"""Phase 6: record rollouts of model B, train/calibrate failure predictors, evaluate on test_id.

Per model-B seed: record intro_train / intro_val / intro_cal / test_id rollouts, train the
introspection head (C = B + head) and the baselines on intro_train, early-stop the head on
intro_val, calibrate everything on intro_cal, fix alarm thresholds on intro_cal, and evaluate
on the 150 test_id routes (the same scenarios as Phase 5).

Usage:  python scripts/run_introspection.py [--seeds 0 1 2] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from driveaware.core.logging import setup_logging
from driveaware.core.run import REPO_ROOT, create_run_dir
from driveaware.core.seeding import seed_everything
from driveaware.evaluation.closed_loop import AgentSpec, scenario_set
from driveaware.introspection import data as D
from driveaware.introspection import metrics as M
from driveaware.introspection.models import (
    HeadConfig,
    LogisticModel,
    PlattCalibrator,
    head_logits,
    train_head,
)
from driveaware.introspection.rollouts import load_rollouts, record_rollouts
from driveaware.sim.lightweight.scenario import ScenarioSpec

SETS = ("intro_train", "intro_val", "intro_cal", "test_id")
TARGET_FPR = 0.10  # frame-level false-positive rate used to fix alarm thresholds (on intro_cal)
HEURISTICS = {
    "speed": ["speed"],
    "context (speed + command)": ["context"],
    "novelty": ["novelty"],
    "inconsistency": ["inconsistency"],
    "all heuristics": ["context", "novelty", "inconsistency", "predicted_turn"],
}


@dataclass
class IntrospectionRunConfig:
    seeds: list[int]
    workers: int
    target_fpr: float = TARGET_FPR
    sets: tuple[str, ...] = SETS
    head: HeadConfig = field(default_factory=HeadConfig)


def scenarios(name: str) -> list[ScenarioSpec]:
    cfg = yaml.safe_load((REPO_ROOT / "configs/evaluation/closed_loop.yaml").read_text())
    sc = cfg["sets"][name]
    base = ScenarioSpec(0, 0, 0, route_length=cfg["route_length"], max_duration=cfg["max_duration"])
    maps = list(range(sc["map_seed_start"], sc["map_seed_start"] + sc["num_maps"]))
    return scenario_set(name, maps, sc["routes_per_map"], base, cfg["seed"])


def stack(eps: list[dict], fn) -> np.ndarray:
    return np.concatenate([fn(ep) for ep in eps])


def evaluate_scores(eps, y_eps, probs_eps, thr, with_ci=False) -> dict:
    y = np.concatenate(y_eps)
    p = np.concatenate(probs_eps)
    rel = M.reliability(y, p)
    out = {
        "auroc": M.auroc(y, p),
        "pr_auc": M.average_precision(y, p),
        "prevalence": float(y.mean()),
        "brier": M.brier(y, p),
        "ece": rel["ece"],
        "reliability": {k: np.asarray(v).tolist() for k, v in rel.items() if k != "ece"},
        "event_level": M.event_level(eps, probs_eps, thr),
    }
    if with_ci:
        pairs = list(zip(y_eps, probs_eps, strict=True))
        out["auroc_ci95"] = M.episode_bootstrap(pairs, M.auroc)
        out["pr_auc_ci95"] = M.episode_bootstrap(pairs, M.average_precision)
    return out


def paired_metric_difference(y_eps, pa_eps, pb_eps, fn, n=1000, seed=0) -> dict:
    """Episode-bootstrap CI of fn(a) - fn(b) on the same frames."""
    rng = np.random.default_rng(seed)
    k = len(y_eps)
    full = fn(np.concatenate(y_eps), np.concatenate(pa_eps)) - fn(
        np.concatenate(y_eps), np.concatenate(pb_eps)
    )
    vals = []
    for _ in range(n):
        pick = rng.integers(0, k, k)
        y = np.concatenate([y_eps[i] for i in pick])
        if not (y.any() and (~y).any()):
            continue
        a = np.concatenate([pa_eps[i] for i in pick])
        b = np.concatenate([pb_eps[i] for i in pick])
        vals.append(fn(y, a) - fn(y, b))
    return {
        "mean": float(full),
        "ci95": (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))),
    }


def run_seed(seed: int, workers: int, out_dir: Path, log) -> dict:
    seed_everything(seed)
    b_run = sorted(REPO_ROOT.glob(f"experiments/*_model_b_h4_seed{seed}"))[-1]
    a_run = sorted(REPO_ROOT.glob(f"experiments/*_model_a_seed{seed}"))[-1]
    agent = AgentSpec("policy", str(b_run / "best.pt"))
    eps: dict[str, list[dict]] = {}
    t0 = time.perf_counter()
    for name in SETS:
        paths = record_rollouts(
            scenarios(name), agent, REPO_ROOT / "data/introspection" / b_run.name / name, workers
        )
        eps[name] = load_rollouts(paths)
        n_steps = sum(len(e["timestamp"]) for e in eps[name])
        log.info("seed %d %s: %d episodes, %d steps", seed, name, len(paths), n_steps)
    rollout_s = time.perf_counter() - t0

    # Determinism check: the test rollouts must reproduce the Phase 5 closed-loop episodes.
    p5 = sorted(REPO_ROOT.glob("experiments/*_closed_loop_test_id"))
    repro = None
    for rep in p5:
        f = rep / f"episodes_{b_run.name}.json"
        if f.exists() and len(json.loads(f.read_text())) == len(eps["test_id"]):
            rows = json.loads(f.read_text())
            repro = float(
                np.max(
                    np.abs(
                        np.array([r["completion"] for r in rows])
                        - np.array([float(e["completion"]) for e in eps["test_id"]])
                    )
                )
            )
            break
    log.info("seed %d: max |completion difference| vs Phase 5 = %s", seed, repro)

    feats_train = np.load(a_run / "features" / "train.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    novelty = D.NoveltyModel.fit(
        np.asarray(
            feats_train[
                np.sort(
                    rng.choice(len(feats_train), size=min(50_000, len(feats_train)), replace=False)
                )
            ]
        )
    )

    def labels(name, horizon=2.0, types=D.FAILURE_TYPES):
        return [D.failure_labels(e, horizon, types) for e in eps[name]]

    y = {n: labels(n) for n in SETS}
    xl = {n: [D.learned_features(e) for e in eps[n]] for n in SETS}
    hf = {n: [D.heuristic_features(e, novelty) for e in eps[n]] for n in SETS}
    orc = {n: [D.oracle_features(e) for e in eps[n]] for n in SETS}
    cat = np.concatenate

    results: dict = {
        "seed": seed,
        "policy_run": b_run.name,
        "rollout_seconds": rollout_s,
        "reproduces_phase5_max_completion_diff": repro,
        "predictors": {},
        "data": {},
    }
    for n in SETS:
        yy = cat(y[n])
        ft = [D.failure_times(e) for e in eps[n]]
        types = cat([e["event_type"] for e in eps[n]])
        results["data"][n] = {
            "episodes": len(eps[n]),
            "steps": int(len(yy)),
            "positive_rate": float(yy.mean()),
            "failure_events": int(sum(len(f) for f in ft)),
            "failure_events_by_type": {
                t.name: int((types == int(t)).sum()) for t in D.FAILURE_TYPES
            },
            "episodes_with_failure": int(sum(len(f) > 0 for f in ft)),
        }

    def finish(name: str, logit_fn, with_ci=False) -> dict:
        cal = PlattCalibrator.fit(cat(logit_fn("intro_cal")), cat(y["intro_cal"]))
        p_cal = cal(cat(logit_fn("intro_cal")))
        thr = M.threshold_at_fpr(cat(y["intro_cal"]), p_cal, TARGET_FPR)
        test_logits = logit_fn("test_id")
        probs = [cal(lg) for lg in test_logits]
        res = evaluate_scores(eps["test_id"], y["test_id"], probs, thr, with_ci)
        res["calibration"] = {"a": cal.a, "c": cal.c}
        results["predictors"][name] = res
        log.info(
            "seed %d %-28s AUROC %.3f PR-AUC %.3f Brier %.4f ECE %.4f det %.2f lead %.2fs "
            "FA/km %.2f",
            seed,
            name,
            res["auroc"],
            res["pr_auc"],
            res["brier"],
            res["ece"],
            res["event_level"]["detection_rate"],
            res["event_level"]["lead_time_median_s"],
            res["event_level"]["false_alarms_per_km"],
        )
        return {"probs": probs, "logits": test_logits}

    # --- learned introspection head (model C) -----------------------------------------------
    cfg = HeadConfig(seed=seed)
    head, hist = train_head(
        cat(xl["intro_train"]),
        cat(y["intro_train"]),
        cat(xl["intro_val"]),
        cat(y["intro_val"]),
        cfg,
        device="cuda",
    )
    results["head_training"] = hist

    def head_fn(n):
        return [head_logits(head, x, "cuda") for x in xl[n]]

    out_head = finish("introspection (C)", head_fn, with_ci=True)
    # Uncalibrated reliability (sigmoid of weighted-BCE logits) for the before/after figure.
    raw = [1 / (1 + np.exp(-np.clip(lg, -40, 40))) for lg in out_head["logits"]]
    rel_raw = M.reliability(cat(y["test_id"]), cat(raw))
    results["predictors"]["introspection (C)"]["uncalibrated"] = {
        "ece": rel_raw["ece"],
        "brier": M.brier(cat(y["test_id"]), cat(raw)),
        "reliability": {k: np.asarray(v).tolist() for k, v in rel_raw.items() if k != "ece"},
    }

    # --- non-privileged heuristics and the privileged oracle --------------------------------
    outs = {"introspection (C)": out_head}
    for name, keys in HEURISTICS.items():

        def hx(n, keys=keys):
            return [np.column_stack([h[k] for k in keys]) for h in hf[n]]

        lm = LogisticModel.fit(cat(hx("intro_train")), cat(y["intro_train"]))
        outs[name] = finish(
            name,
            lambda n, lm=lm, hx=hx: [lm.logits(x) for x in hx(n)],
            with_ci=name == "all heuristics",
        )
    om = LogisticModel.fit(cat(orc["intro_train"]), cat(y["intro_train"]))
    outs["oracle (privileged)"] = finish(
        "oracle (privileged)", lambda n: [om.logits(x) for x in orc[n]]
    )

    # --- H2: paired comparison against the best non-privileged heuristic --------------------
    best_h = max(HEURISTICS, key=lambda k: results["predictors"][k]["auroc"])
    results["best_heuristic"] = best_h
    results["h2"] = {
        f"{m} difference (C - {best_h})": paired_metric_difference(
            y["test_id"], outs["introspection (C)"]["probs"], outs[best_h]["probs"], fn
        )
        for m, fn in (("auroc", M.auroc), ("pr_auc", M.average_precision))
    }

    # --- label-horizon and label-definition ablations (head only) ---------------------------
    results["ablations"] = {}
    for tag, horizon, types in (
        ("horizon 1 s", 1.0, D.FAILURE_TYPES),
        ("horizon 3 s", 3.0, D.FAILURE_TYPES),
        ("incl. lane departure", 2.0, D.FAILURE_TYPES_WITH_LANE),
    ):
        ya = {n: labels(n, horizon, types) for n in ("intro_train", "intro_val", "test_id")}
        h2, _ = train_head(
            cat(xl["intro_train"]),
            cat(ya["intro_train"]),
            cat(xl["intro_val"]),
            cat(ya["intro_val"]),
            cfg,
            device="cuda",
        )
        s = cat([head_logits(h2, x, "cuda") for x in xl["test_id"]])
        yt = cat(ya["test_id"])
        lm = LogisticModel.fit(
            np.column_stack([cat([h["context"] for h in hf["intro_train"]])]),
            cat(ya["intro_train"]),
        )
        sc = lm.logits(cat([h["context"] for h in hf["test_id"]]))
        results["ablations"][tag] = {
            "prevalence": float(yt.mean()),
            "auroc": M.auroc(yt, s),
            "pr_auc": M.average_precision(yt, s),
            "context_auroc": M.auroc(yt, sc),
            "context_pr_auc": M.average_precision(yt, sc),
        }
        log.info("seed %d ablation %s: %s", seed, tag, results["ablations"][tag])

    # --- per-step test predictions for figures --------------------------------------------
    np.savez_compressed(
        out_dir / f"test_predictions_seed{seed}.npz",
        **{f"y_{i}": yy for i, yy in enumerate(y["test_id"])},
        **{f"t_{i}": e["timestamp"] for i, e in enumerate(eps["test_id"])},
        **{f"p_head_{i}": p for i, p in enumerate(outs["introspection (C)"]["probs"])},
        **{f"p_ctx_{i}": p for i, p in enumerate(outs["context (speed + command)"]["probs"])},
        **{f"p_all_{i}": p for i, p in enumerate(outs["all heuristics"]["probs"])},
        **{f"p_oracle_{i}": p for i, p in enumerate(outs["oracle (privileged)"]["probs"])},
        **{f"p_novelty_{i}": p for i, p in enumerate(outs["novelty"]["probs"])},
    )
    import torch

    torch.save(
        {
            "state_dict": head.state_dict(),
            "in_dim": xl["test_id"][0].shape[1],
            "config": cfg,
            "calibration": results["predictors"]["introspection (C)"]["calibration"],
            "threshold": results["predictors"]["introspection (C)"]["event_level"]["threshold"],
            "policy_checkpoint": str(b_run / "best.pt"),
        },
        out_dir / f"introspection_head_seed{seed}.pt",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run_dir = create_run_dir("introspection", IntrospectionRunConfig(args.seeds, args.workers))
    log = setup_logging(run_dir)
    all_res = {}
    for s in args.seeds:
        all_res[s] = run_seed(s, args.workers, run_dir, log)
        (run_dir / f"results_seed{s}.json").write_text(
            json.dumps(all_res[s], indent=1, default=str)
        )
    (run_dir / "results.json").write_text(json.dumps(all_res, indent=1, default=str))
    log.info("done: %s", run_dir)


if __name__ == "__main__":
    main()
