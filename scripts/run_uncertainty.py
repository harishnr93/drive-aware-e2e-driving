"""Phase 7: trajectory uncertainty (MC dropout, Laplace head, deep ensemble) and its relation to
prediction error and to failures.

Per encoder seed k (model A seed k's frozen encoder):
- open loop on expert data: test_id clean + image corruptions; sigma-scale calibrated on val;
- failure link: uncertainty recomputed offline on the recorded Phase 6 rollouts of model B
  seed k (histories rebuilt from the recorded per-step encoder features), used as a failure
  predictor alone and as extra input to the introspection head;
- inference latency (batch 1) for each method.

Usage:  python scripts/run_uncertainty.py [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from driveaware.core.hardware import LatencyStats
from driveaware.core.logging import setup_logging
from driveaware.core.run import REPO_ROOT, create_run_dir
from driveaware.core.seeding import seed_everything
from driveaware.data.corruptions import CORRUPTIONS
from driveaware.data.dataset import DrivingDataset
from driveaware.evaluation import uncertainty_metrics as U
from driveaware.introspection import data as D
from driveaware.introspection import metrics as M
from driveaware.introspection.models import (
    HeadConfig,
    LogisticModel,
    PlattCalibrator,
    head_logits,
    train_head,
)
from driveaware.introspection.rollouts import load_rollouts
from driveaware.models.encoders.compact_cnn import normalize_images
from driveaware.models.uncertainty.ensemble import (
    EnsemblePredictor,
    TrajectoryUncertainty,
    mc_dropout,
)
from driveaware.training.bc import load_policy
from driveaware.training.temporal import FeatureSplit, cache_features

DATA = REPO_ROOT / "data/expert_v1"
SUFFIX = ""  # run-name suffix ("" = v1, "_v2" = benchmark v2)
METHODS = ("MC dropout", "Laplace head", "Ensemble", "Ensemble + Laplace")
H, STRIDE, BATCH = 4, 1, 4096


@dataclass
class UncertaintyRunConfig:
    seeds: list[int]
    members: int = 5
    mc_passes: int = 20
    corruptions: tuple[str, ...] = tuple(CORRUPTIONS)
    head: HeadConfig = field(default_factory=HeadConfig)


def run_of(pattern: str) -> Path:
    runs = sorted(d for d in REPO_ROOT.glob(f"experiments/*{pattern}") if (d / "best.pt").exists())
    return runs[-1]


@torch.no_grad()
def encode(model, images: np.ndarray, device: str) -> np.ndarray:
    out = np.empty((len(images), model.encoder.feature_dim), np.float32)
    for s in range(0, len(images), 512):
        x = torch.from_numpy(np.ascontiguousarray(images[s : s + 512])).to(device)
        out[s : s + 512] = (
            model.encoder(normalize_images(x, model.image_mean, model.image_std)).cpu().numpy()
        )
    return out


def predict_all(feats, speed, cmd, ens: EnsemblePredictor, det, passes: int) -> dict:
    """All methods on [N, H, F] inputs, batched. Returns method -> TrajectoryUncertainty."""
    parts: dict[str, list[TrajectoryUncertainty]] = {m: [] for m in METHODS}
    for s in range(0, len(feats), BATCH):
        f = feats[s : s + BATCH].cuda()
        sp = speed[s : s + BATCH].cuda()
        c = cmd[s : s + BATCH].cuda()
        e = ens.predict(f, sp, c)
        mc = mc_dropout(det, f, sp, c, passes=passes, seed=s)
        m0 = e.member_means[0]
        b0 = TrajectoryUncertainty(m0, np.zeros_like(m0), np.zeros_like(m0))
        # Laplace head alone = member 0's own scale (aleatoric only).
        with torch.no_grad():
            _, scale = ens.members[0].forward_dist(f, sp, c)
        b0.aleatoric_var = 2.0 * scale.float().cpu().numpy() ** 2
        parts["MC dropout"].append(mc)
        parts["Laplace head"].append(b0)
        parts["Ensemble"].append(
            TrajectoryUncertainty(e.mean, e.epistemic_var, np.zeros_like(e.mean))
        )
        parts["Ensemble + Laplace"].append(e)
    out = {}
    for m, ps in parts.items():
        out[m] = TrajectoryUncertainty(
            np.concatenate([p.mean for p in ps]),
            np.concatenate([p.epistemic_var for p in ps]),
            np.concatenate([p.aleatoric_var for p in ps]),
        )
    return out


def open_loop_metrics(
    pred: TrajectoryUncertainty, target: np.ndarray, scale: float, err_threshold: float
) -> dict:
    resid = pred.mean - target
    ade = np.linalg.norm(resid, axis=-1).mean(axis=1)
    fde = np.linalg.norm(resid[:, -1], axis=-1)
    var = pred.total_var
    u = TrajectoryUncertainty.summary(var)
    sp = U.sparsification(ade, u)
    high = ade > err_threshold
    return {
        "ade": float(ade.mean()),
        "fde": float(fde.mean()),
        "mean_uncertainty_m": float(u.mean()),
        "spearman_u_ade": U.spearman(u, ade),
        "auroc_high_error": M.auroc(high, u),
        "high_error_rate": float(high.mean()),
        "ause": sp["ause"],
        "nll_raw": U.gaussian_nll(resid, var),
        "nll_scaled": U.gaussian_nll(resid, var * scale**2),
        "coverage_1sigma_raw": U.coverage(resid, var, 1.0),
        "coverage_2sigma_raw": U.coverage(resid, var, 2.0),
        "coverage_1sigma_scaled": U.coverage(resid, var * scale**2, 1.0),
        "coverage_2sigma_scaled": U.coverage(resid, var * scale**2, 2.0),
        "sigma_scale": scale,
        "_curves": {
            "sparsification": {k: np.asarray(v).tolist() for k, v in sp.items() if k != "ause"},
            "binned": U.binned_uncertainty_vs_error(u, ade),
        },
    }


def split_inputs(split: FeatureSplit):
    """Histories, speeds, commands and labels of all labelled samples of a split (CPU)."""
    g = split.indices
    feats, speeds, cmds, labels = [], [], [], []
    for s in range(0, len(g), 20000):
        f, sp, c, wp = split.gather(g[s : s + 20000], H, STRIDE)
        feats.append(f.cpu())
        speeds.append(sp.cpu())
        cmds.append(c.cpu())
        labels.append(wp.cpu())
    return torch.cat(feats), torch.cat(speeds), torch.cat(cmds), torch.cat(labels).numpy()


def rollout_histories(ep: dict):
    feats = torch.from_numpy(ep["feature"].astype(np.float32))
    n = len(feats)
    idx = np.arange(n)[:, None] - STRIDE * np.arange(H - 1, -1, -1)[None]
    idx = np.maximum(idx, 0)
    speed = torch.from_numpy(ep["speed"].astype(np.float32))
    return feats[idx], speed[idx], torch.from_numpy(ep["command"].astype(np.int64))


def run_seed(k: int, cfg: UncertaintyRunConfig, run_dir: Path, log) -> dict:
    seed_everything(k)
    a_run = run_of(f"_model_a{SUFFIX}_seed{k}")
    b_run = run_of(f"_model_b_h4{SUFFIX}_seed{k}")
    members = [
        load_policy(run_of(f"_model_bprob_e{m}_k{k}{SUFFIX}_seed{10 * k + m}") / "best.pt", "cuda")[
            0
        ]
        for m in range(cfg.members)
    ]
    ens = EnsemblePredictor(members, "cuda")
    det, _ = load_policy(b_run / "best.pt", "cuda")
    res: dict = {
        "seed": k,
        "encoder_run": a_run.name,
        "policy_run": b_run.name,
        "member_val_ade": [],
        "open_loop": {},
        "failure": {},
    }
    for m in range(cfg.members):
        r = json.loads(
            (run_of(f"_model_bprob_e{m}_k{k}{SUFFIX}_seed{10 * k + m}") / "result.json").read_text()
        )
        res["member_val_ade"].append(r["best_val"]["ade"])

    # ---- open loop: calibrate sigma scale + error threshold on clean val --------------------
    ck = a_run / "best.pt"
    val_ds = DrivingDataset(DATA / "val")
    val = FeatureSplit(val_ds, cache_features(ck, DATA / "val", torch.device("cuda")), "cuda")
    vf, vs, vc, vw = split_inputs(val)
    vpred = predict_all(vf, vs, vc, ens, det, cfg.mc_passes)
    scales = {m: U.fit_sigma_scale(p.mean - vw, p.total_var) for m, p in vpred.items()}
    val_ade = np.linalg.norm(vpred["Ensemble"].mean - vw, axis=-1).mean(axis=1)
    err_thr = float(np.quantile(val_ade, 0.9))
    res["sigma_scales_val"] = scales
    res["high_error_threshold_m"] = err_thr

    test_ds = DrivingDataset(DATA / "test_id")
    a_model, _ = load_policy(ck, "cuda")
    test_images = np.load(DATA / "test_id" / "images.npy", mmap_mode="r")
    conditions = {"clean": cache_features(ck, DATA / "test_id", torch.device("cuda"))}
    for name in cfg.corruptions:
        feats = np.empty((len(test_images), 256), np.float32)
        for s in range(0, len(test_images), 2048):
            chunk = np.asarray(test_images[s : s + 2048])
            feats[s : s + 2048] = encode(a_model, CORRUPTIONS[name](chunk, 1000 + s), "cuda")
        conditions[name] = feats
    for cond, feats in conditions.items():
        split = FeatureSplit(test_ds, feats, "cuda")
        tf, ts_, tc, tw = split_inputs(split)
        preds = predict_all(tf, ts_, tc, ens, det, cfg.mc_passes)
        res["open_loop"][cond] = {
            m: open_loop_metrics(p, tw, scales[m], err_thr) for m, p in preds.items()
        }
        if cond == "clean":
            res["open_loop"]["clean"]["Model B (deterministic)"] = {
                "ade": float(np.linalg.norm(det_predict(det, tf, ts_, tc) - tw, axis=-1).mean())
            }
        log.info(
            "seed %d %s: %s",
            k,
            cond,
            {
                m: (
                    round(v["ade"], 3),
                    round(v.get("spearman_u_ade", 0), 3),
                    round(v.get("auroc_high_error", 0), 3),
                )
                for m, v in res["open_loop"][cond].items()
            },
        )

    # ---- failure link on recorded rollouts --------------------------------------------------
    sets = ("intro_train", "intro_val", "intro_cal", "test_id")
    eps = {
        s: load_rollouts(
            sorted((REPO_ROOT / "data/introspection" / b_run.name / s).glob("ep_*.npz"))
        )
        for s in sets
    }
    unc: dict[str, list[np.ndarray]] = {s: [] for s in sets}
    max_wp_diff = 0.0
    for s in sets:
        for ep in eps[s]:
            f, sp, c = rollout_histories(ep)
            p = predict_all(f, sp, c, ens, det, cfg.mc_passes)
            wp = det_predict(det, f, sp, c)
            max_wp_diff = max(max_wp_diff, float(np.abs(wp - ep["waypoints"]).max()))
            unc[s].append(
                np.column_stack(
                    [
                        TrajectoryUncertainty.summary(p["Ensemble"].epistemic_var),
                        TrajectoryUncertainty.summary(p["Ensemble + Laplace"].aleatoric_var),
                        TrajectoryUncertainty.summary(p["MC dropout"].epistemic_var),
                    ]
                )
            )
    res["rollout_reconstruction_max_waypoint_diff_m"] = max_wp_diff
    log.info("seed %d: rollout prediction reconstruction max diff %.4f m", k, max_wp_diff)
    y = {s: [D.failure_labels(e) for e in eps[s]] for s in sets}
    cat = np.concatenate

    def finish(name, logits_fn):
        cal = PlattCalibrator.fit(cat(logits_fn("intro_cal")), cat(y["intro_cal"]))
        probs = [cal(lg) for lg in logits_fn("test_id")]
        yt, pt = cat(y["test_id"]), cat(probs)
        out = {
            "auroc": M.auroc(yt, pt),
            "pr_auc": M.average_precision(yt, pt),
            "brier": M.brier(yt, pt),
            "ece": M.reliability(yt, pt)["ece"],
        }
        res["failure"][name] = out
        log.info(
            "seed %d failure %-34s AUROC %.3f PR-AUC %.3f", k, name, out["auroc"], out["pr_auc"]
        )
        return probs

    probs = {}
    for j, name in enumerate(
        ("Ensemble disagreement (epistemic)", "Laplace scale (aleatoric)", "MC dropout variance")
    ):
        lm = LogisticModel.fit(cat(unc["intro_train"])[:, [j]], cat(y["intro_train"]))
        probs[name] = finish(name, lambda s, lm=lm, j=j: [lm.logits(u[:, [j]]) for u in unc[s]])
    lm = LogisticModel.fit(cat(unc["intro_train"]), cat(y["intro_train"]))
    probs["All uncertainty signals"] = finish(
        "All uncertainty signals", lambda s: [lm.logits(u) for u in unc[s]]
    )
    xl = {s: [D.learned_features(e) for e in eps[s]] for s in sets}
    xu = {
        s: [np.column_stack([a, np.log(u + 1e-4)]) for a, u in zip(xl[s], unc[s], strict=True)]
        for s in sets
    }
    hcfg = HeadConfig(seed=k)
    for name, x in (("Introspection (C)", xl), ("Introspection + uncertainty", xu)):
        head, _ = train_head(
            cat(x["intro_train"]),
            cat(y["intro_train"]),
            cat(x["intro_val"]),
            cat(y["intro_val"]),
            hcfg,
            device="cuda",
        )
        probs[name] = finish(
            name, lambda s, head=head, x=x: [head_logits(head, v, "cuda") for v in x[s]]
        )
    yt = y["test_id"]
    rng = np.random.default_rng(k)
    diffs = {"auroc": [], "pr_auc": []}
    for _ in range(1000):
        pick = rng.integers(0, len(yt), len(yt))
        yy = cat([yt[i] for i in pick])
        if not (yy.any() and (~yy).any()):
            continue
        a = cat([probs["Introspection + uncertainty"][i] for i in pick])
        b = cat([probs["Introspection (C)"][i] for i in pick])
        diffs["auroc"].append(M.auroc(yy, a) - M.auroc(yy, b))
        diffs["pr_auc"].append(M.average_precision(yy, a) - M.average_precision(yy, b))
    res["uncertainty_adds_to_introspection"] = {
        m: {
            "mean": res["failure"]["Introspection + uncertainty"][m]
            - res["failure"]["Introspection (C)"][m],
            "ci95": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))),
        }
        for m, v in diffs.items()
    }
    np.savez_compressed(
        run_dir / f"rollout_uncertainty_seed{k}.npz",
        **{f"u_{i}": u for i, u in enumerate(unc["test_id"])},
        **{f"y_{i}": yy for i, yy in enumerate(yt)},
    )
    res["latency_ms"] = latency(ens, det, a_model)
    return res


@torch.no_grad()
def det_predict(det, f, sp, c) -> np.ndarray:
    det.eval()
    out = [
        det.forward_features(
            f[s : s + BATCH].cuda(), sp[s : s + BATCH].cuda(), c[s : s + BATCH].cuda()
        )
        .float()
        .cpu()
        .numpy()
        for s in range(0, len(f), BATCH)
    ]
    return np.concatenate(out)


@torch.no_grad()
def _latency_on(dev: str, ens: EnsemblePredictor, det, a_model) -> dict:
    enc = a_model.to(dev).eval()
    heads = [m.to(dev).eval() for m in ens.members]
    d = det.to(dev).eval()
    img = torch.randint(0, 255, (1, 96, 160, 3), dtype=torch.uint8, device=dev)
    f = torch.randn(1, H, 256, device=dev)
    sp = torch.zeros(1, H, device=dev)
    c = torch.zeros(1, dtype=torch.long, device=dev)

    def encoder():
        enc.encoder(normalize_images(img, enc.image_mean, enc.image_std))

    cases = {
        "deterministic (B)": lambda: (encoder(), d.forward_features(f, sp, c)),
        "Laplace head": lambda: (encoder(), heads[0].forward_dist(f, sp, c)),
        "Ensemble (5 heads)": lambda: (encoder(), [h.forward_dist(f, sp, c) for h in heads]),
        "MC dropout (20 passes)": lambda: (encoder(), mc_dropout(d, f, sp, c, 20)),
    }
    out = {}
    for name, fn in cases.items():
        for _ in range(10):
            fn()
        lat = LatencyStats()
        for _ in range(100):
            with lat.measure(cuda_sync=dev == "cuda"):
                fn()
        out[f"{dev}: {name}"] = lat.summary()["p50_ms"]
    return out


def latency(ens: EnsemblePredictor, det, a_model) -> dict:
    """Per-step latency, batch 1: encoder on the newest frame + temporal heads (1 CPU thread)."""
    torch.set_num_threads(1)
    out = {}
    for dev in ("cuda", "cpu"):
        out.update(_latency_on(dev, ens, det, a_model))
    ens.members = [m.to("cuda") for m in ens.members]
    det.to("cuda")
    a_model.to("cuda")
    return out


def main() -> None:
    global SUFFIX, DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--suffix", default="")
    parser.add_argument("--data-root", default=str(DATA))
    args = parser.parse_args()
    SUFFIX, DATA = args.suffix, Path(args.data_root)
    cfg = UncertaintyRunConfig(args.seeds)
    run_dir = create_run_dir(f"uncertainty{SUFFIX}", cfg)
    log = setup_logging(run_dir)
    all_res = {}
    t0 = time.perf_counter()
    for k in args.seeds:
        res = run_seed(k, cfg, run_dir, log)
        all_res[k] = res
        (run_dir / f"results_seed{k}.json").write_text(json.dumps(res, indent=1, default=str))
    (run_dir / "results.json").write_text(json.dumps(all_res, indent=1, default=str))
    log.info("done in %.0f s: %s", time.perf_counter() - t0, run_dir)


if __name__ == "__main__":
    main()
