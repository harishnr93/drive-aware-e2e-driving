"""Aggregate Phase 5 (temporal model) results into docs/results/phase5_temporal.json.

Inputs: the closed-loop report (150 test_id routes) and the training run directories.
Open-loop test metrics are recomputed here with each checkpoint's own history settings.

Usage:  python scripts/summarize_phase5.py --report experiments/<ts>_closed_loop_test_id/report.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from driveaware.core.run import REPO_ROOT
from driveaware.evaluation.closed_loop import bootstrap_ci, paired_difference
from driveaware.evaluation.open_loop import evaluate_checkpoint, reference_baselines

GROUPS = {
    "Model A": "_model_a_seed",
    "Model B (H=1)": "_model_b_h1_seed",
    "Model B": "_model_b_h4_seed",
}
ABLATIONS = (
    "_model_b_h1_seed0",
    "_ablate_b_h2_seed0",
    "_model_b_h4_seed0",
    "_ablate_b_h8_seed0",
    "_ablate_b_h4_stride2_seed0",
    "_ablate_b_h4_speedhist_seed0",
    "_ablate_b_e2e_seed0",
    "_model_a_seed0",
)


def run_dir(suffix: str) -> Path | None:
    # Completed runs only (a run directory without result.json is still training).
    hits = sorted(
        d for d in REPO_ROOT.glob(f"experiments/*{suffix}") if (d / "result.json").exists()
    )
    return hits[-1] if hits else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "docs/results/phase5_temporal.json"))
    parser.add_argument("--suffix", default="", help='run-name suffix, e.g. "_v2"')
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data/expert_v1"))
    parser.add_argument(
        "--ablation-report",
        default=None,
        help="closed-loop report of single-seed ablations on the same scenarios",
    )
    args = parser.parse_args()
    global GROUPS, ABLATIONS
    GROUPS = {k: v.replace("_seed", f"{args.suffix}_seed") for k, v in GROUPS.items()}
    # Only ablation runs of the same benchmark (none were re-run on v2).
    ABLATIONS = tuple(a.replace("_seed0", f"{args.suffix}_seed0") for a in ABLATIONS)
    report_path = Path(args.report)
    report = json.loads(report_path.read_text())
    episodes = {
        n: json.loads((report_path.parent / f"episodes_{n}.json").read_text())
        for n in report["agents"]
    }

    out: dict = {
        "source_report": report_path.parent.name,
        "num_scenarios": report["num_scenarios"],
        "closed_loop": {},
        "open_loop_test": {},
        "paired": {},
        "val_ablation": {},
        "training": {},
    }
    out["closed_loop"]["Expert"] = report["agents"]["expert"]["closed_loop"]
    test_dir = Path(args.data_root) / "test_id"
    out["open_loop_test"]["reference"] = reference_baselines(test_dir)
    members: dict[str, list[str]] = {}
    for group, key in GROUPS.items():
        names = sorted(n for n in report["agents"] if key in n)
        members[group] = names
        cls = [report["agents"][n]["closed_loop"] for n in names]
        pooled_out: collections.Counter[str] = collections.Counter()
        pooled_loc: collections.Counter[str] = collections.Counter()
        for c in cls:
            pooled_out.update(c["outcomes"])
            pooled_loc.update(c["failure_location"])
        out["closed_loop"][group] = {
            "runs": names,
            "route_completion": [c["route_completion"] for c in cls],
            "success_rate": [c["success_rate"] for c in cls],
            "collision_episode_rate": [c["collision_episode_rate"] for c in cls],
            "driving_score": [c["driving_score"] for c in cls],
            "per_km": {k: [c["per_km"][k] for c in cls] for k in cls[0]["per_km"]},
            "collisions_by_kind": [c["collisions_by_kind"] for c in cls],
            "latency_ms_p50": [c.get("latency_ms_p50") for c in cls],
            "latency_ms_p95": [c.get("latency_ms_p95") for c in cls],
            "outcomes_pooled": dict(pooled_out),
            "failure_location_pooled": dict(pooled_loc),
        }
        ol = []
        for n in names:
            ck = Path(report["agents"][n]["spec"]["checkpoint"])
            ol.append(evaluate_checkpoint(ck, test_dir))
        out["open_loop_test"][group] = {k: [o[k] for o in ol] for k in ol[0]}
        for n in names:
            res = json.loads((REPO_ROOT / "experiments" / n / "result.json").read_text())
            out["training"][n] = {
                "best_val_ade": res["best_val"]["ade"],
                "train_seconds": res["train_seconds"],
                "params": res.get("params", res.get("params_total")),
                "params_trainable": res.get("params_trainable", res.get("params")),
                "feature_cache_seconds": res.get("feature_cache_seconds"),
                "peak_vram_alloc_mib": res["resources"]["torch_peak_allocated_mib"],
                "inference": res.get("inference_latency", res.get("inference_latency_batch1")),
            }

    def compare(a: str, b: str) -> dict:
        res: dict = {"per_seed": {}, "pooled": {}}
        pairs = list(zip(members[a], members[b], strict=True))
        for key in ("completion", "collision", "driving_score"):
            res["per_seed"][key] = [
                paired_difference(episodes[x], episodes[y], key) for x, y in pairs
            ]
            rows_a = [r for x, _ in pairs for r in episodes[x]]
            rows_b = [r for _, y in pairs for r in episodes[y]]
            res["pooled"][key] = paired_difference(rows_a, rows_b, key)
        rl_a = np.array([r.get("n_red_light", 0) for x, _ in pairs for r in episodes[x]], float)
        rl_b = np.array([r.get("n_red_light", 0) for _, y in pairs for r in episodes[y]], float)
        d = rl_a - rl_b
        res["pooled"]["red_light_count_per_episode"] = {
            "mean": float(d.mean()),
            "ci95": bootstrap_ci(d),
        }
        return res

    out["paired"]["Model B - Model A"] = compare("Model B", "Model A")
    out["paired"]["Model B - Model B (H=1)"] = compare("Model B", "Model B (H=1)")
    out["paired"]["Model B (H=1) - Model A"] = compare("Model B (H=1)", "Model A")

    for suffix in ABLATIONS:
        d = run_dir(suffix)
        if d is None:
            continue
        res = json.loads((d / "result.json").read_text())
        cfg = (d / "config.yaml").read_text()
        out["val_ablation"][suffix.strip("_")] = {
            "run": d.name,
            "best_val": res["best_val"],
            "train_seconds": res["train_seconds"],
            "config_excerpt": [
                ln
                for ln in cfg.splitlines()
                if ln.split(":")[0].strip()
                in ("history", "history_stride", "speed_history", "freeze_encoder")
            ],
        }
    if args.ablation_report:
        ab_path = Path(args.ablation_report)
        ab = json.loads(ab_path.read_text())
        out["ablation_closed_loop"] = {"source_report": ab_path.parent.name}
        for name, a in ab["agents"].items():
            rows = json.loads((ab_path.parent / f"episodes_{name}.json").read_text())
            entry = {
                "closed_loop": a["closed_loop"],
                "open_loop_test": a.get("open_loop"),
                "paired": {},
            }
            for ref_group in ("Model B", "Model A"):
                ref = members[ref_group][0]  # seed 0 of the reference group
                entry["paired"][f"{name} - {ref}"] = {
                    k: paired_difference(rows, episodes[ref], k)
                    for k in ("completion", "collision", "driving_score")
                }
            out["ablation_closed_loop"][name] = entry
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
