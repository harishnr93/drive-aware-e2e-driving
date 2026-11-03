"""Aggregate a Phase 6 introspection run (results + analysis) into a committed summary JSON.

Usage:  python scripts/summarize_introspection.py --run-dir experiments/<ts>_introspection[_v2] \
            --out docs/results/phase6_introspection[_v2].json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    res = json.loads((run / "results.json").read_text())
    analysis = json.loads((run / "analysis.json").read_text())
    seeds = sorted(res, key=int)
    preds = list(res[seeds[0]]["predictors"])
    out = {
        "source_run": run.name,
        "label": "failure (collision, off-road, route deviation, red light, stall) in (t, t+2 s]",
        "alarm_threshold_rule": "frame-level FPR 10% on intro_cal",
        "seeds": seeds,
        "data": {s: res[s]["data"] for s in seeds},
        "reproduces_phase5": {s: res[s]["reproduces_phase5_max_completion_diff"] for s in seeds},
        "predictors": {},
        "h2": {s: res[s]["h2"] for s in seeds},
        "best_heuristic": {s: res[s]["best_heuristic"] for s in seeds},
        "ablations": {s: res[s]["ablations"] for s in seeds},
        "analysis": analysis,
    }
    for p in preds:
        d: dict = {}
        for m in ("auroc", "pr_auc", "brier", "ece", "prevalence"):
            d[m] = [res[s]["predictors"][p][m] for s in seeds]
        for m in (
            "detection_rate",
            "lead_time_median_s",
            "false_alarms_per_km",
            "events",
            "threshold",
        ):
            d[m] = [res[s]["predictors"][p]["event_level"][m] for s in seeds]
        for m in ("auroc_ci95", "pr_auc_ci95"):
            if m in res[seeds[0]]["predictors"][p]:
                d[m] = [res[s]["predictors"][p][m] for s in seeds]
        if "uncalibrated" in res[seeds[0]]["predictors"][p]:
            d["uncalibrated_ece"] = [res[s]["predictors"][p]["uncalibrated"]["ece"] for s in seeds]
            d["uncalibrated_brier"] = [
                res[s]["predictors"][p]["uncalibrated"]["brier"] for s in seeds
            ]
        out["predictors"][p] = d
    Path(args.out).write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
