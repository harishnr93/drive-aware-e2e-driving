"""Aggregate a Phase 7 uncertainty run into a committed summary JSON (curves stripped).

Usage:  python scripts/summarize_uncertainty.py --run-dir experiments/<ts>_uncertainty[_v2] \
            --out docs/results/phase7_uncertainty[_v2].json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    res = json.loads((run / "results.json").read_text())
    out = {
        "source_run": run.name,
        "methods": ["MC dropout", "Laplace head", "Ensemble", "Ensemble + Laplace"],
        "notes": {
            "uncertainty_summary": "mean over waypoints of sqrt(var_x + var_y), metres",
            "sigma_scale": "fitted on clean val so that 68.3% of coordinates satisfy "
            "|e| <= s*sigma",
            "high_error": "ADE above the 90th percentile of the ensemble mean's clean val ADE",
        },
        "seeds": {},
    }
    for s, r in res.items():
        r = copy.deepcopy(r)
        for cond in r["open_loop"].values():
            for m in cond.values():
                m.pop("_curves", None)
        out["seeds"][s] = r
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
