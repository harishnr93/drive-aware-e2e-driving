"""Post-hoc analysis of failure prediction: anticipation vs. recognition.

Some failure events are detected by the simulator only once the situation is visibly bad
(route deviation > 5 m, footprint off the road). A predictor can score well on those by
*recognising* an ongoing failure rather than *anticipating* it. This script separates the two,
using privileged state for stratification only (never as predictor input):

- per failure type (collision, red light, off-road, route deviation);
- "nominal" frames only: ego box centre within 1 m of the route centre, i.e. still in its lane;
- an anticipation stratum: positives 1-2 s before the event vs. nominal negatives.

Usage:  python scripts/analyze_introspection.py --run-dir experiments/<ts>_introspection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from driveaware.core.run import REPO_ROOT
from driveaware.core.types import EventType
from driveaware.introspection import data as D
from driveaware.introspection import metrics as M

PREDICTORS = {
    "introspection (C)": "head",
    "context (speed + command)": "ctx",
    "all heuristics": "all",
    "novelty": "novelty",
    "oracle (privileged)": "oracle",
}


def analyze_seed(run_dir: Path, seed: int, policy_run: str) -> dict:
    z = np.load(run_dir / f"test_predictions_seed{seed}.npz")
    files = sorted((REPO_ROOT / "data/introspection" / policy_run / "test_id").glob("ep_*.npz"))
    eps = [dict(np.load(f)) for f in files]
    lat = np.concatenate([e["lateral"] for e in eps])
    ttf = np.concatenate([D.time_to_next_failure(e) for e in eps])
    y = np.concatenate([D.failure_labels(e) for e in eps])
    scores = {
        name: np.concatenate([z[f"p_{key}_{i}"] for i in range(len(eps))])
        for name, key in PREDICTORS.items()
    }
    out: dict = {"by_type": {}, "strata": {}}
    for t in (
        EventType.COLLISION,
        EventType.RED_LIGHT,
        EventType.OFF_ROAD,
        EventType.ROUTE_DEVIATION,
    ):
        yt = np.concatenate([D.failure_labels(e, 2.0, (t,)) for e in eps])
        if yt.sum() == 0:
            continue
        out["by_type"][t.name] = {
            "positives": int(yt.sum()),
            **{
                name: {"auroc": M.auroc(yt, s), "pr_auc": M.average_precision(yt, s)}
                for name, s in scores.items()
            },
        }
    nominal = lat < 1.0
    strata = {
        "all frames": np.ones_like(y, dtype=bool),
        "nominal frames (|lateral| < 1 m)": nominal,
        "anticipation: positives 1-2 s ahead vs nominal negatives": nominal
        & ((y & (ttf > 1.0)) | ~y),
    }
    for sname, m in strata.items():
        ys = y[m]
        out["strata"][sname] = {
            "frames": int(m.sum()),
            "positives": int(ys.sum()),
            "positive_share_of_all_positives": float(ys.sum() / max(y.sum(), 1)),
            **{
                name: {"auroc": M.auroc(ys, s[m]), "pr_auc": M.average_precision(ys, s[m])}
                for name, s in scores.items()
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    res = {
        f.stem.split("seed")[-1]: json.loads(f.read_text())
        for f in sorted(run_dir.glob("results_seed*.json"))
    }
    analysis = {s: analyze_seed(run_dir, int(s), r["policy_run"]) for s, r in res.items()}
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=1))
    print(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main()
