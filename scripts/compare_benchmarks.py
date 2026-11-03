"""Compare benchmark v1 and v2 results (docs/benchmark_v2.md) from the committed summaries.

Writes docs/results/benchmark_v1_v2.json and the README comparison chart.
Usage:  python scripts/compare_benchmarks.py
"""

from __future__ import annotations

import json

import numpy as np

from driveaware.core.run import REPO_ROOT
from driveaware.viz.charts import version_bars

RES = REPO_ROOT / "docs/results"
MODELS = ["Model A", "Model B (H=1)", "Model B"]


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def junction_share(loc: dict) -> float:
    total = sum(loc.values())
    return 100.0 * sum(v for k, v in loc.items() if k.startswith("junction")) / max(total, 1)


def summarise(version: str) -> dict:
    sfx = "" if version == "v1" else "_v2"
    p5, p6, p7 = (
        load(f"phase5_temporal{sfx}.json"),
        load(f"phase6_introspection{sfx}.json"),
        load(f"phase7_uncertainty{sfx}.json"),
    )
    out: dict = {"closed_loop": {}, "open_loop_ade": {}, "introspection": {}, "uncertainty": {}}
    e = p5["closed_loop"]["Expert"]
    out["closed_loop"]["Expert"] = {
        "completion": [100 * e["route_completion"]],
        "success": [100 * e["success_rate"]],
        "collision": [100 * e["collision_episode_rate"]],
        "red_light_per_km": [e["per_km"]["red_light"]],
    }
    for m in MODELS:
        c = p5["closed_loop"][m]
        out["closed_loop"][m] = {
            "completion": [100 * v for v in c["route_completion"]],
            "success": [100 * v for v in c["success_rate"]],
            "collision": [100 * v for v in c["collision_episode_rate"]],
            "red_light_per_km": c["per_km"]["red_light"],
            "off_road_per_km": c["per_km"]["off_road"],
            "route_deviation_per_km": c["per_km"]["route_deviation"],
            "collision_per_km": c["per_km"]["collision"],
            "junction_failure_share": [junction_share(c["failure_location_pooled"])],
        }
        out["open_loop_ade"][m] = p5["open_loop_test"][m]["ade"]
    out["temporal_effect_completion"] = p5["paired"]["Model B - Model B (H=1)"]["pooled"][
        "completion"
    ]
    pr = p6["predictors"]
    out["introspection"] = {
        "prevalence": pr["introspection (C)"]["prevalence"],
        "auroc": pr["introspection (C)"]["auroc"],
        "pr_auc": pr["introspection (C)"]["pr_auc"],
        "ece": pr["introspection (C)"]["ece"],
        "detection_rate": pr["introspection (C)"]["detection_rate"],
        "false_alarms_per_km": pr["introspection (C)"]["false_alarms_per_km"],
        "best_heuristic_auroc": [
            pr[p6["best_heuristic"][s]]["auroc"][i] for i, s in enumerate(p6["seeds"])
        ],
        "anticipation_pr_auc": [
            p6["analysis"][s]["strata"]["anticipation: positives 1-2 s ahead vs nominal negatives"][
                "introspection (C)"
            ]["pr_auc"]
            for s in p6["seeds"]
        ],
        "h2_auroc_diff": [list(p6["h2"][s].values())[0]["mean"] for s in p6["seeds"]],
    }
    seeds = sorted(p7["seeds"], key=int)
    ol = [p7["seeds"][s]["open_loop"] for s in seeds]
    out["uncertainty"] = {
        "ensemble_ade": [o["clean"]["Ensemble"]["ade"] for o in ol],
        "ensemble_spearman": [o["clean"]["Ensemble + Laplace"]["spearman_u_ade"] for o in ol],
        "coverage_1sigma_occlusion": [
            o["occlusion"]["Ensemble + Laplace"]["coverage_1sigma_scaled"] for o in ol
        ],
        "failure_auroc_all_uncertainty": [
            p7["seeds"][s]["failure"]["All uncertainty signals"]["auroc"] for s in seeds
        ],
        "adds_to_introspection_auroc": [
            p7["seeds"][s]["uncertainty_adds_to_introspection"]["auroc"]["mean"] for s in seeds
        ],
    }
    return out


def main() -> None:
    data = {v: summarise(v) for v in ("v1", "v2")}
    (RES / "benchmark_v1_v2.json").write_text(json.dumps(data, indent=1))
    vals = {v: {m: data[v]["closed_loop"][m] for m in MODELS} for v in data}
    version_bars(
        [
            ("completion", "Route completion (%)"),
            ("collision", "Episodes with a collision (%)"),
            ("red_light_per_km", "Red-light violations / km"),
            ("junction_failure_share", "Failures ending in junctions (%)"),
        ],
        MODELS,
        vals,
        REPO_ROOT / "docs/assets/benchmark_v1_v2_results.png",
        "Same models, same protocol: benchmark v1 vs v2 (test_id, 150 routes)",
        caption="bars: mean over 3 seeds · dots: seeds · v2 = realistic junctions and signals "
        "(pre-registered); expert 99-100% on both",
    )
    f = lambda v: f"{np.mean(v):.3f}"  # noqa: E731
    for v in data:
        print(
            v,
            {m: f(data[v]["closed_loop"][m]["completion"]) for m in MODELS},
            "C AUROC",
            f(data[v]["introspection"]["auroc"]),
        )


if __name__ == "__main__":
    main()
