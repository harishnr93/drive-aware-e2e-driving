"""Closed-loop evaluation of the privileged expert (reference for later model comparisons).

Usage:  python scripts/eval_expert.py [--set map_seed_start=200 num_maps=10 routes_per_map=10]
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from dataclasses import dataclass

import numpy as np
import yaml

from driveaware.core.config import from_dict
from driveaware.core.logging import setup_logging
from driveaware.core.run import create_run_dir
from driveaware.expert.expert import RuleBasedExpert
from driveaware.sim.lightweight.env import LightweightDrivingSim, SimConfig
from driveaware.sim.lightweight.scenario import BENCHMARKS, ScenarioSpec


@dataclass
class EvalConfig:
    map_seed_start: int = 200
    num_maps: int = 10
    routes_per_map: int = 10
    route_length: float = 400.0
    max_duration: float = 180.0
    benchmark: str = "v1"  # map preset (scenario.BENCHMARKS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = from_dict(
        EvalConfig, {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    )
    run_dir = create_run_dir(f"eval_expert_{cfg.benchmark}", cfg)
    log = setup_logging(run_dir)
    sim = LightweightDrivingSim(SimConfig(render=False))
    rows = []
    t0 = time.perf_counter()
    for m in range(cfg.map_seed_start, cfg.map_seed_start + cfg.num_maps):
        for r in range(cfg.routes_per_map):
            spec = ScenarioSpec(
                m,
                r,
                r,
                route_length=cfg.route_length,
                max_duration=cfg.max_duration,
                map=BENCHMARKS[cfg.benchmark],
            )
            res = sim.reset(spec)
            expert = RuleBasedExpert()
            counts: collections.Counter[str] = collections.Counter()
            speeds, lateral, dist = [], [], 0.0
            while not res.done:
                prev = res.privileged.ego_pose
                res = sim.step(expert(res))
                p = res.privileged
                dist += float(np.hypot(p.ego_pose.x - prev.x, p.ego_pose.y - prev.y))
                speeds.append(p.ego_speed)
                lateral.append(abs(p.route_lateral))
                counts.update(e.type.name for e in res.events)
            p = res.privileged
            rows.append(
                {
                    "map_seed": m,
                    "route_seed": r,
                    "outcome": [e.type.name for e in res.events][-1],
                    "completion": min(1.0, p.route_progress / (p.route_length - 3.0)),
                    "duration_s": res.observation.timestamp,
                    "distance_km": dist / 1000,
                    "mean_speed": float(np.mean(speeds)),
                    "lateral_p99": float(np.percentile(lateral, 99)),
                    **{f"n_{k.lower()}": v for k, v in counts.items()},
                }
            )
            log.info("%s", rows[-1])
    km = sum(r["distance_km"] for r in rows)
    infra = ("collision", "red_light", "off_road", "lane_departure", "route_deviation", "stall")
    result = {
        "episodes": len(rows),
        "outcomes": dict(collections.Counter(r["outcome"] for r in rows)),
        "route_completion_mean": float(np.mean([r["completion"] for r in rows])),
        "distance_km": km,
        "per_km": {k: sum(r.get(f"n_{k}", 0) for r in rows) / km for k in infra},
        "mean_speed": float(np.mean([r["mean_speed"] for r in rows])),
        "wall_seconds": time.perf_counter() - t0,
    }
    (run_dir / "episodes.json").write_text(json.dumps(rows, indent=1))
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
