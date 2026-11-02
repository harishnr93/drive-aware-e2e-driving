"""Closed-loop (and open-loop) evaluation of agents on a fixed, paired scenario set.

Usage:
    python scripts/eval_closed_loop.py --set-name test_id --expert \
        --checkpoints experiments/*_model_a_seed*/best.pt [--workers 4]

Writes experiments/<ts>_closed_loop_<set>/ with per-episode rows, summaries, paired differences
against the expert, and open-loop metrics of each checkpoint on the matching dataset split.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from driveaware.core.config import load_config
from driveaware.core.logging import setup_logging
from driveaware.core.run import REPO_ROOT, create_run_dir
from driveaware.evaluation.closed_loop import (
    AgentSpec,
    evaluate,
    paired_difference,
    scenario_set,
    summarize,
)
from driveaware.evaluation.open_loop import evaluate_checkpoint, reference_baselines
from driveaware.sim.lightweight.scenario import ScenarioSpec


@dataclass
class SetConfig:
    map_seed_start: int = 0
    num_maps: int = 1
    routes_per_map: int = 1


@dataclass
class ClosedLoopConfig:
    route_length: float = 300.0
    max_duration: float = 150.0
    seed: int = 0
    sets: dict[str, SetConfig] = field(default_factory=dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/closed_loop.yaml")
    parser.add_argument("--set-name", default="val")
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--checkpoints", nargs="*", default=[])
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data/expert_v1"))
    args = parser.parse_args()
    cfg = load_config(ClosedLoopConfig, args.config)
    sc = cfg.sets[args.set_name]
    base = ScenarioSpec(0, 0, 0, route_length=cfg.route_length, max_duration=cfg.max_duration)
    maps = list(range(sc.map_seed_start, sc.map_seed_start + sc.num_maps))
    scenarios = scenario_set(args.set_name, maps, sc.routes_per_map, base, cfg.seed)
    run_dir = create_run_dir(f"closed_loop_{args.set_name}", cfg)
    log = setup_logging(run_dir)

    agents: dict[str, AgentSpec] = {}
    if args.expert:
        agents["expert"] = AgentSpec("expert")
    for ck in args.checkpoints:
        agents[Path(ck).parent.name] = AgentSpec("policy", str(Path(ck).resolve()), args.history)

    report: dict = {"set": args.set_name, "num_scenarios": len(scenarios), "agents": {}}
    rows_by_agent = {}
    for name, spec in agents.items():
        t0 = time.perf_counter()
        rows = evaluate(scenarios, spec, args.workers)
        rows_by_agent[name] = rows
        summary = summarize(rows)
        summary["wall_seconds"] = time.perf_counter() - t0
        report["agents"][name] = {"spec": spec.__dict__, "closed_loop": summary}
        (run_dir / f"episodes_{name}.json").write_text(json.dumps(rows, indent=1))
        log.info(
            "%s: %s",
            name,
            {
                k: summary[k]
                for k in (
                    "route_completion",
                    "success_rate",
                    "collision_episode_rate",
                    "driving_score",
                )
            },
        )
    split_dir = Path(args.data_root) / args.set_name
    if split_dir.exists():
        report["open_loop_reference"] = reference_baselines(split_dir)
        for name, spec in agents.items():
            if spec.kind == "policy":
                report["agents"][name]["open_loop"] = evaluate_checkpoint(
                    spec.checkpoint, split_dir
                )
    if "expert" in rows_by_agent:
        for name, rows in rows_by_agent.items():
            if name == "expert":
                continue
            report["agents"][name]["vs_expert"] = {
                k: paired_difference(rows, rows_by_agent["expert"], k)
                for k in ("completion", "driving_score", "collision")
            }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
