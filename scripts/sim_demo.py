"""Phase 2 simulator demo: run the traffic-unaware route follower on a few scenarios.

Writes replay logs (spec + actions), optional MP4 videos (camera | top-down), and a summary
with outcomes, event counts and step timing to experiments/<run>/.

Usage:  python scripts/sim_demo.py [--config configs/simulation/demo.yaml] [--set key=value ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from dataclasses import dataclass, field

import numpy as np
import yaml

from driveaware.core.config import load_config
from driveaware.core.hardware import LatencyStats, snapshot
from driveaware.core.logging import setup_logging
from driveaware.core.run import create_run_dir
from driveaware.expert.route_follower import RouteFollower
from driveaware.sim.lightweight.env import LightweightDrivingSim
from driveaware.sim.lightweight.renderer import TopDownRenderer
from driveaware.sim.lightweight.scenario import ScenarioSpec, TrafficConfig
from driveaware.viz.replay import EpisodeLog, VideoWriter, compose_frame


@dataclass
class DemoConfig:
    map_seed: int = 0
    route_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    traffic_seed: int = 0
    num_vehicles: int = 12
    num_parked: int = 6
    route_length: float = 400.0
    max_duration: float = 120.0
    video: bool = True
    video_episodes: int = 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulation/demo.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(DemoConfig, args.config, overrides)
    run_dir = create_run_dir("sim_demo", cfg)
    log = setup_logging(run_dir)

    sim = LightweightDrivingSim()
    traffic = TrafficConfig(num_vehicles=cfg.num_vehicles, num_parked=cfg.num_parked)
    step_lat = LatencyStats()
    episodes = []
    totals: collections.Counter[str] = collections.Counter()
    for k, route_seed in enumerate(cfg.route_seeds):
        spec = ScenarioSpec(
            cfg.map_seed,
            route_seed,
            cfg.traffic_seed,
            traffic=traffic,
            route_length=cfg.route_length,
            max_duration=cfg.max_duration,
        )
        t0 = time.perf_counter()
        res = sim.reset(spec)
        reset_ms = (time.perf_counter() - t0) * 1e3
        driver = RouteFollower()
        ep_log = EpisodeLog(spec)
        video = None
        if cfg.video and k < cfg.video_episodes:
            video = VideoWriter(run_dir / f"episode_{route_seed}.mp4")
            topdown = TopDownRenderer(sim.episode.road_map)
            video.write(compose_frame(res, topdown))
        while not res.done:
            action = driver(res)
            with step_lat.measure():
                res = sim.step(action)
            ep_log.add(action, res)
            if video is not None:
                video.write(compose_frame(res, topdown))
        if video is not None:
            video.close()
        ep_log.save(run_dir / f"episode_{route_seed}.json")
        counts = collections.Counter(name for _, name, _ in ep_log.events)
        totals.update(counts)
        summary = {
            "route_seed": route_seed,
            "outcome": ep_log.outcome,
            "steps": len(ep_log.actions),
            "sim_time_s": round(res.observation.timestamp, 2),
            "route_progress_m": round(res.privileged.route_progress, 1),
            "route_length_m": round(res.privileged.route_length, 1),
            "events": dict(counts),
            "reset_ms": round(reset_ms, 1),
        }
        episodes.append(summary)
        log.info("episode %s: %s", route_seed, summary)

    result = {
        "driver": "traffic-unaware route follower (simulator test driver, not the expert)",
        "episodes": episodes,
        "event_totals": dict(totals),
        "step_latency": step_lat.summary(),
        "resources": snapshot().as_dict(),
        "mean_completion": float(
            np.mean([e["route_progress_m"] / e["route_length_m"] for e in episodes])
        ),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
