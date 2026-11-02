"""Collect, consolidate and validate an expert dataset.

Usage:
    python scripts/collect_dataset.py --config configs/data/expert_v1.yaml
    python scripts/collect_dataset.py --set name=expert_tiny splits.train.num_maps=1 ...

Output: data/<name>/<split>/{episodes/, images.npy, frames.npz, episodes.json} and
data/<name>/dataset.json (config, environment, per-split summaries and validation reports).
Collection is deterministic given the config; episodes already on disk are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from driveaware.core.config import load_config
from driveaware.core.hardware import environment_info
from driveaware.core.logging import setup_logging
from driveaware.core.run import REPO_ROOT
from driveaware.data.collection import (
    EpisodeRequest,
    NoiseConfig,
    collect_episode,
    load_episode,
    plan_requests,
    save_episode,
)
from driveaware.data.storage import consolidate_split
from driveaware.data.validation import (
    validate_disjoint_maps,
    validate_episode_arrays,
    validate_split,
)
from driveaware.sim.lightweight.scenario import ScenarioSpec


@dataclass
class SplitConfig:
    map_seed_start: int = 0
    num_maps: int = 1
    routes_per_map: int = 1


@dataclass
class DatasetConfig:
    name: str = "expert_v1"
    dataset_seed: int = 0
    route_length: float = 300.0
    max_duration: float = 150.0
    workers: int = 4
    splits: dict[str, SplitConfig] = field(default_factory=dict)
    noise: NoiseConfig = field(default_factory=NoiseConfig)


_SIM = None


def _work(req: EpisodeRequest, noise: NoiseConfig, out_dir: str) -> dict:
    global _SIM
    from driveaware.sim.lightweight.env import LightweightDrivingSim

    if _SIM is None:
        _SIM = LightweightDrivingSim()
    t0 = time.perf_counter()
    data = collect_episode(req, noise, _SIM)
    errors = validate_episode_arrays(data.arrays)
    save_episode(data, out_dir)
    return {
        "episode_id": req.episode_id,
        "frames": data.meta["num_frames"],
        "seconds": time.perf_counter() - t0,
        "outcome": data.meta["outcome"],
        "expert_failure": data.meta["expert_failure"],
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/expert_v1.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--root", default=str(REPO_ROOT / "data"))
    args = parser.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(DatasetConfig, args.config, overrides)
    root = Path(args.root) / cfg.name
    root.mkdir(parents=True, exist_ok=True)
    log = setup_logging(root)
    base = ScenarioSpec(0, 0, 0, route_length=cfg.route_length, max_duration=cfg.max_duration)

    report: dict = {"config": asdict(cfg), "environment": environment_info(REPO_ROOT), "splits": {}}
    for split, sc in cfg.splits.items():
        maps = list(range(sc.map_seed_start, sc.map_seed_start + sc.num_maps))
        reqs = plan_requests(split, maps, sc.routes_per_map, base, cfg.noise, cfg.dataset_seed)
        ep_dir = root / split / "episodes"
        todo = [r for r in reqs if not (ep_dir / f"{split}_{r.episode_id:05d}.npz").exists()]
        log.info("%s: %d episodes planned, %d to collect", split, len(reqs), len(todo))
        t0 = time.perf_counter()
        results, frames = [], 0
        with ProcessPoolExecutor(max_workers=min(cfg.workers, max(1, len(todo)))) as pool:
            futures = [pool.submit(_work, r, cfg.noise, str(ep_dir)) for r in todo]
            for k, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                frames += r["frames"]
                if r["errors"]:
                    log.error("episode %d invalid: %s", r["episode_id"], r["errors"])
                if k % 10 == 0 or k == len(futures):
                    el = time.perf_counter() - t0
                    log.info(
                        "%s: %d/%d episodes, %d frames, %.0f frames/s",
                        split,
                        k,
                        len(futures),
                        frames,
                        frames / el,
                    )
        # Re-validate episodes collected in earlier runs as well.
        for r in reqs:
            if r in todo:
                continue
            arrays = load_episode(ep_dir / f"{split}_{r.episode_id:05d}.npz").arrays
            errs = validate_episode_arrays(arrays)
            if errs:
                log.error("episode %d invalid: %s", r.episode_id, errs)
        summary = consolidate_split(root / split)
        validation = validate_split(root / split)
        n_fail = sum(e["expert_failure"] for e in summary["episodes"])
        outcomes: dict[str, int] = {}
        for e in summary["episodes"]:
            outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1
        report["splits"][split] = {
            "map_seeds": [maps[0], maps[-1]],
            "num_episodes": summary["num_episodes"],
            "num_frames": summary["num_frames"],
            "num_labelled": summary["num_labelled"],
            "noisy_episodes": int(sum(e["noise"] for e in summary["episodes"])),
            "expert_failure_episodes": int(n_fail),
            "outcomes": outcomes,
            "collect_seconds": round(time.perf_counter() - t0, 1),
            "images_gb": round(os.path.getsize(root / split / "images.npy") / 1e9, 3),
            "episodes_gb": round(sum(p.stat().st_size for p in ep_dir.glob("*")) / 1e9, 3),
            "mean_speed": float(np.mean(np.load(root / split / "frames.npz")["speed"])),
            "validation": validation,
        }
        log.info(
            "%s: %s", split, {k: v for k, v in report["splits"][split].items() if k != "validation"}
        )
        if validation["errors"]:
            log.error("%s validation errors: %s", split, validation["errors"])
    report["leakage_errors"] = validate_disjoint_maps(root, list(cfg.splits))
    if report["leakage_errors"]:
        log.error("split leakage: %s", report["leakage_errors"])
    (root / "dataset.json").write_text(json.dumps(report, indent=1, default=str))
    ok = not report["leakage_errors"] and all(
        not s["validation"]["errors"] for s in report["splits"].values()
    )
    log.info("dataset %s written to %s (valid=%s)", cfg.name, root, ok)


if __name__ == "__main__":
    main()
