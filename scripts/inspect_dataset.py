"""Inspect a consolidated dataset split: statistics plus a montage of frames with their
future-waypoint labels projected into the camera image (a direct frame/label alignment check).

Usage:  python scripts/inspect_dataset.py data/expert_v1/train [--num 16] [--out montage.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from driveaware.core.types import Command
from driveaware.data.dataset import DrivingDataset
from driveaware.sim.lightweight.renderer import CameraConfig, CameraRenderer
from driveaware.sim.lightweight.roadmap import generate_map


def statistics(ds: DrivingDataset) -> dict:
    f = ds.frames
    idx = ds.indices
    wp = f["waypoints"][idx]
    eps = ds.summary["episodes"]
    outcomes: dict[str, int] = {}
    for e in eps:
        outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1
    speed = f["speed"][idx]
    return {
        "episodes": len(eps),
        "frames": int(len(f["speed"])),
        "samples": int(len(idx)),
        "noisy_episodes": int(sum(e["noise"] for e in eps)),
        "expert_failure_episodes": int(sum(e["expert_failure"] for e in eps)),
        "outcomes": outcomes,
        "command_fraction": {
            Command(c).name: round(float(np.mean(f["command"][idx] == c)), 4) for c in Command
        },
        "speed_mps": {
            "mean": float(speed.mean()),
            "p50": float(np.median(speed)),
            "stationary_fraction": float(np.mean(speed < 0.1)),
        },
        "final_waypoint_m": {
            "x_mean": float(wp[:, -1, 0].mean()),
            "x_max": float(wp[:, -1, 0].max()),
            "y_abs_p95": float(np.percentile(np.abs(wp[:, -1, 1]), 95)),
        },
        # Labelled frames within 2 s after injected noise: off-nominal starts, clean futures.
        "recovery_samples": int(recovery_mask(ds)[idx].sum()),
    }


def recovery_mask(ds: DrivingDataset, window: int = 20) -> np.ndarray:
    f = ds.frames
    p = f["perturbed"].astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(p)])
    t = np.arange(len(p))
    start = np.maximum(t - window, ds.episode_start[f["episode_index"]])
    return (csum[t] - csum[start]) > 0


def montage(ds: DrivingDataset, num: int, seed: int, scale: int = 3) -> np.ndarray:
    # The projection depends only on the camera config; any map serves to build the renderer.
    cam = CameraRenderer(generate_map(0), CameraConfig())
    rng = np.random.default_rng(seed)
    tiles = []
    for i in rng.choice(len(ds), size=min(num, len(ds)), replace=False):
        s = ds[int(i)]
        img = s["image"][-1].numpy()
        tile = cv2.resize(
            img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST
        )
        uv, front = cam.project_ego(s["waypoints"].numpy())
        for (u, v), ok in zip(uv * scale, front, strict=True):
            if ok:
                cv2.circle(tile, (int(u), int(v)), 4, (0, 255, 255), -1, cv2.LINE_AA)
        text = (
            f"ep{s['meta']['episode_index']} f{s['meta']['frame_index']} "
            f"v={s['speed'][-1]:.1f} {Command(int(s['command'])).name}"
            + (" PERTURBED" if s["meta"]["perturbed"] else "")
        )
        cv2.putText(
            tile, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r : r + 4]) for r in range(0, len(tiles), 4)]
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split_dir")
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    ds = DrivingDataset(args.split_dir)
    print(json.dumps(statistics(ds), indent=2))
    out = Path(args.out or Path(args.split_dir) / "montage.png")
    cv2.imwrite(str(out), cv2.cvtColor(montage(ds, args.num, args.seed), cv2.COLOR_RGB2BGR))
    print(f"montage written to {out}")


if __name__ == "__main__":
    main()
