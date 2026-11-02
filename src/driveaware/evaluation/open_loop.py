"""Open-loop trajectory metrics for checkpoints and trivial reference predictors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from driveaware.data.dataset import DrivingDataset
from driveaware.data.labels import SITUATIONS, motion_situations
from driveaware.training.bc import evaluate_open_loop, load_policy, resolve_device


def evaluate_checkpoint(
    checkpoint: str | Path, split_dir: str | Path, device: str = "auto", batch_size: int = 256
) -> dict:
    """Open-loop metrics; history settings come from the checkpoint's own training config."""
    dev = resolve_device(device)
    model, cfg = load_policy(checkpoint, dev)
    history = getattr(cfg, "history", 1) if getattr(model, "uses_history", False) else 1
    ds = DrivingDataset(
        split_dir, history=history, history_stride=getattr(cfg, "history_stride", 1)
    )
    loader = DataLoader(ds, batch_size=max(16, batch_size // history), num_workers=4)
    return evaluate_open_loop(model, loader, dev)


def reference_baselines(split_dir: str | Path, waypoint_dt: float = 0.2) -> dict:
    """ADE/FDE of predictors that need no learning (context for the learned numbers)."""
    ds = DrivingDataset(split_dir)
    f = ds.frames
    target = f["waypoints"][ds.indices]  # [N, 10, 2]
    speed = f["speed"][ds.indices]
    t = waypoint_dt * np.arange(1, target.shape[1] + 1)
    preds = {
        "stand_still": np.zeros_like(target),
        "constant_velocity_straight": np.stack(
            [speed[:, None] * t[None], np.zeros((len(speed), len(t)))], axis=-1
        ),
    }
    situ = motion_situations(speed, target)
    out = {}
    for name, p in preds.items():
        err = np.linalg.norm(p - target, axis=-1)
        out[name] = {"ade": float(err.mean()), "fde": float(err[:, -1].mean())}
        for s in SITUATIONS:
            m = situ == s
            if m.any():
                out[name][f"ade_{s}"] = float(err[m].mean())
    return out
