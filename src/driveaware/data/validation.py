"""Dataset integrity checks. Each check appends human-readable errors; empty means valid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from driveaware.core.frames import Pose2D, ego_to_world
from driveaware.core.timing import TimingConfig
from driveaware.core.types import Command
from driveaware.data.labels import label_validity


def validate_episode_arrays(
    a: dict[str, np.ndarray], timing: TimingConfig = TimingConfig()
) -> list[str]:
    errors: list[str] = []
    n = len(a["images"])
    for key, arr in a.items():
        if key.startswith("event_"):
            continue
        if len(arr) != n:
            errors.append(f"{key}: length {len(arr)} != {n}")
    if a["images"].dtype != np.uint8 or a["images"].ndim != 4 or a["images"].shape[-1] != 3:
        errors.append(f"images: bad dtype/shape {a['images'].dtype} {a['images'].shape}")
    elif np.any(a["images"].reshape(n, -1).std(axis=1) < 1.0):
        errors.append("images: blank frame(s)")
    dt = np.diff(a["timestamp"])
    if n > 1 and not np.allclose(dt, timing.control_period, atol=1e-6):
        errors.append(f"timestamp: non-uniform steps (min {dt.min()}, max {dt.max()})")
    if not np.all(np.isfinite(a["speed"])) or a["speed"].min() < 0:
        errors.append("speed: negative or non-finite")
    if not np.isin(a["command"], [int(c) for c in Command]).all():
        errors.append("command: out of range")
    if not np.all(np.isfinite(a["ego_pose"])):
        errors.append("ego_pose: non-finite")
    # Kinematic consistency: distance moved ~ mean speed * dt (exact up to arc/chord). The final
    # transition is skipped: a collision stops physics part-way through that control step.
    if n > 2:
        moved = np.linalg.norm(np.diff(a["ego_pose"][:-1, :2], axis=0), axis=1)
        expected = 0.5 * (a["speed"][1:-1] + a["speed"][:-2]) * timing.control_period
        err = np.abs(moved - expected)
        if err.max() > 0.05:
            errors.append(f"kinematics: displacement/speed mismatch up to {err.max():.3f} m")
    for key in ("expert_action", "applied_action"):
        act = a[key][:-1]
        if not np.all(np.isfinite(act)):
            errors.append(f"{key}: non-finite before last frame")
        elif (np.abs(act[:, 0]) > 1).any() or (act[:, 1:] < 0).any() or (act[:, 1:] > 1).any():
            errors.append(f"{key}: out of range")
    clean = ~a["perturbed"][:-1]
    if not np.allclose(a["expert_action"][:-1][clean], a["applied_action"][:-1][clean]):
        errors.append("actions: applied != expert on unperturbed frames")
    return errors


def validate_split(split_dir: str | Path, timing: TimingConfig = TimingConfig(),
                   sample_labels: int = 500, seed: int = 0) -> dict[str, Any]:
    """Validate a consolidated split (frames.npz + images.npy + episodes.json)."""
    split_dir = Path(split_dir)
    errors: list[str] = []
    summary = json.loads((split_dir / "episodes.json").read_text())
    with np.load(split_dir / "frames.npz") as z:
        f = {k: z[k] for k in z.files}
    images = np.load(split_dir / "images.npy", mmap_mode="r")
    n = len(images)
    if n != summary["num_frames"] or any(len(v) != n for v in f.values()):
        errors.append("frame arrays and images.npy are not aligned")
    stride, num = timing.waypoint_stride, timing.num_waypoints
    for i, ep in enumerate(summary["episodes"]):
        sl = slice(ep["start"], ep["end"])
        if not np.all(f["episode_index"][sl] == i):
            errors.append(f"episode {i}: episode_index mismatch")
        if not np.array_equal(f["frame_index"][sl], np.arange(ep["end"] - ep["start"])):
            errors.append(f"episode {i}: frame_index not contiguous")
        expected = label_validity(f["perturbed"][sl], ep["end"] - ep["start"], stride, num)
        if not np.array_equal(expected, f["label_valid"][sl]):
            errors.append(f"episode {i}: label_valid inconsistent with perturbation flags")
    valid = np.nonzero(f["label_valid"])[0]
    if np.isnan(f["waypoints"][valid]).any():
        errors.append("NaN waypoints on labelled frames")
    # Re-derive a sample of labels from poses (catches frame/transform errors).
    rng = np.random.default_rng(seed)
    worst = 0.0
    for g in rng.choice(valid, size=min(sample_labels, len(valid)), replace=False):
        world = ego_to_world(f["waypoints"][g], Pose2D(*map(float, f["ego_pose"][g])))
        target = f["ego_pose"][g + stride * np.arange(1, num + 1), :2]
        worst = max(worst, float(np.abs(world - target).max()))
        if f["episode_index"][g + stride * num] != f["episode_index"][g]:
            errors.append(f"frame {g}: label window crosses an episode boundary")
    if worst > 1e-3:
        errors.append(f"labels: re-derived waypoints differ by up to {worst:.4f} m")
    cmd_hist = np.bincount(f["command"][valid].astype(int), minlength=len(Command))
    return {
        "split": split_dir.name,
        "errors": errors,
        "num_frames": int(n),
        "num_labelled": int(len(valid)),
        "label_recheck_max_error_m": worst,
        "command_histogram": {Command(i).name: int(c) for i, c in enumerate(cmd_hist)},
    }


def validate_disjoint_maps(root: str | Path, splits: list[str]) -> list[str]:
    """No map seed may appear in more than one split (prevents road/route leakage)."""
    seen: dict[int, str] = {}
    errors = []
    for split in splits:
        summary = json.loads((Path(root) / split / "episodes.json").read_text())
        for ep in summary["episodes"]:
            m = ep["spec"]["map_seed"]
            if seen.setdefault(m, split) != split:
                errors.append(f"map_seed {m} appears in both {seen[m]} and {split}")
    return sorted(set(errors))
