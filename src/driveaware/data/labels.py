"""Future-trajectory labels.

Indexing convention: frame t is the observation at time t (reset gives t = 0). The action
recorded at frame t is applied after observing frame t and produces frame t + 1. The label for
frame t is the ego rear-axle position at frames t + s, t + 2s, ..., t + n*s (s = waypoint stride
in control steps), expressed in the ego frame at frame t. Labels therefore use only the future,
and inputs at frame t use only frames <= t.
"""

from __future__ import annotations

import numpy as np

from driveaware.core.frames import Pose2D, world_to_ego


def future_waypoints(poses: np.ndarray, t: int, stride: int, num: int) -> np.ndarray:
    """[num, 2] future positions in the ego frame at frame ``t``. Requires t + num*stride < T."""
    idx = t + stride * np.arange(1, num + 1)
    if idx[-1] >= len(poses):
        raise IndexError("label window extends past the end of the episode")
    anchor = Pose2D(*map(float, poses[t]))
    return world_to_ego(poses[idx, :2], anchor).astype(np.float32)


def label_validity(perturbed: np.ndarray, num_frames: int, stride: int, num: int) -> np.ndarray:
    """Frames whose label window exists and was driven without injected noise.

    ``perturbed[t]`` marks that the action applied at frame t contained injected noise. The
    window of frame t is driven by the actions at frames t .. t + num*stride - 1.
    """
    horizon = stride * num
    valid = np.zeros(num_frames, dtype=bool)
    if num_frames <= horizon:
        return valid
    p = np.asarray(perturbed[: num_frames - 1], dtype=np.int64)  # last frame has no action
    csum = np.concatenate([[0], np.cumsum(p)])
    t = np.arange(num_frames - horizon)
    noisy = csum[t + horizon] - csum[t]
    valid[: num_frames - horizon] = noisy == 0
    return valid


def all_waypoints(poses: np.ndarray, valid: np.ndarray, stride: int, num: int) -> np.ndarray:
    """[T, num, 2] labels; NaN where ``valid`` is False or the window is incomplete."""
    out = np.full((len(poses), num, 2), np.nan, dtype=np.float32)
    horizon = stride * num
    for t in range(max(0, len(poses) - horizon)):
        if valid[t]:
            out[t] = future_waypoints(poses, t, stride, num)
    return out
