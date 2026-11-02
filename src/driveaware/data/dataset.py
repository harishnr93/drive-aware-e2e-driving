"""PyTorch dataset over a consolidated split.

Each sample is one labelled frame t. Policy inputs use only frames <= t of the same episode:
``image`` is a history of H frames (oldest first) sampled every ``history_stride`` control
steps; frames before the episode start are padded by repeating the first frame and flagged in
``history_mask``. The target is the [10, 2] future waypoint label in the ego frame at t.
Privileged quantities are returned only under ``meta`` (for analysis, never model input).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DrivingDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        history: int = 1,
        history_stride: int = 1,
        exclude_expert_failures: bool = True,
        indices: np.ndarray | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.history = history
        self.history_stride = history_stride
        with np.load(self.split_dir / "frames.npz") as z:
            self.frames = {k: z[k] for k in z.files}
        self.summary = json.loads((self.split_dir / "episodes.json").read_text())
        episodes = self.summary["episodes"]
        self.episode_start = np.array([e["start"] for e in episodes], dtype=np.int64)
        usable = self.frames["label_valid"].copy()
        if exclude_expert_failures:
            failed = np.array([e["expert_failure"] for e in episodes], dtype=bool)
            usable &= ~failed[self.frames["episode_index"]]
        self.indices = np.nonzero(usable)[0] if indices is None else np.asarray(indices)
        self._images: np.ndarray | None = None  # opened lazily (fork-safe for workers)

    @property
    def images(self) -> np.ndarray:
        if self._images is None:
            self._images = np.load(self.split_dir / "images.npy", mmap_mode="r")
        return self._images

    def __len__(self) -> int:
        return len(self.indices)

    def history_indices(self, global_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Global frame indices of the input history (oldest first) and their validity mask."""
        ep_start = self.episode_start[self.frames["episode_index"][global_index]]
        offsets = self.history_stride * np.arange(self.history - 1, -1, -1)
        idx = global_index - offsets
        mask = idx >= ep_start
        return np.where(mask, idx, ep_start), mask

    def __getitem__(self, i: int) -> dict:
        g = int(self.indices[i])
        hist, mask = self.history_indices(g)
        f = self.frames
        return {
            "image": torch.from_numpy(np.ascontiguousarray(self.images[hist])),  # [H,h,w,3] u8
            "speed": torch.from_numpy(f["speed"][hist].astype(np.float32)),  # [H]
            "command": torch.tensor(int(f["command"][g]), dtype=torch.long),
            "history_mask": torch.from_numpy(mask),
            "waypoints": torch.from_numpy(f["waypoints"][g]),  # [10, 2] target
            "meta": {
                "global_index": g,
                "episode_index": int(f["episode_index"][g]),
                "frame_index": int(f["frame_index"][g]),
                "perturbed": bool(f["perturbed"][g]),
            },
        }
