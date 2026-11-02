"""Consolidate per-episode files into memory-mapped per-split arrays for training.

Layout of ``<root>/<split>/``:
    episodes/*.npz|json    raw episodes (source of truth)
    images.npy             uint8 [N, H, W, 3], all frames of all episodes, memory-mapped
    frames.npz             per-frame arrays aligned with images.npy (see FRAME_FIELDS)
    episodes.json          per-episode metadata incl. [start, end) frame ranges

Every frame is kept (so temporal models can gather history), and ``label_valid`` marks which
frames may be used as training samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from driveaware.core.timing import TimingConfig
from driveaware.data.collection import load_episode
from driveaware.data.labels import all_waypoints, label_validity

FRAME_FIELDS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "speed",
    "command",
    "ego_pose",
    "expert_action",
    "applied_action",
    "perturbed",
    "label_valid",
    "waypoints",
)


def consolidate_split(split_dir: str | Path, timing: TimingConfig = TimingConfig()) -> dict:
    split_dir = Path(split_dir)
    files = sorted((split_dir / "episodes").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no episodes in {split_dir / 'episodes'}")
    metas = []
    total = 0
    for f in files:  # first pass: sizes only
        meta = json.loads(f.with_suffix(".json").read_text())
        metas.append(meta)
        total += meta["num_frames"]
    h, w = None, None
    images = None
    fields: dict[str, list[np.ndarray]] = {k: [] for k in FRAME_FIELDS}
    start = 0
    stride, num = timing.waypoint_stride, timing.num_waypoints
    for i, (f, meta) in enumerate(zip(files, metas, strict=True)):
        ep = load_episode(f).arrays
        n = len(ep["images"])
        if images is None:
            h, w = ep["images"].shape[1:3]
            images = np.lib.format.open_memmap(split_dir / "images.npy", mode="w+",
                                               dtype=np.uint8, shape=(total, h, w, 3))
        images[start : start + n] = ep["images"]
        valid = label_validity(ep["perturbed"], n, stride, num)
        fields["episode_index"].append(np.full(n, i, dtype=np.int32))
        fields["frame_index"].append(np.arange(n, dtype=np.int32))
        fields["timestamp"].append(ep["timestamp"])
        fields["speed"].append(ep["speed"])
        fields["command"].append(ep["command"])
        fields["ego_pose"].append(ep["ego_pose"])
        fields["expert_action"].append(ep["expert_action"])
        fields["applied_action"].append(ep["applied_action"])
        fields["perturbed"].append(ep["perturbed"])
        fields["label_valid"].append(valid)
        fields["waypoints"].append(all_waypoints(ep["ego_pose"], valid, stride, num))
        meta.update({"file": f.name, "start": start, "end": start + n,
                     "num_labelled": int(valid.sum())})
        start += n
    images.flush()
    del images
    arrays = {k: np.concatenate(v) for k, v in fields.items()}
    np.savez(split_dir / "frames.npz", **arrays)
    summary = {
        "num_episodes": len(files),
        "num_frames": int(total),
        "num_labelled": int(arrays["label_valid"].sum()),
        "image_shape": [int(h), int(w), 3],
        "timing": {"control_period": timing.control_period, "waypoint_dt": timing.waypoint_dt,
                   "num_waypoints": num},
        "episodes": metas,
    }
    (split_dir / "episodes.json").write_text(json.dumps(summary, indent=1))
    return summary
