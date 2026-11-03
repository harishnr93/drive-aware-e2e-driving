"""Run directories: every experiment stores its resolved config and environment record."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from driveaware.core.config import save_config
from driveaware.core.hardware import environment_info

REPO_ROOT = Path(__file__).resolve().parents[3]


def create_run_dir(name: str, cfg: Any, root: str | Path | None = None) -> Path:
    """Create ``experiments/<timestamp>_<name>/`` with ``config.yaml`` and ``env.json``."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "run"
    base = Path(root) if root is not None else REPO_ROOT / "experiments"
    run_dir = base / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}"
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)
    save_config(cfg, run_dir / "config.yaml")
    with open(run_dir / "env.json", "w") as fh:
        json.dump(environment_info(REPO_ROOT), fh, indent=2)
    return run_dir


def atomic_savez(path: str | Path, **arrays: Any) -> Path:
    """np.savez_compressed via a temporary file + rename: a killed process never leaves a
    truncated file at ``path`` (resumable pipelines treat an existing file as complete)."""
    import os

    import numpy as np

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path
