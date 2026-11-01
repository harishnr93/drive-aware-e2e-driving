"""Deterministic seeding.

Global seeding covers Python, NumPy and PyTorch. Simulator components should not use global
RNG state; they receive their own ``np.random.Generator`` from :func:`make_rng`, with seeds
derived stably (independent of Python's per-process hash randomisation) via :func:`derive_seed`.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def derive_seed(base_seed: int, *keys: str | int) -> int:
    """Stable 63-bit seed from a base seed and a sequence of keys (e.g. split, episode)."""
    payload = "/".join([str(base_seed), *map(str, keys)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") >> 1


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    ``deterministic=True`` additionally requests deterministic cuDNN/algorithms; this can be
    slower and raises on ops without a deterministic implementation.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 (global state is the point)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding: derive NumPy/random seeds from the torch worker seed."""
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)
