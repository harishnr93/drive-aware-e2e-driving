import random

import numpy as np
import torch

from driveaware.core.seeding import derive_seed, make_rng, seed_everything


def test_derive_seed_stable_and_distinct():
    assert derive_seed(0, "train", 3) == derive_seed(0, "train", 3)
    assert derive_seed(0, "train", 3) != derive_seed(0, "val", 3)
    assert derive_seed(0, "train", 3) != derive_seed(1, "train", 3)
    # Pinned value: guards against accidental changes that would silently alter splits.
    assert derive_seed(0, "train", 3) == 6899014207850744573
    assert 0 <= derive_seed(123, "x") < 2**63


def test_make_rng_reproducible():
    assert np.array_equal(make_rng(5).normal(size=4), make_rng(5).normal(size=4))


def test_seed_everything():
    def draw():
        legacy_np = np.random.rand()  # noqa: NPY002 (checks global seeding)
        return random.random(), legacy_np, torch.rand(1).item()

    seed_everything(7)
    a = draw()
    seed_everything(7)
    assert draw() == a
