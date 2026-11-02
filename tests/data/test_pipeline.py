"""End-to-end data pipeline on a tiny collection (no traffic, short routes)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from driveaware.data.collection import (
    NoiseConfig,
    collect_episode,
    load_episode,
    plan_requests,
    save_episode,
)
from driveaware.data.dataset import DrivingDataset
from driveaware.data.storage import consolidate_split
from driveaware.data.validation import (
    validate_disjoint_maps,
    validate_episode_arrays,
    validate_split,
)
from driveaware.sim.lightweight.env import LightweightDrivingSim
from driveaware.sim.lightweight.scenario import ScenarioSpec, TrafficConfig

BASE = ScenarioSpec(0, 0, 0, traffic=TrafficConfig(num_vehicles=2, num_parked=1),
                    route_length=60.0, max_duration=40.0)


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    sim = LightweightDrivingSim()
    noise = NoiseConfig(episode_prob=1.0, mean_interval=2.0)
    for split, maps in (("train", [11]), ("val", [12])):
        reqs = plan_requests(split, maps, 2, BASE, noise, dataset_seed=0)
        for r in reqs:
            save_episode(collect_episode(r, noise, sim), root / split / "episodes")
        consolidate_split(root / split)
    return root


def test_plan_is_deterministic_and_seeds_differ():
    noise = NoiseConfig()
    a = plan_requests("train", [1, 2], 3, BASE, noise, 0)
    b = plan_requests("train", [1, 2], 3, BASE, noise, 0)
    assert a == b
    assert len({r.spec.route_seed for r in a}) == len(a)
    c = plan_requests("val", [1, 2], 3, BASE, noise, 0)
    assert {r.spec.route_seed for r in a}.isdisjoint({r.spec.route_seed for r in c})


def test_collection_is_reproducible():
    noise = NoiseConfig(episode_prob=1.0, mean_interval=2.0)
    req = replace(plan_requests("train", [11], 1, BASE, noise, 0)[0], noise=True)
    a = collect_episode(req, noise).arrays
    b = collect_episode(req, noise).arrays
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_episode_roundtrip_and_validity(dataset_root):
    files = sorted((dataset_root / "train" / "episodes").glob("*.npz"))
    ep = load_episode(files[0])
    assert validate_episode_arrays(ep.arrays) == []
    assert ep.meta["noise"] and ep.arrays["perturbed"].any()
    assert ep.arrays["images"].shape[1:] == (96, 160, 3)


def test_split_validation_and_leakage(dataset_root):
    for split in ("train", "val"):
        report = validate_split(dataset_root / split)
        assert report["errors"] == [], report["errors"]
        assert report["label_recheck_max_error_m"] < 1e-4
    assert validate_disjoint_maps(dataset_root, ["train", "val"]) == []


def test_validation_detects_corruption(dataset_root):
    files = sorted((dataset_root / "train" / "episodes").glob("*.npz"))
    a = dict(load_episode(files[0]).arrays)
    a["timestamp"] = a["timestamp"].copy()
    a["timestamp"][5] += 0.03
    pose = a["ego_pose"].copy()
    pose[10, 0] += 1.0
    a["ego_pose"] = pose
    errors = validate_episode_arrays(a)
    assert any("timestamp" in e for e in errors) and any("kinematics" in e for e in errors)


def test_dataset_sample_contract(dataset_root):
    ds = DrivingDataset(dataset_root / "train", history=4, history_stride=2)
    assert len(ds) > 0
    s = ds[0]
    assert s["image"].shape == (4, 96, 160, 3) and s["image"].dtype == torch.uint8
    assert s["speed"].shape == (4,) and s["waypoints"].shape == (10, 2)
    assert torch.isfinite(s["waypoints"]).all()
    assert set(s) == {"image", "speed", "command", "history_mask", "waypoints", "meta"}


def test_history_is_causal_and_within_episode(dataset_root):
    ds = DrivingDataset(dataset_root / "train", history=5, history_stride=3)
    f = ds.frames
    for i in range(0, len(ds), 7):
        g = int(ds.indices[i])
        hist, mask = ds.history_indices(g)
        assert hist[-1] == g and (np.diff(hist) >= 0).all()  # ordered, ends at t
        assert (hist <= g).all()  # no future frames as input
        assert (f["episode_index"][hist] == f["episode_index"][g]).all()
        assert mask[-1]
        # Padded positions repeat the episode's first frame.
        assert (f["frame_index"][hist[~mask]] == 0).all()


def test_labels_come_from_future_of_same_episode(dataset_root):
    ds = DrivingDataset(dataset_root / "val")
    f = ds.frames
    g = ds.indices
    assert (f["episode_index"][g + 20] == f["episode_index"][g]).all()
    assert (f["timestamp"][g + 20] - f["timestamp"][g] == pytest.approx(2.0))


def test_dataloader_batches(dataset_root):
    ds = DrivingDataset(dataset_root / "train", history=2)
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, num_workers=2,
                                                  shuffle=True)))
    assert batch["image"].shape == (4, 2, 96, 160, 3)
    assert batch["command"].dtype == torch.long
