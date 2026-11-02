import dataclasses

import numpy as np
import torch

from driveaware.evaluation.closed_loop import AgentSpec, scenario_set
from driveaware.introspection.data import NoveltyModel, heuristic_features, learned_features
from driveaware.introspection.rollouts import record_episode
from driveaware.models.temporal.gru_policy import TemporalPolicy
from driveaware.sim.lightweight.scenario import ScenarioSpec, TrafficConfig
from driveaware.training.bc import TrainConfig


def _checkpoint(tmp_path):
    torch.manual_seed(0)
    cfg = TrainConfig(model="temporal", history=4)
    torch.save(
        {"config": cfg, "state_dict": TemporalPolicy(cfg.temporal).state_dict(), "epoch": 0},
        tmp_path / "b.pt",
    )
    return AgentSpec("policy", str(tmp_path / "b.pt"))


def test_recorded_inputs_are_causal(tmp_path):
    """Features at step t must not change when the future is cut off."""
    agent = _checkpoint(tmp_path)
    base = ScenarioSpec(
        0, 0, 0, traffic=TrafficConfig(num_vehicles=4, num_parked=2), route_length=80.0
    )
    spec = scenario_set("unit", [0], 1, base)[0]
    long_ep = record_episode(dataclasses.replace(spec, max_duration=6.0), agent)
    short_ep = record_episode(dataclasses.replace(spec, max_duration=3.0), agent)
    n = len(short_ep["timestamp"])
    assert 0 < n < len(long_ep["timestamp"])
    for key in ("hidden", "feature", "waypoints", "speed", "command"):
        np.testing.assert_array_equal(short_ep[key], long_ep[key][:n], err_msg=key)
    np.testing.assert_array_equal(learned_features(short_ep), learned_features(long_ep)[:n])
    nm = NoveltyModel.fit(np.random.default_rng(0).normal(size=(500, 256)))
    hs, hl = heuristic_features(short_ep, nm), heuristic_features(long_ep, nm)
    for k in hs:
        np.testing.assert_allclose(hs[k], hl[k][:n], err_msg=k)


def test_recorded_shapes(tmp_path):
    agent = _checkpoint(tmp_path)
    base = ScenarioSpec(0, 0, 0, route_length=60.0, max_duration=2.0)
    ep = record_episode(scenario_set("unit", [0], 1, base)[0], agent)
    n = len(ep["timestamp"])
    assert ep["hidden"].shape == (n, 256) and ep["feature"].shape == (n, 256)
    assert ep["waypoints"].shape == (n, 10, 2) and np.isfinite(ep["lateral"]).all()
    assert np.allclose(np.diff(ep["timestamp"]), 0.1)
