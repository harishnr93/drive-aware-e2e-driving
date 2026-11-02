import inspect

import numpy as np
import pytest
import torch

from driveaware.agents.policy_agent import AgentConfig, WaypointPolicyAgent
from driveaware.core.types import IMAGE_SHAPE, Command, Observation
from driveaware.evaluation.closed_loop import (
    AgentSpec,
    bootstrap_ci,
    paired_difference,
    run_episode,
    scenario_set,
    summarize,
)
from driveaware.models.trajectory.single_frame import SingleFramePolicy
from driveaware.sim.lightweight.scenario import ScenarioSpec, TrafficConfig
from driveaware.training.bc import TrainConfig, photometric_jitter


def _obs(t=0.0, speed=3.0):
    img = np.random.default_rng(0).integers(0, 255, IMAGE_SHAPE, dtype=np.uint8)
    return Observation(t, int(t * 10), img, speed, Command.LEFT)


def test_policy_shapes_and_grad():
    m = SingleFramePolicy()
    img = torch.randint(0, 255, (3, 96, 160, 3), dtype=torch.uint8)
    out = m(img, torch.rand(3) * 10, torch.tensor([0, 1, 3]))
    assert out.shape == (3, 10, 2)
    out.sum().backward()
    assert m.encoder.backbone[0][0].weight.grad is not None


def test_normalisation_buffers_in_state_dict():
    sd = SingleFramePolicy().state_dict()
    assert "image_mean" in sd and "image_std" in sd


def test_agent_sees_only_observations():
    params = list(inspect.signature(WaypointPolicyAgent.__call__).parameters)
    assert params == ["self", "obs"]


def test_agent_target_speed_from_waypoints():
    agent = WaypointPolicyAgent(SingleFramePolicy())
    wp = np.column_stack([np.arange(1, 11) * 1.0, np.zeros(10)])  # 1 m per 0.2 s = 5 m/s
    assert agent.target_speed(wp) == pytest.approx(5.0)
    assert agent.target_speed(np.zeros((10, 2))) == 0.0


def test_agent_history_padding_matches_dataset():
    class Probe(torch.nn.Module):
        def forward(self, image, speed, command):
            self.seen = speed.clone()
            return torch.zeros(1, 10, 2)

    probe = Probe()
    agent = WaypointPolicyAgent(probe, config=AgentConfig(history=3, history_stride=2))
    for k in range(3):
        agent(_obs(t=0.1 * k, speed=float(k)))
    # frames seen: speeds 0,1,2; history 3 with stride 2 -> indices [0(pad), 0, 2]
    assert probe.seen.tolist() == [[0.0, 0.0, 2.0]]


def test_photometric_jitter_keeps_dtype_and_range():
    img = torch.randint(0, 255, (4, 96, 160, 3), dtype=torch.uint8)
    out = photometric_jitter(img, TrainConfig(), torch.Generator().manual_seed(0))
    assert out.dtype == torch.uint8 and out.shape == img.shape and not torch.equal(out, img)


def test_closed_loop_episode_with_untrained_policy(tmp_path):
    torch.manual_seed(0)  # weights must not depend on the global RNG state left by other tests
    torch.save(
        {"config": TrainConfig(), "state_dict": SingleFramePolicy().state_dict(), "epoch": 0},
        tmp_path / "m.pt",
    )
    base = ScenarioSpec(
        0,
        0,
        0,
        traffic=TrafficConfig(num_vehicles=0, num_parked=0, ped_mean_interval=1e9),
        route_length=50.0,
        max_duration=5.0,
    )
    spec = scenario_set("unit", [0], 1, base)[0]
    row = run_episode(spec, AgentSpec("policy", str(tmp_path / "m.pt")))
    # The outcome is always the terminal event, never a co-occurring infraction.
    assert row["outcome"] in {
        "TIMEOUT",
        "OFF_ROAD",
        "COLLISION",
        "ROUTE_DEVIATION",
        "STALL",
        "ROUTE_COMPLETE",
    }
    assert row["latency_ms_p50"] > 0, row
    assert 0.0 <= row["completion"] <= 1.0, row
    s = summarize([row, row])
    assert s["episodes"] == 2


def test_paired_difference_and_bootstrap():
    a = [{"map_seed": i, "route_seed": 0, "completion": 1.0} for i in range(5)]
    b = [{"map_seed": i, "route_seed": 0, "completion": 0.5} for i in range(5)]
    d = paired_difference(a, b, "completion")
    assert d["mean"] == pytest.approx(0.5) and d["ci95"] == (pytest.approx(0.5), pytest.approx(0.5))
    lo, hi = bootstrap_ci(np.array([0, 1] * 50))
    assert lo < 0.5 < hi
    with pytest.raises(AssertionError):
        paired_difference(a, b[::-1], "completion")


def test_summarize_agent_that_never_moves():
    row = {
        "map_seed": 0,
        "route_seed": 0,
        "outcome": "TIMEOUT",
        "completion": 0.0,
        "driving_score": 0.0,
        "distance_km": 0.0,
        "mean_speed": 0.0,
        "end_lane_kind": "road",
        "end_lane_turn": "LANE_FOLLOW",
    }
    s = summarize([row])
    assert s["per_km"]["collision"] is None and s["route_completion"] == 0.0
