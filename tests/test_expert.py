import pytest

from driveaware.core.types import EventType
from driveaware.expert.expert import RuleBasedExpert
from driveaware.sim.lightweight.env import LightweightDrivingSim, SimConfig
from driveaware.sim.lightweight.scenario import ScenarioSpec, TrafficConfig


def _drive(spec):
    sim = LightweightDrivingSim(SimConfig(render=False))
    res = sim.reset(spec)
    expert = RuleBasedExpert()
    events = []
    while not res.done:
        res = sim.step(expert(res))
        events += [e.type for e in res.events]
    return res, events


@pytest.mark.parametrize("route_seed", [0, 1])
def test_expert_completes_with_traffic_without_infractions(route_seed):
    spec = ScenarioSpec(
        300,
        route_seed,
        route_seed,
        route_length=200.0,
        traffic=TrafficConfig(num_vehicles=10, ped_mean_interval=6.0),
    )
    _, events = _drive(spec)
    assert events[-1] == EventType.ROUTE_COMPLETE
    for bad in (
        EventType.COLLISION,
        EventType.RED_LIGHT,
        EventType.OFF_ROAD,
        EventType.LANE_DEPARTURE,
        EventType.STALL,
    ):
        assert bad not in events


def test_expert_stops_for_red_and_records_reason():
    sim = LightweightDrivingSim(SimConfig(render=False))
    res = sim.reset(
        ScenarioSpec(
            0, 0, 0, traffic=TrafficConfig(num_vehicles=0, num_parked=0, ped_mean_interval=1e9)
        )
    )
    expert = RuleBasedExpert()
    stopped_at_red = False
    while not res.done:
        res = sim.step(expert(res))
        nxt = res.privileged.extra["next_stop"]
        if nxt is not None and nxt["state"].name == "RED" and res.privileged.ego_speed < 0.05:
            front = res.privileged.route_progress + 3.65  # rear axle + length - rear overhang
            assert 0.0 < nxt["s"] - front < 6.0  # waiting just before the stop line
            assert expert.last_decision.gap is not None
            stopped_at_red = True
    assert stopped_at_red
