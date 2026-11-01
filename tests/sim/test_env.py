import math

import numpy as np
import pytest

from driveaware.control import WaypointController
from driveaware.core.types import IMAGE_SHAPE, Command, ControlCommand, EventType, Observation
from driveaware.sim.interface import DrivingSimulator
from driveaware.sim.lightweight.env import EventConfig, LightweightDrivingSim, SimConfig
from driveaware.sim.lightweight.roadmap import Sem, SignalState
from driveaware.sim.lightweight.scenario import ScenarioSpec, TrafficConfig
from driveaware.sim.lightweight.vehicle import VehicleState

from .helpers import route_follow_action, run_episode

EMPTY = TrafficConfig(num_vehicles=0, num_parked=0, ped_mean_interval=1e9)


def _types(events):
    return [e.type for e in events]


def test_implements_protocol():
    sim: DrivingSimulator = LightweightDrivingSim()
    assert hasattr(sim, "get_state") and hasattr(sim, "set_state")


def test_observation_contract():
    res = LightweightDrivingSim().reset(ScenarioSpec(0, 0, 0))
    obs = res.observation
    assert isinstance(obs, Observation)
    assert obs.image.shape == IMAGE_SHAPE and obs.image.dtype == np.uint8
    assert isinstance(obs.command, Command) and obs.timestamp == 0.0 and obs.frame_id == 0
    assert obs.image.std() > 5  # not blank


def test_timing_advances_one_control_period():
    sim = LightweightDrivingSim(SimConfig(render=False))
    sim.reset(ScenarioSpec(0, 0, 0))
    res = sim.step(ControlCommand(0, 0.5, 0))
    assert res.observation.timestamp == pytest.approx(0.1)
    assert res.observation.frame_id == 1


def _rollout(sim, spec, n, seed=0):
    rng = np.random.default_rng(seed)
    res = sim.reset(spec)
    out = []
    for _ in range(n):
        if res.done:
            break
        res = sim.step(ControlCommand.clipped(rng.uniform(-0.1, 0.1), 0.4, 0.0))
        out.append(res)
    return out


def test_reset_is_deterministic():
    spec = ScenarioSpec(1, 2, 3)
    a = _rollout(LightweightDrivingSim(), spec, 60)
    b = _rollout(LightweightDrivingSim(), spec, 60)
    assert len(a) == len(b)
    for ra, rb in zip(a, b, strict=True):
        assert np.array_equal(ra.observation.image, rb.observation.image)
        assert ra.privileged.ego_pose == rb.privileged.ego_pose
        assert [(x.x, x.y) for x in ra.privileged.agents] == [
            (x.x, x.y) for x in rb.privileged.agents
        ]
        assert _types(ra.events) == _types(rb.events)


def test_traffic_seed_changes_traffic_only():
    a = LightweightDrivingSim().reset(ScenarioSpec(0, 0, 0))
    b = LightweightDrivingSim().reset(ScenarioSpec(0, 0, 1))
    assert a.privileged.ego_pose == b.privileged.ego_pose  # same route start
    assert [(x.x, x.y) for x in a.privileged.agents] != [(x.x, x.y) for x in b.privileged.agents]


def test_snapshot_restore_is_bit_identical():
    sim = LightweightDrivingSim()
    spec = ScenarioSpec(0, 1, 2, traffic=TrafficConfig(ped_mean_interval=3.0))
    res = sim.reset(spec)
    ctrl = WaypointController()
    for _ in range(40):
        res = sim.step(route_follow_action(res, ctrl))
    snap = sim.get_state()
    actions = [ControlCommand.clipped(0.05 * math.sin(i / 5), 0.3, 0.0) for i in range(30)]

    def branch():
        out = [sim.set_state(snap)]
        for a in actions:
            if out[-1].done:
                break
            out.append(sim.step(a))
        return out

    first, second = branch(), branch()
    assert len(first) == len(second)
    for ra, rb in zip(first, second, strict=True):
        assert np.array_equal(ra.observation.image, rb.observation.image)
        assert ra.privileged.ego_pose == rb.privileged.ego_pose
        assert len(ra.privileged.agents) == len(rb.privileged.agents)
        assert _types(ra.events) == _types(rb.events)


def test_snapshot_from_other_scenario_rejected():
    sim = LightweightDrivingSim(SimConfig(render=False))
    sim.reset(ScenarioSpec(0, 0, 0))
    snap = sim.get_state()
    sim.reset(ScenarioSpec(0, 0, 1))
    with pytest.raises(ValueError):
        sim.set_state(snap)


@pytest.mark.parametrize("route_seed", [0, 1, 2])
def test_route_follower_completes_empty_routes_cleanly(route_seed):
    """Acceptance: geometry, controller and events agree on a clean drive (no traffic)."""
    sim = LightweightDrivingSim(SimConfig(render=False))
    spec = ScenarioSpec(0, route_seed, 0, traffic=EMPTY)
    res, events = run_episode(sim, spec)
    types = _types(events)
    assert types[-1] == EventType.ROUTE_COMPLETE
    for bad in (
        EventType.COLLISION,
        EventType.OFF_ROAD,
        EventType.LANE_DEPARTURE,
        EventType.ROUTE_DEVIATION,
    ):
        assert bad not in types
    assert res.privileged.route_progress >= res.privileged.route_length - 3.0


def test_red_light_violation_detected():
    sim = LightweightDrivingSim(SimConfig(render=False))
    found = False
    for route_seed in range(6):
        res = sim.reset(ScenarioSpec(0, route_seed, 0, traffic=EMPTY))
        ctrl = WaypointController()
        while not res.done:
            res = sim.step(route_follow_action(res, ctrl))
            for e in res.events:
                if e.type == EventType.RED_LIGHT:
                    node = int(e.info["node"])
                    route = res.privileged.extra["route"]
                    stop = next(sl for sl in route.stop_lines if sl.node == node)
                    state = sim.episode.road_map.signals[node].state(e.timestamp, stop.approach)
                    assert state == SignalState.RED
                    found = True
        if found:
            break
    assert found, "a signal-ignoring driver should run at least one red light"


def test_off_road_detected_and_terminal():
    sim = LightweightDrivingSim(SimConfig(render=False))
    sim.reset(ScenarioSpec(0, 0, 0, traffic=EMPTY))
    m = sim.episode.road_map
    rows, cols = np.nonzero(m.semantic == Sem.GRASS)
    xy = m.origin + (np.array([cols[len(cols) // 2], rows[len(rows) // 2]]) + 0.5) * 0.1
    sim.state.ego = VehicleState(float(xy[0]), float(xy[1]), 0.0)
    res = sim.step(ControlCommand(0, 0, 1))
    assert EventType.OFF_ROAD in _types(res.events) and res.done
    with pytest.raises(RuntimeError):
        sim.step(ControlCommand(0, 0, 1))


def test_collision_detected():
    # Teleporting off the route would otherwise end the episode with ROUTE_DEVIATION first.
    sim = LightweightDrivingSim(SimConfig(render=False, events=EventConfig(route_deviation=1e9)))
    res = sim.reset(
        ScenarioSpec(
            0, 0, 0, traffic=TrafficConfig(num_vehicles=0, num_parked=4, ped_mean_interval=1e9)
        )
    )
    car = sim.episode.parked[0]
    # Put the ego 6 m behind the parked car, aligned with it, driving forward.
    back = 6.0 + 1.35
    sim.state.ego = VehicleState(
        car.x - back * math.cos(car.yaw), car.y - back * math.sin(car.yaw), car.yaw, v=5.0
    )
    for _ in range(30):
        res = sim.step(ControlCommand(0, 0.3, 0))
        if res.done:
            break
    coll = [e for e in res.events if e.type == EventType.COLLISION]
    assert coll and coll[0].info["agent_kind"] == 2.0 and res.done


def test_lane_departure_then_recovery():
    sim = LightweightDrivingSim(SimConfig(render=False))
    res = sim.reset(ScenarioSpec(0, 0, 0, traffic=EMPTY, initial_speed=5.0))
    events = []
    for _ in range(40):  # drift left until the departure is reported
        res = sim.step(ControlCommand(0.3, 0.2, 0.0))
        events += res.events
        if EventType.LANE_DEPARTURE in _types(res.events):
            break
    assert EventType.LANE_DEPARTURE in _types(events)
    assert abs(res.privileged.route_lateral) > 1.75 and res.privileged.extra["in_lane_departure"]
    ctrl = WaypointController()
    for _ in range(60):  # recover with the route follower
        res = sim.step(route_follow_action(res, ctrl))
        events += res.events
    assert not res.privileged.extra["in_lane_departure"]
    assert _types(events).count(EventType.LANE_DEPARTURE) == 1
    assert EventType.OFF_ROAD not in _types(events)


def test_stall_and_timeout():
    sim = LightweightDrivingSim(SimConfig(render=False, events=EventConfig(stall_timeout=2.0)))
    sim.reset(ScenarioSpec(0, 0, 0, traffic=EMPTY))
    types = []
    for _ in range(25):
        res = sim.step(ControlCommand(0, 0, 1))
        types += _types(res.events)
        if res.done:
            break
    assert types[-1] == EventType.STALL
    sim = LightweightDrivingSim(SimConfig(render=False))
    sim.reset(ScenarioSpec(0, 0, 0, traffic=EMPTY, max_duration=1.0))
    for _ in range(10):
        res = sim.step(ControlCommand(0, 0.3, 0))
    assert res.done and _types(res.events)[-1] == EventType.TIMEOUT


def test_privileged_not_in_observation():
    res = LightweightDrivingSim(SimConfig(render=False)).reset(ScenarioSpec(0, 0, 0))
    assert set(vars(res.observation)) == {"timestamp", "frame_id", "image", "speed", "command"}
