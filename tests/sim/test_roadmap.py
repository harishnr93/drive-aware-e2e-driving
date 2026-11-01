import hashlib
import itertools

import numpy as np
import pytest

from driveaware.core.types import Command
from driveaware.sim.lightweight.roadmap import MapConfig, Sem, SignalState, generate_map

SEEDS = [0, 1, 2]


def _hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def test_map_deterministic_and_seed_dependent():
    a = generate_map.__wrapped__(0, MapConfig())
    b = generate_map.__wrapped__(0, MapConfig())
    c = generate_map.__wrapped__(1, MapConfig())
    assert _hash(a.color) == _hash(b.color) and _hash(a.semantic) == _hash(b.semantic)
    assert _hash(a.semantic) != _hash(c.semantic)


def test_map_is_read_only():
    m = generate_map(0)
    with pytest.raises(ValueError):
        m.semantic[0, 0] = 1


@pytest.mark.parametrize("seed", SEEDS)
def test_topology(seed):
    m = generate_map(seed)
    n = len(m.nodes)
    deg = np.bincount(np.array(m.edges).ravel(), minlength=n)
    assert deg.min() >= 2  # no dead ends
    for lane in m.lanes:
        assert lane.successors, f"lane {lane.id} has no successor"
        for succ in lane.successors:
            nxt = m.lanes[succ]
            gap = np.linalg.norm(lane.path.points[-1] - nxt.path.points[0])
            assert gap < 1e-6  # lanes join exactly
    for lane in m.road_lanes:
        succ_edges = {m.lanes[m.lanes[c].successors[0]].edge for c in lane.successors}
        assert lane.edge not in succ_edges  # no U-turns


@pytest.mark.parametrize("seed", SEEDS)
def test_lanes_drivable_with_vehicle_width(seed):
    m = generate_map(seed)
    for lane in m.lanes:
        pts = lane.path.interpolate(np.arange(0, lane.path.length, 0.5))
        normal = np.stack([-np.sin(pts[:, 2]), np.cos(pts[:, 2])], 1)
        for off in (-1.0, 0.0, 1.0):
            assert m.is_drivable(pts[:, :2] + off * normal).all()


@pytest.mark.parametrize("seed", SEEDS)
def test_connector_turn_radius_feasible(seed):
    m = generate_map(seed)
    min_vehicle_radius = 2.7 / np.tan(0.6)  # 3.95 m
    for lane in m.lanes:
        if lane.kind == "connector":
            assert 1.0 / lane.path.curvature().max() > min_vehicle_radius + 0.5


def test_turn_commands_only_at_junctions():
    m = generate_map(0)
    for lane in m.lanes:
        if lane.kind != "connector":
            continue
        if m.node_degree(lane.node) <= 2:
            assert lane.turn == Command.LANE_FOLLOW
        else:
            assert lane.turn in (Command.LEFT, Command.RIGHT, Command.STRAIGHT)


def test_signals_never_two_greens():
    m = generate_map(0)
    assert m.signals
    for sig in m.signals.values():
        for t in np.arange(0.0, 2 * sig.cycle, 0.25):
            states = [sig.state(t, k) for k in range(len(sig.approach_lanes))]
            active = sum(s != SignalState.RED for s in states)
            assert active <= 1
        # Every approach gets green within one cycle.
        for k in range(len(sig.approach_lanes)):
            assert any(sig.state(t, k) == SignalState.GREEN for t in np.arange(0, sig.cycle, 0.25))


def test_stop_lines_on_signalised_approaches():
    m = generate_map(0)
    for sig in m.signals.values():
        for lane_id in sig.approach_lanes:
            lane = m.lanes[lane_id]
            assert lane.signal is not None and lane.signal[0] == sig.node
            end = lane.path.interpolate(lane.path.length - 0.2)[:2]
            assert m.semantic_at(end[None])[0] == Sem.STOP_LINE


def test_outside_raster_is_not_drivable():
    m = generate_map(0)
    far = m.origin - 1000.0
    assert not m.is_drivable(far[None])[0]
    assert m.semantic_at(far[None])[0] == Sem.GRASS


def test_pixel_lookup_convention():
    m = generate_map(0)
    res = m.config.resolution
    xy = m.origin + np.array([[10.05, 20.05]])
    row, col = m.world_to_pixel(xy)
    assert (row[0], col[0]) == (int(20.05 / res), int(10.05 / res))
    assert list(itertools.islice(m.signals, 1))  # map has signals for later tests
