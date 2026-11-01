import numpy as np
import pytest

from driveaware.core.types import Command
from driveaware.sim.lightweight.roadmap import generate_map
from driveaware.sim.lightweight.route import build_route, random_lane_walk


def _route(seed=0, length=600.0):
    m = generate_map(0)
    rng = np.random.default_rng(seed)
    lanes = random_lane_walk(m, rng, m.road_lanes[0].id, length)
    return m, build_route(m, lanes)


def test_route_concatenation_and_lane_offsets():
    m, r = _route()
    assert r.length >= 600.0 - 1e-6
    lengths = [m.lanes[i].path.length for i in r.lane_ids]
    assert r.length == pytest.approx(sum(lengths), rel=1e-6)
    np.testing.assert_allclose(r.lane_s0[1:], np.cumsum(lengths)[:-1], rtol=1e-6)


def test_commands_cover_junction_connectors():
    m, r = _route(seed=3, length=800.0)
    for lane_id, s0 in zip(r.lane_ids, r.lane_s0, strict=True):
        lane = m.lanes[lane_id]
        if lane.kind == "connector":
            assert r.command_at(s0 + 0.5 * lane.path.length) == lane.turn
        elif lane.path.length > 21.0:
            # Just past the previous junction and before the next announcement.
            assert r.command_at(s0 + 0.5) == Command.LANE_FOLLOW
        if lane.turn != Command.LANE_FOLLOW:
            assert r.command_at(s0 - 15.0) == lane.turn  # announced in advance
            assert r.command_at(s0 - 25.0) == Command.LANE_FOLLOW or s0 < 25.0 + 1e-9


def test_stop_lines_at_signalised_lane_ends():
    m, r = _route(seed=1, length=800.0)
    expected = [
        s0 + m.lanes[i].path.length
        for i, s0 in zip(r.lane_ids, r.lane_s0, strict=True)
        if m.lanes[i].signal
    ]
    np.testing.assert_allclose([sl.s for sl in r.stop_lines], expected)


def test_invalid_lane_sequence_rejected():
    m = generate_map(0)
    with pytest.raises(ValueError):
        build_route(m, [m.road_lanes[0].id, m.road_lanes[1].id])


def test_speed_limits_lower_in_turns():
    m, r = _route(seed=2, length=800.0)
    for lane_id, s0 in zip(r.lane_ids, r.lane_s0, strict=True):
        lane = m.lanes[lane_id]
        assert r.speed_limit_at(s0 + 0.5 * lane.path.length) == pytest.approx(lane.speed_limit)
        if lane.turn in (Command.LEFT, Command.RIGHT):
            assert lane.speed_limit < m.config.speed_limit
