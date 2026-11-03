"""Benchmark v2 (docs/benchmark_v2.md): v1 must stay bit-identical; v2 must stay drivable."""

import hashlib

import numpy as np
import pytest

from driveaware.core.types import ControlCommand
from driveaware.sim.lightweight.env import LightweightDrivingSim
from driveaware.sim.lightweight.roadmap import Sem, generate_map
from driveaware.sim.lightweight.scenario import BENCHMARKS, ScenarioSpec

V2 = BENCHMARKS["v2"]


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def test_v1_maps_and_render_unchanged():
    """Fingerprints recorded before the v2 code existed (2026-11-02)."""
    assert BENCHMARKS["v1"] == generate_map(0).config
    m = generate_map(0)
    assert (_h(m.semantic), _h(m.color)) == ("110caf257ae52c60", "d4ef18d9fd5142ff")
    m = generate_map(3000)
    assert (_h(m.semantic), _h(m.color)) == ("566cf1074764f28d", "2d3f2017146cce2d")
    sim = LightweightDrivingSim()
    res = sim.reset(ScenarioSpec(0, 0, 0))
    for _ in range(30):
        res = sim.step(ControlCommand(0, 0.4, 0))
    assert _h(res.observation.image) == "aabba6e4aa22a300"


@pytest.mark.parametrize("seed", [0, 1, 205])
def test_v2_lanes_drivable_with_vehicle_width(seed):
    m = generate_map(seed, V2)
    for lane in m.lanes:
        pts = lane.path.interpolate(np.arange(0, lane.path.length, 0.5))
        normal = np.stack([-np.sin(pts[:, 2]), np.cos(pts[:, 2])], 1)
        for off in (-1.0, 0.0, 1.0):
            assert m.is_drivable(pts[:, :2] + off * normal).all(), lane.id


def test_v2_lane_graph_identical_to_v1():
    """v2 changes rendering/kerbs only: lanes, signals and crosswalks are the same."""
    a, b = generate_map(1), generate_map(1, V2)
    assert len(a.lanes) == len(b.lanes)
    for la, lb in zip(a.lanes, b.lanes, strict=True):
        np.testing.assert_array_equal(la.path.points, lb.path.points)
        assert la.successors == lb.successors and la.turn == lb.turn
    assert [s.offset for s in a.signals.values()] == [s.offset for s in b.signals.values()]
    assert [c.center.tolist() for c in a.crosswalks] == [c.center.tolist() for c in b.crosswalks]


def test_v2_junctions_smaller_than_v1_discs():
    a, b = generate_map(2), generate_map(2, V2)
    assert (b.semantic == Sem.ROAD).sum() < (a.semantic == Sem.ROAD).sum()
    assert (b.semantic == Sem.EDGE_LINE).sum() > (a.semantic == Sem.EDGE_LINE).sum()  # guides


def test_v2_sidewalk_borders_road():
    m = generate_map(0, V2)
    road = m.semantic != Sem.GRASS
    road &= m.semantic != Sem.SIDEWALK
    # no grass pixel directly adjacent to the road surface (kerb is always lined)
    grass = m.semantic == Sem.GRASS
    adj = grass[1:, :] & road[:-1, :]
    assert adj.sum() == 0


def test_v2_near_side_heads_and_size():
    m = generate_map(0, V2)
    for sig in m.signals.values():
        assert len(sig.near_heads) == len(sig.approach_lanes)
        for h in sig.near_heads:
            assert 3.0 < h.position[2] < 4.5
    assert all(not s.near_heads for s in generate_map(0).signals.values())
    sim = LightweightDrivingSim()
    sim.reset(ScenarioSpec(0, 0, 0, map=V2))
    heads = sim.signal_heads()
    assert heads and all(h.size == (1.4, 0.8, 2.6) for h in heads)
