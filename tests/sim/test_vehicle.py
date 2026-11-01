import math

import numpy as np
import pytest

from driveaware.core.types import ControlCommand
from driveaware.sim.lightweight.vehicle import (
    VehicleParams,
    VehicleState,
    step_vehicle,
    vehicle_footprint,
)

P = VehicleParams(drag=0.0, rolling=0.0)  # idealised for analytic checks
DT = 0.05


def run(state, cmd, steps, params=P):
    for _ in range(steps):
        state = step_vehicle(state, cmd, params, DT)
    return state


def test_straight_line_constant_speed():
    s = run(VehicleState(0, 0, 0.3, v=10.0), ControlCommand(0, 0, 0), 40)  # 2 s
    assert s.v == pytest.approx(10.0)
    np.testing.assert_allclose([s.x, s.y], [20 * math.cos(0.3), 20 * math.sin(0.3)], atol=1e-9)


def test_turning_radius_matches_bicycle_model():
    # Full left steer at constant speed: rear axle traces a circle of R = L / tan(delta).
    s0 = VehicleState(0, 0, 0, v=5.0, steer=P.max_steer)
    radius = P.wheelbase / math.tan(P.max_steer)
    pts = []
    s = s0
    for _ in range(200):
        s = step_vehicle(s, ControlCommand(1.0, 0, 0), P, DT)
        pts.append((s.x, s.y))
    pts = np.array(pts)
    centre = np.array([0.0, radius])  # left turn: centre on the +y side
    np.testing.assert_allclose(np.linalg.norm(pts - centre, axis=1), radius, rtol=1e-9)
    assert s.yaw == pytest.approx(math.remainder(5.0 * 200 * DT / radius, 2 * math.pi), abs=1e-9)


def test_steer_rate_limit():
    s = step_vehicle(VehicleState(0, 0, 0, v=5.0), ControlCommand(1.0, 0, 0), P, DT)
    assert s.steer == pytest.approx(P.max_steer_rate * DT)


def test_braking_distance_and_no_reverse():
    v0 = 10.0
    s = run(VehicleState(0, 0, 0, v=v0), ControlCommand(0, 0, 1.0), 60)
    assert s.v == 0.0
    assert s.x == pytest.approx(v0**2 / (2 * P.accel_brake), rel=1e-9)
    s2 = run(s, ControlCommand(0, 0, 1.0), 10)
    assert s2.x == s.x  # stays stopped, never reverses


def test_acceleration_and_speed_cap():
    s = run(VehicleState(0, 0, 0), ControlCommand(0, 1.0, 0), 20)  # 1 s
    assert s.v == pytest.approx(P.accel_throttle * 1.0)
    s = run(s, ControlCommand(0, 1.0, 0), 400)
    assert s.v == pytest.approx(P.max_speed)


def test_drag_slows_coasting_vehicle():
    s = run(VehicleState(0, 0, 0, v=10.0), ControlCommand(0, 0, 0), 20, VehicleParams())
    assert s.v < 10.0


def test_footprint_rear_axle_reference():
    fp = vehicle_footprint(VehicleState(0, 0, 0), P)
    assert fp[:, 0].min() == pytest.approx(-P.rear_overhang)
    assert fp[:, 0].max() == pytest.approx(P.length - P.rear_overhang)
    assert fp[:, 1].max() == pytest.approx(P.width / 2)
