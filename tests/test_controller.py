import math

import numpy as np
import pytest

from driveaware.control import ControllerConfig, WaypointController
from driveaware.control.controller import lookahead_point
from driveaware.core.frames import Pose2D, world_to_ego
from driveaware.core.types import ControlCommand
from driveaware.sim.lightweight.vehicle import VehicleParams, VehicleState, step_vehicle


def test_lookahead_point_measures_from_origin():
    path = np.array([[1.0, 0.0], [5.0, 0.0], [5.0, 4.0]])
    np.testing.assert_allclose(lookahead_point(path, 3.0), [3.0, 0.0])
    np.testing.assert_allclose(lookahead_point(path, 7.0), [5.0, 2.0])
    np.testing.assert_allclose(lookahead_point(path, 11.0), [5.0, 6.0])  # extrapolated


def test_steering_sign():
    ctrl = WaypointController()
    straight = np.column_stack([np.arange(1, 20.0), np.zeros(19)])
    assert ctrl.steering(straight, 5.0) == pytest.approx(0.0)
    left = np.column_stack([np.arange(1, 20.0), 0.05 * np.arange(1, 20.0) ** 2])
    assert ctrl.steering(left, 5.0) > 0
    assert ctrl.steering(left * [1, -1], 5.0) < 0


def test_longitudinal_modes():
    ctrl = WaypointController()
    thr, brk = ctrl.longitudinal(10.0, 5.0)
    assert thr > 0 and brk == 0
    ctrl.reset()
    thr, brk = ctrl.longitudinal(2.0, 10.0)
    assert thr == 0 and brk > 0
    ctrl.reset()
    assert ctrl.longitudinal(0.0, 0.1) == (0.0, ControllerConfig().hold_brake)


def test_closed_loop_tracks_circle_and_speed():
    params = VehicleParams()
    ctrl = WaypointController(dt=0.1)
    radius, target_v = 25.0, 6.0
    state = VehicleState(0.0, -radius, 0.0, v=0.0)  # on a CCW circle about the origin
    errors, speeds = [], []
    for k in range(400):
        theta = math.atan2(state.y, state.x)
        ahead = theta + np.linspace(0.05, 1.0, 30)
        path = world_to_ego(
            np.column_stack([radius * np.cos(ahead), radius * np.sin(ahead)]),
            Pose2D(state.x, state.y, state.yaw),
        )
        cmd: ControlCommand = ctrl(path, target_v, state.v)
        for _ in range(2):
            state = step_vehicle(state, cmd, params, 0.05)
        if k > 150:
            errors.append(abs(math.hypot(state.x, state.y) - radius))
            speeds.append(state.v)
    assert max(errors) < 0.5  # pure pursuit cuts slightly inside on curves
    assert abs(np.mean(speeds) - target_v) < 0.2
