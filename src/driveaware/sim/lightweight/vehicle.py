"""Kinematic bicycle model referenced at the rear axle.

State: x, y (world, m), yaw (rad), v (m/s, >= 0, no reverse), steer (front wheel angle, rad).
Inputs: normalised :class:`ControlCommand`. Steering follows the command with a rate limit;
longitudinal acceleration a = a_throttle*throttle - a_brake*brake - c_drag*v^2 - c_roll (v>0).
Integration is exact for constant curvature and constant acceleration within one step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from driveaware.core.frames import Pose2D, wrap_angle
from driveaware.core.types import ControlCommand
from driveaware.sim.lightweight.geometry import box_corners


@dataclass(frozen=True)
class VehicleParams:
    length: float = 4.6  # m
    width: float = 1.9  # m
    wheelbase: float = 2.7  # m
    rear_overhang: float = 0.95  # m, rear bumper to rear axle
    height: float = 1.5  # m (rendering only)
    max_steer: float = 0.6  # rad, front wheel angle at |steer| = 1
    max_steer_rate: float = 1.2  # rad/s
    accel_throttle: float = 3.0  # m/s^2 at full throttle
    accel_brake: float = 8.0  # m/s^2 at full brake
    drag: float = 0.0025  # 1/m, aerodynamic term c in c*v^2
    rolling: float = 0.05  # m/s^2
    max_speed: float = 20.0  # m/s

    @property
    def center_offset(self) -> float:
        """Distance from rear axle forward to the bounding-box centre."""
        return 0.5 * self.length - self.rear_overhang


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    v: float = 0.0
    steer: float = 0.0  # actual front wheel angle, rad

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)


def step_vehicle(
    state: VehicleState, cmd: ControlCommand, params: VehicleParams, dt: float
) -> VehicleState:
    """Advance the bicycle model by ``dt`` seconds and return the new state."""
    target = cmd.steer * params.max_steer
    max_delta = params.max_steer_rate * dt
    steer = state.steer + float(np.clip(target - state.steer, -max_delta, max_delta))

    v = state.v
    a = params.accel_throttle * cmd.throttle - params.accel_brake * cmd.brake
    a -= params.drag * v * v + (params.rolling if v > 0 else 0.0)
    v_new = v + a * dt
    if v_new < 0.0:  # stops within the step: integrate distance to standstill only
        dist = 0.5 * v * (v / -a) if a < 0 else 0.0
        v_new = 0.0
    else:
        v_new = min(v_new, params.max_speed)
        dist = 0.5 * (v + v_new) * dt

    kappa = math.tan(steer) / params.wheelbase
    dyaw = kappa * dist
    if abs(dyaw) < 1e-9:
        x = state.x + dist * math.cos(state.yaw)
        y = state.y + dist * math.sin(state.yaw)
    else:
        x = state.x + (math.sin(state.yaw + dyaw) - math.sin(state.yaw)) / kappa
        y = state.y - (math.cos(state.yaw + dyaw) - math.cos(state.yaw)) / kappa
    return VehicleState(x, y, float(wrap_angle(state.yaw + dyaw)), v_new, steer)


def vehicle_footprint(state: VehicleState, params: VehicleParams) -> np.ndarray:
    """World-frame corners [4, 2] of the vehicle bounding box."""
    off = params.center_offset
    cx = state.x + off * math.cos(state.yaw)
    cy = state.y + off * math.sin(state.yaw)
    return box_corners(cx, cy, state.yaw, params.length, params.width)
