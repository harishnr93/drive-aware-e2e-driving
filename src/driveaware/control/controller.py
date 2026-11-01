"""Waypoint tracking controller shared by the expert and the learned policies.

Input: a path in the ego frame (rear axle origin, x forward, y left) and a target speed.
Lateral: pure pursuit, lookahead L_d = clip(k_v * v + L_0, L_min, L_max) metres along the path.
Longitudinal: PID on speed error, mapped to throttle (u > 0) or brake (u < -deadband).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from driveaware.core.types import ControlCommand


@dataclass(frozen=True)
class ControllerConfig:
    wheelbase: float = 2.7  # m, must match the vehicle
    max_steer: float = 0.6  # rad at |steer| = 1
    lookahead_gain: float = 0.6  # s
    lookahead_min: float = 3.0  # m
    lookahead_max: float = 12.0  # m
    lookahead_base: float = 2.0  # m
    kp: float = 0.45
    ki: float = 0.05
    kd: float = 0.0
    integral_limit: float = 4.0
    brake_gain: float = 0.6  # brake = -u * brake_gain
    brake_deadband: float = 0.05
    stop_speed: float = 0.3  # m/s: below this with target ~0, hold the brake
    hold_brake: float = 0.6


def lookahead_point(path_xy: np.ndarray, distance: float) -> np.ndarray:
    """Point at arc length ``distance`` along a polyline starting near the origin.

    The path is prefixed with the origin so that arc length is measured from the rear axle.
    Beyond the end, the last segment is extrapolated.
    """
    pts = np.vstack([[0.0, 0.0], np.asarray(path_xy, dtype=np.float64)])
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-6
    seg, seg_len = seg[keep], seg_len[keep]
    if len(seg) == 0:
        return np.array([distance, 0.0])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    base = np.vstack([[0.0, 0.0], np.cumsum(seg, axis=0)])  # vertex positions after filtering
    i = int(np.clip(np.searchsorted(cum, distance, side="right") - 1, 0, len(seg) - 1))
    return base[i] + seg[i] / seg_len[i] * (distance - cum[i])


class WaypointController:
    def __init__(self, config: ControllerConfig = ControllerConfig(), dt: float = 0.1) -> None:
        self.cfg = config
        self.dt = dt
        self._integral = 0.0
        self._prev_error: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def steering(self, path_xy: np.ndarray, speed: float) -> float:
        c = self.cfg
        ld = float(
            np.clip(c.lookahead_gain * speed + c.lookahead_base, c.lookahead_min, c.lookahead_max)
        )
        target = lookahead_point(path_xy, ld)
        dist_sq = float(target @ target)
        if dist_sq < 1e-6:
            return 0.0
        curvature = 2.0 * target[1] / dist_sq
        delta = math.atan(c.wheelbase * curvature)
        return float(np.clip(delta / c.max_steer, -1.0, 1.0))

    def longitudinal(self, target_speed: float, speed: float) -> tuple[float, float]:
        c = self.cfg
        if target_speed < 0.05 and speed < c.stop_speed:
            self._integral = 0.0
            self._prev_error = None
            return 0.0, c.hold_brake
        error = target_speed - speed
        self._integral = float(
            np.clip(self._integral + error * self.dt, -c.integral_limit, c.integral_limit)
        )
        deriv = 0.0 if self._prev_error is None else (error - self._prev_error) / self.dt
        self._prev_error = error
        u = c.kp * error + c.ki * self._integral + c.kd * deriv
        if u >= 0.0:
            return float(min(u, 1.0)), 0.0
        if u > -c.brake_deadband:
            return 0.0, 0.0
        return 0.0, float(min(-u * c.brake_gain, 1.0))

    def __call__(self, path_xy: np.ndarray, target_speed: float, speed: float) -> ControlCommand:
        throttle, brake = self.longitudinal(target_speed, speed)
        return ControlCommand.clipped(self.steering(path_xy, speed), throttle, brake)
