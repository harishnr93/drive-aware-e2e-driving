"""Sampling-rate conventions (docs/system_architecture.md §4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimingConfig:
    dt_sim: float = 0.05  # s, physics step (20 Hz)
    control_period: float = 0.1  # s, policy/control step (10 Hz)
    num_waypoints: int = 10
    waypoint_dt: float = 0.2  # s between predicted waypoints
    history_length: int = 4  # observation frames at the control rate
    failure_horizon_s: float = 2.0  # introspection label horizon

    def __post_init__(self) -> None:
        if self.dt_sim <= 0 or self.control_period <= 0 or self.waypoint_dt <= 0:
            raise ValueError("time steps must be positive")
        if self.num_waypoints < 1 or self.history_length < 1:
            raise ValueError("num_waypoints and history_length must be >= 1")
        _check_multiple(self.control_period, self.dt_sim, "control_period", "dt_sim")
        _check_multiple(self.waypoint_dt, self.control_period, "waypoint_dt", "control_period")
        _check_multiple(
            self.failure_horizon_s, self.control_period, "failure_horizon_s", "control_period"
        )

    @property
    def substeps(self) -> int:
        """Physics steps per control step."""
        return round(self.control_period / self.dt_sim)

    @property
    def waypoint_stride(self) -> int:
        """Control steps between consecutive waypoints."""
        return round(self.waypoint_dt / self.control_period)

    @property
    def horizon_s(self) -> float:
        return self.num_waypoints * self.waypoint_dt

    @property
    def failure_horizon_steps(self) -> int:
        return round(self.failure_horizon_s / self.control_period)

    def waypoint_times(self) -> np.ndarray:
        """Offsets (s) of the predicted waypoints relative to the anchor time."""
        return self.waypoint_dt * np.arange(1, self.num_waypoints + 1)


def _check_multiple(value: float, base: float, name: str, base_name: str) -> None:
    ratio = value / base
    if ratio < 1 - 1e-9 or abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(
            f"{name}={value} must be a positive integer multiple of {base_name}={base}"
        )
