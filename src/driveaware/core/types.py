"""Observation / prediction / control contracts (see docs/system_architecture.md §5).

These types are shared by every simulator backend and by the learning pipeline. The learned
policy receives only :class:`Observation`; privileged simulator state is a separate type
introduced with the simulator (Phase 2) and is never part of these contracts.

Conventions: SI units, radians, ego frame = rear axle, x forward / y left / z up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 160
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)


class Command(IntEnum):
    """High-level navigation command (route-derived, non-privileged)."""

    LANE_FOLLOW = 0
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3


class EventType(IntEnum):
    """Ground-truth events detected by the simulator (privileged; for labels and metrics)."""

    COLLISION = 0
    OFF_ROAD = 1
    RED_LIGHT = 2
    ROUTE_DEVIATION = 3
    STALL = 4
    ROUTE_COMPLETE = 5
    LANE_DEPARTURE = 6  # infraction; not in FAILURE_EVENTS by default
    TIMEOUT = 7  # episode time limit reached


FAILURE_EVENTS = frozenset(
    {
        EventType.COLLISION,
        EventType.OFF_ROAD,
        EventType.RED_LIGHT,
        EventType.ROUTE_DEVIATION,
        EventType.STALL,
    }
)


@dataclass(frozen=True)
class Event:
    type: EventType
    timestamp: float  # s, sim time
    info: dict[str, float] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return self.type in FAILURE_EVENTS


@dataclass(frozen=True)
class Observation:
    """The only input the learned policy receives."""

    timestamp: float  # s, sim time at which the image was rendered
    frame_id: int
    image: np.ndarray  # uint8 [H, W, 3] RGB
    speed: float  # m/s, ego longitudinal speed
    command: Command

    def __post_init__(self) -> None:
        if self.image.dtype != np.uint8 or self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(
                f"image must be uint8 [H, W, 3], got {self.image.dtype} {self.image.shape}"
            )
        if not np.isfinite(self.speed):
            raise ValueError("speed must be finite")


@dataclass(frozen=True)
class TrajectoryPrediction:
    """Future ego waypoints in the ego frame at ``timestamp``."""

    timestamp: float
    waypoints: np.ndarray  # float32 [N, 2], metres
    target_speed: float  # m/s
    waypoint_std: np.ndarray | None = None  # float32 [N, 2], predictive std

    def __post_init__(self) -> None:
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 2:
            raise ValueError(f"waypoints must be [N, 2], got {self.waypoints.shape}")
        if self.waypoint_std is not None and self.waypoint_std.shape != self.waypoints.shape:
            raise ValueError("waypoint_std must match waypoints shape")
        if self.target_speed < 0:
            raise ValueError("target_speed must be non-negative")


@dataclass(frozen=True)
class IntrospectionOutput:
    failure_prob: float  # P(failure in (t, t + horizon_s]), calibrated
    horizon_s: float
    features: dict[str, float] = field(default_factory=dict)  # analysis only

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_prob <= 1.0:
            raise ValueError(f"failure_prob must be in [0, 1], got {self.failure_prob}")


@dataclass(frozen=True)
class ControlCommand:
    """Normalised vehicle command with the same semantics as ``carla.VehicleControl``.

    ``steer`` is positive to the LEFT (ego frame convention); the CARLA adapter negates it.
    """

    steer: float  # [-1, 1]
    throttle: float  # [0, 1]
    brake: float  # [0, 1]

    def __post_init__(self) -> None:
        if not -1.0 <= self.steer <= 1.0:
            raise ValueError(f"steer out of range: {self.steer}")
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError(f"throttle out of range: {self.throttle}")
        if not 0.0 <= self.brake <= 1.0:
            raise ValueError(f"brake out of range: {self.brake}")

    @classmethod
    def clipped(cls, steer: float, throttle: float, brake: float) -> ControlCommand:
        return cls(
            steer=float(np.clip(steer, -1.0, 1.0)),
            throttle=float(np.clip(throttle, 0.0, 1.0)),
            brake=float(np.clip(brake, 0.0, 1.0)),
        )
