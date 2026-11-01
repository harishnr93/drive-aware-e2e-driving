"""Backend-independent simulator interface (docs/system_architecture.md §5).

The learned policy only ever receives ``StepResult.observation``. ``StepResult.privileged``
is for the expert, label generation and metrics; it must never reach a policy at inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from driveaware.core.frames import Pose2D
from driveaware.core.types import ControlCommand, Event, Observation


@dataclass(frozen=True)
class PrivilegedState:
    """Ground truth at the time of the observation (backend-specific details in ``extra``)."""

    timestamp: float
    ego_pose: Pose2D  # rear axle, world frame
    ego_speed: float  # m/s
    ego_steer: float  # rad, front wheel angle
    ego_footprint: np.ndarray  # [4, 2] world corners
    route_progress: float  # m along the route (rear axle)
    route_length: float  # m
    route_lateral: float  # m, signed offset of the box centre from the route (left +)
    agents: tuple[Any, ...]  # backend agent snapshots (lightweight: agents.Box)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    privileged: PrivilegedState
    events: tuple[Event, ...]
    done: bool


class DrivingSimulator(Protocol):
    def reset(self, scenario: Any) -> StepResult: ...

    def step(self, action: ControlCommand) -> StepResult: ...

    def get_state(self) -> bytes:
        """Serialise the full mutable state (for deterministic counterfactual branching)."""
        ...

    def set_state(self, state: bytes) -> StepResult:
        """Restore a state produced by ``get_state`` within the same scenario."""
        ...
