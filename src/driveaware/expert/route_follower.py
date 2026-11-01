"""Traffic-unaware route follower (privileged).

Tracks the route centreline at the curvature-limited speed limit and ignores other actors and
traffic signals. It is a simulator test driver and the lateral/longitudinal core of the Phase 3
expert, which adds car-following, signal compliance and pedestrian yielding.
"""

from __future__ import annotations

import numpy as np

from driveaware.control import WaypointController
from driveaware.core.frames import world_to_ego
from driveaware.core.types import ControlCommand
from driveaware.sim.interface import StepResult


class RouteFollower:
    def __init__(
        self,
        controller: WaypointController | None = None,
        speed_scale: float = 1.0,
        comfort_decel: float = 2.5,
    ) -> None:
        self.ctrl = controller or WaypointController()
        self.speed_scale = speed_scale
        self.comfort_decel = comfort_decel

    def reset(self) -> None:
        self.ctrl.reset()

    def target_speed(self, res: StepResult) -> float:
        p = res.privileged
        route = p.extra["route"]
        dist = np.arange(0.0, 30.0, 1.0)
        limits = route.speed_limit_at(p.route_progress + dist) * self.speed_scale
        return float(np.min(np.sqrt(limits**2 + 2.0 * self.comfort_decel * dist)))

    def path_ego(self, res: StepResult) -> np.ndarray:
        p = res.privileged
        route = p.extra["route"]
        ahead = route.path.interpolate(p.route_progress + np.arange(1.0, 20.0, 1.0))[:, :2]
        return world_to_ego(ahead, p.ego_pose)

    def __call__(self, res: StepResult) -> ControlCommand:
        return self.ctrl(self.path_ego(res), self.target_speed(res), res.privileged.ego_speed)
