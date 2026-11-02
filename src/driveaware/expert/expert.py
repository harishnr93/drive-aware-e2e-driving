"""Rule-based privileged driving expert.

Lateral: pure pursuit on the route centreline (shared controller).
Longitudinal: IDM towards the curvature-limited speed limit, with the leader being the most
constraining of (i) actors in the lane corridor ahead, (ii) pedestrians on the carriageway
ahead, (iii) a virtual stopped obstacle at a red stop line, or at a yellow one when stopping is
comfortable. The IDM acceleration is converted into a short-horizon target speed for the PID.

The expert reads ``StepResult.privileged`` only. It is the demonstrator for behaviour cloning
and a reference for driving metrics; it is never available to a learned policy at inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from driveaware.control import WaypointController
from driveaware.core.types import ControlCommand
from driveaware.expert.route_follower import RouteFollower
from driveaware.sim.interface import StepResult
from driveaware.sim.lightweight.agents import IDMParams, find_leader, idm_accel
from driveaware.sim.lightweight.vehicle import VehicleParams


@dataclass(frozen=True)
class ExpertConfig:
    idm: IDMParams = field(
        default_factory=lambda: IDMParams(
            a_max=2.0, b_comf=3.0, b_max=8.0, s0=3.0, time_headway=1.2, lookahead=40.0
        )
    )
    response_time: float = 0.5  # s: target speed = v + a * response_time
    stop_speed: float = 0.3  # m/s: below this with a near obstacle, command a full stop
    vehicle: VehicleParams = field(default_factory=VehicleParams)


@dataclass(frozen=True)
class ExpertDecision:
    """Diagnostics for analysis (why the expert chose its target speed)."""

    target_speed: float
    free_speed: float
    gap: float | None
    accel: float


class RuleBasedExpert:
    def __init__(
        self, config: ExpertConfig | None = None, controller: WaypointController | None = None
    ) -> None:
        self.cfg = config or ExpertConfig()
        self.follower = RouteFollower(
            controller or WaypointController(), comfort_decel=self.cfg.idm.b_comf
        )
        self.last_decision: ExpertDecision | None = None

    def reset(self) -> None:
        self.follower.reset()
        self.last_decision = None

    def decide(self, res: StepResult) -> ExpertDecision:
        p = res.privileged
        route = p.extra["route"]
        road_map = p.extra["map"]
        vp = self.cfg.vehicle
        v = p.ego_speed
        free = self.follower.target_speed(res)
        s_centre = p.route_progress + vp.center_offset
        gap, dv = find_leader(
            route,
            s_centre,
            v,
            vp.length,
            vp.width,
            list(p.agents),
            None,
            road_map,
            p.timestamp,
            self.cfg.idm,
        )
        accel = idm_accel(v, free, gap, dv, self.cfg.idm)
        target = min(max(0.0, v + accel * self.cfg.response_time), free)
        if gap is not None and v < self.cfg.stop_speed and gap < self.cfg.idm.s0:
            target = 0.0  # hold position behind an obstacle / at the stop line
        return ExpertDecision(target, free, gap, accel)

    def __call__(self, res: StepResult) -> ControlCommand:
        decision = self.decide(res)
        self.last_decision = decision
        return self.follower.ctrl(
            self.follower.path_ego(res), decision.target_speed, res.privileged.ego_speed
        )
