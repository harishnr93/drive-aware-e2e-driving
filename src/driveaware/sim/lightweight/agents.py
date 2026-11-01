"""Background agents: IDM vehicles on lane paths, crosswalk pedestrians, parked cars.

NPC vehicles follow fixed random lane paths with the Intelligent Driver Model. Their leader is
the closest actor (ego, vehicle or pedestrian) inside a corridor ahead along their path, or a
virtual stopped obstacle at a red/yellow stop line. Pedestrians cross at mid-block crosswalks
and do not yield; vehicles yield to them through the same corridor check. NPC-NPC contacts are
not scored (only ego events are), but the corridor logic keeps them rare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from driveaware.sim.lightweight.geometry import box_corners, boxes_overlap
from driveaware.sim.lightweight.roadmap import Crosswalk, RoadMap, SignalState
from driveaware.sim.lightweight.route import Route

NPC_LENGTH, NPC_WIDTH, NPC_HEIGHT = 4.6, 1.9, 1.5
PED_SIZE, PED_HEIGHT = 0.6, 1.75


class AgentKind:
    VEHICLE = 0
    PEDESTRIAN = 1
    PARKED = 2


@dataclass(frozen=True)
class IDMParams:
    a_max: float = 1.5
    b_comf: float = 2.5
    b_max: float = 8.0
    s0: float = 2.5  # m standstill gap
    time_headway: float = 1.4  # s
    delta: float = 4.0
    lookahead: float = 40.0  # m
    yellow_decel: float = 3.0  # stop on yellow if achievable with this decel


@dataclass
class NPCState:
    path_index: int
    s: float  # arc length of the vehicle centre along its path
    v: float
    speed_factor: float
    active: bool = True


@dataclass
class PedState:
    crosswalk: int
    direction: float  # +1 / -1 across the road
    progress: float  # m travelled across
    speed: float
    along: float  # m offset along the road within the crosswalk


@dataclass(frozen=True)
class ParkedCar:
    x: float
    y: float
    yaw: float
    length: float = NPC_LENGTH
    width: float = NPC_WIDTH


@dataclass(frozen=True)
class Box:
    """Snapshot of an actor for collision checks, leader search and rendering."""

    kind: int
    x: float
    y: float
    yaw: float
    v: float
    length: float
    width: float
    height: float
    ident: int

    def corners(self) -> np.ndarray:
        return box_corners(self.x, self.y, self.yaw, self.length, self.width)


def npc_box(npc: NPCState, path: Route, ident: int) -> Box:
    x, y, yaw = path.path.interpolate(npc.s)
    return Box(AgentKind.VEHICLE, x, y, yaw, npc.v, NPC_LENGTH, NPC_WIDTH, NPC_HEIGHT, ident)


def ped_pose(ped: PedState, cw: Crosswalk) -> tuple[float, float, float]:
    along = np.array([math.cos(cw.road_yaw), math.sin(cw.road_yaw)])
    across = np.array([-along[1], along[0]])
    lateral = ped.direction * (ped.progress - cw.half_span)
    x, y = cw.center + ped.along * along + lateral * across
    yaw = cw.road_yaw + (math.pi / 2 if ped.direction > 0 else -math.pi / 2)
    return float(x), float(y), yaw


def ped_box(ped: PedState, cw: Crosswalk, ident: int) -> Box:
    x, y, yaw = ped_pose(ped, cw)
    return Box(AgentKind.PEDESTRIAN, x, y, yaw, ped.speed, PED_SIZE, PED_SIZE, PED_HEIGHT, ident)


def parked_box(car: ParkedCar, ident: int) -> Box:
    return Box(
        AgentKind.PARKED, car.x, car.y, car.yaw, 0.0, car.length, car.width, NPC_HEIGHT, ident
    )


def idm_accel(v: float, v0: float, gap: float | None, dv: float, p: IDMParams) -> float:
    """IDM acceleration. ``gap`` is bumper-to-bumper distance (None = free road)."""
    v0 = max(v0, 0.1)
    acc = 1.0 - (v / v0) ** p.delta
    if gap is not None:
        s_star = p.s0 + max(0.0, v * p.time_headway + v * dv / (2 * math.sqrt(p.a_max * p.b_comf)))
        acc -= (s_star / max(gap, 0.1)) ** 2
    return float(np.clip(p.a_max * acc, -p.b_max, p.a_max))


def npc_leader(
    npc: NPCState,
    path: Route,
    others: list[Box],
    self_ident: int,
    road_map: RoadMap,
    t: float,
    p: IDMParams,
) -> tuple[float | None, float]:
    """Return (gap, closing speed) to the most constraining obstacle ahead, or (None, 0)."""
    ds = 0.5
    s_ahead = npc.s + np.arange(0.0, p.lookahead, ds)
    pts = path.path.interpolate(s_ahead)
    best_gap: float | None = None
    best_dv = 0.0
    # Lane centre to the far kerb, plus a step onto the pavement.
    ped_yield_reach = 0.5 * road_map.config.lane_width + road_map.config.road_half_width + 0.5
    if others:
        centres = np.array([[b.x, b.y] for b in others])
        d = np.linalg.norm(centres[:, None, :] - pts[None, :, :2], axis=2)  # [A, P]
        k = d.argmin(axis=1)
        lateral = d[np.arange(len(others)), k]
        for i, box in enumerate(others):
            if box.ident == self_ident:
                continue
            is_ped = box.kind == AgentKind.PEDESTRIAN
            # Vehicles: lane corridor. Pedestrians: anyone on the carriageway ahead (crosswalk
            # yielding; they cross from the kerb and do not yield themselves).
            reach = ped_yield_reach if is_ped else 0.5 * NPC_WIDTH + 0.5 * box.width + 0.2
            if lateral[i] > reach:
                continue
            if k[i] == 0:
                # Projects onto the NPC's own centre: only a leader if it is in front of us.
                yaw = pts[0, 2]
                rel = np.array([box.x - pts[0, 0], box.y - pts[0, 1]])
                ahead = rel @ np.array([math.cos(yaw), math.sin(yaw)])
                if ahead <= (0.5 * NPC_LENGTH - 0.5 if is_ped else 0.0):
                    continue
            gap = k[i] * ds - 0.5 * (NPC_LENGTH + box.length)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_dv = npc.v - box.v * math.cos(box.yaw - pts[k[i], 2])
    for stop in path.stop_lines:
        if not (npc.s - 0.5 * NPC_LENGTH < stop.s < npc.s + p.lookahead):
            continue
        state = road_map.signals[stop.node].state(t, stop.approach)
        gap = stop.s - npc.s - 0.5 * NPC_LENGTH
        must_stop = state == SignalState.RED or (
            state == SignalState.YELLOW and gap > npc.v**2 / (2 * p.yellow_decel)
        )
        if must_stop and gap > -0.5 and (best_gap is None or gap < best_gap):
            best_gap, best_dv = max(gap, 0.0), npc.v
        break  # only the next stop line matters
    return best_gap, best_dv


def npc_desired_speed(npc: NPCState, path: Route, p: IDMParams) -> float:
    """Speed limit ahead, discounted by the distance available to brake for it."""
    s_ahead = npc.s + np.arange(0.0, p.lookahead, 1.0)
    limits = path.speed_limit_at(s_ahead) * npc.speed_factor
    reachable = np.sqrt(limits**2 + 2.0 * p.b_comf * (s_ahead - npc.s))
    return float(reachable.min())


def step_npc(npc: NPCState, accel: float, path: Route, dt: float) -> None:
    v_new = max(0.0, npc.v + accel * dt)
    npc.s += 0.5 * (npc.v + v_new) * dt
    npc.v = v_new
    if npc.s >= path.length - 1.0:
        npc.active = False


def step_ped(ped: PedState, cw: Crosswalk, dt: float, vehicles: list[Box] = ()) -> bool:
    """Advance a pedestrian; returns False once it has finished crossing.

    Pedestrians wait at the kerb while a vehicle occupies the crosswalk, and never step into a
    vehicle. They do not otherwise yield (moving vehicles must yield to them).
    """
    kerb = 1.0  # m of pavement before the carriageway (half_span = road half width + 1)
    nxt = PedState(
        ped.crosswalk, ped.direction, ped.progress + ped.speed * dt, ped.speed, ped.along
    )
    if ped.progress <= kerb < nxt.progress and _crosswalk_occupied(cw, vehicles):
        return True
    x, y, yaw = ped_pose(nxt, cw)
    mine = box_corners(x, y, yaw, PED_SIZE + 0.3, PED_SIZE + 0.3)
    for v in vehicles:
        if math.hypot(v.x - x, v.y - y) < 4.0 and boxes_overlap(mine, v.corners()):
            return True
    ped.progress = nxt.progress
    return ped.progress < 2.0 * cw.half_span


def _crosswalk_occupied(cw: Crosswalk, vehicles: list[Box]) -> bool:
    along = np.array([math.cos(cw.road_yaw), math.sin(cw.road_yaw)])
    across = np.array([-along[1], along[0]])
    for v in vehicles:
        rel = np.array([v.x, v.y]) - cw.center
        if (
            abs(rel @ along) < 0.5 * (cw.depth + v.length) + 0.5
            and abs(rel @ across) < cw.half_span
        ):
            return True
    return False
