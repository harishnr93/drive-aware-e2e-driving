"""Lightweight closed-loop driving simulator implementing ``DrivingSimulator``.

Episode-static content (map, ego route, NPC paths, parked cars) is rebuilt deterministically
from the :class:`ScenarioSpec`. All mutable state lives in :class:`SimState`, which is what
``get_state``/``set_state`` serialise; restoring a state and replaying the same actions gives
bit-identical results.

Events (privileged, for labels and metrics):
- COLLISION: ego box overlaps any actor box (checked every physics step). Terminal.
- OFF_ROAD: a footprint point lies outside the drivable raster. Terminal.
- LANE_DEPARTURE: ego box centre more than half a lane width from the route (rising edge,
  with hysteresis). Non-terminal infraction.
- RED_LIGHT: front bumper crosses a route stop line while the signal is red. Non-terminal.
- ROUTE_DEVIATION: rear axle further than ``route_deviation`` m from the route. Terminal.
- STALL: speed below 0.1 m/s for ``stall_timeout`` s. Terminal.
- ROUTE_COMPLETE / TIMEOUT: terminal.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from driveaware.core.frames import Pose2D
from driveaware.core.seeding import derive_seed, make_rng
from driveaware.core.timing import TimingConfig
from driveaware.core.types import ControlCommand, Event, EventType, Observation
from driveaware.sim.interface import PrivilegedState, StepResult
from driveaware.sim.lightweight import agents as ag
from driveaware.sim.lightweight.geometry import boxes_overlap
from driveaware.sim.lightweight.renderer import CameraConfig, CameraRenderer, SignalHeadView
from driveaware.sim.lightweight.roadmap import RoadMap, SignalState, generate_map
from driveaware.sim.lightweight.route import Route, build_route, random_lane_walk
from driveaware.sim.lightweight.scenario import ScenarioSpec
from driveaware.sim.lightweight.vehicle import (
    VehicleParams,
    VehicleState,
    step_vehicle,
    vehicle_footprint,
)

TERMINAL_EVENTS = frozenset(
    {
        EventType.COLLISION,
        EventType.OFF_ROAD,
        EventType.ROUTE_DEVIATION,
        EventType.STALL,
        EventType.ROUTE_COMPLETE,
        EventType.TIMEOUT,
    }
)


@dataclass(frozen=True)
class EventConfig:
    route_deviation: float = 5.0  # m
    stall_timeout: float = 40.0  # s (longer than the worst red-light wait)
    goal_tolerance: float = 3.0  # m before the route end
    lane_departure_hysteresis: float = 0.2  # m


@dataclass(frozen=True)
class SimConfig:
    timing: TimingConfig = field(default_factory=TimingConfig)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    camera: CameraConfig = field(default_factory=CameraConfig)
    events: EventConfig = field(default_factory=EventConfig)
    idm: ag.IDMParams = field(default_factory=ag.IDMParams)
    render: bool = True


@dataclass
class SimState:
    scenario_key: tuple[int, int, int]
    t: float
    step: int
    ego: VehicleState
    route_s: float
    npcs: list[ag.NPCState]
    peds: list[ag.PedState]
    next_ped_spawn: list[float]
    rng_state: dict[str, Any]
    stall_time: float = 0.0
    in_lane_departure: bool = False
    next_stop_index: int = 0
    done: bool = False
    next_ped_ident: int = 0


@dataclass
class _Episode:
    """Episode-static content derived from the scenario (not part of the snapshot)."""

    spec: ScenarioSpec
    road_map: RoadMap
    route: Route
    npc_paths: list[Route]
    parked: list[ag.ParkedCar]
    renderer: CameraRenderer | None


class LightweightDrivingSim:
    def __init__(self, config: SimConfig = SimConfig()) -> None:
        self.cfg = config
        self._ep: _Episode | None = None
        self._state: SimState | None = None
        self._rng: np.random.Generator | None = None

    # ---------------------------------------------------------------- public API

    @property
    def episode(self) -> _Episode:
        if self._ep is None:
            raise RuntimeError("call reset() first")
        return self._ep

    @property
    def state(self) -> SimState:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    def reset(self, scenario: ScenarioSpec) -> StepResult:
        road_map = generate_map(scenario.map_seed, scenario.map)
        route_rng = make_rng(derive_seed(scenario.map_seed, "route", scenario.route_seed))
        road_lanes = road_map.road_lanes
        start = road_lanes[int(route_rng.integers(len(road_lanes)))].id
        route = build_route(
            road_map, random_lane_walk(road_map, route_rng, start, scenario.route_length)
        )
        traffic_rng = make_rng(
            derive_seed(scenario.map_seed, "traffic", scenario.route_seed, scenario.traffic_seed)
        )
        ego_s = 2.0
        x, y, yaw = route.path.interpolate(ego_s)
        ego = VehicleState(float(x), float(y), float(yaw), scenario.initial_speed)
        parked = self._place_parked(road_map, scenario, traffic_rng, np.array([x, y]))
        npc_paths, npcs = self._spawn_npcs(
            road_map, scenario, traffic_rng, np.array([x, y]), parked
        )
        next_spawn = [
            float(traffic_rng.exponential(scenario.traffic.ped_mean_interval))
            for _ in road_map.crosswalks
        ]
        renderer = None
        if self.cfg.render:
            if (
                self._ep is not None
                and self._ep.renderer is not None
                and (self._ep.road_map is road_map)
            ):
                renderer = self._ep.renderer  # reuse precomputed rays for the same map
            else:
                renderer = CameraRenderer(road_map, self.cfg.camera)
        self._ep = _Episode(scenario, road_map, route, npc_paths, parked, renderer)
        self._rng = traffic_rng
        self._state = SimState(
            scenario_key=scenario.key,
            t=0.0,
            step=0,
            ego=ego,
            route_s=ego_s,
            npcs=npcs,
            peds=[],
            next_ped_spawn=next_spawn,
            rng_state=traffic_rng.bit_generator.state,
        )
        return self._result([])

    def step(self, action: ControlCommand) -> StepResult:
        st, ep = self.state, self.episode
        if st.done:
            raise RuntimeError("episode is done; call reset()")
        tm = self.cfg.timing
        events: list[Event] = []

        # Leaders are evaluated once per control step (NPC "reaction" at 10 Hz).
        boxes = self._agent_boxes(include_ego=True)
        npc_acc = []
        for i, npc in enumerate(st.npcs):
            if not npc.active:
                npc_acc.append(0.0)
                continue
            path = ep.npc_paths[npc.path_index]
            gap, dv = ag.npc_leader(npc, path, boxes, i, ep.road_map, st.t, self.cfg.idm)
            v0 = ag.npc_desired_speed(npc, path, self.cfg.idm)
            npc_acc.append(ag.idm_accel(npc.v, v0, gap, dv, self.cfg.idm))

        for k in range(tm.substeps):
            st.ego = step_vehicle(st.ego, action, self.cfg.vehicle, tm.dt_sim)
            for npc, acc in zip(st.npcs, npc_acc, strict=True):
                if npc.active:
                    ag.step_npc(npc, acc, ep.npc_paths[npc.path_index], tm.dt_sim)
            self._step_pedestrians(tm.dt_sim)
            st.t += tm.dt_sim
            collided = self._check_collision()
            if collided is not None:
                events.append(Event(EventType.COLLISION, st.t, collided))
                # Physics freezes at contact, but the clock completes the control period so
                # observations stay on the uniform 10 Hz grid.
                st.t += (tm.substeps - 1 - k) * tm.dt_sim
                break
        st.step += 1
        st.rng_state = self._rng.bit_generator.state
        events.extend(self._detect_events())
        st.done = any(e.type in TERMINAL_EVENTS for e in events)
        return self._result(events)

    def get_state(self) -> bytes:
        return pickle.dumps(self.state, protocol=pickle.HIGHEST_PROTOCOL)

    def set_state(self, blob: bytes) -> StepResult:
        state: SimState = pickle.loads(blob)
        if state.scenario_key != self.episode.spec.key:
            raise ValueError("snapshot belongs to a different scenario")
        self._state = state
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state.rng_state
        return self._result([])

    # ---------------------------------------------------------------- scenario construction

    def _place_parked(self, road_map, spec, rng, ego_xy) -> list[ag.ParkedCar]:
        cfg = spec.map
        cars: list[ag.ParkedCar] = []
        lanes = road_map.road_lanes
        for _ in range(spec.traffic.num_parked * 20):
            if len(cars) >= spec.traffic.num_parked:
                break
            lane = lanes[int(rng.integers(len(lanes)))]
            if lane.path.length < 25.0:
                continue
            s = rng.uniform(8.0, lane.path.length - 12.0)
            x, y, yaw = lane.path.interpolate(s)
            intrusion = rng.uniform(*spec.traffic.parked_intrusion)
            lateral = 0.5 * cfg.lane_width + 0.5 * ag.NPC_WIDTH - intrusion
            pos = np.array([x, y]) + lateral * np.array([math.sin(yaw), -math.cos(yaw)])
            far_from_ego = np.linalg.norm(pos - ego_xy) > spec.traffic.ego_clear_radius
            far_from_cars = all(np.hypot(c.x - pos[0], c.y - pos[1]) > 8.0 for c in cars)
            far_from_cw = all(np.linalg.norm(pos - cw.center) > 10.0 for cw in road_map.crosswalks)
            if far_from_ego and far_from_cars and far_from_cw:
                cars.append(ag.ParkedCar(float(pos[0]), float(pos[1]), float(yaw)))
        return cars

    def _spawn_npcs(self, road_map, spec, rng, ego_xy, parked):
        tc = spec.traffic
        paths: list[Route] = []
        npcs: list[ag.NPCState] = []
        occupied = [ego_xy] + [np.array([c.x, c.y]) for c in parked]
        min_len = spec.max_duration * spec.map.speed_limit * 1.2
        lanes = road_map.road_lanes
        for _ in range(tc.num_vehicles * 20):
            if len(npcs) >= tc.num_vehicles:
                break
            lane = lanes[int(rng.integers(len(lanes)))]
            s = float(rng.uniform(3.0, max(3.5, lane.path.length - 3.0)))
            pos = lane.path.interpolate(s)[:2]
            if np.linalg.norm(pos - ego_xy) < tc.ego_clear_radius:
                continue
            if any(np.linalg.norm(pos - o) < tc.npc_min_spawn_gap for o in occupied[1:]):
                continue
            route = build_route(road_map, random_lane_walk(road_map, rng, lane.id, min_len + s))
            paths.append(route)
            factor = float(rng.uniform(*tc.npc_speed_factor))
            npcs.append(ag.NPCState(len(paths) - 1, s, 0.5 * spec.map.speed_limit * factor, factor))
            occupied.append(pos)
        return paths, npcs

    # ---------------------------------------------------------------- dynamics helpers

    def _step_pedestrians(self, dt: float) -> None:
        st, ep = self.state, self.episode
        cws = ep.road_map.crosswalks
        if st.peds:
            vehicles = [
                b for b in self._agent_boxes(include_ego=True) if b.kind != ag.AgentKind.PEDESTRIAN
            ]
            st.peds = [p for p in st.peds if ag.step_ped(p, cws[p.crosswalk], dt, vehicles)]
        tc = ep.spec.traffic
        for i, cw in enumerate(cws):
            if st.t >= st.next_ped_spawn[i]:
                st.peds.append(
                    ag.PedState(
                        crosswalk=i,
                        direction=float(self._rng.choice([-1.0, 1.0])),
                        progress=0.0,
                        speed=float(self._rng.uniform(*tc.ped_speed)),
                        along=float(self._rng.uniform(-0.4, 0.4) * cw.depth),
                    )
                )
                st.next_ped_spawn[i] = st.t + float(self._rng.exponential(tc.ped_mean_interval))

    def _agent_boxes(self, include_ego: bool = False) -> list[ag.Box]:
        st, ep = self.state, self.episode
        boxes = [
            ag.npc_box(n, ep.npc_paths[n.path_index], i) for i, n in enumerate(st.npcs) if n.active
        ]
        cws = ep.road_map.crosswalks
        boxes += [ag.ped_box(p, cws[p.crosswalk], 10_000 + k) for k, p in enumerate(st.peds)]
        boxes += [ag.parked_box(c, 20_000 + k) for k, c in enumerate(ep.parked)]
        if include_ego:
            vp = self.cfg.vehicle
            off = vp.center_offset
            e = st.ego
            boxes.append(
                ag.Box(
                    ag.AgentKind.VEHICLE,
                    e.x + off * math.cos(e.yaw),
                    e.y + off * math.sin(e.yaw),
                    e.yaw,
                    e.v,
                    vp.length,
                    vp.width,
                    vp.height,
                    -1,
                )
            )
        return boxes

    def _check_collision(self) -> dict[str, float] | None:
        st = self.state
        ego_fp = vehicle_footprint(st.ego, self.cfg.vehicle)
        centre = ego_fp.mean(axis=0)
        for box in self._agent_boxes():
            if math.hypot(box.x - centre[0], box.y - centre[1]) > 8.0:
                continue
            if boxes_overlap(ego_fp, box.corners()):
                return {
                    "agent_kind": float(box.kind),
                    "agent_id": float(box.ident),
                    "ego_speed": st.ego.v,
                    "agent_speed": box.v,
                }
        return None

    def _detect_events(self) -> list[Event]:
        st, ep, ev = self.state, self.episode, self.cfg.events
        route = ep.route
        events: list[Event] = []
        t = st.t

        prev_s = st.route_s
        s, _, dist = route.path.project(np.array([st.ego.x, st.ego.y]), prev_s - 5.0, prev_s + 25.0)
        st.route_s = s

        fp = vehicle_footprint(st.ego, self.cfg.vehicle)
        probe = np.concatenate([fp, 0.5 * (fp + np.roll(fp, -1, axis=0))])
        if not ep.road_map.is_drivable(probe).all():
            events.append(Event(EventType.OFF_ROAD, t))

        lateral = self._centre_lateral(fp)
        half_lane = 0.5 * ep.spec.map.lane_width
        if not st.in_lane_departure and abs(lateral) > half_lane:
            st.in_lane_departure = True
            events.append(Event(EventType.LANE_DEPARTURE, t, {"lateral": lateral}))
        elif st.in_lane_departure and abs(lateral) < half_lane - ev.lane_departure_hysteresis:
            st.in_lane_departure = False

        front_offset = self.cfg.vehicle.length - self.cfg.vehicle.rear_overhang
        while st.next_stop_index < len(route.stop_lines):
            stop = route.stop_lines[st.next_stop_index]
            if s + front_offset < stop.s:
                break
            state = ep.road_map.signals[stop.node].state(t, stop.approach)
            if state == SignalState.RED and prev_s + front_offset < stop.s:
                events.append(Event(EventType.RED_LIGHT, t, {"node": float(stop.node)}))
            st.next_stop_index += 1

        if dist > ev.route_deviation:
            events.append(Event(EventType.ROUTE_DEVIATION, t, {"distance": dist}))

        st.stall_time = st.stall_time + self.cfg.timing.control_period if st.ego.v < 0.1 else 0.0
        if st.stall_time >= ev.stall_timeout:
            events.append(Event(EventType.STALL, t))

        if s >= route.length - ev.goal_tolerance:
            events.append(Event(EventType.ROUTE_COMPLETE, t))
        elif t >= ep.spec.max_duration - 1e-9:
            events.append(Event(EventType.TIMEOUT, t))
        return events

    def _centre_lateral(self, fp: np.ndarray) -> float:
        st = self.state
        centre = fp.mean(axis=0)
        off = self.cfg.vehicle.center_offset
        _, lateral, _ = self.episode.route.path.project(
            centre, st.route_s - 5.0, st.route_s + off + 10.0
        )
        return lateral

    # ---------------------------------------------------------------- outputs

    def signal_heads(self) -> list[SignalHeadView]:
        st, ep = self.state, self.episode
        size = ep.road_map.config.signal_head_size
        out = []
        for sig in ep.road_map.signals.values():
            for heads in (sig.heads, sig.near_heads):
                for k, head in enumerate(heads):
                    dx, dy = head.position[0] - st.ego.x, head.position[1] - st.ego.y
                    if math.hypot(dx, dy) > 90.0:
                        continue
                    out.append(
                        SignalHeadView(
                            float(head.position[0]),
                            float(head.position[1]),
                            float(head.position[2]),
                            head.facing_yaw,
                            sig.state(st.t, k),
                            size,
                        )
                    )
        return out

    def _result(self, events: list[Event]) -> StepResult:
        st, ep = self.state, self.episode
        boxes = self._agent_boxes()
        pose = Pose2D(st.ego.x, st.ego.y, st.ego.yaw)
        if ep.renderer is not None:
            image = ep.renderer.render(pose, boxes, self.signal_heads())
        else:
            image = np.zeros((self.cfg.camera.height, self.cfg.camera.width, 3), np.uint8)
        command = ep.route.command_at(st.route_s)
        obs = Observation(st.t, st.step, image, float(st.ego.v), command)
        fp = vehicle_footprint(st.ego, self.cfg.vehicle)
        next_stop = None
        if st.next_stop_index < len(ep.route.stop_lines):
            stop = ep.route.stop_lines[st.next_stop_index]
            next_stop = {
                "s": stop.s,
                "state": ep.road_map.signals[stop.node].state(st.t, stop.approach),
            }
        priv = PrivilegedState(
            timestamp=st.t,
            ego_pose=pose,
            ego_speed=float(st.ego.v),
            ego_steer=float(st.ego.steer),
            ego_footprint=fp,
            route_progress=st.route_s,
            route_length=ep.route.length,
            route_lateral=self._centre_lateral(fp),
            agents=tuple(boxes),
            extra={
                "next_stop": next_stop,
                "route": ep.route,
                "map": ep.road_map,
                "in_lane_departure": st.in_lane_departure,
            },
        )
        return StepResult(obs, priv, tuple(events), st.done)
