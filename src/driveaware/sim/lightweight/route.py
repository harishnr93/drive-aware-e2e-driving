"""Routes over the lane graph and the navigation commands derived from them.

A route is a sequence of lanes concatenated into one arc-length path. The navigation command
at arc length s is non-privileged route information (equivalent to turn-by-turn GPS): it
announces the manoeuvre at the next junction from ``command_lead`` metres before the junction
until the junction has been crossed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from driveaware.core.types import Command
from driveaware.sim.lightweight.geometry import Polyline
from driveaware.sim.lightweight.roadmap import RoadMap


@dataclass(frozen=True)
class StopLine:
    s: float  # arc length along the route
    node: int
    approach: int


@dataclass(frozen=True)
class Route:
    lane_ids: tuple[int, ...]
    path: Polyline
    lane_s0: np.ndarray  # start arc length of each lane
    lane_speed_limits: np.ndarray
    command_intervals: tuple[tuple[float, float, Command], ...]
    stop_lines: tuple[StopLine, ...]

    @property
    def length(self) -> float:
        return self.path.length

    def lane_index_at(self, s: float | np.ndarray) -> np.ndarray:
        return np.clip(
            np.searchsorted(self.lane_s0, s, side="right") - 1, 0, len(self.lane_ids) - 1
        )

    def speed_limit_at(self, s: float | np.ndarray) -> np.ndarray:
        return self.lane_speed_limits[self.lane_index_at(s)]

    def command_at(self, s: float) -> Command:
        for start, end, cmd in self.command_intervals:
            if start <= s <= end:
                return cmd
        return Command.LANE_FOLLOW


def random_lane_walk(
    road_map: RoadMap, rng: np.random.Generator, start_lane: int, min_length: float
) -> list[int]:
    """Random walk over lane successors until the accumulated length reaches ``min_length``."""
    lanes = [start_lane]
    total = road_map.lanes[start_lane].path.length
    while total < min_length:
        succ = road_map.lanes[lanes[-1]].successors
        nxt = succ[int(rng.integers(len(succ)))]
        lanes.append(nxt)
        total += road_map.lanes[nxt].path.length
    return lanes


def build_route(road_map: RoadMap, lane_ids: list[int], command_lead: float = 20.0) -> Route:
    lanes = [road_map.lanes[i] for i in lane_ids]
    for a, b in zip(lanes[:-1], lanes[1:], strict=False):
        if b.id not in a.successors:
            raise ValueError(f"lane {b.id} does not follow lane {a.id}")
    path = Polyline.concatenate([ln.path for ln in lanes])
    lengths = np.array([ln.path.length for ln in lanes])
    lane_s0 = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    commands = []
    stops = []
    for ln, s0, length in zip(lanes, lane_s0, lengths, strict=True):
        if ln.kind == "connector" and ln.turn != Command.LANE_FOLLOW:
            commands.append((max(0.0, s0 - command_lead), s0 + length, ln.turn))
        if ln.signal is not None:
            stops.append(StopLine(float(s0 + length), *ln.signal))
    return Route(
        lane_ids=tuple(lane_ids),
        path=path,
        lane_s0=lane_s0,
        lane_speed_limits=np.array([ln.speed_limit for ln in lanes]),
        command_intervals=tuple(commands),
        stop_lines=tuple(stops),
    )
