"""Procedural road network and its rasterised semantic map.

A map is a jittered grid of nodes joined by straight or curved two-lane roads (right-hand
traffic) with a parking shoulder on each side. Each directed road carries one lane; lanes are
joined at nodes by Hermite connector lanes. Nodes with >= 3 roads are signalised.

The semantic raster is the single source of truth for both the camera renderer and off-road
detection, so what the policy sees and what the evaluator scores cannot disagree.
"""

from __future__ import annotations

import functools
import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

import cv2
import numpy as np

from driveaware.core.frames import wrap_angle
from driveaware.core.seeding import derive_seed, make_rng
from driveaware.core.types import Command
from driveaware.sim.lightweight.geometry import Polyline, bezier_quadratic, hermite


class Sem(IntEnum):
    GRASS = 0
    SIDEWALK = 1
    ROAD = 2
    CENTER_LINE = 3
    EDGE_LINE = 4
    STOP_LINE = 5
    CROSSWALK = 6


DRIVABLE_CLASSES = (Sem.ROAD, Sem.CENTER_LINE, Sem.EDGE_LINE, Sem.STOP_LINE, Sem.CROSSWALK)

# RGB colours per semantic class.
PALETTE = np.array(
    [
        [76, 112, 56],  # grass
        [158, 154, 146],  # sidewalk
        [72, 72, 76],  # road
        [222, 186, 40],  # centre line (yellow)
        [232, 232, 232],  # edge line
        [240, 240, 240],  # stop line
        [226, 226, 226],  # crosswalk
    ],
    dtype=np.uint8,
)


class SignalState(IntEnum):
    GREEN = 0
    YELLOW = 1
    RED = 2


@dataclass(frozen=True)
class MapConfig:
    rows: int = 4
    cols: int = 4
    spacing: float = 80.0  # m between grid nodes
    jitter: float = 6.0  # m, uniform node jitter
    edge_drop_prob: float = 0.25
    curve_prob: float = 0.35
    curve_offset: tuple[float, float] = (6.0, 16.0)  # m, Bezier control-point offset
    lane_width: float = 3.5
    shoulder: float = 2.0  # parking shoulder outside each lane
    sidewalk: float = 2.5
    intersection_trim: float = 16.0  # m from node centre where road lanes end
    speed_limit: float = 10.0  # m/s on roads
    max_lat_accel: float = 2.5  # m/s^2, sets connector speed limits
    crosswalk_prob: float = 0.4  # per road
    signal_green: float = 6.0  # s
    signal_yellow: float = 2.0
    signal_all_red: float = 1.0
    resolution: float = 0.1  # m per raster pixel
    margin: float = 30.0  # m of grass around the network
    texture_noise: float = 6.0  # std of static colour noise (motion cue)

    @property
    def road_half_width(self) -> float:
        return self.lane_width + self.shoulder


@dataclass
class Lane:
    id: int
    path: Polyline
    kind: str  # "road" | "connector"
    speed_limit: float
    start_node: int | None
    end_node: int | None
    edge: int | None = None
    node: int | None = None
    turn: Command = Command.LANE_FOLLOW
    successors: list[int] = field(default_factory=list)
    signal: tuple[int, int] | None = None  # (node, approach) controlling the stop line at the end


@dataclass(frozen=True)
class Crosswalk:
    id: int
    edge: int
    center: np.ndarray  # (2,)
    road_yaw: float  # heading of the road at the crosswalk
    half_span: float  # m, half the crossing distance (kerb to kerb)
    depth: float = 3.0  # m along the road


@dataclass(frozen=True)
class SignalHead:
    position: np.ndarray  # (3,) x, y, z of the head centre
    facing_yaw: float  # direction the lit face points (towards approaching traffic)


@dataclass
class Signal:
    node: int
    approach_lanes: list[int]
    heads: list[SignalHead]
    offset: float
    green: float
    yellow: float
    all_red: float

    @property
    def slot(self) -> float:
        return self.green + self.yellow + self.all_red

    @property
    def cycle(self) -> float:
        return self.slot * len(self.approach_lanes)

    def state(self, t: float, approach: int) -> SignalState:
        """Split phasing: approaches get green one at a time, so greens never conflict."""
        tau = (t + self.offset) % self.cycle
        active = int(tau // self.slot)
        if active != approach:
            return SignalState.RED
        phase = tau - active * self.slot
        if phase < self.green:
            return SignalState.GREEN
        if phase < self.green + self.yellow:
            return SignalState.YELLOW
        return SignalState.RED


@dataclass
class RoadMap:
    seed: int
    config: MapConfig
    nodes: np.ndarray  # [N, 2]
    edges: list[tuple[int, int]]
    roads: list[Polyline]  # untrimmed centrelines, edge u -> v
    lanes: list[Lane]
    signals: dict[int, Signal]
    crosswalks: list[Crosswalk]
    semantic: np.ndarray  # uint8 [H, W]
    color: np.ndarray  # uint8 [H, W, 3]
    origin: np.ndarray  # world (x, y) of pixel (0, 0)

    @property
    def road_lanes(self) -> list[Lane]:
        return [ln for ln in self.lanes if ln.kind == "road"]

    def node_degree(self, node: int) -> int:
        return sum(node in e for e in self.edges)

    def world_to_pixel(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (row, col) integer indices, clipped to the raster."""
        pts = np.asarray(xy, dtype=np.float64)
        col = np.floor((pts[..., 0] - self.origin[0]) / self.config.resolution).astype(np.int64)
        row = np.floor((pts[..., 1] - self.origin[1]) / self.config.resolution).astype(np.int64)
        h, w = self.semantic.shape
        return np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)

    def in_bounds(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float64)
        h, w = self.semantic.shape
        res = self.config.resolution
        col = (pts[..., 0] - self.origin[0]) / res
        row = (pts[..., 1] - self.origin[1]) / res
        return (col >= 0) & (col < w) & (row >= 0) & (row < h)

    def semantic_at(self, xy: np.ndarray) -> np.ndarray:
        row, col = self.world_to_pixel(xy)
        sem = self.semantic[row, col]
        return np.where(self.in_bounds(xy), sem, Sem.GRASS)

    def is_drivable(self, xy: np.ndarray) -> np.ndarray:
        return np.isin(self.semantic_at(xy), np.array(DRIVABLE_CLASSES, dtype=np.uint8))


# --------------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def generate_map(seed: int, config: MapConfig = MapConfig()) -> RoadMap:
    """Deterministically generate a map from ``seed``. Cached: maps are immutable."""
    rng = make_rng(derive_seed(seed, "map"))
    nodes = _grid_nodes(config, rng)
    edges = _select_edges(config, rng)
    roads = [_road_centerline(nodes[u], nodes[v], config, rng) for u, v in edges]
    lanes = _road_lanes(edges, roads, config)
    _connect_lanes(lanes, nodes, edges, config)
    signals = _signals(lanes, nodes, edges, config, rng)
    crosswalks = _crosswalks(roads, edges, config, rng)
    semantic, origin = _rasterise(nodes, roads, lanes, signals, crosswalks, config)
    color = _colorise(semantic, config, rng)
    for arr in (semantic, color, nodes):
        arr.setflags(write=False)
    return RoadMap(
        seed, config, nodes, edges, roads, lanes, signals, crosswalks, semantic, color, origin
    )


def _grid_nodes(cfg: MapConfig, rng: np.random.Generator) -> np.ndarray:
    rr, cc = np.meshgrid(np.arange(cfg.rows), np.arange(cfg.cols), indexing="ij")
    base = np.stack([cc.ravel(), rr.ravel()], axis=1) * cfg.spacing
    return base + rng.uniform(-cfg.jitter, cfg.jitter, base.shape)


def _select_edges(cfg: MapConfig, rng: np.random.Generator) -> list[tuple[int, int]]:
    def nid(r: int, c: int) -> int:
        return r * cfg.cols + c

    edges = [(nid(r, c), nid(r, c + 1)) for r in range(cfg.rows) for c in range(cfg.cols - 1)]
    edges += [(nid(r, c), nid(r + 1, c)) for r in range(cfg.rows - 1) for c in range(cfg.cols)]
    n = cfg.rows * cfg.cols
    order = [edges[i] for i in rng.permutation(len(edges))]
    for edge in order:
        if rng.random() >= cfg.edge_drop_prob:
            continue
        candidate = [e for e in edges if e != edge]
        deg = np.bincount(np.array(candidate).ravel(), minlength=n)
        # No dead ends (would require U-turns) and the network must stay connected.
        if deg.min() >= 2 and _connected(candidate, n):
            edges = candidate
    return edges


def _connected(edges: list[tuple[int, int]], n: int) -> bool:
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen, queue = {0}, deque([0])
    while queue:
        for nb in adj[queue.popleft()]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen) == n


def _road_centerline(
    p0: np.ndarray, p1: np.ndarray, cfg: MapConfig, rng: np.random.Generator
) -> Polyline:
    if rng.random() < cfg.curve_prob:
        d = p1 - p0
        normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        offset = rng.uniform(*cfg.curve_offset) * rng.choice([-1.0, 1.0])
        pts = bezier_quadratic(p0, 0.5 * (p0 + p1) + offset * normal, p1, 200)
    else:
        pts = np.stack([p0, p1])
    return Polyline(pts).resample(0.5)


def _road_lanes(edges, roads, cfg: MapConfig) -> list[Lane]:
    lanes: list[Lane] = []
    trim = cfg.intersection_trim
    for e, ((u, v), road) in enumerate(zip(edges, roads, strict=True)):
        core = road.slice(trim, road.length - trim)
        reverse = Polyline(core.points[::-1])
        for path, a, b in ((core, u, v), (reverse, v, u)):
            lanes.append(
                Lane(
                    id=len(lanes),
                    path=path.offset(-0.5 * cfg.lane_width).resample(0.5),
                    kind="road",
                    speed_limit=cfg.speed_limit,
                    start_node=a,
                    end_node=b,
                    edge=e,
                )
            )
    return lanes


def _connect_lanes(lanes: list[Lane], nodes, edges, cfg: MapConfig) -> None:
    road = list(lanes)
    for n in range(len(nodes)):
        incoming = [ln for ln in road if ln.end_node == n]
        outgoing = [ln for ln in road if ln.start_node == n]
        degree = sum(n in e for e in edges)
        for lin in incoming:
            p0 = lin.path.points[-1]
            yaw0 = lin.path.heading_at(lin.path.length)
            for lout in outgoing:
                if lout.edge == lin.edge:
                    continue  # no U-turns
                p1 = lout.path.points[0]
                yaw1 = lout.path.heading_at(0.0)
                gap = float(np.linalg.norm(p1 - p0))
                path = Polyline(hermite(p0, yaw0, p1, yaw1, max(8, int(gap / 0.5)))).resample(0.5)
                if degree <= 2:
                    turn = Command.LANE_FOLLOW
                else:
                    dyaw = float(wrap_angle(yaw1 - yaw0))
                    if abs(dyaw) < math.pi / 6:
                        turn = Command.STRAIGHT
                    else:
                        turn = Command.LEFT if dyaw > 0 else Command.RIGHT
                kmax = float(path.curvature().max())
                limit = min(cfg.speed_limit, math.sqrt(cfg.max_lat_accel / max(kmax, 1e-6)))
                conn = Lane(
                    id=len(lanes),
                    path=path,
                    kind="connector",
                    speed_limit=limit,
                    start_node=n,
                    end_node=n,
                    node=n,
                    turn=turn,
                    successors=[lout.id],
                )
                lanes.append(conn)
                lin.successors.append(conn.id)


def _signals(lanes, nodes, edges, cfg: MapConfig, rng: np.random.Generator) -> dict[int, Signal]:
    signals: dict[int, Signal] = {}
    for n in range(len(nodes)):
        if sum(n in e for e in edges) < 3:
            continue
        approaches = [ln for ln in lanes if ln.kind == "road" and ln.end_node == n]
        approaches.sort(key=lambda ln: ln.path.heading_at(ln.path.length))
        heads = []
        for k, ln in enumerate(approaches):
            ln.signal = (n, k)
            end = ln.path.points[-1]
            yaw = ln.path.heading_at(ln.path.length)
            # Head on the far side of the junction, over the approach lane's extension, so
            # it remains in view while waiting at the stop line.
            ahead = end + 2.0 * cfg.intersection_trim * np.array([math.cos(yaw), math.sin(yaw)])
            heads.append(SignalHead(np.array([*ahead, 5.2]), float(wrap_angle(yaw + math.pi))))
        sig = Signal(
            n,
            [ln.id for ln in approaches],
            heads,
            0.0,
            cfg.signal_green,
            cfg.signal_yellow,
            cfg.signal_all_red,
        )
        sig.offset = float(rng.uniform(0.0, sig.cycle))
        signals[n] = sig
    return signals


def _crosswalks(roads, edges, cfg: MapConfig, rng: np.random.Generator) -> list[Crosswalk]:
    out: list[Crosswalk] = []
    lo = cfg.intersection_trim + 12.0
    for e, road in enumerate(roads):
        if road.length - 2 * lo < 5.0 or rng.random() >= cfg.crosswalk_prob:
            continue
        s = rng.uniform(lo, road.length - lo)
        x, y, yaw = road.interpolate(s)
        out.append(Crosswalk(len(out), e, np.array([x, y]), float(yaw), cfg.road_half_width + 1.0))
    return out


# --------------------------------------------------------------------------------------------
# Rasterisation
# --------------------------------------------------------------------------------------------

_SHIFT = 4  # cv2 sub-pixel bits


def _rasterise(nodes, roads, lanes, signals, crosswalks, cfg: MapConfig):
    all_pts = np.concatenate([r.points for r in roads])
    origin = all_pts.min(axis=0) - cfg.margin
    size = all_pts.max(axis=0) + cfg.margin - origin
    w, h = (np.ceil(size / cfg.resolution)).astype(int)
    sem = np.full((h, w), Sem.GRASS, dtype=np.uint8)
    res = cfg.resolution

    def px(points: np.ndarray) -> np.ndarray:
        return np.round((points - origin) / res * (1 << _SHIFT)).astype(np.int32)

    def line(points: np.ndarray, width: float, value: int) -> None:
        thickness = max(1, int(round(width / res)))
        cv2.polylines(sem, [px(points)], False, int(value), thickness, cv2.LINE_8, _SHIFT)

    def disk(center: np.ndarray, radius: float, value: int) -> None:
        c = tuple(int(v) for v in px(center))
        cv2.circle(
            sem, c, int(round(radius / res * (1 << _SHIFT))), int(value), -1, cv2.LINE_8, _SHIFT
        )

    def poly(points: np.ndarray, value: int) -> None:
        cv2.fillPoly(sem, [px(points)], int(value), cv2.LINE_8, _SHIFT)

    half = cfg.road_half_width
    junction_r = 0.9 * cfg.intersection_trim
    for road in roads:
        line(road.points, 2 * (half + cfg.sidewalk), Sem.SIDEWALK)
    for p in nodes:
        disk(p, junction_r + cfg.sidewalk, Sem.SIDEWALK)
    for road in roads:
        line(road.points, 2 * half, Sem.ROAD)
    for p in nodes:
        disk(p, junction_r, Sem.ROAD)
    for ln in lanes:
        if ln.kind == "connector":
            line(ln.path.points, cfg.lane_width, Sem.ROAD)

    trim = cfg.intersection_trim
    for road in roads:
        core = road.slice(trim, road.length - trim, 0.25)
        for side in (-1.0, 1.0):
            line(core.offset(side * cfg.lane_width).points, 0.15, Sem.EDGE_LINE)
        s = 0.0
        while s < core.length:  # 3 m dash, 3 m gap
            line(core.slice(s, min(s + 3.0, core.length), 0.25).points, 0.15, Sem.CENTER_LINE)
            s += 6.0

    for sig in signals.values():
        for lane_id in sig.approach_lanes:
            path = lanes[lane_id].path
            x, y, yaw = path.interpolate(path.length - 0.2)
            n = np.array([-math.sin(yaw), math.cos(yaw)])
            hw = 0.5 * cfg.lane_width
            line(np.array([[x, y] + hw * n, [x, y] - hw * n]), 0.4, Sem.STOP_LINE)

    for cw in crosswalks:
        along = np.array([math.cos(cw.road_yaw), math.sin(cw.road_yaw)])
        across = np.array([-along[1], along[0]])
        for off in np.arange(-half + 0.25, half, 1.0):  # 0.5 m stripes, 1 m pitch
            c = cw.center + off * across
            pts = [
                c + a * along * cw.depth / 2 + b * across * 0.25
                for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))
            ]
            poly(np.array(pts), Sem.CROSSWALK)
    return sem, origin


def _colorise(sem: np.ndarray, cfg: MapConfig, rng: np.random.Generator) -> np.ndarray:
    color = PALETTE[sem].astype(np.int16)
    noise = rng.normal(0.0, cfg.texture_noise, sem.shape).astype(np.int16)
    color += noise[..., None]
    return np.clip(color, 0, 255).astype(np.uint8)
