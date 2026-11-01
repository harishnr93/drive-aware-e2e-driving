"""Pseudo-perspective front camera and a top-down debug view.

Camera model: pinhole camera mounted ``x_offset`` m ahead of the rear axle at height
``height``, pitched down by ``pitch_deg``. Camera axes follow OpenCV (x right, y down,
z forward). For every pixel below the horizon, the ray's ground intersection is constant in
the ego frame, so it is precomputed once; each frame only transforms these points to the
world and looks up the map's colour raster. Agents and signal heads are drawn as shaded,
near-plane-clipped cuboids in far-to-near order. Rendering is supersampled ``ssaa`` times
and area-downsampled to reduce aliasing.

The top-down view is for replay/debugging only and is never a policy input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from driveaware.core.frames import Pose2D, world_to_ego
from driveaware.core.types import IMAGE_HEIGHT, IMAGE_WIDTH
from driveaware.sim.lightweight.agents import AgentKind, Box
from driveaware.sim.lightweight.roadmap import PALETTE, RoadMap, Sem, SignalState

SKY_TOP = np.array([110, 160, 220], dtype=np.float32)
SKY_HORIZON = np.array([190, 210, 230], dtype=np.float32)
FOG = np.array([170, 185, 190], dtype=np.float32)
SIGNAL_COLORS = {
    SignalState.GREEN: (40, 230, 70),
    SignalState.YELLOW: (250, 200, 30),
    SignalState.RED: (240, 40, 30),
}
SIGNAL_BODY = (35, 35, 35)
KIND_COLORS = {AgentKind.PEDESTRIAN: (200, 60, 160), AgentKind.PARKED: (90, 110, 150)}
VEHICLE_COLORS = [
    (180, 30, 30),
    (30, 70, 170),
    (220, 220, 220),
    (30, 30, 30),
    (200, 160, 40),
    (60, 140, 90),
]
LIGHT_DIR = np.array([0.3, 0.5, 0.81])  # world-frame unit-ish light direction


@dataclass(frozen=True)
class CameraConfig:
    width: int = IMAGE_WIDTH
    height: int = IMAGE_HEIGHT
    hfov_deg: float = 90.0
    pitch_deg: float = 8.0  # downwards
    mount_height: float = 1.5  # m
    x_offset: float = 1.5  # m ahead of rear axle
    max_distance: float = 80.0  # m, ground beyond this is fog
    fog_start: float = 35.0  # m
    ssaa: int = 2
    signal_head_size: tuple[float, float, float] = (1.0, 0.8, 1.8)  # width, depth, height (m)


@dataclass(frozen=True)
class SignalHeadView:
    x: float
    y: float
    z: float  # centre height
    facing_yaw: float
    state: SignalState


class CameraRenderer:
    def __init__(self, road_map: RoadMap, config: CameraConfig = CameraConfig()) -> None:
        self.map = road_map
        self.cfg = config
        s = config.ssaa
        self.w, self.h = config.width * s, config.height * s
        self.fx = self.fy = (self.w / 2) / math.tan(math.radians(config.hfov_deg) / 2)
        self.cx, self.cy = self.w / 2, self.h / 2
        p = math.radians(config.pitch_deg)
        # Rows: camera x (right), y (down), z (forward) axes expressed in the ego frame.
        self.R = np.array(
            [[0.0, -1.0, 0.0], [-math.sin(p), 0.0, -math.cos(p)], [math.cos(p), 0.0, -math.sin(p)]]
        )
        self.cam_pos = np.array([config.x_offset, 0.0, config.mount_height])
        self._precompute_ground()

    def _precompute_ground(self) -> None:
        u = np.arange(self.w) + 0.5
        v = np.arange(self.h) + 0.5
        uu, vv = np.meshgrid(u, v)
        rays_cam = np.stack(
            [(uu - self.cx) / self.fx, (vv - self.cy) / self.fy, np.ones_like(uu)], axis=-1
        )
        rays_ego = rays_cam @ self.R  # R^T applied to row vectors
        dz = rays_ego[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dz < -1e-6, -self.cam_pos[2] / dz, np.inf)
        ground = self.cam_pos[None, None, :2] + t[..., None] * rays_ego[..., :2]
        dist = t * np.linalg.norm(rays_ego, axis=-1)
        self.ground_mask = dist < self.cfg.max_distance
        self.ground_ego = ground[self.ground_mask]  # [M, 2]
        fog = np.clip(
            (dist[self.ground_mask] - self.cfg.fog_start)
            / (self.cfg.max_distance - self.cfg.fog_start),
            0.0,
            1.0,
        )
        alpha = np.round(fog**1.5 * 256).astype(np.uint16)[:, None]
        self.fog_keep = (256 - alpha).astype(np.uint16)
        self.fog_term = (alpha * FOG.astype(np.uint16)).astype(np.uint16)
        self.ground_ego32 = self.ground_ego.astype(np.float32)
        flat = self.map.color.reshape(-1, 3)
        # Extra last row = grass, used for out-of-bounds lookups.
        self.color_flat = np.vstack([flat, PALETTE[Sem.GRASS][None]])
        # Sky gradient (also used for ground beyond max_distance, blended into fog colour).
        elev = np.clip((self.cy - vv) / self.h * 2.0, 0.0, 1.0)[..., None]
        bg = SKY_HORIZON * (1 - elev) + SKY_TOP * elev
        bg = np.where((dz < -1e-6)[..., None], FOG, bg)
        self.background = bg.astype(np.uint8)

    # ----------------------------------------------------------------------------------------

    def render(
        self, ego_pose: Pose2D, boxes: list[Box], signal_heads: list[SignalHeadView] = ()
    ) -> np.ndarray:
        img = self.background.copy()
        img[self.ground_mask] = self._ground_colors(ego_pose)

        boxes = [
            b
            for b, keep in zip(
                boxes, self._visible([(b.x, b.y) for b in boxes], ego_pose), strict=True
            )
            if keep
        ]
        heads = [
            h
            for h, keep in zip(
                signal_heads,
                self._visible([(h.x, h.y) for h in signal_heads], ego_pose),
                strict=True,
            )
            if keep
        ]
        solids = [self._box_solid(b, ego_pose) for b in boxes]
        solids += [self._signal_solid(h, ego_pose) for h in heads]
        solids = [s for s in solids if s is not None]
        solids.sort(key=lambda s: -s[0])  # far to near (painter's algorithm)
        for _, faces in solids:
            for poly, color in faces:
                pts = np.round(poly * 16).astype(np.int32)
                cv2.fillConvexPoly(img, pts, color, cv2.LINE_AA, 4)
        if self.cfg.ssaa > 1:
            img = cv2.resize(img, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)
        return img

    def _ground_colors(self, ego_pose: Pose2D) -> np.ndarray:
        res = self.map.config.resolution
        c, s = math.cos(ego_pose.yaw), math.sin(ego_pose.yaw)
        # Ego -> raster pixel coordinates in one affine map (float32).
        a = np.array([[c, s], [-s, c]], dtype=np.float32) / res
        b = (np.array([ego_pose.x, ego_pose.y]) - self.map.origin) / res
        px = self.ground_ego32 @ a + b.astype(np.float32)
        col = px[:, 0].astype(np.int64)
        row = px[:, 1].astype(np.int64)
        h, w = self.map.semantic.shape
        inside = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (col < w) & (row < h)
        idx = np.where(inside, row * w + col, h * w)
        colors = self.color_flat.take(idx, axis=0).astype(np.uint16)
        return ((colors * self.fog_keep + self.fog_term) >> 8).astype(np.uint8)

    def _visible(self, centres: list[tuple[float, float]], ego_pose: Pose2D) -> np.ndarray:
        """Cheap frustum pre-cull on object centres (generous margin for object extent)."""
        if not centres:
            return np.zeros(0, dtype=bool)
        rel = world_to_ego(np.array(centres), ego_pose) - self.cam_pos[:2]
        dist = np.linalg.norm(rel, axis=1)
        half_fov = math.radians(self.cfg.hfov_deg) / 2
        bearing = np.abs(np.arctan2(rel[:, 1], rel[:, 0]))
        near = dist < 6.0  # anything this close may still overlap the view
        return (dist < self.cfg.max_distance) & ((bearing < half_fov + 0.35) | near)

    def _box_solid(self, box: Box, ego_pose: Pose2D):
        if box.kind == AgentKind.VEHICLE:
            base = VEHICLE_COLORS[box.ident % len(VEHICLE_COLORS)]
        else:
            base = KIND_COLORS[box.kind]
        corners = box.corners()
        return self._cuboid(corners, 0.0, box.height, ego_pose, lambda _f: base)

    def _signal_solid(self, head: SignalHeadView, ego_pose: Pose2D):
        w, d, h = self.cfg.signal_head_size
        c, s = math.cos(head.facing_yaw), math.sin(head.facing_yaw)
        local = np.array([[d / 2, w / 2], [-d / 2, w / 2], [-d / 2, -w / 2], [d / 2, -w / 2]])
        corners = local @ np.array([[c, s], [-s, c]]) + np.array([head.x, head.y])
        lit = SIGNAL_COLORS[head.state]

        def color(face_normal: np.ndarray) -> tuple[int, int, int]:
            # The lit face is the one pointing along facing_yaw.
            return lit if face_normal[:2] @ np.array([c, s]) > 0.9 else SIGNAL_BODY

        return self._cuboid(corners, head.z - h / 2, head.z + h / 2, ego_pose, color, shade=False)

    def _cuboid(self, corners_xy, z0, z1, ego_pose, color_fn, shade=True):
        ego_xy = world_to_ego(corners_xy, ego_pose)
        centre = ego_xy.mean(axis=0)
        rel = np.array([centre[0], centre[1], 0.5 * (z0 + z1)]) - self.cam_pos
        depth = float(np.linalg.norm(rel))
        if depth > self.cfg.max_distance or (self.R @ rel)[2] < -5.0:
            return None
        bottom = np.column_stack([ego_xy, np.full(4, z0)])
        top = np.column_stack([ego_xy, np.full(4, z1)])
        faces = [top[::-1]] + [
            np.array([bottom[i], bottom[(i + 1) % 4], top[(i + 1) % 4], top[i]]) for i in range(4)
        ]
        centre3 = np.concatenate([bottom, top]).mean(axis=0)
        out = []
        yaw = ego_pose.yaw
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        for face in faces:
            fc = face.mean(axis=0)
            normal = np.cross(face[1] - face[0], face[2] - face[0])
            normal /= np.linalg.norm(normal) + 1e-12
            if normal @ (fc - centre3) < 0:
                normal = -normal  # make outward
            if normal @ (fc - self.cam_pos) >= 0:
                continue  # back face
            poly = self._project_clipped(face)
            if poly is None:
                continue
            n_world = np.concatenate([rot @ normal[:2], normal[2:]])
            base = np.array(color_fn(n_world), dtype=np.float32)
            if shade:
                base = base * (0.55 + 0.45 * max(0.0, float(n_world @ LIGHT_DIR)))
            out.append((poly, tuple(int(v) for v in np.clip(base, 0, 255))))
        return depth, out

    def _project_clipped(self, face_ego: np.ndarray, near: float = 0.2) -> np.ndarray | None:
        cam = (face_ego - self.cam_pos) @ self.R.T
        poly = _clip_near(cam, near)
        if poly is None:
            return None
        u = self.fx * poly[:, 0] / poly[:, 2] + self.cx
        v = self.fy * poly[:, 1] / poly[:, 2] + self.cy
        uv = np.column_stack([u, v])
        if (
            uv[:, 0].max() < 0
            or uv[:, 0].min() > self.w
            or uv[:, 1].max() < 0
            or uv[:, 1].min() > self.h
        ):
            return None
        return np.clip(uv, -4 * self.w, 4 * self.w)


def _clip_near(poly: np.ndarray, near: float) -> np.ndarray | None:
    """Sutherland-Hodgman clip of a camera-frame polygon against the plane z = near."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        a_in, b_in = a[2] >= near, b[2] >= near
        if a_in:
            out.append(a)
        if a_in != b_in:
            t = (near - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else None


# --------------------------------------------------------------------------------------------


class TopDownRenderer:
    """North-up debug view centred on the ego."""

    def __init__(self, road_map: RoadMap, size_m: float = 80.0, px_per_m: float = 5.0) -> None:
        self.map = road_map
        self.size_m = size_m
        self.ppm = px_per_m
        self.n = int(size_m * px_per_m)

    def render(
        self,
        ego_pose: Pose2D,
        ego_corners: np.ndarray,
        boxes: list[Box],
        route_xy: np.ndarray | None = None,
        signals: list[tuple[np.ndarray, SignalState]] = (),
        predicted_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        half = self.size_m / 2
        xs = ego_pose.x + np.linspace(-half, half, self.n)
        ys = ego_pose.y + np.linspace(half, -half, self.n)  # north up
        gx, gy = np.meshgrid(xs, ys)
        grid = np.stack([gx, gy], axis=-1)
        row, col = self.map.world_to_pixel(grid)
        img = self.map.color[row, col].copy()
        img[~self.map.in_bounds(grid)] = PALETTE[Sem.GRASS]

        def to_px(xy: np.ndarray) -> np.ndarray:
            xy = np.atleast_2d(xy)
            u = (xy[:, 0] - ego_pose.x + half) * self.ppm
            v = (half - (xy[:, 1] - ego_pose.y)) * self.ppm
            return np.round(np.column_stack([u, v]) * 16).astype(np.int32)

        if route_xy is not None:
            cv2.polylines(img, [to_px(route_xy)], False, (60, 160, 255), 2, cv2.LINE_AA, 4)
        for pos, state in signals:
            cv2.circle(
                img,
                tuple(int(v) for v in to_px(pos)[0]),
                6 * 16,
                SIGNAL_COLORS[state],
                -1,
                cv2.LINE_AA,
                4,
            )
        for b in boxes:
            color = (
                VEHICLE_COLORS[b.ident % len(VEHICLE_COLORS)]
                if b.kind == AgentKind.VEHICLE
                else KIND_COLORS[b.kind]
            )
            cv2.fillConvexPoly(img, to_px(b.corners()), color, cv2.LINE_AA, 4)
        cv2.fillConvexPoly(img, to_px(ego_corners), (255, 140, 0), cv2.LINE_AA, 4)
        if predicted_xy is not None and len(predicted_xy):
            for p in to_px(predicted_xy):
                cv2.circle(img, tuple(int(v) for v in p), 2 * 16, (255, 255, 0), -1, cv2.LINE_AA, 4)
        return img
