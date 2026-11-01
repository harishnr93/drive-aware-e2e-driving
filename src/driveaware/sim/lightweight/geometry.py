"""Planar geometry: arc-length polylines, oriented boxes and box-box collision (SAT)."""

from __future__ import annotations

import math

import numpy as np

from driveaware.core.frames import wrap_angle


class Polyline:
    """Piecewise-linear curve parameterised by arc length ``s`` (metres).

    Headings are per segment; :meth:`resample` with a small step gives smooth headings.
    """

    def __init__(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        # Drop consecutive duplicates (zero-length segments).
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("a polyline needs at least two distinct points")
        self.points = pts
        seg = np.diff(pts, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.seg_dir = seg / self.seg_len[:, None]
        self.seg_yaw = np.arctan2(seg[:, 1], seg[:, 0])
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def _segment_index(self, s: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seg_len) - 1)

    def interpolate(self, s: float | np.ndarray) -> np.ndarray:
        """Return [..., 3] array of (x, y, yaw) at arc length(s), clamped to [0, length]."""
        s_arr = np.clip(np.asarray(s, dtype=np.float64), 0.0, self.length)
        i = self._segment_index(s_arr)
        xy = self.points[i] + self.seg_dir[i] * (s_arr - self.s[i])[..., None]
        return np.concatenate([xy, self.seg_yaw[i][..., None]], axis=-1)

    def heading_at(self, s: float) -> float:
        return float(self.seg_yaw[self._segment_index(np.asarray(s))])

    def project(
        self, point: np.ndarray, s_min: float = 0.0, s_max: float | None = None
    ) -> tuple[float, float, float]:
        """Closest point on the curve within [s_min, s_max].

        Returns (s, signed lateral offset (left positive), distance).
        """
        s_max = self.length if s_max is None else s_max
        lo = int(self._segment_index(np.asarray(max(s_min, 0.0))))
        hi = int(self._segment_index(np.asarray(min(s_max, self.length)))) + 1
        p = np.asarray(point, dtype=np.float64)
        a = self.points[lo:hi]
        d = self.seg_dir[lo:hi]
        t = np.clip(np.einsum("ij,ij->i", p - a, d), 0.0, self.seg_len[lo:hi])
        closest = a + d * t[:, None]
        dist = np.linalg.norm(p - closest, axis=1)
        k = int(np.argmin(dist))
        rel = p - closest[k]
        lateral = d[k, 0] * rel[1] - d[k, 1] * rel[0]
        s = float(np.clip(self.s[lo + k] + t[k], s_min, s_max))
        return s, float(lateral), float(dist[k])

    def resample(self, step: float) -> Polyline:
        n = max(2, int(math.ceil(self.length / step)) + 1)
        return Polyline(self.interpolate(np.linspace(0.0, self.length, n))[:, :2])

    def slice(self, s0: float, s1: float, step: float = 0.5) -> Polyline:
        s0, s1 = max(0.0, s0), min(self.length, s1)
        n = max(2, int(math.ceil((s1 - s0) / step)) + 1)
        return Polyline(self.interpolate(np.linspace(s0, s1, n))[:, :2])

    def offset(self, lateral: float) -> Polyline:
        """Parallel curve shifted ``lateral`` metres to the left (negative = right)."""
        normals = np.stack([-self.seg_dir[:, 1], self.seg_dir[:, 0]], axis=1)
        vertex_n = np.empty_like(self.points)
        vertex_n[0], vertex_n[-1] = normals[0], normals[-1]
        avg = normals[:-1] + normals[1:]
        vertex_n[1:-1] = avg / np.linalg.norm(avg, axis=1, keepdims=True)
        return Polyline(self.points + lateral * vertex_n)

    def curvature(self) -> np.ndarray:
        """Unsigned curvature (1/m) at each interior vertex; endpoints copy their neighbours."""
        if len(self.seg_yaw) < 2:
            return np.zeros(len(self.points))
        dyaw = np.abs(wrap_angle(np.diff(self.seg_yaw)))
        ds = 0.5 * (self.seg_len[:-1] + self.seg_len[1:])
        k = dyaw / ds
        return np.concatenate([[k[0]], k, [k[-1]]])

    @staticmethod
    def concatenate(parts: list[Polyline]) -> Polyline:
        return Polyline(np.concatenate([p.points for p in parts], axis=0))


def box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    """Corners [4, 2] of a box centred at (x, y), CCW from front-left."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def boxes_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex quadrilaterals given as [4, 2] corners."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        axes = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
        for axis in axes[:2]:  # rectangles: two unique axes each
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def bezier_quadratic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def hermite(
    p0: np.ndarray, yaw0: float, p1: np.ndarray, yaw1: float, n: int, tangent_scale: float = 1.2
) -> np.ndarray:
    """Cubic Hermite curve between two poses; tangent magnitude scales with the gap."""
    gap = float(np.linalg.norm(p1 - p0))
    k = tangent_scale * gap
    m0 = k * np.array([math.cos(yaw0), math.sin(yaw0)])
    m1 = k * np.array([math.cos(yaw1), math.sin(yaw1)])
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
