"""2D rigid transforms between the world and ego frames, plus CARLA conversions.

Frames (docs/system_architecture.md §3):
- world: right-handed, x east, y north, yaw CCW from +x, radians.
- ego:   origin at rear axle, x forward, y left.
- CARLA: left-handed, x forward, y right, yaw in degrees (clockwise seen from above),
         actor origin at the bounding-box centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    wrapped = np.mod(np.asarray(angle) + np.pi, 2.0 * np.pi) - np.pi
    wrapped = np.where(wrapped <= -np.pi, wrapped + 2.0 * np.pi, wrapped)
    if np.ndim(wrapped) == 0:
        return float(wrapped)
    return wrapped


@dataclass(frozen=True)
class Pose2D:
    """Pose of a frame expressed in a parent frame (metres, radians)."""

    x: float
    y: float
    yaw: float

    def rotation(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

    def compose(self, other: Pose2D) -> Pose2D:
        """Return self ∘ other: ``other`` is expressed in this pose's frame."""
        dx, dy = self.rotation() @ np.array([other.x, other.y])
        return Pose2D(self.x + dx, self.y + dy, float(wrap_angle(self.yaw + other.yaw)))

    def inverse(self) -> Pose2D:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(-c * self.x - s * self.y, s * self.x - c * self.y, -self.yaw)

    def relative_to(self, reference: Pose2D) -> Pose2D:
        """Express this pose in the frame of ``reference``."""
        return reference.inverse().compose(self)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw])


def world_to_ego(points_world: np.ndarray, ego_pose: Pose2D) -> np.ndarray:
    """Transform [N, 2] world points into the ego frame defined by ``ego_pose``."""
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 2)
    delta = pts - np.array([ego_pose.x, ego_pose.y])
    # Row-vector form of R^T @ delta.
    return delta @ ego_pose.rotation()


def ego_to_world(points_ego: np.ndarray, ego_pose: Pose2D) -> np.ndarray:
    """Transform [N, 2] ego-frame points into the world frame."""
    pts = np.asarray(points_ego, dtype=np.float64).reshape(-1, 2)
    return pts @ ego_pose.rotation().T + np.array([ego_pose.x, ego_pose.y])


# --- CARLA (left-handed) <-> internal (right-handed) -----------------------------------------


def carla_to_world_pose(x: float, y: float, yaw_deg: float) -> Pose2D:
    """Convert a CARLA transform (location m, yaw deg) to an internal world pose."""
    return Pose2D(x, -y, float(wrap_angle(-math.radians(yaw_deg))))


def world_to_carla_pose(pose: Pose2D) -> tuple[float, float, float]:
    """Inverse of :func:`carla_to_world_pose`; returns (x, y, yaw_deg)."""
    return pose.x, -pose.y, -math.degrees(pose.yaw)


def center_to_rear_axle(center_pose: Pose2D, rear_axle_offset: float) -> Pose2D:
    """Shift a bounding-box-centre pose back along its heading to the rear axle.

    ``rear_axle_offset`` is the (positive) distance from box centre to rear axle in metres.
    """
    return center_pose.compose(Pose2D(-rear_axle_offset, 0.0, 0.0))
