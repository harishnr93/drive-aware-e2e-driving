import math

import numpy as np
import pytest

from driveaware.core.frames import (
    Pose2D,
    carla_to_world_pose,
    center_to_rear_axle,
    ego_to_world,
    world_to_carla_pose,
    world_to_ego,
    wrap_angle,
)


@pytest.mark.parametrize(
    "angle, expected",
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (3 * math.pi, math.pi),
        (2 * math.pi, 0.0),
        (-3 * math.pi / 2, math.pi / 2),
    ],
)
def test_wrap_angle_range(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected)


def test_wrap_angle_vectorised():
    a = np.linspace(-20, 20, 1001)
    w = wrap_angle(a)
    assert np.all(w > -np.pi) and np.all(w <= np.pi)
    np.testing.assert_allclose(np.sin(w), np.sin(a), atol=1e-12)
    np.testing.assert_allclose(np.cos(w), np.cos(a), atol=1e-12)


def test_ego_axes_convention():
    # Ego at (10, 5) facing north (+y world). A point 1 m north is straight ahead (+x ego);
    # a point 1 m west is to the LEFT (+y ego).
    ego = Pose2D(10.0, 5.0, math.pi / 2)
    pts = world_to_ego(np.array([[10.0, 6.0], [9.0, 5.0]]), ego)
    np.testing.assert_allclose(pts, [[1.0, 0.0], [0.0, 1.0]], atol=1e-12)


def test_world_ego_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        pose = Pose2D(*rng.uniform(-100, 100, 2), rng.uniform(-np.pi, np.pi))
        pts = rng.uniform(-50, 50, (20, 2))
        np.testing.assert_allclose(ego_to_world(world_to_ego(pts, pose), pose), pts, atol=1e-9)


def test_compose_inverse_identity():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = Pose2D(*rng.uniform(-50, 50, 2), rng.uniform(-np.pi, np.pi))
        ident = p.compose(p.inverse())
        np.testing.assert_allclose(ident.as_array(), [0, 0, 0], atol=1e-9)


def test_relative_to_matches_point_transform():
    a = Pose2D(3.0, -2.0, 0.7)
    b = Pose2D(-1.0, 4.0, -2.1)
    rel = b.relative_to(a)
    np.testing.assert_allclose(world_to_ego(np.array([[b.x, b.y]]), a)[0], [rel.x, rel.y])
    assert rel.yaw == pytest.approx(wrap_angle(b.yaw - a.yaw))


def test_carla_conversion():
    # CARLA yaw 90 deg points along +y_carla, which is to the RIGHT -> world yaw -pi/2.
    pose = carla_to_world_pose(5.0, 2.0, 90.0)
    assert (pose.x, pose.y) == (5.0, -2.0)
    assert pose.yaw == pytest.approx(-math.pi / 2)
    x, y, yaw = world_to_carla_pose(pose)
    assert (x, y) == (5.0, 2.0) and yaw == pytest.approx(90.0)


def test_carla_handedness_preserves_geometry():
    # A point 3 m to the right of a CARLA vehicle facing +x_carla must end up at ego y = -3.
    vehicle = carla_to_world_pose(0.0, 0.0, 0.0)
    point = carla_to_world_pose(0.0, 3.0, 0.0)
    np.testing.assert_allclose(world_to_ego(np.array([[point.x, point.y]]), vehicle), [[0, -3]])


def test_center_to_rear_axle():
    center = Pose2D(0.0, 0.0, math.pi / 2)  # facing north
    rear = center_to_rear_axle(center, 1.4)
    np.testing.assert_allclose([rear.x, rear.y, rear.yaw], [0.0, -1.4, math.pi / 2], atol=1e-12)
