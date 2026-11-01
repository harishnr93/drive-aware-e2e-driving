import math

import numpy as np
import pytest

from driveaware.sim.lightweight.geometry import (
    Polyline,
    bezier_quadratic,
    box_corners,
    boxes_overlap,
    hermite,
)


def test_polyline_interpolate_and_length():
    pl = Polyline(np.array([[0, 0], [10, 0], [10, 10]]))
    assert pl.length == pytest.approx(20.0)
    np.testing.assert_allclose(pl.interpolate(5.0), [5, 0, 0])
    np.testing.assert_allclose(pl.interpolate(15.0), [10, 5, math.pi / 2])
    np.testing.assert_allclose(pl.interpolate(100.0)[:2], [10, 10])  # clamped
    assert pl.interpolate(np.array([1.0, 2.0])).shape == (2, 3)


def test_polyline_drops_duplicates():
    pl = Polyline(np.array([[0, 0], [0, 0], [1, 0]]))
    assert len(pl.points) == 2


def test_project_signed_lateral():
    pl = Polyline(np.array([[0, 0], [10, 0]]))
    s, lat, d = pl.project(np.array([3.0, 2.0]))
    assert (s, lat, d) == pytest.approx((3.0, 2.0, 2.0))  # left is positive
    s, lat, _ = pl.project(np.array([4.0, -1.5]))
    assert (s, lat) == pytest.approx((4.0, -1.5))


def test_project_window_disambiguates_overlaps():
    # A loop that passes the same place twice: the window selects the intended pass.
    pl = Polyline(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0], [10, 0]]))
    s_first, *_ = pl.project(np.array([5.0, 0.0]), 0.0, 20.0)
    s_second, *_ = pl.project(np.array([5.0, 0.0]), 35.0, 50.0)
    assert s_first == pytest.approx(5.0) and s_second == pytest.approx(45.0)


def test_offset_left_and_right():
    pl = Polyline(np.array([[0, 0], [10, 0], [20, 0]]))
    np.testing.assert_allclose(pl.offset(1.5).points[:, 1], 1.5)
    np.testing.assert_allclose(pl.offset(-1.5).points[:, 1], -1.5)


def test_curvature_of_circle():
    r = 20.0
    th = np.linspace(0, math.pi, 400)
    pl = Polyline(np.stack([r * np.cos(th), r * np.sin(th)], 1))
    np.testing.assert_allclose(pl.curvature()[5:-5], 1 / r, rtol=1e-3)


def test_box_corners_and_overlap():
    a = box_corners(0, 0, 0, 4, 2)
    np.testing.assert_allclose(a.min(0), [-2, -1]) and np.testing.assert_allclose(a.max(0), [2, 1])
    assert boxes_overlap(a, box_corners(3.9, 0, 0, 4, 2))
    assert not boxes_overlap(a, box_corners(4.1, 0, 0, 4, 2))
    # Rotated case where axis-aligned bounds overlap but the boxes do not.
    assert not boxes_overlap(
        box_corners(0, 0, math.pi / 4, 4, 0.5), box_corners(1.6, -1.6, 0, 1, 1)
    )
    assert boxes_overlap(box_corners(0, 0, math.pi / 4, 4, 0.5), box_corners(0.5, 0.5, 0, 1, 1))


def test_curve_helpers_hit_endpoints():
    p0, p2 = np.array([0.0, 0.0]), np.array([10.0, 0.0])
    b = bezier_quadratic(p0, np.array([5.0, 5.0]), p2, 11)
    np.testing.assert_allclose(b[[0, -1]], [p0, p2])
    h = hermite(p0, 0.0, np.array([10.0, 10.0]), math.pi / 2, 50)
    np.testing.assert_allclose(h[[0, -1]], [[0, 0], [10, 10]])
    d0, d1 = h[1] - h[0], h[-1] - h[-2]
    assert math.atan2(d0[1], d0[0]) == pytest.approx(0.0, abs=0.05)
    assert math.atan2(d1[1], d1[0]) == pytest.approx(math.pi / 2, abs=0.05)
