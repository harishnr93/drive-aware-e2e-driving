import math

import numpy as np
import pytest

from driveaware.data.labels import all_waypoints, future_waypoints, label_validity


def _straight(n, v=2.0, yaw=0.3):
    t = np.arange(n) * 0.1
    return np.column_stack([v * t * math.cos(yaw), v * t * math.sin(yaw), np.full(n, yaw)])


def test_future_waypoints_in_ego_frame():
    poses = _straight(30)  # 0.2 m per frame along the heading
    wp = future_waypoints(poses, 3, stride=2, num=10)
    np.testing.assert_allclose(wp[:, 0], 0.4 * np.arange(1, 11), atol=1e-6)
    np.testing.assert_allclose(wp[:, 1], 0.0, atol=1e-6)


def test_left_turn_label_has_positive_y():
    r = 10.0
    th = np.arange(40) * 0.02
    poses = np.column_stack([r * np.sin(th), r - r * np.cos(th), th])  # CCW circle
    assert (future_waypoints(poses, 0, 2, 10)[:, 1] > 0).all()


def test_window_past_end_raises():
    with pytest.raises(IndexError):
        future_waypoints(_straight(20), 0, 2, 10)


def test_label_validity_excludes_noise_and_end():
    n = 60
    perturbed = np.zeros(n, dtype=bool)
    perturbed[30:33] = True  # actions at frames 30..32 were noisy
    valid = label_validity(perturbed, n, 2, 10)
    # Windows are frames t..t+19 of actions; any overlap with 30..32 is invalid.
    expected = np.zeros(n, dtype=bool)
    for t in range(n - 20):
        expected[t] = not perturbed[t : t + 20].any()
    np.testing.assert_array_equal(valid, expected)
    assert valid[10] and not valid[11] and not valid[32] and valid[33]
    assert not valid[n - 20:].any()


def test_all_waypoints_nan_where_invalid():
    poses = _straight(40)
    valid = label_validity(np.zeros(40, bool), 40, 2, 10)
    wp = all_waypoints(poses, valid, 2, 10)
    assert np.isnan(wp[~valid]).all() and not np.isnan(wp[valid]).any()
