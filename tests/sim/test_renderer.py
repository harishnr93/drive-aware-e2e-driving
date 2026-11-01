import math

import numpy as np
import pytest

from driveaware.core.frames import Pose2D
from driveaware.sim.lightweight.agents import AgentKind, Box
from driveaware.sim.lightweight.renderer import CameraConfig, CameraRenderer, _clip_near
from driveaware.sim.lightweight.roadmap import generate_map


@pytest.fixture(scope="module")
def renderer():
    return CameraRenderer(generate_map(0), CameraConfig())


def test_ground_rays_match_pinhole_geometry(renderer):
    cfg = renderer.cfg
    # Centre column of the image: ground distance follows from pitch and the row's angle.
    v = np.arange(renderer.h) + 0.5
    angle = math.radians(cfg.pitch_deg) + np.arctan((v - renderer.cy) / renderer.fy)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_x = cfg.x_offset + cfg.mount_height / np.tan(angle)
    mask = renderer.ground_mask[:, renderer.w // 2]
    full = np.full((renderer.h, renderer.w, 2), np.nan)
    full[renderer.ground_mask] = renderer.ground_ego
    col = full[:, renderer.w // 2]
    # Pixel centre is half a pixel right of the optical axis: y offset small, x as expected.
    np.testing.assert_allclose(col[mask, 0], expected_x[mask], rtol=2e-3)
    assert np.all(np.abs(col[mask, 1]) < 0.05 * col[mask, 0])


def test_projection_inverts_ground_rays(renderer):
    idx = np.linspace(0, len(renderer.ground_ego) - 1, 200).astype(int)
    pts = np.column_stack([renderer.ground_ego[idx], np.zeros(len(idx))])
    cam = (pts - renderer.cam_pos) @ renderer.R.T
    uv = np.column_stack(
        [
            renderer.fx * cam[:, 0] / cam[:, 2] + renderer.cx,
            renderer.fy * cam[:, 1] / cam[:, 2] + renderer.cy,
        ]
    )
    rows, cols = np.nonzero(renderer.ground_mask)
    np.testing.assert_allclose(uv, np.column_stack([cols[idx] + 0.5, rows[idx] + 0.5]), atol=1e-6)


def test_left_is_left(renderer):
    # A point to the ego's left must appear in the left half of the image.
    cam = (np.array([10.0, 3.0, 0.0]) - renderer.cam_pos) @ renderer.R.T
    u = renderer.fx * cam[0] / cam[2] + renderer.cx
    assert u < renderer.w / 2


def test_render_box_ahead(renderer):
    m = renderer.map
    lane = m.road_lanes[0]
    x, y, yaw = lane.path.interpolate(2.0)
    pose = Pose2D(float(x), float(y), float(yaw))
    empty = renderer.render(pose, [])
    ahead = Box(
        AgentKind.VEHICLE,
        x + 12 * math.cos(yaw),
        y + 12 * math.sin(yaw),
        yaw,
        0.0,
        4.6,
        1.9,
        1.5,
        1,
    )
    behind = Box(
        AgentKind.VEHICLE,
        x - 12 * math.cos(yaw),
        y - 12 * math.sin(yaw),
        yaw,
        0.0,
        4.6,
        1.9,
        1.5,
        2,
    )
    with_ahead = renderer.render(pose, [ahead])
    assert empty.shape == (96, 160, 3) and empty.dtype == np.uint8
    diff = np.abs(with_ahead.astype(int) - empty.astype(int)).sum(axis=2) > 0
    cols = np.nonzero(diff.any(axis=0))[0]
    assert diff.sum() > 50 and abs(cols.mean() - 80) < 10  # centred
    assert np.array_equal(renderer.render(pose, [behind]), empty)  # culled, not drawn


def test_near_plane_clipping():
    quad = np.array([[-1, -1, -1.0], [1, -1, -1.0], [1, 1, 3.0], [-1, 1, 3.0]])
    clipped = _clip_near(quad, 0.2)
    assert clipped is not None and (clipped[:, 2] >= 0.2 - 1e-9).all()
    assert _clip_near(quad - [0, 0, 10], 0.2) is None
