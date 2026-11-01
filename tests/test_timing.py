import numpy as np
import pytest

from driveaware.core.timing import TimingConfig


def test_defaults():
    t = TimingConfig()
    assert t.substeps == 2
    assert t.waypoint_stride == 2
    assert t.horizon_s == pytest.approx(2.0)
    assert t.failure_horizon_steps == 20
    np.testing.assert_allclose(t.waypoint_times(), np.arange(1, 11) * 0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control_period": 0.07},  # not a multiple of dt_sim
        {"waypoint_dt": 0.15},  # not a multiple of control_period
        {"failure_horizon_s": 0.05},  # shorter than one control step
        {"dt_sim": 0.0},
        {"num_waypoints": 0},
    ],
)
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        TimingConfig(**kwargs)
