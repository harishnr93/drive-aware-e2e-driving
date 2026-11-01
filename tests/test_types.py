import numpy as np
import pytest

from driveaware.core.types import (
    IMAGE_SHAPE,
    Command,
    ControlCommand,
    Event,
    EventType,
    IntrospectionOutput,
    Observation,
    TrajectoryPrediction,
)


def test_observation_validation():
    img = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    obs = Observation(0.0, 0, img, 3.0, Command.LEFT)
    assert obs.command == 1
    with pytest.raises(ValueError):
        Observation(0.0, 0, img.astype(np.float32), 3.0, Command.LEFT)
    with pytest.raises(ValueError):
        Observation(0.0, 0, img, float("nan"), Command.LEFT)


def test_observation_has_no_privileged_fields():
    # Leakage guard: the policy input contract must stay minimal.
    fields = set(Observation.__dataclass_fields__)
    assert fields == {"timestamp", "frame_id", "image", "speed", "command"}


def test_control_command_ranges():
    ControlCommand(-1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        ControlCommand(1.1, 0.0, 0.0)
    c = ControlCommand.clipped(2.0, -0.5, 3.0)
    assert (c.steer, c.throttle, c.brake) == (1.0, 0.0, 1.0)


def test_trajectory_prediction_shapes():
    wp = np.zeros((10, 2), np.float32)
    TrajectoryPrediction(0.0, wp, 5.0, np.ones((10, 2), np.float32))
    with pytest.raises(ValueError):
        TrajectoryPrediction(0.0, np.zeros((10, 3)), 5.0)
    with pytest.raises(ValueError):
        TrajectoryPrediction(0.0, wp, 5.0, np.ones((5, 2)))
    with pytest.raises(ValueError):
        TrajectoryPrediction(0.0, wp, -1.0)


def test_introspection_prob_range():
    IntrospectionOutput(0.3, 2.0)
    with pytest.raises(ValueError):
        IntrospectionOutput(1.2, 2.0)


def test_failure_events():
    assert Event(EventType.COLLISION, 1.0).is_failure
    assert not Event(EventType.ROUTE_COMPLETE, 1.0).is_failure
