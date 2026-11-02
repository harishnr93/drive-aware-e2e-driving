import numpy as np
import pytest
import torch

from driveaware.agents.policy_agent import AgentConfig, WaypointPolicyAgent
from driveaware.core.types import IMAGE_SHAPE, Command, Observation
from driveaware.data.labels import motion_situations
from driveaware.models.temporal.gru_policy import TemporalConfig, TemporalPolicy
from driveaware.training.temporal import FeatureSplit


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = TemporalPolicy(TemporalConfig()).eval()
    m.image_mean.fill_(0.45)
    m.image_std.fill_(0.2)
    return m


def test_forward_matches_cached_feature_path(model):
    img = torch.randint(0, 255, (2, 4, 96, 160, 3), dtype=torch.uint8)
    speed = torch.rand(2, 4) * 8
    cmd = torch.tensor([0, 2])
    with torch.no_grad():
        full = model(img, speed, cmd)
        feats = model.encode_frames(img.flatten(0, 1)).view(2, 4, -1)
        cached = model.forward_features(feats, speed, cmd)
    assert full.shape == (2, 10, 2)
    torch.testing.assert_close(full, cached)


def test_speed_history_flag_changes_gru_input():
    plain = TemporalPolicy(TemporalConfig(speed_history=False))
    hist = TemporalPolicy(TemporalConfig(speed_history=True))
    assert hist.gru.input_size == plain.gru.input_size + 1


def test_only_head_trains_when_encoder_frozen(model):
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.train()
    feats = torch.randn(3, 4, 256)
    out = model.forward_features(feats, torch.rand(3, 4), torch.tensor([0, 1, 3]))
    out.sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.gru.weight_ih_l0.grad is not None


def _obs(k, rng):
    return Observation(
        0.1 * k, k, rng.integers(0, 255, IMAGE_SHAPE, dtype=np.uint8), float(k), Command.STRAIGHT
    )


def test_agent_incremental_cache_matches_full_forward(model):
    rng = np.random.default_rng(1)
    obs = [_obs(k, rng) for k in range(7)]
    cfg = AgentConfig(history=4, history_stride=2)
    inc = WaypointPolicyAgent(model, config=cfg)
    assert inc._incremental
    full = WaypointPolicyAgent(model, config=cfg)
    full._incremental = False  # force the image path
    for o in obs:
        a = inc.predict(o).waypoints
        b = full.predict(o).waypoints
        np.testing.assert_allclose(a, b, atol=1e-5)


class _FakeDS:
    """Minimal stand-in for DrivingDataset with two episodes."""

    def __init__(self):
        n = 30
        ep = np.array([0] * 12 + [1] * 18)
        self.frames = {
            "speed": np.arange(n, dtype=np.float32),
            "command": np.zeros(n, dtype=np.int8),
            "waypoints": np.zeros((n, 10, 2), dtype=np.float32),
            "episode_index": ep,
        }
        self.episode_start = np.array([0, 12])
        self.indices = np.arange(n)


def test_feature_gather_is_causal_and_padded():
    ds = _FakeDS()
    feats = np.arange(30, dtype=np.float32)[:, None].repeat(4, axis=1)
    split = FeatureSplit(ds, feats, torch.device("cpu"))
    f, speed, _, _ = split.gather(torch.tensor([13, 25, 2]), history=4, stride=2)
    # frame 13 is the 2nd frame of episode 1 (start 12): history 7,9,11,13 -> padded to 12
    assert speed[0].tolist() == [12.0, 12.0, 12.0, 13.0]
    assert speed[1].tolist() == [19.0, 21.0, 23.0, 25.0]
    assert speed[2].tolist() == [0.0, 0.0, 0.0, 2.0]
    assert f[1, :, 0].tolist() == [19.0, 21.0, 23.0, 25.0]


def test_motion_situations():
    speed = np.array([0.0, 0.0, 6.0, 6.0])
    wp = np.zeros((4, 10, 2))
    wp[1, -1, 0] = 2.0  # starts moving
    wp[2, -1, 0] = 3.0  # 3 m in 2 s from 6 m/s -> stopping
    wp[3, -1, 0] = 12.0  # cruising
    assert motion_situations(speed, wp).tolist() == [
        "holding",
        "start_from_rest",
        "stopping",
        "cruising",
    ]
