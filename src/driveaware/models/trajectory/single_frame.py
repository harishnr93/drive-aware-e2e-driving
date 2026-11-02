"""Model A: single-frame CNN behaviour-cloning policy.

Inputs (non-privileged only): the current RGB image, ego speed and navigation command.
Output: 10 future waypoints [B, 10, 2] in the ego frame (metres), following the label
convention in ``driveaware.data.labels``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from driveaware.models.encoders.compact_cnn import CompactCNNEncoder, normalize_images

NUM_COMMANDS = 4


@dataclass(frozen=True)
class SingleFrameConfig:
    channels: tuple[int, ...] = (32, 64, 96, 128, 192)
    feature_dim: int = 256
    measurement_dim: int = 64
    hidden_dim: int = 256
    num_waypoints: int = 10
    speed_scale: float = 10.0  # m/s, speed input is divided by this
    dropout: float = 0.1


class SingleFramePolicy(nn.Module):
    def __init__(self, config: SingleFrameConfig = SingleFrameConfig()) -> None:
        super().__init__()
        self.cfg = config
        self.encoder = CompactCNNEncoder(config.channels, config.feature_dim)
        self.measure = nn.Sequential(
            nn.Linear(1 + NUM_COMMANDS, config.measurement_dim), nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Linear(config.feature_dim + config.measurement_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_waypoints * 2),
        )
        # Dataset image statistics (set before training, saved with the checkpoint).
        self.register_buffer("image_mean", torch.full((3,), 0.5))
        self.register_buffer("image_std", torch.full((3,), 0.25))

    def features(self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        """image: uint8 [B, H, W, 3]; speed: [B] m/s; command: long [B]."""
        z_img = self.encoder(normalize_images(image, self.image_mean, self.image_std))
        meas = torch.cat(
            [
                (speed / self.cfg.speed_scale).unsqueeze(1),
                nn.functional.one_hot(command, NUM_COMMANDS).float(),
            ],
            dim=1,
        )
        return torch.cat([z_img, self.measure(meas)], dim=1)

    def forward(self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        z = self.features(image, speed, command)
        return self.head(z).view(-1, self.cfg.num_waypoints, 2)
