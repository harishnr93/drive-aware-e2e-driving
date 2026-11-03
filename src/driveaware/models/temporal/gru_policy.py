"""Model B: CNN features over an observation history, aggregated by a GRU.

Inputs (non-privileged only): the last H images (oldest first, same padding rule as the
dataset), the current ego speed (and optionally the speed history) and the navigation command.
Output: 10 future waypoints [B, 10, 2] in the ego frame of the newest frame.

The image encoder is the compact CNN. In the primary configuration it is initialised from a
trained model A and frozen, so per-frame features can be cached for training and reused frame
by frame at inference (only the newest frame is encoded each step).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from driveaware.models.encoders.compact_cnn import CompactCNNEncoder, normalize_images

NUM_COMMANDS = 4


@dataclass(frozen=True)
class TemporalConfig:
    channels: tuple[int, ...] = (32, 64, 96, 128, 192)
    feature_dim: int = 256
    gru_hidden: int = 256
    measurement_dim: int = 64
    hidden_dim: int = 256
    num_waypoints: int = 10
    speed_scale: float = 10.0
    speed_history: bool = False  # feed each frame's speed to the GRU (copycat-risk ablation)
    dropout: float = 0.1
    freeze_encoder: bool = True
    probabilistic: bool = False  # also predict a Laplace scale per waypoint coordinate
    min_scale: float = 0.01  # m, lower bound on the Laplace scale b


class TemporalPolicy(nn.Module):
    uses_history = True

    def __init__(self, config: TemporalConfig = TemporalConfig()) -> None:
        super().__init__()
        self.cfg = config
        self.encoder = CompactCNNEncoder(config.channels, config.feature_dim)
        self.register_buffer("image_mean", torch.full((3,), 0.5))
        self.register_buffer("image_std", torch.full((3,), 0.25))
        gru_in = config.feature_dim + (1 if config.speed_history else 0)
        self.gru = nn.GRU(gru_in, config.gru_hidden, batch_first=True)
        self.measure = nn.Sequential(
            nn.Linear(1 + NUM_COMMANDS, config.measurement_dim), nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Linear(config.gru_hidden + config.measurement_dim, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.hidden_dim, config.num_waypoints * 2 * (2 if config.probabilistic else 1)
            ),
        )

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 [N, H, W, 3] -> features [N, feature_dim]."""
        return self.encoder(normalize_images(images, self.image_mean, self.image_std))

    def forward_features(
        self, feats: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        """feats [B, T, F] (oldest first); speed [B, T] m/s; command [B]."""
        x = feats
        if self.cfg.speed_history:
            x = torch.cat([x, (speed / self.cfg.speed_scale).unsqueeze(-1)], dim=-1)
        _, h = self.gru(x)
        # Read-only side output for introspection (never fed back into the policy).
        self.last_hidden = h[-1].detach()
        meas = torch.cat(
            [
                (speed[:, -1] / self.cfg.speed_scale).unsqueeze(1),
                nn.functional.one_hot(command, NUM_COMMANDS).float(),
            ],
            dim=1,
        )
        z = torch.cat([h[-1], self.measure(meas)], dim=1)
        out = self.head(z)
        n = self.cfg.num_waypoints
        if self.cfg.probabilistic:
            # Mean is returned by forward(); the scale is kept as a read-only side output.
            self.last_scale = self.cfg.min_scale + nn.functional.softplus(out[:, 2 * n :]).view(
                -1, n, 2
            )
            return out[:, : 2 * n].view(-1, n, 2)
        return out.view(-1, n, 2)

    def forward_dist(
        self, feats: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(mean, Laplace scale b), both [B, N, 2]. Requires ``probabilistic=True``."""
        if not self.cfg.probabilistic:
            raise ValueError("model was not trained with a probabilistic head")
        mean = self.forward_features(feats, speed, command)
        return mean, self.last_scale

    def forward(self, image: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        """image uint8 [B, T, H, W, 3]; speed [B, T]; command [B]."""
        b, t = image.shape[:2]
        feats = self.encode_frames(image.flatten(0, 1)).view(b, t, -1)
        return self.forward_features(feats, speed, command)
