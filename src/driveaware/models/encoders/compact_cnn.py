"""Compact CNN image encoder (docs/system_architecture.md §7).

Five stride-2 conv blocks reduce a 96x160 image to 3x5, followed by global average pooling and
a linear projection. Input: float tensor [B, 3, H, W] normalised by :func:`normalize_images`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

# Per-channel statistics used to normalise uint8 RGB in [0, 1]. Placeholder values; to be
# replaced by dataset statistics once the Phase 3 dataset exists.
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.25, 0.25, 0.25)


def normalize_images(
    images_uint8: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    """[B, H, W, 3] uint8 -> [B, 3, H, W] float32, normalised per channel (values in [0, 1])."""
    x = images_uint8.permute(0, 3, 1, 2).float().div_(255.0)
    mean = x.new_tensor(IMAGE_MEAN) if mean is None else mean.to(x)
    std = x.new_tensor(IMAGE_STD) if std is None else std.to(x)
    return (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CompactCNNEncoder(nn.Module):
    def __init__(
        self, channels: Sequence[int] = (32, 64, 96, 128, 192), feature_dim: int = 256
    ) -> None:
        super().__init__()
        layers, c_in = [], 3
        for c_out in channels:
            layers.append(_block(c_in, c_out))
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(c_in, feature_dim), nn.ReLU(inplace=True))
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.backbone(x)).flatten(1))
