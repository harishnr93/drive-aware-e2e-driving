"""Trajectory uncertainty: deep ensemble of probabilistic heads, and MC dropout.

Variances are per waypoint coordinate (m^2):
- aleatoric: mean over members of the Laplace variance 2 b^2 (data noise the model expects)
- epistemic: variance of the members' means (disagreement; model uncertainty)
- total:     aleatoric + epistemic (moment-matched mixture)

All members share the frozen model-A encoder, so encoder (feature) uncertainty is not captured;
the ensemble spans GRU + head only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from driveaware.models.temporal.gru_policy import TemporalPolicy


@dataclass
class TrajectoryUncertainty:
    mean: np.ndarray  # [B, N, 2]
    epistemic_var: np.ndarray  # [B, N, 2]
    aleatoric_var: np.ndarray  # [B, N, 2]
    member_means: np.ndarray | None = None  # [M, B, N, 2]

    @property
    def total_var(self) -> np.ndarray:
        return self.epistemic_var + self.aleatoric_var

    @staticmethod
    def summary(var: np.ndarray) -> np.ndarray:
        """Per-sample scalar: mean over waypoints of the positional std sqrt(var_x + var_y) (m)."""
        return np.sqrt(var.sum(axis=-1)).mean(axis=-1)


class EnsemblePredictor:
    def __init__(self, members: list[TemporalPolicy], device: str | torch.device = "cpu") -> None:
        if not members or not all(m.cfg.probabilistic for m in members):
            raise ValueError("ensemble members must be probabilistic TemporalPolicy models")
        self.members = [m.to(device).eval() for m in members]
        self.device = torch.device(device)

    @torch.no_grad()
    def predict(
        self, feats: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> TrajectoryUncertainty:
        means, scales = [], []
        for m in self.members:
            mu, b = m.forward_dist(
                feats.to(self.device), speed.to(self.device), command.to(self.device)
            )
            means.append(mu.float().cpu().numpy())
            scales.append(b.float().cpu().numpy())
        mm = np.stack(means)
        sc = np.stack(scales)
        return TrajectoryUncertainty(
            mean=mm.mean(axis=0),
            epistemic_var=mm.var(axis=0),
            aleatoric_var=(2.0 * sc**2).mean(axis=0),
            member_means=mm,
        )


@torch.no_grad()
def mc_dropout(
    model: TemporalPolicy,
    feats: torch.Tensor,
    speed: torch.Tensor,
    command: torch.Tensor,
    passes: int = 20,
    seed: int = 0,
) -> TrajectoryUncertainty:
    """Monte-Carlo dropout: dropout layers active, everything else in eval mode."""
    model.eval()
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for d in drops:
        d.train()
    torch.manual_seed(seed)
    outs = np.stack(
        [model.forward_features(feats, speed, command).float().cpu().numpy() for _ in range(passes)]
    )
    for d in drops:
        d.eval()
    return TrajectoryUncertainty(
        mean=outs.mean(axis=0),
        epistemic_var=outs.var(axis=0),
        aleatoric_var=np.zeros_like(outs[0]),
        member_means=outs,
    )
