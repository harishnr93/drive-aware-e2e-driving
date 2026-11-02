"""Introspection head, logistic baselines and post-hoc calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class HeadConfig:
    hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 512
    pos_weight: str = "balanced"  # "balanced" (neg/pos) or "none"
    seed: int = 0


class IntrospectionHead(nn.Module):
    def __init__(self, in_dim: int, cfg: HeadConfig = HeadConfig()) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in cfg.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sigma", torch.ones(in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mu) / self.sigma).squeeze(-1)


def _auroc_fast(y: np.ndarray, s: np.ndarray) -> float:
    from driveaware.introspection.metrics import auroc

    return auroc(y, s)


def train_head(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: HeadConfig = HeadConfig(),
    device: str = "cpu",
) -> tuple[IntrospectionHead, list[dict]]:
    """Weighted-BCE training; the epoch with the best validation PR-AUC is kept."""
    from driveaware.introspection.metrics import average_precision

    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    model = IntrospectionHead(x_tr.shape[1], cfg).to(dev)
    xt = torch.as_tensor(x_tr, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=dev)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=dev)
    model.mu.copy_(xt.mean(0))
    model.sigma.copy_(xt.std(0).clamp_min(1e-6))
    pos = float(y_tr.mean())
    pw = (1 - pos) / max(pos, 1e-6) if cfg.pos_weight == "balanced" else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    best, best_state, history = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen, device=dev)
        total = 0.0
        for k in range(0, len(xt), cfg.batch_size):
            idx = perm[k : k + cfg.batch_size]
            loss = loss_fn(model(xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            sv = model(xv).cpu().numpy()
        ap = average_precision(y_val, sv)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(xt),
                "val_pr_auc": ap,
                "val_auroc": _auroc_fast(y_val, sv),
            }
        )
        if ap > best:
            best = ap
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), history


@torch.no_grad()
def head_logits(model: IntrospectionHead, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    out = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return out.cpu().numpy().astype(np.float64)


@dataclass
class LogisticModel:
    """Standardised logistic regression (L2), fitted with L-BFGS. Used for heuristics/oracle."""

    w: np.ndarray
    b: float
    mu: np.ndarray
    sigma: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> LogisticModel:
        x = np.asarray(x, dtype=np.float64)
        mu, sigma = x.mean(0), x.std(0) + 1e-9
        xt = torch.as_tensor((x - mu) / sigma)
        yt = torch.as_tensor(y, dtype=torch.float64)
        w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt)
            loss = loss + l2 * (w**2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return cls(w.detach().numpy(), float(b.item()), mu, sigma)

    def logits(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mu) / self.sigma) @ self.w + self.b


@dataclass
class PlattCalibrator:
    """p = sigmoid(a * logit + c), fitted on a held-out calibration split.

    A temperature alone (c = 0) cannot undo the prior shift introduced by class-weighted
    training, so an intercept is fitted as well.
    """

    a: float = 1.0
    c: float = 0.0

    @classmethod
    def fit(cls, logits: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        m = LogisticModel.fit(np.asarray(logits)[:, None], y, l2=0.0)
        a = float(m.w[0] / m.sigma[0])
        return cls(a, float(m.b - a * m.mu[0]))

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        z = np.clip(self.a * np.asarray(logits) + self.c, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))
