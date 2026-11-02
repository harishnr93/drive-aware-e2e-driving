"""Behaviour-cloning training for waypoint policies.

Loss: L1 between predicted and expert future waypoints (ego frame, metres).
Sampling: frames that stay stationary for the whole label window (long red-light waits) are
down-weighted by ``stationary_weight``; frames that are stopped but about to move keep weight 1.
Model selection: lowest validation ADE. The test split is never touched here.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from driveaware.core.hardware import LatencyStats, reset_peak_gpu_memory, snapshot
from driveaware.core.logging import MetricsWriter, get_logger
from driveaware.core.seeding import seed_everything, worker_init_fn
from driveaware.core.types import Command
from driveaware.data.dataset import DrivingDataset
from driveaware.models.trajectory.single_frame import SingleFrameConfig, SingleFramePolicy

log = get_logger("training")


@dataclass
class TrainConfig:
    name: str = "model_a"
    model: str = "single_frame"
    single_frame: SingleFrameConfig = field(default_factory=SingleFrameConfig)
    data_root: str = "data/expert_v1"
    train_split: str = "train"
    val_split: str = "val"
    seed: int = 0
    device: str = "auto"
    epochs: int = 12
    batch_size: int = 64
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 5.0
    workers: int = 4
    stationary_weight: float = 0.3
    stationary_speed: float = 0.1  # m/s
    stationary_motion: float = 0.5  # m, final-waypoint displacement below which it stays put
    brightness_jitter: float = 0.1  # +- fraction
    contrast_jitter: float = 0.1
    deterministic: bool = False
    norm_stat_frames: int = 2000


def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model == "single_frame":
        return SingleFramePolicy(cfg.single_frame)
    raise ValueError(f"unknown model {cfg.model}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def sample_weights(ds: DrivingDataset, cfg: TrainConfig) -> np.ndarray:
    f = ds.frames
    idx = ds.indices
    final = np.linalg.norm(f["waypoints"][idx, -1], axis=1)
    stationary = (f["speed"][idx] < cfg.stationary_speed) & (final < cfg.stationary_motion)
    return np.where(stationary, cfg.stationary_weight, 1.0)


def image_statistics(ds: DrivingDataset, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(ds.indices, size=min(n, len(ds.indices)), replace=False))
    imgs = np.asarray(ds.images[pick], dtype=np.float64) / 255.0
    return imgs.mean(axis=(0, 1, 2)), imgs.std(axis=(0, 1, 2))


def photometric_jitter(img: torch.Tensor, cfg: TrainConfig, gen: torch.Generator) -> torch.Tensor:
    """Per-sample brightness/contrast jitter on uint8 [B, ..., H, W, 3] images (on device)."""
    b = img.shape[0]
    shape = (b,) + (1,) * (img.dim() - 1)
    u = torch.rand((2, b), generator=gen).to(img.device)
    contrast = 1.0 + cfg.contrast_jitter * (2 * u[0] - 1)
    bright = 255.0 * cfg.brightness_jitter * (2 * u[1] - 1)
    x = img.float()
    mean = x.mean(dim=tuple(range(1, x.dim())), keepdim=True)
    x = (x - mean) * contrast.view(shape) + mean + bright.view(shape)
    return x.clamp_(0, 255).to(torch.uint8)


def model_inputs(batch: dict, device: torch.device, history: bool = False):
    image = batch["image"].to(device, non_blocking=True)
    speed = batch["speed"].to(device, non_blocking=True)
    if not history:  # single-frame models use the newest frame only
        image, speed = image[:, -1], speed[:, -1]
    return image, speed, batch["command"].to(device, non_blocking=True)


@torch.no_grad()
def evaluate_open_loop(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ade, fde, l1, cmds, moving = [], [], [], [], []
    for batch in loader:
        pred = model(*model_inputs(batch, device)).float().cpu()
        target = batch["waypoints"]
        err = torch.linalg.norm(pred - target, dim=-1)  # [B, 10]
        ade.append(err.mean(dim=1))
        fde.append(err[:, -1])
        l1.append((pred - target).abs().mean(dim=(1, 2)))
        cmds.append(batch["command"])
        moving.append(torch.linalg.norm(target[:, -1], dim=-1) >= 0.5)
    ade_t, fde_t, l1_t = torch.cat(ade), torch.cat(fde), torch.cat(l1)
    cmd_t, mov_t = torch.cat(cmds), torch.cat(moving)
    out: dict[str, Any] = {
        "ade": float(ade_t.mean()),
        "fde": float(fde_t.mean()),
        "l1": float(l1_t.mean()),
        "n": int(len(ade_t)),
    }
    for c in Command:
        m = cmd_t == int(c)
        if m.any():
            out[f"ade_{c.name.lower()}"] = float(ade_t[m].mean())
            out[f"fde_{c.name.lower()}"] = float(fde_t[m].mean())
    for name, m in (("moving", mov_t), ("stationary", ~mov_t)):
        if m.any():
            out[f"ade_{name}"] = float(ade_t[m].mean())
    return out


def train(cfg: TrainConfig, run_dir: Path) -> dict:
    seed_everything(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    root = Path(cfg.data_root)
    train_ds = DrivingDataset(root / cfg.train_split)
    val_ds = DrivingDataset(root / cfg.val_split)
    model = build_model(cfg).to(device)
    mean, std = image_statistics(train_ds, cfg.norm_stat_frames, cfg.seed)
    model.image_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
    model.image_std.copy_(torch.as_tensor(std, dtype=torch.float32))
    log.info("image mean %s std %s", np.round(mean, 4), np.round(std, 4))

    weights = sample_weights(train_ds, cfg)
    gen = torch.Generator().manual_seed(cfg.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights), num_samples=len(train_ds), replacement=True, generator=gen
    )
    loader_kw = dict(
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, **loader_kw)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total))
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "model %s: %d params; %d train samples (stationary-weighted %.1f%%), %d val",
        cfg.model,
        n_params,
        len(train_ds),
        100 * np.mean(weights < 1),
        len(val_ds),
    )

    metrics = MetricsWriter(run_dir / "metrics.jsonl")
    reset_peak_gpu_memory()
    jitter_gen = torch.Generator().manual_seed(cfg.seed + 1)
    best = {"ade": float("inf")}
    history = []
    t_start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.perf_counter()
        running, n = 0.0, 0
        for batch in train_loader:
            image, speed, command = model_inputs(batch, device)
            image = photometric_jitter(image, cfg, jitter_gen)
            target = batch["waypoints"].to(device, non_blocking=True)
            loss = nn.functional.l1_loss(model(image, speed, command), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            running += loss.item() * len(target)
            n += len(target)
        epoch_s = time.perf_counter() - t0
        val = evaluate_open_loop(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_l1": running / n,
            "lr": sched.get_last_lr()[0],
            "epoch_seconds": epoch_s,
            "samples_per_s": n / epoch_s,
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)
        metrics.write(row)
        log.info(
            "epoch %2d train_l1 %.4f val ADE %.3f FDE %.3f (%.0f s, %.0f samples/s)",
            epoch,
            row["train_l1"],
            val["ade"],
            val["fde"],
            epoch_s,
            row["samples_per_s"],
        )
        torch.save(
            {"config": cfg, "state_dict": model.state_dict(), "epoch": epoch}, run_dir / "last.pt"
        )
        if val["ade"] < best["ade"]:
            best = {**val, "epoch": epoch}
            torch.save(
                {"config": cfg, "state_dict": model.state_dict(), "epoch": epoch},
                run_dir / "best.pt",
            )
    metrics.close()

    resources = snapshot().as_dict()
    lat = measure_latency(model, device)
    result = {
        "params": n_params,
        "best_val": best,
        "final_val": history[-1],
        "train_seconds": time.perf_counter() - t_start,
        "resources": resources,
        "inference_latency_batch1": lat,
        "image_mean": mean.tolist(),
        "image_std": std.tolist(),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


@torch.no_grad()
def measure_latency(model: nn.Module, device: torch.device, iters: int = 200) -> dict:
    model.eval()
    image = torch.randint(0, 255, (1, 96, 160, 3), dtype=torch.uint8, device=device)
    speed = torch.zeros(1, device=device)
    command = torch.zeros(1, dtype=torch.long, device=device)
    for _ in range(20):
        model(image, speed, command)
    lat = LatencyStats()
    for _ in range(iters):
        with lat.measure(cuda_sync=device.type == "cuda"):
            model(image, speed, command)
    return {"device": str(device), **lat.summary()}


def load_policy(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["config"]
