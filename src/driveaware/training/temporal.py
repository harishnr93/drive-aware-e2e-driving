"""Training of temporal policies on cached, frozen per-frame CNN features.

The encoder comes from a trained single-frame model (model A) and is frozen. Its features are
computed once per split (eval mode, no augmentation) and cached next to that checkpoint. All
features, labels and history indices then live on the GPU, so an epoch over the training set
takes seconds. The saved checkpoint contains the full model (encoder included), so closed-loop
agents run from raw images exactly as model A does.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from driveaware.core.hardware import LatencyStats, reset_peak_gpu_memory, snapshot
from driveaware.core.logging import MetricsWriter, get_logger
from driveaware.core.seeding import seed_everything
from driveaware.data.dataset import DrivingDataset
from driveaware.models.temporal.gru_policy import TemporalPolicy
from driveaware.training.bc import (
    TrainConfig,
    load_policy,
    resolve_device,
    sample_weights,
    trajectory_metrics,
)

log = get_logger("training.temporal")


@torch.no_grad()
def cache_features(
    encoder_checkpoint: str | Path, split_dir: str | Path, device: torch.device, batch: int = 512
) -> np.ndarray:
    """Frozen-encoder features for every frame of a split; cached as float32 .npy."""
    ck = Path(encoder_checkpoint)
    out = ck.parent / "features" / f"{Path(split_dir).name}.npy"
    if out.exists():
        return np.load(out)
    model, _ = load_policy(ck, device)
    images = np.load(Path(split_dir) / "images.npy", mmap_mode="r")
    feats = np.empty((len(images), model.encoder.feature_dim), dtype=np.float32)
    from driveaware.models.encoders.compact_cnn import normalize_images

    for s in range(0, len(images), batch):
        x = torch.from_numpy(np.ascontiguousarray(images[s : s + batch])).to(device)
        z = model.encoder(normalize_images(x, model.image_mean, model.image_std))
        feats[s : s + batch] = z.float().cpu().numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feats)
    return feats


class FeatureSplit:
    """GPU-resident features + labels for one split, with causal history gathering."""

    def __init__(self, ds: DrivingDataset, feats: np.ndarray, device: torch.device) -> None:
        f = ds.frames
        self.feats = torch.from_numpy(feats).to(device)
        self.speed = torch.from_numpy(f["speed"].astype(np.float32)).to(device)
        self.command = torch.from_numpy(f["command"].astype(np.int64)).to(device)
        self.waypoints = torch.from_numpy(np.nan_to_num(f["waypoints"])).to(device)
        ep_start = ds.episode_start[f["episode_index"]]
        self.ep_start = torch.from_numpy(ep_start).to(device)
        self.indices = torch.from_numpy(ds.indices).to(device)

    def gather(self, g: torch.Tensor, history: int, stride: int):
        offsets = stride * torch.arange(history - 1, -1, -1, device=g.device)
        idx = g[:, None] - offsets[None]  # [B, H], oldest first
        idx = torch.maximum(idx, self.ep_start[g][:, None])  # pad with the episode's first frame
        return self.feats[idx], self.speed[idx], self.command[g], self.waypoints[g]


@torch.no_grad()
def evaluate_features(
    model: TemporalPolicy, split: FeatureSplit, history: int, stride: int, batch: int = 2048
) -> dict:
    model.eval()
    preds, targets, cmds, speeds = [], [], [], []
    for s in range(0, len(split.indices), batch):
        g = split.indices[s : s + batch]
        feats, speed, cmd, wp = split.gather(g, history, stride)
        preds.append(model.forward_features(feats, speed, cmd).float().cpu())
        targets.append(wp.cpu())
        cmds.append(cmd.cpu())
        speeds.append(speed[:, -1].cpu())
    return trajectory_metrics(
        torch.cat(preds), torch.cat(targets), torch.cat(cmds), torch.cat(speeds)
    )


def build_from_encoder(cfg: TrainConfig, device: torch.device) -> TemporalPolicy:
    model = TemporalPolicy(cfg.temporal).to(device)
    base, _ = load_policy(cfg.encoder_checkpoint, device)
    model.encoder.load_state_dict(base.encoder.state_dict())
    model.image_mean.copy_(base.image_mean)
    model.image_std.copy_(base.image_std)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    return model


def train_temporal(cfg: TrainConfig, run_dir: Path) -> dict:
    if not cfg.temporal.freeze_encoder:
        raise ValueError(
            "train_temporal trains on cached frozen features; use train() for "
            "end-to-end temporal training"
        )
    epochs, batch_size, lr = cfg.epochs, cfg.batch_size, cfg.lr
    seed_everything(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    root = Path(cfg.data_root)
    t_cache = time.perf_counter()
    splits = {}
    for name in (cfg.train_split, cfg.val_split):
        ds = DrivingDataset(root / name)
        feats = cache_features(cfg.encoder_checkpoint, root / name, device)
        splits[name] = (ds, FeatureSplit(ds, feats, device))
    cache_s = time.perf_counter() - t_cache
    train_ds, train = splits[cfg.train_split]
    _, val = splits[cfg.val_split]
    model = build_from_encoder(cfg, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    weights = torch.as_tensor(sample_weights(train_ds, cfg), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train.indices) // batch_size
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: 0.5 * (1 + math.cos(math.pi * min(k, total) / total))
    )
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    metrics = MetricsWriter(run_dir / "metrics.jsonl")
    reset_peak_gpu_memory()
    best: dict = {"ade": float("inf")}
    log.info(
        "temporal H=%d stride=%d speed_history=%s: %d trainable / %d params; "
        "features cached in %.0f s",
        cfg.history,
        cfg.history_stride,
        cfg.temporal.speed_history,
        n_train,
        n_total,
        cache_s,
    )
    t_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        model.encoder.eval()
        t0 = time.perf_counter()
        pick = torch.multinomial(
            weights, steps_per_epoch * batch_size, replacement=True, generator=gen
        )
        running = 0.0
        for k in range(steps_per_epoch):
            g = train.indices[pick[k * batch_size : (k + 1) * batch_size]]
            feats, speed, cmd, wp = train.gather(g, cfg.history, cfg.history_stride)
            loss = nn.functional.l1_loss(model.forward_features(feats, speed, cmd), wp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            sched.step()
            running += loss.item()
        val_m = evaluate_features(model, val, cfg.history, cfg.history_stride)
        row = {
            "epoch": epoch,
            "train_l1": running / steps_per_epoch,
            "epoch_seconds": time.perf_counter() - t0,
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        metrics.write(row)
        if val_m["ade"] < best["ade"]:
            best = {**val_m, "epoch": epoch}
            torch.save(
                {"config": cfg, "state_dict": model.state_dict(), "epoch": epoch},
                run_dir / "best.pt",
            )
        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info(
                "epoch %2d train_l1 %.4f val ADE %.3f FDE %.3f (%.1f s)",
                epoch,
                row["train_l1"],
                val_m["ade"],
                val_m["fde"],
                row["epoch_seconds"],
            )
    metrics.close()
    result = {
        "params_total": n_total,
        "params_trainable": n_train,
        "best_val": best,
        "train_seconds": time.perf_counter() - t_start,
        "feature_cache_seconds": cache_s,
        "resources": snapshot().as_dict(),
        "inference_latency": measure_temporal_latency(model, device, cfg.history),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


@torch.no_grad()
def measure_temporal_latency(
    model: TemporalPolicy, device: torch.device, history: int, iters: int = 200
) -> dict:
    """Full H-frame forward vs incremental step (encode newest frame + GRU over cached)."""
    model.eval()
    img = torch.randint(0, 255, (1, history, 96, 160, 3), dtype=torch.uint8, device=device)
    speed = torch.zeros(1, history, device=device)
    cmd = torch.zeros(1, dtype=torch.long, device=device)
    feats = model.encode_frames(img[0]).unsqueeze(0)
    out = {}
    for name, fn in (
        ("full_history_forward", lambda: model(img, speed, cmd)),
        (
            "incremental_step",
            lambda: model.forward_features(
                torch.cat([feats[:, 1:], model.encode_frames(img[0, -1:]).unsqueeze(0)], 1),
                speed,
                cmd,
            ),
        ),
    ):
        for _ in range(20):
            fn()
        lat = LatencyStats()
        for _ in range(iters):
            with lat.measure(cuda_sync=device.type == "cuda"):
                fn()
        out[name] = {"device": str(device), **lat.summary()}
    return out
