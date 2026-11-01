"""Phase 1 smoke test: train the compact encoder + waypoint head on a synthetic task.

The synthetic task has a learnable signal: a bright vertical bar is drawn at a random column,
and the target waypoints curve laterally in proportion to the bar offset. The loss must fall
well below the predict-the-mean baseline, which shows that data, model, optimiser and device
work end-to-end. Reports peak VRAM, host RAM, throughput and batch-1 inference latency.

Usage:  python scripts/smoke_train.py [--config configs/training/smoke.yaml] [--set key=value ...]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import yaml
from torch import nn

from driveaware.core.config import load_config
from driveaware.core.hardware import LatencyStats, reset_peak_gpu_memory, snapshot
from driveaware.core.logging import MetricsWriter, setup_logging
from driveaware.core.run import create_run_dir
from driveaware.core.seeding import make_rng, seed_everything
from driveaware.models.encoders import CompactCNNEncoder
from driveaware.models.encoders.compact_cnn import normalize_images


@dataclass
class SmokeConfig:
    seed: int = 0
    device: str = "auto"
    batch_size: int = 4
    steps: int = 300
    lr: float = 1e-3
    num_samples: int = 512
    image_height: int = 96
    image_width: int = 160
    num_waypoints: int = 10
    latency_iters: int = 200
    deterministic: bool = False


class SmokeModel(nn.Module):
    def __init__(self, num_waypoints: int) -> None:
        super().__init__()
        self.encoder = CompactCNNEncoder()
        self.measure = nn.Sequential(nn.Linear(1 + 4, 64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_waypoints * 2),
        )
        self.num_waypoints = num_waypoints

    def forward(self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        meas = torch.cat([speed[:, None], nn.functional.one_hot(command, 4).float()], dim=1)
        z = torch.cat([self.encoder(images), self.measure(meas)], dim=1)
        return self.head(z).view(-1, self.num_waypoints, 2)


def make_synthetic(cfg: SmokeConfig) -> dict[str, np.ndarray]:
    rng = make_rng(cfg.seed)
    n, h, w = cfg.num_samples, cfg.image_height, cfg.image_width
    images = rng.integers(0, 40, size=(n, h, w, 3), dtype=np.uint8)
    offset = rng.uniform(-1.0, 1.0, size=n)  # normalised bar position
    cols = ((offset + 1) / 2 * (w - 8)).astype(int)
    for i, c in enumerate(cols):
        images[i, :, c : c + 8] = 230
    speed = rng.uniform(0.0, 10.0, size=n).astype(np.float32)
    command = rng.integers(0, 4, size=n)
    t = np.linspace(0.2, 2.0, cfg.num_waypoints)
    x = speed[:, None] * t[None]  # forward progress from speed
    y = 1.5 * offset[:, None] * t[None] ** 2  # lateral curvature from image
    waypoints = np.stack([x, y], axis=-1).astype(np.float32)
    return {"images": images, "speed": speed, "command": command, "waypoints": waypoints}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/smoke.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="overrides, e.g. batch_size=16")
    args = parser.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(SmokeConfig, args.config, overrides)

    run_dir = create_run_dir(f"smoke_bs{cfg.batch_size}", cfg)
    log = setup_logging(run_dir)
    seed_everything(cfg.seed, cfg.deterministic)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    )
    log.info("run_dir=%s device=%s", run_dir, device)

    data = make_synthetic(cfg)
    images = torch.from_numpy(data["images"])  # kept uint8 on host; normalised per batch
    speed = torch.from_numpy(data["speed"])
    command = torch.from_numpy(data["command"]).long()
    target = torch.from_numpy(data["waypoints"])
    baseline_l1 = float((target - target.mean(0, keepdim=True)).abs().mean())

    model = SmokeModel(cfg.num_waypoints).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    reset_peak_gpu_memory()

    metrics = MetricsWriter(run_dir / "metrics.jsonl")
    model.train()
    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        idx = torch.randint(0, cfg.num_samples, (cfg.batch_size,), generator=gen)
        x = normalize_images(images[idx].to(device, non_blocking=True))
        pred = model(x, speed[idx].to(device) / 10.0, command[idx].to(device))
        loss = nn.functional.l1_loss(pred, target[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0 or step == cfg.steps - 1:
            metrics.write({"step": step, "train_l1": losses[-1]})
            log.info("step %4d  l1 %.4f", step, losses[-1])
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    train_snap = snapshot()

    # Held-out evaluation on fresh synthetic samples (different seed).
    eval_cfg = SmokeConfig(**{**cfg.__dict__, "seed": cfg.seed + 1, "num_samples": 256})
    ev = make_synthetic(eval_cfg)
    model.eval()
    with torch.no_grad():
        pred = model(
            normalize_images(torch.from_numpy(ev["images"]).to(device)),
            torch.from_numpy(ev["speed"]).to(device) / 10.0,
            torch.from_numpy(ev["command"]).long().to(device),
        )
        eval_l1 = float((pred.cpu() - torch.from_numpy(ev["waypoints"])).abs().mean())

        lat = LatencyStats()
        x1 = normalize_images(images[:1].to(device))
        s1, c1 = speed[:1].to(device) / 10.0, command[:1].to(device)
        for _ in range(20):  # warm-up
            model(x1, s1, c1)
        for _ in range(cfg.latency_iters):
            with lat.measure(cuda_sync=device.type == "cuda"):
                model(x1, s1, c1)

    result = {
        "device": str(device),
        "batch_size": cfg.batch_size,
        "params_total": n_params,
        "params_encoder": n_enc,
        "steps": cfg.steps,
        "train_seconds": round(train_s, 2),
        "train_samples_per_s": round(cfg.steps * cfg.batch_size / train_s, 1),
        "train_l1_first50_mean": float(np.mean(losses[:50])),
        "train_l1_last50_mean": float(np.mean(losses[-50:])),
        "eval_l1": eval_l1,
        "mean_predictor_l1": baseline_l1,
        "learned": eval_l1 < 0.5 * baseline_l1,
        "inference_batch1": lat.summary(),
        "resources_after_train": train_snap.as_dict(),
    }
    metrics.write({"final": result})
    metrics.close()
    with open(run_dir / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
