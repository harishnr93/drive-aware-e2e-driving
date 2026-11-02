"""Train a behaviour-cloning waypoint policy.

Usage:  python scripts/train_bc.py --config configs/training/model_a.yaml [--set seed=1 ...]
"""

from __future__ import annotations

import argparse
import json

import yaml

from driveaware.core.config import load_config
from driveaware.core.logging import setup_logging
from driveaware.core.run import REPO_ROOT, create_run_dir
from driveaware.training.bc import TrainConfig, train
from driveaware.training.temporal import train_temporal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/model_a.yaml")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(TrainConfig, args.config, overrides)
    if not cfg.data_root.startswith("/"):
        cfg.data_root = str(REPO_ROOT / cfg.data_root)
    if cfg.encoder_checkpoint and not cfg.encoder_checkpoint.startswith("/"):
        cfg.encoder_checkpoint = str(REPO_ROOT / cfg.encoder_checkpoint)
    run_dir = create_run_dir(f"{cfg.name}_seed{cfg.seed}", cfg)
    setup_logging(run_dir)
    cached = cfg.model == "temporal" and cfg.temporal.freeze_encoder
    result = train_temporal(cfg, run_dir) if cached else train(cfg, run_dir)
    print(json.dumps({"run_dir": str(run_dir), **result}, indent=2, default=str))


if __name__ == "__main__":
    main()
