# drive-aware-e2e-driving

Uncertainty-Aware End-to-End Autonomous Driving with Self-Introspection and Risk-Adaptive Planning.

A compact camera-based driving policy (CNN + GRU) that predicts future ego waypoints, plus an
introspection head that predicts the probability of an impending driving failure, used by a
risk-adaptive planner. The study asks whether calibrated introspection improves closed-loop
safety, and under which conditions. See [docs/research_plan.md](docs/research_plan.md).

## Status

| Phase | Content | State |
|---|---|---|
| 0 | Research design, architecture, environment assessment | done |
| 1 | Repository, environment, core contracts, smoke training | done |
| 2+ | Lightweight simulator, expert data, models, introspection, … | not started |

No research results exist yet.

## Installation (Ubuntu 22.04 / WSL2, Python 3.10)

```bash
python3.10 -m venv .venv
source scripts/env.sh                       # activates .venv and isolates it from ROS 2 (see below)
pip install --upgrade pip
pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

CPU-only machines: install `torch` from `https://download.pytorch.org/whl/cpu` instead.
The full CUDA environment is about 6.7 GB on disk.

**Always activate with `source scripts/env.sh`.** If ROS 2 is sourced in `~/.bashrc`, its
`PYTHONPATH` takes precedence over the venv and makes pytest load ROS plugins, which fails at
import. `env.sh` clears it for the current shell only.

## Usage

```bash
source scripts/env.sh
pytest                          # CPU unit tests
ruff check . && ruff format --check . && mypy
python scripts/smoke_train.py --set batch_size=4      # environment validation (synthetic data)
python scripts/smoke_train.py --set device=cpu steps=100
```

Each run writes `experiments/<timestamp>_<name>/` containing `config.yaml`, `env.json`
(versions, GPU, git commit), `run.log`, `metrics.jsonl` and `result.json`.

## Layout

```text
configs/            YAML configs (strictly validated against dataclasses)
docs/               research plan, architecture, protocol, environment, decisions
scripts/            env.sh and command-line entry points
src/driveaware/
  core/             contracts (types, frames, timing), config, seeding, logging, hardware, run dirs
  models/encoders/  compact CNN encoder
tests/              CPU-only unit tests
```

Conventions: SI units, radians; ego frame at the rear axle with x forward, y left; 20 Hz physics,
10 Hz policy, 10 waypoints at 0.2 s. See [docs/system_architecture.md](docs/system_architecture.md).

## License

MIT
