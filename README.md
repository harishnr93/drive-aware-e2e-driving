# drive-aware-e2e-driving

Uncertainty-Aware End-to-End Autonomous Driving with Self-Introspection and Risk-Adaptive Planning.

A compact camera-based driving policy (CNN, later CNN + GRU) predicts future ego waypoints. An
introspection head will predict the probability of an impending driving failure, and a
risk-adaptive planner will act on it. The study asks whether calibrated introspection improves
closed-loop safety, and under which conditions. Hypotheses, baselines and the evaluation
protocol were fixed before any results: see [docs/research_plan.md](docs/research_plan.md) and
[docs/experiment_protocol.md](docs/experiment_protocol.md).

<p align="center">
  <img src="docs/assets/rollout_model_a.gif" alt="Model A driving in closed loop: front camera with predicted waypoints (left) and top-down debug view (right); the clip ends with a rear-end collision" width="720">
  <br>
  <sub>Model A (single-frame CNN) in closed loop on the first <code>test_id</code> scenario (not
  cherry-picked): the last 24 s of the episode in real time (every 4th frame). Left: the 160×96 policy
  camera with predicted waypoints (dots). Right: the privileged top-down view (planned route in
  blue). The episode ends in a rear-end collision at 87% route completion.</sub>
</p>

## Status

| Phase | Content | State |
|---|---|---|
| 0 | Research design, architecture, environment assessment | done |
| 1 | Repository, environment, core contracts, smoke training | done |
| 2 | Lightweight closed-loop simulator, controller, replay/video | done |
| 3 | Rule-based expert, expert dataset (128k frames), labels, validation | done |
| 4 | Model A: single-frame CNN behaviour cloning, open- and closed-loop evaluation | done |
| 5–12 | Temporal model, introspection, uncertainty, risk adaptation, robustness, CARLA, report | not started |

## Simulator and data

A purpose-built 2D simulator (numpy + OpenCV, CPU-only). It has procedural towns, signalised
junctions, IDM traffic, pedestrians, parked cars, a pseudo-perspective front camera, and
deterministic snapshots for counterfactual replays.

<table>
<tr>
<td width="38%"><img src="docs/assets/map_overview.png" alt="Procedurally generated town: grid of roads with curved segments, junctions, crosswalks; the ego route in blue, traffic in red"></td>
<td><img src="docs/assets/camera_topdown.png" alt="Three moments of the expert's drive: cruising, turning right at a junction, waiting at a red light; each shows the policy camera and the top-down view"></td>
</tr>
<tr>
<td><sub>A procedural town (map seed 205, not used for training or testing). Planned route in
blue, vehicles in red, ego in orange.</sub></td>
<td><sub>The rule-based expert cruising, turning right, and waiting at a red light. Left: what
the policy sees (160×96, upscaled). Right: the top-down debug view.</sub></td>
</tr>
</table>

<p align="center">
  <img src="docs/assets/label_montage.png" alt="Eight training frames with the 2-second future-waypoint labels projected into the camera image as dots" width="760">
  <br>
  <sub>Training samples with their labels: the expert's future rear-axle positions over 2 s,
  projected into the camera image. Stopped frames show no dots, because all labels sit at the
  rear axle, behind the camera.</sub>
</p>

## Results so far (Phase 4)

All numbers are on the held-out `test_id` maps. Agents are evaluated on the same 50 scenarios,
and model A is reported as the mean ± sd over 3 training seeds. Full details, with the
diagnosis, are in [docs/results.md](docs/results.md).

| | Open-loop ADE (m) | Route completion | Routes completed | Episodes with a collision | Red-light violations / km |
|---|---|---|---|---|---|
| Expert (privileged, rule-based) | — | 1.000 | 100% | 0% | 0 |
| Constant-velocity extrapolation | 0.768 | — | — | — | — |
| **Model A** (single-frame CNN, 1.15 M params) | **0.162 ± 0.002** | **0.413 ± 0.011** | **8.7 ± 2.3%** | **18.0 ± 2.0%** | **4.05 ± 0.13** |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/closed_loop_model_a_dark.png">
    <img src="docs/assets/closed_loop_model_a_light.png" alt="Bar charts: route completion 100% for the expert vs 41.3% for model A; episodes with a collision 0% for the expert vs 18.0% for model A, with individual seeds shown as dots" width="720">
  </picture>
</p>

**What this shows.** Model A predicts waypoints accurately on logged data: open-loop error is
4.7× lower than constant-velocity extrapolation. Yet it completes few routes when it drives
itself.
- **It is the policy, not the pipeline.** An oracle feeding privileged route waypoints through
  the same controller completes every route.
- **Where it fails.** 76% of failures happen inside junctions. These are large, unmarked
  asphalt areas where the camera has little to anchor on, so small heading errors compound.
- **Signals.** It also runs red lights. Signal heads are a few pixels wide, and one frame
  carries no phase information.

This is the expected compounding-error weakness of behaviour cloning, measured rather than
assumed. It is the baseline that the temporal model (Phase 5) and the introspection work
(Phases 6–8) build on.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/training_curves_model_a_dark.png">
    <img src="docs/assets/training_curves_model_a_light.png" alt="Validation ADE per epoch for three seeds, falling from about 0.25 m to 0.16 m, well below the 0.79 m constant-velocity baseline" width="560">
  </picture>
</p>

### Open decisions (need sign-off; none applied)

These options would change the benchmark or the training method, so they are not applied
silently:
1. **Junction appearance (simulator v2):** smaller corner fillets and guide markings through
   junctions. This requires re-collecting the dataset; v1 results stay reported.
2. **DAgger** with the expert as the labeller, for compounding errors.
3. **Route target-point conditioning** in addition to discrete commands (TransFuser-style).

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
The full CUDA environment is about 6.7 GB on disk. Development hardware: GTX 1650 Ti (4 GB),
Ryzen 7 4800H, 7.4 GiB RAM under WSL2 (see [docs/environment.md](docs/environment.md)).

**Always activate with `source scripts/env.sh`.** If ROS 2 is sourced in `~/.bashrc`, its
`PYTHONPATH` takes precedence over the venv and makes pytest load ROS plugins, which fails at
import. `env.sh` clears it for the current shell only.

## Reproducing

```bash
source scripts/env.sh
pytest                                                    # CPU unit tests
ruff check . && ruff format --check . && mypy

python scripts/collect_dataset.py --config configs/data/expert_v1.yaml   # ~9 min, 6.7 GB
python scripts/inspect_dataset.py data/expert_v1/train                  # stats + label montage
python scripts/eval_expert.py                                           # expert reference

for s in 0 1 2; do                                        # ~13 min per seed on a GTX 1650 Ti
  python scripts/train_bc.py --config configs/training/model_a.yaml --set seed=$s
done
python scripts/eval_closed_loop.py --set-name test_id --expert \
    --checkpoints experiments/*_model_a_seed*/best.pt     # ~6 min with 4 workers

python scripts/make_figures.py sim                        # README figures
python scripts/make_figures.py charts --report experiments/<ts>_closed_loop_test_id/report.json
python scripts/make_figures.py rollout --checkpoint experiments/<ts>_model_a_seed0/best.pt
```

Each run writes `experiments/<timestamp>_<name>/` containing `config.yaml`, `env.json`
(versions, GPU, git commit), `run.log`, metrics and results. Summaries quoted in the docs are
committed under `docs/results/`.

## Layout

```text
configs/            YAML configs (strictly validated against dataclasses)
docs/               research plan, architecture, protocol, dataset, results, decisions
scripts/            env.sh and command-line entry points
src/driveaware/
  core/             contracts (types, frames, timing), config, seeding, logging, hardware
  sim/              simulator interface; lightweight/ = maps, vehicle, traffic, events, camera
  control/          pure pursuit + PID waypoint controller (shared by expert and policies)
  expert/           privileged drivers: route follower, rule-based expert
  data/             collection, labels, storage (memmaps), validation, PyTorch Dataset
  models/           encoders/ (compact CNN), trajectory/ (model A)
  training/         behaviour-cloning trainer
  agents/           closed-loop policy agent (observations only, no privileged state)
  evaluation/       open-loop metrics, closed-loop evaluation, paired statistics
  viz/              replay, video export, result charts
tests/              CPU-only unit tests
```

Conventions: SI units, radians; ego frame at the rear axle with x forward, y left; 20 Hz physics,
10 Hz policy, 10 waypoints at 0.2 s. See [docs/system_architecture.md](docs/system_architecture.md).

## Documentation

| Document | Content |
|---|---|
| [research_plan.md](docs/research_plan.md) | Questions, hypotheses, model ladder, scope of claims |
| [experiment_protocol.md](docs/experiment_protocol.md) | Splits, metrics, statistics, deviation log |
| [system_architecture.md](docs/system_architecture.md) | Interfaces, frames, timing, simulator as built |
| [dataset.md](docs/dataset.md) | Expert, collection, labels, validation, expert reference |
| [results.md](docs/results.md) | Results by phase, with diagnosis |
| [technical_decisions.md](docs/technical_decisions.md) | Decision log |
| [environment.md](docs/environment.md) | Hardware/software assessment and measurements |

## License

MIT
