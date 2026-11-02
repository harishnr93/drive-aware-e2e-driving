# drive-aware-e2e-driving

Uncertainty-Aware End-to-End Autonomous Driving with Self-Introspection and Risk-Adaptive Planning.

A compact camera-based driving policy (a CNN, and a CNN + GRU variant) predicts future ego
waypoints. An introspection head will predict the probability of an impending driving failure,
and a risk-adaptive planner will act on it. The study asks whether calibrated introspection
improves closed-loop safety, and under which conditions. Hypotheses, baselines and the
evaluation protocol were fixed before any results: see
[docs/research_plan.md](docs/research_plan.md) and
[docs/experiment_protocol.md](docs/experiment_protocol.md).

<table>
<tr>
<td><img src="docs/assets/rollout_model_a.gif" alt="Model A in closed loop on the first test scenario: front camera with predicted waypoints and top-down view; ends with a rear-end collision at 87% route completion"></td>
<td><img src="docs/assets/rollout_model_b.gif" alt="Model B in closed loop on the same scenario: front camera with predicted waypoints and top-down view; ends off-road at 49% route completion"></td>
</tr>
<tr>
<td><sub><b>Model A</b> (single-frame CNN), first <code>test_id</code> scenario, last 24 s in
real time: rear-end collision at 87% route completion.</sub></td>
<td><sub><b>Model B</b> (frozen CNN + GRU over 4 frames), <i>same</i> scenario: leaves the road
at 49% route completion.</sub></td>
</tr>
</table>
<sub>Each clip: left, the 160×96 policy camera with predicted waypoints (dots); right, the
privileged top-down view (planned route in blue). One scenario, not cherry-picked. Aggregate
results are below.</sub>

## Status

| Phase | Content | State |
|---|---|---|
| 0 | Research design, architecture, environment assessment | done |
| 1 | Repository, environment, core contracts, smoke training | done |
| 2 | Lightweight closed-loop simulator, controller, replay/video | done |
| 3 | Rule-based expert, expert dataset (128k frames), labels, validation | done |
| 4 | Model A: single-frame CNN behaviour cloning, open- and closed-loop evaluation | done |
| 5 | Model B: temporal (GRU) policy with controls and ablations | done |
| 6–12 | Introspection, uncertainty, risk adaptation, robustness, CARLA, report | not started |

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

## Results

All numbers are on held-out `test_id` maps. All agents drive the same scenarios, and learned
models are reported as the mean ± sd over 3 training seeds. Full tables, paired statistics and
diagnoses are in [docs/results.md](docs/results.md).

### Phase 5: does temporal context help? Not in this setup.

Model B feeds the frozen model-A encoder's features for the last 4 frames through a GRU. A
**control, B (H=1)**, has the same frozen features and head training but no history. It
separates "temporal context" from "retraining the head".

| 150 paired routes | Route completion | Routes completed | Episodes with a collision | Red-light violations / km |
|---|---|---|---|---|
| Expert | 1.000 | 99.3% | 0% | 0 |
| Model A (single frame) | 0.431 ± 0.002 | 11.3 ± 1.8% | 20.9 ± 1.0% | 3.53 ± 0.15 |
| Model B (H=1) control | 0.473 ± 0.016 | 15.1 ± 1.0% | 23.6 ± 2.0% | 3.68 ± 0.40 |
| **Model B (H=4)** | **0.463 ± 0.026** | **13.8 ± 2.5%** | **23.3 ± 1.2%** | **3.75 ± 0.18** |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/closed_loop_phase5_dark.png">
    <img src="docs/assets/closed_loop_phase5_light.png" alt="Three bar panels for Expert, Model A, Model B (H=1) and Model B: route completion 100, 43.1, 47.3, 46.3 percent; episodes with a collision 0, 20.9, 23.6, 23.3 percent; red-light violations per km 0, 3.5, 3.7, 3.8" width="860">
  </picture>
</p>

- **History adds nothing measurable.** Paired B(H=4) − B(H=1): route completion −0.010
  [95% CI −0.026, 0.006]. Collisions and red-light violations are also unchanged. This is
  a negative result for hypothesis H1.
- **The gain over model A is a training-recipe effect.** B(H=1) − A is +0.042 [0.023, 0.061]
  route completion, the same gain B(H=4) shows. Without the control, it would have looked
  like evidence for H1.
- **Starting from rest improves slightly**, the case where one frame lacks timing cues:
  0.287 → 0.266 → 0.254 m ADE (A → B(H=1) → B). The change is within seed variation.
  History lengths from 1 to 8 frames change val ADE by under 0.003 m.

<table>
<tr>
<td width="56%">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/situations_phase5_dark.png">
    <img src="docs/assets/situations_phase5_light.png" alt="Grouped bars of open-loop ADE by motion situation for Model A, Model B (H=1) and Model B: start from rest 0.29, 0.27, 0.25; stopping 0.08, 0.07, 0.07; cruising 0.19 for all; holding 0.04, 0.05, 0.06">
  </picture>
</td>
<td>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/history_ablation_dark.png">
    <img src="docs/assets/history_ablation_light.png" alt="Validation ADE versus history length 1, 2, 4, 8 frames: 0.157, 0.155, 0.154, 0.155; reference lines for model A at 0.157 and for H=4 with speed history at 0.149">
  </picture>
</td>
</tr>
</table>

Feeding past speeds lowers open-loop error (0.149 m), but almost entirely on cruising frames.
That is consistent with the inertia ("copycat") shortcut, so it is kept as an ablation, not
adopted. Fine-tuning the encoder end to end (one seed, 55 min instead of about 1 min) does not help
either: route completion is 0.436 vs 0.447 for frozen B on the same seed. The result covers
temporal aggregation over ≤ 0.7 s of history.

### Phase 4: single-frame behaviour cloning

Model A predicts waypoints well on logged data: test ADE is 0.162 ± 0.002 m, against 0.768 m for
constant-velocity extrapolation. But it drives poorly: on the original 50-route set it completes
0.413 ± 0.011 of the route distance and finishes 8.7% of routes.
- **The pipeline is sound.** An oracle feeding privileged route waypoints through the same
  controller completes every route.
- **Junctions.** 76% of failures happen inside junctions. These are large, unmarked asphalt
  areas where the camera has little to anchor on.
- **Signals.** Signal heads are only a few pixels wide.

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

for s in 0 1 2; do                                        # model A: ~13 min per seed
  python scripts/train_bc.py --config configs/training/model_a.yaml --set seed=$s
done
for s in 0 1 2; do                                        # model B and its H=1 control: ~2 min
  A=$(ls -d experiments/*_model_a_seed$s)/best.pt
  python scripts/train_bc.py --config configs/training/model_b.yaml --set seed=$s encoder_checkpoint=$A
  python scripts/train_bc.py --config configs/training/model_b.yaml --set seed=$s encoder_checkpoint=$A name=model_b_h1 history=1
done
python scripts/eval_closed_loop.py --set-name test_id --expert --workers 6 \
    --checkpoints experiments/*_model_a_seed*/best.pt experiments/*_model_b_h*_seed*/best.pt
python scripts/summarize_phase5.py --report experiments/<ts>_closed_loop_test_id/report.json

python scripts/make_figures.py sim                        # README figures
python scripts/make_figures.py charts --report experiments/<ts>_closed_loop_test_id/report.json
python scripts/make_figures.py phase5 --summary docs/results/phase5_temporal.json
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
  models/           encoders/ (compact CNN), trajectory/ (model A), temporal/ (model B)
  training/         behaviour-cloning trainer; cached-feature trainer for temporal models
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
