# Technical Decisions

Format: decision, alternatives considered, rationale. Status values: **accepted**
(Phase 0), **provisional** (to be confirmed by measurement), **deferred**.

| # | Decision | Alternatives | Rationale | Status |
|---|---|---|---|---|
| D1 | Python 3.10 venv | conda env; Python 3.12 | Matches ROS 2 Humble and CARLA client ABI; venv is lightweight | accepted |
| D2 | `torch 2.14.1+cu126` | cu128 (max 2.11); CPU only | Newest CUDA build with cp310; driver supports CUDA 13.1 | accepted: sm_75 verified, Phase 1 |
| D3 | FP32 default; FP16 AMP later | BF16 | Turing has no BF16 hardware | accepted |
| D4 | Single package `src/driveaware/` | top-level `models/`, `datasets/`, … | Avoids import shadowing (HF `datasets`); installable and testable | accepted |
| D5 | Unset `PYTHONPATH`/AMENT vars in `scripts/env.sh` | rely on venv isolation | ROS Humble paths override venv site-packages and auto-load pytest plugins | accepted |
| D6 | Own lightweight simulator (numpy + OpenCV) | highway-env, MetaDrive, CARLA only | Needs junctions, signals, pedestrians, camera-like view, deterministic snapshots for counterfactuals, and a CPU-only footprint. Existing simulators lack one or more of these or are too heavy | accepted |
| D7 | Pseudo-perspective ground-plane renderer | top-down BEV raster as policy input | Keeps a *front camera* contract shared with CARLA; precomputed per-pixel ground lookup is cheap | accepted; measured 1.8 ms empty / 4.3 ms busy (< 2 ms target missed for busy scenes) |
| D8 | Ego frame at rear axle, FLU (REP-103) | vehicle centre; CARLA's left-handed frame | Natural for bicycle model and pure pursuit; ROS-consistent; CARLA converted at the adapter | accepted |
| D9 | 20 Hz physics, 10 Hz policy, 10 waypoints × 0.2 s | 20 Hz policy; 4 waypoints | Matches common E2E settings (2 s horizon); halves inference cost | accepted |
| D10 | Image 160×96 | 160×90 | Divisible by 32 for 5 stride-2 stages, close to the suggested resolution | accepted |
| D11 | Target speed derived from waypoints | separate speed head | Avoids inconsistent outputs; the planner scales it explicitly | accepted |
| D12 | Introspection head reads detached features; C drives identically to B | joint training | Preserves the B/C driving equivalence and separates "can it predict" from "does acting help" | accepted |
| D13 | Failure label = event in (t, t+2 s]; collision/off-road/red-light/deviation/stall | collisions only | Collisions alone are too rare for stable training; event types are also reported separately | accepted |
| D14 | Deep ensemble (M = 5) + heteroscedastic head; MC dropout baseline | Bayesian NN, evidential | Strong, simple baselines; small models make M = 5 affordable | provisional |
| D15 | Deterministic sim snapshots → counterfactual false-intervention metric | proxy via label agreement | Gives a direct estimate of whether an intervention was necessary | accepted |
| D16 | Matched-speed and heuristic-trigger controls for D | compare D vs. C only | Without these, safety gains could come from driving slower | accepted |
| D17 | Map-level data splits | random frame split | Prevents leakage between nearly identical frames and routes | accepted |
| D18 | Plain YAML + dataclass config schema | Hydra, OmegaConf | Fewer dependencies; explicit validation; config saved with each run | accepted |
| D19 | Per-episode `npz` → per-split memmap | HDF5, LMDB, JPEG | Lossless, no extra deps, fits 7.4 GiB RAM via memmap | provisional |
| D20 | CARLA via Windows-native server + WSL client, or remote server | Linux CARLA inside WSL2 | WSL2 lacks native NVIDIA Vulkan; GPU is below CARLA's recommended spec | deferred to Phase 10 |
| D21 | ruff + pytest (+ mypy on `core/`) | flake8/black/pylint | One fast tool for lint and format | accepted |
| D22 | Split-phase signals at every junction with ≥ 3 roads | two-phase with permissive lefts; unsignalised junctions | Conflict-free without a right-of-way model. Cost: unrealistic waits (documented) | accepted (Phase 2) |
| D23 | One semantic raster for rendering and off-road checks | separate polygon geometry for events | What the camera shows and what is scored cannot disagree | accepted |
| D24 | Connector feasibility: curve offset ≤ 16 m, trim 16 m, Hermite tangent scale 1.2 | sharper geometry | The first version had 3.1 m turn radii, below the vehicle's 3.95 m minimum. Now ≥ 4.86 m over 40 maps, enforced by a test (≥ 4.45 m) | accepted |
| D25 | Pedestrians wait at the kerb and never walk into vehicles; vehicles yield to pedestrians on the carriageway | pedestrians ignore vehicles | Without this, NPC–pedestrian overlaps were frequent (34 in 4×120 s). Now 0 in 6×120 s | accepted |
| D26 | LANE_DEPARTURE and TIMEOUT added to `EventType`; not part of the default failure set | lane departure counted as a failure | Lane departure is an infraction metric. Its inclusion in failure labels is decided in Phase 6 | accepted |
| D27 | Controller (pure pursuit + PID) and a traffic-unaware route follower implemented in Phase 2 | wait until Phase 3 | The simulator acceptance test needs a closed-loop driver. The Phase 3 expert extends the route follower | accepted |

## Rejected or deferred ideas

- **Vision transformers or foundation-model encoders:** do not fit the 4 GB budget, and
  they are not needed for the research question.
- **Feeding past actions to the policy:** worsens copycat behaviour. Speed is the only
  ego-motion input.
- **Target-point conditioning (TransFuser-style):** possible later ablation. The primary
  contract uses discrete commands, as specified.
