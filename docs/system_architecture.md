# System Architecture

Status: Phase 0 design. Interfaces here are the contracts that later phases implement
and test.

## 1. Data flow

```text
 Simulator backend (lightweight | CARLA)            privileged side (never seen by policy)
 ┌──────────────────────────────────────┐          ┌───────────────────────────────────┐
 │ step(ControlCommand) -> StepResult   │──────────▶ PrivilegedState: poses, agents,   │
 │ reset(ScenarioSpec)                  │          │ lanes, signals, events             │
 │ get_state()/set_state()  (snapshot)  │          │  → expert, labeler, metrics only   │
 └───────────────┬──────────────────────┘          └───────────────────────────────────┘
                 │ Observation (image, speed, command, timestamp)
                 ▼
       ObservationHistory (ring buffer, H frames, optional delay k)
                 ▼
   CNN encoder (shared per frame) ─▶ GRU ─┬─▶ Trajectory head ─▶ waypoints [10×2], σ
                                          └─▶ Introspection head (detached input) ─▶ p_fail
                 ▼
   Risk-adaptive planner (mode FSM with hysteresis) ─▶ adjusted waypoints + target speed
                 ▼
   Controller (pure pursuit + PID) ─▶ ControlCommand(steer, throttle, brake)
```

## 2. Package layout

All code lives in one installable package, `src/driveaware/`. Generic top-level names
such as `datasets/` or `models/` would shadow third-party packages (e.g. HuggingFace
`datasets`).

```text
drive-aware-e2e-driving/
├── pyproject.toml
├── configs/{simulation,models,training,evaluation}/*.yaml
├── docs/
├── scripts/                 # thin CLIs: collect, train, evaluate, env.sh
├── src/driveaware/
│   ├── core/                # contracts: types.py, frames.py, timing.py, config.py, seeding.py, logging.py
│   ├── sim/
│   │   ├── interface.py     # DrivingSimulator protocol
│   │   ├── lightweight/     # roads, vehicle, agents, signals, collision, renderer, scenarios
│   │   └── carla_adapter/   # Phase 10 only
│   ├── expert/              # privileged rule-based driver
│   ├── data/                # collection, labeling, storage, torch Dataset, validation
│   ├── models/              # encoders, temporal, trajectory, introspection, uncertainty
│   ├── training/
│   ├── planning/            # risk-adaptive mode FSM
│   ├── control/             # pure pursuit, PID, longitudinal mapping
│   ├── evaluation/          # closed-loop runner, metrics, statistics
│   └── viz/
├── tests/                   # CPU-only
├── experiments/             # run outputs (git-ignored except summaries)
└── ros2_ws/                 # optional, Phase 10+ (RViz2/Foxglove bridge)
```

Directories are created only when a phase needs them.

## 3. Coordinate frames and units

All internal quantities use **SI units and radians**.

| Frame | Origin | Axes | Notes |
|---|---|---|---|
| `world` | Map origin | x east, y north, z up (right-handed, REP-103) | yaw measured CCW from +x, wrapped to (−π, π] |
| `ego` | **Rear-axle centre** projected to the ground | x forward, y left, z up | Bicycle-model reference point; all waypoint labels live here |
| `camera` | Camera optical centre | x right, y down, z forward (OpenCV) | Extrinsics given relative to `ego` |
| `image` | Top-left pixel | u right, v down | H×W = 96×160 |

**CARLA conversion (Phase 10):** CARLA is left-handed (x fwd, y right, z up; yaw in degrees,
clockwise when seen from above). Adapter: `y = −y_carla`, `yaw = −deg2rad(yaw_carla)`.
CARLA's vehicle origin is the bounding-box centre, so it is shifted to the rear axle.
Unit tests check this conversion on known poses.

## 4. Timing

| Quantity | Value | Configurable |
|---|---|---|
| Physics step `dt_sim` | 0.05 s (20 Hz) | yes |
| Policy/control rate | 10 Hz (control held for 2 physics steps) | yes |
| Observation timestamp | sim time at which the image was rendered | — |
| History | H frames at 10 Hz, default H = 4 (0.4 s); ablation H ∈ {1, 2, 4, 8} | yes |
| Prediction horizon | 10 waypoints at t + 0.2 k s, k = 1..10 (2.0 s) | yes |
| Observation delay | k ∈ {0, 1, 2, 3} control steps (OOD suite) | yes |
| Failure label horizon `T_h` | 2.0 s (ablation 1/2/3 s) | yes |

Inference is synchronous by default (simulation time is frozen during inference).
Wall-clock latency is measured and reported separately. Delay is modelled explicitly via `k`.

## 5. Contracts (to become `core/types.py`)

```python
class Command(IntEnum):          # high-level navigation command
    LANE_FOLLOW = 0; LEFT = 1; RIGHT = 2; STRAIGHT = 3

@dataclass(frozen=True)
class Observation:               # the ONLY input the learned policy receives
    timestamp: float             # s, sim time
    frame_id: int
    image: np.ndarray            # uint8 [96, 160, 3] RGB
    speed: float                 # m/s, ego longitudinal speed (odometry-equivalent)
    command: Command

@dataclass(frozen=True)
class TrajectoryPrediction:
    timestamp: float
    waypoints: np.ndarray        # float32 [10, 2], ego frame at `timestamp`, metres
    waypoint_std: np.ndarray | None   # float32 [10, 2], predictive std (Phase 7)
    target_speed: float          # m/s, derived from waypoint spacing unless overridden

@dataclass(frozen=True)
class IntrospectionOutput:
    failure_prob: float          # P(failure in (t, t+T_h]) — calibrated
    horizon_s: float
    features: dict[str, float]   # e.g. ensemble variance, for analysis only

@dataclass(frozen=True)
class ControlCommand:            # identical semantics to carla.VehicleControl
    steer: float                 # [-1, 1], + = left (ego frame); scaled by max steer angle
    throttle: float              # [0, 1]
    brake: float                 # [0, 1]

@dataclass(frozen=True)
class StepResult:
    observation: Observation
    privileged: PrivilegedState  # consumed by expert/labeler/metrics; never by the policy
    events: list[Event]          # collision, off_road, red_light, route_deviation, stall, route_complete
    done: bool
```

Simulator protocol:

```python
class DrivingSimulator(Protocol):
    def reset(self, scenario: ScenarioSpec) -> StepResult: ...
    def step(self, action: ControlCommand) -> StepResult: ...     # advances 1 control period
    def get_state(self) -> bytes: ...                               # full deterministic snapshot
    def set_state(self, state: bytes) -> None: ...
```

`get_state/set_state` enables **counterfactual branching**. When D intervenes, the
evaluator replays from the snapshot without adaptation to check whether a failure would
actually have happened. This is the basis of the false-intervention metric.

**Leakage guard.** The policy's `act(history: Sequence[Observation])` cannot access
`PrivilegedState` (enforced by types and covered by a test). A dataset test also asserts
that every input timestamp ≤ the label anchor time t, and every label timestamp > t.

## 6. Lightweight simulator (Phase 2, as built)

Code: `src/driveaware/sim/lightweight/`. Tests: `tests/sim/`.

- **Map (`roadmap.py`):** a 4×4 jittered grid of nodes (80 m spacing, ±6 m). 4-neighbour
  roads are dropped at random, but never so as to create a dead end or disconnect the network.
  35% of roads are quadratic Bézier curves (control-point offset 6–16 m). Each road has
  one 3.5 m lane per direction plus a 2 m parking shoulder and a 2.5 m sidewalk. Road lanes
  stop 16 m from the node centre, and Hermite connector lanes join each incoming lane to
  every outgoing lane except the U-turn. Junction areas are drawn as discs, which makes them
  visually plaza-like.
- **Single source of truth:** the map is rasterised once (0.1 m/px, semantic classes plus a
  colour raster with static texture noise as a motion cue). The camera renderer and
  off-road detection both read this raster.
- **Commands (`route.py`):** LEFT/RIGHT/STRAIGHT for connectors at junctions with ≥ 3 roads,
  announced 20 m before the junction. Bends at degree-2 nodes are LANE_FOLLOW.
- **Signals:** every junction with ≥ 3 roads is signalised with **split phasing**: approaches
  get green one at a time (6 s green, 2 s yellow, 1 s all-red), so greens never conflict.
  Each junction has a random offset. The cost is long red waits of up to ~27 s at 4-way
  junctions. Signal state is a pure function of time, so it needs no snapshot state.
- **Ego dynamics (`vehicle.py`):** kinematic bicycle at the rear axle, as specified, with a
  1.2 rad/s steering-rate limit. Integration is exact for constant curvature and
  acceleration within a 0.05 s step.
- **Traffic (`agents.py`):** IDM vehicles on random lane walks. The leader is the nearest
  actor in a lane corridor ahead, or a virtual obstacle at a red stop line (or a yellow one
  when stopping is comfortable). NPCs yield to pedestrians on the carriageway ahead.
  Pedestrians arrive at mid-block crosswalks (Poisson, mean 12 s), wait at the kerb while a
  vehicle occupies the crosswalk, and never step into a vehicle. Parked cars sit on the
  shoulder and intrude 0–0.3 m into the lane.
- **Events (`env.py`):** COLLISION (SAT test every physics step, terminal), OFF_ROAD
  (footprint corners and edge midpoints vs. raster, terminal), LANE_DEPARTURE (box centre
  more than half a lane from the route, rising edge with 0.2 m hysteresis, non-terminal),
  RED_LIGHT (front bumper crosses a red stop line, non-terminal), ROUTE_DEVIATION (> 5 m,
  terminal), STALL (< 0.1 m/s for 40 s, terminal), ROUTE_COMPLETE, TIMEOUT.
  NPC–NPC contacts are not scored.
- **Camera (`renderer.py`):** 90° HFOV, 8° pitch, 1.5 m high, 1.5 m ahead of the rear axle,
  2× supersampling, fog from 35–80 m. Agents and signal heads are shaded cuboids with
  back-face culling, near-plane clipping, and far-to-near painting. Each signal head sits over
  its approach lane on the far side of the junction, 5.2 m up, so it stays visible while
  waiting at the stop line. Only the face pointing at the approach is lit.
- **Determinism:** episode-static content (map, route, NPC paths, parked cars) is rebuilt
  from `ScenarioSpec` seeds. `get_state()` pickles only `SimState` (about 1 KB) plus the
  traffic RNG state. A replay log is `(spec, actions)`.
- **Debug output (`viz/replay.py`, `scripts/sim_demo.py`):** a top-down view and MP4 export.
  The top-down view is never a policy input.

Measured performance (one CPU core, default traffic: 12 vehicles, 6 parked cars):

| Item | Measured |
|---|---|
| Map generation (cached per seed) | 0.4–0.6 s |
| Reset on a cached map | 12–14 ms |
| Camera render: ground only / empty scene / 20 agents + 12 signal heads | 0.9 / 1.8 / 4.3 ms |
| Full control step incl. render (demo run, 1271 steps) | mean 9.7 ms, p95 12.5 ms (~10× real time) |
| Full control step without render (20 NPCs) | ~12 ms incl. test-side overlap checks |
| Process RSS | ~0.6 GiB |

The < 2 ms render target is met only for scenes without agents. Cuboid drawing in Python
dominates busy scenes. Vectorising it is deferred until data collection is shown to be
render-bound.

Known limitations:
- Signal heads are about 3×5 px at the stop line at 160×96. Whether a policy can read the
  signal state at this resolution is an open question for Phase 4 (red-light violations will
  show it).
- Split phasing is unrealistic and produces long waits. Unsignalised junctions and permissive
  left turns are not modelled.
- Pedestrians appear only at crosswalks. Jaywalking is reserved for the OOD suite (Phase 9).
- Static obstacles are parked cars only. Blocked lanes that require an overtake are not modelled.

## 7. Model design (Phases 4–7)

| Block | Design | Approx. size |
|---|---|---|
| Encoder | 5 stride-2 conv blocks (32-64-96-128-192 ch, BN, ReLU) → global avg pool → 256-d | 1.07 M params (measured) |
| Measurement fusion | speed (normalised) + one-hot command → MLP(64), concatenated with image feature | small |
| A: trajectory head | MLP 320→256→20 (10×(x, y)); L1 loss on waypoints | ~0.1 M |
| B: temporal | shared encoder per frame → GRU(256) over H frames → same head | +0.4 M |
| C: introspection | MLP on [detached GRU state, speed, command, uncertainty features] → logit; weighted BCE or focal loss; temperature scaling on a calibration split | ~0.05 M |
| Uncertainty | (i) heteroscedastic Laplace head (aleatoric); (ii) ensemble of M = 5 models (epistemic). MC dropout as a cheap comparison | ×M at inference |
| Ablation | ImageNet-pretrained ResNet-18 encoder | 11 M |

Memory estimate (FP32, 96×160 input, B = 32 frames): activations ≪ 1 GiB, so batch 4 is a
safe *start*. Batch size is raised after measuring peak VRAM.

Target speed comes from the spacing of the first waypoints (v ≈ ‖w₂ − w₀‖ / 0.4 s). This
avoids a separate speed output that could contradict the waypoints.

## 8. Risk-adaptive planner (Phase 8)

A finite-state machine over `NORMAL → CAUTIOUS → FALLBACK`:
- Enter CAUTIOUS when p > τ_c, enter FALLBACK when p > τ_f, with exit thresholds
  τ' < τ (hysteresis) and a minimum dwell time.
- CAUTIOUS: scale target speed by α (e.g. 0.5) and cap lateral acceleration.
- FALLBACK: decelerate to a stop along the predicted path at ≤ 3 m/s² and hold.
  Release requires p < τ'_f for N consecutive steps.
- The thresholds are fixed on the **validation** split to meet a target false-alarm rate,
  then frozen for test.
- Every mode change is logged with its timestamp, p, and a state snapshot ID.

## 9. Controller (shared by expert and policy; implemented in Phase 2)

- Lateral: pure pursuit on a waypoint path, lookahead `L_d = clip(k·v + L_0, 2, 10) m`.
- Longitudinal: PID on target speed, mapped to throttle/brake with a dead-band to avoid
  chattering.
- The expert and the learned policy use the same controller. Their driving difference
  therefore comes from the trajectory source, not the controller.

## 10. Optional integrations

- **ROS 2 (Humble):** a bridge node in `ros2_ws/` republishes Observation, predicted path,
  p_fail, and mode for RViz2/Foxglove. It runs in a separate ROS-sourced shell and is not
  needed for training.
- **CARLA:** implements `DrivingSimulator`. Expert data in CARLA uses a privileged autopilot.
- **Remote GPU:** training scripts depend only on config + dataset path, so they run
  unchanged on a remote machine.
