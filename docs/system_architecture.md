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

## 6. Lightweight simulator design (Phase 2)

- **Map:** procedurally generated road graph from a seed. Lane centrelines are polylines
  (arc-length parameterised), lane width 3.5 m. Elements: straights, constant-curvature arcs,
  3-way and 4-way intersections, crosswalks, signal-controlled junctions.
- **Ego dynamics:** kinematic bicycle at the rear axle. Wheelbase 2.7 m, footprint
  4.6×1.9 m, max steer 0.6 rad, steer-rate limit, `a = 3.0·throttle − 8.0·brake − c·v²`.
  Exact arc integration.
- **Agents:** IDM vehicles on lanes, pedestrians on crosswalks (+ jaywalkers in OOD),
  static obstacles. All deterministic given the scenario seed.
- **Events:** oriented-box collisions (SAT), off-road via footprint corners vs. drivable
  area, red-light crossing of a stop line, route deviation, stall.
- **Renderer (pseudo-perspective camera):** pinhole camera 1.5 m above the ground,
  pitched down. The ground-plane point behind each pixel is constant in the ego frame and is
  precomputed once. Each frame, those points are transformed to world coordinates and
  looked up in a rasterised semantic texture of the map (road, markings, sidewalk, grass,
  stop lines). Agents and traffic lights are drawn as projected 3D boxes (painter's
  algorithm, `cv2.fillPoly`). Target: < 2 ms/frame on the CPU at 160×96. Lighting,
  texture noise, blur, noise, and occlusion are applied as image-space perturbations
  (training augmentation and OOD suites).
- **Top-down debug view** is rendered separately for replay videos. It is never a policy input.

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

## 9. Controller (shared by expert and policy)

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
