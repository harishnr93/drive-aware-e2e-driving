# Research Plan

**Uncertainty-Aware End-to-End Autonomous Driving with Self-Introspection and Risk-Adaptive Planning**

Status: Phase 0 design. No experiments have been run. Everything below is a plan or a
hypothesis, not a result.

## 1. Question

Can a compact, camera-based end-to-end driving policy **predict its own impending
failures**, and does acting on calibrated failure predictions **improve closed-loop safety
at an acceptable cost** in progress and unnecessary interventions?

## 2. Research questions

| ID | Question | Primary evidence |
|---|---|---|
| RQ1 | Can a compact visual model learn trajectory prediction and closed-loop driving? | ADE/FDE; route completion and collision rate vs. expert |
| RQ2 | Does temporal modelling help over single-frame BC? | Model B vs. A, closed loop, same routes |
| RQ3 | Can introspection predict future failures before they occur? | AUROC, PR-AUC, lead time on held-out rollouts |
| RQ4 | Are failure probabilities and trajectory uncertainty calibrated? | Brier, ECE, reliability diagrams; interval coverage, NLL |
| RQ5 | Does risk-adaptive behaviour reduce safety events without excessive interventions? | Model D vs. C and vs. matched cautious baselines |
| RQ6 | Do benefits persist under distribution shift? | All of the above on the OOD test suites |

## 3. Hypotheses and how each could be falsified

| ID | Hypothesis | Falsified if |
|---|---|---|
| H1 | Temporal representations improve closed-loop driving | B's driving score is not better than A's across seeds (paired CI includes 0 or favours A) |
| H2 | Introspection predicts failures better than simple heuristics | Introspection AUROC/PR-AUC CI overlaps or falls below the best non-privileged heuristic |
| H3 | Calibrated failure probability is useful for choosing conservative behaviour | Thresholds tuned on validation fail to transfer to test (large ECE, or the false-alarm rate deviates strongly from target) |
| H4 | Risk-adaptive control reduces collisions/safety events | D does not reduce safety events vs. C **and** vs. a uniformly slower baseline with equal mean speed |
| H5 | Introspection degrades under distribution shift | OOD AUROC/ECE is not worse than in-distribution |

Notes:
- **H1 has a known risk of being false.** Temporal BC is prone to the *copycat / inertia*
  problem (the policy infers the expert's action from history instead of from the scene).
  A negative result here is reportable and informative.
- **H4 needs a matched control.** A policy that simply drives slower will crash less. D
  must beat (a) C with a constant speed scale chosen to match D's mean speed and (b) C with
  a heuristic trigger (speed, ensemble variance). Otherwise the gain cannot be attributed
  to introspection.

## 4. Model ladder (primary comparison)

| Model | Temporal | Introspection | Risk adaptation |
|---|---|---|---|
| A: CNN baseline | No | No | No |
| B: CNN + GRU | Yes | No | No |
| C: B + introspection head | Yes | Yes | No |
| D: C + risk-adaptive planner | Yes | Yes | Yes |

**By design, C drives exactly like B.** The introspection head reads detached features and
does not change the trajectory output. C is therefore evaluated on *prediction quality*
(RQ3/RQ4), and D is the closed-loop test of introspection (RQ5). Saying this explicitly
avoids presenting identical driving numbers as two separate results.

Additional controls (Phase 8/11):
- **C + constant slowdown**: speed scaled so that the mean speed matches D's.
- **C + heuristic trigger**: the same adaptive planner driven by a heuristic risk score.
- **Expert (privileged)**: upper reference for driving metrics.
- **Oracle trigger (privileged TTC)**: upper reference for the value of perfect risk
  information. Clearly labelled as not deployable.

## 5. Failure definition (summary — full spec in `experiment_protocol.md`)

A *failure event* at time t_f is any of: collision; off-road (ego footprint leaves the
drivable area); red-light violation; route deviation > 5 m; stall > 40 s without progress (changed from 20 s, see the deviation log in `experiment_protocol.md`).
The introspection label at time t is `y_t = 1` iff a failure occurs in `(t, t + T_h]`,
with `T_h = 2.0 s` by default (ablation: 1, 2, 3 s). Inputs at time t use only data from
`≤ t`.

## 6. Scope of claims

- Results come from a purpose-built 2D/pseudo-perspective simulator. They validate the
  **methodology**, not real-world driving performance.
- Transfer to CARLA will be reported as a domain-shift experiment. It will not be assumed.
- Introspection is trained on rollouts of one specific policy. If the policy changes, the
  introspection model must be retrained. This dependency is part of the study, not a footnote.

## 7. Related work to position against (for the report)

Conditional imitation learning (Codevilla et al. 2018; CIL++), learning by cheating (Chen
et al. 2019), TransFuser/TCP-style waypoint prediction, DART noise injection and DAgger for
covariate shift, copycat analyses (Wen et al. 2020), introspective failure prediction
(Daftry et al. 2016; Kuhn et al. 2020 on introspective failure prediction for AD), deep
ensembles (Lakshminarayanan et al. 2017), and calibration via temperature scaling (Guo et
al. 2017). Citations will be verified before the report is written.

## 8. Timeline (phase gates)

Each phase ends with its acceptance criteria verified and recorded. The next phase starts
only after review. Phases 0–4 are the minimum viable study (RQ1). Phases 5–8 address
the core contribution (RQ2–RQ5). Phase 9 covers RQ6. Phase 10 (CARLA) is optional and depends on hardware.
