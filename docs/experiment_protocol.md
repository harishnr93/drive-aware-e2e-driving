# Experiment Protocol

Status: Phase 0. This protocol is fixed **before** results exist. Any later deviation
will be logged in §9 with its reason.

## 1. Scenario generation and splits

Scenarios are generated from `(map_seed, route_seed, traffic_seed)` by a versioned
generator. **Splits are made at the map level**: no map, and therefore no road segment,
appears in more than one split.

| Split | Purpose | Maps | Distribution |
|---|---|---|---|
| `train` | BC training | disjoint set | nominal |
| `val` | BC model selection and early stopping | disjoint | nominal |
| `intro_train` | closed-loop rollouts for introspection training | disjoint | nominal |
| `intro_cal` | temperature scaling + risk-threshold selection | disjoint | nominal |
| `test_id` | in-distribution test | disjoint | nominal |
| `test_ood_*` | shift suites (§4) | disjoint | shifted |

Seed lists are stored in `configs/evaluation/splits.yaml` and hashed into every result.
Test splits are not looked at until the final evaluation of each phase (no test-set tuning).

## 2. Data collection

- **Expert:** privileged rule-based driver (route pure pursuit, IDM car-following, red-light
  and pedestrian yielding) using the shared controller.
- **Covariate shift mitigation:** in some episodes, DART-style steering/speed noise
  bursts are injected. A sample is kept only if its 2 s label window contains **no**
  injected noise, so labels show recovery from perturbed states and never the noise itself.
- **Labels:** the future ego positions at t + 0.2k s (k = 1..10), transformed into the ego
  frame at t. Samples whose horizon extends past the end of the episode are dropped.
- **Storage:** per-episode `npz` shards plus a JSON/CSV index (episode, map seed, frame,
  command, speed, events). For training, each split is consolidated into a memory-mapped
  `uint8` image array.
- **Integrity checks:** shapes, dtypes, timestamp monotonicity, label/kinematics
  consistency (re-integrating labels gives the recorded poses), command distribution,
  no cross-split map IDs.

## 3. Training protocol

- Each model variant is trained with **3 seeds** (5 if variance is high in the pilot).
- Same data, same augmentation, same epoch budget, and selection by validation ADE for A and B.
- Introspection: trained on `intro_train` rollouts of the **frozen** model B, then
  temperature-scaled on `intro_cal`. Class imbalance is handled with weighted BCE (focal
  loss as ablation). The prevalence is reported.
- Introspection inputs are from time ≤ t only. A unit test enforces this.

## 4. Distribution-shift suites (Phase 9)

| Suite | Shift |
|---|---|
| `ood_geometry` | sharper curves and unseen junction layouts |
| `ood_traffic` | 2–3× traffic density |
| `ood_behaviour` | jaywalkers, cut-ins, hard-braking leads |
| `ood_sensor` | Gaussian noise, motion blur, partial occlusion (3 severity levels each) |
| `ood_delay` | observation delay k ∈ {1, 2, 3} steps |

Each suite has a fixed seed list and a documented parameter range.

## 5. Metrics

**Driving (closed loop):**
- route completion (%), and driving score = completion × infraction penalty (reported as
  secondary)
- collisions per km and per route, off-road rate, red-light violations, stalls
- mean speed and time-to-complete (needed to interpret any safety gain)

**Open loop:** ADE and FDE (m) on held-out frames, per command.

**Failure prediction:**
- AUROC, PR-AUC (with prevalence shown), Brier score
- ECE (15 equal-mass bins) and reliability diagrams
- **Warning lead time:** for each failure event, the time from the first alarm
  (p > τ, held for ≥ 2 steps) to the event. Reported as a distribution, plus the
  detection rate at a fixed false-alarm rate per km.

**Uncertainty:** Spearman ρ(σ, error), AUROC of σ for detecting high-error predictions,
NLL, coverage of 1σ/2σ intervals, sparsification error (AUSE).

**Risk adaptation:**
- number of interventions per km and time spent in each mode
- **False intervention rate:** the fraction of interventions after which a counterfactual
  rollout from the same snapshot, *without* adaptation, has no failure within T_h.
  The deterministic simulator snapshots make this measurable.
- safety–progress trade-off curve while sweeping τ (validation only for selection, test
  reported once)

**Compute:** parameters, peak VRAM, training throughput (samples/s), inference latency
(p50/p95, batch 1), host RAM.

## 6. Heuristic failure predictors (H2 baselines)

All of these use only information available to the policy:
1. speed-only logistic model;
2. trajectory ensemble disagreement / predicted σ;
3. feature-space novelty (Mahalanobis distance to training features);
4. temporal inconsistency of consecutive trajectory predictions.

Reference only: privileged time-to-collision (an oracle upper bound, not a competitor).

## 7. Statistics

- **Paired design:** all models run on identical scenario seeds.
- Report mean ± std across training seeds and 95% bootstrap CIs over routes (10k
  resamples). Pairwise comparisons use the paired bootstrap of the per-route difference.
- Route counts per test suite are fixed after a pilot (Phase 4) from the observed variance
  of the collision rate. Rare-event metrics need many routes, and power will be reported
  honestly.
- Negative and null results are reported with the same detail as positive ones.

## 8. Artefacts per run

`experiments/<date>_<name>/`: resolved config, `env.json`, git commit, seed lists, metrics
JSON, per-route CSV, plots, and checkpoint hashes. Summary tables are generated from these
files by a script, never typed by hand.

## 9. Deviation log

| Date | Deviation | Reason |
|---|---|---|
| 2026-11-02 | Phase 5 adds a B(H=1) control agent (not in the original model ladder) | Model B's training recipe differs from model A's, so A vs B alone confounds history with recipe |
| 2026-11-02 | The first Phase 5 evaluation run computed model B's *open-loop* test metrics with history 1 (a CLI default bug). Closed loop was unaffected (the agent reads the history from the checkpoint). Fixed: `evaluate_checkpoint` now uses the checkpoint's history, with a regression test. The reported open-loop numbers were recomputed by `scripts/summarize_phase5.py` | Bug found during the run; nothing reported used the wrong numbers |
| 2026-11-02 | Route counts after the Phase 4 pilot (§7): with 50 test_id routes, the per-model 95% CI on the collision-episode rate is about ±0.11 (p ≈ 0.18). Detecting a 0.10 absolute difference between two policies at that rate needs roughly 120+ paired routes. **From Phase 5 on, test sets use 150 routes (10 maps × 15)**; Phase 4 numbers stay at 50 | Fixed from pilot variance, as the protocol requires, before any Phase 5 result exists |
| 2026-11-02 | Model A results on test_id are reported as obtained. The junction-appearance change, DAgger and target-point conditioning are listed as open decisions, not applied | Changing the benchmark after seeing test results requires explicit sign-off and re-running all baselines |
| 2026-11-01 | Split seed ranges are defined in `configs/data/expert_v1.yaml` (map seeds 1000–1039 train, 2000–2005 val, 3000–3009 test_id; 4000+/5000+/6000+ reserved), not in `configs/evaluation/splits.yaml` | The dataset config is the single source; the evaluation config will reference the same ranges |
| 2026-11-01 | Noise injection is steering-only (§2 said steering/speed) | Lateral recovery is the main covariate-shift concern. Speed noise is deferred and listed as a limitation |
| 2026-11-01 | test_id statistics and montage were inspected for data QA | No model has been evaluated on test_id. The no-tuning rule applies to model selection |
| 2026-11-01 | Default `max_duration` 180 s (dataset: 150 s for 300 m routes) | Split-phase waits make 400 m routes take about 113 s on average. 120 s caused timeouts of a correct expert |
| 2026-11-01 | Added LANE_DEPARTURE (non-terminal infraction) and TIMEOUT events. Stall timeout is 40 s instead of the 20 s in `research_plan.md` §5 | Lane-departure rate is a required metric. 20 s is shorter than the worst split-phase red wait (~27 s), so a compliant driver would be flagged as stalled |
