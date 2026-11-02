# Results Log

Each section reports one completed experiment. Numbers are copied from committed summaries in
`docs/results/` (generated from run directories, never typed by hand), and every number is
labelled with the split it comes from.

## Phase 4 — Model A (single-frame CNN behaviour cloning)

Summary file: [`docs/results/phase4_model_a.json`](results/phase4_model_a.json).

### Setup

| Item | Value |
|---|---|
| Model | Compact CNN encoder (1.07 M) + speed/command MLP + waypoint MLP head; 1.15 M parameters in total |
| Inputs | Current 96×160 RGB image, ego speed, navigation command (no privileged state) |
| Output | 10 waypoints at 0.2 s spacing (2 s) in the ego frame |
| Data | `expert_v1` train (75,488 samples) |
| Loss / optimiser | L1 on waypoints; AdamW 3e-4, weight decay 1e-4, cosine schedule, 12 epochs, batch 64, ±10% brightness/contrast jitter |
| Sampling | Frames that stay stationary for the whole 2 s window are weighted 0.3 (22.5% of samples) |
| Selection | Best validation ADE (val split only) |
| Seeds | 3 training seeds (0, 1, 2) |
| Closed loop | Predicted waypoints → target speed \|w₂ − w₀\| / 0.4 s → the expert's pure-pursuit + PID controller |
| Test set | `test_id`: maps 3000–3009, 5 routes each (50 scenarios), routes/traffic not seen in the dataset, identical for all agents |

### Training cost (GTX 1650 Ti, FP32)

| Measure | Value |
|---|---|
| Time per seed | 761–796 s (12 epochs, ~1,200–1,290 samples/s, data-loading bound) |
| Peak GPU memory | 744 MiB allocated / 1,350 MiB reserved (includes batch-256 validation) |
| Training process RSS | 1.5–1.7 GiB |
| Inference, batch 1 | GPU 3.4 ms p50; CPU (1 thread, 4 parallel evaluation workers) 8.6–9.3 ms p50, 12.6–14.0 ms p95 |

### Open loop (`test_id`, 17,137 samples; mean ± sd over 3 seeds)

| Predictor | ADE (m) | FDE (m) |
|---|---|---|
| Stand still | 4.399 | 8.026 |
| Constant velocity, straight | 0.768 | 1.963 |
| **Model A** | **0.162 ± 0.002** | **0.409 ± 0.006** |

By command (ADE, m): LANE_FOLLOW 0.192, LEFT 0.169, RIGHT 0.109, STRAIGHT 0.125. Moving
frames: 0.199. Stationary frames: 0.040.
Validation ADE at the selected epoch: 0.157 / 0.161 / 0.161.

### Closed loop (`test_id`, 50 scenarios)

| Metric | Expert | Model A (mean ± sd, 3 seeds) |
|---|---|---|
| Route completion | 1.000 | 0.413 ± 0.011 |
| Success rate (route completed) | 1.00 | 0.087 ± 0.023 |
| Episodes with a collision | 0.00 | 0.180 ± 0.020 |
| Driving score (secondary) | 1.000 | 0.323 ± 0.014 |
| Collisions / km | 0 | 1.39 ± 0.18 |
| Red-light violations / km | 0 | 4.05 ± 0.13 |
| Off-road / km | 0 | 2.41 ± 0.12 |
| Lane departures / km | 0 | 8.78 ± 0.79 |
| Route deviations / km | 0 | 3.28 ± 0.12 |

Paired difference in route completion, model A − expert (per-seed bootstrap 95% CI over the 50
scenarios): −0.578 [−0.649, −0.502], −0.599 [−0.662, −0.532], −0.584 [−0.658, −0.504].

Terminal outcomes pooled over the 3 seeds (150 episodes): 13 route complete, 63 route
deviation, 47 off-road, 27 collisions (22 with vehicles, 5 with pedestrians).

![Closed-loop results](assets/closed_loop_model_a_light.png)

### Diagnosis

1. **The learned policy fails, not the pipeline.** An oracle feeds privileged route waypoints
   through the *same* target-speed rule and controller. On the 30 val scenarios it completes
   30/30 with |lateral error| ≤ 0.44 m.
2. **Failures concentrate at junctions.** 76% of failure-terminated episodes (104/137) end on a
   junction connector: LEFT 40, STRAIGHT 37, RIGHT 21, LANE_FOLLOW 6. Rendered rollouts show
   the cause: inside the large, unmarked junction discs the camera sees nearly uniform
   asphalt. The policy predicts a curved path even under STRAIGHT and drifts out of the route.
   Once off the expert's line, nothing in the input pulls it back. This is the compounding-error
   failure of behaviour cloning, amplified by a stylised junction design in the simulator.
3. **Signals are rarely obeyed** (about 4 red-light violations per km). Signal heads are about 3×5 px at
   160×96 (a limitation flagged in Phase 2), and a single frame gives no cue about the phase.
4. **Open-loop error does not predict closed-loop performance.** ADE is 4.7× better than
   constant velocity, yet only 8.7% of routes are completed. Seeds agree closely, so this is
   systematic, not seed noise.

### Assessment against the Phase 4 acceptance criterion

"The learned model can complete basic driving routes in closed loop, and its performance is
measured against a baseline." **Partially met.** The model completes some routes (13/150) and
41% of route distance on average, and is measured against the expert on paired scenarios. But
it is not a competent driver. Under the protocol this is reported as-is.

### Implications for later phases

- For RQ2 (temporal model), model A is a weak but well-measured baseline. A temporal model
  could plausibly help with signal phase and with recovery inside junctions.
- For introspection (Phase 6), model A fails often (about 1.4 collisions/km plus frequent route
  deviations). That gives many failure events, but a policy this weak makes "predicting
  failure" partly trivial (e.g. "at a junction → risky"). The heuristic baselines in the
  protocol (§6) must include such context cues.
- Options that change the *benchmark* or the *training method* (junction markings, DAgger,
  route target points) are listed in the README's open decisions and require sign-off before
  use. None has been applied.

## Phase 5 — Model B (temporal) vs. model A

Summary file: [`docs/results/phase5_temporal.json`](results/phase5_temporal.json)
(generated by `scripts/summarize_phase5.py` from closed-loop run
`20261102-124917_closed_loop_test_id` and the training runs).

### Design (fixed before results)

| Item | Value |
|---|---|
| Model B | Model A's compact CNN encoder, **frozen** → per-frame 256-d features → GRU(256) over H frames (oldest first) → MLP head that also gets [current speed, command]. 1.55 M parameters, 0.48 M trainable |
| Primary history | H = 4 frames at 10 Hz (0.3 s span), the protocol default. Speed history **not** fed (copycat risk) |
| Control | **B (H=1)**: identical architecture and training recipe without history. Separates "temporal context" from "retrained head" |
| Pairing | B seed k and B(H=1) seed k use model A seed k's encoder. All agents drive the same scenarios |
| Training | Cached features (computed once per split, eval mode). 30 epochs, AdamW 1e-3, cosine, batch 256, same stationary weighting as A, no photometric jitter (features are fixed) |
| Selection | Best val ADE |
| Test set | `test_id`, **150 routes** (10 maps × 15), fixed after the Phase 4 pilot. Phase 4's 50 scenarios are a subset |

### Closed loop (`test_id`, 150 routes; mean ± sd over 3 seeds)

| Metric | Expert | Model A | Model B (H=1) | **Model B (H=4)** |
|---|---|---|---|---|
| Route completion | 1.000 | 0.431 ± 0.002 | 0.473 ± 0.016 | 0.463 ± 0.026 |
| Success rate | 0.993 | 0.113 ± 0.018 | 0.151 ± 0.010 | 0.138 ± 0.025 |
| Episodes with a collision | 0.000 | 0.209 ± 0.010 | 0.236 ± 0.020 | 0.233 ± 0.012 |
| Driving score | — | 0.346 ± 0.006 | 0.362 ± 0.010 | 0.361 ± 0.019 |
| Collisions / km | 0 | 1.55 ± 0.07 | 1.59 ± 0.09 | 1.62 ± 0.17 |
| Red-light violations / km | 0 | 3.53 ± 0.15 | 3.68 ± 0.40 | 3.75 ± 0.18 |
| Off-road / km | 0 | 2.08 ± 0.10 | 1.92 ± 0.31 | 1.93 ± 0.35 |
| Route deviations / km | 0 | 2.97 ± 0.10 | 2.24 ± 0.15 | 2.43 ± 0.29 |
| Lane departures / km | 0 | 8.08 ± 0.66 | 8.43 ± 0.70 | 8.34 ± 0.20 |
| CPU latency p50 (6 parallel workers) | — | 11.5 ms | 12.4 ms | 15.0 ms |

The expert's single non-completion over 150 routes is a timeout (success 0.993, completion
0.9999).

**Paired differences** (per scenario, bootstrap 95% CI; "pooled" concatenates the 3 seed pairs
and ignores seed clustering, so per-seed CIs are given as well):

| Comparison | Route completion, pooled | per seed | Collision episodes, pooled | Red lights per episode, pooled |
|---|---|---|---|---|
| **B(H=4) − B(H=1)** (effect of history) | **−0.010 [−0.026, 0.006]** | −0.033 [−0.059, −0.007], +0.009 [−0.026, 0.045], −0.005 [−0.024, 0.013] | −0.002 [−0.027, 0.022] | −0.002 [−0.036, 0.031] |
| B(H=1) − A (effect of retrained head) | +0.042 [0.023, 0.061] | +0.046, +0.055, +0.024 (lower bounds 0.014, 0.015, −0.000) | +0.027 [−0.004, 0.058] | +0.069 [0.036, 0.104] |
| B(H=4) − A | +0.032 [0.014, 0.050] | +0.014, +0.064, +0.018 | +0.024 [−0.004, 0.051] | +0.067 [0.036, 0.100] |

Per-episode red-light counts rise with route completion (more junctions are reached). The
per-km rates are similar across models.

![Closed-loop results, Phase 5](assets/closed_loop_phase5_light.png)

### Open loop (`test_id`, 17,137 samples; mean ± sd over 3 seeds)

| ADE (m) | All | Start from rest (n=543) | Stopping (n=344) | Cruising (n=12,842) | Holding (n=3,408) |
|---|---|---|---|---|---|
| Constant velocity | 0.768 | 0.485 | 1.147 | 0.972 | 0.007 |
| Model A | 0.162 ± 0.002 | 0.287 ± 0.029 | 0.077 ± 0.010 | 0.192 ± 0.001 | 0.036 ± 0.014 |
| Model B (H=1) | 0.162 ± 0.003 | 0.266 ± 0.034 | 0.073 ± 0.005 | 0.190 ± 0.001 | 0.050 ± 0.015 |
| Model B (H=4) | 0.163 ± 0.004 | 0.254 ± 0.037 | 0.073 ± 0.009 | 0.188 ± 0.002 | 0.061 ± 0.019 |

Situations are defined in `data/labels.py::motion_situations` from the current speed and the 2 s
label.

![Open-loop ADE by situation](assets/situations_phase5_light.png)

### History ablations (val ADE, seed 0, frozen encoder unless stated)

| Variant | Val ADE (m) |
|---|---|
| Model A (reference) | 0.157 |
| B, H=1 | 0.157 |
| B, H=2 | 0.155 |
| **B, H=4 (primary)** | 0.154 |
| B, H=8 | 0.155 |
| B, H=4, stride 2 (0.6 s span) | 0.155 |
| B, H=4 + speed history | **0.149** |
| B, H=4, end-to-end (encoder initialised from A and fine-tuned, 12 epochs, 55 min) | 0.160 (epoch 4; val is noisy early, 0.163 at the last epoch) |

The speed-history gain comes mostly from *cruising* frames (0.169 vs 0.177 m), not from the
situations that need temporal reasoning (start from rest 0.335 vs 0.334). That is consistent
with an inertia shortcut, so it was not adopted as the primary model.

![History ablation](assets/history_ablation_light.png)

### Cost

| | Model A | Model B (H=4) |
|---|---|---|
| Training | 761–796 s (end to end, GPU) | 52–64 s + 43 s one-off feature caching per encoder |
| Peak GPU memory (training) | 744 MiB | 233 MiB |
| GPU inference, batch 1 (p50) | 3.4 ms | 3.4 ms (incremental: encode newest frame + GRU); 3.3 ms full 4-frame forward |
| Parameters (trainable) | 1.15 M (1.15 M) | 1.55 M (0.48 M) |

### Findings

1. **H1 (temporal representations improve closed-loop driving) is not supported in this
   setup.** Against its proper control (same frozen features, same head training, no
   history), B(H=4) changes neither route completion (pooled −0.010 [−0.026, 0.006]) nor
   collisions nor red-light violations. The history ablation (H = 1…8) moves val ADE by at most
   0.003 m.
2. **The apparent B > A gain is a training-recipe effect, not a temporal one.** B(H=1) beats A
   by +0.042 route completion, the same as B(H=4). Without the H=1 control this would have
   been misread as evidence for H1.
3. **Where history might help, the effect is small and uncertain.** Start-from-rest ADE goes
   0.287 → 0.266 → 0.254 m (A → B(H=1) → B(H=4)), which is within one seed sd. Holding gets
   slightly worse with history (0.036 → 0.061 m): the model occasionally predicts creeping
   forward while stopped.
4. **Failure modes are unchanged.** 76–79% of failure-terminated episodes still end at junctions
   (A 308/399, B(H=1) 291/382, B(H=4) 307/388), and the
   red-light violation rate stays near 3.5–3.8 / km.
5. **Scope.** The encoder was trained single-frame and then frozen, and the history is short
   (0.3 s by default, up to 0.7 s). The negative result applies to *temporal aggregation of
   single-frame features*. It is not a statement about temporal models in general.
6. **Freezing the encoder is not the bottleneck** (single seed). Fine-tuning the encoder
   jointly with the GRU (H=4, end to end) reaches val ADE 0.160 / test ADE 0.167. On the same
   150 routes it completes 0.436 of the route distance, vs 0.447 for frozen B seed 0 (paired
   −0.012 [−0.054, 0.032]) and 0.433 for A seed 0 (+0.002 [−0.039, 0.044]). Its training loss
   keeps falling (0.086) while val ADE plateaus, i.e. mild overfitting. It costs 55 min vs about 1
   min for the cached-feature model (run `20261102-142523_closed_loop_test_id`, summarised
   under `ablation_closed_loop` in the summary file).

### Assessment against the Phase 5 acceptance criterion

"The temporal model has been trained and evaluated under the same conditions as the
single-frame baseline." **Met.** Same data, the same 150 paired test routes, 3 seeds, the same
controller and metrics, plus a control that isolates the temporal effect. Training and inference
overhead were measured. Prediction failures were analysed by motion situation and by failure
location.

## Phase 6 — Failure prediction (self-introspection)

Summary file: [`docs/results/phase6_introspection.json`](results/phase6_introspection.json)
(run `20261102-161901_introspection`, generated by `scripts/run_introspection.py` and
`scripts/analyze_introspection.py`).

### Design (fixed before results)

| Item | Value |
|---|---|
| Policy under study | Model B (H=4), seeds 0–2. **C = B + introspection head**, so C drives exactly like B |
| Label | y_t = 1 iff a collision, off-road, route deviation, red-light violation or stall occurs in (t, t + 2 s]. Lane departure is excluded (ablation below) |
| Head inputs (non-privileged, time ≤ t) | Detached GRU state (256), speed, command, predicted waypoints (20) |
| Head | MLP 284 → 128 → 64 → 1, class-weighted BCE, up to 40 epochs; the epoch with the best intro_val PR-AUC is kept |
| Calibration | Platt scaling (scale + intercept) on intro_cal, applied to **every** predictor |
| Alarm threshold | Fixed on intro_cal at a 10% frame-level false-positive rate; an alarm needs 2 consecutive steps above it |
| Non-privileged baselines | Logistic models on: speed; **context** (speed + command, i.e. "near a junction"); novelty (Mahalanobis distance of encoder features to the expert data); prediction inconsistency between steps; all heuristics combined (+ predicted turn) |
| Privileged reference | Logistic model on time-to-collision along the route, gap, lateral offset, junction flag and speed. **Not deployable**; a simple privileged reference, not an upper bound |
| Splits (by map) | intro_train 4000–4015 (160 episodes), intro_val 4016–4019 (40), intro_cal 5000–5005 (60), test = the 150 test_id routes from Phase 5 |

Ensemble-disagreement baselines (protocol §6, item 2) need Phase 7's uncertainty models and
are deferred to Phase 7.

**Determinism check.** For every seed, the 150 recorded test rollouts reproduce the Phase 5
closed-loop episodes of model B exactly (maximum completion difference 0.0). The
introspection test set is literally the driving test set.

**Causality.** Recording the same scenario with a shorter time limit gives identical
introspection inputs for the shared prefix (`tests/introspection/test_rollouts.py`).

### Data (per seed; test_id)

About 47–50k decision steps per seed, positive rate 8.4–9.0%, and 208–214 failure events in
135–136 of 150 episodes. Events per seed: collision 33–36, off-road 37–49, route deviation
47–58, red light 78–83, stall 0.

### Test results (mean ± sd over 3 policy seeds)

| Predictor | AUROC | PR-AUC | Brier | ECE | Events detected | Median lead time | False alarms / km |
|---|---|---|---|---|---|---|---|
| **Introspection (C)** | **0.971 ± 0.003** | **0.796 ± 0.012** | **0.033** | **0.011** | **98.7%** | 2.1 s | 19.2 |
| All heuristics | 0.828 ± 0.021 | 0.307 ± 0.018 | 0.069 | 0.020 | 70.3% | 1.0 s | 19.2 |
| Context (speed + command) | 0.780 ± 0.013 | 0.194 ± 0.020 | 0.073 | 0.036 | 36.8% | 4.5 s | 4.6 |
| Novelty | 0.745 ± 0.048 | 0.220 ± 0.032 | 0.073 | 0.018 | 76.6% | 0.6 s | 17.0 |
| Inconsistency | 0.668 ± 0.022 | 0.144 ± 0.015 | 0.078 | 0.039 | 44.5% | 0.6 s | 39.5 |
| Speed | 0.559 ± 0.023 | 0.087 ± 0.001 | 0.079 | 0.061 | 3.2% | — | 9.6 |
| Oracle (privileged) | 0.865 ± 0.015 | 0.364 ± 0.011 | 0.067 | 0.035 | 59.7% | 1.9 s | 9.0 |

Introspection AUROC 95% CIs (episode bootstrap): [0.962, 0.978], [0.967, 0.980],
[0.960, 0.977]. Prevalence is 0.084–0.090, which is the PR-AUC of a random predictor.

**H2 (introspection beats the best non-privileged heuristic).** The best heuristic is "all
heuristics" for every seed. Paired differences, episode bootstrap 95% CI:

| Seed | ΔAUROC (C − heuristics) | ΔPR-AUC |
|---|---|---|
| 0 | +0.166 [0.145, 0.189] | +0.509 [0.451, 0.558] |
| 1 | +0.138 [0.119, 0.160] | +0.485 [0.432, 0.535] |
| 2 | +0.124 [0.108, 0.141] | +0.473 [0.415, 0.522] |

![ROC and PR curves](assets/introspection_roc_pr_light.png)
![Metrics across seeds](assets/introspection_metrics_light.png)

### Anticipation vs. recognition

Some events register only once the situation is visibly bad (route deviation > 5 m, footprint
off the road), so a predictor could score well by *recognising* an ongoing failure. The analysis
stratifies with privileged state, used for analysis only:

| Stratum (share of all positives) | Introspection AUROC / PR-AUC | All heuristics | Context | Oracle |
|---|---|---|---|---|
| All frames (100%) | 0.971 / 0.796 | 0.828 / 0.307 | 0.780 / 0.194 | 0.865 / 0.364 |
| Nominal frames, \|lateral\| < 1 m (55%) | 0.975 / 0.748 | 0.801 / 0.156 | 0.774 / 0.126 | 0.862 / 0.224 |
| **Anticipation**: positives 1–2 s ahead and nominal, vs nominal negatives (32%) | **0.972 / 0.570** | 0.752 / 0.067 | 0.769 / 0.078 | 0.824 / 0.106 |

By failure type (AUROC, mean of 3 seeds): collision 0.890 (heuristics 0.739, oracle 0.808);
red light 0.953 (0.750 / 0.806); off-road 0.938 (0.826 / 0.812); route deviation 0.959
(0.923 / 0.945).

### Calibration and warnings

- **Calibration:** the weighted-BCE head is miscalibrated as expected (ECE 0.037–0.076).
  Platt scaling on intro_cal brings test ECE to 0.006–0.014.
- **Warnings:** at the intro_cal threshold, 98.7% of failure events have an active alarm within
  the 2 s before them. The median lead time is 2.1 s; it peaks near the 2 s training horizon,
  and persistent alarms reach up to about 6 s. The price is about 19 false alarm onsets per km at
  this operating point. Phase 8 must choose thresholds by the cost of interventions, not by
  this fixed FPR.

![Calibration](assets/introspection_reliability_light.png)
![Lead time](assets/introspection_lead_time_light.png)

### Ablations (introspection head, mean of 3 seeds; context heuristic for comparison)

| Label | Prevalence | AUROC | PR-AUC | Context AUROC |
|---|---|---|---|---|
| Failure within 1 s | 0.043 | 0.979 | 0.735 | 0.763 |
| **Failure within 2 s (primary)** | 0.087 | 0.971 | 0.796 | 0.780 |
| Failure within 3 s | 0.130 | 0.958 | 0.805 | 0.802 |
| Within 2 s, lane departure counted as failure | 0.152 | 0.970 | 0.849 | 0.768 |

### Findings

1. **RQ3 / H2 supported in this setting.** A small head on the frozen policy's own GRU state
   and predictions anticipates failures far better than every non-privileged heuristic and
   than the simple privileged reference. AUROC is 0.971 vs 0.828, and PR-AUC 0.80 vs 0.31.
   This holds for every seed with tight CIs, and it holds in the anticipation stratum.
2. **It is not just "junctions are risky".** The context heuristic (speed + command)
   captures the junction prior (AUROC 0.78). The learned head adds most of its value in the
   cases the prior cannot separate: PR-AUC 0.57 vs 0.08 in the anticipation stratum.
3. **Why it works so well here, and why that is a limitation.** Model B fails often and
   systematically: 135 of 150 test episodes contain a failure, and red-light runs are common
   because it rarely stops. Its internal state therefore carries much of the information
   needed to predict its own failures (e.g. approaching a stop line without slowing; drifting
   inside a junction). With a more competent policy, failures will be rarer and less
   systematic, and these numbers should not be expected to transfer. H5 (degradation under
   shift) is tested in Phase 9.
4. **Calibration requires a held-out step.** Raw head probabilities are not calibrated.
   Calibration on intro_cal transfers to test, so the probabilities can be used for
   thresholding (H3 is tested in Phase 8).
5. **High recall comes with frequent alarms.** At the 10%-FPR operating point there are about
   19 alarm onsets per km. Whether acting on them helps more than it costs (RQ5) is Phase 8's
   question.

### Assessment against the Phase 6 acceptance criterion

"The introspection model produces evaluated future-failure probabilities without using future
information as input." **Met.**
- **Inputs:** strictly causal, with a test.
- **Labels:** strictly future.
- **Evaluation:** AUROC, PR-AUC, Brier, ECE, reliability, lead time and false alarms, against
  four heuristics plus a privileged reference, with 3 seeds and episode-level CIs.
