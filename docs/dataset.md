# Expert Dataset (`expert_v1`)

Status: Phase 3, collected 2026-11-01. Reproduce with
`python scripts/collect_dataset.py --config configs/data/expert_v1.yaml`. Collection is
deterministic given the config; `data/` is git-ignored.

## 1. Demonstrator

`RuleBasedExpert` (`src/driveaware/expert/expert.py`) is a privileged driver:
- **Lateral control:** pure pursuit on the route centreline.
- **Longitudinal control:** IDM towards the curvature-limited speed limit. The leader is the
  most constraining of:
  - vehicles in the lane corridor ahead;
  - pedestrians on the carriageway ahead;
  - a virtual stopped obstacle at a red stop line, or a yellow one when stopping is
    comfortable.
- **Controller:** the same one the learned policies use.

Evaluation on 100 held-out episodes (maps 200–209, 400 m routes, default traffic):
see §6.

## 2. Collection

| Setting | Value |
|---|---|
| Routes | ≥ 300 m random lane walks, `max_duration` 150 s |
| Traffic | 12 IDM vehicles, 6 parked cars, crosswalk pedestrians (mean 12 s between arrivals) |
| Seeds | route / traffic / noise seeds derived from `(dataset_seed, split, map_seed, k)` |
| Noise injection | 50% of episodes. Steering bursts of ±0.15–0.35 (normalised) lasting 0.5–1.2 s, with exponential gaps averaging 8 s |
| Recorded per frame | image, timestamp, speed, command, ego pose, steer angle, route progress / lateral offset, expert action, applied action, `perturbed` flag, expert target speed, events |

**Noise strength.** In a 24-episode probe, this setting produced lateral deviations up to
1.9 m and 1 noise-induced collision. A milder setting (±0.08–0.20, 0.3–0.8 s) gave no
collisions but only 0.76 m deviations. The stronger setting was kept so that recovery data
covers larger drifts; noise-induced failures are excluded from training (§4).

## 3. Labels

Frame t is the observation at time t. The action recorded at frame t produces frame t+1.

- **Target:** ego rear-axle positions at t + 0.2k s (k = 1..10), in the ego frame at t
  (x forward, y left, metres).
- **Validity:** the 2 s window exists within the episode, and none of the actions at frames
  t … t+19 contained injected noise. A frame may *start* in a perturbed state; these
  recovery samples are kept.
- **Inputs:** only frames ≤ t of the same episode. The history is padded by repeating the
  episode's first frame, with a mask.

## 4. Splits and contents

Splits are made by **map seed** (no shared roads). `intro_train` (4000–4999), `intro_cal`
(5000–5999) and OOD suites (6000+) are reserved for later phases.

| Split | Maps | Episodes (noisy) | Frames | Labelled | Used samples¹ | Recovery samples² | Outcomes |
|---|---|---|---|---|---|---|---|
| train | 1000–1039 | 120 (58) | 91,305 | 76,486 | 75,488 | 6,422 | 117 complete, 3 collision |
| val | 2000–2005 | 18 (10) | 14,190 | 11,029 | 11,029 | 1,475 | 18 complete |
| test_id | 3000–3009 | 30 (18) | 22,499 | 17,825 | 17,137 | 1,882 | 28 complete, 2 collision |

¹ Excluding episodes flagged `expert_failure`.
² Labelled frames within 2 s after injected noise.

**All 5 collisions happened in noisy episodes, 0.1–0.6 s after a noise burst**, so they are
noise-induced. No unperturbed expert episode failed. These episodes are kept on disk and
flagged, and excluded by `DrivingDataset` by default.

Training-sample composition (train): commands LANE_FOLLOW 48%, LEFT 16%, RIGHT 19%,
STRAIGHT 17%. Mean speed is 4.1 m/s and **25.5% of samples are stationary** (< 0.1 m/s),
mostly red-light waits under split phasing. Phase 4 must decide how to handle this (e.g.
subsampling or loss weighting) and report the choice.

## 5. Storage, integrity and cost

- **Per episode:** `episodes/<split>_<id>.npz` (compressed) + `.json` metadata (spec, seeds,
  outcome, events, noise flag).
- **Per split:** `images.npy` (uint8 memmap [N, 96, 160, 3]), `frames.npz` (aligned
  per-frame arrays including labels and `label_valid`) and `episodes.json`.
- **Validation** (`data/validation.py`), all passing:
  - array alignment;
  - uniform 0.1 s timestamps;
  - displacement vs. speed consistency;
  - action ranges, and applied == expert action on unperturbed frames;
  - `label_valid` consistent with the noise flags;
  - 500 labels per split re-derived from poses: max error 9.5e-7 m;
  - no label window crossing an episode boundary;
  - map seeds disjoint across splits.
- **Inspection:** `scripts/inspect_dataset.py <split_dir>` prints statistics and writes a
  montage with labels projected into the camera image (`montage.png` per split). On visual
  check the waypoints follow the lane centre and bend into commanded turns.

| Cost | Measured |
|---|---|
| Wall time | 8 min 36 s (4 workers, 168 episodes, 128k frames, about 260 frames/s) |
| Peak RSS of the collection process tree | 4.6 GB (`/usr/bin/time` maximum) |
| Disk | 6.7 GB in total: memmaps 5.9 GB (46 KB/frame raw) + compressed episodes 1.2 GB (about 9.5 KB/frame) |

The compressed episodes are the source of truth. The memmaps can be rebuilt with
`consolidate_split` and could be deleted to save disk.

## 6. Expert reference performance

`python scripts/eval_expert.py` (run `experiments/20261101-215300_eval_expert/`, which is
git-ignored; config and environment are stored with it). 100 episodes: maps 200–209 (outside
all dataset splits), 10 routes each, ≥ 400 m, default traffic, `max_duration` 180 s.

| Metric | Expert |
|---|---|
| Outcomes | 99 route complete, 1 timeout (map 201, route 8, 180 s) |
| Mean route completion | 0.9999 |
| Distance driven | 42.2 km |
| Collisions / red-light / off-road / lane departure / route deviation / stall per km | 0 / 0 / 0 / 0 / 0 / 0 |
| Mean speed | 3.94 m/s (split-phase waits dominate) |
| Wall time | 647 s for 100 episodes without rendering (one process) |

The expert is therefore a near-ceiling reference in this simulator. Learned-policy results
will be compared against it on the same scenario seeds. A single run on 100 episodes cannot
establish a failure rate below about 3% per episode (rule of three, 95%).

## 7. Known limitations

- The data is from a stylised simulator. Absolute driving numbers do not transfer.
- Labels come from a privileged, rule-based expert. Its behaviour (split-phase waits, IDM
  gaps) defines what "correct" means for behaviour cloning.
- Noise is lateral (steering) only. Speed perturbations are not injected.
- 25% stationary samples and the command imbalance may bias behaviour cloning towards stopping or
  lane following.
- The test_id split was opened only for data QA (statistics, montage), not for model
  evaluation.
