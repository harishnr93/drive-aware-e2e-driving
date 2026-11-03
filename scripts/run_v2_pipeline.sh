#!/usr/bin/env bash
# Re-run the full Phase 3-7 pipeline on benchmark v2 (docs/benchmark_v2.md), same protocol/seeds.
# Every stage is resumable: finished work is detected and skipped, and incomplete training
# runs (killed before writing result.json) are removed before retraining.
#
# Usage (detached, survives logout):
#   setsid nohup bash scripts/run_v2_pipeline.sh > experiments/v2_pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
stage() { echo "=== $(date +%T) $*"; }

# A JSON file that exists and parses (a crash or reboot can leave empty/partial files).
valid_json() { [ -s "$1" ] && python -c 'import json,sys; json.load(open(sys.argv[1]))' "$1" 2>/dev/null; }
# Latest *complete* run directory for a run name (has a valid result.json).
run_dir() {
  local d
  for d in $(ls -d experiments/*_"$1" 2>/dev/null | sort -r); do
    valid_json "$d/result.json" && { echo "$d"; return 0; }
  done
  return 1
}
# train NAME_WITH_SEED CONFIG ARGS... : skip if complete, else clear partial runs and train.
train() {
  local name=$1 config=$2; shift 2
  if run_dir "$name" >/dev/null; then echo "skip $name"; return 0; fi
  rm -rf experiments/*_"$name"
  python scripts/train_bc.py --config "$config" --set "$@" >/dev/null
  run_dir "$name" >/dev/null || { echo "training $name failed" >&2; exit 1; }
}
A() { echo "$(run_dir "model_a_v2_seed$1")/best.pt"; }

stage "1/8 collect expert_v2"
python scripts/collect_dataset.py --config configs/data/expert_v2.yaml

stage "2/8 model A v2 (3 seeds)"
for s in 0 1 2; do
  train "model_a_v2_seed$s" configs/training/model_a.yaml \
    seed=$s data_root=data/expert_v2 name=model_a_v2
done

stage "3/8 model B v2 + H=1 control (3 seeds)"
for s in 0 1 2; do
  train "model_b_h4_v2_seed$s" configs/training/model_b.yaml \
    seed=$s data_root=data/expert_v2 encoder_checkpoint="$(A $s)" name=model_b_h4_v2
  train "model_b_h1_v2_seed$s" configs/training/model_b.yaml \
    seed=$s data_root=data/expert_v2 encoder_checkpoint="$(A $s)" name=model_b_h1_v2 history=1
done

stage "4/8 closed loop v2 (150 test_id routes)"
REPORT=""
for d in $(ls -d experiments/*_closed_loop_test_id_v2 2>/dev/null | sort -r); do
  valid_json "$d/report.json" && { REPORT="$d/report.json"; break; }
done
if [ -z "$REPORT" ]; then
  RESUME=()
  PARTIAL=$(ls -d experiments/*_closed_loop_test_id_v2 2>/dev/null | sort | tail -1 || true)
  [ -n "$PARTIAL" ] && RESUME=(--resume-dir "$PARTIAL") && echo "resume closed loop ($PARTIAL)"
  ckpts=()
  for s in 0 1 2; do ckpts+=("$(A $s)"); done
  for s in 0 1 2; do ckpts+=("$(run_dir model_b_h1_v2_seed$s)/best.pt"); done
  for s in 0 1 2; do ckpts+=("$(run_dir model_b_h4_v2_seed$s)/best.pt"); done
  python scripts/eval_closed_loop.py --config configs/evaluation/closed_loop_v2.yaml \
    --set-name test_id --expert --workers 6 --data-root data/expert_v2 "${RESUME[@]}" \
    --checkpoints "${ckpts[@]}" > /dev/null
  REPORT=$(ls -d experiments/*_closed_loop_test_id_v2 | sort | tail -1)/report.json
else
  echo "skip closed loop ($REPORT)"
fi

stage "5/8 summarise temporal comparison"
python scripts/summarize_phase5.py --report "$REPORT" --suffix _v2 --data-root data/expert_v2 \
  --out docs/results/phase5_temporal_v2.json

stage "6/8 introspection v2"
INTRO=""
for d in $(ls -d experiments/*_introspection_v2 2>/dev/null | sort -r); do
  valid_json "$d/results.json" && { INTRO="$d"; break; }
done
if [ -z "$INTRO" ]; then
  rm -rf experiments/*_introspection_v2   # rollouts under data/introspection are reused
  python scripts/run_introspection.py --seeds 0 1 2 --workers 6 \
    --eval-config configs/evaluation/closed_loop_v2.yaml --suffix _v2
  INTRO=$(ls -d experiments/*_introspection_v2 | sort | tail -1)
else
  echo "skip introspection ($INTRO)"
fi
python scripts/analyze_introspection.py --run-dir "$INTRO" > /dev/null

stage "7/8 ensemble members v2 (15)"
for k in 0 1 2; do for m in 0 1 2 3 4; do
  train "model_bprob_e${m}_k${k}_v2_seed$((10*k+m))" configs/training/model_b_prob.yaml \
    seed=$((10*k+m)) data_root=data/expert_v2 encoder_checkpoint="$(A $k)" \
    name=model_bprob_e${m}_k${k}_v2
done; done

stage "8/8 uncertainty v2"
UNC=""
for d in $(ls -d experiments/*_uncertainty_v2 2>/dev/null | sort -r); do
  valid_json "$d/results.json" && { UNC="$d"; break; }
done
if [ -z "$UNC" ]; then
  rm -rf experiments/*_uncertainty_v2
  python scripts/run_uncertainty.py --seeds 0 1 2 --suffix _v2 --data-root data/expert_v2
else
  echo "skip uncertainty ($UNC)"
fi

stage "done"
