#!/usr/bin/env bash
set -euo pipefail


PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

# The evaluation profile lives in scripts/validate_high_level.py. Only
# checkpoint locations, CUDA, output paths, and seed are configurable here.
# Arguments: high-level-dir walk-dir dribble-dir shoot-dir opponent-pool.pt
#            cuda-index video-output plot-output csv-output seed
HIGH_LEVEL_POLICY_DIR="${1:-wandb/run-20260920_145323-1s2mw9rb/files/tmp/legged_data/high_level}"
WALK_POLICY_DIR="${2:-checkpoints/reproduction/walk}"
DRIBBLE_POLICY_DIR="${3:-checkpoints/reproduction/dribble}"
SHOOT_POLICY_DIR="${4:-checkpoints/reproduction/shoot}"
OPPONENT_CHECKPOINT="${5:-wandb/run-20260920_145323-1s2mw9rb/files/tmp/legged_data/high_level/opponent_pool_latest.pt}"
CUDA_NUM="${6:-4}"
VIDEO_PATH="${7:-outputs/high_level_eval.mp4}"
PLOT_PATH="${8:-outputs/high_level_eval_metrics.png}"
CSV_PATH="${9:-outputs/high_level_eval_metrics.csv}"
SEED="${10:-10}"

# FAST=1 skips rendering, plotting, and detailed state diagnostics.
FAST="${FAST:-0}"
EXTRA_ARGS=()
if [[ "${FAST}" == "1" ]]; then
  EXTRA_ARGS+=(--no-video --no-plot --outcomes-only)
fi

"${PYTHON_BIN}" scripts/validate_high_level.py \
  --high-level-policy-dir "${HIGH_LEVEL_POLICY_DIR}" \
  --opponent-pool-checkpoint "${OPPONENT_CHECKPOINT}" \
  --video "${VIDEO_PATH}" \
  --plot "${PLOT_PATH}" \
  --csv "${CSV_PATH}" \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --cuda "${CUDA_NUM}" \
  --seed "${SEED}" \
  --random-init \
  "${EXTRA_ARGS[@]}"
