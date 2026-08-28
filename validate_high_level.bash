#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

# ===== Paths to edit for a new run =====
HIGH_LEVEL_POLICY_DIR="wandb/run-20260825_101144-zpkeqtxq/files/tmp/legged_data/high_level"
# HIGH_LEVEL_POLICY_DIR="wandb/run-20260825_211452-008kqv9p/files/tmp/legged_data/high_level_online_mpc"
OPPONENT_HIGH_LEVEL_POLICY_DIR="${HIGH_LEVEL_POLICY_DIR}"
WALK_POLICY_DIR="checkpoints/reproduction/walk"
DRIBBLE_POLICY_DIR="checkpoints/reproduction/dribble"
SHOOT_POLICY_DIR="checkpoints/reproduction/shoot"
VIDEO_PATH="outputs/high_level_eval.mp4"
PLOT_PATH="outputs/high_level_eval_metrics.png"
CSV_PATH="outputs/high_level_eval_metrics.csv"
SEED="${SEED:-0}"
# =======================================

HIGH_LEVEL_CHECKPOINT="latest"
OPPONENT_HIGH_LEVEL_CHECKPOINT="0"

"${PYTHON_BIN}" scripts/play_high_level.py \
  --num-robots 2 \
  --high-level-policy-source local \
  --high-level-policy-dir "${HIGH_LEVEL_POLICY_DIR}" \
  --high-level-checkpoint "${HIGH_LEVEL_CHECKPOINT}" \
  --opponent-high-level-policy-dir "${OPPONENT_HIGH_LEVEL_POLICY_DIR}" \
  --opponent-high-level-checkpoint "${OPPONENT_HIGH_LEVEL_CHECKPOINT}" \
  --skill-policy-source local \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --video "${VIDEO_PATH}" \
  --plot "${PLOT_PATH}" \
  --csv "${CSV_PATH}" \
  --seed "${SEED}" \
  --headless \
  "$@"
