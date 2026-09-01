#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

# ===== Paths to edit for a new run =====
# HIGH_LEVEL_POLICY_DIR="wandb/run-20260830_153222-1viubo0h/files/tmp/legged_data/high_level_mpc_replay"
HIGH_LEVEL_POLICY_DIR="wandb/run-20260830_153204-xjhoe0b9/files/tmp/legged_data/high_level"
OPPONENT_HIGH_LEVEL_POLICY_DIR="${HIGH_LEVEL_POLICY_DIR}"
WALK_POLICY_DIR="checkpoints/reproduction/walk"
# Dribble/shoot must be replaced with newly retrained checkpoints whose
# config.yaml contains Cfg.commands.ball_xy_frame: body. The bundled
# reproduction checkpoints are legacy world-frame policies and are rejected.
DRIBBLE_POLICY_DIR="checkpoints/reproduction/dribble"
SHOOT_POLICY_DIR="checkpoints/reproduction/shoot"
VIDEO_PATH="outputs/high_level_eval.mp4"
PLOT_PATH="outputs/high_level_eval_metrics.png"
CSV_PATH="outputs/high_level_eval_metrics.csv"
SEED="${SEED:-5}"
# Deployment-only emergency guard. Set ENABLE_COLLISION_AVOIDANCE=0 to
# evaluate the raw learned policy; the CSV records every enabled override.
ENABLE_COLLISION_AVOIDANCE="${ENABLE_COLLISION_AVOIDANCE:-1}"
# =======================================

HIGH_LEVEL_CHECKPOINT="latest"
OPPONENT_HIGH_LEVEL_CHECKPOINT="latest"

COLLISION_ARGS=()
if [[ "${ENABLE_COLLISION_AVOIDANCE}" == "1" ]]; then
  COLLISION_ARGS=(
    --collision-avoidance
    --collision-avoidance-distance "${COLLISION_AVOIDANCE_DISTANCE:-0.55}"
    --collision-avoidance-lookahead "${COLLISION_AVOIDANCE_LOOKAHEAD:-0.25}"
    --collision-avoidance-speed "${COLLISION_AVOIDANCE_SPEED:-0.5}"
  )
fi

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
  --device cuda:4 \
  --policy-device cuda:4 \
  --video "${VIDEO_PATH}" \
  --plot "${PLOT_PATH}" \
  --csv "${CSV_PATH}" \
  --seed "${SEED}" \
  "${COLLISION_ARGS[@]}" \
  --headless \
  "$@"
