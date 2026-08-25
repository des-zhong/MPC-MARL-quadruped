#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dribblebot_matplotlib}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

# ===== Paths to edit for a new run =====
MPC_CONFIG="configs/mpc_joint_teams.yaml"
WORLD_MODEL_CHECKPOINT="outputs/iterative_mpc_world_model_finetune/checkpoints/world_model_iteration_002/best.pt"
TERMINAL_VALUE_CHECKPOINT="checkpoints/terminal_value_mpc/best.pt"
WALK_POLICY_DIR="checkpoints/reproduction/walk"
DRIBBLE_POLICY_DIR="checkpoints/reproduction/dribble"
SHOOT_POLICY_DIR="checkpoints/reproduction/shoot"
OUTPUT_DIR="outputs/mpc_visualizations"
# =======================================

exec "${PYTHON_BIN}" scripts/visualize_mpc.py \
  --config "${MPC_CONFIG}" \
  --profile teacher_high_quality \
  --skill-policy-source local \
  --num-robots 2 \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --world-model-checkpoint "${WORLD_MODEL_CHECKPOINT}" \
  --terminal-value-checkpoint "${TERMINAL_VALUE_CHECKPOINT}" \
  --save-dir "${OUTPUT_DIR}" \
  "$@"
