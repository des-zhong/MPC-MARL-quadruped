#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

# ===== Paths to edit for a new run =====
WORLD_MODEL_CHECKPOINT="checkpoints/reproduction/world_model/best.pt"
MPC_CONFIG="configs/mpc_joint_teams.yaml"
WALK_POLICY_DIR="checkpoints/reproduction/walk"
DRIBBLE_POLICY_DIR="checkpoints/reproduction/dribble"
SHOOT_POLICY_DIR="checkpoints/reproduction/shoot"
# Empty keeps the default W&B run checkpoint directory.
CHECKPOINT_DIR=""
# =======================================

OUTPUT_ARGS=()
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  OUTPUT_ARGS=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi

exec "${PYTHON_BIN}" scripts/train_high_level_with_mpc_teacher.py \
  --world-model-checkpoint "${WORLD_MODEL_CHECKPOINT}" \
  --mpc-config "${MPC_CONFIG}" \
  --mpc-profile teacher_training \
  --num-robots 2 \
  --self-play-update-interval 2000 \
  --skill-policy-source local \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  "${OUTPUT_ARGS[@]}" \
  "$@"
