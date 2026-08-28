#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

# The world-model checkpoint is only an initialization.  It is updated online
# from the bounded replay buffer and is never reset when self-play opponents
# rotate.  Point the three low-level folders at immutable Walk/Dribble/Shoot
# exports; this script never trains those policies.
WORLD_MODEL_CHECKPOINT="checkpoints/reproduction/world_model/best.pt"
HIGH_LEVEL_CHECKPOINT="checkpoints/reproduction/high_level/ac_weights_latest.pt"
TERMINAL_VALUE_CHECKPOINT="${TERMINAL_VALUE_CHECKPOINT:-checkpoints/terminal_value_env_v2_bootstrap/best.pt}"
WORLD_MODEL_CONFIG="configs/world_model_as2.yaml"
MPC_CONFIG="configs/mpc_joint_teams.yaml"
WALK_POLICY_DIR="checkpoints/reproduction/walk"
DRIBBLE_POLICY_DIR="checkpoints/reproduction/dribble"
SHOOT_POLICY_DIR="checkpoints/reproduction/shoot"
CHECKPOINT_DIR=""

OUTPUT_ARGS=()
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  OUTPUT_ARGS=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi

exec "${PYTHON_BIN}" scripts/train_high_level_online_mpc_full.py \
  --world-model-checkpoint "${WORLD_MODEL_CHECKPOINT}" \
  --world-model-config "${WORLD_MODEL_CONFIG}" \
  --mpc-config "${MPC_CONFIG}" \
  --mpc-profile teacher_training \
  --num-envs 32 \
  --num-robots 2 \
  --self-play-update-interval 2000 \
  --opponent-pool-size 8 \
  --opponent-latest-probability 0.5 \
  --world-model-update-interval 25 \
  --world-model-replay-buffer-size 100000 \
  --world-model-replay-recent-fraction 0.5 \
  --world-model-replay-recent-window 10000 \
  --mpc-horizon 2 \
  --mpc-num-samples 256 \
  --mpc-num-iterations 4 \
  --terminal-value-checkpoint "${TERMINAL_VALUE_CHECKPOINT}" \
  --mpc-kl-coefficient 0.05 \
  --mpc-guidance-reward-coefficient 1.0 \
  --mpc-warmup-steps 500 \
  --resume \
  --resume-mode full \
  --device cuda:7 \
  --resume-checkpoint "${HIGH_LEVEL_CHECKPOINT}" \
  --skill-policy-source local \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  "${OUTPUT_ARGS[@]}" \
  "$@"
