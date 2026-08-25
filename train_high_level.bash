#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

# ===== Paths to edit for a new run =====
WALK_POLICY_DIR="checkpoints/reproduction/walk"
DRIBBLE_POLICY_DIR="checkpoints/reproduction/dribble"
SHOOT_POLICY_DIR="checkpoints/reproduction/shoot"
# Warm-start from the strongest fixed-window checkpoint in the analyzed run.
# Policy-only mode intentionally resets the critic and continuous exploration.
RESUME_CHECKPOINT="wandb/run-20260824_155604-f3sipxl6/files/tmp/legged_data/high_level/ac_weights_latest.pt"
# Empty keeps the default W&B run checkpoint directory.
CHECKPOINT_DIR=""
# =======================================

OUTPUT_ARGS=()
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  OUTPUT_ARGS=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi

exec "${PYTHON_BIN}" scripts/train_high_level.py \
  --num-robots 2 \
  --self-play-update-interval 2000 \
  --opponent-pool-size 8 \
  --opponent-latest-probability 0.5 \
  --skill-entropy-coef 0.002 \
  --skill-entropy-final-coef 0.0002 \
  --skill-entropy-anneal-iterations 4000 \
  --skill-policy-source local \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --resume \
  --resume-mode policy-only \
  --resume-checkpoint "${RESUME_CHECKPOINT}" \
  "${OUTPUT_ARGS[@]}" \
  "$@"
