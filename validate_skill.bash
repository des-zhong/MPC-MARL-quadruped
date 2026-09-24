#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

# ===== Paths to edit for a new run =====
WALK_POLICY_DIR="checkpoints/reproduction/walk"
DRIBBLE_POLICY_DIR="checkpoints/reproduction/dribble"
SHOOT_POLICY_DIR="checkpoints/reproduction/shoot"
OUTPUT_DIR="outputs/ability_validation"
DEVICE="${DEVICE:-cuda:0}"
POLICY_DEVICE="${POLICY_DEVICE:-${DEVICE}}"
# =======================================
		
exec "${PYTHON_BIN}" scripts/validate_skill.py \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --policy-device "${POLICY_DEVICE}"
