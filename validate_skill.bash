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
# =======================================
		
exec "${PYTHON_BIN}" scripts/validate_robot_abilities.py \
  --ability dribble \
  --skill-policy-source local \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --walk-x 1.1811809539794922 \
  --walk-y 0.2554563879966736 \
  --walk-yaw 0.3178353011608124 \
  --headless \
  "$@"
