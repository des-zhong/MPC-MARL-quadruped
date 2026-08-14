#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

exec "${PYTHON_BIN}" scripts/validate_robot_abilities.py \
  --ability all \
  --skill-policy-source local \
  --walk-policy-dir checkpoints/reproduction/walk \
  --dribble-policy-dir checkpoints/reproduction/dribble \
  --shoot-policy-dir checkpoints/reproduction/shoot \
  --headless \
  "$@"
