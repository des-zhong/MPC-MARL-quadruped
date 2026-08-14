#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

"${PYTHON_BIN}" scripts/play_high_level.py \
  --num-robots 2 \
  --high-level-policy-source local \
  --high-level-policy-dir checkpoints/reproduction/high_level \
  --skill-policy-source local \
  --walk-policy-dir checkpoints/reproduction/walk \
  --dribble-policy-dir checkpoints/reproduction/dribble \
  --shoot-policy-dir checkpoints/reproduction/shoot \
  --headless \
  "$@"
