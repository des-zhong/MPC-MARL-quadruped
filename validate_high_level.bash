#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

"${PYTHON_BIN}" scripts/play_high_level.py \
  --num-robots 2 \
  --high-level-policy-source local \
  --high-level-policy-dir wandb/run-20260812_233656-nkgeuto7/files/tmp/legged_data/high_level \
  --skill-policy-source local \
  --walk-policy-dir wandb/as2_walk-3a6g1def/files/tmp/legged_data \
  --dribble-policy-dir wandb/as2_dribble-ofbwcsz3/files/tmp/legged_data/dribble \
  --shoot-policy-dir wandb/as2_shoot-a95j09x7/files/tmp/legged_data/shoot \
  --headless \
  "$@"
