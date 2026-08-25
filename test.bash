#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

# ===== Paths to edit for a new run =====
WALK_POLICY_DIR="wandb/as2_walk-3a6g1def/files/tmp/legged_data"
DRIBBLE_POLICY_DIR="wandb/run-20260809_220330-ofbwcsz3/files/tmp/legged_data/dribble"
SHOOT_POLICY_DIR="wandb/run-20260728_144658-lphndlu9/files/tmp/legged_data"
VIDEO_PATH="outputs/walk_dribble_shoot.mp4"
PLOT_PATH="outputs/walk_dribble_shoot_metrics.png"
CSV_PATH="outputs/walk_dribble_shoot_metrics.csv"
# =======================================

exec "${PYTHON_BIN}" scripts/play_walk_dribble_shoot.py \
  --skill-policy-source local \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --video "${VIDEO_PATH}" \
  --plot "${PLOT_PATH}" \
  --csv "${CSV_PATH}" \
  "$@"
