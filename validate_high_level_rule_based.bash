#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dribblebot_matplotlib}"

# Policy and output paths are kept here; validation behavior is defined by
# scripts/validate_high_level_rule_based.py.
# HIGH_LEVEL_POLICY_DIR="wandb/run-20260901_193801-7p3sykvg/files/tmp/legged_data/high_level_mpc_replay"
HIGH_LEVEL_POLICY_DIR="wandb/run-20260905_021328-ekwy8emt/files/tmp/legged_data/high_level"
WALK_POLICY_DIR="${WALK_POLICY_DIR:-checkpoints/reproduction/walk}"
DRIBBLE_POLICY_DIR="${DRIBBLE_POLICY_DIR:-checkpoints/reproduction/dribble}"
SHOOT_POLICY_DIR="${SHOOT_POLICY_DIR:-checkpoints/reproduction/shoot}"

VIDEO_PATH="${VIDEO_PATH:-outputs/high_level_rule_based_eval.mp4}"
PLOT_PATH="${PLOT_PATH:-outputs/high_level_rule_based_eval_metrics.png}"
CSV_PATH="${CSV_PATH:-outputs/high_level_rule_based_eval_metrics.csv}"
DEVICE="${DEVICE:-cuda:4}"
POLICY_DEVICE="${POLICY_DEVICE:-${DEVICE}}"

"${PYTHON_BIN}" scripts/validate_high_level_rule_based.py \
  --num-robots 2 \
  --high-level-policy-dir "${HIGH_LEVEL_POLICY_DIR}" \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --device "${DEVICE}" \
  --policy-device "${POLICY_DEVICE}" \
  --video "${VIDEO_PATH}" \
  --plot "${PLOT_PATH}" \
  --csv "${CSV_PATH}"
