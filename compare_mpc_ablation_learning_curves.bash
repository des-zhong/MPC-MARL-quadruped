#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/dribblebot_torch_extensions}"
export PYTHONUNBUFFERED=1

# Editable defaults, with optional positional overrides like evaluate_goal_rate.bash.
MPC_DIR="wandb/run-20260920_145323-1s2mw9rb/files/tmp/legged_data/high_level"
NO_TERMINAL_VALUE_DIR="wandb/run-20260921_224940-qen5gkl3/files/outputs/mpc_ablations/no_terminal_value/seed_42"
NO_UNCERTAINTY_DIR="wandb/run-20260921_224920-cljr8hbn/files/outputs/mpc_ablations/no_uncertainty/seed_42"
MAX_OPPONENT_ITERATION="${MAX_OPPONENT_ITERATION:-2800}"

exec "${PYTHON_BIN}" scripts/compare_mpc_ablation_learning_curves.py \
  --mpc-dir "${MPC_DIR}" \
  --no-terminal-value-dir "${NO_TERMINAL_VALUE_DIR}" \
  --no-uncertainty-dir "${NO_UNCERTAINTY_DIR}" \
  --opponent-max-iteration "${MAX_OPPONENT_ITERATION}" \
  --max-iteration "${MAX_OPPONENT_ITERATION}" \
  "$@"
