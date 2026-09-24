#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/dribblebot_torch_extensions}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

# Edit these paths to visualize your checkpoints.
WORLD_MODEL_CHECKPOINT="${WORLD_MODEL_CHECKPOINT:-wandb/run-20260918_145325-5iw09e94/files/tmp/legged_data/high_level_mpc_replay/world_model_online_latest.pt}"
# Optional: set a terminal value checkpoint and use reward_plus_terminal_value below.
TERMINAL_VALUE_CHECKPOINT="${TERMINAL_VALUE_CHECKPOINT:-wandb/run-20260918_145325-5iw09e94/files/tmp/legged_data/high_level_mpc_replay/terminal_value_online_latest.pt}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-reward_plus_terminal_value}"
CAMERA_SCALE="${CAMERA_SCALE:-0.65}"

args=(
  --world-model-checkpoint "$WORLD_MODEL_CHECKPOINT"
  --objective-mode "$OBJECTIVE_MODE"
  --camera-scale "$CAMERA_SCALE"
)
if [[ -n "$TERMINAL_VALUE_CHECKPOINT" ]]; then
  args+=(--terminal-value-checkpoint "$TERMINAL_VALUE_CHECKPOINT")
fi

# Explicit command-line arguments override the settings above.
exec "$PYTHON_BIN" scripts/visualize_mpc.py "${args[@]}" "$@"
