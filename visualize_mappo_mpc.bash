#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/dribblebot_torch_extensions}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

# MAPPO drives the learning team; the saved opponent pool drives the other team.
HIGH_LEVEL_POLICY_DIR="${HIGH_LEVEL_POLICY_DIR:-wandb/run-20260920_145323-1s2mw9rb/files/tmp/legged_data/high_level}"
HIGH_LEVEL_CHECKPOINT="${HIGH_LEVEL_CHECKPOINT:-latest}"
OPPONENT_POOL_CHECKPOINT="${OPPONENT_POOL_CHECKPOINT:-${HIGH_LEVEL_POLICY_DIR}/opponent_pool_2000.pt}"
WORLD_MODEL_CHECKPOINT="${WORLD_MODEL_CHECKPOINT:-wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay/world_model_online_latest.pt}"
TERMINAL_VALUE_CHECKPOINT="${TERMINAL_VALUE_CHECKPOINT:-wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay/terminal_value_online_latest.pt}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-reward_plus_terminal_value}"
CAMERA_SCALE="${CAMERA_SCALE:-0.65}"
# Prediction horizon in high-level steps (10 steps = 2 seconds at the current control rate).
MPC_HORIZON="${MPC_HORIZON:-10}"
DEVICE="${DEVICE:-cuda:0}"
### 40 defense
### 38 get defensed

## 25 goal
SEED="${SEED:-24}"
EPISODES="${EPISODES:-10}"

exec "$PYTHON_BIN" scripts/visualize_mappo_mpc.py \
  --high-level-policy-dir "$HIGH_LEVEL_POLICY_DIR" \
  --high-level-checkpoint "$HIGH_LEVEL_CHECKPOINT" \
  --opponent-pool-checkpoint "$OPPONENT_POOL_CHECKPOINT" \
  --world-model-checkpoint "$WORLD_MODEL_CHECKPOINT" \
  --terminal-value-checkpoint "$TERMINAL_VALUE_CHECKPOINT" \
  --objective-mode "$OBJECTIVE_MODE" \
  --camera-scale "$CAMERA_SCALE" \
  --mpc-horizon "$MPC_HORIZON" \
  --seed "$SEED" --episodes "$EPISODES" --random-learning-start \
  --device "$DEVICE" --policy-device "$DEVICE" "$@"
