#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

PURE_MAPPO_DIR="${1:-wandb/run-20260920_145323-1s2mw9rb/files/tmp/legged_data/high_level}"
MPC_REPLAY_DIR="${2:-wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay}"
MAX_OPPONENT_ITERATION="${3:-3200}"
METRIC="${METRIC:-goal-rate}"
SKILL_ARGS=()
if [[ -n "${SHOOT_POLICY_DIR:-}" ]]; then
  SKILL_ARGS+=(--shoot-policy-dir "${SHOOT_POLICY_DIR}")
fi

exec "${PYTHON_BIN}" scripts/compare_high_level_learning_curves.py \
  --method "Pure MAPPO=${PURE_MAPPO_DIR}" \
  --method "MPC replay=${MPC_REPLAY_DIR}" \
  --opponent-method "Pure MAPPO=${PURE_MAPPO_DIR}" \
  --opponent-method "MPC replay=${MPC_REPLAY_DIR}" \
  --opponent-max-iteration "${MAX_OPPONENT_ITERATION}" \
  --candidate-max-iteration "${MAX_OPPONENT_ITERATION}" \
  --opponent-stride 400 \
  --episodes-per-opponent 10 \
  --steps "${MAX_EVAL_STEPS:-1000}" \
  --metric "${METRIC}" \
  --x-axis environment-steps \
  --incremental-plot \
  --no-overwrite \
  --output-dir "${OUTPUT_DIR:-outputs/learning_curve_comparison}" \
  --num-robots 2 \
  --device "${DEVICE:-cuda:4}" \
  --policy-device "${POLICY_DEVICE:-cuda:4}" \
  "${SKILL_ARGS[@]}"
