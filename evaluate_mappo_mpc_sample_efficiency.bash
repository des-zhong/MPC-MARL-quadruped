#!/usr/bin/env bash
set -euo pipefail

# Evaluate MAPPO and MPC on the same benchmark, then count the training
# transitions needed to reach a stable expected-match-score target.
#
# Positional arguments (all optional):
#   1 MAPPO policy directory
#   2 MPC policy directory
#   3 walk skill directory
#   4 dribble skill directory
#   5 shoot skill directory
#   6 output directory
#
# Other settings are exposed as environment variables below. Override them
# before invoking this file, for example:
#   DEVICE=cuda:4 TARGET_SCORE=60 ./evaluate_mappo_mpc_sample_efficiency.bash

cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

MAPPO_LABEL="${MAPPO_LABEL:-MAPPO}"
MPC_LABEL="${MPC_LABEL:-MPC}"
MAPPO_POLICY_DIR="${1:-${MAPPO_POLICY_DIR:-wandb/run-20260918_145325-5iw09e94/files/tmp/legged_data/high_level_mpc_replay}}"
MPC_POLICY_DIR="${2:-${MPC_POLICY_DIR:-wandb/run-20260918_145316-c9nr12in/files/tmp/legged_data/high_level}}"
WALK_POLICY_DIR="${3:-${WALK_POLICY_DIR:-checkpoints/reproduction/walk}}"
DRIBBLE_POLICY_DIR="${4:-${DRIBBLE_POLICY_DIR:-checkpoints/reproduction/dribble}}"
SHOOT_POLICY_DIR="${5:-${SHOOT_POLICY_DIR:-checkpoints/reproduction/shoot}}"
OUTPUT_DIR="${6:-${OUTPUT_DIR:-outputs/mappo_mpc_sample_efficiency}}"

# Learning-curve evaluation settings.
DEVICE="${DEVICE:-cuda:3}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:3}"
NUM_ROBOTS="${NUM_ROBOTS:-2}"
MAX_ITERATION="${MAX_ITERATION:-2000}"
OPPONENT_STRIDE="${OPPONENT_STRIDE:-400}"
EPISODES_PER_OPPONENT="${EPISODES_PER_OPPONENT:-10}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SEEDS="${SEEDS:-0}"
DOMAIN_RAND="${DOMAIN_RAND:-0}"
FIXED_INIT="${FIXED_INIT:-0}"
CANDIDATE_CHECKPOINTS="${CANDIDATE_CHECKPOINTS:-auto}"
STEPS_PER_ITERATION_MAPPO="${STEPS_PER_ITERATION_MAPPO:-}"
STEPS_PER_ITERATION_MPC="${STEPS_PER_ITERATION_MPC:-}"
STEP_OFFSET_MAPPO="${STEP_OFFSET_MAPPO:-}"
STEP_OFFSET_MPC="${STEP_OFFSET_MPC:-}"

# Sample-efficiency criterion.
TARGET_SCORE="${TARGET_SCORE:-55}"
CONSECUTIVE_CHECKPOINTS="${CONSECUTIVE_CHECKPOINTS:-3}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,32p' "${BASH_SOURCE[0]}"
  exit 0
fi

COMPARE_ARGS=(
  --method "${MAPPO_LABEL}=${MAPPO_POLICY_DIR}"
  --method "${MPC_LABEL}=${MPC_POLICY_DIR}"
  --opponent-method "${MAPPO_LABEL}=${MAPPO_POLICY_DIR}"
  --opponent-method "${MPC_LABEL}=${MPC_POLICY_DIR}"
  --candidate-max-iteration "${MAX_ITERATION}"
  --candidate-checkpoints "${CANDIDATE_CHECKPOINTS}"
  --opponent-max-iteration "${MAX_ITERATION}"
  --opponent-stride "${OPPONENT_STRIDE}"
  --episodes-per-opponent "${EPISODES_PER_OPPONENT}"
  --steps "${EVAL_STEPS}"
  --seeds "${SEEDS}"
  --metric expected-score
  --x-axis environment-steps
  --incremental-plot
  --no-overwrite
  --output-dir "${OUTPUT_DIR}"
  --num-robots "${NUM_ROBOTS}"
  --device "${DEVICE}"
  --policy-device "${POLICY_DEVICE}"
  --walk-policy-dir "${WALK_POLICY_DIR}"
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}"
  --shoot-policy-dir "${SHOOT_POLICY_DIR}"
)

if [[ "${DOMAIN_RAND}" == "1" ]]; then COMPARE_ARGS+=(--domain-rand); fi
if [[ "${FIXED_INIT}" == "1" ]]; then COMPARE_ARGS+=(--fixed-init); fi
if [[ -n "${STEPS_PER_ITERATION_MAPPO}" ]]; then
  COMPARE_ARGS+=(--steps-per-iteration "${MAPPO_LABEL}=${STEPS_PER_ITERATION_MAPPO}")
fi
if [[ -n "${STEPS_PER_ITERATION_MPC}" ]]; then
  COMPARE_ARGS+=(--steps-per-iteration "${MPC_LABEL}=${STEPS_PER_ITERATION_MPC}")
fi
if [[ -n "${STEP_OFFSET_MAPPO}" ]]; then
  COMPARE_ARGS+=(--step-offset "${MAPPO_LABEL}=${STEP_OFFSET_MAPPO}")
fi
if [[ -n "${STEP_OFFSET_MPC}" ]]; then
  COMPARE_ARGS+=(--step-offset "${MPC_LABEL}=${STEP_OFFSET_MPC}")
fi

"${PYTHON_BIN}" scripts/compare_high_level_learning_curves.py "${COMPARE_ARGS[@]}"

"${PYTHON_BIN}" scripts/evaluate_sample_efficiency.py \
  --method "${MAPPO_LABEL}=${OUTPUT_DIR}/learning_curve.csv" \
  --method "${MPC_LABEL}=${OUTPUT_DIR}/learning_curve.csv" \
  --target-score "${TARGET_SCORE}" \
  --consecutive-checkpoints "${CONSECUTIVE_CHECKPOINTS}" \
  --output-dir "${OUTPUT_DIR}"
