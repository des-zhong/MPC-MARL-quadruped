#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dribblebot_matplotlib}"

# The learning team still uses its trained high-level coordinator. The
# opponent high-level decisions are generated entirely by hard-coded rules;
# only the shared low-level walk/dribble/shoot skill policies are loaded.
HIGH_LEVEL_POLICY_DIR="${HIGH_LEVEL_POLICY_DIR:-wandb/run-20260825_101144-zpkeqtxq/files/tmp/legged_data/high_level}"
WALK_POLICY_DIR="${WALK_POLICY_DIR:-checkpoints/reproduction/walk}"
DRIBBLE_POLICY_DIR="${DRIBBLE_POLICY_DIR:-checkpoints/reproduction/dribble}"
SHOOT_POLICY_DIR="${SHOOT_POLICY_DIR:-checkpoints/reproduction/shoot}"

VIDEO_PATH="${VIDEO_PATH:-outputs/high_level_rule_based_eval.mp4}"
PLOT_PATH="${PLOT_PATH:-outputs/high_level_rule_based_eval_metrics.png}"
CSV_PATH="${CSV_PATH:-outputs/high_level_rule_based_eval_metrics.csv}"
SEED="${SEED:-50}"
DEVICE="${DEVICE:-cuda:4}"
POLICY_DEVICE="${POLICY_DEVICE:-${DEVICE}}"

# Rule-based opponent tuning. Team one attacks the -x goal.
RULE_SHOOT_DISTANCE="${RULE_SHOOT_DISTANCE:-0.75}"
RULE_DRIBBLE_DISTANCE="${RULE_DRIBBLE_DISTANCE:-1.0}"
RULE_WALK_SPEED="${RULE_WALK_SPEED:-0.9}"
RULE_DRIBBLE_SPEED="${RULE_DRIBBLE_SPEED:-1.0}"
RULE_SHOOT_SPEED="${RULE_SHOOT_SPEED:-1.5}"
RULE_BLOCK_DISTANCE="${RULE_BLOCK_DISTANCE:-0.75}"
RULE_COLLISION_DISTANCE="${RULE_COLLISION_DISTANCE:-0.65}"
RULE_COLLISION_STRENGTH="${RULE_COLLISION_STRENGTH:-1.5}"

# This final emergency guard applies to every robot after both coordinators
# choose actions. Set ENABLE_COLLISION_AVOIDANCE=0 for raw-policy evaluation.
ENABLE_COLLISION_AVOIDANCE="${ENABLE_COLLISION_AVOIDANCE:-1}"
COLLISION_ARGS=()
if [[ "${ENABLE_COLLISION_AVOIDANCE}" == "1" ]]; then
  COLLISION_ARGS=(
    --collision-avoidance
    --collision-avoidance-distance "${COLLISION_AVOIDANCE_DISTANCE:-0.55}"
    --collision-avoidance-lookahead "${COLLISION_AVOIDANCE_LOOKAHEAD:-0.25}"
    --collision-avoidance-speed "${COLLISION_AVOIDANCE_SPEED:-0.5}"
  )
fi

"${PYTHON_BIN}" scripts/play_high_level.py \
  --num-robots 2 \
  --high-level-policy-source local \
  --high-level-policy-dir "${HIGH_LEVEL_POLICY_DIR}" \
  --high-level-checkpoint latest \
  --opponent-rule-based \
  --rule-opponent-shoot-distance "${RULE_SHOOT_DISTANCE}" \
  --rule-opponent-dribble-distance "${RULE_DRIBBLE_DISTANCE}" \
  --rule-opponent-walk-speed "${RULE_WALK_SPEED}" \
  --rule-opponent-dribble-speed "${RULE_DRIBBLE_SPEED}" \
  --rule-opponent-shoot-speed "${RULE_SHOOT_SPEED}" \
  --rule-opponent-block-distance "${RULE_BLOCK_DISTANCE}" \
  --rule-opponent-collision-distance "${RULE_COLLISION_DISTANCE}" \
  --rule-opponent-collision-strength "${RULE_COLLISION_STRENGTH}" \
  --skill-policy-source local \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  --device "${DEVICE}" \
  --policy-device "${POLICY_DEVICE}" \
  --video "${VIDEO_PATH}" \
  --plot "${PLOT_PATH}" \
  --csv "${CSV_PATH}" \
  --seed "${SEED}" \
  "${COLLISION_ARGS[@]}" \
  --headless \
  "$@"
