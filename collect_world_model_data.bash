#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"

PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
POLICY_DIR="${1:-wandb/run-20260915_234323-ft1q8tbg/files/tmp/legged_data/high_level_mpc_replay}"
if (($#)); then shift; fi
# Remaining positional arguments select checkpoint suffixes, e.g. 0 400 latest.
# With none supplied, collect from every ac_weights_<suffix>.pt in POLICY_DIR.
CHECKPOINTS=("$@")
if ((${#CHECKPOINTS[@]} == 0)); then
  shopt -s nullglob
  for checkpoint_path in "${POLICY_DIR}"/ac_weights_*.pt; do
    suffix="${checkpoint_path##*/ac_weights_}"
    CHECKPOINTS+=("${suffix%.pt}")
  done
fi
if ((${#CHECKPOINTS[@]} == 0)); then 
  echo "No PPO checkpoints found in ${POLICY_DIR}" >&2
  exit 1
fi
WORLD_MODEL_CHECKPOINT="${WORLD_MODEL_CHECKPOINT:-${POLICY_DIR}/world_model_online_latest.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/world_model_ppo_eval_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda:0}"
POLICY_DEVICE="${POLICY_DEVICE:-${DEVICE}}"
ROLLOUTS="${ROLLOUTS:-3}"

for suffix in "${CHECKPOINTS[@]}"; do
  if [[ ! "${suffix}" =~ ^(latest|[0-9]+)$ ]]; then
    echo "Expected a numbered checkpoint suffix or latest, got: ${suffix}" >&2
    exit 1
  fi
  output="${OUTPUT_ROOT}/checkpoint_${suffix}"
  opponent="${OPPONENT_CHECKPOINT:-${suffix}}"
  "${PYTHON_BIN}" scripts/collect_world_model_data.py \
    --high-level-policy-dir "${POLICY_DIR}" \
    --high-level-checkpoint "${suffix}" \
    --opponent-high-level-checkpoint "${opponent}" \
    --world-model-checkpoint "${WORLD_MODEL_CHECKPOINT}" \
    --world-model-output "${output}" \
    --csv "${OUTPUT_ROOT}/collection_${suffix}.csv" \
    --num-envs "${NUM_ENVS:-3}" --num-robots "${NUM_ROBOTS:-2}" \
    --stop-after-episodes "${ROLLOUTS}" --steps "${MAX_STEPS:-10000}" \
    --seed "${SEED:-20260915}" \
    --device "${DEVICE}" --policy-device "${POLICY_DEVICE}"
done
echo "Evaluation datasets saved under ${OUTPUT_ROOT}/checkpoint_<suffix>"
