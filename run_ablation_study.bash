#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dribblebot_matplotlib}"
OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-outputs/ablation_study}"

if [[ $# -eq 0 || "${1}" == -* ]]; then
  # The three paths are produced by train_ablation_method.bash. Supplying
  # explicit paths remains useful when runs live on separate machines.
  EVALUATION_ARGS=("$@")
  TRAINING_SEED="${ABLATION_TRAINING_SEED:-42}"
  CHECKPOINT_ROOT="${ABLATION_CHECKPOINT_ROOT:-checkpoints/ablation}"
  set -- \
    "${CHECKPOINT_ROOT}/MAPPO-FSP/seed_${TRAINING_SEED}" \
    "${CHECKPOINT_ROOT}/Ours-no-terminal/seed_${TRAINING_SEED}" \
    "${CHECKPOINT_ROOT}/Ours/seed_${TRAINING_SEED}" \
    "${EVALUATION_ARGS[@]}"
elif [[ $# -lt 3 ]]; then
  echo "Usage: $0 MAPPO_FSP_DIR NO_TERMINAL_DIR OURS_DIR [evaluation options]" >&2
  echo "       $0 [evaluation options]  # use checkpoints/ablation/* defaults" >&2
  echo "" >&2
  echo "NO_TERMINAL_DIR differs from Ours only by disabling terminal value." >&2
  echo "All directories need numbered ac_weights/body/adaptation_module exports." >&2
  echo "Use --cuda N (for example --cuda 6) or --device cuda:N for evaluation." >&2
  exit 2
fi

MAPPO_FSP_DIR="$1"
NO_TERMINAL_DIR="$2"
OURS_DIR="$3"
shift 3

"${PYTHON_BIN}" scripts/compare_high_level_learning_curves.py \
  --method "MAPPO-FSP=${MAPPO_FSP_DIR}" \
  --method "Ours w/o Terminal Value=${NO_TERMINAL_DIR}" \
  --method "Ours=${OURS_DIR}" \
  --opponent-dir "${MAPPO_FSP_DIR}" \
  --opponent-checkpoints suite \
  --seeds "${ABLATION_EVAL_SEEDS:-0}" \
  --steps "${ABLATION_ROLLOUT_STEPS:-300}" \
  --eval-num-envs "${ABLATION_EVAL_ENVS:-4}" \
  "$@" \
  --output-dir "${OUTPUT_DIR}"

"${PYTHON_BIN}" scripts/analyze_ablation_study.py \
  --evaluations "${OUTPUT_DIR}/evaluations.csv" \
  --target-score "${ABLATION_TARGET_SCORE:-60}" \
  --target-consecutive "${ABLATION_TARGET_CONSECUTIVE:-1}" \
  --bootstrap-samples "${ABLATION_BOOTSTRAP_SAMPLES:-10000}" \
  --seed "${ABLATION_ANALYSIS_SEED:-2026}" \
  --output-dir "${OUTPUT_DIR}"
