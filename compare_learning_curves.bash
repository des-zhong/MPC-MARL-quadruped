#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dribblebot_matplotlib}"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 MAPPO_FSP_DIR NO_TERMINAL_DIR OURS_DIR [evaluation options]" >&2
  echo "" >&2
  echo "Each directory must contain numbered ac_weights_*.pt, body_*.jit," >&2
  echo "adaptation_module_*.jit, and the training config.yaml." >&2
  exit 2
fi

MAPPO_FSP_DIR="$1"
NO_TERMINAL_DIR="$2"
OURS_DIR="$3"
shift 3

exec "${PYTHON_BIN}" scripts/compare_high_level_learning_curves.py \
  --method "MAPPO-FSP=${MAPPO_FSP_DIR}" \
  --method "Ours w/o Terminal Value=${NO_TERMINAL_DIR}" \
  --method "Ours=${OURS_DIR}" \
  --opponent-dir "${MAPPO_FSP_DIR}" \
  --opponent-checkpoints auto \
  --seeds 0 \
  --steps 300 \
  --eval-num-envs 4 \
  --policy-device cuda:5 \
  --overwrite
  "$@"
