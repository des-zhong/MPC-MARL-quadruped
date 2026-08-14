#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/dribblebot_matplotlib}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

exec "${PYTHON_BIN}" scripts/visualize_mpc.py \
  --config configs/mpc_joint_teams.yaml \
  --profile teacher_high_quality \
  --skill-policy-source local \
  --num-robots 2 \
  --walk-policy-dir checkpoints/reproduction/walk \
  --dribble-policy-dir checkpoints/reproduction/dribble \
  --shoot-policy-dir checkpoints/reproduction/shoot \
  --opponent-policy-source local \
  --opponent-policy-dir checkpoints/reproduction/high_level \
  --world-model-checkpoint checkpoints/reproduction/world_model/best.pt \
  "$@"
