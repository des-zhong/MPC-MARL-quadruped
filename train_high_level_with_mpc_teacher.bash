#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

exec "${PYTHON_BIN}" scripts/train_high_level_with_mpc_teacher.py \
  --world-model-checkpoint checkpoints/reproduction/world_model/best.pt \
  --mpc-config configs/mpc_joint_teams.yaml \
  --mpc-profile teacher_training \
  --num-robots 2 \
  --self-play-update-interval 500 \
  --skill-policy-source local \
  --walk-policy-dir checkpoints/reproduction/walk \
  --dribble-policy-dir checkpoints/reproduction/dribble \
  --shoot-policy-dir checkpoints/reproduction/shoot \
  "$@"
