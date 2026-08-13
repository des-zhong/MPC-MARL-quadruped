#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

exec "${PYTHON_BIN}" scripts/train_high_level_with_mpc_teacher.py \
  --world-model-checkpoint checkpoints/world_model_as2/best.pt \
  --mpc-config configs/mpc_joint_teams.yaml \
  --mpc-profile teacher_training \
  --num-robots 2 \
  --self-play-update-interval 500 \
  --skill-policy-source local \
  --walk-policy-dir wandb/as2_walk-3a6g1def/files/tmp/legged_data \
  --dribble-policy-dir wandb/as2_dribble-ofbwcsz3/files/tmp/legged_data/dribble \
  --shoot-policy-dir wandb/as2_shoot-a95j09x7/files/tmp/legged_data/shoot \
  --checkpoint-dir tmp/legged_data/high_level_mpc_teacher \
  "$@"
