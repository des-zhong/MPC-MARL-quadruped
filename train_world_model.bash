#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"

exec "${PYTHON_BIN}" scripts/train_world_model.py \
  --config configs/world_model_as2.yaml \
  --dataset data/world_model_as2 \
  --output checkpoints/world_model_as2 \
  --num-robots 2 \
  "$@"
