#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

# Completed W&B runs can also be used with:
#   --skill-policy-source wandb
#   --walk-wandb-run des_zhong/as2_walking/3a6g1def
#   --dribble-wandb-run <entity>/as2_dribbling/<run-id>
#   --shoot-wandb-run <entity>/as2_shooting/<run-id>
exec "${PYTHON_BIN}" scripts/collect_world_model_data.py \
  --output data/world_model_as2 \
  --num-robots 2 \
  --device cuda:6 \
  --skill-policy-source local \
  --walk-policy-dir wandb/as2_walk-3a6g1def/files/tmp/legged_data \
  --dribble-policy-dir wandb/as2_dribble-ofbwcsz3/files/tmp/legged_data/dribble \
  --shoot-policy-dir wandb/as2_shoot-a95j09x7/files/tmp/legged_data/shoot \
  --num-episodes 20000 \
  "$@"
