#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/dribblebot_torch_extensions}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
exec "$PYTHON_BIN" scripts/train_shooting.py --headless --option-training \
  --device "${DEVICE:-cuda:7}" --checkpoint-dir tmp/legged_data/shoot_option \
  --project as2_shooting_option "$@"
