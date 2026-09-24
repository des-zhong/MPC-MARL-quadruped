#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
FONT_SIZE="${FONT_SIZE:-10}"
FIGURE_WIDTH="${FIGURE_WIDTH:-8}"
FIGURE_HEIGHT="${FIGURE_HEIGHT:-4.2}"

exec "${PYTHON_BIN}" scripts/replot_learning_curves.py \
  --font-size "${FONT_SIZE}" \
  --figure-width "${FIGURE_WIDTH}" \
  --figure-height "${FIGURE_HEIGHT}" \
  "$@"
