#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
FONT_SIZE="${FONT_SIZE:-10}"
FIGURE_WIDTH="${FIGURE_WIDTH:-8}"
FIGURE_HEIGHT="${FIGURE_HEIGHT:-4.2}"

exec "${PYTHON_BIN}" scripts/plot_training_process.py \
    --mpc-dir wandb/run-20260920_145324-clcesqb3 \
    --mappo-dir wandb/run-20260920_145323-1s2mw9rb \
    --font-size "${FONT_SIZE}" \
    --figure-width "${FIGURE_WIDTH}" \
    --figure-height "${FIGURE_HEIGHT}" \
    "$@"
    # --figures attack_milestones ball_progress skill_usage plot_timeout_causes plot_world_model_velocity_error plot_world_model_ball_error plot_terminal_value_error plot_teacher_acceptance\
    # --window 100
