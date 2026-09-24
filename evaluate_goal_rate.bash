#!/usr/bin/env bash
set -euo pipefail

# if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -lt 2 ]]; then
#   cat <<'USAGE'
# Usage: bash evaluate_goal_rate.bash TEAM_A.pt TEAM_B.pt [options]

# Run 100 matches x 10 repetitions concurrently, without visualization.
# Checkpoint paths should point to ac_weights_<checkpoint>.pt files.
# Team B also accepts opponent_ac_weights_<checkpoint>.pt.
# Relative paths are resolved from the directory where you run this command.

# Examples:
#   bash evaluate_goal_rate.bash /path/to/A/ac_weights_latest.pt /path/to/B/ac_weights_latest.pt
#   bash evaluate_goal_rate.bash /path/to/A/ac_weights_100.pt /path/to/B/ac_weights_200.pt --device cuda:4
#   bash evaluate_goal_rate.bash /path/to/A/ac_weights_latest.pt /path/to/B/ac_weights_latest.pt --repeats-per-batch 2

# Options are forwarded to scripts/evaluate_goal_rate.py, including:
#   --device cuda:0           Simulation and policy GPU
#   --matches 100             Matches per repetition
#   --repeats 10              Number of repetitions
#   --repeats-per-batch N     Reduce concurrent repetitions to save GPU memory
#   --steps 600               Step limit; increase if matches do not finish
#   --output DIRECTORY        New or empty output directory

# Reports: summary.json and repeats.csv in a unique outputs/goal_rate_* directory
# unless --output is supplied. Set DRIBBLEBOT_PYTHON to override Python.
# USAGE
#   if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
#     exit 0
#   fi
#   exit 2
# fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"
export PYTHONUNBUFFERED=1

TEAM_A="wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay/ac_weights_latest.pt"
TEAM_B="wandb/run-20260920_145325-mn957gzl/files/tmp/legged_data/high_level_discrete/ac_weights_latest.pt"
if [[ $# -gt 0 && "$1" != --* ]]; then
  TEAM_A="$1"
  shift
fi
if [[ $# -gt 0 && "$1" != --* ]]; then
  TEAM_B="$1"
  shift
fi

OUTPUT_DIR="${PROJECT_ROOT}/outputs/goal_rate_$(date +%Y%m%d_%H%M%S)_$$"

# OPPONENT_POOL_DIR_1="${1:-wandb/run-20260920_145323-1s2mw9rb/files/tmp/legged_data/high_level}"
# OPPONENT_POOL_DIR_2="${2:-wandb/run-20260920_145325-mn957gzl/files/tmp/legged_data/high_level_discrete}"
# OPPONENT_POOL_DIR_3="${3:-wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay}"

python scripts/evaluate_goal_rate.py \
  --team-a "${TEAM_A}" \
  --team-b "${TEAM_B}" \
  --matches 100 \
  --repeats 10 \
  --device cuda:2 \
  --output "${OUTPUT_DIR}" \
  "$@"
