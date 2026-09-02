#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the IsaacLab four-AS2 self-play baseline.
# Every default can be overridden with an environment variable; additional
# train_self_play.py arguments can be appended directly to this script.

REBUILD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${REBUILD_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

ISAAC_PYTHON="${ISAAC_PYTHON:-${REBUILD_ROOT}/.venv/bin/python}"
TASK="${TASK:-Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-8}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-24}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-match_selfplay_$(date +%Y%m%d_%H%M%S)}"
HEADLESS="${HEADLESS:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
OPPONENT_CHECKPOINT_ROOT="${OPPONENT_CHECKPOINT_ROOT:-}"
OPPONENT_POLICY_DEVICE="${OPPONENT_POLICY_DEVICE:-cpu}"
OPPONENT_MODE="${OPPONENT_MODE:-zero}"
OPPONENT_POOL_SIZE="${OPPONENT_POOL_SIZE:-8}"
OPPONENT_LATEST_PROBABILITY="${OPPONENT_LATEST_PROBABILITY:-0.5}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "IsaacLab Python 不存在或不可执行: ${ISAAC_PYTHON}" >&2
  echo "请先配置 isaaclab-rebuild/.venv，或设置 ISAAC_PYTHON=/path/to/python。" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REBUILD_ROOT}/source/dribblebot_isaaclab${PYTHONPATH:+:${PYTHONPATH}}"
export DRIBBLEBOT_EXPERIENCE="${DRIBBLEBOT_EXPERIENCE:-isaacsim.exp.base.python.kit}"

command=(
  "${ISAAC_PYTHON}"
  -u
  "${REBUILD_ROOT}/scripts/train_self_play.py"
  --task "${TASK}"
  --device "${DEVICE}"
  --num_envs "${NUM_ENVS}"
  --num_steps_per_env "${NUM_STEPS_PER_ENV}"
  --max_iterations "${MAX_ITERATIONS}"
  --seed "${SEED}"
  --run_name "${RUN_NAME}"
  --opponent_policy_device "${OPPONENT_POLICY_DEVICE}"
  --opponent_mode "${OPPONENT_MODE}"
  --opponent_pool_size "${OPPONENT_POOL_SIZE}"
  --opponent_latest_probability "${OPPONENT_LATEST_PROBABILITY}"
)

if [[ "${HEADLESS}" == "1" || "${HEADLESS}" == "true" ]]; then
  command+=(--headless)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  command+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ -n "${OPPONENT_CHECKPOINT_ROOT}" ]]; then
  command+=(--opponent_checkpoint_root "${OPPONENT_CHECKPOINT_ROOT}")
fi

# Explicit CLI arguments are appended last, so they can override defaults.
command+=("$@")

printf '启动 IsaacLab self-play 训练：\n'
printf '  task=%s device=%s num_envs=%s rollout=%s target_iterations=%s\n' \
  "${TASK}" "${DEVICE}" "${NUM_ENVS}" "${NUM_STEPS_PER_ENV}" "${MAX_ITERATIONS}"
printf '  run_name=%s headless=%s\n' "${RUN_NAME}" "${HEADLESS}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  printf '  resume=%s\n' "${RESUME_CHECKPOINT}"
fi
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${command[@]}"
