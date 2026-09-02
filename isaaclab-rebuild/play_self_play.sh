#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${REBUILD_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

ISAAC_PYTHON="${ISAAC_PYTHON:-${REBUILD_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EPISODES="${EPISODES:-3}"
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-7}"
FRAME_DELAY="${FRAME_DELAY:-0.03}"
OPPONENT="${OPPONENT:-self}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "IsaacLab Python 不存在或不可执行: ${ISAAC_PYTHON}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REBUILD_ROOT}/source/dribblebot_isaaclab${PYTHONPATH:+:${PYTHONPATH}}"
export DRIBBLEBOT_EXPERIENCE="${DRIBBLEBOT_EXPERIENCE:-isaacsim.exp.base.python.kit}"

command=(
  "${ISAAC_PYTHON}"
  -u
  "${REBUILD_ROOT}/scripts/play_self_play.py"
  --device "${DEVICE}"
  --episodes "${EPISODES}"
  --num_envs "${NUM_ENVS}"
  --seed "${SEED}"
  --frame_delay "${FRAME_DELAY}"
  --opponent "${OPPONENT}"
)
if [[ -n "${CHECKPOINT}" ]]; then
  command+=(--checkpoint "${CHECKPOINT}")
fi
command+=("$@")

printf '启动 checkpoint 可视化：'
printf ' %q' "${command[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${command[@]}"
