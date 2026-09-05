#!/usr/bin/env bash
set -euo pipefail

# Sequential train -> evaluate -> video orchestration.
# A training chunk exits Isaac Sim before the evaluator starts, avoiding two
# concurrent Kit/PhysX processes competing for the same GPU and Omni resources.

REBUILD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${REBUILD_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

ISAAC_PYTHON="${ISAAC_PYTHON:-${REBUILD_ROOT}/.venv/bin/python}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1}"
EVAL_OPPONENT="${EVAL_OPPONENT:-self}"
EVAL_VIDEO="${EVAL_VIDEO:-1}"
EVAL_VIDEO_WIDTH="${EVAL_VIDEO_WIDTH:-640}"
EVAL_VIDEO_HEIGHT="${EVAL_VIDEO_HEIGHT:-360}"
EVAL_VIDEO_FPS="${EVAL_VIDEO_FPS:-5}"
EVAL_FRAME_DELAY="${EVAL_FRAME_DELAY:-0}"
RUN_NAME="${RUN_NAME:-match_selfplay_eval_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REBUILD_ROOT}/logs/rsl_rl/dribblebot_as2_match_self_play/${RUN_NAME}}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "IsaacLab Python 不存在或不可执行: ${ISAAC_PYTHON}" >&2
  exit 1
fi
if [[ "${EVAL_INTERVAL}" -le 0 || "${MAX_ITERATIONS}" -le 0 || "${EVAL_EPISODES}" -le 0 ]]; then
  echo "MAX_ITERATIONS、EVAL_INTERVAL、EVAL_EPISODES 必须为正数" >&2
  exit 1
fi
if [[ "${EVAL_NUM_ENVS}" -le 0 ]]; then
  echo "EVAL_NUM_ENVS 必须为正数" >&2
  exit 1
fi

LOG_DIR="$(realpath -m -- "${LOG_DIR}")"
mkdir -p "${LOG_DIR}/eval"

current_end=-1
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  checkpoint_name="$(basename -- "${RESUME_CHECKPOINT}")"
  if [[ "${checkpoint_name}" =~ ^model_([0-9]+)\.pt$ ]]; then
    current_end="${BASH_REMATCH[1]}"
  else
    echo "无法从 RESUME_CHECKPOINT 解析 model_<iteration>.pt: ${RESUME_CHECKPOINT}" >&2
    exit 1
  fi
fi

while [[ "${current_end}" -lt $((MAX_ITERATIONS - 1)) ]]; do
  if [[ "${current_end}" -lt 0 ]]; then
    segment_end=$((EVAL_INTERVAL - 1))
  else
    segment_end=$((current_end + EVAL_INTERVAL))
  fi
  if [[ "${segment_end}" -ge $((MAX_ITERATIONS - 1)) ]]; then
    segment_end=$((MAX_ITERATIONS - 1))
  fi

  echo "[TRAIN-EVAL] training through iteration ${segment_end}" >&2
  resume_next_arg=()
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    resume_next_arg=(--resume_next_iteration)
  fi
  RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" \
  LOG_DIR="${LOG_DIR}" \
  RUN_NAME="${RUN_NAME}" \
  MAX_ITERATIONS=$((segment_end + 1)) \
  "${REBUILD_ROOT}/train_self_play.sh" "${resume_next_arg[@]}" "$@"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[TRAIN-EVAL] dry-run checkpoint=${LOG_DIR}/model_${segment_end}.pt" >&2
    exit 0
  fi

  checkpoint="${LOG_DIR}/model_${segment_end}.pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "训练 chunk 未生成 checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  video_arg=()
  if [[ "${EVAL_VIDEO}" == "1" || "${EVAL_VIDEO}" == "true" ]]; then
    video_arg=(--video_output "${LOG_DIR}/eval/iter_${segment_end}.mp4")
  fi
  metrics="${LOG_DIR}/eval/iter_${segment_end}.json"
  echo "[TRAIN-EVAL] evaluating ${checkpoint}" >&2
  "${ISAAC_PYTHON}" -u "${REBUILD_ROOT}/scripts/play_self_play.py" \
    --checkpoint "${checkpoint}" \
    --episodes "${EVAL_EPISODES}" \
    --num_envs "${EVAL_NUM_ENVS}" \
    --opponent "${EVAL_OPPONENT}" \
    --headless \
    --frame_delay "${EVAL_FRAME_DELAY}" \
    --metrics_output "${metrics}" \
    --video_width "${EVAL_VIDEO_WIDTH}" \
    --video_height "${EVAL_VIDEO_HEIGHT}" \
    --video_fps "${EVAL_VIDEO_FPS}" \
    "${video_arg[@]}"
  "${ISAAC_PYTHON}" "${REBUILD_ROOT}/scripts/log_eval_metrics.py" \
    --metrics "${metrics}" \
    --log_dir "${LOG_DIR}" \
    --iteration "${segment_end}"

  RESUME_CHECKPOINT="${checkpoint}"
  current_end="${segment_end}"
done

echo "[TRAIN-EVAL] finished through iteration ${current_end}" >&2
