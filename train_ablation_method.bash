#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {mappo_fsp|no_terminal|ours} [training options]" >&2
  exit 2
fi

METHOD="$1"
shift

PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

TRAINING_SEED="${ABLATION_TRAINING_SEED:-42}"
CHECKPOINT_ROOT="${ABLATION_CHECKPOINT_ROOT:-checkpoints/ablation}"
NUM_ENVS="${ABLATION_NUM_ENVS:-32}"
ITERATIONS="${ABLATION_ITERATIONS:-10000}"
DEVICE="${ABLATION_DEVICE:-cuda:0}"
POLICY_DEVICE="${ABLATION_POLICY_DEVICE:-cpu}"
INITIAL_POLICY="${ABLATION_INITIAL_POLICY:-}"
WORLD_MODEL_CHECKPOINT="${WORLD_MODEL_CHECKPOINT:-checkpoints/reproduction/world_model/best.pt}"
WORLD_MODEL_CONFIG="${WORLD_MODEL_CONFIG:-configs/world_model_as2.yaml}"
MPC_CONFIG="${MPC_CONFIG:-configs/mpc_joint_teams.yaml}"
TERMINAL_VALUE_CHECKPOINT="${TERMINAL_VALUE_CHECKPOINT:-checkpoints/terminal_value_env_v2_bootstrap/best.pt}"
WALK_POLICY_DIR="${WALK_POLICY_DIR:-checkpoints/reproduction/walk}"
DRIBBLE_POLICY_DIR="${DRIBBLE_POLICY_DIR:-checkpoints/reproduction/dribble}"
SHOOT_POLICY_DIR="${SHOOT_POLICY_DIR:-checkpoints/reproduction/shoot}"

case "${METHOD}" in
  mappo_fsp)
    METHOD_LABEL="MAPPO-FSP"
    PROJECT="as2_ablation_mappo_fsp"
    ;;
  no_terminal)
    METHOD_LABEL="Ours-no-terminal"
    PROJECT="as2_ablation_ours_no_terminal"
    ;;
  ours)
    METHOD_LABEL="Ours"
    PROJECT="as2_ablation_ours"
    ;;
  *)
    echo "Unknown method '${METHOD}'. Expected mappo_fsp, no_terminal, or ours." >&2
    exit 2
    ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${METHOD_LABEL}/seed_${TRAINING_SEED}"
COMMON_ARGS=(
  --device "${DEVICE}"
  --policy-device "${POLICY_DEVICE}"
  --seed "${TRAINING_SEED}"
  --headless
  --project "${PROJECT}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --num-envs "${NUM_ENVS}"
  --num-robots 2
  --iterations "${ITERATIONS}"
  --self-play-update-interval "${SELF_PLAY_UPDATE_INTERVAL:-2000}"
  --skill-entropy-coef 0.002
  --skill-entropy-final-coef 0.0002
  --skill-entropy-anneal-iterations 4000
  --skill-policy-source local
  --walk-policy-dir "${WALK_POLICY_DIR}"
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}"
  --shoot-policy-dir "${SHOOT_POLICY_DIR}"
)
if [[ -n "${INITIAL_POLICY}" ]]; then
  COMMON_ARGS+=(
    --resume
    --resume-mode policy-only
    --resume-checkpoint "${INITIAL_POLICY}"
  )
fi

echo "Training ${METHOD_LABEL} (seed ${TRAINING_SEED})"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"

if [[ "${METHOD}" == "mappo_fsp" ]]; then
  exec "${PYTHON_BIN}" scripts/train_high_level_mappo_fsp.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --opponent-pool-size "${OPPONENT_POOL_SIZE:-8}" \
    --opponent-latest-probability "${OPPONENT_LATEST_PROBABILITY:-0.5}"
fi

MPC_ARGS=(
  --world-model-checkpoint "${WORLD_MODEL_CHECKPOINT}"
  --world-model-config "${WORLD_MODEL_CONFIG}"
  --mpc-config "${MPC_CONFIG}"
  --mpc-profile teacher_training
  --world-model-update-interval "${WORLD_MODEL_UPDATE_INTERVAL:-25}"
  --world-model-replay-buffer-size "${WORLD_MODEL_REPLAY_BUFFER_SIZE:-100000}"
  --world-model-replay-recent-fraction "${WORLD_MODEL_REPLAY_RECENT_FRACTION:-0.5}"
  --world-model-replay-recent-window "${WORLD_MODEL_REPLAY_RECENT_WINDOW:-10000}"
  --mpc-horizon "${MPC_HORIZON:-2}"
  --mpc-num-samples "${MPC_NUM_SAMPLES:-256}"
  --mpc-num-iterations "${MPC_NUM_ITERATIONS:-4}"
  --mpc-warmup-steps "${MPC_WARMUP_STEPS:-500}"
  --mpc-guidance-reward-coefficient "${MPC_GUIDANCE_REWARD_COEFFICIENT:-1.0}"
  --opponent-pool-size "${OPPONENT_POOL_SIZE:-8}"
  --opponent-latest-probability "${OPPONENT_LATEST_PROBABILITY:-0.5}"
)

TERMINAL_ARGS=()
if [[ -n "${TERMINAL_VALUE_CHECKPOINT}" ]]; then
  TERMINAL_ARGS=(--terminal-value-checkpoint "${TERMINAL_VALUE_CHECKPOINT}")
fi

case "${METHOD}" in
  no_terminal)
    exec "${PYTHON_BIN}" scripts/train_high_level_online_mpc_no_terminal.py \
      "${COMMON_ARGS[@]}" "${MPC_ARGS[@]}" "$@" \
      --mpc-terminal-value disabled \
      --mpc-kl-coefficient "${MPC_KL_COEFFICIENT:-0.05}"
    ;;
  ours)
    exec "${PYTHON_BIN}" scripts/train_high_level_online_mpc_full.py \
      "${COMMON_ARGS[@]}" "${MPC_ARGS[@]}" "${TERMINAL_ARGS[@]}" "$@" \
      --mpc-terminal-value enabled \
      --mpc-kl-coefficient "${MPC_KL_COEFFICIENT:-0.05}"
    ;;
esac
