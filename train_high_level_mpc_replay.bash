#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"
CPU_THREADS="${DRIBBLEBOT_CPU_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"

usage() {
  cat <<EOF
Usage: $0 [--cuda N] [training options]

Launch replay-first MPC policy training. All unrecognized options are passed
to scripts/train_high_level_mpc_replay.py.

  --cuda N   CUDA device index (default: ${DRIBBLEBOT_CUDA_INDEX:-7})
  -h, --help Show this launcher help

The default can also be set with DRIBBLEBOT_CUDA_INDEX.
EOF
}

CUDA_INDEX="${DRIBBLEBOT_CUDA_INDEX:-7}"
TRAINING_ARGS=()
while (( $# > 0 )); do
  case "$1" in
    --cuda)
      if (( $# < 2 )); then
        echo "Error: --cuda requires a device index." >&2
        usage >&2
        exit 2
      fi
      CUDA_INDEX="$2"
      shift 2
      ;;
    --cuda=*)
      CUDA_INDEX="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TRAINING_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! "${CUDA_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "Error: CUDA device index must be a non-negative integer, got '${CUDA_INDEX}'." >&2
  exit 2
fi

# These match train_high_level_online_mpc.bash, except policy learning uses a
# persistent MPC-teacher replay with repeated KL-distillation updates and no
# on-policy PPO epochs by default.
# The online world-model ensemble is randomly initialized and trained from
# this run's replay. No --world-model-checkpoint is passed.
HIGH_LEVEL_CHECKPOINT="checkpoints/reproduction/high_level/ac_weights_latest.pt"
TERMINAL_VALUE_CHECKPOINT="${TERMINAL_VALUE_CHECKPOINT:-checkpoints/terminal_value_env_v2_bootstrap/best.pt}"
WORLD_MODEL_CONFIG="configs/world_model_as2.yaml"
MPC_CONFIG="configs/mpc_joint_teams.yaml"
WALK_POLICY_DIR="checkpoints/reproduction/walk"
DRIBBLE_POLICY_DIR="checkpoints/reproduction/dribble"
SHOOT_POLICY_DIR="checkpoints/reproduction/shoot"
CHECKPOINT_DIR="${MPC_REPLAY_CHECKPOINT_DIR:-}"

OUTPUT_ARGS=()
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  OUTPUT_ARGS=(--checkpoint-dir "${CHECKPOINT_DIR}")
fi

# Use the same conservative collision shaping and stable role arbitration as
# the online launcher.
exec "${PYTHON_BIN}" scripts/train_high_level_mpc_replay.py \
  --world-model-config "${WORLD_MODEL_CONFIG}" \
  --mpc-config "${MPC_CONFIG}" \
  --mpc-profile teacher_training \
  --num-envs 32 \
  --num-robots 2 \
  --robot-collision-penalty "${ROBOT_COLLISION_PENALTY:-3.0}" \
  --robot-collision-distance "${ROBOT_COLLISION_DISTANCE:-0.70}" \
  --robot-collision-lookahead "${ROBOT_COLLISION_LOOKAHEAD:-0.25}" \
  --attacker-switch-margin "${ATTACKER_SWITCH_MARGIN:-0.15}" \
  --support-command-deadband "${SUPPORT_COMMAND_DEADBAND:-0.08}" \
  --self-play-update-interval "${SELF_PLAY_UPDATE_INTERVAL:-400}" \
  --opponent-pool-size 8 \
  --opponent-latest-probability 0.5 \
  --world-model-update-interval 25 \
  --world-model-replay-buffer-size 100000 \
  --world-model-replay-recent-fraction 0.5 \
  --world-model-replay-recent-window 10000 \
  --mpc-horizon 2 \
  --mpc-num-samples 256 \
  --mpc-num-iterations 4 \
  --terminal-value-checkpoint "${TERMINAL_VALUE_CHECKPOINT}" \
  --mpc-kl-coefficient 0.05 \
  --mpc-guidance-reward-coefficient 0.0 \
  --mpc-warmup-steps "${MPC_WARMUP_STEPS:-2400}" \
  --mpc-min-replay-size 20000 \
  --mpc-min-world-model-updates 48 \
  --mpc-min-terminal-value-updates 4 \
  --on-policy-ppo-epochs 0 \
  --mpc-replay-capacity 100000 \
  --mpc-replay-recent-fraction 0.5 \
  --mpc-replay-recent-window 10000 \
  --mpc-replay-batch-size 1024 \
  --mpc-replay-updates-per-rollout 20 \
  --mpc-replay-min-size 1024 \
  --save-mpc-replay \
  --no-auto-resume-online-replay \
  --headless \
  --policy-device "cuda:${CUDA_INDEX}" \
  --physx-num-threads "${PHYSX_NUM_THREADS:-4}" \
  --save-video-interval "${SAVE_VIDEO_INTERVAL:-0}" \
  --resume \
  --resume-mode full \
  --resume-checkpoint "${HIGH_LEVEL_CHECKPOINT}" \
  --device "cuda:${CUDA_INDEX}" \
  --skill-policy-source local \
  --walk-policy-dir "${WALK_POLICY_DIR}" \
  --dribble-policy-dir "${DRIBBLE_POLICY_DIR}" \
  --shoot-policy-dir "${SHOOT_POLICY_DIR}" \
  "${OUTPUT_ARGS[@]}" \
  "${TRAINING_ARGS[@]}"
