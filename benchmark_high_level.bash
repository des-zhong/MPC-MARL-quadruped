#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON_BIN="${DRIBBLEBOT_PYTHON:-/home/zhz/anaconda3/envs/legged_env/bin/python}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/dribblebot_torch_extensions}"

# Uses every complete numbered learner checkpoint as a candidate. The fixed
# opponent set is selected from the early, middle, and final learner snapshots.
POLICY_DIR="wandb/run-20260817_191642-yz0y52ly/files/tmp/legged_data/high_level"
OUTPUT_DIR="outputs/high_level_benchmark"
DEVICE="${DEVICE:-cuda:0}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"

exec "${PYTHON_BIN}" scripts/benchmark_high_level_checkpoints.py \
  --policy-dir "${POLICY_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-robots 2 \
  --device "${DEVICE}" \
  --policy-device "${POLICY_DEVICE}"
