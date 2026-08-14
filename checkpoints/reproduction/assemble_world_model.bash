#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORLD_MODEL_DIR="${SCRIPT_DIR}/world_model"
TARGET="${WORLD_MODEL_DIR}/best.pt"
EXPECTED_SHA256="da6d12ee2cb59e09d98cfc6b14f80b11843072755d186e1fc02e1a3886f4ebb3"
PARTS=(
  "${WORLD_MODEL_DIR}/best.pt.part-00"
  "${WORLD_MODEL_DIR}/best.pt.part-01"
  "${WORLD_MODEL_DIR}/best.pt.part-02"
)

if [[ -f "${TARGET}" ]] && printf '%s  %s\n' "${EXPECTED_SHA256}" "${TARGET}" | sha256sum --check --status; then
  echo "World-model checkpoint is already assembled and verified: ${TARGET}"
  exit 0
fi

for part in "${PARTS[@]}"; do
  if [[ ! -f "${part}" ]]; then
    echo "Missing checkpoint part: ${part}" >&2
    echo "Run 'git lfs pull' before assembling the world model." >&2
    exit 1
  fi
done

TEMP_TARGET="${TARGET}.tmp.$$"
trap 'rm -f -- "${TEMP_TARGET}"' EXIT
cat "${PARTS[@]}" > "${TEMP_TARGET}"
printf '%s  %s\n' "${EXPECTED_SHA256}" "${TEMP_TARGET}" | sha256sum --check --status
mv -- "${TEMP_TARGET}" "${TARGET}"
trap - EXIT

echo "Assembled and verified world-model checkpoint: ${TARGET}"
