#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PROJECT_ROOT}/train_world_model_pipeline.bash" terminal_final "$@"
