#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/.venv/bin/activate"
FILM_OUT="${1:-$REPO_DIR/outputs/blender/policy_match_20260924}"
BASE_SCENE="${2:-$REPO_DIR/isaaclab-rebuild/outputs/blender/parallel_pitch_20260924/parallel_pitch_continuous_turf.blend}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
if [[ ! -f "$BASE_SCENE" ]]; then
  echo "Missing base scene: $BASE_SCENE (see scripts/blender/README.md)" >&2
  exit 1
fi
python scripts/blender/record_policy_rollout.py \
  --rollout-output "$FILM_OUT" \
  --high-level-policy-source local \
  --high-level-policy-dir checkpoints/reproduction/high_level \
  --skill-policy-source local \
  --walk-policy-dir checkpoints/reproduction/walk \
  --dribble-policy-dir checkpoints/reproduction/dribble \
  --shoot-policy-dir checkpoints/reproduction/shoot \
  --training-environment --num-envs 64 --num-robots 2 \
  --device cuda:0 --steps 150 --seed 17 --headless --no-video --no-plot \
  --csv "$FILM_OUT/metrics.csv"
python scripts/blender/select_policy_clips.py --output "$FILM_OUT"
"$BLENDER_BIN" -b --factory-startup --python-exit-code 1 \
  --python scripts/blender/build_policy_film.py -- \
  --output "$FILM_OUT" --base-blend "$BASE_SCENE"
# The renderer resumes existing frame sequences. Use a new output directory
# when changing policy, scene, camera or render settings.
"$BLENDER_BIN" -b --factory-startup --python-exit-code 1 \
  --python scripts/blender/render_policy_film.py -- --output "$FILM_OUT"
python "$HOME/.codex/skills/isaaclab-blender-render/scripts/encode_video.py" \
  --output "$FILM_OUT" --name policy_match_camera.mp4
