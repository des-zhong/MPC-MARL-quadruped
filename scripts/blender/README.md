# Trained-policy football film

This pipeline runs the committed Isaac Gym high-level policy and its three
frozen skills, records link poses at 50 Hz, and bakes them into Blender.
Blender does not execute the policy or replace the original physics engine.

The default film is 10 seconds, 24 FPS, 1280 × 720, rendered with Cycles/OptiX.
It shows 36 separate matches (144 AS2 robots). Each match is an independently
trimmed, reset-free excerpt from a different simulated environment; their source
times differ. Neither poses nor playback speed are procedurally generated.
Both teams use the trained high-level policy. The camera follows the foreground
match, then rises and pulls back to show the field array.

Run from the repository root:

```bash
bash scripts/blender/demo_policy_video.sh
```

This uses the root `.venv` (Python 3.8 / Isaac Gym), local reproduction checkpoints,
Blender 5.2, FFmpeg, and the local `isaaclab-blender-render` skill helpers.
The presentation scene is the previously generated
`isaaclab-rebuild/outputs/blender/parallel_pitch_20260924/parallel_pitch_continuous_turf.blend`.
That generated binary is not tracked in Git. Supply another copy as the second
argument and an output directory as the first. The base scene must retain the
36-field object names and accompanying `snapshot.json` layout metadata.

Outputs under `outputs/blender/policy_match_20260924/`:

- `policy_match_camera.mp4`: finished video, no audio.
- `policy_match_camera.blend`: packed materials and ordinary baked animation
  curves, with no external NPZ dependency or runtime frame handler.
- `rollout.npz` / `rollout.json`: full recorded trajectories, reset episode IDs,
  coordinate convention, seed, policy hashes and root/body frame validation.
- `clips_24fps.npz` / `clip_selection.json`: interpolation and source clip mapping.
- `animation_report.json`, `motion_validation.json`, `video_verification.json`:
  rendering and motion/video checks.
- `frames/`: resumable PNG sequence; `preview_*.png`: framing checks.

To resume only rendering, run:

```bash
blender -b --factory-startup --python-exit-code 1 \
  --python scripts/blender/render_policy_film.py -- \
  --output outputs/blender/policy_match_20260924
python "$HOME/.codex/skills/isaaclab-blender-render/scripts/encode_video.py" \
  --output outputs/blender/policy_match_20260924 --name policy_match_camera.mp4
```

Use a fresh output directory when changing the policy or scene; the renderer
skips existing PNGs. The original 17 STL visual links share their mesh data.
World translations are interpolated linearly, rotations by shortest-arc SLERP,
and source environment offsets are replaced by the Blender pitch layout.
Independent world-link interpolation introduces only millimetre-scale residual
fixed-joint error in this recording. Robot materials remain the previous
approximate silver/red shading. Boundary walls remain active in physics and
hidden in the presentation scene.

## Localized blue/orange team colors

The approved material variant keeps the limbs gray and feet dark, and colors
only the back and left/right body flanks. The mask uses each base link's local
coordinates, so the panels follow the recorded motion. It also uses 35-degree
sharp edges on the shared STL meshes.

After generating the blue single-robot preview and original film, run:

```bash
blender -b --python-exit-code 1 --python scripts/blender/apply_team_colors.py
blender -b --python-exit-code 1 --python scripts/blender/render_policy_film.py -- \
  --output outputs/blender/policy_match_team_colors_20260924
python "$HOME/.codex/skills/isaaclab-blender-render/scripts/encode_video.py" \
  --output outputs/blender/policy_match_team_colors_20260924 --name policy_match_camera.mp4
```

`apply_team_colors.py` accepts `--source`, `--output`, and `--materials` paths.
It appends the approved materials from the single-robot `.blend`, preserves the
baked animation, camera and lighting, and writes a new scene and three previews.
The generated source `.blend` files are local artifacts, not tracked assets.

## 10 × 10 grid with staged camera motion

The current `build_grid_film.py` builds 100 distinct matches / 400 robots from
the approved blue/orange scene. In a top view, +X points right and +Y points up.
The default opening is `(2, 2)` counting from the **lower-left** corner, at world
`(-35, -35)`, diagonally opposite the previous version's upper-right opening.

The current 13.25-second camera timeline is:

- 0–3 s: fixed position, orientation and 36 mm focal length.
- 3–7 s: level backward translation, +50 m in X and +50 m in Y, passing five
  diagonal grid intervals. Quadratic ease-in (`u**2`) increases speed throughout;
  orientation stays fixed.
- 7–9.25 s: continue backward while rising to 30 m. The Hermite position curve
  matches the incoming velocity, then decelerates to a stop.
- 9.25–13.25 s: fixed ending camera for four seconds; matches keep playing.

The ending crops the outer grid so fields fill the entire image. Camera corner
rays are checked to hit the ground inside the grid. It does not try to include
all 100 pitches with a border around them.

Current output: `outputs/blender/policy_grid10_accelerating_camera_20260924/`.
The 100 independent 13.25-second clips are selected from the existing
192-environment, 40-second recording in `policy_grid10_20260924`. No match
animation is duplicated or frozen during the camera hold. To rebuild the current version:

```bash
python scripts/blender/select_policy_clips.py \
  --recording outputs/blender/policy_grid10_20260924 \
  --output outputs/blender/policy_grid10_accelerating_camera_20260924 \
  --num-envs 100 --seconds 13.25
blender -b -t 16 --python-exit-code 1 --python scripts/blender/build_grid_film.py -- \
  --output outputs/blender/policy_grid10_accelerating_camera_20260924
blender -b -t 16 --python-exit-code 1 --python scripts/blender/render_policy_film.py -- \
  --output outputs/blender/policy_grid10_accelerating_camera_20260924
python "$HOME/.codex/skills/isaaclab-blender-render/scripts/encode_video.py" \
  --output outputs/blender/policy_grid10_accelerating_camera_20260924 --name policy_match_camera.mp4
```

Use a fresh output directory when changing the camera; copy `clips_24fps.npz`,
`clip_selection.json` and `motion_validation.json` into it before building.
The renderer resumes existing PNG frames, so old frames should not be mixed
with a newly built scene.

Camera controls include `--hero-corner`, `--hero-row`, `--hero-col`,
`--hold-seconds`, `--flat-envs`, `--flat-end-seconds`, `--move-end-seconds`,
and `--end-height`. The start/end positions and Hermite curve are grouped under
`Camera controls` inside `build_grid_film.py`.

Robot materials are bound to 18 mesh variants (two base colors plus 16 limb
meshes), rather than object-level material overrides. Cycles instances the
geometry instead of rebuilding thousands of separate meshes on every frame.
The accepted materials and 32-sample quality are retained; off-camera objects
use a 25% culling margin. Grass and studio lighting cover the full field array.

## Isolated skill studio previews

`record_skill_rollout.py` wraps the existing single-ability validator and records
pre-reset link/ball poses at 50 Hz. It maps the historical checkpoint asset path
to this repository's AS2 URDF. Walk, dribble and shoot run in separate simulators.
`build_skill_studio.py` bakes a continuous six-second excerpt at 30 FPS into
`walk.blend`, `dribble.blend`, or `shoot.blend` and renders framing checks.

Output: `outputs/blender/skills_studio_neutral_20260924/`. The accepted dribble case is
`trial_f`: body-frame command `(0.2, 0, 0)`, ball initially `(0.35, 0.08, 0.1)`.
It passed the existing continuous-control checks. Repositioning uses walking
command `(0.45, 0.2, 0.35)`; kicking uses world-frame ball command `(3, 0)` and
initial ball `(0.45, 0, 0.1)`. These are true trained-policy executions; camera
tracking only changes the presentation. No motion is synthesized in Blender.

The floor material and composite backdrop use sRGB `(253,250,244)` / `#FDFAF4`,
converted to scene-linear values. Standard display transform preserves the
chosen backdrop; a shadow catcher supplies soft contact shadows, whose pixels
are naturally darker. All three previews use a neutral gray shell and limbs with dark feet, without
team-colored panels. The original real-motion recordings are reused unchanged.

For example, after recording:

```bash
blender -b --python-exit-code 1 --python scripts/blender/build_skill_studio.py -- \
  --output outputs/blender/skills_studio_neutral_20260924 --skill dribble
blender -b --python-exit-code 1 --python scripts/blender/render_policy_film.py -- \
  --output outputs/blender/skills_studio_neutral_20260924/dribble \
  --blend outputs/blender/skills_studio_neutral_20260924/dribble/dribble.blend
python "$HOME/.codex/skills/isaaclab-blender-render/scripts/encode_video.py" \
  --output outputs/blender/skills_studio_neutral_20260924/dribble --name dribbling.mp4
```

Encode the other abilities as `walk/repositioning.mp4` and `shoot/kicking.mp4`,
then run `compose_skill_panels.py --output outputs/blender/skills_studio_neutral_20260924`
for the three-panel video and poster.
