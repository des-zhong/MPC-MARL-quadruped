# Parallel football pitches — Blender preview

36 actual Isaac Lab match environments / 144 AS2 robots. Z-up, metres, 10 m
match spacing. Geometry comes from the current aligned project scene.

- `02_grass_studio.png`: main perspective render, 1800 × 1125.
- `03_overview.png`: overview of all 36 pitches.
- `parallel_pitch_studio.blend`: editable Blender scene with packed field image.
- `parallel_scene.usdc`: flattened static USD with actual GPU link poses baked.
- `snapshot.json`: simulation snapshot metadata and environment origins.
- `render_report.json`: import/render details and limitations.

The pose is from randomized reset followed by one 0.2-second macro step with a
scripted walking command, using the existing frozen low-level skills. It is a
static simulation snapshot, not a trained high-level policy evaluation or an
animation. Source simulation code and physics were not changed for this render.

Presentation adds continuous procedural turf, original texture-derived white
field lines, a black camera background, broad key lights and cool rim/fill.
AS2 shell/foot materials are approximate Principled replacements for MDL that
Blender did not correctly import. Visible USD Cube/Sphere shapes are reconstructed
using their composed world transforms because native import lost intermediate
untyped parents for those shapes. This restores 216 walls and 36 balls.

Rendered successfully using Blender 5.2.2 LTS, Cycles/OptiX, RTX 4090, 48 samples.
No procedural grass blades are included in this first preview.

Reproduce from repository root:

```bash
TASK_OUT="$PWD/isaaclab-rebuild/outputs/blender/parallel_pitch_20260924"
isaaclab-rebuild/.venv/bin/python "$TASK_OUT/export_scene.py" \
  --headless --num_envs 36 --output "$TASK_OUT"
blender --background --factory-startup --python-exit-code 1 \
  --python "$TASK_OUT/build_preview.py"
```

These task scripts use the local project and skill paths. Rerunning replaces
outputs in this directory. The `.blend` contains its imported meshes, materials
and packed field image; it does not need the unsupported MDL shader to render.

## Open-grass variant

`04_open_grass.png` and `05_open_grass_overview.png` hide all 216 visual boundary
walls while retaining the 72 goal meshes, field markings and robots. Physics in
Isaac Lab is unchanged. The original walled scene remains available.

Editable scene: `parallel_pitch_open_grass.blend`.
Reproduce with `blender -b --factory-startup --python-exit-code 1 --python
isaaclab-rebuild/outputs/blender/parallel_pitch_20260924/hide_walls_render.py`.

## Football turf asset variant

`parallel_pitch_football_turf.blend` uses the CC0 [ambientCG Grass004](https://ambientcg.com/view?id=Grass004)
2K PBR asset, with packed color, OpenGL normal and roughness maps. The asset is
procedurally authored, not a photographic scan. It tiles every 1.4 metres in
world coordinates across all fields and the surrounding lawn. The original
field image remains only as a white-line mask. Grass is a shaded surface, without
individual blade geometry.

Each playable 8 × 5 m rectangle has alternating 1 m mowing bands. The field
margins and connecting lawn retain neutral grass shading, so the surface remains
continuous. The node group `Football turf — Grass004 PBR and mowing stripes`
exposes `Mowing Mask` and `Line Mask`; the internal Map Range controls band
brightness (0.65 / 1.30). A green tint and reduced specular level avoid washed-out
turf under the existing studio lights. All 216 boundary wall visuals stay hidden;
simulation physics, field geometry, robot poses and original camera/light setup
are unchanged.

- `06_football_turf.png`: matching main view.
- `07_football_turf_overview.png`: matching overview.
- `08_football_turf_detail.png`: closer field view.
- `football_turf_report.json`: asset and render settings.
- `assets/Grass004/SOURCE.json`: source and license record.

Reproduce with `blender -b --factory-startup --python-exit-code 1 --python
isaaclab-rebuild/outputs/blender/parallel_pitch_20260924/replace_turf_render.py`.

## Continuous mowing across the whole lawn

`parallel_pitch_continuous_turf.blend` extends the same Grass004 PBR material and
mowing pattern across every field, margin and connecting lawn. Both material
variants share one shader group with world-coordinate mapping; only the white
field markings use each field's local UVs. There are no field-local grass or
stripe boundaries. Texture tiles remain 1.4 m; mowing bands remain 1 m wide,
with one shared phase and direction over the entire 200 m lawn.

- `09_continuous_turf.png`: same main camera and lighting as the previous version.
- `10_continuous_turf_overview.png`: overview.
- `11_continuous_turf_detail.png`: close view.
- `continuous_turf_report.json`: settings and packed asset verification.

Reproduce with `blender -b --factory-startup --python-exit-code 1 --python
isaaclab-rebuild/outputs/blender/parallel_pitch_20260924/continuous_turf_render.py`.
Edit the shared shader group's `Mowing band width, metres` Divide node to adjust
stripe spacing throughout the lawn at once. The three PBR images remain packed.

## Files kept in Git

The export/render scripts, this guide, and `assets/Grass004/SOURCE.json` are
versioned. Generated USD, Blender scenes, screenshots, render reports and the
third-party texture package remain local under the ignored `outputs/` directory.
For a fresh checkout, reproduce the export and preview first, then run
`hide_walls_render.py`, `replace_turf_render.py`, and `continuous_turf_render.py`
in that order. Before the turf steps, download `Grass004_2K-JPG.zip` from the
asset page above and extract it into `assets/Grass004/` beside these scripts.
The scripts currently also require the local `isaaclab-blender-render` skill's
`blender_common.py` helper at the path shown in their source.
