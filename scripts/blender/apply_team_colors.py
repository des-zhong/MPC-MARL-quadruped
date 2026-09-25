"""Apply the approved dorsal/flank colors to the baked policy movie."""
import argparse
import json
import re
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / '.codex/skills/isaaclab-blender-render/scripts'))
from blender_common import configure_cycles

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=ROOT / 'outputs/blender/policy_match_20260924')
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/blender/policy_match_team_colors_20260924')
parser.add_argument('--materials', type=Path, default=ROOT / 'outputs/blender/as2_team_color_preview/as2_blue_back_side.blend')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
(out / 'frames').mkdir(exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(args.source / 'policy_match_camera.blend'))
scene = bpy.context.scene
names = ['AS2 neutral shell with localized team color', 'AS2 neutral limbs', 'AS2 dark feet']
with bpy.data.libraries.load(str(args.materials), link=False) as (available, loaded):
    assert all(name in available.materials for name in names)
    loaded.materials = names
blue, neutral, rubber = loaded.materials
blue.name = 'AS2 blue back and flanks'
orange = blue.copy()
orange.name = 'AS2 orange back and flanks'
orange.node_tree.nodes['Team Color'].outputs[0].default_value = (.95, .19, .025, 1)
counts = {'links': 0, 'blue_bodies': 0, 'orange_bodies': 0, 'feet': 0}
meshes = set()
actions_before = {ob.name: ob.animation_data.action.name for ob in scene.objects if ob.animation_data and ob.animation_data.action}
for ob in scene.objects:
    match = re.fullmatch(r'Rollout_E\d+_R([0-3])_(.+)', ob.name)
    if not match:
        continue
    robot, link = int(match[1]), match[2]
    if link == 'base_link':
        material = blue if robot < 2 else orange
        counts['blue_bodies' if robot < 2 else 'orange_bodies'] += 1
    elif link.endswith('_foot'):
        material = rubber
        counts['feet'] += 1
    else:
        material = neutral
    assert ob.material_slots, ob.name
    for slot in ob.material_slots:
        slot.link = 'OBJECT'
        slot.material = material
    if ob.data.name not in meshes:
        for polygon in ob.data.polygons:
            polygon.use_smooth = True
        ob.data.set_sharp_from_angle(angle=.610865)
        meshes.add(ob.data.name)
    counts['links'] += 1
assert counts == {'links': 2448, 'blue_bodies': 72, 'orange_bodies': 72, 'feet': 576}, counts
assert actions_before == {ob.name: ob.animation_data.action.name for ob in scene.objects if ob.animation_data and ob.animation_data.action}
configure_cycles(scene, 'auto')
scene.cycles.denoising_use_gpu = True
scene.render.use_persistent_data = True
scene.frame_set(1)
scene.render.filepath = str(out / 'frames/frame_')
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'policy_match_camera.blend'))
report = json.loads((args.source / 'animation_report.json').read_text())
report['robot_materials'] = 'Neutral gray shell and limbs, dark feet; blue/orange only on dorsal and flank body panels'
report['material_source'] = str(args.materials.resolve())
report['animation_source'] = str((args.source / 'policy_match_camera.blend').resolve())
report['team_color_validation'] = {**counts, 'shared_meshes': len(meshes), 'animated_objects_preserved': len(actions_before)}
(out / 'animation_report.json').write_text(json.dumps(report, indent=2))
for frame in (1, 120, 240):
    scene.frame_set(frame)
    scene.render.filepath = str(out / f'preview_{frame:04}.png')
    bpy.ops.render.render(write_still=True)
print('TEAM_COLORS_APPLIED', counts, flush=True)
