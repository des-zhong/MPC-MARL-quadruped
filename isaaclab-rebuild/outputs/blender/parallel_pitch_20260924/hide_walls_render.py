"""Create an open-grass presentation variant without changing simulation physics."""
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector

OUT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/xander/.codex/skills/isaaclab-blender-render/scripts')
from blender_common import configure_cycles
bpy.ops.wm.open_mainfile(filepath=str(OUT/'parallel_pitch_studio.blend'))
scene=bpy.context.scene
walls=[ob for ob in scene.objects if ob.name.startswith('USDShape_') and '_Wall' in ob.name]
assert len(walls)==216, f'Expected 216 boundary walls, got {len(walls)}'
for ob in walls:
    ob.hide_render=True
    ob.hide_set(True)
scene.cycles.samples=48
device=configure_cycles(scene,'auto')
cam=scene.camera

def aim(target):
    cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler()

cam.location=(32,-36,12);aim((10,-5,0));cam.data.lens=32
scene.render.filepath=str(OUT/'04_open_grass.png')
bpy.ops.render.render(write_still=True)
cam.location=(69,-73,61);aim((0,0,0));cam.data.lens=43
scene.render.filepath=str(OUT/'05_open_grass_overview.png')
bpy.ops.render.render(write_still=True)
cam.location=(32,-36,12);aim((10,-5,0));cam.data.lens=32
scene.render.filepath=str(OUT/'04_open_grass.png')
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'parallel_pitch_open_grass.blend'))
(OUT/'open_grass_report.json').write_text(json.dumps({'hidden_boundary_walls':len(walls),'goals_retained':72,'num_envs':36,'source_physics_modified':False,'device':device,'variant':'Blender boundary-wall geometry hidden in render and viewport; goal meshes and white field lines retained'},indent=2))
print('OPEN_GRASS_COMPLETE',len(walls),device,flush=True)
