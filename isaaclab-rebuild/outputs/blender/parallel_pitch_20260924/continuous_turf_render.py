"""Extend the same PBR turf and mowing bands continuously across the entire lawn."""
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector
OUT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/xander/.codex/skills/isaaclab-blender-render/scripts')
from blender_common import configure_cycles
bpy.ops.wm.open_mainfile(filepath=str(OUT/'parallel_pitch_football_turf.blend'))
scene=bpy.context.scene
group=bpy.data.node_groups['Football turf — Grass004 PBR and mowing stripes']
group.name='Continuous football turf — world-space PBR and mowing'
n=group.nodes;l=group.links
geo=next(v for v in n if v.type=='NEW_GEOMETRY')
stripe=next(v for v in n if v.label=='Dark / light mowing brightness')
separate=n.new('ShaderNodeSeparateXYZ');separate.label='Global mowing axis';separate.location=(-1100,700)
l.new(geo.outputs['Position'],separate.inputs[0])
phase=n.new('ShaderNodeMath');phase.operation='ADD';phase.inputs[1].default_value=1;phase.label='Shared stripe phase, metres';phase.location=(-880,700)
l.new(separate.outputs['X'],phase.inputs[0])
width=n.new('ShaderNodeMath');width.operation='DIVIDE';width.inputs[1].default_value=1; width.label='Mowing band width, metres';width.location=(-660,700)
l.new(phase.outputs[0],width.inputs[0])
floor=n.new('ShaderNodeMath');floor.operation='FLOOR';floor.location=(-440,700)
l.new(width.outputs[0],floor.inputs[0])
mod=n.new('ShaderNodeMath');mod.operation='FLOORED_MODULO';mod.inputs[1].default_value=2;mod.location=(-220,700)
l.new(floor.outputs[0],mod.inputs[0]);l.new(mod.outputs[0],stripe.inputs['Value'])
# Eliminate the old field-local stripe controls. White line masks remain local.
for socket in list(group.interface.items_tree):
    if socket.item_type=='SOCKET' and socket.name=='Mowing Mask':group.interface.remove(socket)
for name in ('Continuous grass','Continuous grass with original field markings'):
    mat=bpy.data.materials[name];nodes=mat.node_tree.nodes
    used=set()
    def visit(v):
        if v in used:return
        used.add(v)
        for inp in v.inputs:
            for link in inp.links:visit(link.from_node)
    for v in nodes:
        if v.type=='OUTPUT_MATERIAL':visit(v)
    for v in list(nodes):
        if v not in used:nodes.remove(v)
assert stripe.inputs['Value'].links[0].from_node==mod
assert sum(ob.hide_render for ob in scene.objects if ob.name.startswith('USDShape_') and '_Wall' in ob.name)==216
images=[i for i in bpy.data.images if 'Grass004_2K' in i.name]
assert len(images)==3 and all(i.packed_file for i in images)
scene.cycles.samples=64
device=configure_cycles(scene,'auto')
cam=scene.camera
def camera(pos,target,lens):
    cam.location=pos;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=lens
def render(name):
    scene.render.filepath=str(OUT/name);bpy.ops.render.render(write_still=True)
camera((32,-36,12),(10,-5,0),32);render('09_continuous_turf.png')
camera((69,-73,61),(0,0,0),43);render('10_continuous_turf_overview.png')
camera((31,-31,5),(24,-24,0),40);render('11_continuous_turf_detail.png')
camera((32,-36,12),(10,-5,0),32)
scene.render.filepath=str(OUT/'09_continuous_turf.png')
bpy.ops.file.pack_all()
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'parallel_pitch_continuous_turf.blend'))
(OUT/'continuous_turf_report.json').write_text(json.dumps({'source_blend':'parallel_pitch_football_turf.blend','asset':'ambientCG Grass004 2K-JPG','mapping':'Shared world coordinates for all PBR textures and mowing stripes across fields, margins and lawn','tile_m':1.4,'stripe_width_m':1,'stripe_phase_m':1,'white_line_masks_preserved':True,'packed_pbr_maps':[i.name for i in images],'hidden_walls':216,'physics_modified':False,'device':device},indent=2))
print('CONTINUOUS_TURF_COMPLETE',flush=True)
