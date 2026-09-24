"""Use a packed CC0 PBR turf asset and one-metre football mowing stripes."""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import bpy
from mathutils import Vector
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
ASSET=OUT/'assets/Grass004'
sys.path.insert(0,'/home/xander/.codex/skills/isaaclab-blender-render/scripts')
from blender_common import configure_cycles
bpy.ops.wm.open_mainfile(filepath=str(OUT/'parallel_pitch_open_grass.blend'))
scene=bpy.context.scene
group=bpy.data.node_groups['Continuous turf world coordinates']
group.name='Football turf — Grass004 PBR and mowing stripes'
slot=group.interface.new_socket(name='Mowing Mask',in_out='INPUT',socket_type='NodeSocketFloat')
slot.default_value=.5
n=group.nodes;l=group.links;n.clear()
def node(kind,name,x,y):
    v=n.new(kind);v.label=name;v.location=(x,y);return v
inp=node('NodeGroupInput','Field markings and mowing mask',-700,450)
out=node('NodeGroupOutput','Turf surface',1100,150)
geo=node('ShaderNodeNewGeometry','Continuous world-space mapping',-1100,0)
coords=node('ShaderNodeVectorMath','Asset physical tile: 1.4 metres',-880,0);coords.operation='SCALE';coords.inputs['Scale'].default_value=1/1.4
l.new(geo.outputs['Position'],coords.inputs[0])
def texture(suffix,color,y):
    v=node('ShaderNodeTexImage',suffix,-650,y)
    v.image=bpy.data.images.load(str(ASSET/f'Grass004_2K-JPG_{suffix}.jpg'),check_existing=True)
    v.image.colorspace_settings.name='sRGB' if color else 'Non-Color'
    v.extension='REPEAT';v.interpolation='Linear'
    l.new(coords.outputs['Vector'],v.inputs['Vector'])
    return v
albedo=texture('Color',True,100)
normal=texture('NormalGL',False,-180)
rough=texture('Roughness',False,-450)
tint=node('ShaderNodeMixRGB','Fresh football lawn tint',-360,180);tint.blend_type='MULTIPLY';tint.inputs[0].default_value=1;tint.inputs[2].default_value=(.28,.62,.22,1)
l.new(albedo.outputs['Color'],tint.inputs[1])
stripe=node('ShaderNodeMapRange','Dark / light mowing brightness',-350,440);stripe.inputs['To Min'].default_value=.65;stripe.inputs['To Max'].default_value=1.3
l.new(inp.outputs['Mowing Mask'],stripe.inputs['Value'])
shade=node('ShaderNodeMixRGB','Mowing direction appearance',-80,220);shade.blend_type='MULTIPLY';shade.inputs[0].default_value=1
l.new(tint.outputs[0],shade.inputs[1]);l.new(stripe.outputs['Result'],shade.inputs[2])
paint=node('ShaderNodeMixRGB','Original white field markings',180,220);paint.inputs[2].default_value=(.76,.80,.69,1)
l.new(inp.outputs['Line Mask'],paint.inputs[0]);l.new(shade.outputs[0],paint.inputs[1])
normalmap=node('ShaderNodeNormalMap','Grass blade micro normals',-70,-140);normalmap.space='WORLD';normalmap.inputs['Strength'].default_value=.6
l.new(normal.outputs['Color'],normalmap.inputs['Color'])
roughmap=node('ShaderNodeMapRange','Matte short turf',-70,-380);roughmap.inputs['To Min'].default_value=.65;roughmap.inputs['To Max'].default_value=.95
l.new(rough.outputs['Color'],roughmap.inputs['Value'])
bs=node('ShaderNodeBsdfPrincipled','Short grass PBR',650,170)
bs.inputs['Specular IOR Level'].default_value=.2
l.new(paint.outputs[0],bs.inputs['Base Color']);l.new(normalmap.outputs['Normal'],bs.inputs['Normal']);l.new(roughmap.outputs['Result'],bs.inputs['Roughness']);l.new(bs.outputs[0],out.inputs['Shader'])
# Both surfaces share the same PBR coordinates. Only playable fields get bands.
fm=bpy.data.materials['Continuous grass with original field markings'];fn=fm.node_tree.nodes;fl=fm.node_tree.links
gn=next(v for v in fn if v.type=='GROUP');gn.inputs['Mowing Mask'].default_value=.5
field_size=tuple(float(v) for v in ET.parse(ROOT/'resources/objects/soccer_field/soccer_field.urdf').find('.//collision/geometry/box').get('size').split())
uv=fn.new('ShaderNodeTexCoord');uv.label='Field-local mowing layout'
xy=fn.new('ShaderNodeSeparateXYZ');fl.new(uv.outputs['UV'],xy.inputs[0])
def mathnode(op,a,b=None,label=''):
    v=fn.new('ShaderNodeMath');v.operation=op;v.label=label
    if isinstance(a,(float,int)):v.inputs[0].default_value=a
    else:fl.new(a,v.inputs[0])
    if b is not None:
        if isinstance(b,(float,int)):v.inputs[1].default_value=b
        else:fl.new(b,v.inputs[1])
    return v.outputs[0]
x=mathnode('MULTIPLY_ADD',xy.outputs['X'],field_size[0])
# Convert UV to metres measured from the left painted touchline, x=-4.
x.node.inputs[2].default_value=4-field_size[0]/2
band=mathnode('FLOORED_MODULO',mathnode('FLOOR',x),2.0,'One metre alternating stripes')
# Trim the mowing pattern at the playable rectangle; neutral grass in margins.
inside_x=mathnode('MULTIPLY',mathnode('GREATER_THAN',x,0),mathnode('LESS_THAN',x,8))
y=mathnode('MULTIPLY_ADD',xy.outputs['Y'],field_size[1]);y.node.inputs[2].default_value=2.5-field_size[1]/2
inside_y=mathnode('MULTIPLY',mathnode('GREATER_THAN',y,0),mathnode('LESS_THAN',y,5))
inside=mathnode('MULTIPLY',inside_x,inside_y)
mask=mathnode('ADD',mathnode('MULTIPLY',mathnode('SUBTRACT',band,.5),inside),.5)
fl.new(mask,gn.inputs['Mowing Mask'])
# Arrange the compact material's nodes so the asset is editable after opening.
for i,v in enumerate(fn):v.location=((i%5)*230,-(i//5)*200)
assert sum(ob.hide_render for ob in scene.objects if ob.name.startswith('USDShape_') and '_Wall' in ob.name)==216
scene.cycles.samples=64
scene.cycles.use_denoising=True
device=configure_cycles(scene,'auto')
cam=scene.camera

def camera(pos,target,lens):
    cam.location=pos;cam.rotation_euler=(Vector(target)-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.lens=lens

def render(name):
    scene.render.filepath=str(OUT/name);bpy.ops.render.render(write_still=True)
camera((32,-36,12),(10,-5,0),32)
render('06_football_turf.png')
camera((69,-73,61),(0,0,0),43)
render('07_football_turf_overview.png')
camera((31,-31,5),(24,-24,0),40)
render('08_football_turf_detail.png')
camera((32,-36,12),(10,-5,0),32)
scene.render.filepath=str(OUT/'06_football_turf.png')
bpy.ops.file.pack_all()
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'parallel_pitch_football_turf.blend'))
report={'asset':'ambientCG Grass004 2K-JPG','source':'https://ambientcg.com/view?id=Grass004','license':'CC0','technique':'procedural PBR asset plus shader mowing bands','packed_maps':['Color','NormalGL','Roughness','original field line mask'],'physical_tile_m':1.4,'mowing_band_width_m':1,'mowing_brightness':[.65,1.3],'num_envs':36,'hidden_walls':216,'geometry_or_physics_changed':False,'device':device,'samples':64,'field_size':field_size}
(OUT/'football_turf_report.json').write_text(json.dumps(report,indent=2))
print('FOOTBALL_TURF_COMPLETE',flush=True)
