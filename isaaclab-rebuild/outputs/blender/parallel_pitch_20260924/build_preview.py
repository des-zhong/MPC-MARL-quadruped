"""Import the parallel USD snapshot and build a continuous-grass studio preview."""
import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector, Matrix
from pxr import Usd, UsdGeom, UsdShade, UsdUtils

SKILL=Path('/home/xander/.codex/skills/isaaclab-blender-render/scripts')
sys.path.insert(0,str(SKILL))
from blender_common import configure_cycles
OUT=Path(__file__).resolve().parent
SOURCE=OUT/'parallel_scene.usdc'
TEXTURE=Path('/home/xander/Code/01_Locomotion/MPC-MARL-quadruped/resources/textures/field.png')

bpy.ops.wm.read_factory_settings(use_empty=True)
stage=Usd.Stage.Open(str(SOURCE))
layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(SOURCE))
print('[PREVIEW] dependencies',len(layers),len(assets),'unresolved',unresolved,flush=True)
result=bpy.ops.wm.usd_import(filepath=str(SOURCE),import_cameras=False,import_lights=False,
    support_scene_instancing=True,import_materials=True,import_usd_preview=True,
    import_proxy=False,import_guide=False,import_visible_only=True,import_shapes=False,merge_parent_xform=False)
assert 'FINISHED' in result
scene=bpy.context.scene
report={'source_usd':str(SOURCE),'blender_version':bpy.app.version_string,
        'up_axis':str(UsdGeom.GetStageUpAxis(stage)),'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),
        'unresolved_dependencies':unresolved,'imported_objects':len(bpy.data.objects),
        'unique_meshes':len(bpy.data.meshes),
        'instanced_collections':len([o for o in bpy.data.objects if o.instance_collection]),
        'pose':'Actual Isaac Lab static simulation snapshot; no generated gait or animation',
        'presentation_changes':'Shared procedural grass, original texture-derived field lines, studio lights; source collision geometry unchanged',
        'materials':'USD Preview imported; scalar OmniPBR fallback mapped approximately to Principled BSDF'}
(OUT/'import_inventory.json').write_text(json.dumps({'objects':[{'name':o.name,'type':o.type,'parent':o.parent.name if o.parent else None,'instance':o.instance_collection.name if o.instance_collection else None} for o in bpy.data.objects], 'materials':[m.name for m in bpy.data.materials]},indent=2))
print('[PREVIEW] imported',report,flush=True)


def material(name,color,roughness=.6,metallic=0):
    mat=bpy.data.materials.new(name); mat.use_nodes=True
    bs=mat.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value=(*color,1)
    bs.inputs['Roughness'].default_value=roughness
    bs.inputs['Metallic'].default_value=metallic
    return mat

# Preserve source material colors, approximating only scalar MDL parameters.
for prim in stage.Traverse():
    if not prim.IsA(UsdShade.Material): continue
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdShade.Shader): continue
        inputs={str(i.GetBaseName()):i.Get() for i in UsdShade.Shader(child).GetInputs()}
        if 'diffuse_color_constant' not in inputs: continue
        mat=bpy.data.materials.get(prim.GetName())
        if mat is None: continue
        mat.use_nodes=True
        bs=next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
        if bs:
            bs.inputs['Base Color'].default_value=(*inputs['diffuse_color_constant'][:3],1)
            bs.inputs['Roughness'].default_value=float(inputs.get('reflection_roughness_constant',.5))
            bs.inputs['Metallic'].default_value=float(inputs.get('metallic_constant',0))


def aim(obj,target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()

world=bpy.data.worlds.new('Black studio with faint ambient');world.use_nodes=True
world.node_tree.nodes['Background'].inputs['Color'].default_value=(.15,.18,.23,1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value=.06
lp=world.node_tree.nodes.new('ShaderNodeLightPath');mix=world.node_tree.nodes.new('ShaderNodeMixShader')
black=world.node_tree.nodes.new('ShaderNodeBackground');black.inputs['Color'].default_value=(0,0,0,1)
world.node_tree.links.new(lp.outputs['Is Camera Ray'],mix.inputs[0])
world.node_tree.links.new(world.node_tree.nodes['Background'].outputs[0],mix.inputs[1])
world.node_tree.links.new(black.outputs[0],mix.inputs[2])
world.node_tree.links.new(mix.outputs[0],world.node_tree.nodes['World Output'].inputs['Surface'])
scene.world=world


def area(name,pos,target,energy,size,color):
    data=bpy.data.lights.new(name,'AREA');data.energy=energy;data.shape='DISK';data.size=size;data.color=color
    ob=bpy.data.objects.new(name,data);scene.collection.objects.link(ob);ob.location=pos;aim(ob,target)

area('Foreground soft key',(24,-30,17),(19,-19,0),14000,13,(1,.95,.86))
area('Cool lateral fill',(3,-27,14),(17,-17,0),4200,17,(.63,.76,1))
area('Rim across foreground',(26,-5,12),(21,-21,0),17000,12,(.86,.94,1))
area('Rear grid canopy',(-13,15,27),(-5,6,0),47000,30,(.83,.90,1))
area('Back edge strip',(10,38,18),(0,15,0),24000,20,(1,1,1))
camdata=bpy.data.cameras.new('Pitch camera');cam=bpy.data.objects.new('Pitch camera',camdata)
scene.collection.objects.link(cam);scene.camera=cam
cam.location=(38,-44,20);aim(cam,(7,-5,0));camdata.lens=35;camdata.clip_end=1000
scene.render.engine='CYCLES';scene.cycles.use_denoising=True;scene.cycles.seed=17
report['device']=configure_cycles(scene,'auto')
scene.cycles.max_bounces=6;scene.cycles.diffuse_bounces=3;scene.cycles.glossy_bounces=3
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.view_settings.view_transform='AgX';scene.view_settings.exposure=0.0
scene.render.resolution_percentage=100
scene.render.resolution_x=1200;scene.render.resolution_y=750;scene.cycles.samples=12
scene.render.filepath=str(OUT/'01_direct_import.png')
print('[PREVIEW] native import render',flush=True)
bpy.ops.render.render(write_still=True)

# One world-coordinate shader shared by the field patches and the continuous lawn.
group=bpy.data.node_groups.new('Continuous turf world coordinates','ShaderNodeTree')
group.interface.new_socket(name='Line Mask',in_out='INPUT',socket_type='NodeSocketFloat')
group.interface.new_socket(name='Shader',in_out='OUTPUT',socket_type='NodeSocketShader')
n=group.nodes;l=group.links
inp=n.new('NodeGroupInput');out=n.new('NodeGroupOutput');geo=n.new('ShaderNodeNewGeometry')
noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=1.5;noise.inputs['Detail'].default_value=3
l.new(geo.outputs['Position'],noise.inputs['Vector'])
ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.2;ramp.color_ramp.elements[0].color=(.012,.028,.007,1)
ramp.color_ramp.elements[1].position=.8;ramp.color_ramp.elements[1].color=(.055,.115,.018,1)
l.new(noise.outputs['Fac'],ramp.inputs['Fac'])
mix=n.new('ShaderNodeMixRGB');mix.blend_type='MIX';mix.inputs[2].default_value=(.76,.80,.69,1)
l.new(inp.outputs['Line Mask'],mix.inputs[0]);l.new(ramp.outputs['Color'],mix.inputs[1])
fine=n.new('ShaderNodeTexNoise');fine.inputs['Scale'].default_value=210;fine.inputs['Detail'].default_value=2
l.new(geo.outputs['Position'],fine.inputs['Vector'])
bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.28;bump.inputs['Distance'].default_value=.004
l.new(fine.outputs['Fac'],bump.inputs['Height'])
bs=n.new('ShaderNodeBsdfPrincipled');bs.inputs['Roughness'].default_value=.90
l.new(mix.outputs['Color'],bs.inputs['Base Color']);l.new(bump.outputs['Normal'],bs.inputs['Normal'])
l.new(bs.outputs['BSDF'],out.inputs['Shader'])


def grass_mat(name,lines=False):
    mat=bpy.data.materials.new(name);mat.use_nodes=True;n=mat.node_tree.nodes;l=mat.node_tree.links;n.clear()
    gn=n.new('ShaderNodeGroup');gn.node_tree=group;out=n.new('ShaderNodeOutputMaterial');l.new(gn.outputs['Shader'],out.inputs['Surface'])
    if lines:
        tex=n.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(TEXTURE),check_existing=True)
        tex.extension='CLIP';tex.interpolation='Linear'
        split=n.new('ShaderNodeSeparateColor');l.new(tex.outputs['Color'],split.inputs['Color'])
        ab=n.new('ShaderNodeMath');ab.operation='MINIMUM';l.new(split.outputs['Red'],ab.inputs[0]);l.new(split.outputs['Green'],ab.inputs[1])
        abc=n.new('ShaderNodeMath');abc.operation='MINIMUM';l.new(ab.outputs[0],abc.inputs[0]);l.new(split.outputs['Blue'],abc.inputs[1])
        remap=n.new('ShaderNodeMapRange');remap.inputs['From Min'].default_value=.38;remap.inputs['From Max'].default_value=.72
        l.new(abc.outputs[0],remap.inputs['Value']);l.new(remap.outputs['Result'],gn.inputs['Line Mask'])
    return mat

groundmat=grass_mat('Continuous grass');fieldmat=grass_mat('Continuous grass with original field markings',True)


def ancestry(obj):
    names=[]
    while obj:
        names.append(obj.name.split('.')[0]);obj=obj.parent
    return names

fieldcount=0
for obj in list(bpy.data.objects):
    names=ancestry(obj)
    if 'ground' in names:
        obj.hide_render=True
    if obj.type=='MESH' and 'SoccerFieldVisual' in names:
        obj.data.materials.clear();obj.data.materials.append(fieldmat);fieldcount+=1
assert fieldcount==36, f'Expected 36 imported field meshes, found {fieldcount}'
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.001))
ground=bpy.context.object;ground.name='Continuous grass across all 36 pitches';ground.data.materials.append(groundmat)
report['grass_patches_rebound']=fieldcount
# Blender loses untyped intermediate parents of some USD basic shapes.
# Rebuild visible Cube/Sphere geometry using its composed USD world transform.
# AS2 meshes remain imported, with shared geometry/collection instances.
white=material('AS2 silver approximation',(.58,.64,.73),.42,.12)
red=material('AS2 opponent red approximation',(.50,.025,.016),.45,.08)
foot=material('AS2 dark feet',(.025,.028,.032),.72,0)
wallmat=material('Original wall ivory',(.70,.70,.65),.72,0)
yellow=material('Football yellow',(1,.72,.008),.47,0)

def set_mat(obj,mat):
    if not len(obj.data.materials):obj.data.materials.append(mat)
    for slot in obj.material_slots:
        slot.link='OBJECT';slot.material=mat

cache={}
def tint_collection(source,mat):
    key=(source.name,mat.name)
    if key in cache:return cache[key]
    col=bpy.data.collections.new(source.name+'_'+mat.name)
    col.instance_offset=source.instance_offset
    cache[key]=col;copies={}
    for ob in source.objects:
        copy=ob.copy();col.objects.link(copy);copies[ob]=copy
        if copy.type=='MESH':set_mat(copy,mat)
        if copy.instance_collection:copy.instance_collection=tint_collection(copy.instance_collection,mat)
    for old,copy in copies.items():
        if old.parent in copies:copy.parent=copies[old.parent]
    for child in source.children:col.children.link(tint_collection(child,mat))
    return col

for obj in list(scene.objects):
    names=ancestry(obj)
    team=next((name for name in names if name in ('Robot_0','Robot_1','Robot_2','Robot_3')),None)
    if team:
        mat=red if team in ('Robot_2','Robot_3') else white
        if any('foot' in name for name in names):mat=foot
        if obj.type=='MESH':set_mat(obj,mat)
        if obj.instance_collection:obj.instance_collection=tint_collection(obj.instance_collection,mat)

bpy.ops.mesh.primitive_cube_add(size=2)
ob=bpy.context.object;cube=ob.data;bpy.data.objects.remove(ob,do_unlink=True)
bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=1)
ob=bpy.context.object;sphere=ob.data
for face in sphere.polygons:face.use_smooth=True
bpy.data.objects.remove(ob,do_unlink=True)
shape_count=0
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    if not (prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Sphere)):continue
    path=str(prim.GetPath())
    if '/collisions/' in path or '/Collider/' in path or path.endswith('/Collider'):continue
    imageable=UsdGeom.Imageable(prim)
    if imageable.ComputeVisibility()=='invisible' or imageable.ComputePurpose() in ('proxy','guide'):continue
    data=cube if prim.IsA(UsdGeom.Cube) else sphere
    ob=bpy.data.objects.new('USDShape_'+path.replace('/','_'),data);scene.collection.objects.link(ob)
    matrix=UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    ob.matrix_world=Matrix([list(row) for row in matrix]).transposed()
    scale=float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2 if prim.IsA(UsdGeom.Cube) else float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
    ob.scale*=scale
    mat=yellow if '/Ball/' in path else wallmat
    if '/Robot_' in path:
        mat=red if ('/Robot_2/' in path or '/Robot_3/' in path) else white
        if 'foot' in path:mat=foot
    set_mat(ob,mat);shape_count+=1
report['visible_usd_shapes_reconstructed']=shape_count
report['robot_materials']='Approximate Principled silver/red shells and dark feet, because MDL did not import correctly'
# Slightly darker, richer grass and more directional contrast than the first draft.
ramp.color_ramp.elements[0].color=(.007,.018,.003,1)
ramp.color_ramp.elements[1].color=(.026,.066,.009,1)
for ob in scene.objects:
    if ob.type=='LIGHT':ob.data.energy*=.50
cam.location=(32,-36,12);aim(cam,(10,-5,0));camdata.lens=32

scene.render.resolution_x=1800;scene.render.resolution_y=1125;scene.cycles.samples=48
scene.render.filepath=str(OUT/'02_grass_studio.png')
print('[PREVIEW] adapted grass render',flush=True)
bpy.ops.render.render(write_still=True)
# Second camera makes the actual extent and independently positioned envs inspectable.
cam.location=(69,-73,61);aim(cam,(0,0,0));camdata.lens=43
scene.render.filepath=str(OUT/'03_overview.png')
bpy.ops.render.render(write_still=True)
cam.location=(32,-36,12);aim(cam,(10,-5,0));camdata.lens=32
scene.render.filepath=str(OUT/'02_grass_studio.png')
for ob in scene.objects:ob.select_set(False)
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA'
bpy.ops.file.pack_all()
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'parallel_pitch_studio.blend'))
report['renders']=['01_direct_import.png','02_grass_studio.png','03_overview.png']
report['render_resolution']=[1800,1125];report['samples']=48
(OUT/'render_report.json').write_text(json.dumps(report,indent=2))
print('[PREVIEW] complete',report,flush=True)
