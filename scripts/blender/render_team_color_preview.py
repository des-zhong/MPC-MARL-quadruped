"""Single-AS2 studio preview with localized dorsal and flank team colors."""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector,Euler,Quaternion
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path.home()/'.codex/skills/isaaclab-blender-render/scripts'))
from blender_common import configure_cycles
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/blender/as2_team_color_preview');p.add_argument('--team',choices=('blue','orange'),default='blue')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []);OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
team_color=(.015,.125,.55,1) if a.team=='blue' else (.95,.19,.025,1)

def basic(name,color,roughness=.45,metallic=.1):
    m=bpy.data.materials.new(name);m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=color;bs.inputs['Roughness'].default_value=roughness;bs.inputs['Metallic'].default_value=metallic
    return m
shell=basic('AS2 neutral shell with localized team color',(.28,.31,.35,1),.53,.08)
neutral=basic('AS2 neutral limbs',(.28,.31,.35,1),.52,.08)
rubber=basic('AS2 dark feet',(.022,.025,.03,1),.8,0)
# Object-local masks stick to the rigid base_link. No world-space swimming,
# UV unwrap, full-body recoloring, added body geometry or collision edits.
n=shell.node_tree.nodes;l=shell.node_tree.links
tex=n.new('ShaderNodeTexCoord');tex.label='Rigid-body local coordinates';tex.location=(-1100,100)
sep=n.new('ShaderNodeSeparateXYZ');l.new(tex.outputs['Object'],sep.inputs[0]);sep.location=(-880,100)
def mathop(op,x,y=None,label=''):
    v=n.new('ShaderNodeMath');v.operation=op;v.label=label
    if isinstance(x,(float,int)):v.inputs[0].default_value=x
    else:l.new(x,v.inputs[0])
    if y is not None:
        if isinstance(y,(float,int)):v.inputs[1].default_value=y
        else:l.new(y,v.inputs[1])
    return v.outputs[0]
def rect(value,lo,hi):return mathop('MULTIPLY',mathop('GREATER_THAN',value,lo),mathop('LESS_THAN',value,hi))
x,y,z=(sep.outputs[s] for s in ('X','Y','Z'))
ay=mathop('ABSOLUTE',y)
back=mathop('MULTIPLY',rect(x,-.14,.175),mathop('MULTIPLY',mathop('LESS_THAN',ay,.088),mathop('GREATER_THAN',z,.058)),label='Dorsal team panel')
sides=mathop('MULTIPLY',rect(x,-.095,.11),mathop('MULTIPLY',mathop('GREATER_THAN',ay,.092),rect(z,-.022,.042)),label='Left and right flank panels')
mask=mathop('MAXIMUM',back,sides,label='Only back and flanks')
color=n.new('ShaderNodeRGB');color.label='TEAM COLOR — change this for the other team';color.name='Team Color';color.outputs[0].default_value=team_color
mix=n.new('ShaderNodeMixRGB');mix.inputs[1].default_value=(.28,.31,.35,1);l.new(mask,mix.inputs[0]);l.new(color.outputs[0],mix.inputs[2]);l.new(mix.outputs[0],n.get('Principled BSDF').inputs['Base Color'])
for i,v in enumerate(n):v.location=((i%6)*210,-(i//6)*170)
# A balanced, static presentation stance computed from the original URDF.
robot=ET.parse(ROOT/'resources/robots/as2/urdf/as2.urdf').getroot()
transforms={'base_link':Matrix.Identity(4)}
pending=list(robot.findall('joint'))
while pending:
    progressed=False
    for joint in list(pending):
        parent=joint.find('parent').get('link');child=joint.find('child').get('link')
        if parent not in transforms:continue
        origin=joint.find('origin');xyz=tuple(map(float,origin.get('xyz','0 0 0').split()));rpy=tuple(map(float,origin.get('rpy','0 0 0').split()))
        local=Matrix.Translation(Vector(xyz))@Euler(rpy,'XYZ').to_matrix().to_4x4()
        if joint.get('type') in ('revolute','continuous'):
            name=joint.get('name');angle=(.1 if name.startswith(('FL','RL')) else -.1) if 'hip' in name else (.8 if 'thigh' in name else -1.6)
            axis=Vector(tuple(map(float,joint.find('axis').get('xyz').split())))
            local=local@Quaternion(axis,angle).to_matrix().to_4x4()
        transforms[child]=transforms[parent]@local;pending.remove(joint);progressed=True
    if not progressed:raise RuntimeError('Unresolved URDF hierarchy')
objects=[]
for link in robot.findall('link'):
    visual=link.find('visual')
    if visual is None:continue
    name=link.get('name');path=ROOT/'resources/robots/as2/meshes'/Path(visual.find('geometry/mesh').get('filename')).name
    bpy.ops.wm.stl_import(filepath=str(path));ob=bpy.context.object;ob.name='AS2_'+name
    origin=visual.find('origin');xyz=tuple(map(float,origin.get('xyz','0 0 0').split()));rpy=tuple(map(float,origin.get('rpy','0 0 0').split()))
    ob.matrix_world=transforms[name]@Matrix.Translation(Vector(xyz))@Euler(rpy,'XYZ').to_matrix().to_4x4()
    ob.data.materials.clear();ob.data.materials.append(shell if name=='base_link' else rubber if name.endswith('_foot') else neutral)
    for poly in ob.data.polygons:poly.use_smooth=True
    if hasattr(ob.data,'set_sharp_from_angle'):ob.data.set_sharp_from_angle(angle=.610865)
    objects.append(ob)
# Rest the four foot meshes on the studio floor.
lowest=100
for ob in objects:
    if not ob.name.endswith('_foot'):continue
    coords=np.empty(len(ob.data.vertices)*3);ob.data.vertices.foreach_get('co',coords)
    verts=coords.reshape(-1,3);mat=np.array(ob.matrix_world)
    lowest=min(lowest,float((verts@mat[:3,:3].T+mat[:3,3])[:,2].min()))
for ob in objects:ob.location.z-=lowest-.001
floor=basic('Warm gray studio floor',(.22,.24,.27,1),.82,0)
bpy.ops.mesh.primitive_plane_add(size=200);bpy.context.object.name='Studio floor';bpy.context.object.data.materials.append(floor)
world=bpy.data.worlds.new('Soft studio environment');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.55,.61,.72,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.28;scene.world=world

def area(name,pos,target,energy,size,color):
    d=bpy.data.lights.new(name,'AREA');d.energy=energy;d.shape='DISK';d.size=size;d.color=color
    o=bpy.data.objects.new(name,d);scene.collection.objects.link(o);o.location=pos;o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
area('Large neutral key',(0.7,-1.4,2.0),(0,0,.25),110,1.5,(1,.94,.86))
area('Soft frontal fill',(1.4,1.0,1.0),(.1,0,.3),45,1.2,(.80,.89,1))
area('Back rim',(-1.1,.55,1.5),(0,0,.3),80,1.0,(1,1,1))
d=bpy.data.cameras.new('Three-quarter team-color preview');cam=bpy.data.objects.new(d.name,d);scene.collection.objects.link(cam)
cam.location=(1.03,-1.30,1.06);target=Vector((.04,0,.27));cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler();d.type='ORTHO';d.ortho_scale=1.07;scene.camera=cam
scene.render.resolution_x=1536;scene.render.resolution_y=1152;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast'
scene.cycles.samples=96;scene.cycles.use_denoising=True;scene.cycles.denoising_use_gpu=True
scene.cycles.max_bounces=6;device=configure_cycles(scene,'auto')
scene.render.filepath=str(OUT/f'as2_{a.team}_back_side.png')
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/f'as2_{a.team}_back_side.blend'))
bpy.ops.render.render(write_still=True)
(OUT/'preview_report.json').write_text(json.dumps({'team':a.team,'team_color_linear_rgba':team_color,'geometry':'Original 17 AS2 STL visual links','pose':'Static URDF-based presentation stance','team_color_regions':'Dorsal surface and symmetric left/right body flanks only','material_scope':'Neutral gray body and limbs; dark feet; no full-body team recolor','device':device},indent=2))
print('TEAM_COLOR_PREVIEW_COMPLETE',flush=True)
