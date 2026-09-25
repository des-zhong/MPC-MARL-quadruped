"""Render isolated real-policy AS2 skill clips on a precisely colored studio floor."""
import argparse,json,sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path.home()/'.codex/skills/isaaclab-blender-render/scripts'))
from blender_common import configure_cycles
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--skill',choices=('walk','dribble','shoot'),required=True);p.add_argument('--seconds',type=float,default=6)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=a.output.resolve()/a.skill;out.mkdir(parents=True,exist_ok=True)
z=np.load(out/'trajectory.npz');meta=json.loads((out/'trajectory.json').read_text())
fps=30;frames=round(a.seconds*fps);start=.1 if a.skill=='shoot' else .5
samples=(start+np.arange(frames)/fps)/meta['dt'];idx=np.floor(samples).astype(int);w=samples-idx
assert idx[-1]+1<len(z['links']) and not z['done'][idx[0]:idx[-1]+2].any()
def interpolate(data):
    x=data[idx];y=data[idx+1];weight=w.reshape((len(w),)+(1,)*(data.ndim-1))
    q0=x[...,3:7];q1=y[...,3:7];dot=np.sum(q0*q1,axis=-1,keepdims=True)
    q1=np.where(dot<0,-q1,q1);dot=np.clip(np.abs(dot),0,1);theta=np.arccos(dot);den=np.maximum(np.sin(theta),1e-8)
    q=np.sin((1-weight)*theta)/den*q0+np.sin(weight*theta)/den*q1
    q=np.where(dot>.9995,(1-weight)*q0+weight*q1,q);q/=np.linalg.norm(q,axis=-1,keepdims=True)
    return np.concatenate(((1-weight)*x[...,:3]+weight*y[...,:3],q),axis=-1)
links=interpolate(z['links']);balls=interpolate(z['ball']);names=z['body_names'].tolist()
assert np.isfinite(links).all()
bpy.ops.wm.read_factory_settings(use_empty=True);s=bpy.context.scene
source=ROOT/'outputs/blender/as2_team_color_preview/as2_blue_back_side.blend'
with bpy.data.libraries.load(str(source),link=False) as (available,loaded):
    loaded.objects=['AS2_'+name for name in names]
objects=loaded.objects
frame_numbers=np.arange(1,frames+1,dtype=np.float32)
def animate(ob,poses):
    ob.animation_data_clear();ob.rotation_mode='QUATERNION';q=poses[:,[6,3,4,5]].copy()
    for i in range(1,len(q)):
        if np.dot(q[i-1],q[i])<0:q[i]*=-1
    values=np.concatenate([poses[:,:3],q],axis=1)
    action=bpy.data.actions.new(ob.name+' true skill motion');slot=action.slots.new(id_type='OBJECT',name=ob.name)
    bag=action.layers.new('Baked motion').strips.new(type='KEYFRAME').channelbag(slot,ensure=True)
    ob.animation_data_create();ob.animation_data.action=action;ob.animation_data.action_slot=slot
    for k in range(7):
        curve=bag.fcurves.new(data_path='location' if k<3 else 'rotation_quaternion',index=k if k<3 else k-3)
        curve.keyframe_points.add(frames);curve.keyframe_points.foreach_set('co',np.column_stack((frame_numbers,values[:,k])).astype(np.float32).ravel())
        for point in curve.keyframe_points:point.interpolation='LINEAR'
        curve.update()
    ob.location=values[0,:3];ob.rotation_quaternion=values[0,3:]
for i,ob in enumerate(objects):
    s.collection.objects.link(ob);ob.parent=None;animate(ob,links[:,i])
def material(name,color,roughness=.5,metallic=0):
    m=bpy.data.materials.new(name);m.use_nodes=True;bs=m.node_tree.nodes['Principled BSDF']
    bs.inputs['Base Color'].default_value=color;bs.inputs['Roughness'].default_value=roughness;bs.inputs['Metallic'].default_value=metallic;return m
# Skill demonstrations use an unmarked neutral shell, without team panels.
neutral_shell=material('AS2 neutral shell without team colors',(.28,.31,.35,1),.53,.08)
base=objects[names.index('base_link')]
base.data.materials.clear();base.data.materials.append(neutral_shell)
assert not neutral_shell.node_tree.nodes['Principled BSDF'].inputs['Base Color'].is_linked
if a.skill!='walk':
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48,ring_count=24,radius=.0889)
    ball=bpy.context.object;ball.name='Actual skill ball';ball.data.materials.append(material('Football yellow',(.82,.64,.01,1),.45))
    for polygon in ball.data.polygons:polygon.use_smooth=True
    animate(ball,balls)
def linear(v):
    v=v/255
    return v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4
cream=tuple(linear(v) for v in (253,250,244))+(1,)
bpy.ops.mesh.primitive_plane_add(size=200)
floor=bpy.context.object;floor.name='Floor RGB 253 250 244';floor.data.materials.append(material('Warm white #FDFAF4',cream,.78));floor.is_shadow_catcher=True
world=bpy.data.worlds.new('Neutral soft studio');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.35;s.world=world
# Large lights give soft, readable contact shadows and restrained highlights.
def area(name,pos,energy,size):
    d=bpy.data.lights.new(name,'AREA');d.energy=energy;d.shape='DISK';d.size=size
    ob=bpy.data.objects.new(name,d);s.collection.objects.link(ob);ob.location=pos;ob.rotation_euler=(Vector((1,0,.2))-ob.location).to_track_quat('-Z','Y').to_euler()
area('Soft key',(0,-4,6),400,5)
area('Soft fill',(4,3,5),180,5)
# Composite the shadow-catching floor over the exact display color. This keeps
# unshadowed background pixels #FDFAF4 rather than shifting them with exposure.
s.render.film_transparent=True
comp=bpy.data.node_groups.new('Warm white shadow composite','CompositorNodeTree')
comp.interface.new_socket(name='Image',in_out='OUTPUT',socket_type='NodeSocketColor')
rl=comp.nodes.new('CompositorNodeRLayers');over=comp.nodes.new('CompositorNodeAlphaOver');over.inputs['Factor'].default_value=1;over.inputs['Background'].default_value=cream
output=comp.nodes.new('NodeGroupOutput');comp.links.new(rl.outputs['Image'],over.inputs['Foreground']);comp.links.new(over.outputs[0],output.inputs['Image']);s.compositing_node_group=comp
s.view_settings.view_transform='Standard';s.view_settings.look='None';s.view_settings.exposure=0;s.view_settings.gamma=1
root=links[:,names.index('base_link'),:3]
tracked=np.stack([root[max(0,i-6):min(frames,i+7)].mean(axis=0) for i in range(frames)])
tracked[:,2]=.22
cam_data=bpy.data.cameras.new('Skill studio camera');cam=bpy.data.objects.new(cam_data.name,cam_data);s.collection.objects.link(cam);s.camera=cam
cam_data.type='ORTHO';cam_data.ortho_scale=2.2;cam_data.clip_end=500
camera=[]
for target in tracked:
    pos=target+np.array([2.8,-3.8,5.0]);q=(Vector(target)-Vector(pos)).to_track_quat('-Z','Y');camera.append([*pos,q.x,q.y,q.z,q.w])
animate(cam,np.array(camera))
s.frame_start=1;s.frame_end=frames;s.render.fps=fps;s.render.resolution_x=960;s.render.resolution_y=640;s.render.resolution_percentage=100
s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.render.filepath=str(out/'frames/frame_')
s.cycles.samples=48;s.cycles.use_denoising=True;s.cycles.denoising_use_gpu=True;s.cycles.max_bounces=4;s.render.use_persistent_data=True
configure_cycles(s,'auto');s.frame_set(1);bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out/f'{a.skill}.blend'))
report={'fps':fps,'frames':frames,'frame_start':1,'frame_end':frames,'seconds':frames/fps,'resolution':[960,640],'motion_source':f'Actual isolated {a.skill} trained-policy recording','source_start_s':start,'team_colors':False,'floor_srgb':[253,250,244],'floor_linear':cream,'presentation':'Neutral gray AS2 without team colors, dark feet; orthographic studio follow camera','camera_motion':'Position follows robot with fixed orientation; does not modify recorded poses'}
(out/'animation_report.json').write_text(json.dumps(report,indent=2))
for frame in (1,22,91,frames):
    s.frame_set(frame);s.render.filepath=str(out/f'preview_{frame:04}.png');bpy.ops.render.render(write_still=True)
print('SKILL_STUDIO_BUILT',a.skill,flush=True)
