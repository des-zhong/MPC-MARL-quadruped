"""Bake real AS2 match tracks into a portable Blender scene and camera move."""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector,Quaternion
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path.home()/'.codex/skills/isaaclab-blender-render/scripts'))
from blender_common import configure_cycles
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--base-blend',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(a.base_blend.resolve()))
scene=bpy.context.scene
z=np.load(OUT/'clips_24fps.npz');links=z['links'];balls=z['ball'];names=z['body_names'].tolist()
origins=np.array(json.loads((a.base_blend.parent/'snapshot.json').read_text())['env_origins'])
frames,num_envs,num_robots,num_bodies,_=links.shape
assert len(origins)==num_envs==36 and num_bodies==17
# Remove static imported robot hierarchies. Original goals, field lines, PBR
# lawn and studio lights stay in place. STL visuals use recorded link frames.
def robot_ancestor(ob):
    while ob:
        if ob.name.split('.')[0] in ('Robot_0','Robot_1','Robot_2','Robot_3'):return True
        ob=ob.parent
    return False
old=[ob for ob in scene.objects if robot_ancestor(ob)]
bpy.data.batch_remove(ids=old)
print('REMOVED_STATIC_ROBOT_OBJECTS',len(old),flush=True)
materials=[bpy.data.materials['AS2 silver approximation'],bpy.data.materials['AS2 opponent red approximation']]
foot=bpy.data.materials['AS2 dark feet']
mesh_by_name={}
urdf=ET.parse(ROOT/'resources/robots/as2/urdf/as2.urdf').getroot()
for name in names:
    visual=urdf.find(f"link[@name='{name}']/visual")
    origin=visual.find('origin')
    assert all(float(v)==0 for v in origin.get('xyz').split()) and all(float(v)==0 for v in origin.get('rpy').split())
    path=ROOT/'resources/robots/as2/meshes'/Path(visual.find('geometry/mesh').get('filename')).name
    bpy.ops.wm.stl_import(filepath=str(path))
    obj=bpy.context.object;mesh=obj.data;mesh.name='AS2 shared '+name
    for poly in mesh.polygons:poly.use_smooth=True
    mesh_by_name[name]=mesh;bpy.data.objects.remove(obj,do_unlink=True)
# All channels are baked into ordinary F-curves. No frame handlers or runtime
# NPZ dependency is needed when opening the final .blend.
frame_numbers=np.arange(1,frames+1,dtype=np.float32)
def animate(ob,poses):
    ob.rotation_mode='QUATERNION';poses=poses.copy();q=poses[:,[6,3,4,5]]
    for i in range(1,len(q)):
        if np.dot(q[i-1],q[i])<0:q[i]*=-1
    values=np.concatenate([poses[:,:3],q],axis=1)
    action=bpy.data.actions.new(ob.name+' recorded motion')
    slot=action.slots.new(id_type='OBJECT',name=ob.name)
    strip=action.layers.new('Baked real rollout').strips.new(type='KEYFRAME')
    bag=strip.channelbag(slot,ensure=True)
    ob.animation_data_create();ob.animation_data.action=action;ob.animation_data.action_slot=slot
    for k in range(7):
        curve=bag.fcurves.new(data_path='location' if k<3 else 'rotation_quaternion',index=k if k<3 else k-3)
        curve.keyframe_points.add(frames)
        coords=np.column_stack((frame_numbers,values[:,k])).astype(np.float32)
        curve.keyframe_points.foreach_set('co',coords.ravel())
        for point in curve.keyframe_points:point.interpolation='LINEAR'
        curve.update()
    ob.location=values[0,:3];ob.rotation_quaternion=values[0,3:]
collection=bpy.data.collections.new('Real policy rollout — 144 AS2');scene.collection.children.link(collection)
for env in range(num_envs):
    for robot in range(num_robots):
        for body,name in enumerate(names):
            mesh=mesh_by_name[name]
            ob=bpy.data.objects.new(f'Rollout_E{env:02}_R{robot}_{name}',mesh);collection.objects.link(ob)
            if not len(mesh.materials):mesh.materials.append(materials[0])
            for slot in ob.material_slots:slot.link='OBJECT';slot.material=foot if name.endswith('_foot') else materials[int(robot>=2)]
            poses=links[:,env,robot,body,:].copy();poses[:,:3]+=origins[env]
            animate(ob,poses)
    ball=bpy.data.objects[f'USDShape__World_envs_env_{env}_Ball_geometry_mesh']
    ball.animation_data_clear();ball.parent=None
    poses=balls[:,env,:].copy();poses[:,:3]+=origins[env];animate(ball,poses)
    if env%6==0:print('BAKED_FIELDS',env+1,flush=True)
# Smooth the camera's subject tracking, leaving every recorded robot pose intact.
hero_ball=balls[:,0,:3]+origins[0]
def smooth_signal(x,radius=20):
    return np.stack([x[max(0,i-radius):min(len(x),i+radius+1)].mean(axis=0) for i in range(len(x))])
tracked=smooth_signal(hero_ball);tracked[:,:2]=np.clip(tracked[:,:2],origins[0,:2]-1.3,origins[0,:2]+1.3)
cam=scene.camera;cam.animation_data_clear();cam.data.lens=36;cam.data.clip_end=1000
camera_poses=[]
for f in range(frames):
    t=f/24;u=np.clip((t-3.5)/6.0,0,1);u=u*u*(3-2*u)
    target_close=.45*tracked[f]+.55*(origins[0]+np.array([0,0,.25]))
    close=target_close+np.array([7.2+.8*t/10,-8.5+.5*t/10,5.2])
    target=(1-u)*target_close+u*np.array([0,-2,.25])
    pos=(1-u)*close+u*np.array([47,-53,34])
    q=(Vector(target)-Vector(pos)).to_track_quat('-Z','Y')
    camera_poses.append([*pos,q.x,q.y,q.z,q.w])
animate(cam,np.array(camera_poses));cam.data.dof.use_dof=False
scene.frame_start=1;scene.frame_end=frames;scene.render.fps=24
scene.render.engine='CYCLES';scene.cycles.samples=32;scene.cycles.use_denoising=True
scene.cycles.denoising_use_gpu=True
scene.cycles.max_bounces=4;scene.cycles.diffuse_bounces=2;scene.cycles.glossy_bounces=2
scene.render.use_persistent_data=True
scene.render.resolution_x=1280;scene.render.resolution_y=720;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
scene.render.filepath=str(OUT/'frames/frame_')
device=configure_cycles(scene,'auto')
scene.frame_set(1)
# Discard orphaned import collections and meshes, retaining shared active meshes.
bpy.ops.outliner.orphans_purge(do_local_ids=True,do_linked_ids=False,do_recursive=True)
bpy.ops.file.pack_all();bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'policy_match_camera.blend'))
report={'frames':frames,'resolution':[1280,720],'seconds':frames/24,'motion_source':'True Isaac Gym policy rollout','fps':24,'frame_start':1,'frame_end':frames,'duration_s':frames/24,'width':1280,'height':720,'samples':32,'device':device,'robots':144,'distinct_matches':36,'animation':'Recorded link poses, linear positions and shortest-arc SLERP rotations; ordinary baked F-curves','camera':'Smooth close follow into a rising pullback over the parallel field array','source':'Real Isaac Gym trained policy, independently trimmed reset-free excerpts','robot_materials':'Existing approximate silver/red material preserved; no claimed original textures','physics_boundary_walls':'Retained in simulation, hidden in Blender presentation'}
(OUT/'animation_report.json').write_text(json.dumps(report,indent=2))
for frame in (1,120,240):
    scene.frame_set(frame);scene.render.filepath=str(OUT/f'preview_{frame:04}.png');bpy.ops.render.render(write_still=True)
print('POLICY_FILM_BAKED',flush=True)
