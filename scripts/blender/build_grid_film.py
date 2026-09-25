"""Build a 10x10 real-policy match film, holding a corner pitch before pullback."""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / '.codex/skills/isaaclab-blender-render/scripts'))
from blender_common import configure_cycles

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--source', type=Path, default=ROOT / 'outputs/blender/policy_match_team_colors_20260924/policy_match_camera.blend')
p.add_argument('--grid-size', type=int, default=10)
p.add_argument('--hero-row', type=int, default=2)
p.add_argument('--hero-col', type=int, default=2)
p.add_argument('--hold-seconds', type=float, default=3)
p.add_argument('--hero-corner', choices=('lower-left', 'upper-right'), default='lower-left')
p.add_argument('--flat-envs', type=float, default=5)
p.add_argument('--flat-end-seconds', type=float, default=7)
p.add_argument('--move-end-seconds', type=float, default=9.25)
p.add_argument('--end-height', type=float, default=30)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
z = np.load(out / 'clips_24fps.npz')
links, balls, names = z['links'], z['ball'], z['body_names'].tolist()
frames, envs, robots, bodies, _ = links.shape
assert envs == a.grid_size ** 2 and robots == 4 and bodies == 17
assert 1 <= a.hero_row <= a.grid_size and 1 <= a.hero_col <= a.grid_size
fps = 24
assert 0 <= a.hold_seconds < a.flat_end_seconds < a.move_end_seconds <= (frames - 1) / fps
bpy.ops.wm.open_mainfile(filepath=str(a.source))
s = bpy.context.scene
s.frame_set(1)
originals = list(s.objects)
# Retain shared mesh data and approved object-level material assignments.
robot_templates = {(r, name): bpy.data.objects[f'Rollout_E00_R{r}_{name}'].copy()
                   for r in range(4) for name in names}
# Bind materials to mesh data so Cycles can actually instance geometry.
# Object-level overrides force a separate Cycles mesh for every robot link.
visual_meshes = {}
for (robot, name), template in robot_templates.items():
    material = template.material_slots[0].material
    key = (name, material.name)
    if key not in visual_meshes:
        mesh = template.data.copy()
        mesh.name = 'Instanced AS2 ' + name + ' ' + material.name
        mesh.materials.clear()
        mesh.materials.append(material)
        visual_meshes[key] = mesh
    template.data = visual_meshes[key]
    for slot in template.material_slots:
        slot.link = 'DATA'
assert len(visual_meshes) == 18, len(visual_meshes)
ball_template = bpy.data.objects['USDShape__World_envs_env_0_Ball_geometry_mesh'].copy()
field_templates = []
for parent_name in ('GoalVisual', 'SoccerFieldVisual'):
    parent = bpy.data.objects[parent_name]
    matches = [ob for ob in parent.children_recursive if ob.type == 'MESH']
    assert len(matches) == 1, (parent_name, len(matches))
    ob = matches[0]
    clone = ob.copy()
    clone.parent = None
    clone.matrix_world = ob.matrix_world.copy()
    clone.location -= Vector((25, -25, 0))
    field_templates.append(clone)
grass = bpy.data.objects['Continuous grass across all 36 pitches']
grass.name = f'Continuous grass across all {envs} pitches'
grass.scale *= 2  # World-coordinate material keeps the same turf and stripe scale.
keep = {grass, s.camera, *[ob for ob in s.objects if ob.type == 'LIGHT']}
bpy.data.batch_remove(ids=[ob for ob in originals if ob not in keep])
# Scale the existing studio lighting to the larger grid, mirrored toward +Y.
scale = a.grid_size / 6
for ob in s.objects:
    if ob.type != 'LIGHT':
        continue
    direction = ob.rotation_euler.to_quaternion() @ Vector((0, 0, -1))
    direction.y *= -1
    ob.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    ob.location = Vector((ob.location.x * scale, -ob.location.y * scale, ob.location.z * scale))
    ob.data.size *= scale
    ob.data.energy *= scale ** 2
collection = bpy.data.collections.new(f'{a.grid_size}x{a.grid_size} real policy matches')
s.collection.children.link(collection)
spacing = 10
edge = (a.grid_size - 1) * spacing / 2
# Top view: +Y is up, +X is right; one-based rows/columns from upper right.
origins = np.array([(edge - col * spacing, edge - row * spacing, 0)
                    for row in range(a.grid_size) for col in range(a.grid_size)])
hero_row = a.grid_size - a.hero_row if a.hero_corner == 'lower-left' else a.hero_row - 1
hero_col = a.grid_size - a.hero_col if a.hero_corner == 'lower-left' else a.hero_col - 1
hero = hero_row * a.grid_size + hero_col
# Prefer an active opening near midfield, with no goal obscuring the players.
opening = int(a.hold_seconds * fps) + 1
hero_candidates = []
for e in range(envs):
    initial = np.linalg.norm(balls[0, e, :2])
    mean = np.linalg.norm(balls[:opening, e, :2], axis=1).mean()
    edge_extent = np.abs(links[:opening, e, :, 0, 0]).max()
    travel = np.linalg.norm(np.diff(balls[:opening, e, :2], axis=0), axis=1).sum()
    if initial < 1.8 and mean < 2 and edge_extent < 3.9:
        hero_candidates.append((float(travel - mean * .25), e))
hero_source = max(hero_candidates)[1] if hero_candidates else 0
source_indices = np.arange(envs)
source_indices[hero_source], source_indices[hero] = source_indices[hero], source_indices[hero_source]
frame_numbers = np.arange(1, frames + 1, dtype=np.float32)

def animate(ob, poses):
    ob.animation_data_clear()
    ob.rotation_mode = 'QUATERNION'
    q = poses[:, [6, 3, 4, 5]].copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    values = np.concatenate([poses[:, :3], q], axis=1)
    action = bpy.data.actions.new(ob.name + ' recorded motion')
    slot = action.slots.new(id_type='OBJECT', name=ob.name)
    strip = action.layers.new('Baked motion').strips.new(type='KEYFRAME')
    bag = strip.channelbag(slot, ensure=True)
    ob.animation_data_create()
    ob.animation_data.action = action
    ob.animation_data.action_slot = slot
    for k in range(7):
        curve = bag.fcurves.new(data_path='location' if k < 3 else 'rotation_quaternion', index=k if k < 3 else k - 3)
        curve.keyframe_points.add(frames)
        curve.keyframe_points.foreach_set('co', np.column_stack((frame_numbers, values[:, k])).astype(np.float32).ravel())
        for point in curve.keyframe_points:
            point.interpolation = 'LINEAR'
        curve.update()
    ob.location = values[0, :3]
    ob.rotation_quaternion = values[0, 3:]

for env, origin in enumerate(origins):
    source = source_indices[env]
    for i, template in enumerate(field_templates):
        ob = template.copy()
        ob.name = f'Pitch_{env:03}_' + ('Goals' if i == 0 else 'Lines')
        collection.objects.link(ob)
        ob.location += Vector(origin)
    for robot in range(robots):
        for body, name in enumerate(names):
            ob = robot_templates[robot, name].copy()
            ob.name = f'Rollout_E{env:03}_R{robot}_{name}'
            ob.parent = None
            collection.objects.link(ob)
            poses = links[:, source, robot, body, :].copy()
            poses[:, :3] += origin
            animate(ob, poses)
    ob = ball_template.copy()
    ob.name = f'Ball_{env:03}'
    collection.objects.link(ob)
    poses = balls[:, source, :].copy()
    poses[:, :3] += origin
    animate(ob, poses)
    if env % 10 == 0:
        print('BAKED_PITCHES', env + 1, flush=True)
bpy.data.batch_remove(ids=[*robot_templates.values(), ball_template, *field_templates])
# A truly fixed first shot: no ball tracking, drift, lens animation or zoom.
cam = s.camera
cam.animation_data_clear()
cam.data.animation_data_clear()
cam.data.lens = 36
cam.data.clip_end = 1000
cam.data.dof.use_dof = False
# Camera controls: fixed opening, level retreat across five diagonal cells,
# then a low ascending retreat. The final composition crops the outer grid.
sign = 1 if a.hero_corner == 'lower-left' else -1
start_target = origins[hero] + np.array([0, 0, .25])
start_pos = start_target + np.array([sign * 7.2, sign * 8.5, 5.2])
flat_translation = np.array([sign * spacing * a.flat_envs, sign * spacing * a.flat_envs, 0])
flat_pos = start_pos + flat_translation
end_pos = np.array([sign * 43, sign * 47, a.end_height])
end_target = np.array([sign * 15, sign * 15, .25])
start_q = (Vector(start_target) - Vector(start_pos)).to_track_quat('-Z', 'Y')
end_q = (Vector(end_target) - Vector(end_pos)).to_track_quat('-Z', 'Y')
# Shared tangent keeps the two moving segments continuous in speed.
flat_duration = a.flat_end_seconds - a.hold_seconds
handoff_velocity = 2 * flat_translation / flat_duration
def hermite(p0, p1, v0, v1, u, seconds):
    return ((2*u**3-3*u**2+1)*p0 + (u**3-2*u**2+u)*seconds*v0
            + (-2*u**3+3*u**2)*p1 + (u**3-u**2)*seconds*v1)

s.render.resolution_x = 1280
s.render.resolution_y = 720
s.render.resolution_percentage = 100
poses = []
for f in range(frames):
    t = f / fps
    if t <= a.hold_seconds:
        pos, q = start_pos, start_q
    elif t <= a.flat_end_seconds:
        duration = a.flat_end_seconds - a.hold_seconds
        u = (t - a.hold_seconds) / duration
        pos = start_pos + flat_translation * u**2  # Quadratic ease-in: speed increases throughout.
        q = start_q  # Pure translation: constant camera orientation and height.
    elif t < a.move_end_seconds:
        duration = a.move_end_seconds - a.flat_end_seconds
        u = (t - a.flat_end_seconds) / duration
        pos = hermite(flat_pos, end_pos, handoff_velocity, np.zeros(3), u, duration)
        q = start_q.slerp(end_q, u*u*(3-2*u))
    else:
        pos, q = end_pos, end_q
    poses.append([*pos, q.x, q.y, q.z, q.w])
poses = np.array(poses)
hold_end = int(a.hold_seconds * fps)
flat_end = int(a.flat_end_seconds * fps)
move_end = int(a.move_end_seconds * fps)
assert np.max(np.abs(poses[:hold_end + 1] - poses[0])) == 0
assert np.max(np.abs(poses[:flat_end + 1, 2] - poses[0, 2])) < 1e-7
assert np.max(np.abs(poses[:flat_end + 1, 3:] - poses[0, 3:])) < 1e-7
assert np.max(np.abs(poses[move_end:] - poses[-1])) < 1e-7
assert np.all(np.diff(poses[:, 2]) >= -1e-7)
assert np.all(sign * np.diff(poses[:, :2], axis=0) >= -1e-7)
flat_speeds = np.linalg.norm(np.diff(poses[hold_end:flat_end + 1, :3], axis=0), axis=1) * fps
assert np.all(np.diff(flat_speeds) >= -1e-7)
# Intersect final camera corner rays with the ground to verify a filled frame.
direction = end_target - end_pos
direction /= np.linalg.norm(direction)
right = np.cross(direction, [0, 0, 1]); right /= np.linalg.norm(right)
up = np.cross(right, direction)
corner_ground = []
for x, y in ((-1,-1), (1,-1), (-1,1), (1,1)):
    ray = direction + x*.5*right + y*.28125*up
    assert ray[2] < 0, 'The final frame must contain ground, not sky'
    point = end_pos - ray*end_pos[2]/ray[2]
    corner_ground.append(point.tolist())
assert np.abs(np.array(corner_ground)[:, :2]).max() < edge + 5
animate(cam, poses)
s.frame_start = 1
s.frame_end = frames
s.render.fps = fps
# Preserve a generous offscreen shadow margin while skipping distant unseen links.
s.render.use_simplify = True
s.cycles.use_camera_cull = True
s.cycles.camera_cull_margin = .25
for ob in collection.objects:
    ob.cycles.use_camera_cull = True
s.cycles.samples = 32
s.cycles.use_denoising = True
s.cycles.denoising_use_gpu = True
s.render.use_persistent_data = True
s.render.image_settings.file_format = 'PNG'
s.render.image_settings.color_mode = 'RGB'
s.render.filepath = str(out / 'frames/frame_')
device = configure_cycles(s, 'auto')
s.frame_set(1)
bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)
bpy.ops.file.pack_all()
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(out / 'policy_match_camera.blend'))
report = {'frames': frames, 'resolution': [1280, 720], 'seconds': frames / fps,
          'fps': fps, 'frame_start': 1, 'frame_end': frames, 'samples': 32,
          'motion_source': f'{envs} distinct real Isaac Gym policy matches, independently trimmed',
          'grid': [a.grid_size, a.grid_size], 'robots': envs * robots, 'distinct_matches': envs,
          'hero_corner': a.hero_corner, 'hero_row_col_from_corner_one_based': [a.hero_row, a.hero_col],
          'hero_env_index': hero, 'hero_source_clip_index': int(hero_source), 'hero_world_origin_m': origins[hero].tolist(),
          'camera_hold_s': a.hold_seconds, 'camera_hold_max_pose_delta': 0,
          'camera_start_m': start_pos.tolist(), 'camera_end_m': end_pos.tolist(),
          'camera_motion': 'Fixed opening; level diagonal retreat across five cells; low ascending retreat; fixed ending',
          'camera_flat_translation_m': flat_translation.tolist(),
          'camera_flat_speed_curve': 'quadratic ease-in (u squared)',
          'camera_flat_start_end_speed_m_s': [float(flat_speeds[0]), float(flat_speeds[-1])],
          'camera_flat_end_s': a.flat_end_seconds, 'camera_move_end_s': a.move_end_seconds,
          'camera_end_hold_s': frames / fps - a.move_end_seconds,
          'final_frame_ground_corners_m': corner_ground,
          'source_clip_index_per_pitch': source_indices.tolist(), 'env_origins': origins.tolist(),
          'shared_robot_meshes': len(visual_meshes),
          'robot_materials': 'Gray shell and limbs, dark feet, blue/orange back and flank panels',
          'device': device}
(out / 'animation_report.json').write_text(json.dumps(report, indent=2))
for frame in (1, 72, flat_end + 1, 200, move_end + 1, frames):
    s.frame_set(frame)
    s.render.filepath = str(out / f'preview_{frame:04}.png')
    bpy.ops.render.render(write_still=True)
print('GRID_FILM_BUILT', envs, 'matches', flush=True)
