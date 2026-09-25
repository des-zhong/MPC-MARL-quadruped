"""Render a baked policy scene to a resumable PNG sequence."""
import argparse,json,sys,time
from pathlib import Path
import bpy
sys.path.insert(0,str(Path.home()/'.codex/skills/isaaclab-blender-render/scripts'))
from blender_common import configure_cycles
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--blend',type=Path);p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=a.output.resolve()
bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve() if a.blend else out/'policy_match_camera.blend'))
s=bpy.context.scene;device=configure_cycles(s,'auto');s.render.use_persistent_data=True
s.cycles.denoising_use_gpu=True
end=a.end if a.end is not None else s.frame_end
(out/'frames').mkdir(exist_ok=True)
for frame in range(a.start,end+1):
    path=out/'frames'/f'frame_{frame:04}.png'
    if path.exists():continue
    started=time.monotonic();s.frame_set(frame);evaluated=time.monotonic();s.render.filepath=str(path);bpy.ops.render.render(write_still=True)
    if frame==a.start:print(f'TIMING evaluate={evaluated-started:.2f}s render={time.monotonic()-evaluated:.2f}s',flush=True)
    if frame%12==0:print(f'PROGRESS {frame}/{end}',flush=True)
print('RENDER_COMPLETE',device,flush=True)
