"""Record single-skill AS2 link poses through the existing ability validator."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import isaacgym
import numpy as np
from scripts import play_walk_dribble_shoot as playback
from scripts import validate_robot_abilities as validation
from scripts.playback_utils import get_raw_env

args=validation.build_parser().parse_args()
assert args.ability in ('walk','dribble','shoot')
args.no_video=True
args.headless=True
out=Path(args.output_dir).resolve()/args.ability
out.mkdir(parents=True,exist_ok=True)
records={}
original_configure=playback.configure_rollout_cfg
def configure(*args,**kwargs):
    original_configure(*args,**kwargs)
    playback.Cfg.asset.file=str(ROOT/'resources/robots/as2/urdf/as2.urdf')
playback.configure_rollout_cfg=configure
original_make_env=playback.make_env

def make_env(*positional,**kwargs):
    env=original_make_env(*positional,**kwargs)
    raw=get_raw_env(env)
    names=list(raw.gym.get_asset_rigid_body_names(raw.robot_asset))
    assert len(names)==17
    frames=[];balls=[];done=[];base_errors=[]
    original_check=raw.check_termination
    def check():
        original_check()
        link=raw.rigid_body_state[0,:,:7].detach().cpu().numpy().copy()
        ball=raw.rigid_body_state_object[0,0,:7].detach().cpu().numpy().copy()
        root=raw.root_states[raw.robot_actor_idxs[0],:7].detach().cpu().numpy()
        base_errors.append(float(np.abs(link[names.index('base_link'),:3]-root[:3]).max()))
        origin=raw.env_origins[0].detach().cpu().numpy()
        link[:,:3]-=origin;ball[:3]-=origin
        frames.append(link);balls.append(ball);done.append(bool(raw.reset_buf[0].item()))
    raw.check_termination=check
    records.update(raw=raw,names=names,frames=frames,balls=balls,done=done,base_errors=base_errors,original_check=original_check)
    return env

playback.make_env=make_env
status=validation.run_single(args)
raw=records['raw'];raw.check_termination=records['original_check']
links=np.asarray(records['frames']);balls=np.asarray(records['balls'])
assert np.isfinite(links).all() and np.isfinite(balls).all()
assert max(records['base_errors'])<1e-4
np.savez_compressed(out/'trajectory.npz',links=links,ball=balls,done=records['done'],body_names=records['names'])
meta={'ability':args.ability,'dt':float(raw.dt),'frames':len(links),'source':'Actual isolated trained-policy rollout, captured before automatic reset','coordinates':'environment-local metres, Z up; xyzw quaternions','root_body_max_error_m':max(records['base_errors']),'command_frame':args.command_frame,'seed':args.seed,'arguments':vars(args)}
(out/'trajectory.json').write_text(json.dumps(meta,indent=2))
print('SKILL_RECORDED',args.ability,links.shape,flush=True)
sys.stdout.flush();sys.stderr.flush();os._exit(status)
