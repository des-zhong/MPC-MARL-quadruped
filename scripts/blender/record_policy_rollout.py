"""Record true Isaac Gym link poses at the low-level control rate for Blender."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
import isaacgym
import numpy as np
import torch
from scripts import play_high_level as playback

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--rollout-output',type=Path,required=True)
record_args,remaining=parser.parse_known_args()
OUT=record_args.rollout_output.resolve();OUT.mkdir(parents=True,exist_ok=True)
args=playback.parse_args(remaining)
args.no_video=True;args.no_plot=True;args.headless=True

class LinkRecorder:
    def __init__(self,env,args):
        self.raw=env.env.env;raw=self.raw
        self.frames=[];self.balls=[];self.episodes=[];self.roots=[];self.joints=[]
        self.episode=np.zeros(raw.num_envs,dtype=np.int32)
        self.names=list(raw.gym.get_asset_rigid_body_names(raw.robot_asset))
        self.original_check=raw.check_termination;self.original_reset=raw.reset_idx
        def check():
            self.original_check()
            # The rigid-body tensor is freshly refreshed at this point and no
            # auto-reset has yet overwritten roots. Poses are actor link frames.
            self.frames.append(raw.robot_rigid_body_state_all[...,:7].detach().cpu().numpy().copy())
            self.balls.append(raw.rigid_body_state_object[:,0,:7].detach().cpu().numpy().copy())
            self.roots.append(raw.root_states[raw.robot_actor_idxs_all,:7].detach().cpu().numpy().copy())
            self.joints.append(raw.dof_pos.detach().cpu().numpy().copy())
            self.episodes.append(self.episode.copy())
        def reset(ids):
            if ids.numel():self.episode[ids.detach().cpu().numpy()]+=1
            return self.original_reset(ids)
        raw.check_termination=check;raw.reset_idx=reset
        self.origins=raw.env_origins.detach().cpu().numpy().copy()
    def before_step(self,action):pass
    def after_step(self,done,info):pass
    def close(self):
        raw=self.raw;raw.check_termination=self.original_check;raw.reset_idx=self.original_reset
        frames=np.asarray(self.frames);ball=np.asarray(self.balls);roots=np.asarray(self.roots)
        assert np.isfinite(frames).all() and np.isfinite(ball).all()
        base_idx=self.names.index('base_link')
        delta=float(np.max(np.abs(frames[:,:, :,base_idx,:3]-roots[...,:3])))
        # Roots and body base must use the same link frame, not the body's COM.
        assert delta<1e-4, f'Root/body frame mismatch: {delta}'
        frames[...,:3]-=self.origins[None,:,None,None,:]
        ball[...,:3]-=self.origins[None,:,:]
        roots[...,:3]-=self.origins[None,:,None,:]
        np.savez_compressed(OUT/'rollout.npz',links=frames,ball=ball,roots=roots,
            joint_positions=np.asarray(self.joints),episode_ids=np.asarray(self.episodes),
            body_names=np.asarray(self.names),env_origins=self.origins,
            timestamps=np.arange(len(frames),dtype=np.float64)*raw.dt)
        def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
        meta={'source':'Real Isaac Gym policy rollout; both teams use the trained high-level policy',
          'dt':raw.dt,'sample_hz':1/raw.dt,'frames':len(frames),'num_envs':raw.num_envs,'robots_per_env':raw.num_robots,
          'body_names':self.names,'quaternion_order':'xyzw','coordinates':'Z-up metres; environment origin removed',
          'root_body_max_error_m':delta,'seed':args.seed,'episode_counts':(self.episode+1).tolist(),
          'policy_hashes':{skill:digest(ROOT/f'checkpoints/reproduction/{skill}/body_latest.jit') for skill in ('high_level','walk','dribble','shoot')}}
        (OUT/'rollout.json').write_text(json.dumps(meta,indent=2))
        print('RECORDED',frames.shape,'root/body delta',delta,flush=True)

playback.run(args,observer_factory=LinkRecorder)
