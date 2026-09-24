"""Export a static, GPU-pose-baked snapshot of actual parallel Isaac Lab matches."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path('/home/xander/Code/01_Locomotion/MPC-MARL-quadruped')
sys.path.insert(0, str(ROOT/'isaaclab-rebuild/source/dribblebot_isaaclab'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=36)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import dribblebot_isaaclab
from isaaclab_tasks.utils import parse_env_cfg


def main():
    task='Isaac-DribbleBot-AS2-Match-Macro-Flat-Play-v0'
    cfg=parse_env_cfg(task,device=args.device,num_envs=args.num_envs,use_fabric=True)
    cfg.seed=17
    cfg.wait_for_textures=False
    # Actual independent match resets; no synthetic articulation or walking animation.
    cfg.events.reset_match.params['randomize']=True
    env=gym.make(task,cfg=cfg)
    raw=env.unwrapped
    try:
        env.reset()
        actions=torch.zeros(raw.num_envs,16,device=raw.device)
        for k in range(4):
            actions[:,4*k+1]=0.3
        with torch.inference_mode():
            env.step(actions)
        print('[EXPORT] baking GPU link frames into a separate USD stage',flush=True)
        frozen=Usd.Stage.Open(raw.sim.stage.Flatten())
        UsdGeom.SetStageUpAxis(frozen,UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(frozen,1.0)
        snapshots=[]
        for robot_index in range(4):
            robot=raw.scene[f'robot_{robot_index}']
            positions=robot.data.body_link_pos_w.cpu().tolist()
            rotations=robot.data.body_link_quat_w.cpu().tolist()
            for env_index in range(raw.num_envs):
                root=f'/World/envs/env_{env_index}/Robot_{robot_index}'
                bodies={p.GetName():p for p in Usd.PrimRange(frozen.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
                for body_index,name in enumerate(robot.body_names):
                    snapshots.append((bodies[name],positions[env_index][body_index],rotations[env_index][body_index]))
        ball=raw.scene['ball']
        for env_index in range(raw.num_envs):
            root=f'/World/envs/env_{env_index}/Ball'
            prim=next(p for p in Usd.PrimRange(frozen.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.RigidBodyAPI))
            snapshots.append((prim,ball.data.root_pos_w[env_index].cpu().tolist(),ball.data.root_quat_w[env_index].cpu().tolist()))
        # Parents first, then convert each measured world link frame to USD local.
        for prim,pos,q in sorted(snapshots,key=lambda item:len(str(item[0].GetPath()).split('/'))):
            world=Gf.Matrix4d(1.0)
            world.SetRotate(Gf.Quatd(q[0],Gf.Vec3d(*q[1:])))
            world.SetTranslateOnly(Gf.Vec3d(*pos))
            parent=UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
            xf=UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            op=xf.AddTransformOp(opSuffix='snapshot')
            op.Set(world*parent.GetInverse())
        args.output.mkdir(parents=True,exist_ok=True)
        path=args.output/'parallel_scene.usdc'
        frozen.GetRootLayer().Export(str(path))
        report={'task':task,'num_envs':raw.num_envs,'robots':4*raw.num_envs,
                'env_origins':raw.scene.env_origins.cpu().tolist(),'seed':17,
                'snapshot':'Actual simulation after randomized reset and one 0.2 s macro step; scripted walk command, not trained high-level policy',
                'body_poses_baked':len(snapshots),'source_usd':str(path),'source_physics_modified':False}
        (args.output/'snapshot.json').write_text(json.dumps(report,indent=2))
        print('[EXPORT] saved',path,flush=True)
    finally:
        env.close()

if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc()
        raise
    finally:
        app.close()
