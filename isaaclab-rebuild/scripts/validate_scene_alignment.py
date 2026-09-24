"""Check the loaded Lab scene against Gym assets and exercise wall/goal physics.

Run with --headless --output outputs/scene-alignment.json. This checks scene
parity, not equality of trajectories from different PhysX versions.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab-rebuild/source/dribblebot_isaaclab"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
import dribblebot_isaaclab  # noqa: F401
from dribblebot_isaaclab.tasks.manager_based.football import mdp
from isaaclab_tasks.utils import parse_env_cfg


def gym_wall_layout():
    # Execute only the simulator-independent reference function, so this check
    # cannot accidentally compare two copies of the Lab configuration.
    path = ROOT / "quadruped/envs/base/legged_robot_two.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "boundary_wall_layout")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["boundary_wall_layout"](8.0, 5.0, 0.12, 0.50, 0.05)


def colliders(stage, path):
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.HasAPI(UsdPhysics.CollisionAPI)]


def box_bounds(stage, path):
    prims = colliders(stage, path)
    assert len(prims) == 1, (path, len(prims))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], False, True)
    box = cache.ComputeWorldBound(prims[0]).ComputeAlignedRange()
    return np.asarray(box.GetSize()), np.asarray(box.GetMidpoint())


def check_assets(raw):
    stage = raw.sim.stage
    origins = raw.scene.env_origins.cpu().numpy()
    np.testing.assert_allclose(origins, raw.scene._default_env_origins.cpu().numpy(), atol=1e-6)
    assert len(np.unique(origins, axis=0)) == raw.num_envs, "Matches must have distinct reset origins"
    goal = ET.parse(ROOT / "resources/objects/goalpost/goalpost.urdf").getroot()
    goal_names = ("GoalEastNorth", "GoalEastSouth", "GoalEastCrossbar", "GoalWestNorth", "GoalWestSouth", "GoalWestCrossbar")
    wall_names = ("WallNorth", "WallSouth", "WallEastNorth", "WallEastSouth", "WallWestNorth", "WallWestSouth")
    layout = gym_wall_layout()
    for index, origin in enumerate(origins):
        prefix = f"/World/envs/env_{index}"
        for name, element in zip(goal_names, goal.findall(".//collision")):
            size, position = box_bounds(stage, f"{prefix}/{name}")
            np.testing.assert_allclose(size, np.fromstring(element.find("geometry/box").get("size"), sep=" "), atol=1e-6)
            np.testing.assert_allclose(position-origin, np.fromstring(element.find("origin").get("xyz"), sep=" "), atol=1e-6)
        for name, (kind, pos) in zip(wall_names, layout["walls"]):
            size, position = box_bounds(stage, f"{prefix}/{name}")
            np.testing.assert_allclose(size, layout[f"{kind}_dimensions"], atol=1e-6)
            np.testing.assert_allclose(position-origin, pos, atol=1e-6)
        size, position = box_bounds(stage, f"{prefix}/SoccerFieldVisual/Collider")
        field = ET.parse(ROOT / "resources/objects/soccer_field/soccer_field.urdf")
        np.testing.assert_allclose(size, np.fromstring(field.find(".//collision/geometry/box").get("size"), sep=" "), atol=1e-6)
        np.testing.assert_allclose(position-origin, (0, 0, -0.003), atol=1e-6)
        assert not colliders(stage, f"{prefix}/GoalVisual"), "Goal net must be visual-only"
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{prefix}/GoalVisual/Mesh"))
        assert len(mesh.GetFaceVertexCountsAttr().Get()) > 1000, "Missing detailed Gym goal mesh"
        for robot_index in (2, 3):
            root = stage.GetPrimAtPath(f"{prefix}/Robot_{robot_index}")
            material, _ = UsdShade.MaterialBindingAPI(root).ComputeBoundMaterial()
            shaders = [p for p in Usd.PrimRange(material.GetPrim()) if p.IsA(UsdShade.Shader)]
            color = next(UsdShade.Shader(p).GetInput("diffuseColor").Get() for p in shaders if UsdShade.Shader(p).GetInput("diffuseColor"))
            np.testing.assert_allclose(color, (0.85, 0.10, 0.10), atol=1e-6)
    ball = raw.scene["ball"]
    view = ball.root_physx_view
    masses = view.get_masses()
    assert bool(((masses >= 0.318*0.5) & (masses <= 0.318*0.8)).all())
    material = view.get_material_properties()
    np.testing.assert_allclose(material.cpu().numpy(), np.broadcast_to([1.0, 1.0, 0.85], material.shape), atol=1e-6)
    np.testing.assert_allclose(view.get_contact_offsets().cpu().numpy(), 0.01, atol=1e-6)
    expected_inertia = 0.4 * masses.cpu().numpy().reshape(-1, 1) * 0.0889**2
    inertia = view.get_inertias().cpu().numpy().reshape(-1, 9)[:, [0, 4, 8]]
    np.testing.assert_allclose(inertia, np.broadcast_to(expected_inertia, inertia.shape), rtol=1e-4)
    for index in range(4):
        robot = raw.scene[f"robot_{index}"]
        assert Path(robot.cfg.spawn.asset_path).resolve() == ROOT / "resources/robots/as2/urdf/as2.urdf"
        np.testing.assert_allclose(robot.root_physx_view.get_contact_offsets().cpu().numpy(), 0.01, atol=1e-6)
    return {"envs_checked": len(origins), "goal_colliders_per_env": 6, "walls_per_env": 6,
            "ball_masses_kg": masses.cpu().tolist(), "ball_material": material.cpu().tolist()}


def launch_ball(raw, position, velocity, steps=80, drag=False):
    ball = raw.scene["ball"]
    state = ball.data.default_root_state.clone()
    state[:, :3] = raw.scene.env_origins + state.new_tensor(position)
    state[:, 7:] = 0.0
    state[:, 7:10] = state.new_tensor(velocity)
    ball.write_root_state_to_sim(state)
    ball.set_external_force_and_torque(torch.zeros(raw.num_envs, 1, 3, device=raw.device),
                                       torch.zeros(raw.num_envs, 1, 3, device=raw.device), is_global=True)
    raw.scene.update(0.0)
    trace = []
    term = raw.action_manager.get_term("ball_drag")
    for step in range(steps):
        if drag:
            if step % raw.cfg.decimation == 0:
                term.process_actions(term.raw_actions)
            term.apply_actions()
        raw.scene.write_data_to_sim()
        raw.sim.step(render=False)
        raw.scene.update(raw.physics_dt)
        trace.append(torch.cat((ball.data.root_pos_w-raw.scene.env_origins, ball.data.root_lin_vel_w), dim=-1).clone())
    return torch.stack(trace)


def main():
    cfg = parse_env_cfg("Isaac-DribbleBot-AS2-Match-Macro-Flat-Play-v0", device=args.device, num_envs=args.num_envs)
    cfg.seed = 17
    cfg.wait_for_textures = False
    env = gym.make("Isaac-DribbleBot-AS2-Match-Macro-Flat-Play-v0", cfg=cfg)
    raw = env.unwrapped
    try:
        print("[ALIGN] resetting", flush=True)
        env.reset()
        print("[ALIGN] checking loaded assets", flush=True)
        report = check_assets(raw)
        assert raw.action_manager.total_action_dim == 16
        action = torch.zeros(raw.num_envs, 16, device=raw.device)
        for _ in range(3):
            obs, reward, _, _, _ = env.step(action)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
        # Park robots away from the contact experiments; step physics directly
        # so scoring/reset logic cannot hide a blocked goal or failed bounce.
        for index in range(4):
            robot = raw.scene[f"robot_{index}"]
            state = robot.data.default_root_state.clone()
            state[:, :3] = raw.scene.env_origins + state.new_tensor((-2.0+index, -1.5, 0.34))
            robot.write_root_state_to_sim(state)
        wall = launch_ball(raw, (0, 2.20, 0.25), (0, 2, 0))
        assert bool((wall[:, :, 4].min(dim=0).values < -0.3).all()), "Ball failed to rebound from side wall"
        goal = launch_ball(raw, (3.65, 0, 0.25), (2, 0, 0))
        assert bool((goal[-1, :, 0] > 4.1).all()), "Goal mouth blocked"
        assert bool(mdp.match_goal(raw).all())
        assert not bool(mdp.match_ball_out_of_bounds(raw, boundary_walls=True).any())
        drag = launch_ball(raw, (0, 0, 2.0), (1, 0, 0), steps=20, drag=True)
        assert bool((drag[-1, :, 3] < 0.99).all()), "Drag did not slow airborne ball"
        assert bool((drag[-1, :, 3] > 0.0).all()), "Drag reversed velocity"
        term = raw.action_manager.get_term("ball_drag")
        coefficients = term.coefficients.clone()
        term.reset(torch.tensor([0], device=raw.device))
        assert torch.equal(coefficients, term.coefficients), "Episode reset must not resample drag"
        assert not bool(term._forces[0].any())
        raw.common_step_counter = term._interval
        term.process_actions(term.raw_actions)
        assert not torch.equal(coefficients, term.coefficients), "15 s drag refresh missing"
        report.update({"passed": True, "action_dim": 16,
                       "wall_min_y_velocity": wall[:, :, 4].min(dim=0).values.cpu().tolist(),
                       "goal_final_x": goal[-1, :, 0].cpu().tolist(),
                       "drag_final_x_velocity": drag[-1, :, 3].cpu().tolist()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        app.close()
    raise SystemExit(exit_code)
