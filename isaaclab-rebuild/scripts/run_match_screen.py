"""Run the four-AS2 match in a real Isaac Sim window for screen capture."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REBUILD_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REBUILD_ROOT / "source" / "dribblebot_isaaclab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--capture_delay", type=float, default=3.0)
parser.add_argument("--frame_delay", type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _action(skill_id: int, command_xy: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(command_xy.shape[0], 6, device=command_xy.device)
    result[:, skill_id] = 3.0
    result[:, 3:5] = command_xy.clamp(-2.0, 2.0)
    return result


def _team_action(raw_env) -> torch.Tensor:
    ball_xy = raw_env.scene["ball"].data.root_pos_w[:, :2]
    actions = []
    for name in ("robot_0", "robot_1"):
        robot = raw_env.scene[name]
        delta = ball_xy - robot.data.root_pos_w[:, :2]
        delta = quat_apply_inverse(
            robot.data.root_quat_w,
            torch.cat((delta, torch.zeros_like(delta[:, :1])), dim=-1),
        )[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / distance.clamp_min(1.0e-6).unsqueeze(-1)
        walk = _action(0, 1.25 * direction)
        shoot = (distance <= 0.70) & (direction[:, 0] >= -0.10) & (direction[:, 1].abs() <= 0.40)
        if bool(shoot.any()):
            kick = _action(
                2,
                torch.cat((torch.ones_like(direction[:, :1]), torch.zeros_like(direction[:, :1])), dim=-1),
            )
            walk[shoot] = kick[shoot]
        actions.append(walk)
    return torch.stack(actions, dim=1).reshape(-1, 6)


def _all_robot_action(raw_env) -> torch.Tensor:
    """Build one manager action for all four robots in the match."""

    ball_xy = raw_env.scene["ball"].data.root_pos_w[:, :2]
    actions = []
    for name in ("robot_0", "robot_1", "robot_2", "robot_3"):
        robot = raw_env.scene[name]
        delta = ball_xy - robot.data.root_pos_w[:, :2]
        delta = quat_apply_inverse(
            robot.data.root_quat_w,
            torch.cat((delta, torch.zeros_like(delta[:, :1])), dim=-1),
        )[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / distance.clamp_min(1.0e-6).unsqueeze(-1)
        action = _action(0, 1.0 * direction)
        shoot = (distance <= 0.65) & (direction[:, 0] >= -0.10) & (direction[:, 1].abs() <= 0.45)
        if bool(shoot.any()):
            kick = _action(
                2,
                torch.cat((torch.ones_like(direction[:, :1]), torch.zeros_like(direction[:, :1])), dim=-1),
            )
            action[shoot] = kick[shoot]
        actions.append(action)
    return torch.stack(actions, dim=1).reshape(raw_env.num_envs, -1)


def _opponent(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    history = observation["obs_history"]
    result = torch.zeros(history.shape[0], 6, device=history.device)
    result[:, 0] = 3.0
    result[:, 3] = 1.25
    return result


def _reset_without_view_forward(env) -> None:
    raw_env = env.unwrapped
    env_ids = torch.arange(raw_env.num_envs, dtype=torch.long, device=raw_env.device)
    raw_env.recorder_manager.record_pre_reset(env_ids)
    raw_env._reset_idx(env_ids)
    raw_env.scene.write_data_to_sim()
    raw_env.sim.step(render=False)
    raw_env.scene.update(dt=raw_env.physics_dt)
    raw_env.recorder_manager.record_post_reset(env_ids)
    raw_env.obs_buf = raw_env.observation_manager.compute(update_history=True)
    env._history.zero_()
    env._last_team_actions.zero_()
    env._update_observations(raw_env.obs_buf, reset_mask=None)


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    cfg.seed = 7
    cfg.wait_for_textures = False
    cfg.viewer.eye = (7.5, -8.5, 7.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env = gym.make(args_cli.task, cfg=cfg, render_mode="human")
    match_env = env
    while not hasattr(match_env, "_history") and hasattr(match_env, "env"):
        match_env = match_env.env
    if not hasattr(match_env, "_history"):
        raise RuntimeError("Could not locate MatchSelfPlayWrapper in the Gym wrapper chain")
    match_env.set_opponent_callable(_opponent)
    raw_env = env.unwrapped
    _reset_without_view_forward(match_env)
    robot_xy = torch.stack(
        [raw_env.scene[name].data.root_pos_w[:, :2] for name in ("robot_0", "robot_1", "robot_2", "robot_3")],
        dim=1,
    )
    print(f"[SCREEN] robot_xy={robot_xy.detach().cpu().tolist()}", flush=True)
    for _ in range(3):
        raw_env.sim.render()
    print("[SCREEN] match ready; capture the Isaac Sim window now", flush=True)
    deadline = time.monotonic() + max(0.0, args_cli.capture_delay)
    while time.monotonic() < deadline:
        simulation_app.update()
        time.sleep(0.02)
    with torch.inference_mode():
        for index in range(args_cli.steps):
            _, reward, terminated, truncated, _ = raw_env.step(_all_robot_action(raw_env))
            raw_env.sim.render()
            if not torch.isfinite(reward).all():
                raise RuntimeError(f"non-finite reward at step {index}")
            if (index + 1) % 5 == 0:
                print(
                    f"[SCREEN] step={index + 1}/{args_cli.steps} "
                    f"reward={reward.detach().cpu().tolist()} "
                    f"done={(terminated | truncated).detach().cpu().tolist()}",
                    flush=True,
                )
            if args_cli.frame_delay > 0.0:
                time.sleep(args_cli.frame_delay)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
