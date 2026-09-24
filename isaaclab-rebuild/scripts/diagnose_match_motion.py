"""Check whether the four-AS2 match actually moves under an explicit walk action."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REBUILD_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REBUILD_ROOT / "source" / "dribblebot_isaaclab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Match-Macro-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--walk_speed", type=float, default=1.0)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")
if args_cli.steps <= 0 or args_cli.num_envs <= 0:
    raise ValueError("--steps and --num_envs must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import dribblebot_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> dict:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    cfg.seed = 17
    cfg.wait_for_textures = False
    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped
    try:
        env.reset()
        robot_names = ("robot_0", "robot_1", "robot_2", "robot_3")
        start = torch.stack([raw_env.scene[name].data.root_pos_w[:, :2].clone() for name in robot_names], dim=1)
        action = torch.zeros(raw_env.num_envs, raw_env.action_manager.total_action_dim, device=raw_env.device)
        for slot in range(4):
            action[:, 4 * slot] = 0.0  # walk skill
            action[:, 4 * slot + 1] = float(args_cli.walk_speed)
        root_trace = [start]
        target_trace = []
        with torch.inference_mode():
            for step in range(args_cli.steps):
                _, reward, terminated, truncated, _ = env.step(action)
                if not torch.isfinite(reward).all():
                    raise RuntimeError(f"non-finite reward at step {step}")
                root_trace.append(torch.stack([raw_env.scene[name].data.root_pos_w[:, :2].clone() for name in robot_names], dim=1))
                target_trace.append(raw_env.action_manager.get_term("skill_policy_0").processed_actions[:, :3].clone())
                if bool((terminated | truncated).any()):
                    break
        roots = torch.stack(root_trace, dim=0)
        targets = torch.stack(target_trace, dim=0) if target_trace else torch.empty(0, device=raw_env.device)
        displacement = roots[-1] - roots[0]
        result = {
            "task": args_cli.task,
            "num_envs": raw_env.num_envs,
            "requested_steps": args_cli.steps,
            "executed_steps": int(roots.shape[0] - 1),
            "walk_speed_input": args_cli.walk_speed,
            "start_xy": start.detach().cpu().tolist(),
            "end_xy": roots[-1].detach().cpu().tolist(),
            "displacement_xy": displacement.detach().cpu().tolist(),
            "max_abs_displacement": float(displacement.abs().max().item()),
            "max_processed_target": float(targets.abs().max().item()) if targets.numel() else 0.0,
            "passed_motion": bool(displacement.abs().max().item() > 1.0e-3),
        }
        if args_cli.output is not None:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        env.close()


if __name__ == "__main__":
    try:
        result = main()
        raise SystemExit(0 if result["passed_motion"] else 1)
    finally:
        simulation_app.close()
