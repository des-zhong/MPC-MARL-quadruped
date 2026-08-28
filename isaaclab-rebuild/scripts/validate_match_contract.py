"""Validate the four-AS2 match action, fallback, observation, and snapshot seams."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401
from dribblebot_isaaclab.world_model_adapter import IsaacLabFootballWorldModelStateAdapter  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped
    try:
        reset_obs, _ = env.reset()
        action = torch.zeros(env.action_space.shape, device=raw_env.device)
        # Ask both learning-team robots to shoot while the deterministic reset
        # places the ball outside strike range.  The legacy geometric fallback
        # should execute walk and mark the request invalid.
        action[:, 2] = 3.0
        _, _, terminated, truncated, info = env.step(action)
        episode_data = raw_env.recorder_manager.get_episode(0).data["football"]["snapshot"]
        snapshot = {key: torch.stack(values) for key, values in episode_data.items()}
        world_model_state = IsaacLabFootballWorldModelStateAdapter(max_obstacles=0).extract_state(snapshot)["tensor"]
        terms = [raw_env.action_manager.get_term(f"skill_policy_{index}") for index in range(4)]
        report = {
            "task": args_cli.task,
            "observation_shape": list(reset_obs["obs"].shape),
            "observation_history_shape": list(reset_obs["obs_history"].shape),
            "world_model_state_shape": list(world_model_state.shape),
            "action_shape": list(env.action_space.shape),
            "executed_skill_ids": [term.skill_ids.detach().cpu().tolist() for term in terms],
            "requested_skill_ids": [term.requested_skill_ids.detach().cpu().tolist() for term in terms],
            "invalid_skill_masks": [term.invalid_skill_mask.detach().cpu().tolist() for term in terms],
            "elapsed_low_level_steps": info["elapsed_low_level_steps"].detach().cpu().tolist(),
            "terminated": terminated.detach().cpu().tolist(),
            "truncated": truncated.detach().cpu().tolist(),
            "passed": (
                reset_obs["obs"].shape[-1] == 34
                and reset_obs["obs"].shape[0] == 2
                and reset_obs["obs_history"].shape == (2, 136)
                and world_model_state.shape[-1] > 0
                and torch.isfinite(world_model_state).all().item()
                and all(term.requested_skill_ids.eq(2).all() for term in terms[:2])
                and all(term.invalid_skill_mask.all() for term in terms[:2])
                and all(term.skill_ids.eq(0).all() for term in terms[:2])
            ),
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
