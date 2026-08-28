"""Run a bounded coordinator-rate smoke test for a macro-wrapped Lab task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REBUILD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REBUILD_ROOT not in sys.path:
    sys.path.insert(0, REBUILD_ROOT)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Coordinator-Macro-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=cfg)
    raw_env = env.unwrapped
    try:
        reset_obs, reset_info = env.reset()
        print(f"[MACRO] reset groups={list(reset_obs)} info={reset_info}", flush=True)
        action_shape = env.action_space.shape
        action = torch.zeros(action_shape, device=raw_env.device)
        action_dim = int(action_shape[-1])
        records = []
        for index in range(args_cli.steps):
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = info["elapsed_low_level_steps"]
            print(
                f"[MACRO] step={index} action_dim={action_dim} "
                f"elapsed={elapsed.detach().cpu().tolist()} "
                f"reward={reward.detach().cpu().tolist()} "
                f"terminated={terminated.detach().cpu().tolist()} "
                f"truncated={truncated.detach().cpu().tolist()}",
                flush=True,
            )
            if not torch.isfinite(reward).all():
                raise RuntimeError("macro reward contains non-finite values")
            records.append(
                {
                    "step": index,
                    "elapsed_low_level_steps": elapsed.detach().cpu().tolist(),
                    "reward": reward.detach().cpu().tolist(),
                    "terminated": terminated.detach().cpu().tolist(),
                    "truncated": truncated.detach().cpu().tolist(),
                    "finite": True,
                }
            )
        recorder_snapshot = {}
        recorder_manager = getattr(raw_env, "recorder_manager", None)
        if recorder_manager is not None and len(getattr(recorder_manager, "active_terms", [])) > 0:
            episode_data = recorder_manager.get_episode(0).data
            snapshot = episode_data.get("football", {}).get("snapshot", {})
            recorder_snapshot = {
                key: {
                    "samples": len(value),
                    "shape": list(value[0].shape),
                }
                for key, value in snapshot.items()
                if isinstance(value, list) and value
            }
        if args_cli.output is not None:
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(
                json.dumps(
                    {
                        "task": args_cli.task,
                        "num_envs": args_cli.num_envs,
                        "requested_steps": args_cli.steps,
                        "action_shape": list(action_shape),
                        "records": records,
                        "recorder_snapshot": recorder_snapshot,
                        "passed": all(record["finite"] for record in records),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[MACRO] report={args_cli.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
