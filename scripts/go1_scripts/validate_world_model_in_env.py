"""Compare learned Go1 open-loop predictions with identical simulator actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import isaacgym
import numpy as np
import torch

from dribblebot.world_model.config import load_config
from dribblebot.world_model.state_adapter import FootballWorldModelStateAdapter
from dribblebot.world_model.trainer import load_checkpoint
from scripts.go1_scripts.collect_world_model_data import TerminalStateCapture, build_environment


@torch.no_grad()
def main(args):
    config = load_config(args.config)
    config["environment"]["num_envs"] = args.num_trajectories
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    num_robots = model.action_adapter.num_robots if args.num_robots is None else int(args.num_robots)
    if num_robots != model.action_adapter.num_robots:
        raise ValueError(
            f"--num-robots={num_robots} does not match checkpoint robot count "
            f"{model.action_adapter.num_robots}"
        )
    config["environment"]["num_robots"] = num_robots
    config["world_model"]["max_obstacles"] = num_robots
    config["environment"]["robot"] = "go1"
    requested_robot = "go1"
    checkpoint_robot = checkpoint.get("training_config", {}).get("environment", {}).get("robot")
    if checkpoint_robot and checkpoint_robot != requested_robot:
        raise ValueError(
            f"Checkpoint was trained for {checkpoint_robot!r}, but validation requested {requested_robot!r}. "
            "Use a robot-matched dataset/checkpoint and config."
        )
    model.eval()
    env = build_environment(args, config)
    adapter = FootballWorldModelStateAdapter(
        env,
        sum(feature.name.startswith("obstacle_") for feature in model.schema.features),
        model.schema,
        num_robots=model.action_adapter.num_robots,
    )
    capture = TerminalStateCapture(env, adapter)
    env.reset(); capture.clear()
    initial = adapter.extract_state(env)["tensor"]
    actions = model.action_adapter.random_valid((env.num_envs, 1, args.horizon), initial.device)
    prediction = model.rollout(initial.to(args.device), actions.to(args.device))["predicted_states"].cpu()
    actual = [initial.cpu()]
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=initial.device)
    try:
        for step in range(args.horizon):
            action = actions[:, 0, step]
            capture.clear()
            _, _, done, _ = env.step(model.action_adapter.to_wrapper_action(action))
            live = adapter.extract_state(env)["tensor"]
            terminal = torch.where(capture.valid[:, None], capture.states, live)
            actual.append(terminal.cpu())
            alive &= ~done
    finally:
        capture.restore(); env.close()
    actual = torch.stack(actual, dim=1)
    predicted = prediction[:, 0]
    error = predicted - actual
    metrics = {"state_rmse": float(error.square().mean().sqrt()), "state_mae": float(error.abs().mean())}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trajectories.npz", predicted=predicted.numpy(), actual=actual.numpy(), actions=actions[:, 0].cpu().numpy())
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/world_model/best.pt")
    parser.add_argument("--config", default="configs/world_model.yaml")
    parser.add_argument("--output", default="outputs/world_model_env_validation")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--skill-checkpoint", default="latest")
    parser.add_argument("--walk-wandb-run", default="des_zhong/walking/fdmj3ehy")
    parser.add_argument("--dribble-wandb-run", default="des_zhong/dribbling/uu2vgloi")
    parser.add_argument("--shoot-wandb-run", default="des_zhong/shooting/lj807eqa")
    parser.add_argument("--walk-x-speed-scale", type=float, default=1.5); parser.add_argument("--walk-y-speed-scale", type=float, default=1.5); parser.add_argument("--walk-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--dribble-x-speed-scale", type=float, default=1.5); parser.add_argument("--dribble-y-speed-scale", type=float, default=1.5); parser.add_argument("--dribble-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--shoot-x-speed-scale", type=float, default=3.0); parser.add_argument("--shoot-y-speed-scale", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
