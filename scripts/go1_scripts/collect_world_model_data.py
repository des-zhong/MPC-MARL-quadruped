"""Collect complete Go1 skill-timescale episodes from the vectorized simulator."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

if "isaacgym" in sys.modules:
    isaacgym = sys.modules["isaacgym"]
elif "torch" not in sys.modules:
    import isaacgym
else:
    isaacgym = None
from dribblebot.world_model.behavior_policies import BehaviorMixture
from dribblebot.world_model.config import load_config
from dribblebot.world_model.dataset import Episode, EpisodeShardWriter, split_episodes
from dribblebot.world_model.schema import EVENT_NAMES
from dribblebot.world_model.state_adapter import FootballWorldModelStateAdapter
import torch

class TerminalStateCapture:
    """Capture terminal simulator state immediately before the env's automatic reset."""

    def __init__(self, wrapper, adapter: FootballWorldModelStateAdapter):
        self.wrapper = wrapper
        self.raw = wrapper.env
        self.adapter = adapter
        self.original_reset_idx = self.raw.reset_idx
        self.states = torch.zeros(self.raw.num_envs, adapter.state_dim, device=self.raw.device)
        self.valid = torch.zeros(self.raw.num_envs, dtype=torch.bool, device=self.raw.device)

        def reset_with_capture(env_ids):
            if env_ids.numel():
                uncaptured = env_ids[~self.valid[env_ids]]
                if uncaptured.numel():
                    snapshot = self.adapter.extract_state(self.wrapper)["tensor"]
                    self.states[uncaptured] = snapshot[uncaptured]
                    self.valid[uncaptured] = True
            return self.original_reset_idx(env_ids)

        self.raw.reset_idx = reset_with_capture

    def clear(self) -> None:
        self.valid.zero_()

    def restore(self) -> None:
        self.raw.reset_idx = self.original_reset_idx


def build_environment(args, config):
    # Imports are intentionally local so offline model tools do not require Isaac Gym.

    if isaacgym is None:
        raise ImportError("Import isaacgym before torch to build the GO1 simulator environment.")
    from dribblebot.envs.base.legged_robot_config import Cfg
    from dribblebot.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from scripts.go1_scripts.training_high_level import configure_high_level_cfg, load_skill_policies

    env_config = config["environment"]
    env_config["robot"] = "go1"
    robot = "go1"
    from dribblebot.envs.go1.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    high_args = argparse.Namespace(
        robot=robot, device=args.device, policy_device=args.policy_device,
        headless=True, project=None, num_envs=int(env_config.get("num_envs", 256)), iterations=0,
        num_robots=int(getattr(args, "num_robots", None) or env_config.get("num_robots", 2)),
        episode_length=float(env_config.get("episode_length", 20.0)),
        control_interval=int(config["world_model"].get("macro_action_steps", 10)),
        high_level_history=int(config["world_model"].get("history_length", 1)),
        field_length=float(env_config.get("field_length", 8.0)), field_width=float(env_config.get("field_width", 5.0)),
        goal_half_width=float(env_config.get("goal_half_width", 1.0)),
        walk_x_speed_scale=args.walk_x_speed_scale, walk_y_speed_scale=args.walk_y_speed_scale,
        walk_yaw_speed_scale=args.walk_yaw_speed_scale, walk_yaw_reward_scale=args.walk_yaw_speed_scale,
        dribble_x_speed_scale=args.dribble_x_speed_scale, dribble_y_speed_scale=args.dribble_y_speed_scale,
        dribble_yaw_speed_scale=args.dribble_yaw_speed_scale,
        shoot_x_speed_scale=args.shoot_x_speed_scale, shoot_y_speed_scale=args.shoot_y_speed_scale,
        skill_checkpoint=args.skill_checkpoint, walk_wandb_run=args.walk_wandb_run,
        dribble_wandb_run=args.dribble_wandb_run, shoot_wandb_run=args.shoot_wandb_run,
    )
    configure_high_level_cfg(Cfg, high_args)
    policies = load_skill_policies(high_args)
    raw = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=True, cfg=Cfg)
    return HighLevelSkillWrapper(raw, policies, control_interval=high_args.control_interval, history_length=high_args.high_level_history)


def _episode_arrays(episode_id: int, records):
    result = {}
    for key in records[0]:
        values = [record[key] for record in records]
        result[key] = np.asarray(values)
    length = len(records)
    result["episode_id"] = np.full(length, episode_id, dtype=np.int64)
    result["step_id"] = np.arange(length, dtype=np.int64)
    return result


def collect(args) -> None:
    config = load_config(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))
    env = build_environment(args, config)
    num_robots = int(env.num_robots)
    config["environment"]["num_robots"] = num_robots
    config["world_model"]["max_obstacles"] = num_robots
    adapter = FootballWorldModelStateAdapter(
        env, num_robots, num_robots=num_robots
    )
    action_adapter = adapter.action_adapter
    metadata = {
        "robot": str(env.env.cfg.robot.name),
        "num_robots": num_robots,
        "num_obstacles": env.env.num_static_opponents,
        "state_schema": adapter.schema.to_dict(), "action_schema": action_adapter.to_dict(),
        "event_names": list(EVENT_NAMES), "macro_action_steps": env.control_interval,
        "low_level_control_dt_seconds": float(env.env.dt),
        "physics_dt_seconds": float(env.env.sim_params.dt),
        "coordinate_frame": "fixed team frame, attacking +x", "config": config,
    }
    writer = EpisodeShardWriter(args.output, metadata)
    behavior_cfg = config["data_collection"]
    policy = BehaviorMixture(
        action_adapter, adapter.schema, behavior_cfg["behavior_mixture"],
        float(behavior_cfg.get("repeat_previous_skill_probability", 0.35)), seed=int(config.get("seed", 42)),
    )
    capture = TerminalStateCapture(env, adapter)
    env.reset()
    capture.clear()
    buffers = [[] for _ in range(env.num_envs)]
    episode_ids = np.arange(env.num_envs, dtype=np.int64)
    next_episode_id = env.num_envs
    previous_actions = None
    completed = 0
    target = int(args.num_episodes or behavior_cfg["num_episodes"])
    try:
        while completed < target:
            state = adapter.extract_state(env)["tensor"]
            actions, sources = policy.sample(state, previous_actions)
            capture.clear()
            _, reward, done, info = env.step(action_adapter.to_wrapper_action(actions))
            live_next_state = adapter.extract_state(env)["tensor"]
            next_state = torch.where(capture.valid[:, None], capture.states, live_next_state)
            events = adapter.extract_event_labels(state, next_state, info)
            timeouts = torch.as_tensor(info.get("time_outs", np.zeros(env.num_envs)), device=done.device).bool()
            elapsed = np.asarray(info["elapsed_low_level_steps"])
            for env_index in range(env.num_envs):
                buffers[env_index].append({
                    "state": state[env_index].detach().cpu().numpy().astype(np.float32),
                    "joint_action": actions[env_index].detach().cpu().numpy().astype(np.float32),
                    "reward": np.float32(reward[env_index].item()),
                    "next_state": next_state[env_index].detach().cpu().numpy().astype(np.float32),
                    "terminated": np.bool_(done[env_index].item() and not timeouts[env_index].item()),
                    "truncated": np.bool_(timeouts[env_index].item()),
                    "elapsed_low_level_steps": np.int16(elapsed[env_index]),
                    "event_labels": events[env_index].detach().cpu().numpy().astype(np.float32),
                    "behavior_source": sources[env_index],
                })
                if done[env_index] and completed < target:
                    writer.write_episode(Episode(int(episode_ids[env_index]), _episode_arrays(int(episode_ids[env_index]), buffers[env_index])))
                    completed += 1
                    buffers[env_index] = []
                    episode_ids[env_index] = next_episode_id
                    next_episode_id += 1
            previous_actions = actions
            if completed and completed % 100 == 0:
                print(f"collected {completed}/{target} complete episodes")
    finally:
        capture.restore()
        env.close()
    dataset_cfg = config["dataset"]
    splits = split_episodes(args.output, dataset_cfg["train_fraction"], dataset_cfg["validation_fraction"], dataset_cfg["test_fraction"], int(config.get("seed", 42)))
    print(json.dumps({"output": str(args.output), "completed_episodes": completed, "splits": {k: len(v) for k, v in splits.items()}}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/world_model.yaml")
    parser.add_argument("--output", default="data/world_model")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--skill-checkpoint", default="latest")
    parser.add_argument("--walk-wandb-run", default="des_zhong/walking/fdmj3ehy")
    parser.add_argument("--dribble-wandb-run", default="des_zhong/dribbling/uu2vgloi")
    parser.add_argument("--shoot-wandb-run", default="des_zhong/shooting/lj807eqa")
    parser.add_argument("--walk-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--dribble-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--shoot-x-speed-scale", type=float, default=3.0)
    parser.add_argument("--shoot-y-speed-scale", type=float, default=3.0)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
