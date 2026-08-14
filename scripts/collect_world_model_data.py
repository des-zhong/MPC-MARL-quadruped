"""Collect complete AS2 skill-timescale episodes from the vectorized simulator."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

if "isaacgym" in sys.modules:
    isaacgym = sys.modules["isaacgym"]
elif "torch" not in sys.modules:
    import isaacgym
else:
    # Offline users may import dataset helpers after torch. Isaac Gym itself
    # requires the opposite order, so defer the simulator-only failure until
    # build_environment is called.
    isaacgym = None
from dribblebot.world_model.behavior_policies import BehaviorMixture
from dribblebot.world_model.config import load_config
from dribblebot.world_model.dataset import Episode, EpisodeShardWriter, split_episodes
from dribblebot.world_model.schema import EVENT_NAMES
from dribblebot.world_model.state_adapter import FootballWorldModelStateAdapter
from scripts.targeted_collection import (
    TargetedScenarioManager,
    configured_minimum_counts,
    coverage_deficits,
)
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
        raise ImportError("Import isaacgym before torch to build the AS2 simulator environment.")
    from dribblebot.envs.base.legged_robot_config import Cfg
    from dribblebot.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from scripts.train_high_level import configure_high_level_cfg, load_skill_policies

    env_config = config["environment"]
    configured_robot = env_config.get("robot", "as2")
    if configured_robot != "as2":
        raise ValueError(
            f"The AS2 collector requires environment.robot='as2', got {configured_robot!r}. "
            "Use scripts/go1_scripts/collect_world_model_data.py for GO1."
        )
    from dribblebot.envs.as2.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    joint_teams = "team_size" in env_config
    configured_count = (
        env_config.get("team_size", 2)
        if joint_teams
        else env_config.get("num_robots", 2)
    )
    high_args = argparse.Namespace(
        device=args.device, policy_device=args.policy_device,
        headless=True, project=None, num_envs=int(env_config.get("num_envs", 256)), iterations=0,
        self_play=joint_teams,
        num_robots=int(
            getattr(args, "num_robots", None)
            or configured_count
        ),
        episode_length=float(env_config.get("episode_length", 20.0)),
        control_interval=int(config["world_model"].get("macro_action_steps", 10)),
        high_level_history=int(config["world_model"].get("history_length", 1)),
        field_length=float(env_config.get("field_length", 8.0)), field_width=float(env_config.get("field_width", 5.0)),
        goal_half_width=float(env_config.get("goal_half_width", 1.0)),
        near_ball_init_probability=float(env_config.get("near_ball_init_probability", 0.4)),
        near_ball_init_min_distance=float(env_config.get("near_ball_init_min_distance", 0.4)),
        near_ball_init_max_distance=float(env_config.get("near_ball_init_max_distance", 0.95)),
        near_ball_init_max_angle=float(env_config.get("near_ball_init_max_angle", 0.35)),
        walk_x_speed_scale=args.walk_x_speed_scale, walk_y_speed_scale=args.walk_y_speed_scale,
        walk_yaw_speed_scale=args.walk_yaw_speed_scale, walk_yaw_reward_scale=args.walk_yaw_speed_scale,
        dribble_x_speed_scale=args.dribble_x_speed_scale, dribble_y_speed_scale=args.dribble_y_speed_scale,
        dribble_yaw_speed_scale=args.dribble_yaw_speed_scale,
        shoot_x_speed_scale=args.shoot_x_speed_scale, shoot_y_speed_scale=args.shoot_y_speed_scale,
        skill_checkpoint=args.skill_checkpoint, walk_wandb_run=args.walk_wandb_run,
        dribble_wandb_run=args.dribble_wandb_run, shoot_wandb_run=args.shoot_wandb_run,
        skill_policy_source=getattr(args, "skill_policy_source", "wandb"),
        walk_policy_dir=getattr(args, "walk_policy_dir", None),
        dribble_policy_dir=getattr(args, "dribble_policy_dir", None),
        shoot_policy_dir=getattr(args, "shoot_policy_dir", None),
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


def _executed_actions(action_adapter, info, device) -> torch.Tensor:
    """Repack what the wrapper actually ran, including affordance downgrades."""

    skills = torch.as_tensor(info["high_level_skill_ids"], device=device, dtype=torch.long)
    commands = torch.as_tensor(info["high_level_commands"], device=device, dtype=torch.float)
    actions = action_adapter.pack(skills, commands)
    action_adapter.assert_within_bounds(actions, atol=1e-5)
    return actions


def _termination_reason(env_index, done, timeout, events, next_state, schema, info) -> str:
    if not bool(done[env_index].item()):
        return "none"
    if bool(timeout[env_index].item()):
        return "timeout"
    event_row = events[env_index]
    for name in ("goal", "own_goal", "out_of_bounds", "ball_obstacle_collision"):
        if name in EVENT_NAMES and bool(event_row[EVENT_NAMES.index(name)].item() > 0.5):
            return name
    num_robots = len(
        [
            feature
            for feature in schema.features
            if feature.name.startswith("robot_")
            and feature.name.endswith(".position")
        ]
    )
    fallen = [
        bool(next_state[env_index, schema.slice(f"robot_{robot}.fallen")].item() > 0.5)
        for robot in range(num_robots)
    ]
    if any(fallen):
        return "all_robots_fell" if all(fallen) else f"robot_{fallen.index(True)}_fell"
    accidental = np.asarray(info.get("high_level_accidental_termination", []))
    if accidental.size and bool(accidental.reshape(-1)[env_index]):
        return "accidental_termination"
    return "other_termination"


def _update_event_counts(counter: Counter, event_labels: np.ndarray, event_names) -> None:
    totals = np.asarray(event_labels).sum(axis=0)
    counter.update({name: int(totals[index]) for index, name in enumerate(event_names) if totals[index]})


def _episode_limits(args, behavior_cfg, coverage_enabled: bool) -> tuple[int, int]:
    """Return the base target and absolute collection cap.

    A CLI-provided ``--num-episodes`` is an explicit hard limit. Configuration-
    driven runs retain the optional rare-event coverage extension for backward
    compatibility.
    """

    requested = getattr(args, "num_episodes", None)
    target = int(behavior_cfg["num_episodes"] if requested is None else requested)
    if target < 1:
        raise ValueError("--num-episodes must be at least 1")
    allow_coverage_extension = requested is None and coverage_enabled
    max_extra = (
        int(behavior_cfg.get("max_extra_episodes", 0))
        if allow_coverage_extension
        else 0
    )
    if max_extra < 0:
        raise ValueError("data_collection.max_extra_episodes must be non-negative")
    return target, target + max_extra


def collect(args) -> None:
    config = load_config(args.config)
    torch.manual_seed(int(config.get("seed", 42)))
    np.random.seed(int(config.get("seed", 42)))
    env = build_environment(args, config)
    num_robots = int(env.num_robots)
    team_size = int(getattr(env.env.cfg.env, "num_team_robots", num_robots // 2))
    config["environment"]["num_robots"] = num_robots
    config["environment"]["team_size"] = team_size
    config["world_model"]["max_obstacles"] = 0
    adapter = FootballWorldModelStateAdapter(
        env, 0, num_robots=num_robots
    )
    action_adapter = adapter.action_adapter
    behavior_cfg = config["data_collection"]
    minimum_counts = {} if getattr(args, "no_coverage_quota", False) else configured_minimum_counts(behavior_cfg)
    unknown_coverage_events = sorted(set(minimum_counts) - set(EVENT_NAMES))
    if unknown_coverage_events:
        raise ValueError(f"minimum_event_counts contains unknown events: {unknown_coverage_events}")
    metadata = {
        "robot": str(env.env.cfg.robot.name),
        "num_robots": env.num_robots,
        "team_size": team_size,
        "team_robot_slots": {
            "learning": list(range(team_size)),
            "opponent": list(range(team_size, 2 * team_size)),
        },
        "team_behavior": {
            "learning": "shared configured behavior mixture; attacks +x",
            "opponent": "same behavior mixture; attacks -x",
        },
        "num_obstacles": env.env.num_static_opponents,
        "state_schema": adapter.schema.to_dict(), "action_schema": action_adapter.to_dict(),
        "event_names": list(EVENT_NAMES), "macro_action_steps": env.control_interval,
        "low_level_control_dt_seconds": float(env.env.dt),
        "physics_dt_seconds": float(env.env.sim_params.dt),
        "coordinate_frame": (
            "fixed learning-team frame: learning slots attack +x; opponent slots attack -x"
        ), "config": config,
        "minimum_event_counts": minimum_counts,
        "skill_policy_metadata": {
            skill: record.get("policy_metadata", {})
            for skill, record in getattr(env, "skill_policies", {}).items()
        },
    }
    writer = EpisodeShardWriter(args.output, metadata)
    policy = BehaviorMixture(
        action_adapter, adapter.schema, behavior_cfg["behavior_mixture"],
        float(behavior_cfg.get("repeat_previous_skill_probability", 0.35)),
        seed=int(config.get("seed", 42)),
        random_sampling=behavior_cfg.get("random_sampling"),
        team_size=team_size,
    )
    scenario_manager = TargetedScenarioManager(
        env,
        behavior_cfg.get("targeted_rare_events", {"enabled": False}),
        seed=int(config.get("seed", 42)) + 1,
    )
    capture = TerminalStateCapture(env, adapter)
    env.reset()
    capture.clear()
    buffers = [[] for _ in range(env.num_envs)]
    episode_ids = np.arange(env.num_envs, dtype=np.int64)
    next_episode_id = env.num_envs
    previous_actions = None
    # There is no action to repeat on the first collection step.  Once the
    # first action has executed, the validity mask below prevents reset rows
    # from repeating an action from their previous episode.
    previous_action_valid = None
    completed = 0
    target, total_cap = _episode_limits(args, behavior_cfg, bool(minimum_counts))
    event_counts = Counter()
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    next_report = 100
    try:
        scenario_manager.stage(all_env_ids, event_counts, minimum_counts)
        while completed < total_cap and (completed < target or coverage_deficits(event_counts, minimum_counts)):
            state = adapter.extract_state(env)["tensor"]
            targeted_scenarios = scenario_manager.policy_scenarios()
            actions, sources = policy.sample(
                state,
                previous_actions,
                targeted_scenarios=targeted_scenarios,
                previous_action_valid=previous_action_valid,
            )
            capture.clear()
            _, reward, done, info = env.step(action_adapter.to_wrapper_action(actions))
            executed_actions = _executed_actions(action_adapter, info, state.device)
            live_next_state = adapter.extract_state(env)["tensor"]
            next_state = torch.where(capture.valid[:, None], capture.states, live_next_state)
            events = adapter.extract_event_labels(state, next_state, info)
            timeouts = torch.as_tensor(info.get("time_outs", np.zeros(env.num_envs)), device=done.device).bool()
            elapsed = np.asarray(info["elapsed_low_level_steps"])
            invalid_skills = np.asarray(
                info.get(
                    "high_level_invalid_skill_mask",
                    np.zeros((env.num_envs, env.num_robots), dtype=bool),
                )
            ).any(axis=1)
            done_ids = []
            for env_index in range(env.num_envs):
                buffers[env_index].append({
                    "state": state[env_index].detach().cpu().numpy().astype(np.float32),
                    "joint_action": executed_actions[env_index].detach().cpu().numpy().astype(np.float32),
                    "requested_joint_action": actions[env_index].detach().cpu().numpy().astype(np.float32),
                    "requested_action_modified": np.bool_(
                        not torch.allclose(actions[env_index], executed_actions[env_index], atol=1e-5)
                    ),
                    "invalid_skill_requested": np.bool_(invalid_skills[env_index]),
                    "reward": np.float32(reward[env_index].item()),
                    "next_state": next_state[env_index].detach().cpu().numpy().astype(np.float32),
                    "terminated": np.bool_(done[env_index].item() and not timeouts[env_index].item()),
                    "truncated": np.bool_(timeouts[env_index].item()),
                    "elapsed_low_level_steps": np.int16(elapsed[env_index]),
                    "event_labels": events[env_index].detach().cpu().numpy().astype(np.float32),
                    "behavior_source": sources[env_index],
                    "targeted_scenario": targeted_scenarios[env_index],
                    "termination_reason": _termination_reason(
                        env_index, done, timeouts, events, next_state, adapter.schema, info
                    ),
                })
                if bool(done[env_index].item()):
                    done_ids.append(env_index)
                    need_episode = completed < target or bool(coverage_deficits(event_counts, minimum_counts))
                    if completed < total_cap and need_episode:
                        episode = Episode(
                            int(episode_ids[env_index]),
                            _episode_arrays(int(episode_ids[env_index]), buffers[env_index]),
                        )
                        writer.write_episode(episode)
                        _update_event_counts(event_counts, episode.arrays["event_labels"], EVENT_NAMES)
                        completed += 1
                    buffers[env_index] = []
                    episode_ids[env_index] = next_episode_id
                    next_episode_id += 1
            scenario_manager.observe(events, EVENT_NAMES)
            previous_actions = actions.detach().clone()
            previous_action_valid = ~done.bool()
            if done_ids and completed < total_cap and (
                completed < target or coverage_deficits(event_counts, minimum_counts)
            ):
                reset_ids = torch.as_tensor(done_ids, dtype=torch.long, device=env.device)
                force_targeted = completed >= target and bool(coverage_deficits(event_counts, minimum_counts))
                scenario_manager.stage(
                    reset_ids,
                    event_counts,
                    minimum_counts,
                    force=force_targeted,
                )
            if completed >= next_report:
                deficits = coverage_deficits(event_counts, minimum_counts)
                suffix = f", deficits={deficits}" if deficits else ""
                print(f"collected {completed}/{target} base episodes{suffix}")
                next_report = ((completed // 100) + 1) * 100
    finally:
        capture.restore()
        env.close()
    dataset_cfg = config["dataset"]
    splits = split_episodes(args.output, dataset_cfg["train_fraction"], dataset_cfg["validation_fraction"], dataset_cfg["test_fraction"], int(config.get("seed", 42)))
    remaining = coverage_deficits(event_counts, minimum_counts)
    summary = {
        "output": str(args.output),
        "completed_episodes": completed,
        "base_episode_target": target,
        "collection_episode_cap": total_cap,
        "coverage_extension_enabled": total_cap > target,
        "event_counts": dict(event_counts),
        "coverage_deficits": remaining,
        "targeted_scenarios_staged": dict(scenario_manager.staged_counts),
        "splits": {key: len(value) for key, value in splits.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if remaining and bool(behavior_cfg.get("require_minimum_event_counts", True)):
        raise RuntimeError(
            f"Rare-event coverage was not reached after {completed} episodes; remaining deficits: {remaining}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/world_model_as2.yaml")
    parser.add_argument("--output", default="data/world_model_as2")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument(
        "--num-robots",
        type=int,
        default=None,
        help="Robots per team; the joint world-model contains twice this count.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--skill-checkpoint", default="latest")
    parser.add_argument("--walk-wandb-run", default="des_zhong/as2_walking/3a6g1def")
    parser.add_argument("--dribble-wandb-run", default="des_zhong/as2_dribbling/cp9m21ay")
    parser.add_argument("--shoot-wandb-run", default="des_zhong/as2_shooting/bve3isir")
    parser.add_argument(
        "--skill-policy-source",
        choices=("wandb", "local"),
        default="wandb",
        help="Load low-level skills from online W&B or directly from --*-policy-dir folders.",
    )
    parser.add_argument("--walk-policy-dir", default=None)
    parser.add_argument("--dribble-policy-dir", default=None)
    parser.add_argument("--shoot-policy-dir", default=None)
    parser.add_argument("--walk-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--dribble-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--shoot-x-speed-scale", type=float, default=3.0)
    parser.add_argument("--shoot-y-speed-scale", type=float, default=3.0)
    parser.add_argument(
        "--no-coverage-quota",
        action="store_true",
        help="Collect exactly --num-episodes without enforcing configured rare-event minimums.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
