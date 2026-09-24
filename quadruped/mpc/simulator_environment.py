"""Simulator construction shared by standalone MPC tools."""

import argparse

def build_environment(args, config):
    # Imports are intentionally local so offline model tools do not require Isaac Gym.

    from quadruped.envs.base.legged_robot_config import Cfg
    from quadruped.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from scripts.train_high_level import configure_high_level_cfg, load_skill_policies

    env_config = config["environment"]
    configured_robot = env_config.get("robot", "as2")
    if configured_robot != "as2":
        raise ValueError(
            f"The AS2 collector requires environment.robot='as2', got {configured_robot!r}. "
            "Use scripts/go1_scripts/collect_world_model_data.py for GO1."
        )
    from quadruped.envs.as2.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
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
        boundary_walls=bool(env_config.get("boundary_walls", True)),
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
    if bool(env_config.get("fixed_initial_state", False)):
        Cfg.env.randomize_match_init = False
        Cfg.env.deterministic_match_init = True
    if bool(env_config.get("disable_domain_randomization", False)):
        for name in dir(Cfg.domain_rand):
            if name.startswith("randomize_") or name == "push_robots":
                value = getattr(Cfg.domain_rand, name)
                if isinstance(value, bool):
                    setattr(Cfg.domain_rand, name, False)
    policies = load_skill_policies(high_args)
    raw = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=True, cfg=Cfg)
    return HighLevelSkillWrapper(raw, policies, control_interval=high_args.control_interval, history_length=high_args.high_level_history)

