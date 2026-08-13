import argparse


DEFAULT_GO1_SKILL_RUNS = {
    "walk": "des_zhong/walking/fdmj3ehy",
    "dribble": "des_zhong/dribbling/uu2vgloi",
    "shoot": "des_zhong/shooting/lj807eqa",
}


def configure_high_level_cfg(Cfg, args):
    from dribblebot.envs.go1.go1_config import config_go1

    config_go1(Cfg)
    walk_x_scale = abs(float(args.walk_x_speed_scale))
    walk_y_scale = abs(float(args.walk_y_speed_scale))
    walk_yaw_scale = abs(float(args.walk_yaw_speed_scale))
    walk_yaw_reward_scale = abs(float(args.walk_yaw_reward_scale))
    dribble_x_scale = abs(float(args.dribble_x_speed_scale))
    dribble_y_scale = abs(float(args.dribble_y_speed_scale))
    dribble_yaw_scale = abs(float(args.dribble_yaw_speed_scale))
    shoot_x_scale = abs(float(args.shoot_x_speed_scale))
    shoot_y_scale = abs(float(args.shoot_y_speed_scale))

    num_robots = int(getattr(args, "num_robots", 2))
    if num_robots < 1:
        raise ValueError("--num-robots must be at least 1")
    Cfg.robot.name = "go1"
    Cfg.env.num_envs = args.num_envs
    Cfg.env.num_actions = 12 * num_robots
    Cfg.env.num_observations = 3
    Cfg.env.num_privileged_obs = 3
    Cfg.env.num_observation_history = 15
    Cfg.env.episode_length_s = args.episode_length
    Cfg.env.env_spacing = max(args.field_length, args.field_width) + 2.0
    Cfg.env.add_balls = True
    Cfg.env.num_robots = num_robots
    Cfg.env.control_all_robots = True
    Cfg.env.high_level_control = True
    Cfg.env.high_level_control_interval = args.control_interval
    Cfg.env.high_level_history_length = args.high_level_history
    Cfg.env.high_level_num_observations = 25 * num_robots + 6
    Cfg.env.high_level_num_actions = 6 * num_robots
    Cfg.env.high_level_walk_command_scale = [
        walk_x_scale,
        walk_y_scale,
        walk_yaw_scale,
    ]
    Cfg.env.high_level_dribble_command_scale = [
        dribble_x_scale,
        dribble_y_scale,
        dribble_yaw_scale,
    ]
    Cfg.env.high_level_shoot_command_scale = [
        shoot_x_scale,
        shoot_y_scale,
        0.0,
    ]
    max_cmd_x = max(walk_x_scale, dribble_x_scale, shoot_x_scale, 1e-6)
    max_cmd_y = max(walk_y_scale, dribble_y_scale, shoot_y_scale, 1e-6)
    max_cmd_yaw = max(walk_yaw_scale, dribble_yaw_scale, 1.0)
    Cfg.env.high_level_command_obs_scale = [max_cmd_x, max_cmd_y, max_cmd_yaw]
    Cfg.env.record_video = True
    Cfg.env.num_recording_envs = 1
    Cfg.env.recording_width_px = 640
    Cfg.env.recording_height_px = 640
    Cfg.env.recording_horizontal_fov = 70.0

    Cfg.env.randomize_match_init = True
    Cfg.env.field_length = args.field_length
    Cfg.env.field_width = args.field_width
    Cfg.env.field_margin = 0.35
    Cfg.env.team_goal_x = 0.5 * args.field_length
    Cfg.env.team_goal_half_width = args.goal_half_width
    Cfg.env.high_level_camera_height = 1.6 * max(args.field_length, args.field_width)
    Cfg.env.add_field_markers = True
    Cfg.env.field_marker_width = 0.05
    Cfg.env.field_marker_height = 0.035
    Cfg.env.robot_init_x_range = [-0.5 * args.field_length + 0.6, 0.0]
    Cfg.env.robot_init_y_range = [-0.5 * args.field_width + 0.6, 0.5 * args.field_width - 0.6]
    Cfg.env.robot_yaw_init_range = [-3.14159265, 3.14159265]
    Cfg.env.teammate_init_x_range = [-0.5 * args.field_length + 0.6, 0.0]
    Cfg.env.teammate_init_y_range = [-0.5 * args.field_width + 0.6, 0.5 * args.field_width - 0.6]
    Cfg.env.teammate_yaw_range = [-3.14159265, 3.14159265]
    Cfg.env.ball_init_x_range = [-0.5 * args.field_length + 0.8, 0.5 * args.field_length - 1.2]
    Cfg.env.ball_init_y_range = [-0.5 * args.field_width + 0.8, 0.5 * args.field_width - 0.8]
    Cfg.env.match_init_min_clearance = 0.75
    Cfg.env.num_static_opponents = num_robots
    Cfg.env.static_opponent_size = [0.45, 0.45, 0.50]
    Cfg.env.static_opponent_x_range = [-0.5 * args.field_length + 1.0, 0.5 * args.field_length - 1.0]
    Cfg.env.static_opponent_y_range = [-0.5 * args.field_width + 0.8, 0.5 * args.field_width - 0.8]
    Cfg.env.static_opponent_yaw_range = [-3.14159265, 3.14159265]
    Cfg.env.static_opponent_min_clearance = 0.85

    Cfg.sensors.sensor_names = ["OrientationSensor"]
    Cfg.sensors.sensor_args = {"OrientationSensor": {}}
    Cfg.sensors.privileged_sensor_names = {"BodyVelocitySensor": {}}
    Cfg.sensors.privileged_sensor_args = {"BodyVelocitySensor": {}}

    Cfg.commands.num_commands = 15
    Cfg.commands.distributional_commands = False
    Cfg.commands.heading_command = False
    Cfg.commands.resampling_time = 1_000_000.0
    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.balance_gait_distribution = False
    Cfg.commands.binary_phases = False
    Cfg.commands.gaitwise_curricula = False
    Cfg.commands.lin_vel_x = [-max_cmd_x, max_cmd_x]
    Cfg.commands.lin_vel_y = [-max_cmd_y, max_cmd_y]
    Cfg.commands.ang_vel_yaw = [-max_cmd_yaw, max_cmd_yaw]
    Cfg.commands.body_height_cmd = [0.0, 0.0]
    Cfg.commands.gait_frequency_cmd_range = [3.0, 3.0]
    Cfg.commands.gait_phase_cmd_range = [0.5, 0.5]
    Cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.09, 0.09]
    Cfg.commands.body_pitch_range = [0.0, 0.0]
    Cfg.commands.body_roll_range = [0.0, 0.0]
    Cfg.commands.stance_width_range = [0.05, 0.05]
    Cfg.commands.stance_length_range = [0.05, 0.05]

    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.num_rows = 10
    Cfg.terrain.num_cols = 10
    Cfg.terrain.terrain_length = args.field_length
    Cfg.terrain.terrain_width = args.field_width
    Cfg.terrain.num_border_boxes = 0.0
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.terrain.yaw_init_range = 0.0
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False
    Cfg.terrain.horizontal_scale = 0.05
    Cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.curriculum = False
    Cfg.terrain.max_init_terrain_level = 1

    Cfg.ball.pos_reset_prob = 0.0
    Cfg.ball.vel_reset_prob = 0.0
    Cfg.ball.vision_receive_prob = 1.0
    Cfg.ball.init_pos_range = [0.0, 0.0, 0.0]
    Cfg.ball.init_vel_range = [0.0, 0.0, 0.0]

    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.domain_rand.ground_friction_range = [0.7, 4.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.99, 1.01]
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.002, 0.002]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_ball_drag = True
    Cfg.domain_rand.drag_range = [0.1, 0.8]
    Cfg.domain_rand.ball_drag_rand_interval_s = 15.0
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    for key in list(vars(Cfg.reward_scales).keys()):
        if not key.startswith("_"):
            setattr(Cfg.reward_scales, key, 0.0)
    Cfg.reward_scales.high_level_goal = 400.0
    Cfg.reward_scales.high_level_accidental_termination = -150.0
    Cfg.reward_scales.high_level_ball_goal_progress = 2.0
    Cfg.reward_scales.high_level_possession = 0.5
    Cfg.reward_scales.high_level_robot_spacing = 0.25
    Cfg.reward_scales.high_level_obstacle_clearance = 2.0
    Cfg.reward_scales.high_level_pass = 5.0
    Cfg.reward_scales.high_level_invalid_skill = -3.0
    Cfg.reward_scales.high_level_approach_ball = 1.0

    Cfg.rewards.reward_container_name = "HighLevelRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = False
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.2
    Cfg.rewards.use_terminal_roll_pitch = False
    Cfg.rewards.use_high_level_match_termination = True
    Cfg.rewards.high_level_border_margin = 0.0
    Cfg.rewards.high_level_min_robot_spacing = 0.65
    Cfg.rewards.high_level_target_robot_spacing = 1.5
    Cfg.rewards.high_level_obstacle_safe_distance = 0.55
    Cfg.rewards.high_level_dribble_skill_distance = 1.0
    Cfg.rewards.high_level_shoot_skill_distance = 0.75
    Cfg.rewards.high_level_shoot_alignment = 0.35
    Cfg.rewards.high_level_approach_walk_speed = 0.9
    Cfg.rewards.walking_command_scale = [
        walk_x_scale,
        walk_y_scale,
        max(walk_yaw_reward_scale, walk_yaw_scale, 1e-6),
    ]
    Cfg.rewards.dribbling_command_scale = [
        dribble_x_scale,
        dribble_y_scale,
        max(dribble_yaw_scale, 1e-6),
    ]
    Cfg.rewards.shooting_command_scale = [
        shoot_x_scale,
        shoot_y_scale,
    ]

    Cfg.normalization.clip_actions = 10.0
    Cfg.asset.terminate_after_contacts_on = []


def load_skill_policies(args):
    runs = {
        "walk": args.walk_wandb_run,
        "dribble": args.dribble_wandb_run,
        "shoot": args.shoot_wandb_run,
    }
    for skill, run in runs.items():
        if not run:
            raise ValueError(f"Missing W&B run for {skill}. Pass --{skill}-wandb-run.")
    from scripts.go1_scripts.play_walk_dribble_shoot import load_policy_record, resolve_wandb_policy_files

    policy_types = {
        "walk": "walking",
        "dribble": "ball",
        "shoot": "ball",
    }
    policies = {}
    for skill, run in runs.items():
        body_path, adaptation_module_path, ac_weights_path, run_path = resolve_wandb_policy_files(
            run,
            args.skill_checkpoint,
        )
        policies[skill] = load_policy_record(
            skill,
            body_path,
            adaptation_module_path,
            ac_weights_path,
            policy_types[skill],
            args.policy_device,
            f"W&B {run_path}@{args.skill_checkpoint}",
        )
    return policies


def train_robot(args):
    import isaacgym
    assert isaacgym
    import wandb

    from dribblebot.envs.base.legged_robot_config import Cfg
    from dribblebot.envs.go1.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    from dribblebot.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from dribblebot_learn.ppo_cse import Runner, RunnerArgs
    from dribblebot_learn.ppo_cse.actor_critic import AC_Args
    from dribblebot_learn.ppo_cse.ppo import PPO_Args

    configure_high_level_cfg(Cfg, args)
    skill_policies = load_skill_policies(args)

    RunnerArgs.resume = False
    RunnerArgs.save_video_interval = 500
    AC_Args.adaptation_labels = []
    AC_Args.adaptation_dims = []

    wandb.init(
        project=args.project or "high_level_soccer",
        config={
            "AC_Args": vars(AC_Args),
            "PPO_Args": vars(PPO_Args),
            "RunnerArgs": vars(RunnerArgs),
            "Cfg": vars(Cfg),
            "skill_runs": {
                "walk": args.walk_wandb_run,
                "dribble": args.dribble_wandb_run,
                "shoot": args.shoot_wandb_run,
                "checkpoint": args.skill_checkpoint,
            },
        },
    )

    raw_env = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=args.headless, cfg=Cfg)
    env = HighLevelSkillWrapper(raw_env, skill_policies)
    runner = Runner(env, device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True, eval_freq=100)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-robot high-level soccer coordinator.")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--project", default=None)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-robots", type=int, default=2, help="Controlled robot count; creates the same number of static obstacles.")
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--episode-length", type=float, default=30.0)
    parser.add_argument("--control-interval", type=int, default=10)
    parser.add_argument("--high-level-history", type=int, default=4)
    parser.add_argument("--field-length", type=float, default=8.0)
    parser.add_argument("--field-width", type=float, default=5.0)
    parser.add_argument("--goal-half-width", type=float, default=1.0)
    parser.add_argument("--walk-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--walk-yaw-reward-scale", type=float, default=1.0)
    parser.add_argument("--dribble-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--shoot-x-speed-scale", type=float, default=3.0)
    parser.add_argument("--shoot-y-speed-scale", type=float, default=3.0)
    parser.add_argument("--skill-checkpoint", default="latest")
    parser.add_argument("--walk-wandb-run", default=DEFAULT_GO1_SKILL_RUNS["walk"])
    parser.add_argument("--dribble-wandb-run", default=DEFAULT_GO1_SKILL_RUNS["dribble"])
    parser.add_argument("--shoot-wandb-run", default=DEFAULT_GO1_SKILL_RUNS["shoot"])
    return parser.parse_args()


if __name__ == "__main__":
    train_robot(parse_args())
