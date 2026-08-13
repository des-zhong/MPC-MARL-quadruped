"""AS2 shooting training entry point."""

import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train AS2 shooting with normalized command-scale rewards.")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--shoot-x-speed-scale", type=float, default=3.0)
    parser.add_argument("--shoot-y-speed-scale", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--schedule", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument(
        "--action-clip",
        type=float,
        default=1.0,
        help="Symmetric actuator-action limit used by the environment.",
    )
    parser.add_argument(
        "--action-mean-bound",
        type=float,
        default=0.75,
        help="Tanh bound for the Gaussian mean, leaving room for exploration.",
    )
    parser.add_argument("--init-noise-std", type=float, default=0.20)
    parser.add_argument("--max-noise-std", type=float, default=0.25)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--reset-longitudinal-min", type=float, default=0.35)
    parser.add_argument("--reset-longitudinal-max", type=float, default=0.60)
    parser.add_argument("--reset-lateral-range", type=float, default=0.20)
    parser.add_argument("--reset-yaw-error-range", type=float, default=0.25)
    parser.add_argument(
        "--randomize-ball-physics",
        action="store_true",
        help="Enable ball drag and ground-friction randomization after the base skill converges.",
    )
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default="tmp/legged_data/shoot",
        help="Run-local checkpoint output directory.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-run", default=None)
    parser.add_argument(
        "--resume-checkpoint",
        default="tmp/legged_data/shoot/ac_weights_latest.pt",
    )
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def symmetric_range(scale):
    scale = abs(float(scale))
    return [-scale, scale]


def train_robot(args=None, headless=True):
    if args is None:
        args = build_arg_parser().parse_args([])
    headless = bool(getattr(args, "headless", headless))

    import isaacgym
    assert isaacgym
    import torch

    from dribblebot.envs.base.legged_robot_config import Cfg
    from dribblebot.envs.as2.as2_config import config_as2
    from dribblebot.envs.as2.velocity_tracking import VelocityTrackingEasyEnv

    from dribblebot_learn.ppo_cse import Runner
    from dribblebot.envs.wrappers.history_wrapper import HistoryWrapper
    from dribblebot_learn.ppo_cse.actor_critic import AC_Args
    from dribblebot_learn.ppo_cse.ppo import PPO_Args
    from dribblebot_learn.ppo_cse import RunnerArgs

    config_as2(Cfg)
    Cfg.env.num_envs = args.num_envs

    if args.action_clip <= 0.0:
        raise ValueError("--action-clip must be positive")
    if args.action_mean_bound <= 0.0:
        raise ValueError("--action-mean-bound must be positive")
    if args.init_noise_std <= 0.0:
        raise ValueError("--init-noise-std must be positive")
    if args.max_noise_std < args.init_noise_std:
        raise ValueError("--max-noise-std must be at least --init-noise-std")
    if args.entropy_coef < 0.0:
        raise ValueError("--entropy-coef must be non-negative")
    if args.desired_kl <= 0.0:
        raise ValueError("--desired-kl must be positive")
    if args.reset_longitudinal_min <= 0.0:
        raise ValueError("--reset-longitudinal-min must be positive")
    if args.reset_longitudinal_max < args.reset_longitudinal_min:
        raise ValueError("--reset-longitudinal-max must be at least --reset-longitudinal-min")
    if args.reset_lateral_range < 0.0:
        raise ValueError("--reset-lateral-range must be non-negative")
    if args.reset_yaw_error_range < 0.0:
        raise ValueError("--reset-yaw-error-range must be non-negative")

    if args.resume and not args.resume_run:
        raise ValueError("--resume requires --resume-run for a compatible shooting checkpoint.")
    RunnerArgs.resume = bool(args.resume)
    RunnerArgs.resume_path = args.resume_run
    RunnerArgs.resume_checkpoint = args.resume_checkpoint
    RunnerArgs.checkpoint_dir = args.checkpoint_dir
    PPO_Args.learning_rate = args.learning_rate
    PPO_Args.adaptation_module_learning_rate = args.learning_rate
    PPO_Args.schedule = args.schedule
    PPO_Args.entropy_coef = args.entropy_coef
    PPO_Args.desired_kl = args.desired_kl

    Cfg.robot.name = "as2"
    Cfg.sensors.sensor_names = [
        "ObjectSensor",
        "OrientationSensor",
        "RCSensor",
        "JointPositionSensor",
        "JointVelocitySensor",
        "ActionSensor",
        "LastActionSensor",
        "ClockSensor",
        "YawSensor",
        "TimingSensor",
    ]
    Cfg.sensors.sensor_args = {
        "ObjectSensor": {},
        "OrientationSensor": {},
        "RCSensor": {},
        "JointPositionSensor": {},
        "JointVelocitySensor": {},
        "ActionSensor": {},
        "LastActionSensor": {"delay": 1},
        "ClockSensor": {},
        "YawSensor": {},
        "TimingSensor": {},
    }
    Cfg.sensors.privileged_sensor_names = {
        "BodyVelocitySensor": {},
        "ObjectVelocitySensor": {},
    }
    Cfg.sensors.privileged_sensor_args = {
        "BodyVelocitySensor": {},
        "ObjectVelocitySensor": {},
    }

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90
    Cfg.curriculum_thresholds.dribbling_ball_vel = 0.8
    Cfg.commands.distributional_commands = True

    Cfg.domain_rand.lag_timesteps = 0
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "P"

    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_friction = False # True
    Cfg.domain_rand.randomize_restitution = False # True
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.domain_rand.randomize_ground_friction = args.randomize_ball_physics
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.99, 1.01]
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.002, 0.002]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = True
    Cfg.domain_rand.randomize_Kd_factor = True
    Cfg.domain_rand.randomize_ball_drag = args.randomize_ball_physics
    Cfg.domain_rand.drag_range = [0.1, 0.8]
    Cfg.domain_rand.ball_drag_rand_interval_s = 15.0

    Cfg.env.num_observation_history = 15
    Cfg.reward_scales.feet_contact_forces = 0.0
    Cfg.env.num_envs = args.num_envs

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.balance_gait_distribution = False
    Cfg.commands.binary_phases = False
    Cfg.commands.gaitwise_curricula = False

    ###############################
    # soccer shooting configuration
    ###############################

    Cfg.env.add_balls = True
    # Sample attainable shooting setups in the desired kick frame. This
    # requires the robot to translate around the ball instead of solving a
    # fully random pose, moving-ball, and kick-direction problem at once.
    Cfg.env.shooting_reset_relative_to_command = True
    Cfg.env.shooting_reset_longitudinal_range = [
        args.reset_longitudinal_min,
        args.reset_longitudinal_max,
    ]
    Cfg.env.shooting_reset_lateral_range = [
        -args.reset_lateral_range,
        args.reset_lateral_range,
    ]
    Cfg.env.shooting_reset_yaw_error_range = [
        -args.reset_yaw_error_range,
        args.reset_yaw_error_range,
    ]
    Cfg.env.shooting_reset_zero_velocities = True
    Cfg.ball.init_pos_range = [0.0, 0.0, 0.0]
    Cfg.ball.init_vel_range = [0.0, 0.0, 0.0]
    Cfg.ball.pos_reset_prob = 0.0
    Cfg.ball.vel_reset_prob = 0.0
    Cfg.ball.vision_receive_prob = 1.0

    # domain randomization ranges
    Cfg.domain_rand.rand_interval_s = 6
    Cfg.domain_rand.friction_range = [0.0, 1.5]
    Cfg.domain_rand.randomize_ground_friction = args.randomize_ball_physics
    Cfg.domain_rand.ground_friction_range = [0.7, 4.0]
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.motor_strength_range = [0.99, 1.01]
    Cfg.domain_rand.motor_offset_range = [-0.002, 0.002]
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.0]

    # privileged obs in use
    Cfg.env.num_privileged_obs = 6
    Cfg.env.priv_observe_ball_drag = True

    # sensory observation
    Cfg.commands.num_commands = 15
    Cfg.env.episode_length_s = 10.
    Cfg.env.num_observations = 75

    # terrain configuration
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "boxes_tm"
    Cfg.terrain.num_cols = 20
    Cfg.terrain.num_rows = 20
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.num_border_boxes = 5.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False
    Cfg.terrain.center_span = 3
    Cfg.terrain.horizontal_scale = 0.05
    Cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.curriculum = False
    Cfg.terrain.difficulty_scale = 1.0
    Cfg.terrain.max_step_height = 0.26
    Cfg.terrain.min_step_run = 0.25
    Cfg.terrain.max_step_run = 0.4
    Cfg.terrain.max_init_terrain_level = 1

    # terminal conditions
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.2
    Cfg.rewards.use_terminal_roll_pitch = False
    Cfg.rewards.terminal_body_ori = 0.5

    # command sampling
    # commands[:, :2] = desired post-kick ball xy velocity.
    # commands[:, 2] is unused for shooting and is kept fixed.
    Cfg.commands.resampling_time = 10
    Cfg.commands.heading_command = False

    Cfg.commands.lin_vel_x = symmetric_range(args.shoot_x_speed_scale)
    Cfg.commands.lin_vel_y = symmetric_range(args.shoot_y_speed_scale)
    Cfg.commands.ang_vel_yaw = [-0.0, 0.0]
    Cfg.commands.body_height_cmd = [-0.05, 0.05]
    Cfg.commands.gait_frequency_cmd_range = [3.0, 3.0]
    Cfg.commands.gait_phase_cmd_range = [0.5, 0.5]
    Cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.09, 0.09]
    Cfg.commands.body_pitch_range = [-0.0, 0.0]
    Cfg.commands.body_roll_range = [-0.0, 0.0]
    Cfg.commands.stance_width_range = [0.0, 0.1]
    Cfg.commands.stance_length_range = [0.0, 0.1]

    Cfg.commands.limit_vel_x = symmetric_range(args.shoot_x_speed_scale)
    Cfg.commands.limit_vel_y = symmetric_range(args.shoot_y_speed_scale)
    Cfg.commands.limit_vel_yaw = [-0.0, 0.0]
    Cfg.commands.limit_body_height = [-0.05, 0.05]
    Cfg.commands.limit_gait_frequency = [3.0, 3.0]
    Cfg.commands.limit_gait_phase = [0.5, 0.5]
    Cfg.commands.limit_gait_offset = [0.0, 0.0]
    Cfg.commands.limit_gait_bound = [0.0, 0.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.09, 0.09]
    Cfg.commands.limit_body_pitch = [-0.0, 0.0]
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.0, 0.1]
    Cfg.commands.limit_stance_length = [0.0, 0.1]

    Cfg.commands.num_bins_vel_x = 1
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 1
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.rewards.constrict = False
    Cfg.rewards.use_shooting_phase_termination = True
    Cfg.rewards.shooting_min_command_speed = 0.3
    Cfg.rewards.shooting_setup_distance = 0.45
    Cfg.rewards.shooting_setup_position_gain = 6.0
    Cfg.rewards.shooting_setup_progress_speed = 1.0
    Cfg.rewards.shooting_free_yaw_rate = 0.75
    Cfg.rewards.shooting_yaw_rate_scale = 2.0
    Cfg.rewards.shooting_reward_min_separation = 0.55
    # Admit partial, correctly directed strikes to the post-kick phase. The
    # ungated forward-velocity term below supplies signal before this threshold.
    Cfg.rewards.shooting_launch_speed_fraction = 0.20
    Cfg.rewards.shooting_launch_alignment = 0.50
    Cfg.rewards.shooting_success_distance = 0.7
    Cfg.rewards.shooting_success_speed_fraction = 0.25
    Cfg.rewards.shooting_success_alignment = 0.65
    Cfg.rewards.shooting_max_attempt_time_s = 5.0
    Cfg.rewards.shooting_command_scale = [
        abs(float(args.shoot_x_speed_scale)),
        abs(float(args.shoot_y_speed_scale)),
    ]

    Cfg.reward_scales.orientation = -5.0
    Cfg.reward_scales.torques = -3.0e-5
    Cfg.reward_scales.dof_vel = -0.0001
    Cfg.reward_scales.dof_acc = -2.5e-7
    Cfg.reward_scales.collision = -5.0
    Cfg.reward_scales.action_rate = -0.01
    Cfg.reward_scales.tracking_contacts_shaped_force = 4.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    Cfg.reward_scales.dof_pos_limits = -10.0
    Cfg.reward_scales.dof_pos = -0.05
    Cfg.reward_scales.action_smoothness_1 = -0.1
    Cfg.reward_scales.action_smoothness_2 = -0.1

    Cfg.reward_scales.shooting_ball_vel = 5.0
    # Adapted from the reference kick task: reward the first useful contact
    # instead of waiting for both launch and robot-ball separation.
    Cfg.reward_scales.shooting_ball_forward = 8.0
    Cfg.reward_scales.shooting_ball_vel_norm = 1.5
    Cfg.reward_scales.shooting_ball_vel_angle = 1.5
    Cfg.reward_scales.shooting_ball_out = 7.0
    # The FR-hip proximity term can be improved by rotating in place, so the
    # setup-pose and signed-progress terms replace it.
    Cfg.reward_scales.shooting_robot_ball_pos = 0.0
    Cfg.reward_scales.shooting_robot_ball_behind = 0.5
    Cfg.reward_scales.shooting_robot_forward_cmd = 0.5
    Cfg.reward_scales.shooting_ball_in_front = 0.25
    Cfg.reward_scales.shooting_robot_approach_ball = 1.5
    Cfg.reward_scales.shooting_excess_yaw = -0.1
    Cfg.reward_scales.shooting_launch = 50.0
    Cfg.reward_scales.shooting_success = 200.0
    # Before kick discovery a large timeout penalty is nearly constant and
    # swamps the dense contact signal. Success remains strongly preferred.
    Cfg.reward_scales.shooting_failure = -25.0

    Cfg.reward_scales.tracking_lin_vel = 0.0
    Cfg.reward_scales.tracking_ang_vel = 0.0
    Cfg.reward_scales.lin_vel_z = 0.0
    Cfg.reward_scales.ang_vel_xy = 0.0
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.dribbling_robot_ball_vel = 0.0
    Cfg.reward_scales.dribbling_robot_ball_pos = 0.0
    Cfg.reward_scales.dribbling_ball_vel = 0.0
    Cfg.reward_scales.dribbling_robot_ball_yaw = 0.0
    Cfg.reward_scales.dribbling_ball_vel_norm = 0.0
    Cfg.reward_scales.dribbling_ball_vel_angle = 0.0

    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.0
    Cfg.rewards.gait_vel_sigma = 10.0
    Cfg.rewards.reward_container_name = "SoccerRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = False
    Cfg.rewards.sigma_rew_neg = 0.02

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0.7, 4.0]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    Cfg.reward_scales.feet_slip = -0.0
    Cfg.reward_scales.jump = 0.0
    Cfg.reward_scales.base_height = -0.0
    Cfg.reward_scales.feet_impact_vel = -0.0
    Cfg.reward_scales.feet_air_time = 0.0

    Cfg.asset.terminate_after_contacts_on = []

    Cfg.control.control_type = "P"
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.normalization.clip_actions = args.action_clip
    Cfg.asset.terminate_after_contacts_on = ["base_link"]
    Cfg.rewards.use_terminal_roll_pitch = True
    # PPO scores the raw Gaussian action while the simulator executes the
    # clipped action. Keep both the mean and exploration noise comfortably
    # inside the actuator range so those two actions rarely disagree.
    AC_Args.init_noise_std = args.init_noise_std
    AC_Args.max_action_std = args.max_noise_std
    AC_Args.action_mean_bound = min(args.action_mean_bound, args.action_clip)

    AC_Args.adaptation_labels = []
    AC_Args.adaptation_dims = []

    RunnerArgs.save_video_interval = 500

    import wandb
    wandb.init(
        project=args.project or "as2_shooting",
        config={
            "AC_Args": vars(AC_Args),
            "PPO_Args": vars(PPO_Args),
            "RunnerArgs": vars(RunnerArgs),
            "Cfg": vars(Cfg),
        }
    )

    device = args.device
    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg)

    env = HistoryWrapper(env)
    runner = Runner(env, device=device)
    if args.resume:
        # Existing checkpoints can contain the saturated exploration std that
        # caused the reward collapse. Preserve learned mean weights but do not
        # restore that unstable distribution.
        with torch.no_grad():
            runner.alg.actor_critic.std.fill_(args.init_noise_std)
        print(f"Reset resumed shooting policy action std to {args.init_noise_std}")
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False, eval_freq=100)


if __name__ == "__main__":
    train_robot(parse_args())
