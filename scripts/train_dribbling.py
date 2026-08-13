"""AS2 dribbling training entry point."""

import argparse


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train AS2 dribbling with normalized command-scale rewards.")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--ball-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--ball-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
        default=1.0,
        help="Tanh bound applied to the Gaussian policy mean.",
    )
    parser.add_argument("--init-noise-std", type=float, default=0.3)
    parser.add_argument("--max-noise-std", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default="tmp/legged_data/dribble",
        help="Run-local checkpoint output directory.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-run",
        default=None,
        help=(
            "Optional W&B source run in entity/project/run_id form. Omit this "
            "when --resume-checkpoint is an existing local file."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="wandb/as2_dribble-ihrruh16/files/tmp/legged_data/dribble/ac_weights_latest.pt",
        help=(
            "Local checkpoint path when --resume-run is omitted, or the "
            "artifact-relative filename to restore from the W&B source run."
        ),
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

    RunnerArgs.resume = bool(args.resume)
    RunnerArgs.resume_path = args.resume_run
    RunnerArgs.resume_checkpoint = args.resume_checkpoint
    RunnerArgs.checkpoint_dir = args.checkpoint_dir
    PPO_Args.learning_rate = args.learning_rate
    PPO_Args.adaptation_module_learning_rate = args.learning_rate
    PPO_Args.schedule = args.schedule
    PPO_Args.entropy_coef = args.entropy_coef


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
                        "TimingSensor":{},
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
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.99, 1.01]
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.002, 0.002]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = True
    Cfg.domain_rand.randomize_Kd_factor = True
    Cfg.domain_rand.randomize_ball_drag = True
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
    # soccer dribbling configuration
    ###############################

    # ball parameters
    Cfg.env.add_balls = True

    # domain randomization ranges
    Cfg.domain_rand.rand_interval_s = 6
    Cfg.domain_rand.friction_range = [0.0, 1.5]
    Cfg.domain_rand.randomize_ground_friction = True
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
    Cfg.env.episode_length_s = 40.
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
    # commands[:, :2] = desired dribbled-ball xy velocity.
    # commands[:, 2] = desired robot yaw velocity.
    Cfg.commands.resampling_time = 7
    Cfg.commands.heading_command = False

    Cfg.commands.lin_vel_x = symmetric_range(args.ball_x_speed_scale)
    Cfg.commands.lin_vel_y = symmetric_range(args.ball_y_speed_scale)
    Cfg.commands.ang_vel_yaw = symmetric_range(args.yaw_speed_scale)
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

    Cfg.commands.limit_vel_x = symmetric_range(args.ball_x_speed_scale)
    Cfg.commands.limit_vel_y = symmetric_range(args.ball_y_speed_scale)
    Cfg.commands.limit_vel_yaw = symmetric_range(args.yaw_speed_scale)
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
    Cfg.commands.num_bins_vel_yaw = Cfg.commands.num_ang_vel_bins
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

    # reward function
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

    Cfg.reward_scales.dribbling_robot_ball_vel = 0.5
    Cfg.reward_scales.dribbling_robot_ball_pos = 4.0
    Cfg.reward_scales.dribbling_backward_motion = 2.0
    Cfg.reward_scales.dribbling_ball_vel = 4.0
    Cfg.reward_scales.dribbling_robot_ball_yaw = 4.0
    Cfg.reward_scales.dribbling_ball_vel_norm = 4.0
    Cfg.reward_scales.dribbling_ball_vel_angle = 4.0

    Cfg.reward_scales.tracking_lin_vel = 0.0
    Cfg.reward_scales.tracking_ang_vel = 1.0
    Cfg.reward_scales.lin_vel_z = 0.0
    Cfg.reward_scales.ang_vel_xy = 0.0
    Cfg.reward_scales.feet_air_time = 0.0

    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.
    Cfg.rewards.dribbling_command_scale = [
        abs(float(args.ball_x_speed_scale)),
        abs(float(args.ball_y_speed_scale)),
        max(abs(float(args.yaw_speed_scale)), 1e-6),
    ]
    Cfg.rewards.dribbling_target_forward = 0.35
    Cfg.rewards.dribbling_target_lateral = 0.0
    Cfg.rewards.dribbling_position_gain = 10.0
    Cfg.rewards.dribbling_forward_speed_scale = 1.0
    
    Cfg.rewards.reward_container_name = "SoccerRewards"
    Cfg.rewards.only_positive_rewards = False
    # Ji-style composition exponentially suppresses every positive task reward
    # once action-rate/smoothness penalties grow.  That made a saturated policy
    # lose its recovery signal, so dribbling uses the ordinary signed sum.
    Cfg.rewards.only_positive_rewards_ji22_style = False
    Cfg.rewards.sigma_rew_neg = 0.02

    # normalization
    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0.7, 4.0]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = args.action_clip

    # reward function (not in use)
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

    # Keep the policy distribution inside the same range the simulator
    # executes.  Without this bound PPO scores raw Gaussian samples while the
    # environment clips them, allowing the mean and standard deviation to
    # drift into a many-to-one saturated region.
    AC_Args.init_noise_std = args.init_noise_std
    AC_Args.max_action_std = args.max_noise_std
    AC_Args.action_mean_bound = min(args.action_mean_bound, args.action_clip)
    AC_Args.adaptation_labels = []
    AC_Args.adaptation_dims = []

    RunnerArgs.save_video_interval = 500

    import wandb
    wandb.init(
      # set the wandb project where this run will be logged
      project=args.project or "as2_dribbling",

      # track hyperparameters and run metadata
      config={
      "AC_Args": vars(AC_Args),
      "PPO_Args": vars(PPO_Args),
      "RunnerArgs": vars(RunnerArgs),
      "Cfg": vars(Cfg),
      }
    )

    device = args.device
    # device = 'cpu'
    env = VelocityTrackingEasyEnv(sim_device=device, headless=headless, cfg=Cfg)

    env = HistoryWrapper(env)
    runner = Runner(env, device=device)
    if args.resume:
        # The old checkpoint may contain exploration standard deviations near
        # the hard maximum.  The mean weights remain useful behind the new
        # parameter-free bound, but stale noise would immediately recreate the
        # saturation failure.
        with torch.no_grad():
            runner.alg.actor_critic.std.fill_(args.init_noise_std)
        print(f"Reset resumed policy action std to {args.init_noise_std}")
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    train_robot(parse_args())
