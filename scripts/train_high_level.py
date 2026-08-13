"""AS2 high-level soccer training entry point."""

import argparse
import tempfile
from pathlib import Path
from typing import Mapping


DEFAULT_AS2_SKILL_RUNS = {
    "walk": "des_zhong/as2_walking/3a6g1def",
    "dribble": "des_zhong/as2_dribbling/cp9m21ay",
    "shoot": "des_zhong/as2_shooting/bve3isir",
}

SKILL_NAMES = ("walk", "dribble", "shoot")

HIGH_LEVEL_REWARD_SCALES = {
    # Reward scales are multiplied by the raw environment dt (0.02 s).  The
    # dense terms deliberately cover every transition in the skill sequence so
    # the coordinator cannot maximize return by collapsing to walk forever:
    # approach -> controlled dribble -> aligned launch -> goal.
    "high_level_goal": 500.0,
    "high_level_accidental_termination": -200.0,
    "high_level_ball_goal_progress": 2.0,
    # Smooth body-clearance cost. At complete base overlap this contributes
    # -2 reward over a default 10-step high-level interval.
    "high_level_robot_collision": -10.0,
    "high_level_pass": 2.0,
    "high_level_invalid_skill": -3.0,
    "high_level_approach_ball": 1.0,
    "high_level_walk_command_alignment": 0.5,
    "high_level_face_ball_while_approaching": 0.5,
    "high_level_face_goal_while_moving": 0.75,
    "high_level_dribble_ball_control": 2.0,
    # A launch is a short event rather than a reward emitted throughout the
    # control interval, so it needs a larger nominal coefficient than the
    # continuously evaluated approach and dribble terms.
    "high_level_shoot_launch": 10.0,
}


def _wandb_config_value(value):
    """Unwrap W&B's ``{desc, value}`` representation when present."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def high_level_checkpoint_contract(policy_record):
    """Read the environment contract saved alongside a coordinator policy."""

    metadata = policy_record.get("policy_metadata", {})
    config_path = metadata.get("config_path")
    if not config_path:
        return None
    from dribblebot.world_model.config import load_config

    payload = load_config(config_path)
    cfg = _wandb_config_value(payload.get("Cfg", {}))
    env = _wandb_config_value(cfg.get("env", {})) if isinstance(cfg, Mapping) else {}
    self_play = _wandb_config_value(payload.get("self_play", {}))
    if not isinstance(env, Mapping):
        env = {}
    if not isinstance(self_play, Mapping):
        self_play = {}
    team_size = self_play.get("team_size", env.get("num_team_robots"))
    return {
        "config_path": str(config_path),
        "team_size": team_size,
        "control_interval": env.get("high_level_control_interval"),
        "history_length": env.get("high_level_history_length"),
        "walk_scale": env.get("high_level_walk_command_scale"),
        "dribble_scale": env.get("high_level_dribble_command_scale"),
        "shoot_scale": env.get("high_level_shoot_command_scale"),
        "geometric_fallback": env.get("high_level_use_geometric_skill_fallback"),
        "near_ball_probability": env.get("high_level_near_ball_init_probability"),
    }


def validate_high_level_evaluation_contract(policy_record, args):
    """Reject silent train/evaluation changes that alter policy semantics."""

    contract = high_level_checkpoint_contract(policy_record)
    if contract is None:
        return
    actual = {
        "team_size": int(args.num_robots),
        "control_interval": int(args.control_interval),
        "history_length": int(args.high_level_history),
        "walk_scale": [
            abs(float(args.walk_x_speed_scale)),
            abs(float(args.walk_y_speed_scale)),
            abs(float(args.walk_yaw_speed_scale)),
        ],
        "dribble_scale": [
            abs(float(args.dribble_x_speed_scale)),
            abs(float(args.dribble_y_speed_scale)),
            abs(float(args.dribble_yaw_speed_scale)),
        ],
        "shoot_scale": [
            abs(float(args.shoot_x_speed_scale)),
            abs(float(args.shoot_y_speed_scale)),
            0.0,
        ],
        "geometric_fallback": bool(args.use_geometric_skill_fallback),
        "near_ball_probability": float(args.near_ball_init_probability),
    }
    mismatches = []
    for name, actual_value in actual.items():
        expected = contract.get(name)
        if expected is None:
            continue
        if isinstance(expected, (list, tuple)):
            matches = len(expected) == len(actual_value) and all(
                abs(float(left) - float(right)) <= 1e-6
                for left, right in zip(expected, actual_value)
            )
        elif isinstance(expected, bool):
            matches = bool(actual_value) == expected
        elif isinstance(expected, (int, float)):
            matches = abs(float(actual_value) - float(expected)) <= 1e-6
        else:
            matches = actual_value == expected
        if not matches:
            mismatches.append(f"{name}: checkpoint={expected!r}, evaluation={actual_value!r}")
    if not mismatches:
        return
    message = (
        f"High-level evaluation does not match {contract['config_path']}:\n- "
        + "\n- ".join(mismatches)
    )
    if getattr(args, "allow_training_config_mismatch", False):
        import warnings

        warnings.warn(message)
        return
    raise ValueError(
        message
        + "\nUse matching settings, or pass --allow-training-config-mismatch "
        "for an intentional out-of-distribution evaluation."
    )


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "t", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def newest_complete_local_checkpoint(policy_dir):
    """Return the newest immutable generation available under a policy dir.

    The low-level runner writes actor weights, adaptation module, and body in
    that order.  A numbered body therefore acts as the completion marker for
    the same numbered generation, unlike the three mutable ``*_latest`` files
    which can be observed halfway through a concurrent save.
    """

    root = Path(policy_dir)
    complete = []
    for body in root.rglob("body_*.jit"):
        suffix = body.stem[len("body_") :]
        if not suffix.isdigit():
            continue
        directory = body.parent
        if (
            (directory / f"adaptation_module_{suffix}.jit").is_file()
            and (directory / f"ac_weights_{suffix}.pt").is_file()
        ):
            complete.append(int(suffix))
    return str(max(complete)) if complete else None


def find_run_level_config(policy_dir):
    """Prefer W&B's run-level config over a mutable checkpoint symlink."""

    path = Path(policy_dir)
    for candidate_root in (path, *path.parents):
        if candidate_root.name == "files":
            candidate = candidate_root / "config.yaml"
            if candidate.is_file():
                return candidate.resolve()
    return None


def validate_high_level_training_args(args):
    """Fail before simulator startup on unstable or inconsistent PPO settings."""

    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.desired_kl <= 0.0:
        raise ValueError("--desired-kl must be positive")
    if args.init_noise_std <= 0.0:
        raise ValueError("--init-noise-std must be positive")
    if args.max_noise_std < args.init_noise_std:
        raise ValueError("--max-noise-std must be at least --init-noise-std")
    if args.entropy_coef < 0.0:
        raise ValueError("--entropy-coef must be non-negative")
    if args.skill_entropy_coef < 0.0:
        raise ValueError("--skill-entropy-coef must be non-negative")
    if args.ppo_epochs < 1:
        raise ValueError("--ppo-epochs must be at least 1")
    if args.max_kl_factor <= 1.0:
        raise ValueError("--max-kl-factor must be greater than 1")
    if args.action_mean_bound <= 0.0:
        raise ValueError("--action-mean-bound must be positive")
    if args.max_skill_action_clip <= 0.0:
        raise ValueError("--max-skill-action-clip must be positive")
    if args.self_play_update_interval < 1:
        raise ValueError("--self-play-update-interval must be at least 1")
    if args.resume and not args.resume_checkpoint:
        raise ValueError("--resume requires --resume-checkpoint")
    if args.resume and not args.resume_run:
        checkpoint = Path(args.resume_checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Local high-level resume checkpoint does not exist: {checkpoint.resolve()}"
            )
    teacher_checkpoint = getattr(args, "world_model_checkpoint", None)
    if teacher_checkpoint and not Path(teacher_checkpoint).expanduser().is_file():
        raise FileNotFoundError(
            f"World-model teacher checkpoint does not exist: {teacher_checkpoint}"
        )
    if getattr(args, "teacher_reward_coefficient", 0.0) < 0.0:
        raise ValueError("--teacher-reward-coefficient must be non-negative")


def add_skill_policy_source_args(parser):
    """Add explicit low-level checkpoint source options to an argument parser."""

    parser.add_argument(
        "--skill-policy-source",
        choices=("wandb", "local"),
        default="wandb",
        help=(
            "Load every low-level skill from online W&B (downloaded to a process-local "
            "temporary directory) or directly from the three --*-policy-dir folders."
        ),
    )
    parser.add_argument(
        "--walk-policy-dir",
        default=None,
        help="Local walking checkpoint directory, for example ./wandb/<run>/files/tmp/legged_data.",
    )
    parser.add_argument(
        "--dribble-policy-dir",
        default=None,
        help="Local dribbling checkpoint directory, for example ./wandb/<run>/files/tmp/legged_data.",
    )
    parser.add_argument(
        "--shoot-policy-dir",
        default=None,
        help="Local shooting checkpoint directory, for example ./wandb/<run>/files/tmp/legged_data.",
    )
    return parser


def configure_high_level_cfg(Cfg, args):
    from dribblebot.envs.as2.as2_config import config_as2

    config_as2(Cfg)
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
    Cfg.robot.name = "as2"
    Cfg.env.num_envs = args.num_envs
    # Competitive training instantiates two equal teams. Offline world-model
    # collection opts out explicitly until its multi-team schema is decided.
    self_play_enabled = bool(getattr(args, "self_play", True))
    physical_robots = 2 * num_robots if self_play_enabled else num_robots
    Cfg.env.num_actions = 12 * physical_robots
    Cfg.env.num_observations = 3
    Cfg.env.num_privileged_obs = 3
    Cfg.env.num_observation_history = 15
    Cfg.env.episode_length_s = args.episode_length
    Cfg.env.env_spacing = max(args.field_length, args.field_width) + 2.0
    Cfg.env.add_balls = True
    Cfg.env.num_robots = physical_robots
    Cfg.env.num_team_robots = num_robots
    Cfg.env.opponent_team_color = [0.85, 0.10, 0.10]
    Cfg.env.control_all_robots = True
    Cfg.env.high_level_control = True
    Cfg.env.high_level_control_interval = args.control_interval
    Cfg.env.high_level_history_length = args.high_level_history
    Cfg.env.high_level_num_observations = 25 * physical_robots + 6
    Cfg.env.high_level_num_actions = 6 * physical_robots
    # High-level logits/raw command inputs and low-level actuator actions use
    # different numeric ranges.  Do not reuse normalization.clip_actions for
    # both: the latter is set from the loaded skill checkpoints below.
    Cfg.env.high_level_action_input_clip = 10.0
    Cfg.env.high_level_use_geometric_skill_fallback = bool(
        getattr(args, "use_geometric_skill_fallback", False)
    )
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
    Cfg.env.high_level_camera_height = 1.2 * max(args.field_length, args.field_width)
    Cfg.env.add_field_markers = True
    Cfg.env.field_marker_width = 0.05
    Cfg.env.field_marker_height = 0.035
    Cfg.env.robot_init_x_range = [-0.5 * args.field_length + 0.6, 0.0]
    Cfg.env.robot_init_y_range = [-0.5 * args.field_width + 0.6, 0.5 * args.field_width - 0.6]
    Cfg.env.robot_yaw_init_range = [-3.14159265, 3.14159265]
    Cfg.env.teammate_init_x_range = [-0.5 * args.field_length + 0.6, 0.0]
    Cfg.env.teammate_init_y_range = [-0.5 * args.field_width + 0.6, 0.5 * args.field_width - 0.6]
    Cfg.env.teammate_yaw_range = [-3.14159265, 3.14159265]
    Cfg.env.opponent_init_x_range = [0.0, 0.5 * args.field_length - 0.6]
    Cfg.env.opponent_init_y_range = [-0.5 * args.field_width + 0.6, 0.5 * args.field_width - 0.6]
    Cfg.env.opponent_yaw_range = [-3.14159265, 3.14159265]
    Cfg.env.ball_init_x_range = [-0.5 * args.field_length + 0.8, 0.5 * args.field_length - 1.2]
    Cfg.env.ball_init_y_range = [-0.5 * args.field_width + 0.8, 0.5 * args.field_width - 0.8]
    Cfg.env.match_init_min_clearance = 0.75
    near_ball_probability = float(getattr(args, "near_ball_init_probability", 0.4))
    near_ball_min_distance = float(getattr(args, "near_ball_init_min_distance", 0.4))
    near_ball_max_distance = float(getattr(args, "near_ball_init_max_distance", 0.95))
    if not 0.0 <= near_ball_probability <= 1.0:
        raise ValueError(
            "--near-ball-init-probability must be between 0 and 1, "
            f"got {near_ball_probability}."
        )
    if near_ball_min_distance <= 0.0 or near_ball_max_distance < near_ball_min_distance:
        raise ValueError(
            "--near-ball-init distances must be positive and ordered, "
            f"got [{near_ball_min_distance}, {near_ball_max_distance}]."
        )
    Cfg.env.high_level_near_ball_init_probability = near_ball_probability
    Cfg.env.high_level_near_ball_init_distance_range = [
        near_ball_min_distance,
        near_ball_max_distance,
    ]
    near_ball_angle = abs(float(getattr(args, "near_ball_init_max_angle", 0.35)))
    Cfg.env.high_level_near_ball_init_angle_range = [-near_ball_angle, near_ball_angle]
    Cfg.env.num_static_opponents = 0 if self_play_enabled else num_robots
    if not self_play_enabled:
        Cfg.env.static_opponent_size = [0.45, 0.45, 0.50]
        Cfg.env.static_opponent_x_range = [
            -0.5 * args.field_length + 1.0,
            0.5 * args.field_length - 1.0,
        ]
        Cfg.env.static_opponent_y_range = [
            -0.5 * args.field_width + 0.8,
            0.5 * args.field_width - 0.8,
        ]
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
    Cfg.domain_rand.lag_timesteps = 0
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.control.control_type = "P"

    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_gravity = False

    for key in list(vars(Cfg.reward_scales).keys()):
        if not key.startswith("_"):
            setattr(Cfg.reward_scales, key, 0.0)
    for reward_name, scale in HIGH_LEVEL_REWARD_SCALES.items():
        setattr(Cfg.reward_scales, reward_name, scale)
    # A pass has no receiver and is identically zero in the single-robot task.
    # Remove it from the active objective so the saved configuration accurately
    # describes what can contribute to learning.
    if num_robots == 1:
        Cfg.reward_scales.high_level_pass = 0.0

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
    Cfg.rewards.high_level_robot_collision_distance = 0.75
    Cfg.rewards.high_level_obstacle_safe_distance = 0.55
    Cfg.rewards.high_level_dribble_skill_distance = 1.0
    Cfg.rewards.high_level_dribble_control_distance = 0.8
    Cfg.rewards.high_level_skill_command_min_speed = 0.2
    Cfg.rewards.high_level_dribble_min_ball_speed = 0.1
    Cfg.rewards.high_level_dribble_target_ball_speed = 1.0
    Cfg.rewards.high_level_shoot_skill_distance = 0.75
    Cfg.rewards.high_level_shoot_min_forward = -0.1
    Cfg.rewards.high_level_shoot_lateral_reach = 0.45
    Cfg.rewards.high_level_shoot_min_ball_speed = 0.8
    Cfg.rewards.high_level_shoot_min_delta_speed = 0.25
    Cfg.rewards.high_level_shoot_target_delta_speed = 1.5
    Cfg.rewards.high_level_shoot_min_command_alignment = 0.6
    # Retained for goal-directed observation/reward shaping consumers; the
    # wrapper deliberately does not treat it as a universal validity rule.
    Cfg.rewards.high_level_shoot_alignment = 0.35
    Cfg.rewards.high_level_approach_walk_speed = 0.9
    Cfg.rewards.high_level_goal_facing_target_speed = 0.5
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

    # Conservative until load_skill_policies resolves the exact per-policy
    # training clips and raises this raw-environment ceiling as needed.
    Cfg.normalization.clip_actions = 1.0
    Cfg.asset.terminate_after_contacts_on = []


def load_skill_policies(args):
    source_kind = str(getattr(args, "skill_policy_source", "wandb"))
    if source_kind not in ("wandb", "local"):
        raise ValueError(
            f"Unsupported skill_policy_source={source_kind!r}; expected 'wandb' or 'local'."
        )
    runs = {
        skill: getattr(args, f"{skill}_wandb_run", None)
        for skill in SKILL_NAMES
    }
    local_dirs = {
        skill: getattr(args, f"{skill}_policy_dir", None)
        for skill in SKILL_NAMES
    }
    if source_kind == "wandb":
        for skill, run in runs.items():
            if not run:
                raise ValueError(f"Missing W&B run for {skill}. Pass --{skill}-wandb-run.")
    else:
        missing = [f"--{skill}-policy-dir" for skill, value in local_dirs.items() if not value]
        if missing:
            raise ValueError(
                "--skill-policy-source local requires " + ", ".join(missing)
            )

    from scripts.play_walk_dribble_shoot import load_policy_record, resolve_wandb_policy_files
    from scripts.playback_utils import (
        build_policy_metadata,
        find_policy_config_path,
        resolve_ac_weights_file,
        resolve_policy_files,
    )

    policy_types = {
        "walk": "walking",
        "dribble": "ball",
        "shoot": "ball",
    }
    from dribblebot.envs.base.legged_robot_config import Cfg

    history_length = int(Cfg.env.num_observation_history)
    expected_history_dims = {
        "walking": 72 * history_length,
        "ball": 75 * history_length,
    }
    policies = {}
    for skill in SKILL_NAMES:
        temporary_download = None
        if source_kind == "wandb":
            run = runs[skill]
            # TorchScript must ultimately read bytes from a filesystem path.
            # Online mode therefore downloads a fresh copy for this process,
            # but deliberately does not consult or populate the repository's
            # persistent tmp/wandb_restore_cache directory.
            temporary_download = tempfile.TemporaryDirectory(
                prefix=f"dribblebot-{skill}-wandb-"
            )
            try:
                (
                    body_path,
                    adaptation_module_path,
                    ac_weights_path,
                    run_path,
                    policy_metadata,
                ) = resolve_wandb_policy_files(
                    run,
                    args.skill_checkpoint,
                    skill=skill,
                    return_metadata=True,
                    cache_root=temporary_download.name,
                    refresh=True,
                    local_config_fallback=False,
                )
                if policy_metadata.get("config_path") is None:
                    raise FileNotFoundError(
                        f"W&B run {run_path!r} did not provide config.yaml; "
                        "the policy action clip cannot be validated."
                    )
                source_label = f"online W&B {run_path}@{args.skill_checkpoint}"
            except Exception:
                temporary_download.cleanup()
                raise
        else:
            policy_dir = Path(local_dirs[skill]).expanduser().resolve()
            if not policy_dir.is_dir():
                raise FileNotFoundError(
                    f"Local {skill} policy directory does not exist: {policy_dir}"
                )
            requested_latest = args.skill_checkpoint in ("latest", "last")
            local_checkpoint = str(args.skill_checkpoint)
            if requested_latest:
                local_checkpoint = newest_complete_local_checkpoint(policy_dir) or "latest"
                if local_checkpoint != "latest":
                    print(
                        f"Pinned local {skill} policy request @{args.skill_checkpoint} "
                        f"to complete checkpoint @{local_checkpoint}."
                    )
            body_path, adaptation_module_path = resolve_policy_files(
                policy_dir,
                local_checkpoint,
            )
            ac_weights_path = resolve_ac_weights_file(
                policy_dir,
                local_checkpoint,
                policy_dir=body_path.parent,
            )
            config_path = find_run_level_config(policy_dir)
            if config_path is None:
                config_path = find_policy_config_path(body_path, (policy_dir.parent,))
            if config_path is None:
                raise FileNotFoundError(
                    f"Could not find config.yaml for local {skill} policy under or above {policy_dir}. "
                    "The W&B layout should contain it in the run's files directory."
                )
            policy_metadata = build_policy_metadata(
                body_path,
                adaptation_module_path,
                ac_weights_path,
                checkpoint=local_checkpoint,
                config_path=config_path,
            )
            run_path = None
            source_label = f"local directory {policy_dir}@{local_checkpoint}"

            resolved_body = Path(body_path).resolve()
            for previous_skill, previous_record in policies.items():
                if Path(previous_record["body_path"]).resolve() == resolved_body:
                    raise ValueError(
                        f"Local {previous_skill} and {skill} policy directories both resolve to "
                        f"the same checkpoint file: {resolved_body}. W&B run folders often contain "
                        "symlinks into a shared tmp/legged_data directory; those links are not "
                        "independent archived checkpoints. Point each --*-policy-dir at real, "
                        "run-specific files or use --skill-policy-source wandb."
                    )

        policy_metadata["source_kind"] = source_kind
        policy_metadata["source_location"] = (
            run_path if source_kind == "wandb" else str(policy_dir)
        )
        try:
            record = load_policy_record(
                skill,
                body_path,
                adaptation_module_path,
                ac_weights_path,
                policy_types[skill],
                args.policy_device,
                source_label,
                policy_metadata=policy_metadata,
            )
        except Exception:
            if temporary_download is not None:
                temporary_download.cleanup()
            raise
        max_skill_action_clip = getattr(args, "max_skill_action_clip", None)
        if max_skill_action_clip is not None:
            max_skill_action_clip = float(max_skill_action_clip)
            if max_skill_action_clip <= 0.0:
                raise ValueError("--max-skill-action-clip must be positive")
            policy_clip = float(record["action_clip"])
            if policy_clip > max_skill_action_clip + 1e-6:
                if temporary_download is not None:
                    temporary_download.cleanup()
                raise ValueError(
                    f"Refusing {skill} policy with action clip {policy_clip:g}; "
                    f"the allowed maximum is {max_skill_action_clip:g}. This is usually an "
                    "old saturated checkpoint. Retrain/export the stabilized skill or explicitly "
                    "raise --max-skill-action-clip after validating it."
                )
        if temporary_download is not None:
            # Keep the online download alive for the training process so paths
            # recorded in policy metadata remain auditable during the run.
            record["_temporary_policy_download"] = temporary_download
        expected_dim = expected_history_dims[policy_types[skill]]
        if record["expected_history_dim"] != expected_dim:
            raise ValueError(
                f"{skill} policy from {source_label!r} expects history dimension "
                f"{record['expected_history_dim']}, but {policy_types[skill]} AS2 skills require "
                f"{expected_dim}. This usually means the wrong W&B run/checkpoint was restored."
            )
        policies[skill] = record

    # The wrapper clips each selected policy independently.  The raw env must
    # then accept the widest of those already-safe ranges without silently
    # truncating one skill or exposing another skill to that wider range.
    raw_action_clip = max(float(record["action_clip"]) for record in policies.values())
    Cfg.normalization.clip_actions = raw_action_clip
    print(f"Raw low-level action clip (max of skill policies): {raw_action_clip}")
    return policies


def train_robot(args):
    validate_high_level_training_args(args)

    import isaacgym
    assert isaacgym

    from dribblebot.envs.base.legged_robot_config import Cfg

    configure_high_level_cfg(Cfg, args)
    skill_policies = load_skill_policies(args)
    if getattr(args, "validate_skill_policies_only", False):
        print("Validated all AS2 low-level skill policies; training was not started.")
        return

    import wandb

    from dribblebot.envs.as2.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    from dribblebot.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from dribblebot.envs.wrappers.shared_self_play_wrapper import SharedPolicySelfPlayWrapper
    from dribblebot_learn.ppo_cse import Runner, RunnerArgs
    from dribblebot_learn.ppo_cse.actor_critic import AC_Args
    from dribblebot_learn.ppo_cse.ppo import PPO_Args

    RunnerArgs.resume = bool(args.resume)
    RunnerArgs.resume_path = args.resume_run
    RunnerArgs.resume_checkpoint = args.resume_checkpoint
    RunnerArgs.save_video_interval = 500
    # Keep coordinator checkpoints out of the legacy low-level walking
    # directory (tmp/legged_data).  The runner writes generic names such as
    # body_latest.jit, so sharing that directory silently replaces the walking
    # policy while high-level training is running.
    RunnerArgs.checkpoint_dir = args.checkpoint_dir
    RunnerArgs.self_play_update_interval = args.self_play_update_interval
    PPO_Args.learning_rate = args.learning_rate
    PPO_Args.adaptation_module_learning_rate = args.learning_rate
    # Adaptive KL control may reduce the step size, but it must not grow back
    # toward the old unstable 1e-3 regime.
    PPO_Args.max_learning_rate = args.learning_rate
    PPO_Args.schedule = args.schedule
    PPO_Args.desired_kl = args.desired_kl
    PPO_Args.entropy_coef = args.entropy_coef
    PPO_Args.num_learning_epochs = args.ppo_epochs
    PPO_Args.skill_entropy_coef = args.skill_entropy_coef
    PPO_Args.skill_action_stride = 6
    PPO_Args.num_skill_logits = 3
    PPO_Args.stop_on_excessive_kl = True
    PPO_Args.max_kl_factor = args.max_kl_factor
    AC_Args.init_noise_std = args.init_noise_std
    AC_Args.max_action_std = args.max_noise_std
    AC_Args.action_mean_bound = args.action_mean_bound
    AC_Args.adaptation_labels = []
    AC_Args.adaptation_dims = []

    wandb.init(
        project=args.project or "as2_high_level_soccer",
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
            "skill_policy_source": args.skill_policy_source,
            "skill_source_locations": {
                skill: record["policy_metadata"]["source_location"]
                for skill, record in skill_policies.items()
            },
            "skill_policy_metadata": {
                skill: record["policy_metadata"] for skill, record in skill_policies.items()
            },
            "self_play": {
                "enabled": True,
                "team_size": args.num_robots,
                "shared_actor_parameters": True,
                "opponent_snapshot_interval": args.self_play_update_interval,
                "local_observation_dim": 34,
            },
            "mpc_teacher": (
                {
                    "enabled": True,
                    "world_model_checkpoint": args.world_model_checkpoint,
                    "mpc_config": args.mpc_config,
                    "mpc_profile": args.mpc_profile,
                    "reward_coefficient": args.teacher_reward_coefficient,
                    "opponent_forecast": "frozen policy current action held over horizon",
                }
                if getattr(args, "world_model_checkpoint", None)
                else {"enabled": False}
            ),
        },
    )

    raw_env = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=args.headless, cfg=Cfg)
    match_env = HighLevelSkillWrapper(raw_env, skill_policies)
    env = SharedPolicySelfPlayWrapper(
        match_env,
        team_size=args.num_robots,
        opponent_device=args.policy_device,
    )
    if getattr(args, "world_model_checkpoint", None):
        from dribblebot.envs.wrappers.mpc_teacher_guidance_wrapper import (
            MPCTeacherGuidanceWrapper,
        )
        from dribblebot.mpc import HybridCEMMPC, MPCObjective
        from dribblebot.mpc.config import load_mpc_config
        from dribblebot.world_model.state_adapter import FootballWorldModelStateAdapter
        from dribblebot.world_model.trainer import load_checkpoint

        world_model, checkpoint = load_checkpoint(
            args.world_model_checkpoint, args.device
        )
        world_model.eval()
        expected_robots = 2 * args.num_robots
        if world_model.action_adapter.num_robots != expected_robots:
            raise ValueError(
                "Teacher checkpoint robot count does not match this match: "
                f"checkpoint={world_model.action_adapter.num_robots}, expected={expected_robots}"
            )
        obstacle_count = sum(
            feature.name.startswith("obstacle_")
            for feature in world_model.schema.features
        )
        if obstacle_count:
            raise ValueError(
                "Teacher checkpoint still models static obstacles; train it from "
                "the new two-team collector (max_obstacles=0)."
            )
        stored_steps = int(
            checkpoint.get("training_config", {})
            .get("world_model", {})
            .get("macro_action_steps", args.control_interval)
        )
        if stored_steps != args.control_interval:
            raise ValueError(
                f"World-model macro_action_steps={stored_steps} but training uses "
                f"control_interval={args.control_interval}"
            )
        mpc_config, _ = load_mpc_config(args.mpc_config, args.mpc_profile)
        state_adapter = FootballWorldModelStateAdapter(
            match_env,
            max_obstacles=0,
            schema=world_model.schema,
            event_names=world_model.event_names,
            num_robots=expected_robots,
        )
        if (
            state_adapter.action_adapter.to_dict()["bounds"]
            != world_model.action_adapter.to_dict()["bounds"]
        ):
            raise ValueError(
                "Current high-level command scales differ from the world-model "
                "checkpoint action bounds. Use the same skill command scales as collection."
            )
        objective = MPCObjective(
            world_model.schema,
            world_model.action_adapter,
            world_model.event_names,
            mpc_config,
            controlled_robot_count=args.num_robots,
        )
        planner = HybridCEMMPC(
            world_model,
            state_adapter,
            world_model.action_adapter,
            objective=objective,
            config=mpc_config,
        )
        env = MPCTeacherGuidanceWrapper(
            env,
            planner,
            state_adapter,
            reward_coefficient=args.teacher_reward_coefficient,
        )
    runner = Runner(env, device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True, eval_freq=100)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train an AS2 multi-robot high-level soccer coordinator.")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--checkpoint-dir",
        default="tmp/legged_data/high_level",
        help="Checkpoint output directory reserved for the high-level coordinator.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        type=str_to_bool,
        help="Initialize the high-level coordinator from pretrained actor-critic weights.",
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help=(
            "Optional W&B run path (entity/project/run_id). When omitted, "
            "--resume-checkpoint is loaded from the local filesystem."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="tmp/legged_data/high_level/ac_weights_latest.pt",
        help="Local checkpoint path or W&B artifact name used with --resume.",
    )
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument(
        "--num-robots",
        type=int,
        default=2,
        help="Number of shared-policy AS2 actors per team (supports 1 or more).",
    )
    parser.add_argument(
        "--self-play-update-interval",
        type=int,
        default=500,
        help="PPO iterations between frozen opponent-policy snapshot updates.",
    )
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--episode-length", type=float, default=30.0)
    parser.add_argument("--control-interval", type=int, default=10)
    parser.add_argument("--high-level-history", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--schedule", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--desired-kl", type=float, default=0.01)
    parser.add_argument("--init-noise-std", type=float, default=0.20)
    parser.add_argument("--max-noise-std", type=float, default=0.30)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument(
        "--skill-entropy-coef",
        type=float,
        default=0.02,
        help="Entropy bonus for the categorical walk/dribble/shoot choice.",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=2,
        help="PPO passes over each rollout; kept low to limit policy drift.",
    )
    parser.add_argument(
        "--max-kl-factor",
        type=float,
        default=4.0,
        help="Reject a minibatch update once KL exceeds this multiple of --desired-kl.",
    )
    parser.add_argument(
        "--action-mean-bound",
        type=float,
        default=2.0,
        help="Bound high-level skill logits and raw command parameters.",
    )
    parser.add_argument(
        "--max-skill-action-clip",
        type=float,
        default=1.0,
        help="Reject stale low-level policies trained outside this actuator-action range.",
    )
    parser.add_argument(
        "--use-geometric-skill-fallback",
        dest="use_geometric_skill_fallback",
        action="store_true",
        default=True,
        help="Replace and penalize geometrically invalid dribble/shoot requests (default).",
    )
    parser.add_argument(
        "--no-geometric-skill-fallback",
        dest="use_geometric_skill_fallback",
        action="store_false",
        help="Execute every requested skill even when the ball is out of reach.",
    )
    parser.add_argument("--field-length", type=float, default=8.0)
    parser.add_argument("--field-width", type=float, default=5.0)
    parser.add_argument("--goal-half-width", type=float, default=1.0)
    parser.add_argument(
        "--near-ball-init-probability",
        type=float,
        default=0.6,
        help="Fraction of randomized resets initialized with the ball in front of one robot.",
    )
    parser.add_argument("--near-ball-init-min-distance", type=float, default=0.4)
    parser.add_argument("--near-ball-init-max-distance", type=float, default=0.95)
    parser.add_argument(
        "--near-ball-init-max-angle",
        type=float,
        default=0.35,
        help="Maximum absolute ball bearing in radians for skill-ready resets.",
    )
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
    parser.add_argument(
        "--validate-skill-policies-only",
        action="store_true",
        help="Load and validate walk/dribble/shoot checkpoints, then exit before W&B and simulator startup.",
    )
    parser.add_argument("--walk-wandb-run", default=DEFAULT_AS2_SKILL_RUNS["walk"])
    parser.add_argument("--dribble-wandb-run", default=DEFAULT_AS2_SKILL_RUNS["dribble"])
    parser.add_argument("--shoot-wandb-run", default=DEFAULT_AS2_SKILL_RUNS["shoot"])
    add_skill_policy_source_args(parser)
    return parser


def parse_args():
    return build_arg_parser().parse_args()


if __name__ == "__main__":
    train_robot(parse_args())
