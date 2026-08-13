"""Evaluate Go1 walking, dribbling, and shooting policies in one rollout."""

import argparse
import csv
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import isaacgym

assert isaacgym
import imageio
import numpy as np
import torch
from tqdm import trange

from dribblebot.envs.base.legged_robot_config import Cfg
from dribblebot.envs.go1.go1_config import config_go1
from dribblebot.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from dribblebot.envs.wrappers.history_wrapper import HistoryWrapper
from scripts.playback_utils import (
    GAITS,
    get_raw_env,
    get_sensor_slice,
    load_ac_weights,
    normalize_wandb_run_path,
    patch_obs_command,
    resolve_ac_weights_file,
    resolve_policy_files,
    restore_wandb_file,
    set_walking_command,
)


DEFAULT_RUN_DIR = "runs/improbableailab/dribbling/bvggoq26/dribbling_pretrained"
PHASES = ("approach", "dribble", "shoot")


def command_xy_from_speed_angle(speed, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return np.array(
        [speed * np.cos(angle_rad), speed * np.sin(angle_rad)],
        dtype=np.float32,
    )


def phase_ball_command(phase, args):
    speed = getattr(args, f"{phase}_speed")
    angle_deg = getattr(args, f"{phase}_angle_deg")
    x_override = getattr(args, f"{phase}_x")
    y_override = getattr(args, f"{phase}_y")
    yaw = getattr(args, f"{phase}_yaw", 0.0)

    if x_override is not None or y_override is not None:
        xy = np.array(
            [
                speed if x_override is None else x_override,
                0.0 if y_override is None else y_override,
            ],
            dtype=np.float32,
        )
    else:
        xy = command_xy_from_speed_angle(speed, 0.0 if angle_deg is None else angle_deg)

    return np.array([xy[0], xy[1], yaw], dtype=np.float32)


def max_abs_command_component(args):
    commands = [
        phase_ball_command("dribble", args)[:2],
        phase_ball_command("shoot", args)[:2],
    ]
    max_component = max(float(np.max(np.abs(cmd))) for cmd in commands)
    max_speed = max(abs(args.dribble_speed), abs(args.shoot_speed))
    return max(1.5, max_component, max_speed)


def xy_angle_deg(xy):
    if float(np.linalg.norm(xy)) < 1e-6:
        return 0.0
    return float(np.rad2deg(np.arctan2(xy[1], xy[0])))


def wrap_angle_deg(angle):
    return float((angle + 180.0) % 360.0 - 180.0)


def apply_cfg_dict(cfg_dict):
    if "value" in cfg_dict and isinstance(cfg_dict["value"], dict):
        cfg_dict = cfg_dict["value"]

    for section, values in cfg_dict.items():
        if not hasattr(Cfg, section) or not isinstance(values, dict):
            continue
        target = getattr(Cfg, section)
        for key, value in values.items():
            setattr(target, key, value)


def load_cfg_yaml(config_path):
    import yaml

    with Path(config_path).open("rb") as file:
        payload = yaml.safe_load(file)
    apply_cfg_dict(payload.get("Cfg", payload))


def find_config_path(args, policy_dir):
    if args.config:
        return Path(args.config)

    candidates = []
    if policy_dir is not None:
        candidates.append(Path(policy_dir) / "config.yaml")
    if args.run_dir:
        candidates.append(Path(args.run_dir) / "config.yaml")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_policy_files_any(run_dir, checkpoint, body_path=None, adaptation_module_path=None):
    if body_path and adaptation_module_path:
        return Path(body_path), Path(adaptation_module_path)

    try:
        return resolve_policy_files(run_dir, checkpoint, body_path, adaptation_module_path)
    except FileNotFoundError as original_error:
        root = Path(run_dir)
        candidate_dirs = [
            root,
            root / "tmp" / "legged_data",
            root / "files" / "tmp" / "legged_data",
        ]
        for directory in candidate_dirs:
            body = directory / "body.jit"
            adaptation = directory / "adaptation_module.jit"
            if body.exists() and adaptation.exists():
                return body, adaptation
        raise original_error


def resolve_wandb_policy_files(wandb_run, checkpoint):
    run_path = normalize_wandb_run_path(wandb_run)
    suffix = "latest" if checkpoint in ("latest", "last") else checkpoint

    body_path = restore_wandb_file(run_path, [f"tmp/legged_data/body_{suffix}.jit"])
    adaptation_module_path = restore_wandb_file(run_path, [f"tmp/legged_data/adaptation_module_{suffix}.jit"])

    try:
        ac_weights_path = restore_wandb_file(
            run_path,
            [
                f"tmp/legged_data/ac_weights_{suffix}.pt",
                "tmp/legged_data/ac_weights_last.pt",
            ] if suffix == "latest" else [f"tmp/legged_data/ac_weights_{suffix}.pt"],
        )
    except FileNotFoundError:
        ac_weights_path = None

    return body_path, adaptation_module_path, ac_weights_path, run_path


def infer_first_linear_input_dim(module):
    for _, parameter in module.named_parameters():
        if parameter.ndim == 2:
            return int(parameter.shape[1])
    return None


def load_policy_with_metadata(body_path, adaptation_module_path, policy_device):
    body = torch.jit.load(str(body_path), map_location=policy_device).eval()
    adaptation_module = torch.jit.load(str(adaptation_module_path), map_location=policy_device).eval()
    expected_history_dim = infer_first_linear_input_dim(adaptation_module)

    def policy(obs):
        obs_history = obs["obs_history"].to(policy_device)
        latent = adaptation_module.forward(obs_history)
        action = body.forward(torch.cat((obs_history, latent), dim=-1))
        return action

    return policy, expected_history_dim


def load_policy_record(label, body_path, adaptation_module_path, ac_weights_path, policy_type, policy_device, source):
    print(f"{label}:")
    print(f"  source: {source}")
    print(f"  policy type: {policy_type}")
    print(f"  body: {body_path}")
    print(f"  adaptation module: {adaptation_module_path}")
    print(f"  actor-critic weights: {ac_weights_path}")

    load_ac_weights(ac_weights_path, map_location=policy_device)
    policy, expected_history_dim = load_policy_with_metadata(body_path, adaptation_module_path, policy_device)
    print(f"  expected obs_history dim: {expected_history_dim}")

    return {
        "policy": policy,
        "policy_type": policy_type,
        "expected_history_dim": expected_history_dim,
        "body_path": Path(body_path),
        "adaptation_module_path": Path(adaptation_module_path),
        "ac_weights_path": Path(ac_weights_path) if ac_weights_path is not None else None,
        "source": source,
    }


def load_local_phase_policy(args, phase, checkpoint):
    run_dir = getattr(args, f"{phase}_run_dir") or args.run_dir
    body_path, adaptation_module_path = resolve_policy_files_any(run_dir, checkpoint)
    ac_weights_path = resolve_ac_weights_file(
        run_dir,
        checkpoint,
        policy_dir=body_path.parent,
    )
    return load_policy_record(
        phase,
        body_path,
        adaptation_module_path,
        ac_weights_path,
        getattr(args, f"{phase}_policy_type"),
        args.policy_device,
        f"local {run_dir}@{checkpoint}",
    )


def load_wandb_phase_policy(args, phase, checkpoint):
    wandb_run = getattr(args, f"{phase}_wandb_run") or args.wandb_run
    if not wandb_run:
        raise ValueError(
            f"Missing --{phase}-wandb-run. Provide it, provide --wandb-run as a fallback, "
            f"or use --{phase}-local with --{phase}-run-dir."
        )

    body_path, adaptation_module_path, ac_weights_path, run_path = resolve_wandb_policy_files(
        wandb_run,
        checkpoint,
    )
    return load_policy_record(
        phase,
        body_path,
        adaptation_module_path,
        ac_weights_path,
        getattr(args, f"{phase}_policy_type"),
        args.policy_device,
        f"W&B {run_path}@{checkpoint}",
    )


def load_phase_policies(args):
    using_phase_sources = any(
        getattr(args, f"{phase}_wandb_run") or getattr(args, f"{phase}_local") or getattr(args, f"{phase}_run_dir")
        for phase in PHASES
    )
    policies = {}

    if using_phase_sources:
        for phase in PHASES:
            checkpoint = getattr(args, f"{phase}_checkpoint") or args.checkpoint
            if getattr(args, f"{phase}_local") or getattr(args, f"{phase}_run_dir"):
                policies[phase] = load_local_phase_policy(args, phase, checkpoint)
            else:
                policies[phase] = load_wandb_phase_policy(args, phase, checkpoint)
        return policies

    if args.wandb_run and not args.local:
        body_path, adaptation_module_path, ac_weights_path, run_path = resolve_wandb_policy_files(
            args.wandb_run,
            args.checkpoint,
        )
        record = load_policy_record(
            "all phases",
            body_path,
            adaptation_module_path,
            ac_weights_path,
            args.policy_type,
            args.policy_device,
            f"W&B {run_path}@{args.checkpoint}",
        )
    else:
        body_path, adaptation_module_path = resolve_policy_files_any(
            args.run_dir,
            args.checkpoint,
            body_path=args.body,
            adaptation_module_path=args.adaptation_module,
        )
        ac_weights_path = resolve_ac_weights_file(
            args.run_dir,
            args.checkpoint,
            ac_weights_path=args.pt,
            policy_dir=body_path.parent,
        )
        record = load_policy_record(
            "all phases",
            body_path,
            adaptation_module_path,
            ac_weights_path,
            args.policy_type,
            args.policy_device,
            f"local {args.run_dir}@{args.checkpoint}",
        )

    for phase in PHASES:
        policies[phase] = record
    return policies


def configure_rollout_cfg(args, config_path=None):
    config_go1(Cfg)
    if config_path is not None:
        load_cfg_yaml(config_path)

    Cfg.env.num_envs = 1
    Cfg.env.num_recording_envs = 1
    Cfg.env.record_video = True
    Cfg.env.num_observation_history = 15
    Cfg.env.num_observations = 75
    Cfg.env.num_privileged_obs = 6
    Cfg.env.add_balls = True
    Cfg.env.priv_observe_ball_drag = False
    Cfg.env.episode_length_s = args.episode_length

    Cfg.robot.name = "go1"
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

    Cfg.commands.num_commands = 15
    Cfg.commands.distributional_commands = True
    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.balance_gait_distribution = False
    Cfg.commands.binary_phases = False
    Cfg.commands.gaitwise_curricula = False
    Cfg.commands.heading_command = False
    Cfg.commands.resampling_time = 1_000_000.0

    max_xy_command = max_abs_command_component(args)
    max_yaw_command = max(args.approach_max_yaw, abs(args.dribble_yaw))

    Cfg.commands.lin_vel_x = [-max_xy_command, max_xy_command]
    Cfg.commands.lin_vel_y = [-max_xy_command, max_xy_command]
    Cfg.commands.ang_vel_yaw = [-max_yaw_command, max_yaw_command]
    Cfg.commands.body_height_cmd = [-0.05, 0.05]
    Cfg.commands.gait_frequency_cmd_range = [args.step_frequency, args.step_frequency]
    Cfg.commands.gait_phase_cmd_range = [0.5, 0.5]
    Cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_duration_cmd_range = [args.gait_duration, args.gait_duration]
    Cfg.commands.footswing_height_range = [args.footswing_height, args.footswing_height]
    Cfg.commands.body_pitch_range = [args.pitch, args.pitch]
    Cfg.commands.body_roll_range = [args.roll, args.roll]
    Cfg.commands.stance_width_range = [0.0, 0.1]
    Cfg.commands.stance_length_range = [0.0, 0.1]

    Cfg.commands.limit_vel_x = [-max_xy_command, max_xy_command]
    Cfg.commands.limit_vel_y = [-max_xy_command, max_xy_command]
    Cfg.commands.limit_vel_yaw = [-max_yaw_command, max_yaw_command]
    Cfg.commands.limit_body_height = [-0.05, 0.05]
    Cfg.commands.limit_gait_frequency = [args.step_frequency, args.step_frequency]
    Cfg.commands.limit_gait_phase = [0.5, 0.5]
    Cfg.commands.limit_gait_offset = [0.0, 0.0]
    Cfg.commands.limit_gait_bound = [0.0, 0.0]
    Cfg.commands.limit_gait_duration = [args.gait_duration, args.gait_duration]
    Cfg.commands.limit_footswing_height = [args.footswing_height, args.footswing_height]
    Cfg.commands.limit_body_pitch = [args.pitch, args.pitch]
    Cfg.commands.limit_body_roll = [args.roll, args.roll]
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

    Cfg.terrain.mesh_type = "boxes_tm"
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.num_border_boxes = 0.0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.terrain_length = 8.0
    Cfg.terrain.terrain_width = 8.0
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.terrain.yaw_init_range = 0.0
    Cfg.terrain.x_init_offset = 0.0
    Cfg.terrain.y_init_offset = 0.0
    Cfg.terrain.horizontal_scale = 0.05
    Cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.curriculum = False
    Cfg.terrain.max_init_terrain_level = 1
    Cfg.terrain.max_step_height = 0.26
    Cfg.terrain.min_step_run = 0.25
    Cfg.terrain.max_step_run = 0.4

    Cfg.ball.ball_init_pos = [args.ball_x, args.ball_y, args.ball_z]
    Cfg.ball.init_pos_range = [0.0, 0.0, 0.0]
    Cfg.ball.init_vel_range = [0.0, 0.0, 0.0]
    Cfg.ball.pos_reset_prob = 0.0
    Cfg.ball.vel_reset_prob = 0.0
    Cfg.ball.pos_reset_range = [0.0, 0.0, 0.0]
    Cfg.ball.vel_reset_range = [0.0, 0.0, 0.0]
    Cfg.ball.vision_receive_prob = 1.0

    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_ground_restitution = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_ball_drag = False
    Cfg.domain_rand.randomize_ball_restitution = False
    Cfg.domain_rand.randomize_ball_friction = False
    Cfg.domain_rand.randomize_lag_timesteps = False
    Cfg.domain_rand.lag_timesteps = 0
    Cfg.domain_rand.rand_interval_s = 1_000_000.0
    Cfg.domain_rand.gravity_rand_interval_s = 1_000_000.0
    Cfg.domain_rand.ball_drag_rand_interval_s = 1_000_000.0
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.0]

    Cfg.control.control_type = "actuator_net"

    Cfg.reward_scales.orientation = -5.0
    Cfg.reward_scales.torques = -0.0001
    Cfg.reward_scales.dof_vel = -0.0001
    Cfg.reward_scales.dof_acc = -2.5e-7
    Cfg.reward_scales.collision = -5.0
    Cfg.reward_scales.action_rate = -0.01
    Cfg.reward_scales.dof_pos_limits = -10.0
    Cfg.reward_scales.dof_pos = -0.05
    Cfg.reward_scales.action_smoothness_1 = -0.1
    Cfg.reward_scales.action_smoothness_2 = -0.1
    Cfg.reward_scales.tracking_contacts_shaped_force = 0.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 0.0
    Cfg.reward_scales.tracking_lin_vel = 0.0
    Cfg.reward_scales.tracking_ang_vel = 0.0
    Cfg.reward_scales.lin_vel_z = 0.0
    Cfg.reward_scales.ang_vel_xy = 0.0
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.feet_slip = 0.0
    Cfg.reward_scales.jump = 0.0
    Cfg.reward_scales.base_height = 0.0
    Cfg.reward_scales.feet_impact_vel = 0.0
    Cfg.reward_scales.dribbling_robot_ball_vel = 0.5
    Cfg.reward_scales.dribbling_robot_ball_pos = 4.0
    Cfg.reward_scales.dribbling_ball_vel = 4.0
    Cfg.reward_scales.dribbling_robot_ball_yaw = 4.0
    Cfg.reward_scales.dribbling_ball_vel_norm = 4.0
    Cfg.reward_scales.dribbling_ball_vel_angle = 4.0
    Cfg.reward_scales.shooting_ball_vel = 6.0
    Cfg.reward_scales.shooting_ball_vel_norm = 2.0
    Cfg.reward_scales.shooting_ball_vel_angle = 2.0
    Cfg.reward_scales.shooting_ball_out = 3.0
    Cfg.reward_scales.shooting_robot_ball_pos = 1.0
    Cfg.reward_scales.shooting_robot_ball_behind = 1.0
    Cfg.reward_scales.shooting_robot_approach_ball = 0.5

    Cfg.rewards.reward_container_name = "SoccerRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.2
    Cfg.rewards.use_terminal_roll_pitch = False
    Cfg.rewards.terminal_body_ori = 0.5
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.0
    Cfg.rewards.gait_vel_sigma = 10.0

    Cfg.normalization.clip_actions = 10.0
    Cfg.normalization.friction_range = [0.0, 1.0]
    Cfg.normalization.ground_friction_range = [0.7, 4.0]
    Cfg.asset.terminate_after_contacts_on = []


def make_env(args, config_path=None):
    configure_rollout_cfg(args, config_path)
    env = VelocityTrackingEasyEnv(sim_device=args.device, headless=args.headless, cfg=Cfg)
    env = HistoryWrapper(env)
    return env


def set_ball_command(raw_env, cmd, args):
    raw_env.commands[:, :] = 0.0
    raw_env.commands[:, 0] = float(cmd[0])
    raw_env.commands[:, 1] = float(cmd[1])
    raw_env.commands[:, 2] = float(cmd[2])

    if raw_env.cfg.commands.num_commands > 3:
        raw_env.commands[:, 3] = args.body_height
    if raw_env.cfg.commands.num_commands > 4:
        raw_env.commands[:, 4] = args.step_frequency
    if raw_env.cfg.commands.num_commands > 7:
        gait = torch.tensor(GAITS[args.gait], dtype=raw_env.commands.dtype, device=raw_env.device)
        raw_env.commands[:, 5:8] = gait
    if raw_env.cfg.commands.num_commands > 8:
        raw_env.commands[:, 8] = args.gait_duration
    if raw_env.cfg.commands.num_commands > 9:
        raw_env.commands[:, 9] = args.footswing_height
    if raw_env.cfg.commands.num_commands > 10:
        raw_env.commands[:, 10] = args.pitch
    if raw_env.cfg.commands.num_commands > 11:
        raw_env.commands[:, 11] = args.roll
    if raw_env.cfg.commands.num_commands > 12:
        raw_env.commands[:, 12] = args.stance_width
    if raw_env.cfg.commands.num_commands > 13:
        raw_env.commands[:, 13] = args.stance_length
    if raw_env.cfg.commands.num_commands > 14:
        raw_env.commands[:, 14] = args.aux_reward_coef


def phase_command(phase, args):
    if phase == "approach":
        return np.array([args.approach_speed, 0.0, 0.0], dtype=np.float32)
    if phase in ("dribble", "shoot"):
        return phase_ball_command(phase, args)
    raise ValueError(f"Unknown phase: {phase}")


def walking_approach_command(raw_env, args):
    ball_xy_body = raw_env.object_local_pos[0, :2]
    distance = torch.norm(ball_xy_body).clamp(min=1e-6)
    bearing = torch.atan2(ball_xy_body[1], ball_xy_body[0]).item()
    alignment = max(0.0, float(np.cos(bearing)))
    speed = min(args.approach_speed, max(0.0, distance.item() - 0.25))
    x_vel = speed * (args.approach_min_forward_scale + (1.0 - args.approach_min_forward_scale) * alignment)
    y_vel = float(np.clip(
        args.approach_lateral_gain * ball_xy_body[1].item(),
        -args.approach_max_y_vel,
        args.approach_max_y_vel,
    ))
    yaw_vel = float(np.clip(args.approach_yaw_gain * bearing, -args.approach_max_yaw, args.approach_max_yaw))
    return np.array([
        x_vel,
        y_vel,
        yaw_vel,
    ], dtype=np.float32)


def apply_phase_command(raw_env, phase, policy_type, args):
    if phase == "approach" and policy_type == "walking":
        cmd = walking_approach_command(raw_env, args)
        set_walking_command(raw_env, cmd, args)
        return cmd

    cmd = phase_command(phase, args)
    set_ball_command(raw_env, cmd, args)
    return cmd


def strip_object_sensor_from_obs(obs, raw_env, object_slice):
    obs_without_object = torch.cat(
        (
            obs["obs"][:, :object_slice.start],
            obs["obs"][:, object_slice.stop:],
        ),
        dim=-1,
    )
    history = obs["obs_history"].view(obs["obs_history"].shape[0], -1, raw_env.num_obs)
    history_without_object = torch.cat(
        (
            history[:, :, :object_slice.start],
            history[:, :, object_slice.stop:],
        ),
        dim=-1,
    ).reshape(obs["obs_history"].shape[0], -1)

    return {
        "obs": obs_without_object,
        "privileged_obs": obs.get("privileged_obs"),
        "obs_history": history_without_object,
    }


def adapt_obs_for_policy(obs, raw_env, policy_record, object_slice):
    expected_history_dim = policy_record.get("expected_history_dim")
    current_history_dim = obs["obs_history"].shape[1]
    if expected_history_dim is None or expected_history_dim == current_history_dim:
        return obs

    if object_slice is not None:
        history_length = raw_env.cfg.env.num_observation_history
        object_dim = object_slice.stop - object_slice.start
        without_object_history_dim = (raw_env.num_obs - object_dim) * history_length
        if expected_history_dim == without_object_history_dim:
            return strip_object_sensor_from_obs(obs, raw_env, object_slice)

    raise ValueError(
        f"{policy_record['source']} expects obs_history dim {expected_history_dim}, "
        f"but the rollout provides {current_history_dim}. "
        "This script can only auto-adapt by removing ObjectSensor."
    )


def maybe_advance_phase(phase_idx, phase_step, raw_env, dribble_start_ball_xy, args):
    phase = PHASES[phase_idx]
    robot_xy = raw_env.base_pos[0, :2]
    ball_xy = raw_env.object_pos_world_frame[0, :2]
    robot_ball_dist = torch.norm(ball_xy - robot_xy).item()
    ball_dribble_dist = torch.norm(ball_xy - dribble_start_ball_xy).item()
    next_step = phase_step + 1
    approach_timed_out = args.approach_steps > 0 and next_step >= args.approach_steps
    dribble_timed_out = args.dribble_steps > 0 and next_step >= args.dribble_steps
    shoot_ready_to_reapproach = next_step >= args.shoot_min_steps

    if phase == "approach":
        if robot_ball_dist <= args.approach_distance or approach_timed_out:
            return 1, 0, ball_xy.clone()
    elif phase == "dribble":
        if robot_ball_dist >= args.reapproach_distance:
            return 0, 0, ball_xy.clone()
        if ball_dribble_dist >= args.dribble_distance or dribble_timed_out:
            return 2, 0, dribble_start_ball_xy
    elif phase == "shoot":
        if shoot_ready_to_reapproach and robot_ball_dist >= args.reapproach_distance:
            return 0, 0, ball_xy.clone()

    return phase_idx, next_step, dribble_start_ball_xy


def initial_phase(raw_env, args):
    robot_xy = raw_env.base_pos[0, :2]
    ball_xy = raw_env.object_pos_world_frame[0, :2]
    robot_ball_dist = torch.norm(ball_xy - robot_xy).item()
    if robot_ball_dist <= args.approach_distance:
        return PHASES.index("dribble"), ball_xy.clone()
    return PHASES.index("approach"), ball_xy.clone()


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, rows, show):
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([row["time_s"] for row in rows], dtype=np.float32)
    command_speed = np.array(
        [np.hypot(row["cmd_x"], row["cmd_y"]) for row in rows],
        dtype=np.float32,
    )
    ball_speed = np.array(
        [np.hypot(row["ball_vx"], row["ball_vy"]) for row in rows],
        dtype=np.float32,
    )
    robot_ball_dist = np.array([row["robot_ball_dist"] for row in rows], dtype=np.float32)
    rewards = np.array([row["reward"] for row in rows], dtype=np.float32)
    phase_ids = np.array([PHASES.index(row["phase"]) for row in rows], dtype=np.float32)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(times, ball_speed, label="ball", color="black")
    axes[0].plot(times, command_speed, label="command", color="tab:blue", linestyle="--")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, robot_ball_dist, color="tab:orange")
    axes[1].set_ylabel("robot-ball dist (m)")
    axes[1].grid(True, alpha=0.25)

    axes[2].step(times, phase_ids, where="post", color="tab:purple")
    axes[2].set_yticks(range(len(PHASES)))
    axes[2].set_yticklabels(PHASES)
    axes[2].set_ylabel("phase")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(times, rewards, color="tab:green")
    axes[3].set_ylabel("reward")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def validate_command_args(args, parser):
    for phase in ("dribble", "shoot"):
        angle = getattr(args, f"{phase}_angle_deg")
        x_override = getattr(args, f"{phase}_x")
        y_override = getattr(args, f"{phase}_y")
        speed = getattr(args, f"{phase}_speed")

        if speed < 0.0:
            parser.error(f"--{phase}-speed must be non-negative.")
        if angle is not None and (x_override is not None or y_override is not None):
            parser.error(
                f"Use either --{phase}-speed/--{phase}-angle-deg or direct "
                f"--{phase}-x/--{phase}-y, not both."
            )


def run(args):
    phase_policies = load_phase_policies(args)
    config_anchor = phase_policies["dribble"]["body_path"].parent
    config_path = find_config_path(args, config_anchor)
    print(f"Using config: {config_path or 'built-in ball rollout config'}")

    env = make_env(args, config_path)
    raw_env = get_raw_env(env)
    command_slice = get_sensor_slice(raw_env, "RCSensor")
    object_slice = get_sensor_slice(raw_env, "ObjectSensor")

    output_video = Path(args.video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_video), fps=args.fps or int(round(1.0 / raw_env.dt)))

    obs = env.reset()
    rows = []
    phase_idx, dribble_start_ball_xy = initial_phase(raw_env, args)
    phase_step = 0

    try:
        for step in trange(args.steps, desc="Walk-dribble-shoot"):
            phase = PHASES[phase_idx]
            phase_policy = phase_policies[phase]
            cmd = apply_phase_command(raw_env, phase, phase_policy["policy_type"], args)
            patch_obs_command(raw_env, obs, command_slice)
            policy_obs = adapt_obs_for_policy(obs, raw_env, phase_policy, object_slice)

            with torch.no_grad():
                action = phase_policy["policy"](policy_obs).to(raw_env.device)

            obs, reward, done, _ = env.step(action)

            robot_xy = raw_env.base_pos[0, :2].detach().cpu().numpy()
            ball_xy = raw_env.object_pos_world_frame[0, :2].detach().cpu().numpy()
            ball_vel = raw_env.object_lin_vel[0, :2].detach().cpu().numpy()
            cmd_xy = cmd[:2]
            cmd_speed = float(np.linalg.norm(cmd_xy))
            ball_speed = float(np.linalg.norm(ball_vel))
            cmd_angle = xy_angle_deg(cmd_xy)
            ball_angle = xy_angle_deg(ball_vel)
            rows.append({
                "time_s": step * raw_env.dt,
                "phase": phase,
                "cmd_x": float(cmd[0]),
                "cmd_y": float(cmd[1]),
                "cmd_yaw": float(cmd[2]) if len(cmd) > 2 else 0.0,
                "cmd_speed": cmd_speed,
                "cmd_angle_deg": cmd_angle,
                "policy_type": phase_policy["policy_type"],
                "robot_x": float(robot_xy[0]),
                "robot_y": float(robot_xy[1]),
                "ball_x": float(ball_xy[0]),
                "ball_y": float(ball_xy[1]),
                "ball_vx": float(ball_vel[0]),
                "ball_vy": float(ball_vel[1]),
                "ball_speed": ball_speed,
                "ball_angle_deg": ball_angle,
                "ball_cmd_angle_error_deg": wrap_angle_deg(ball_angle - cmd_angle),
                "robot_ball_dist": float(np.linalg.norm(ball_xy - robot_xy)),
                "reward": float(reward[0].item()),
                "done": int(done[0].item()),
                "action_norm": float(torch.norm(action[0]).item()),
            })

            frame = env.render(mode="rgb_array")
            writer.append_data(frame)

            phase_idx, phase_step, dribble_start_ball_xy = maybe_advance_phase(
                phase_idx,
                phase_step,
                raw_env,
                dribble_start_ball_xy,
                args,
            )
    finally:
        writer.close()

    metrics_csv = Path(args.csv)
    plot_path = Path(args.plot)
    write_metrics_csv(metrics_csv, rows)
    save_plot(plot_path, rows, args.show_plot)

    ball_vel_xy = np.array([[row["ball_vx"], row["ball_vy"]] for row in rows], dtype=np.float32)
    cmd_xy = np.array([[row["cmd_x"], row["cmd_y"]] for row in rows], dtype=np.float32)
    ball_speed = np.linalg.norm(ball_vel_xy, axis=1)
    cmd_speed = np.linalg.norm(cmd_xy, axis=1)
    shoot_rows = [row for row in rows if row["phase"] == "shoot"]
    ball_policy_rows = [
        row for row in rows
        if row["phase"] in ("dribble", "shoot") and row["cmd_speed"] > 1e-6 and row["ball_speed"] > 1e-6
    ]
    max_shoot_speed = 0.0
    if shoot_rows:
        max_shoot_speed = max(float(np.hypot(row["ball_vx"], row["ball_vy"])) for row in shoot_rows)
    mean_ball_angle_error = 0.0
    if ball_policy_rows:
        mean_ball_angle_error = float(np.mean([abs(row["ball_cmd_angle_error_deg"]) for row in ball_policy_rows]))

    print(f"Saved video: {output_video}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print(f"Mean ball speed error: {float(np.mean(np.abs(ball_speed - cmd_speed))):.4f}")
    print(f"Mean ball direction error: {mean_ball_angle_error:.2f} deg")
    print(f"Max shoot-phase ball speed: {max_shoot_speed:.4f}")
    print(f"Terminations: {sum(row['done'] for row in rows)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a single rollout that approaches the ball, dribbles it, then shoots."
    )
    parser.add_argument("--wandb-run", default=None, help="Optional W&B run URL or entity/project/run_id.")
    parser.add_argument("--approach-wandb-run", default=None, help="W&B run for the approach phase.")
    parser.add_argument("--dribble-wandb-run", default=None, help="W&B run for the dribble phase.")
    parser.add_argument("--shoot-wandb-run", default=None, help="W&B run for the shoot phase.")
    parser.add_argument("--local", action="store_true", help="Use local checkpoint files from --run-dir instead of W&B.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Local run directory or checkpoint directory to search.")
    parser.add_argument("--approach-local", action="store_true", help="Load the approach policy from local files.")
    parser.add_argument("--dribble-local", action="store_true", help="Load the dribble policy from local files.")
    parser.add_argument("--shoot-local", action="store_true", help="Load the shoot policy from local files.")
    parser.add_argument("--approach-run-dir", default=None, help="Local run directory for the approach policy.")
    parser.add_argument("--dribble-run-dir", default=None, help="Local run directory for the dribble policy.")
    parser.add_argument("--shoot-run-dir", default=None, help="Local run directory for the shoot policy.")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint suffix, for example latest or 62800.")
    parser.add_argument("--approach-checkpoint", default=None, help="Checkpoint suffix for the approach phase.")
    parser.add_argument("--dribble-checkpoint", default=None, help="Checkpoint suffix for the dribble phase.")
    parser.add_argument("--shoot-checkpoint", default=None, help="Checkpoint suffix for the shoot phase.")
    parser.add_argument("--body", default=None, help="Direct path to body JIT file.")
    parser.add_argument("--adaptation-module", default=None, help="Direct path to adaptation_module JIT file.")
    parser.add_argument("--pt", default=None, help="Direct path to ac_weights .pt file.")
    parser.add_argument("--policy-type", choices=["ball", "walking"], default="ball", help="Observation layout for a single policy.")
    parser.add_argument("--approach-policy-type", choices=["ball", "walking"], default="walking", help="Observation layout for the approach policy.")
    parser.add_argument("--dribble-policy-type", choices=["ball", "walking"], default="ball", help="Observation layout for the dribble policy.")
    parser.add_argument("--shoot-policy-type", choices=["ball", "walking"], default="ball", help="Observation layout for the shoot policy.")
    parser.add_argument("--config", default=None, help="Optional config.yaml to load before evaluation overrides.")
    parser.add_argument("--device", default="cuda:0", help="Isaac Gym simulation device.")
    parser.add_argument("--policy-device", default="cpu", help="Device for the JIT policy modules.")
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Gym viewer.")
    parser.add_argument("--steps", type=int, default=900, help="Number of control steps to evaluate.")
    parser.add_argument("--episode-length", type=float, default=60.0, help="Episode length in seconds.")
    parser.add_argument("--fps", type=int, default=None, help="Video FPS. Defaults to 1 / env.dt.")
    parser.add_argument("--video", default="outputs/walk_dribble_shoot.mp4", help="Output MP4 path.")
    parser.add_argument("--plot", default="outputs/walk_dribble_shoot_metrics.png", help="Output plot path.")
    parser.add_argument("--csv", default="outputs/walk_dribble_shoot_metrics.csv", help="Output metrics CSV path.")
    parser.add_argument("--show-plot", action="store_true", help="Show the matplotlib window after saving the plot.")

    parser.add_argument("--approach-speed", type=float, default=0.45, help="Ball velocity command during approach.")
    parser.add_argument("--approach-yaw-gain", type=float, default=1.5, help="Yaw-rate gain for walking approach.")
    parser.add_argument("--approach-max-yaw", type=float, default=1.0, help="Max yaw-rate command during walking approach.")
    parser.add_argument("--approach-max-y-vel", type=float, default=0.0, help="Max lateral velocity during walking approach.")
    parser.add_argument("--approach-lateral-gain", type=float, default=0.0, help="Small optional lateral gain during walking approach.")
    parser.add_argument("--approach-min-forward-scale", type=float, default=0.25, help="Forward speed scale while turning toward the ball.")
    parser.add_argument("--dribble-speed", type=float, default=0.9, help="World-frame dribbled-ball speed command.")
    parser.add_argument("--dribble-angle-deg", type=float, default=None, help="World-frame dribble command angle in degrees.")
    parser.add_argument("--dribble-x", type=float, default=None, help="Direct world-frame dribble x velocity. Overrides speed/angle.")
    parser.add_argument("--dribble-y", type=float, default=None, help="Direct world-frame dribble y velocity. Overrides speed/angle.")
    parser.add_argument("--dribble-yaw", type=float, default=0.5, help="Robot body-frame yaw-rate command during dribbling.")
    parser.add_argument("--shoot-speed", type=float, default=3.0, help="World-frame post-shot ball speed command.")
    parser.add_argument("--shoot-angle-deg", type=float, default=None, help="World-frame shooting command angle in degrees.")
    parser.add_argument("--shoot-x", type=float, default=None, help="Direct world-frame shoot x velocity. Overrides speed/angle.")
    parser.add_argument("--shoot-y", type=float, default=None, help="Direct world-frame shoot y velocity. Overrides speed/angle.")
    parser.add_argument("--approach-steps", type=int, default=0, help="Optional max approach steps before dribbling. Set 0 to disable.")
    parser.add_argument("--dribble-steps", type=int, default=0, help="Optional max dribble steps before shooting. Set 0 to disable.")
    parser.add_argument("--shoot-min-steps", type=int, default=25, help="Minimum shoot steps before returning to approach.")
    parser.add_argument("--approach-distance", type=float, default=0.55, help="Robot-ball distance that starts dribbling.")
    parser.add_argument("--reapproach-distance", type=float, default=1.0, help="Robot-ball distance that returns to approach.")
    parser.add_argument("--dribble-distance", type=float, default=1.2, help="Ball travel distance that starts shooting.")
    parser.add_argument("--ball-x", type=float, default=1.0, help="Initial ball x position in world/env frame.")
    parser.add_argument("--ball-y", type=float, default=0.0, help="Initial ball y position in world/env frame.")
    parser.add_argument("--ball-z", type=float, default=0.5, help="Initial ball z position in world/env frame.")

    parser.add_argument("--gait", choices=sorted(GAITS.keys()), default="trotting")
    parser.add_argument("--body-height", type=float, default=0.0)
    parser.add_argument("--step-frequency", type=float, default=3.0)
    parser.add_argument("--gait-duration", type=float, default=0.5)
    parser.add_argument("--footswing-height", type=float, default=0.09)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--stance-width", type=float, default=0.05)
    parser.add_argument("--stance-length", type=float, default=0.05)
    parser.add_argument("--aux-reward-coef", type=float, default=0.005)
    args = parser.parse_args()
    validate_command_args(args, parser)
    return args


if __name__ == "__main__":
    run(parse_args())
