"""Record a deterministic legacy Isaac Gym rollout for physics parity."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


REBUILD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Calling a virtual-environment interpreter by absolute path does not activate
# the environment, so console tools installed into ``.venv/bin`` (notably
# ninja, required by Isaac Gym's gymtorch JIT extension) may otherwise be
# invisible to PyTorch.
INTERPRETER_BIN = str(Path(sys.executable).parent)
os.environ["PATH"] = os.pathsep.join((INTERPRETER_BIN, os.environ.get("PATH", "")))
for import_root in (REBUILD_ROOT, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from parity.excitation import ExcitationConfig, generate_action_sequence
from parity.physics_profile import make_physics_profile
from parity.policy_contract import LEGACY_PARITY_COMMAND, make_legacy_policy_contract
from parity.schema import SCHEMA_VERSION, save_rollout


TASK_NAMES = {
    "velocity": "Isaac-DribbleBot-AS2-Velocity-Flat-v0",
    "dribble": "Isaac-DribbleBot-AS2-Dribble-Flat-v0",
}

# The legacy LeggedRobot lifecycle reads gait parameters unconditionally even
# when the parity recorder only compares state/action trajectories. Keep its
# complete internal command layout deterministic instead of shrinking it to
# the three velocity commands exposed by the migrated task.
LEGACY_COMMAND_VALUES = LEGACY_PARITY_COMMAND

LEGACY_POLICY_SENSOR_NAMES = (
    "OrientationSensor",
    "RCSensor",
    "JointPositionSensor",
    "JointVelocitySensor",
    "ActionSensor",
    "LastActionSensor",
    "ClockSensor",
    "YawSensor",
    "TimingSensor",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_NAMES), default="velocity")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--excitation", choices=("zero", "sine"), default="zero")
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--frequency_hz", type=float, default=0.75)
    parser.add_argument("--phase_stride_rad", type=float, default=0.37)
    parser.add_argument("--contact_threshold_n", type=float, default=1.0)
    parser.add_argument("--ball_position", type=float, nargs=3, default=(0.6, 0.0, 0.10))
    parser.add_argument("--allow_resets", action="store_true")
    return parser.parse_args()


def _configure_legacy_env(args: argparse.Namespace) -> object:
    from quadruped.envs.as2.as2_config import config_as2
    from quadruped.envs.base.legged_robot_config import Cfg

    config_as2(Cfg)
    # ``config_as2`` configures the asset path, gains, and initial state but
    # legacy training entry points select the concrete robot adapter
    # separately.  Without this assignment the base config silently keeps
    # ``robot.name == "go1"`` and the parity recorder loads the wrong asset.
    Cfg.robot.name = "as2"
    profile = make_physics_profile(include_ball=args.task == "dribble")
    solver = profile["solver"]
    ground = profile["ground"]
    Cfg.env.num_envs = args.num_envs
    Cfg.env.env_spacing = 5.5 if args.task == "dribble" else 2.5
    Cfg.env.add_balls = args.task == "dribble"
    include_ball = args.task == "dribble"
    policy_contract = make_legacy_policy_contract(include_ball=include_ball)
    Cfg.env.num_observations = policy_contract["observation_dim"]
    Cfg.env.num_privileged_obs = 3
    Cfg.env.episode_length_s = max(40.0, (args.steps + 10) * Cfg.sim.dt * Cfg.control.decimation)
    Cfg.sensors.sensor_names = list(LEGACY_POLICY_SENSOR_NAMES)
    if include_ball:
        Cfg.sensors.sensor_names.insert(0, "ObjectSensor")
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
    Cfg.sensors.privileged_sensor_names = ["BodyVelocitySensor"]
    Cfg.sensors.privileged_sensor_args = {"BodyVelocitySensor": {}}
    Cfg.noise.add_noise = False
    Cfg.sim.use_gpu_pipeline = args.device.startswith("cuda")

    Cfg.terrain.mesh_type = "plane"
    Cfg.terrain.static_friction = ground["static_friction"]
    Cfg.terrain.dynamic_friction = ground["dynamic_friction"]
    Cfg.terrain.restitution = ground["restitution"]
    Cfg.terrain.curriculum = False
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.terrain.yaw_init_range = 0.0

    Cfg.commands.command_curriculum = False
    Cfg.commands.distributional_commands = False
    Cfg.commands.heading_command = False
    Cfg.commands.num_commands = len(LEGACY_COMMAND_VALUES)
    Cfg.commands.lin_vel_x = [0.0, 0.0]
    Cfg.commands.lin_vel_y = [0.0, 0.0]
    Cfg.commands.ang_vel_yaw = [0.0, 0.0]

    for name in (
        "randomize_rigids_after_start",
        "randomize_friction",
        "randomize_friction_indep",
        "randomize_ground_friction",
        "randomize_restitution",
        "randomize_ground_restitution",
        "randomize_tile_roughness",
        "randomize_base_mass",
        "randomize_com_displacement",
        "randomize_motor_strength",
        "randomize_motor_offset",
        "randomize_Kp_factor",
        "randomize_Kd_factor",
        "randomize_gravity",
        "randomize_ball_drag",
        "randomize_ball_restitution",
        "randomize_ball_friction",
        "randomize_lag_timesteps",
        "push_robots",
    ):
        setattr(Cfg.domain_rand, name, False)
    Cfg.domain_rand.lag_timesteps = 0

    Cfg.asset.terminate_after_contacts_on = []
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_roll_pitch = False
    Cfg.normalization.clip_actions = 1.0
    Cfg.ball.ball_init_pos = list(args.ball_position)
    Cfg.ball.ball_init_rot = [0.0, 0.0, 0.0, 1.0]
    Cfg.ball.ball_init_lin_vel = [0.0, 0.0, 0.0]
    Cfg.ball.ball_init_ang_vel = [0.0, 0.0, 0.0]
    Cfg.ball.init_pos_range = [0.0, 0.0, 0.0]
    Cfg.ball.init_vel_range = [0.0, 0.0, 0.0]
    Cfg.ball.pos_reset_prob = 0.0
    Cfg.ball.vel_reset_prob = 0.0
    Cfg.ball.vision_receive_prob = 1.0
    Cfg.sim.gravity = list(profile["gravity_mps2"])
    Cfg.sim.physx.solver_type = 1 if solver["type"] == "tgs" else 0
    Cfg.sim.physx.num_position_iterations = solver["position_iterations"]
    Cfg.sim.physx.num_velocity_iterations = solver["velocity_iterations"]
    Cfg.sim.physx.contact_offset = solver["contact_offset_m"]
    Cfg.sim.physx.rest_offset = solver["rest_offset_m"]
    Cfg.sim.physx.bounce_threshold_velocity = solver["bounce_threshold_velocity_mps"]
    Cfg.sim.physx.max_depenetration_velocity = solver["max_depenetration_velocity_mps"]
    return Cfg


def _install_deterministic_command_sampler(env: object) -> None:
    """Replace curriculum sampling with the fixed legacy 15D command row."""

    import torch

    command_row = torch.tensor(LEGACY_COMMAND_VALUES, dtype=env.commands.dtype, device=env.device)

    def resample_commands(env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        env.commands[env_ids] = command_row
        for values in env.command_sums.values():
            values[env_ids] = 0.0

    env._resample_commands = resample_commands
    resample_commands(torch.arange(env.num_envs, dtype=torch.long, device=env.device))


def _install_deterministic_ball_asset_profile() -> None:
    """Make the legacy procedural sphere use explicit Preview 4 defaults."""

    from isaacgym import gymapi

    from quadruped.assets.ball import Ball

    ball_profile = make_physics_profile(include_ball=True)["ball"]

    def initialize(ball_asset: object) -> tuple[object, object]:
        options = gymapi.AssetOptions()
        options.linear_damping = ball_profile["linear_damping"]
        options.angular_damping = ball_profile["angular_damping"]
        options.max_linear_velocity = ball_profile["max_linear_velocity_mps"]
        options.max_angular_velocity = ball_profile["max_angular_velocity_radps"]
        options.enable_gyroscopic_forces = ball_profile["enable_gyroscopic_forces"]
        asset = ball_asset.env.gym.create_sphere(
            ball_asset.env.sim,
            ball_profile["radius_m"],
            options,
        )
        rigid_shape_props = ball_asset.env.gym.get_asset_rigid_shape_properties(asset)
        ball_asset.num_bodies = ball_asset.env.gym.get_asset_rigid_body_count(asset)
        return asset, rigid_shape_props

    Ball.initialize = initialize


def _enforce_actor_dynamics(env: object, include_ball: bool) -> None:
    """Remove legacy actor-creation randomization from a parity rollout."""

    profile = make_physics_profile(include_ball=include_ball)
    ground = profile["ground"]
    ball = profile["ball"]
    for env_index, env_handle in enumerate(env.envs):
        robot_handle = env.robot_actor_handles[env_index]
        robot_shapes = env.gym.get_actor_rigid_shape_properties(env_handle, robot_handle)
        for shape in robot_shapes:
            shape.friction = ground["static_friction"]
            shape.restitution = ground["restitution"]
        env.gym.set_actor_rigid_shape_properties(env_handle, robot_handle, robot_shapes)

        if not include_ball:
            continue
        ball_handle = env.object_actor_handles[env_index]
        ball_shapes = env.gym.get_actor_rigid_shape_properties(env_handle, ball_handle)
        for shape in ball_shapes:
            shape.friction = ball["static_friction"]
            shape.restitution = ball["restitution"]
        env.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shapes)

        body_props = env.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
        body_props[0].mass = ball["mass_kg"]
        env.gym.set_actor_rigid_body_properties(
            env_handle,
            ball_handle,
            body_props,
            recomputeInertia=True,
        )
        actual_mass = env.gym.get_actor_rigid_body_properties(env_handle, ball_handle)[0].mass
        if not np.isclose(actual_mass, ball["mass_kg"], rtol=1.0e-6, atol=1.0e-8):
            raise RuntimeError(
                f"Failed to set deterministic ball mass in env {env_index}: "
                f"expected {ball['mass_kg']}, got {actual_mass}"
            )


def _xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return np.concatenate((quaternion[..., 3:4], quaternion[..., :3]), axis=-1)


def _to_numpy(tensor: object) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def _vec3(value: object) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _mat33(value: object) -> list[list[float]]:
    return [_vec3(value.x), _vec3(value.y), _vec3(value.z)]


def _runtime_asset_properties(env: object) -> dict[str, object]:
    """Read back the first robot actor's effective PhysX properties."""

    env_handle = env.envs[0]
    robot_handle = env.robot_actor_handles[0]
    body_names = list(env.gym.get_actor_rigid_body_names(env_handle, robot_handle))
    body_props = env.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
    dof_names = list(env.gym.get_actor_dof_names(env_handle, robot_handle))
    dof_props = env.gym.get_actor_dof_properties(env_handle, robot_handle)
    return {
        "body_names": body_names,
        "body_mass_kg": [float(prop.mass) for prop in body_props],
        # Preview 4 exposes a COM translation and a full inertia tensor, not
        # a separate principal-axis orientation.
        "body_com_pos_b_m": [_vec3(prop.com) for prop in body_props],
        "body_inertia_kg_m2": [_mat33(prop.inertia) for prop in body_props],
        "joint_names": dof_names,
        "joint_has_limits": dof_props["hasLimits"].astype(bool).tolist(),
        "joint_pos_limits_rad": np.stack((dof_props["lower"], dof_props["upper"]), axis=-1).tolist(),
        "joint_velocity_limit_radps": dof_props["velocity"].astype(float).tolist(),
        "joint_effort_limit_nm": dof_props["effort"].astype(float).tolist(),
        "joint_drive_mode": dof_props["driveMode"].astype(int).tolist(),
        "joint_drive_stiffness": dof_props["stiffness"].astype(float).tolist(),
        "joint_drive_damping": dof_props["damping"].astype(float).tolist(),
        "joint_friction": dof_props["friction"].astype(float).tolist(),
        "joint_armature": dof_props["armature"].astype(float).tolist(),
    }


def _write_deterministic_state(env: object, args: argparse.Namespace) -> None:
    import torch
    from isaacgym import gymtorch
    from isaacgym.torch_utils import quat_rotate_inverse

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    robot_actor_ids = env.robot_actor_idxs[env_ids]
    env.root_states[robot_actor_ids] = env.base_init_state
    env.root_states[robot_actor_ids, :3] += env.env_origins[env_ids]
    env.root_states[robot_actor_ids, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0],
        device=env.device,
    )
    env.root_states[robot_actor_ids, 7:] = 0.0
    actor_ids = [robot_actor_ids]

    if env.cfg.env.add_balls:
        ball_actor_ids = env.object_actor_idxs[env_ids]
        env.root_states[ball_actor_ids] = env.object_init_state
        env.root_states[ball_actor_ids, :3] = env.env_origins[env_ids] + torch.tensor(
            args.ball_position,
            device=env.device,
        )
        env.root_states[ball_actor_ids, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=env.device,
        )
        env.root_states[ball_actor_ids, 7:] = 0.0
        actor_ids.append(ball_actor_ids)

    env.dof_pos[:] = env.default_dof_pos
    env.dof_vel.zero_()
    actor_ids_int32 = torch.cat(actor_ids).to(dtype=torch.int32)
    robot_actor_ids_int32 = robot_actor_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(actor_ids_int32),
        len(actor_ids_int32),
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(robot_actor_ids_int32),
        len(robot_actor_ids_int32),
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)

    env.base_pos[:] = env.root_states[robot_actor_ids, :3]
    env.base_quat[:] = env.root_states[robot_actor_ids, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[robot_actor_ids, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[robot_actor_ids, 10:13])
    env.joint_pos_target[:] = env.default_dof_pos
    env.last_joint_pos_target[:] = env.default_dof_pos
    env.last_last_joint_pos_target[:] = env.default_dof_pos
    env.actions.zero_()
    env.last_actions.zero_()
    env.last_last_actions.zero_()
    env.torques.zero_()
    env.commands[:] = torch.tensor(
        LEGACY_COMMAND_VALUES,
        dtype=env.commands.dtype,
        device=env.device,
    )
    env.episode_length_buf.zero_()
    env.reset_buf.zero_()
    env.common_step_counter = 0
    if env.cfg.env.add_balls:
        env.object_pos_world_frame[:] = env.root_states[env.object_actor_idxs, :3]
        env.object_lin_vel[:] = env.root_states[env.object_actor_idxs, 7:10]
        env.object_ang_vel[:] = env.root_states[env.object_actor_idxs, 10:13]


def _snapshot(env: object, foot_ids: list[int], args: argparse.Namespace) -> dict[str, np.ndarray]:
    robot_root = env.root_states[env.robot_actor_idxs]
    foot_force = env.contact_forces[:, foot_ids]
    state = {
        "root_pos_w": _to_numpy(robot_root[:, :3]),
        "root_quat_w": _xyzw_to_wxyz(_to_numpy(robot_root[:, 3:7])),
        "root_lin_vel_w": _to_numpy(robot_root[:, 7:10]),
        "root_ang_vel_w": _to_numpy(robot_root[:, 10:13]),
        "root_lin_vel_b": _to_numpy(env.base_lin_vel),
        "root_ang_vel_b": _to_numpy(env.base_ang_vel),
        "joint_pos": _to_numpy(env.dof_pos),
        "joint_vel": _to_numpy(env.dof_vel),
        "joint_pos_target": _to_numpy(env.joint_pos_target),
        "applied_torque": _to_numpy(env.torques),
        "foot_pos_w": _to_numpy(env.rigid_body_state[:, foot_ids, :3]),
        "foot_force_w": _to_numpy(foot_force),
        "foot_contact": _to_numpy(foot_force[..., 2] > args.contact_threshold_n),
    }
    if env.cfg.env.add_balls:
        ball_root = env.root_states[env.object_actor_idxs]
        state.update(
            {
                "ball_pos_w": _to_numpy(ball_root[:, :3]),
                "ball_quat_w": _xyzw_to_wxyz(_to_numpy(ball_root[:, 3:7])),
                "ball_lin_vel_w": _to_numpy(ball_root[:, 7:10]),
                "ball_ang_vel_w": _to_numpy(ball_root[:, 10:13]),
            }
        )
    return state


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num_envs and --steps must be positive")
    if args.contact_threshold_n < 0.0:
        raise ValueError("--contact_threshold_n must be non-negative")

    import isaacgym
    import torch

    from quadruped.envs.as2.velocity_tracking import VelocityTrackingEasyEnv

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = _configure_legacy_env(args)
    if args.task == "dribble":
        _install_deterministic_ball_asset_profile()
    env = VelocityTrackingEasyEnv(sim_device=args.device, headless=True, cfg=cfg)
    try:
        _enforce_actor_dynamics(env, include_ball=args.task == "dribble")
        _install_deterministic_command_sampler(env)
        env.reset()
        _write_deterministic_state(env, args)
        joint_names = list(env.dof_names)
        action_names = joint_names[: env.num_actions]
        if set(action_names) != set(joint_names):
            raise RuntimeError("Legacy parity recording requires one action per robot joint")
        body_names = list(env.gym.get_asset_rigid_body_names(env.robot_asset))
        if cfg.robot.name != "as2" or "base_link" not in body_names:
            raise RuntimeError(
                "Legacy parity recorder did not load the AS2 asset: "
                f"robot.name={cfg.robot.name!r}, body_names={body_names}"
            )
        foot_ids = [int(index) for index in env.feet_indices.detach().cpu().tolist()]
        foot_names = [body_names[index] for index in foot_ids]

        step_dt = float(env.dt)
        excitation = ExcitationConfig(
            kind=args.excitation,
            amplitude=args.amplitude,
            frequency_hz=args.frequency_hz,
            phase_stride_rad=args.phase_stride_rad,
        )
        actions = generate_action_sequence(
            args.steps,
            step_dt,
            env.num_envs,
            action_names,
            excitation,
        )
        frames: dict[str, list[np.ndarray]] = {}
        for name, value in _snapshot(env, foot_ids, args).items():
            frames[name] = [value]
        done_frames = []
        policy_contract = make_legacy_policy_contract(include_ball=args.task == "dribble")
        history_width = int(policy_contract["history_dim"])
        observation_width = int(policy_contract["observation_dim"])
        policy_history = torch.zeros(env.num_envs, history_width, device=env.device)
        policy_observations = []
        policy_histories = []

        with torch.inference_mode():
            for step_index in range(args.steps):
                action = torch.from_numpy(actions[step_index]).to(env.device)
                observation, _, done, _ = env.step(action)
                if observation.shape != (env.num_envs, observation_width):
                    raise RuntimeError(
                        f"Legacy policy observation has shape {tuple(observation.shape)}, "
                        f"expected ({env.num_envs}, {observation_width})"
                    )
                policy_history = torch.cat(
                    (policy_history[:, observation_width:], observation),
                    dim=-1,
                )
                policy_observations.append(_to_numpy(observation))
                policy_histories.append(_to_numpy(policy_history))
                done_numpy = _to_numpy(done.to(dtype=torch.bool)).astype(np.bool_, copy=False)
                done_frames.append(done_numpy)
                if done.any() and not args.allow_resets:
                    raise RuntimeError(
                        f"Environment reset at rollout step {step_index}; rerun with fewer steps or --allow_resets"
                    )
                for name, value in _snapshot(env, foot_ids, args).items():
                    frames[name].append(value)

        arrays = {name: np.stack(values, axis=0) for name, values in frames.items()}
        arrays.update(
            {
                "time_s": np.arange(args.steps + 1, dtype=np.float64) * step_dt,
                "action": actions,
                "done": np.stack(done_frames, axis=0),
                "env_origin_w": _to_numpy(env.env_origins),
                "legacy_policy_observation": np.stack(policy_observations, axis=0),
                "legacy_policy_history": np.stack(policy_histories, axis=0),
            }
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "source": "isaacgym",
            "source_version": getattr(isaacgym, "__version__", "Preview 4"),
            "task": TASK_NAMES[args.task],
            "step_dt": step_dt,
            "physics_dt": float(cfg.sim.dt),
            "decimation": int(cfg.control.decimation),
            "joint_names": joint_names,
            "action_names": action_names,
            "body_names": body_names,
            "foot_names": foot_names,
            "quaternion_order": "wxyz",
            "contact_force_semantics": "Isaac Gym net contact force including normal and friction components",
            "contact_threshold_n": args.contact_threshold_n,
            "physics_profile": make_physics_profile(include_ball=args.task == "dribble"),
            "policy_contract": policy_contract,
            "runtime_asset_properties": _runtime_asset_properties(env),
            "excitation": asdict(excitation),
            "terminations_disabled": True,
            "deterministic_reset": True,
        }
        output_path = save_rollout(args.output, metadata, arrays)
        print(f"[INFO] Saved Isaac Gym parity rollout to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
