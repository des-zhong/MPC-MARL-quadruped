"""Record a deterministic manager-based Isaac Lab rollout for physics parity."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Velocity-Flat-Play-v0")
parser.add_argument("--output", type=Path, required=True)
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
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401
import isaaclab  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from parity.excitation import ExcitationConfig, generate_action_sequence  # noqa: E402
from parity.physics_profile import make_physics_profile  # noqa: E402
from parity.policy_contract import make_legacy_policy_contract  # noqa: E402
from parity.runtime_profiles import configure_isaaclab_env_for_parity  # noqa: E402
from parity.schema import SCHEMA_VERSION, save_rollout  # noqa: E402


def _configure_for_parity(env_cfg: object) -> None:
    configure_isaaclab_env_for_parity(env_cfg, seed=args_cli.seed, num_envs=args_cli.num_envs)


def _verify_collider_offsets(asset: object, contact_offset: float, rest_offset: float) -> None:
    """Fail parity recording when the active PhysX shapes do not match the profile."""

    actual_contact = asset.root_physx_view.get_contact_offsets()
    actual_rest = asset.root_physx_view.get_rest_offsets()
    expected_contact = torch.full_like(actual_contact, float(contact_offset))
    expected_rest = torch.full_like(actual_rest, float(rest_offset))
    if not torch.allclose(actual_contact, expected_contact, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            f"PhysX contact offsets for '{asset.cfg.prim_path}' do not match {contact_offset}: "
            f"observed range [{actual_contact.min().item()}, {actual_contact.max().item()}]"
        )
    if not torch.allclose(actual_rest, expected_rest, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            f"PhysX rest offsets for '{asset.cfg.prim_path}' do not match {rest_offset}: "
            f"observed range [{actual_rest.min().item()}, {actual_rest.max().item()}]"
        )


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def _first_instance(tensor: torch.Tensor) -> list:
    return _to_numpy(tensor[0]).tolist()


def _runtime_asset_properties(robot: object) -> dict[str, object]:
    """Read back effective PhysX articulation properties for the first environment."""

    view = robot.root_physx_view
    friction_properties = view.get_dof_friction_properties()
    result = {
        "body_names": list(robot.body_names),
        "body_mass_kg": _first_instance(view.get_masses()),
        "body_com_pose_b_xyzw": _first_instance(view.get_coms()),
        "body_inertia_kg_m2_flat": _first_instance(view.get_inertias()),
        "joint_names": list(robot.joint_names),
        "joint_pos_limits_rad": _first_instance(view.get_dof_limits()),
        "joint_velocity_limit_radps": _first_instance(view.get_dof_max_velocities()),
        "joint_effort_limit_nm": _first_instance(view.get_dof_max_forces()),
        "joint_drive_stiffness": _first_instance(view.get_dof_stiffnesses()),
        "joint_drive_damping": _first_instance(view.get_dof_dampings()),
        "joint_armature": _first_instance(view.get_dof_armatures()),
        "joint_friction_properties": _first_instance(friction_properties),
        "actuator_models": {
            name: type(actuator).__name__ for name, actuator in robot.actuators.items()
        },
    }
    return result


def _action_names(raw_env: object) -> list[str]:
    if raw_env.action_manager.active_terms != ["joint_pos"]:
        raise RuntimeError(
            "Parity recording currently requires one joint_pos action term; "
            f"found {raw_env.action_manager.active_terms}"
        )
    descriptor = raw_env.action_manager.get_term("joint_pos").IO_descriptor
    return list(descriptor.joint_names)


def _foot_indices(robot: object, contact_sensor: object) -> tuple[list[str], list[int], list[int]]:
    foot_names = [name for name in robot.body_names if name.endswith("_foot")]
    if not foot_names:
        raise RuntimeError(f"No foot bodies found in {robot.body_names}")
    robot_index_by_name = {name: index for index, name in enumerate(robot.body_names)}
    sensor_index_by_name = {name: index for index, name in enumerate(contact_sensor.body_names)}
    missing_sensor_feet = [name for name in foot_names if name not in sensor_index_by_name]
    if missing_sensor_feet:
        raise RuntimeError(f"Contact sensor is missing foot bodies: {missing_sensor_feet}")
    return (
        foot_names,
        [robot_index_by_name[name] for name in foot_names],
        [sensor_index_by_name[name] for name in foot_names],
    )


def _write_deterministic_state(raw_env: object, ball: object | None) -> None:
    robot = raw_env.scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += raw_env.scene.env_origins
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))

    if ball is not None:
        ball_state = ball.data.default_root_state.clone()
        ball_state[:, :3] = raw_env.scene.env_origins + torch.tensor(
            args_cli.ball_position,
            device=raw_env.device,
            dtype=ball_state.dtype,
        )
        ball_state[:, 3:7] = 0.0
        ball_state[:, 3] = 1.0
        ball_state[:, 7:] = 0.0
        ball.write_root_state_to_sim(ball_state)

    zero_action = torch.zeros(
        (raw_env.num_envs, raw_env.action_manager.total_action_dim),
        device=raw_env.device,
    )
    raw_env.action_manager.process_action(zero_action)
    raw_env.action_manager.apply_action()
    raw_env.scene.write_data_to_sim()
    raw_env.sim.forward()
    raw_env.scene.update(dt=0.0)
    raw_env.episode_length_buf.zero_()
    raw_env.common_step_counter = 0


def _snapshot(
    raw_env: object,
    ball: object | None,
    robot_foot_ids: list[int],
    sensor_foot_ids: list[int],
) -> dict[str, np.ndarray]:
    robot = raw_env.scene["robot"]
    contact_sensor = raw_env.scene["contact_forces"]
    foot_force = contact_sensor.data.net_forces_w[:, sensor_foot_ids]
    state = {
        "root_pos_w": _to_numpy(robot.data.root_pos_w),
        "root_quat_w": _to_numpy(robot.data.root_quat_w),
        "root_lin_vel_w": _to_numpy(robot.data.root_lin_vel_w),
        "root_ang_vel_w": _to_numpy(robot.data.root_ang_vel_w),
        "root_lin_vel_b": _to_numpy(robot.data.root_lin_vel_b),
        "root_ang_vel_b": _to_numpy(robot.data.root_ang_vel_b),
        "joint_pos": _to_numpy(robot.data.joint_pos),
        "joint_vel": _to_numpy(robot.data.joint_vel),
        "joint_pos_target": _to_numpy(robot.data.joint_pos_target),
        "applied_torque": _to_numpy(robot.data.applied_torque),
        "foot_pos_w": _to_numpy(robot.data.body_pos_w[:, robot_foot_ids]),
        "foot_force_w": _to_numpy(foot_force),
        "foot_contact": _to_numpy(foot_force[..., 2] > args_cli.contact_threshold_n),
    }
    if ball is not None:
        state.update(
            {
                "ball_pos_w": _to_numpy(ball.data.root_pos_w),
                "ball_quat_w": _to_numpy(ball.data.root_quat_w),
                "ball_lin_vel_w": _to_numpy(ball.data.root_lin_vel_w),
                "ball_ang_vel_w": _to_numpy(ball.data.root_ang_vel_w),
            }
        )
    return state


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("--num_envs and --steps must be positive")
    if args_cli.contact_threshold_n < 0.0:
        raise ValueError("--contact_threshold_n must be non-negative")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_for_parity(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    try:
        env.reset(seed=args_cli.seed)
        robot = raw_env.scene["robot"]
        ball = raw_env.scene.rigid_objects.get("ball")
        solver = make_physics_profile(include_ball=ball is not None)["solver"]
        _verify_collider_offsets(
            robot,
            contact_offset=solver["contact_offset_m"],
            rest_offset=solver["rest_offset_m"],
        )
        if ball is not None:
            _verify_collider_offsets(
                ball,
                contact_offset=solver["contact_offset_m"],
                rest_offset=solver["rest_offset_m"],
            )
        action_names = _action_names(raw_env)
        joint_names = list(robot.joint_names)
        if set(action_names) != set(joint_names):
            raise RuntimeError("The joint_pos action term does not cover exactly the robot joints")
        foot_names, robot_foot_ids, sensor_foot_ids = _foot_indices(
            robot,
            raw_env.scene["contact_forces"],
        )
        _write_deterministic_state(raw_env, ball)

        excitation = ExcitationConfig(
            kind=args_cli.excitation,
            amplitude=args_cli.amplitude,
            frequency_hz=args_cli.frequency_hz,
            phase_stride_rad=args_cli.phase_stride_rad,
        )
        actions = generate_action_sequence(
            args_cli.steps,
            raw_env.step_dt,
            raw_env.num_envs,
            action_names,
            excitation,
        )
        frames: dict[str, list[np.ndarray]] = {}
        for name, value in _snapshot(raw_env, ball, robot_foot_ids, sensor_foot_ids).items():
            frames[name] = [value]
        done_frames = []
        policy_contract = make_legacy_policy_contract(include_ball=ball is not None)
        observation_width = int(policy_contract["observation_dim"])
        history_width = int(policy_contract["history_dim"])
        policy_observations = []
        policy_histories = []

        with torch.inference_mode():
            for step_index in range(args_cli.steps):
                action = torch.from_numpy(actions[step_index]).to(raw_env.device)
                observations, _, terminated, truncated, _ = env.step(action)
                policy_observation = observations["legacy_policy"]
                policy_history = observations["legacy_history"]
                if policy_observation.shape != (raw_env.num_envs, observation_width):
                    raise RuntimeError(
                        f"Isaac Lab legacy_policy has shape {tuple(policy_observation.shape)}, "
                        f"expected ({raw_env.num_envs}, {observation_width})"
                    )
                if policy_history.shape != (raw_env.num_envs, history_width):
                    raise RuntimeError(
                        f"Isaac Lab legacy_history has shape {tuple(policy_history.shape)}, "
                        f"expected ({raw_env.num_envs}, {history_width})"
                    )
                policy_observations.append(_to_numpy(policy_observation))
                policy_histories.append(_to_numpy(policy_history))
                done = torch.logical_or(terminated, truncated)
                done_numpy = _to_numpy(done).astype(np.bool_, copy=False)
                done_frames.append(done_numpy)
                if done.any() and not args_cli.allow_resets:
                    raise RuntimeError(
                        f"Environment reset at rollout step {step_index}; rerun with fewer steps or --allow_resets"
                    )
                for name, value in _snapshot(raw_env, ball, robot_foot_ids, sensor_foot_ids).items():
                    frames[name].append(value)

        arrays = {name: np.stack(values, axis=0) for name, values in frames.items()}
        arrays.update(
            {
                "time_s": np.arange(args_cli.steps + 1, dtype=np.float64) * raw_env.step_dt,
                "action": actions,
                "done": np.stack(done_frames, axis=0),
                "env_origin_w": _to_numpy(raw_env.scene.env_origins),
                "legacy_policy_observation": np.stack(policy_observations, axis=0),
                "legacy_policy_history": np.stack(policy_histories, axis=0),
            }
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "source": "isaaclab",
            "source_version": getattr(isaaclab, "__version__", "unknown"),
            "task": args_cli.task,
            "step_dt": float(raw_env.step_dt),
            "physics_dt": float(raw_env.physics_dt),
            "decimation": int(raw_env.cfg.decimation),
            "joint_names": joint_names,
            "action_names": action_names,
            "body_names": list(robot.body_names),
            "foot_names": foot_names,
            "quaternion_order": "wxyz",
            "contact_force_semantics": "Isaac Lab ContactSensor net normal contact force in world frame",
            "contact_threshold_n": args_cli.contact_threshold_n,
            "physics_profile": make_physics_profile(include_ball=ball is not None),
            "policy_contract": policy_contract,
            "runtime_asset_properties": _runtime_asset_properties(robot),
            "excitation": asdict(excitation),
            "terminations_disabled": True,
            "deterministic_reset": True,
        }
        output_path = save_rollout(args_cli.output, metadata, arrays)
        print(f"[INFO] Saved Isaac Lab parity rollout to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
