"""Small deterministic rollout fixtures used by simulator-free tests."""

from __future__ import annotations

import copy

import numpy as np

from parity.physics_profile import make_physics_profile
from parity.policy_contract import make_legacy_policy_contract
from parity.schema import SCHEMA_VERSION


def make_rollout(source: str = "isaacgym") -> tuple[dict, dict[str, np.ndarray]]:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "task": "test-task",
        "step_dt": 0.02,
        "physics_dt": 0.005,
        "decimation": 4,
        "joint_names": ["joint_b", "joint_a"],
        "action_names": ["joint_b", "joint_a"],
        "body_names": ["base", "right_foot", "left_foot"],
        "foot_names": ["right_foot", "left_foot"],
        "quaternion_order": "wxyz",
        "contact_force_semantics": "full net contact force",
        "physics_profile": make_physics_profile(include_ball=False),
    }
    frames = 4
    steps = frames - 1
    num_envs = 2
    num_joints = 2
    num_feet = 2
    time_s = np.arange(frames, dtype=np.float64) * metadata["step_dt"]
    root_pos = np.zeros((frames, num_envs, 3), dtype=np.float32)
    root_pos[..., 2] = 0.34
    root_pos[:, 1, 0] = 10.0
    root_quat = np.zeros((frames, num_envs, 4), dtype=np.float32)
    root_quat[..., 0] = 1.0
    joint_pos = np.arange(frames * num_envs * num_joints, dtype=np.float32).reshape(
        frames, num_envs, num_joints
    )
    foot_pos = np.zeros((frames, num_envs, num_feet, 3), dtype=np.float32)
    foot_pos[..., 2] = 0.02
    foot_pos[:, 1, :, 0] = 10.0
    foot_force = np.zeros_like(foot_pos)
    foot_force[..., 2] = np.asarray([20.0, 25.0], dtype=np.float32)
    arrays = {
        "time_s": time_s,
        "action": np.arange(steps * num_envs * num_joints, dtype=np.float32).reshape(
            steps, num_envs, num_joints
        ),
        "done": np.zeros((steps, num_envs), dtype=np.bool_),
        "env_origin_w": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "root_lin_vel_w": np.zeros((frames, num_envs, 3), dtype=np.float32),
        "root_ang_vel_w": np.zeros((frames, num_envs, 3), dtype=np.float32),
        "root_lin_vel_b": np.zeros((frames, num_envs, 3), dtype=np.float32),
        "root_ang_vel_b": np.zeros((frames, num_envs, 3), dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": joint_pos * 0.1,
        "joint_pos_target": joint_pos * 0.2,
        "applied_torque": joint_pos * 0.3,
        "foot_pos_w": foot_pos,
        "foot_force_w": foot_force,
        "foot_contact": foot_force[..., 2] > 1.0,
    }
    return metadata, arrays


def clone_rollout(metadata: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    return copy.deepcopy(metadata), {name: value.copy() for name, value in arrays.items()}


def add_policy_contract(metadata: dict, arrays: dict[str, np.ndarray], include_ball: bool = False) -> None:
    """Add deterministic step-aligned observation/history signals to a fixture."""

    contract = make_legacy_policy_contract(include_ball=include_ball)
    steps, num_envs = arrays["action"].shape[:2]
    observation_dim = contract["observation_dim"]
    history_length = contract["history_length"]
    observation = np.arange(
        steps * num_envs * observation_dim,
        dtype=np.float32,
    ).reshape(steps, num_envs, observation_dim) / 100.0
    history = np.zeros((steps, num_envs, contract["history_dim"]), dtype=np.float32)
    for step in range(steps):
        start = max(0, step - history_length + 1)
        recent = observation[start : step + 1].transpose(1, 0, 2).reshape(num_envs, -1)
        history[step, :, -recent.shape[1] :] = recent
    metadata["policy_contract"] = contract
    arrays["legacy_policy_observation"] = observation
    arrays["legacy_policy_history"] = history
