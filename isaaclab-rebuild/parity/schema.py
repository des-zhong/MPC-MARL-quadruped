"""Versioned NumPy archive format shared by the Gym and Lab recorders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .physics_profile import validate_physics_profile
from .policy_contract import validate_legacy_policy_contract

SCHEMA_VERSION = 2

REQUIRED_METADATA = {
    "schema_version",
    "source",
    "task",
    "step_dt",
    "physics_dt",
    "decimation",
    "joint_names",
    "action_names",
    "body_names",
    "foot_names",
    "quaternion_order",
    "contact_force_semantics",
    "physics_profile",
}

ROOT_VECTOR_SIGNALS = (
    "root_pos_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "root_lin_vel_b",
    "root_ang_vel_b",
)
JOINT_SIGNALS = (
    "joint_pos",
    "joint_vel",
    "joint_pos_target",
    "applied_torque",
)
BALL_SIGNALS = (
    "ball_pos_w",
    "ball_quat_w",
    "ball_lin_vel_w",
    "ball_ang_vel_w",
)
POLICY_CONTRACT_SIGNALS = (
    "legacy_policy_observation",
    "legacy_policy_history",
)


def _string_list(metadata: Mapping[str, Any], key: str) -> list[str]:
    value = metadata[key]
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"metadata[{key!r}] must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"metadata[{key!r}] contains duplicate names")
    return value


def _require_shape(name: str, array: np.ndarray, expected: tuple[int, ...]) -> None:
    if array.shape != expected:
        raise ValueError(f"{name} has shape {array.shape}, expected {expected}")


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric, got dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def validate_rollout(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    """Validate metadata, signal names, dimensions, time base, and finite values."""

    missing_metadata = sorted(REQUIRED_METADATA.difference(metadata))
    if missing_metadata:
        raise ValueError(f"rollout metadata is missing keys: {missing_metadata}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema version {metadata['schema_version']!r}; expected {SCHEMA_VERSION}"
        )
    if metadata["source"] not in {"isaacgym", "isaaclab"}:
        raise ValueError("metadata['source'] must be 'isaacgym' or 'isaaclab'")
    if not isinstance(metadata["task"], str) or not metadata["task"]:
        raise ValueError("metadata['task'] must be a non-empty string")
    if metadata["quaternion_order"] != "wxyz":
        raise ValueError("the parity schema requires wxyz quaternion ordering")

    step_dt = float(metadata["step_dt"])
    physics_dt = float(metadata["physics_dt"])
    decimation = int(metadata["decimation"])
    if step_dt <= 0.0 or physics_dt <= 0.0 or decimation <= 0:
        raise ValueError("step_dt, physics_dt, and decimation must be positive")
    if not np.isclose(step_dt, physics_dt * decimation, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError("step_dt must equal physics_dt * decimation")

    joint_names = _string_list(metadata, "joint_names")
    action_names = _string_list(metadata, "action_names")
    _string_list(metadata, "body_names")
    foot_names = _string_list(metadata, "foot_names")
    if set(action_names) != set(joint_names):
        raise ValueError("action_names and joint_names must describe the same named joints")
    if not isinstance(metadata["contact_force_semantics"], str) or not metadata["contact_force_semantics"]:
        raise ValueError("contact_force_semantics must be a non-empty string")
    validate_physics_profile(metadata["physics_profile"])

    required_arrays = {
        "time_s",
        "action",
        "done",
        "env_origin_w",
        "root_quat_w",
        "foot_pos_w",
        "foot_force_w",
        "foot_contact",
        *ROOT_VECTOR_SIGNALS,
        *JOINT_SIGNALS,
    }
    missing_arrays = sorted(required_arrays.difference(arrays))
    if missing_arrays:
        raise ValueError(f"rollout archive is missing arrays: {missing_arrays}")

    time_s = np.asarray(arrays["time_s"])
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must have shape (T+1,) with at least two samples")
    _require_finite("time_s", time_s)
    if not np.isclose(time_s[0], 0.0, atol=1.0e-9):
        raise ValueError("time_s must start at zero")
    if not np.allclose(np.diff(time_s), step_dt, rtol=1.0e-5, atol=1.0e-8):
        raise ValueError("time_s increments do not match metadata step_dt")

    frames = time_s.size
    steps = frames - 1
    action = np.asarray(arrays["action"])
    if action.ndim != 3:
        raise ValueError("action must have shape (T, N, A)")
    num_envs = action.shape[1]
    num_joints = len(joint_names)
    num_feet = len(foot_names)
    _require_shape("action", action, (steps, num_envs, len(action_names)))
    _require_finite("action", action)

    done = np.asarray(arrays["done"])
    _require_shape("done", done, (steps, num_envs))
    if done.dtype != np.bool_:
        raise ValueError("done must use boolean dtype")

    env_origin = np.asarray(arrays["env_origin_w"])
    _require_shape("env_origin_w", env_origin, (num_envs, 3))
    _require_finite("env_origin_w", env_origin)

    for name in ROOT_VECTOR_SIGNALS:
        value = np.asarray(arrays[name])
        _require_shape(name, value, (frames, num_envs, 3))
        _require_finite(name, value)

    root_quat = np.asarray(arrays["root_quat_w"])
    _require_shape("root_quat_w", root_quat, (frames, num_envs, 4))
    _require_finite("root_quat_w", root_quat)
    if np.any(np.linalg.norm(root_quat, axis=-1) < 1.0e-8):
        raise ValueError("root_quat_w contains a zero-norm quaternion")

    for name in JOINT_SIGNALS:
        value = np.asarray(arrays[name])
        _require_shape(name, value, (frames, num_envs, num_joints))
        _require_finite(name, value)

    foot_pos = np.asarray(arrays["foot_pos_w"])
    foot_force = np.asarray(arrays["foot_force_w"])
    foot_contact = np.asarray(arrays["foot_contact"])
    _require_shape("foot_pos_w", foot_pos, (frames, num_envs, num_feet, 3))
    _require_shape("foot_force_w", foot_force, (frames, num_envs, num_feet, 3))
    _require_shape("foot_contact", foot_contact, (frames, num_envs, num_feet))
    _require_finite("foot_pos_w", foot_pos)
    _require_finite("foot_force_w", foot_force)
    if foot_contact.dtype != np.bool_:
        raise ValueError("foot_contact must use boolean dtype")

    present_ball_signals = [name for name in BALL_SIGNALS if name in arrays]
    if present_ball_signals and len(present_ball_signals) != len(BALL_SIGNALS):
        missing_ball = sorted(set(BALL_SIGNALS).difference(present_ball_signals))
        raise ValueError(f"ball rollout is incomplete; missing arrays: {missing_ball}")
    profile_has_ball = metadata["physics_profile"]["ball"] is not None
    if bool(present_ball_signals) != profile_has_ball:
        raise ValueError("physics_profile.ball must agree with the presence of ball state arrays")
    for name in present_ball_signals:
        width = 4 if "quat" in name else 3
        value = np.asarray(arrays[name])
        _require_shape(name, value, (frames, num_envs, width))
        _require_finite(name, value)
        if width == 4 and np.any(np.linalg.norm(value, axis=-1) < 1.0e-8):
            raise ValueError(f"{name} contains a zero-norm quaternion")

    present_policy_signals = [name for name in POLICY_CONTRACT_SIGNALS if name in arrays]
    has_policy_metadata = "policy_contract" in metadata
    if present_policy_signals and len(present_policy_signals) != len(POLICY_CONTRACT_SIGNALS):
        missing_policy = sorted(set(POLICY_CONTRACT_SIGNALS).difference(present_policy_signals))
        raise ValueError(f"policy-contract rollout is incomplete; missing arrays: {missing_policy}")
    if bool(present_policy_signals) != has_policy_metadata:
        raise ValueError("policy_contract metadata must agree with policy-contract arrays")
    if has_policy_metadata:
        validate_legacy_policy_contract(metadata["policy_contract"])
        contract = metadata["policy_contract"]
        observation_dim = int(contract["observation_dim"])
        history_dim = int(contract["history_dim"])
        observation = np.asarray(arrays["legacy_policy_observation"])
        history = np.asarray(arrays["legacy_policy_history"])
        _require_shape(
            "legacy_policy_observation",
            observation,
            (steps, num_envs, observation_dim),
        )
        _require_shape(
            "legacy_policy_history",
            history,
            (steps, num_envs, history_dim),
        )
        _require_finite("legacy_policy_observation", observation)
        _require_finite("legacy_policy_history", history)


def save_rollout(
    path: str | Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> Path:
    """Validate and save a compressed rollout without pickle-dependent values."""

    output_path = Path(path)
    if output_path.suffix != ".npz":
        raise ValueError("rollout path must end in .npz")
    normalized_arrays = {name: np.asarray(value) for name, value in arrays.items()}
    validate_rollout(metadata, normalized_arrays)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
    np.savez_compressed(output_path, metadata_json=np.asarray(metadata_json), **normalized_arrays)
    return output_path


def load_rollout(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load and validate a rollout archive with pickle explicitly disabled."""

    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("rollout archive does not contain metadata_json")
        raw_metadata = archive["metadata_json"]
        if raw_metadata.ndim != 0:
            raise ValueError("metadata_json must be a scalar string")
        metadata = json.loads(str(raw_metadata.item()))
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to a JSON object")
    validate_rollout(metadata, arrays)
    return metadata, arrays
