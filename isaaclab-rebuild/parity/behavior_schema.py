"""Versioned trace format for frozen low-level skill behavior parity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .behavior_excitation import SKILL_NAMES
from .policy_contract import validate_legacy_policy_contract


BEHAVIOR_SCHEMA_VERSION = 1
REQUIRED_METADATA = {
    "schema_version",
    "source",
    "task",
    "step_dt",
    "physics_dt",
    "decimation",
    "joint_names",
    "skill_names",
    "policy_contract",
    "checkpoint_contract",
    "action_history_semantics",
}
REQUIRED_ARRAYS = {
    "time_s",
    "coordinator_action",
    "skill_id",
    "skill_command",
    "policy_observation",
    "policy_history",
    "policy_action",
    "processed_joint_target",
    "done",
    "env_origin_w",
    "root_pos_w",
    "root_quat_w",
    "joint_pos",
    "ball_pos_w",
}


def validate_behavior_trace(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    missing_metadata = sorted(REQUIRED_METADATA.difference(metadata))
    if missing_metadata:
        raise ValueError(f"behavior metadata is missing keys: {missing_metadata}")
    if metadata["schema_version"] != BEHAVIOR_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported behavior schema {metadata['schema_version']!r}; expected {BEHAVIOR_SCHEMA_VERSION}"
        )
    if metadata["source"] not in {"isaacgym", "isaaclab"}:
        raise ValueError("behavior source must be 'isaacgym' or 'isaaclab'")
    if list(metadata["skill_names"]) != list(SKILL_NAMES):
        raise ValueError(f"skill_names must be {list(SKILL_NAMES)}")
    joint_names = _unique_string_list(metadata, "joint_names")
    validate_legacy_policy_contract(metadata["policy_contract"])
    contract = metadata["policy_contract"]
    if not contract["include_ball"]:
        raise ValueError("frozen football behavior traces require the 75D ball policy contract")
    _validate_checkpoint_contract(metadata["checkpoint_contract"])
    if not isinstance(metadata["action_history_semantics"], str) or not metadata["action_history_semantics"]:
        raise ValueError("action_history_semantics must be a non-empty string")
    if "contract_horizon_steps" in metadata and int(metadata["contract_horizon_steps"]) <= 0:
        raise ValueError("contract_horizon_steps must be positive")

    step_dt = float(metadata["step_dt"])
    physics_dt = float(metadata["physics_dt"])
    decimation = int(metadata["decimation"])
    if step_dt <= 0.0 or physics_dt <= 0.0 or decimation <= 0:
        raise ValueError("step_dt, physics_dt, and decimation must be positive")
    if not np.isclose(step_dt, physics_dt * decimation, rtol=1.0e-6, atol=1.0e-9):
        raise ValueError("step_dt must equal physics_dt * decimation")

    missing_arrays = sorted(REQUIRED_ARRAYS.difference(arrays))
    if missing_arrays:
        raise ValueError(f"behavior archive is missing arrays: {missing_arrays}")
    time_s = np.asarray(arrays["time_s"])
    if time_s.ndim != 1 or time_s.size < 2:
        raise ValueError("time_s must have shape (T+1,) with at least two values")
    _finite("time_s", time_s)
    if not np.isclose(time_s[0], 0.0, atol=1.0e-9):
        raise ValueError("time_s must start at zero")
    if not np.allclose(np.diff(time_s), step_dt, rtol=1.0e-5, atol=1.0e-8):
        raise ValueError("time_s increments do not match step_dt")

    steps = time_s.size - 1
    coordinator = np.asarray(arrays["coordinator_action"])
    if coordinator.ndim != 3 or coordinator.shape[0] != steps or coordinator.shape[2] != 6:
        raise ValueError("coordinator_action must have shape (T,N,6)")
    num_envs = coordinator.shape[1]
    num_joints = len(joint_names)
    observation_dim = int(contract["observation_dim"])
    history_dim = int(contract["history_dim"])
    expected = {
        "coordinator_action": (steps, num_envs, 6),
        "skill_id": (steps, num_envs),
        "skill_command": (steps, num_envs, 3),
        "policy_observation": (steps, num_envs, observation_dim),
        "policy_history": (steps, num_envs, history_dim),
        "policy_action": (steps, num_envs, num_joints),
        "processed_joint_target": (steps, num_envs, num_joints),
        "done": (steps, num_envs),
        "env_origin_w": (num_envs, 3),
        "root_pos_w": (steps + 1, num_envs, 3),
        "root_quat_w": (steps + 1, num_envs, 4),
        "joint_pos": (steps + 1, num_envs, num_joints),
        "ball_pos_w": (steps + 1, num_envs, 3),
    }
    for name, shape in expected.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        if name == "done":
            if value.dtype != np.bool_:
                raise ValueError("done must use boolean dtype")
        elif name == "skill_id":
            if not np.issubdtype(value.dtype, np.integer):
                raise ValueError("skill_id must use an integer dtype")
            if np.any((value < 0) | (value >= len(SKILL_NAMES))):
                raise ValueError("skill_id contains an unknown skill")
        else:
            _finite(name, value)
    if np.any(np.linalg.norm(arrays["root_quat_w"], axis=-1) < 1.0e-8):
        raise ValueError("root_quat_w contains a zero-norm quaternion")
    history_tail = arrays["policy_history"][..., -observation_dim:]
    if not np.array_equal(history_tail, arrays["policy_observation"]):
        raise ValueError("policy_observation must equal the newest frame of policy_history")


def save_behavior_trace(
    path: str | Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> Path:
    output = Path(path)
    if output.suffix != ".npz":
        raise ValueError("behavior trace path must end in .npz")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    validate_behavior_trace(metadata, normalized)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
    np.savez_compressed(output, metadata_json=np.asarray(metadata_json), **normalized)
    return output


def load_behavior_trace(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("behavior archive does not contain metadata_json")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
    validate_behavior_trace(metadata, arrays)
    return metadata, arrays


def _validate_checkpoint_contract(contract: object) -> None:
    if not isinstance(contract, Mapping) or set(contract) != set(SKILL_NAMES):
        raise ValueError(f"checkpoint_contract must contain {list(SKILL_NAMES)}")
    for skill, entry in contract.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"checkpoint_contract[{skill!r}] must be a mapping")
        required = {"body_sha256", "adaptation_sha256", "action_clip", "history_dim"}
        if set(entry) != required:
            raise ValueError(f"checkpoint_contract[{skill!r}] must contain {sorted(required)}")
        for key in ("body_sha256", "adaptation_sha256"):
            digest = entry[key]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"checkpoint_contract[{skill!r}].{key} must be a sha256 digest")
        if float(entry["action_clip"]) <= 0.0 or int(entry["history_dim"]) <= 0:
            raise ValueError(f"checkpoint_contract[{skill!r}] has invalid numeric values")


def _unique_string_list(metadata: Mapping[str, Any], key: str) -> list[str]:
    values = metadata[key]
    if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"metadata[{key!r}] must be a non-empty string list")
    if len(values) != len(set(values)):
        raise ValueError(f"metadata[{key!r}] contains duplicates")
    return values


def _finite(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite numeric values")
