"""Deterministic frozen-skill trace fixtures."""

from __future__ import annotations

import copy

import numpy as np

from parity.behavior_excitation import SKILL_NAMES, generate_coordinator_sequence
from parity.behavior_schema import BEHAVIOR_SCHEMA_VERSION
from parity.policy_contract import make_legacy_policy_contract


def make_behavior_trace(source: str = "isaacgym") -> tuple[dict, dict[str, np.ndarray]]:
    contract = make_legacy_policy_contract(include_ball=True)
    joint_names = ["joint_b", "joint_a"]
    steps = 4
    num_envs = 2
    coordinator = generate_coordinator_sequence(steps, num_envs, "switch", switch_interval=1)
    observation = np.arange(steps * num_envs * contract["observation_dim"], dtype=np.float32).reshape(
        steps, num_envs, contract["observation_dim"]
    ) / 1000.0
    history = np.zeros((steps, num_envs, contract["history_dim"]), dtype=np.float32)
    for step in range(steps):
        recent = observation[: step + 1].transpose(1, 0, 2).reshape(num_envs, -1)
        history[step, :, -recent.shape[1] :] = recent
    frames = steps + 1
    root_pos = np.zeros((frames, num_envs, 3), dtype=np.float32)
    root_pos[:, 1, 0] = 10.0
    ball_pos = root_pos.copy()
    ball_pos[..., 0] += 0.6
    root_quat = np.zeros((frames, num_envs, 4), dtype=np.float32)
    root_quat[..., 0] = 1.0
    checkpoint = {
        skill: {
            "body_sha256": str(index + 1) * 64,
            "adaptation_sha256": str(index + 4) * 64,
            "action_clip": 1.0,
            "history_dim": 1080 if skill == "walk" else 1125,
        }
        for index, skill in enumerate(SKILL_NAMES)
    }
    metadata = {
        "schema_version": BEHAVIOR_SCHEMA_VERSION,
        "source": source,
        "task": "test-frozen-skill",
        "step_dt": 0.02,
        "physics_dt": 0.005,
        "decimation": 4,
        "joint_names": joint_names,
        "skill_names": list(SKILL_NAMES),
        "policy_contract": contract,
        "checkpoint_contract": checkpoint,
        "action_history_semantics": "duplicate_previous_policy_output",
        "contract_horizon_steps": steps,
    }
    skill_id = np.argmax(coordinator[..., :3], axis=-1).astype(np.int64)
    arrays = {
        "time_s": np.arange(frames, dtype=np.float64) * 0.02,
        "coordinator_action": coordinator,
        "skill_id": skill_id,
        "skill_command": np.zeros((steps, num_envs, 3), dtype=np.float32),
        "policy_observation": observation,
        "policy_history": history,
        "policy_action": np.zeros((steps, num_envs, len(joint_names)), dtype=np.float32),
        "processed_joint_target": np.zeros((steps, num_envs, len(joint_names)), dtype=np.float32),
        "done": np.zeros((steps, num_envs), dtype=np.bool_),
        "env_origin_w": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
        "root_pos_w": root_pos,
        "root_quat_w": root_quat,
        "joint_pos": np.zeros((frames, num_envs, len(joint_names)), dtype=np.float32),
        "ball_pos_w": ball_pos,
    }
    return metadata, arrays


def clone_behavior(metadata: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    return copy.deepcopy(metadata), {name: value.copy() for name, value in arrays.items()}
