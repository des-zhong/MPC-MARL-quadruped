"""Export MPC teacher episodes for actor BC and centralized critic pretraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .teacher_dataset import TeacherDataset, file_sha256


def return_to_go(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Discounted return from each step using only real recorded rewards."""

    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    rewards = np.asarray(rewards, dtype=np.float32)
    result = np.zeros_like(rewards, dtype=np.float32)
    running = np.float32(0.0)
    for step in range(len(rewards) - 1, -1, -1):
        running = rewards[step] + np.float32(gamma) * running
        result[step] = running
    return result


def generalized_lambda_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """GAE/lambda targets with an explicit value for every state, including T."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(rewards) + 1,):
        raise ValueError("values must contain T+1 entries")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must lie in [0, 1]")
    done = np.asarray(terminated).astype(bool) | np.asarray(truncated).astype(bool)
    advantages = np.zeros_like(rewards)
    running = np.float32(0.0)
    for step in range(len(rewards) - 1, -1, -1):
        alive = np.float32(not done[step])
        delta = rewards[step] + gamma * alive * values[step + 1] - values[step]
        running = delta + gamma * gae_lambda * alive * running
        advantages[step] = running
    return advantages + values[:-1]


def _atomic_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def export_teacher_dataset_for_rl(
    input_root: Union[str, Path],
    output_root: Union[str, Path],
    gamma: float = 0.99,
    gae_lambda: Optional[float] = None,
    value_key: Optional[str] = None,
    verify_hashes: bool = False,
) -> Dict[str, object]:
    """Write episode shards with actor and critic pretraining targets."""

    source = TeacherDataset(input_root, verify_hashes=verify_hashes)
    output = Path(output_root)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(
            f"RL export {output} already exists; use a new version directory"
        )
    entries = []
    transitions = 0
    for index, entry in enumerate(source.entries):
        episode = source.load_episode(index)
        arrays = episode.arrays
        action_width = arrays["teacher_joint_action"].shape[-1]
        if action_width % 4:
            raise ValueError(f"Teacher action width {action_width} is not divisible by 4")
        num_robots = action_width // 4
        teacher_action = arrays["teacher_joint_action"].reshape(-1, num_robots, 4)
        skills = teacher_action[..., 0].astype(np.int64)
        parameters = teacher_action[..., 1:].astype(np.float32)
        rewards = arrays["real_reward"].astype(np.float32)
        exported = {
            "episode_id": arrays["episode_id"].astype(np.int64),
            "step_id": arrays["step_id"].astype(np.int64),
            "critic_global_state": arrays["global_state"].astype(np.float32),
            "critic_next_global_state": arrays[
                "real_next_global_state"
            ].astype(np.float32),
            "real_reward": rewards,
            "terminated": arrays["terminated"].astype(bool),
            "truncated": arrays["truncated"].astype(bool),
            "undiscounted_return_to_go": return_to_go(rewards, 1.0),
            "discounted_return_to_go": return_to_go(rewards, gamma),
            "teacher_intervention": arrays["teacher_intervention"].astype(bool),
            "student_teacher_disagreement": arrays[
                "student_teacher_disagreement"
            ].astype(np.float32),
        }
        for robot in range(num_robots):
            prefix = f"actor_{robot}"
            exported[f"{prefix}_local_observation"] = arrays[
                f"robot_{robot}_local_observation"
            ].astype(np.float32)
            exported[f"{prefix}_teacher_skill_id"] = skills[:, robot]
            exported[f"{prefix}_teacher_skill_probabilities"] = arrays[
                "teacher_skill_probabilities"
            ][:, robot].astype(np.float32)
            exported[f"{prefix}_teacher_parameter_target"] = parameters[:, robot]
            exported[f"{prefix}_teacher_parameter_means"] = arrays[
                "teacher_parameter_means"
            ][:, robot].astype(np.float32)
            exported[f"{prefix}_teacher_parameter_stds"] = arrays[
                "teacher_parameter_stds"
            ][:, robot].astype(np.float32)
            exported[f"{prefix}_parameter_masks"] = arrays[
                "teacher_parameter_masks"
            ][:, robot].astype(np.float32)
        if gae_lambda is not None:
            if not value_key:
                raise ValueError(
                    "Optional generalized-lambda targets require --value-key "
                    "containing T+1 centralized value predictions"
                )
            if value_key not in arrays:
                raise KeyError(
                    f"Teacher episode does not contain requested value key {value_key!r}"
                )
            exported["generalized_lambda_return"] = generalized_lambda_returns(
                rewards,
                arrays[value_key],
                arrays["terminated"],
                arrays["truncated"],
                gamma,
                gae_lambda,
            )
        filename = f"episode_{episode.episode_id:08d}.npz"
        path = episodes_dir / filename
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **exported)
        temporary.replace(path)
        entries.append(
            {
                "episode_id": int(episode.episode_id),
                "path": f"episodes/{filename}",
                "length": int(len(rewards)),
                "sha256": file_sha256(path),
            }
        )
        transitions += len(rewards)

    metadata = {
        "format": "dribblebot_mpc_rl_pretraining_v1",
        "source_teacher_dataset": str(Path(input_root).resolve()),
        "source_teacher_format": TeacherDataset(input_root).metadata.get(
            "format", "dribblebot_mpc_teacher_v1"
        ),
        "gamma": float(gamma),
        "gae_lambda": None if gae_lambda is None else float(gae_lambda),
        "value_key": value_key,
        "actor_target": (
            "Behavior cloning/distillation targets from the MPC first-step "
            "distribution; one independent actor view per robot."
        ),
        "critic_target": (
            "Centralized state and returns calculated exclusively from real "
            "simulator rewards."
        ),
        "ppo_importance_ratios_present": False,
        "num_robots": int(source.metadata.get("num_robots", num_robots if entries else 2)),
    }
    _atomic_json(output / "metadata.json", metadata)
    _atomic_json(
        output / "manifest.json",
        {"format": metadata["format"], "episodes": entries},
    )
    return {"episodes": len(entries), "transitions": transitions, **metadata}
