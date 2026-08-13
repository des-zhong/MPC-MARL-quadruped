"""Episode-preserving MPC teacher data and world-model expansion records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from dribblebot.world_model.dataset import Episode


TEACHER_REQUIRED_KEYS = (
    "episode_id",
    "step_id",
    "global_state",
    "robot_0_local_observation",
    "robot_1_local_observation",
    "selected_joint_action",
    "teacher_joint_action",
    "robot_0_skill_id",
    "robot_0_skill_parameters",
    "robot_1_skill_id",
    "robot_1_skill_parameters",
    "real_reward",
    "real_next_global_state",
    "real_next_robot_0_local_observation",
    "real_next_robot_1_local_observation",
    "terminated",
    "truncated",
    "elapsed_low_level_steps",
    "real_event_labels",
    "predicted_reward",
    "predicted_next_state",
    "predicted_done_probability",
    "predicted_plan",
    "predicted_plan_states",
    "predicted_plan_rewards",
    "mpc_objective",
    "mpc_objective_components",
    "state_uncertainty",
    "reward_uncertainty",
    "return_uncertainty",
    "teacher_skill_probabilities",
    "teacher_parameter_means",
    "teacher_parameter_stds",
    "teacher_parameter_masks",
    "planning_time_seconds",
    "world_model_checkpoint_id",
    "behavior_source",
    "student_action",
    "has_student_action",
    "teacher_intervention",
    "student_teacher_disagreement",
)

# Written by value-aware collectors but intentionally optional when reading
# legacy reward-only teacher shards.
TERMINAL_VALUE_DIAGNOSTIC_KEYS = (
    "predicted_plan_values",
    "predicted_finite_horizon_return",
    "predicted_terminal_value",
    "discounted_terminal_value",
    "terminal_value_contribution",
    "terminal_state_uncertainty",
    "terminal_value_clipped",
    "selected_plan_total_objective",
    "real_return_to_go",
    "real_episode_return",
)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def file_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TeacherEpisode:
    episode_id: int
    arrays: Dict[str, np.ndarray]

    def validate(self, require_terminal_end: bool = True) -> None:
        action = np.asarray(self.arrays.get("teacher_joint_action", np.empty((0, 0))))
        if action.ndim < 2 or action.shape[-1] % 4:
            raise ValueError("teacher_joint_action must have shape [T, 4 * num_robots]")
        num_robots = action.shape[-1] // 4
        dynamic_robot_keys = tuple(
            key
            for robot in range(num_robots)
            for key in (
                f"robot_{robot}_local_observation",
                f"robot_{robot}_skill_id",
                f"robot_{robot}_skill_parameters",
                f"real_next_robot_{robot}_local_observation",
            )
        )
        fixed_keys = tuple(
            key for key in TEACHER_REQUIRED_KEYS
            if not key.startswith("robot_") and not key.startswith("real_next_robot_")
        )
        missing = [key for key in fixed_keys + dynamic_robot_keys if key not in self.arrays]
        if missing:
            raise ValueError(f"Teacher episode {self.episode_id} is missing fields {missing}")
        length = len(self.arrays["step_id"])
        if length == 0:
            raise ValueError(f"Teacher episode {self.episode_id} is empty")
        for key, value in self.arrays.items():
            if len(value) != length:
                raise ValueError(
                    f"Teacher episode {self.episode_id} field {key} has length "
                    f"{len(value)}, expected {length}"
                )
            if value.dtype.kind not in "USO" and not np.isfinite(value).all():
                raise ValueError(
                    f"Teacher episode {self.episode_id} field {key} contains NaN/Inf"
                )
        if not np.array_equal(self.arrays["step_id"], np.arange(length)):
            raise ValueError("Teacher step IDs must be contiguous from zero")
        if not np.all(np.asarray(self.arrays["episode_id"]) == self.episode_id):
            raise ValueError("Teacher episode_id array does not match shard ID")
        terminal = np.asarray(self.arrays["terminated"]).astype(bool) | np.asarray(
            self.arrays["truncated"]
        ).astype(bool)
        if terminal[:-1].any():
            raise ValueError("Teacher episode contains a terminal flag before its final step")
        if require_terminal_end and not terminal[-1]:
            raise ValueError("A complete teacher episode must end in termination or truncation")


class TeacherEpisodeWriter:
    """Atomically write immutable teacher episodes with resume protection."""

    format = "dribblebot_mpc_teacher_v1"

    def __init__(
        self,
        output: Union[str, Path],
        metadata: Mapping[str, object],
        resume: bool = False,
    ):
        self.output = Path(output)
        self.episodes_dir = self.output / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if not resume and existing.get("episodes"):
                raise FileExistsError(
                    f"Teacher dataset {self.output} already contains episodes; "
                    "use a new version directory or resume=True"
                )
            if existing.get("format") != self.format:
                raise ValueError(f"Unsupported teacher dataset format in {manifest_path}")
            self.entries = list(existing.get("episodes", []))
        else:
            self.entries = []
        existing_metadata = self.output / "metadata.json"
        if resume and existing_metadata.exists():
            stored = json.loads(existing_metadata.read_text())
            if stored != self.metadata:
                raise ValueError("Refusing to resume teacher collection with different metadata")
        _atomic_json(existing_metadata, self.metadata)
        self._ids = {int(entry["episode_id"]) for entry in self.entries}

    def write_episode(self, episode: TeacherEpisode) -> Path:
        episode.validate()
        if episode.episode_id in self._ids:
            raise FileExistsError(f"Teacher episode ID {episode.episode_id} already exists")
        filename = f"episode_{episode.episode_id:08d}.npz"
        path = self.episodes_dir / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing teacher shard {path}")
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **episode.arrays)
        temporary.replace(path)
        entry = {
            "episode_id": int(episode.episode_id),
            "path": f"episodes/{filename}",
            "length": int(len(episode.arrays["step_id"])),
            "sha256": file_sha256(path),
        }
        self.entries.append(entry)
        self.entries.sort(key=lambda item: int(item["episode_id"]))
        self._ids.add(episode.episode_id)
        _atomic_json(
            self.output / "manifest.json",
            {"format": self.format, "episodes": self.entries},
        )
        return path


class TeacherDataset:
    """Lazy episode reader; full predicted plans are loaded one shard at a time."""

    def __init__(self, root: Union[str, Path], verify_hashes: bool = False):
        self.root = Path(root)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest.get("format") != TeacherEpisodeWriter.format:
            raise ValueError(f"Unsupported teacher dataset format in {self.root}")
        self.entries = list(manifest["episodes"])
        self.verify_hashes = bool(verify_hashes)

    def __len__(self) -> int:
        return len(self.entries)

    def load_episode(self, index: int) -> TeacherEpisode:
        entry = self.entries[index]
        path = self.root / entry["path"]
        if self.verify_hashes and file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Teacher shard checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as shard:
            arrays = {key: shard[key] for key in shard.files}
        episode = TeacherEpisode(int(entry["episode_id"]), arrays)
        episode.validate()
        return episode

    @property
    def transition_count(self) -> int:
        return sum(int(entry["length"]) for entry in self.entries)


def episode_arrays(episode_id: int, records: Sequence[Mapping[str, object]]) -> Dict[str, np.ndarray]:
    if not records:
        raise ValueError("Cannot create an episode from no records")
    result = {
        key: np.asarray([record[key] for record in records])
        for key in records[0]
    }
    length = len(records)
    result["episode_id"] = np.full(length, episode_id, dtype=np.int64)
    result["step_id"] = np.arange(length, dtype=np.int64)
    return result


def teacher_parameter_masks(action_adapter, dtype=np.float32) -> np.ndarray:
    return np.asarray(
        [action_adapter.bounds[skill].mask for skill in range(3)],
        dtype=dtype,
    )


def action_disagreement(action_adapter, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student_skill, student_params = action_adapter.unpack(student)
    teacher_skill, teacher_params = action_adapter.unpack(teacher)
    student_norm = action_adapter.normalize_parameters(student_skill, student_params)
    teacher_norm = action_adapter.normalize_parameters(teacher_skill, teacher_params)
    compatible = student_skill == teacher_skill
    common_mask = (
        action_adapter._selected(student_skill, "mask", student.dtype)
        * action_adapter._selected(teacher_skill, "mask", teacher.dtype)
        * compatible.unsqueeze(-1).to(student.dtype)
    )
    parameter_error = ((student_norm - teacher_norm).square() * common_mask).sum(
        dim=(-1, -2)
    )
    skill_error = (student_skill != teacher_skill).to(student.dtype).sum(-1)
    return skill_error + parameter_error


def transition_to_teacher_record(
    transition,
    env_index: int,
    checkpoint_id: str,
    objective_component_names: Sequence[str],
    action_adapter,
    student_action: Optional[torch.Tensor] = None,
    teacher_intervention: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Convert a real controller transition to one serializable teacher row."""

    plan = transition.plan
    teacher = plan.first_joint_action
    selected = transition.executed_action
    skill_ids, skill_parameters = action_adapter.unpack(selected)
    if student_action is None:
        student = torch.zeros_like(teacher)
        has_student = torch.zeros(
            teacher.shape[0], dtype=torch.bool, device=teacher.device
        )
        disagreement = torch.zeros(
            teacher.shape[0], dtype=teacher.dtype, device=teacher.device
        )
    else:
        student = student_action
        has_student = torch.ones(
            teacher.shape[0], dtype=torch.bool, device=teacher.device
        )
        disagreement = action_disagreement(action_adapter, student, teacher)
    intervention = (
        torch.zeros_like(has_student)
        if teacher_intervention is None
        else teacher_intervention.bool()
    )
    components = torch.stack(
        [plan.objective_components[name] for name in objective_component_names],
        dim=-1,
    )
    num_robots = action_adapter.num_robots
    masks = np.broadcast_to(
        teacher_parameter_masks(action_adapter)[None, ...],
        (num_robots, 3, 3),
    ).copy()

    def array(value, dtype=np.float32):
        if torch.is_tensor(value):
            value = value[env_index].detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    record = {
        "global_state": array(transition.state),
        "selected_joint_action": array(selected),
        "teacher_joint_action": array(teacher),
        "requested_joint_action": array(transition.requested_action),
        "requested_action_modified": np.bool_(
            transition.requested_action_modified[env_index].item()
        ),
        "real_reward": np.float32(transition.reward[env_index].item()),
        "real_next_global_state": array(transition.next_state),
        "terminated": np.bool_(transition.terminated[env_index].item()),
        "truncated": np.bool_(transition.truncated[env_index].item()),
        "elapsed_low_level_steps": np.int16(
            transition.elapsed_low_level_steps[env_index].item()
        ),
        "real_event_labels": array(transition.event_labels),
        "predicted_reward": np.float32(plan.predicted_rewards[env_index, 0].item()),
        "predicted_next_state": array(plan.predicted_states[:, 1]),
        "predicted_done_probability": np.float32(
            plan.predicted_done_probabilities[env_index, 0].item()
        ),
        "predicted_plan": array(plan.best_action_sequence),
        "predicted_plan_states": array(plan.predicted_states),
        "predicted_plan_rewards": array(plan.predicted_rewards),
        "predicted_plan_values": array(plan.predicted_state_values),
        "predicted_finite_horizon_return": np.float32(
            plan.predicted_discounted_reward_return[env_index].item()
        ),
        "predicted_terminal_value": np.float32(
            plan.terminal_state_value[env_index].item()
        ),
        "discounted_terminal_value": np.float32(
            plan.discounted_terminal_value[env_index].item()
        ),
        "terminal_value_contribution": np.float32(
            plan.terminal_value_contribution[env_index].item()
        ),
        "terminal_state_uncertainty": np.float32(
            plan.terminal_state_uncertainty[env_index].item()
        ),
        "terminal_value_clipped": np.bool_(
            plan.terminal_value_clipped[env_index].item() > 0
        ),
        "selected_plan_total_objective": np.float32(
            plan.best_objective[env_index].item()
        ),
        # Replaced during complete-episode postprocessing by the collector.
        "real_return_to_go": np.float32(0.0),
        "real_episode_return": np.float32(0.0),
        "mpc_objective": np.float32(plan.best_objective[env_index].item()),
        "mpc_objective_components": array(components),
        "state_uncertainty": array(plan.uncertainty["state"]),
        "reward_uncertainty": array(plan.uncertainty["reward"]),
        "return_uncertainty": np.float32(plan.uncertainty["return"][env_index].item()),
        "teacher_skill_probabilities": array(
            plan.final_skill_probabilities[:, 0]
        ),
        "teacher_parameter_means": array(plan.final_parameter_means[:, 0]),
        "teacher_parameter_stds": array(plan.final_parameter_stds[:, 0]),
        "teacher_parameter_masks": masks,
        "selected_skill_ids": array(skill_ids, np.int64),
        "selected_skill_parameters": array(skill_parameters),
        "planning_time_seconds": np.float32(plan.planning_time_seconds),
        "world_model_checkpoint_id": str(checkpoint_id),
        "behavior_source": (
            "mpc" if bool(transition.teacher_action_executed[env_index]) else "student_labeled"
        ),
        "student_action": array(student),
        "has_student_action": np.bool_(has_student[env_index].item()),
        "teacher_intervention": np.bool_(intervention[env_index].item()),
        "student_teacher_disagreement": np.float32(disagreement[env_index].item()),
        "fallback_used": np.bool_(plan.fallback_used[env_index].item()),
    }
    for robot in range(num_robots):
        record[f"robot_{robot}_local_observation"] = array(
            transition.local_observations[:, robot]
        )
        record[f"robot_{robot}_skill_id"] = np.int64(
            skill_ids[env_index, robot].item()
        )
        record[f"robot_{robot}_skill_parameters"] = array(
            skill_parameters[:, robot]
        )
        record[f"real_next_robot_{robot}_local_observation"] = array(
            transition.next_local_observations[:, robot]
        )
    return record


def transition_to_world_model_record(transition, env_index: int) -> Dict[str, object]:
    """Return only real simulator values suitable for dynamics retraining."""

    def array(value, dtype=np.float32):
        return np.asarray(
            value[env_index].detach().cpu().numpy(),
            dtype=dtype,
        )

    return {
        "state": array(transition.state),
        "joint_action": array(transition.executed_action),
        "requested_joint_action": array(transition.requested_action),
        "requested_action_modified": np.bool_(
            transition.requested_action_modified[env_index].item()
        ),
        "invalid_skill_requested": np.bool_(
            transition.requested_action_modified[env_index].item()
        ),
        "reward": np.float32(transition.reward[env_index].item()),
        "next_state": array(transition.next_state),
        "terminated": np.bool_(transition.terminated[env_index].item()),
        "truncated": np.bool_(transition.truncated[env_index].item()),
        "elapsed_low_level_steps": np.int16(
            transition.elapsed_low_level_steps[env_index].item()
        ),
        "event_labels": array(transition.event_labels),
        "behavior_source": "mpc",
        "targeted_scenario": "",
        "termination_reason": "timeout" if bool(transition.truncated[env_index].item()) else (
            "terminal" if bool(transition.terminated[env_index].item()) else "none"
        ),
        "ground_truth_source": "real_simulator",
    }
