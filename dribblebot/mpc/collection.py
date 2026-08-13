"""MPC teacher collection and real-transition world-model expansion."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Union

import numpy as np
import torch

from dribblebot.world_model.dataset import Episode, EpisodeShardWriter, split_episodes

from .objective import MPCObjective
from .terminal_value import compute_discounted_returns
from .teacher_dataset import (
    TeacherEpisode,
    TeacherEpisodeWriter,
    episode_arrays,
    transition_to_teacher_record,
    transition_to_world_model_record,
)


class TeacherRolloutCollector:
    """Collect complete real episodes under MPC or student-labeled execution."""

    def __init__(
        self,
        controller,
        teacher_writer: TeacherEpisodeWriter,
        expansion_writer: EpisodeShardWriter,
        checkpoint_id: str,
        behavior_mode: str = "mpc",
        student_policy: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        intervention_rule: Optional[
            Callable[[torch.Tensor, object, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        if behavior_mode not in ("mpc", "student_labels"):
            raise ValueError("behavior_mode must be 'mpc' or 'student_labels'")
        if behavior_mode == "student_labels" and student_policy is None:
            raise ValueError("student_labels mode requires a student_policy callable")
        self.controller = controller
        self.teacher_writer = teacher_writer
        self.expansion_writer = expansion_writer
        self.checkpoint_id = str(checkpoint_id)
        self.behavior_mode = behavior_mode
        self.student_policy = student_policy
        self.intervention_rule = intervention_rule
        self.component_names = tuple(MPCObjective.COMPONENT_NAMES)
        self._last_student_action = None
        self._last_intervention = None
        self.gamma = float(controller.planner.config.gamma)

    def _execution_policy(self, local_observations, plan):
        student = self.student_policy(local_observations)
        student = student.to(
            device=plan.first_joint_action.device,
            dtype=plan.first_joint_action.dtype,
        )
        self.controller.action_adapter.assert_within_bounds(student)
        if self.intervention_rule is None:
            intervention = torch.zeros(
                student.shape[0], dtype=torch.bool, device=student.device
            )
        else:
            intervention = self.intervention_rule(local_observations, plan, student)
            intervention = torch.as_tensor(
                intervention, device=student.device, dtype=torch.bool
            )
            if intervention.shape != (student.shape[0],):
                raise ValueError(
                    f"intervention_rule must return [{student.shape[0]}], got {tuple(intervention.shape)}"
                )
        self._last_student_action = student.detach()
        self._last_intervention = intervention.detach()
        return torch.where(intervention[:, None], plan.first_joint_action, student)

    @staticmethod
    def _force_collection_cutoff(records) -> None:
        if not records:
            return
        records[-1]["truncated"] = np.bool_(True)
        records[-1]["collection_cutoff"] = np.bool_(True)

    def _teacher_episode(self, episode_id, records):
        arrays = episode_arrays(episode_id, records)
        returns = compute_discounted_returns(
            arrays["real_reward"], arrays["terminated"], arrays["truncated"], self.gamma
        )
        arrays["real_return_to_go"] = returns
        arrays["real_episode_return"] = np.full(
            len(returns), returns[0], dtype=np.float32
        )
        return TeacherEpisode(episode_id, arrays)

    def collect(
        self,
        num_episodes: int,
        max_macro_steps: Optional[int] = None,
    ) -> Dict[str, object]:
        if num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        env = self.controller.env
        self.controller.reset()
        teacher_buffers = [[] for _ in range(env.num_envs)]
        expansion_buffers = [[] for _ in range(env.num_envs)]
        episode_ids = np.arange(env.num_envs, dtype=np.int64)
        next_episode_id = env.num_envs
        completed = 0
        macro_steps = 0
        event_counts = Counter()
        num_robots = self.controller.action_adapter.num_robots
        skill_counts = [Counter() for _ in range(num_robots)]
        planning_times = []
        prediction_errors = []
        try:
            while completed < num_episodes:
                if max_macro_steps is not None and macro_steps >= max_macro_steps:
                    break
                self._last_student_action = None
                self._last_intervention = None
                execution_policy = (
                    self._execution_policy
                    if self.behavior_mode == "student_labels"
                    else None
                )
                transition = self.controller.step(execution_policy)
                macro_steps += 1
                planning_times.append(float(transition.plan.planning_time_seconds))
                first_error = (
                    transition.plan.predicted_states[:, 1] - transition.next_state
                ).square().mean(-1).sqrt()
                prediction_errors.extend(first_error.detach().cpu().tolist())
                teacher_skills, _ = self.controller.action_adapter.unpack(
                    transition.plan.first_joint_action
                )
                for robot in range(num_robots):
                    skill_counts[robot].update(
                        teacher_skills[:, robot].detach().cpu().tolist()
                    )
                totals = transition.event_labels.sum(0).detach().cpu().numpy()
                event_counts.update(
                    {
                        name: int(totals[index])
                        for index, name in enumerate(
                            self.controller.planner.world_model.event_names
                        )
                        if totals[index]
                    }
                )

                for env_index in range(env.num_envs):
                    teacher_record = transition_to_teacher_record(
                        transition,
                        env_index,
                        self.checkpoint_id,
                        self.component_names,
                        self.controller.action_adapter,
                        self._last_student_action,
                        self._last_intervention,
                    )
                    teacher_record["collection_cutoff"] = np.bool_(False)
                    expansion_record = transition_to_world_model_record(
                        transition, env_index
                    )
                    expansion_record["collection_cutoff"] = np.bool_(False)
                    teacher_buffers[env_index].append(teacher_record)
                    expansion_buffers[env_index].append(expansion_record)
                    if not bool(transition.done[env_index].item()):
                        continue
                    if completed < num_episodes:
                        episode_id = int(episode_ids[env_index])
                        self.teacher_writer.write_episode(
                            self._teacher_episode(
                                episode_id, teacher_buffers[env_index]
                            )
                        )
                        self.expansion_writer.write_episode(
                            Episode(
                                episode_id,
                                episode_arrays(
                                    episode_id, expansion_buffers[env_index]
                                ),
                            )
                        )
                        completed += 1
                    teacher_buffers[env_index] = []
                    expansion_buffers[env_index] = []
                    episode_ids[env_index] = next_episode_id
                    next_episode_id += 1

            if completed < num_episodes and max_macro_steps is not None:
                for env_index in range(env.num_envs):
                    if completed >= num_episodes:
                        break
                    if not teacher_buffers[env_index]:
                        continue
                    self._force_collection_cutoff(teacher_buffers[env_index])
                    self._force_collection_cutoff(expansion_buffers[env_index])
                    episode_id = int(episode_ids[env_index])
                    self.teacher_writer.write_episode(
                        self._teacher_episode(
                            episode_id, teacher_buffers[env_index]
                        )
                    )
                    self.expansion_writer.write_episode(
                        Episode(
                            episode_id,
                            episode_arrays(episode_id, expansion_buffers[env_index]),
                        )
                    )
                    completed += 1
        finally:
            self.controller.close()

        summary = {
            "episodes": completed,
            "macro_steps": macro_steps,
            "behavior_mode": self.behavior_mode,
            "checkpoint_id": self.checkpoint_id,
            "event_counts": dict(event_counts),
            "teacher_skill_frequencies": [
                {str(key): int(value) for key, value in counts.items()}
                for counts in skill_counts
            ],
            "planning_time_seconds": _summary_statistics(planning_times),
            "one_step_encoded_state_rmse": _summary_statistics(prediction_errors),
        }
        return summary


def _summary_statistics(values) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_collection_metadata(
    model,
    checkpoint_payload: Mapping[str, object],
    checkpoint_id: str,
    mpc_config,
    environment_config: Mapping[str, object],
    local_observation_adapter,
    skill_policy_metadata: Optional[Mapping[str, object]] = None,
) -> tuple[Dict[str, object], Dict[str, object]]:
    common = {
        "num_robots": model.action_adapter.num_robots,
        "num_obstacles": sum(
            feature.name.startswith("obstacle_") for feature in model.schema.features
        ),
        "robot": checkpoint_payload.get("training_config", {})
        .get("environment", {})
        .get("robot", "as2"),
        "state_schema": model.schema.to_dict(),
        "action_schema": model.action_adapter.to_dict(),
        "event_names": list(model.event_names),
        "world_model_checkpoint_id": checkpoint_id,
        "world_model_checkpoint_epoch": checkpoint_payload.get("epoch"),
        "world_model_repository_commit": checkpoint_payload.get(
            "repository_commit"
        ),
        "mpc_config": mpc_config.to_dict(),
        "environment_config": dict(environment_config),
        "skill_policy_metadata": dict(skill_policy_metadata or {}),
    }
    teacher = dict(common)
    teacher.update(
        {
            "format": TeacherEpisodeWriter.format,
            "objective_component_names": list(MPCObjective.COMPONENT_NAMES),
            "local_observation_schema": local_observation_adapter.metadata(),
            "ground_truth_policy": (
                "Only real simulator state/reward/next-state values are ground truth; "
                "predicted plans are teacher diagnostics."
            ),
            "dagger_fields_present": True,
        }
    )
    expansion = dict(common)
    expansion.update(
        {
            "format": "dribblebot_world_model_v1",
            "behavior_source": "mpc",
            "ground_truth_source": "real_simulator",
            "contains_imagined_ground_truth": False,
        }
    )
    return teacher, expansion


def finalize_expansion_dataset(
    root: Union[str, Path],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> Dict[str, object]:
    splits = split_episodes(
        root,
        train_fraction,
        validation_fraction,
        test_fraction,
        seed,
    )
    manifest = {
        "splits": {name: len(ids) for name, ids in splits.items()},
        "seed": int(seed),
    }
    path = Path(root) / "iteration_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
