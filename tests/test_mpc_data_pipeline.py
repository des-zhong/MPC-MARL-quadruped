import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc.acceptance import (
    ModelAcceptanceConfig,
    ModelAcceptanceGate,
)
from dribblebot.mpc.rl_export import return_to_go
from dribblebot.mpc.teacher_dataset import (
    TEACHER_REQUIRED_KEYS,
    TeacherDataset,
    TeacherEpisode,
    TeacherEpisodeWriter,
)


def _teacher_arrays(length=3, episode_id=5):
    arrays = {}
    vector_fields = {
        "global_state": (4,),
        "robot_0_local_observation": (3,),
        "robot_1_local_observation": (3,),
        "selected_joint_action": (8,),
        "teacher_joint_action": (8,),
        "robot_0_skill_parameters": (3,),
        "robot_1_skill_parameters": (3,),
        "real_next_global_state": (4,),
        "real_next_robot_0_local_observation": (3,),
        "real_next_robot_1_local_observation": (3,),
        "real_event_labels": (2,),
        "predicted_next_state": (4,),
        "predicted_plan": (2, 8),
        "predicted_plan_states": (3, 4),
        "predicted_plan_rewards": (2,),
        "mpc_objective_components": (13,),
        "state_uncertainty": (2,),
        "reward_uncertainty": (2,),
        "teacher_skill_probabilities": (2, 3),
        "teacher_parameter_means": (2, 3, 3),
        "teacher_parameter_stds": (2, 3, 3),
        "teacher_parameter_masks": (2, 3, 3),
        "student_action": (8,),
    }
    scalar_float = {
        "real_reward",
        "predicted_reward",
        "predicted_done_probability",
        "mpc_objective",
        "return_uncertainty",
        "planning_time_seconds",
        "student_teacher_disagreement",
    }
    scalar_bool = {
        "terminated",
        "truncated",
        "has_student_action",
        "teacher_intervention",
    }
    scalar_int = {
        "robot_0_skill_id",
        "robot_1_skill_id",
        "elapsed_low_level_steps",
    }
    for key in TEACHER_REQUIRED_KEYS:
        if key == "episode_id":
            arrays[key] = np.full(length, episode_id, dtype=np.int64)
        elif key == "step_id":
            arrays[key] = np.arange(length, dtype=np.int64)
        elif key in vector_fields:
            arrays[key] = np.zeros((length,) + vector_fields[key], dtype=np.float32)
        elif key in scalar_float:
            arrays[key] = np.zeros(length, dtype=np.float32)
        elif key in scalar_bool:
            arrays[key] = np.zeros(length, dtype=bool)
        elif key in scalar_int:
            arrays[key] = np.zeros(length, dtype=np.int64)
        else:
            arrays[key] = np.asarray(["test"] * length)
    arrays["truncated"][-1] = True
    return arrays


def test_teacher_dataset_preserves_complete_episode_and_hash(tmp_path):
    writer = TeacherEpisodeWriter(tmp_path, {"format": "test"})
    writer.write_episode(TeacherEpisode(5, _teacher_arrays()))
    dataset = TeacherDataset(tmp_path, verify_hashes=True)
    episode = dataset.load_episode(0)
    assert len(dataset) == 1
    assert dataset.transition_count == 3
    assert episode.arrays["step_id"].tolist() == [0, 1, 2]
    with pytest.raises(FileExistsError):
        writer.write_episode(TeacherEpisode(5, _teacher_arrays()))


def test_return_to_go_uses_real_reward_order():
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(return_to_go(rewards, 1.0), [6.0, 5.0, 3.0])
    assert np.allclose(return_to_go(rewards, 0.5), [2.75, 3.5, 3.0])


def _metrics(original, recent=None, **overrides):
    base = {
        "normalized_state_rmse": original,
        "reward_rmse": 1.0,
        "termination_brier": 0.1,
        "mean_state_uncertainty": 1.0,
        "ball_rollout_rmse": 1.0,
        "finite_horizon_rollout": True,
        "rollout_horizon": 20,
        "rollout_sequence_count": 10,
    }
    base.update(overrides)
    recent_values = dict(base)
    recent_values["normalized_state_rmse"] = (
        original if recent is None else recent
    )
    return {"original": base, "recent": recent_values}


def test_model_acceptance_gate_accepts_and_rejects_toy_metrics():
    gate = ModelAcceptanceGate(ModelAcceptanceConfig())
    accepted = gate.compare(_metrics(1.0, 1.2), _metrics(1.01, 1.0))
    assert accepted["accepted"]
    rejected = gate.compare(_metrics(1.0, 1.0), _metrics(1.2, 1.1))
    assert not rejected["accepted"]
    assert "original_validation" in rejected["failed_criteria"]

