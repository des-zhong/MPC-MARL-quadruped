"""Simulator-free tests for the canonical snapshot world-model adapter."""

import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
    / "world_model_adapter.py"
)
REPOSITORY_ROOT = MODULE_PATH.parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
SPEC = importlib.util.spec_from_file_location("dribblebot_world_model_adapter_test", MODULE_PATH)
adapter_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(adapter_module)


def _snapshot(batch: int = 2, robots: int = 4) -> dict[str, torch.Tensor]:
    root = torch.zeros(batch, robots, 13)
    root[..., 2] = 0.34
    root[..., 3] = 1.0
    ball = torch.zeros(batch, 13)
    ball[:, 2] = 0.10
    ball[:, 3] = 1.0
    commands = torch.zeros(batch, robots, 3)
    commands[..., 0] = 1.0
    return {
        "env_origin_w": torch.zeros(batch, 3),
        "robot_root_state_w": root,
        "robot_joint_pos": torch.zeros(batch, robots, 12),
        "robot_joint_vel": torch.zeros(batch, robots, 12),
        "robot_joint_target": torch.zeros(batch, robots, 12),
        "ball_root_state_w": ball,
        "skill_id": torch.zeros(batch, robots, dtype=torch.long),
        "requested_skill_id": torch.zeros(batch, robots, dtype=torch.long),
        "invalid_skill": torch.zeros(batch, robots, dtype=torch.bool),
        "skill_command": commands,
        "gait_phase": torch.zeros(batch, robots),
    }


def test_canonical_snapshot_encodes_existing_world_model_schema() -> None:
    adapter = adapter_module.IsaacLabFootballWorldModelStateAdapter(max_obstacles=0)
    structured = adapter.extract_state(_snapshot())
    assert structured["tensor"].shape == (2, adapter.state_dim)
    assert torch.isfinite(structured["tensor"]).all()
    assert structured["ball.possessor_one_hot"].shape == (2, 5)


def test_canonical_snapshot_rejects_wrong_robot_shape() -> None:
    adapter = adapter_module.IsaacLabFootballWorldModelStateAdapter(max_obstacles=0)
    snapshot = _snapshot(robots=3)
    try:
        adapter.extract_state(snapshot)
    except ValueError as exc:
        assert "robot_root_state_w" in str(exc)
    else:
        raise AssertionError("wrong robot count should be rejected")


def test_recorder_episode_is_stacked_into_world_model_transitions() -> None:
    adapter = adapter_module.IsaacLabFootballWorldModelStateAdapter(max_obstacles=0)
    first = _snapshot(batch=1)
    second = _snapshot(batch=1)
    second["ball_root_state_w"][:, 0] = 0.25
    recorder_data = {
        "football": {
            "snapshot": {
                key: [first[key][0], second[key][0]]
                for key in first
            }
        }
    }
    transitions = adapter.extract_episode_transitions(recorder_data)
    assert transitions["state"].shape == (1, adapter.state_dim)
    assert transitions["next_state"].shape == (1, adapter.state_dim)
    assert transitions["event_labels"].shape == (1, len(adapter.event_names))
    assert torch.isfinite(transitions["state"]).all()


def test_recorder_episode_rejects_single_snapshot() -> None:
    adapter = adapter_module.IsaacLabFootballWorldModelStateAdapter(max_obstacles=0)
    snapshot = _snapshot(batch=1)
    recorder_data = {"football": {"snapshot": {key: [value[0]] for key, value in snapshot.items()}}}
    try:
        adapter.extract_episode_transitions(recorder_data)
    except ValueError as exc:
        assert "At least two snapshots" in str(exc)
    else:
        raise AssertionError("a single recorder sample cannot form a transition")


def test_recorder_singleton_vector_axis_is_removed() -> None:
    adapter = adapter_module.IsaacLabFootballWorldModelStateAdapter(max_obstacles=0)
    snapshot = _snapshot(batch=1)
    recorder_data = {
        "football": {
            "snapshot": {
                key: [value[:1], value[:1]]
                for key, value in snapshot.items()
            }
        }
    }
    stacked = adapter.stack_recorder_snapshot(recorder_data)
    assert stacked["robot_root_state_w"].shape == (2, 4, 13)


def test_one_robot_snapshot_keeps_the_robot_axis() -> None:
    adapter = adapter_module.IsaacLabFootballWorldModelStateAdapter(num_robots=1, max_obstacles=0)
    snapshot = _snapshot(batch=1, robots=1)
    recorder_data = {"football": {"snapshot": {key: [value[0], value[0]] for key, value in snapshot.items()}}}
    stacked = adapter.stack_recorder_snapshot(recorder_data)
    assert stacked["robot_root_state_w"].shape == (2, 1, 13)
    transitions = adapter.extract_episode_transitions(recorder_data)
    assert transitions["state"].shape == (1, adapter.state_dim)
