"""Simulator-free tests for the migrated self-play team-frame contract."""

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
    / "tasks"
    / "manager_based"
    / "football"
    / "team_frame.py"
)
SPEC = importlib.util.spec_from_file_location("dribblebot_team_frame_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_commands_mirror_only_field_frame_skills() -> None:
    commands = torch.tensor([[0.8, -0.3, 0.4], [1.1, -0.7, 0.2], [-1.4, 0.5, 0.0]])
    skills = torch.tensor([0, 1, 2])
    mirrored = MODULE.mirror_high_level_commands(commands, skills)
    torch.testing.assert_close(mirrored[0], commands[0])
    torch.testing.assert_close(mirrored[1], torch.tensor([-1.1, 0.7, 0.2]))
    torch.testing.assert_close(mirrored[2], torch.tensor([1.4, -0.5, 0.0]))
    torch.testing.assert_close(MODULE.mirror_high_level_commands(mirrored, skills), commands)


def test_policy_action_mirror_preserves_logits_and_yaw() -> None:
    actions = torch.tensor(
        [
            [4.0, 1.0, 0.0, 0.8, -0.3, 0.4],
            [0.0, 4.0, 1.0, 1.1, -0.7, 0.2],
            [0.0, 1.0, 4.0, -1.4, 0.5, 0.0],
        ]
    )
    mirrored = MODULE.mirror_high_level_policy_actions(actions)
    torch.testing.assert_close(mirrored[:, :3], actions[:, :3])
    torch.testing.assert_close(mirrored[0, 3:], actions[0, 3:])
    torch.testing.assert_close(mirrored[1, 3:], torch.tensor([-1.1, 0.7, 0.2]))
    torch.testing.assert_close(mirrored[2, 3:], torch.tensor([1.4, -0.5, 0.0]))
    torch.testing.assert_close(MODULE.mirror_high_level_policy_actions(mirrored), actions)


def test_team_signs_are_slot_ordered() -> None:
    torch.testing.assert_close(MODULE.team_signs(4, 2), torch.tensor([1.0, 1.0, -1.0, -1.0]))
    with pytest.raises(ValueError):
        MODULE.team_signs(3, 2)
