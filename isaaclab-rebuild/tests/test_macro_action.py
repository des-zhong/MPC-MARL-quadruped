"""Tests for the coordinator-rate adapter around a manager environment."""

import importlib.util
from pathlib import Path

import gymnasium as gym
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
    / "macro.py"
)
SPEC = importlib.util.spec_from_file_location("dribblebot_macro_test", MODULE_PATH)
macro = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(macro)


class FakeManagerEnv(gym.Env):
    is_vector_env = True

    def __init__(self):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=float)
        self.observation_space = gym.spaces.Box(-100.0, 100.0, shape=(2, 1), dtype=float)
        self.step_index = 0
        self.actions = []

    def reset(self, **kwargs):
        self.step_index = 0
        self.actions.clear()
        return torch.zeros(2, 1), {"reset": True}

    def step(self, action):
        self.actions.append(action.clone())
        self.step_index += 1
        terminated = torch.tensor([self.step_index == 2, False])
        truncated = torch.zeros(2, dtype=torch.bool)
        reward = torch.ones(2) * float(self.step_index)
        return torch.full((2, 1), float(self.step_index)), reward, terminated, truncated, {
            "low_step": self.step_index
        }


def test_macro_wrapper_sums_active_rewards_and_masks_finished_rows() -> None:
    env = macro.MacroActionWrapper(FakeManagerEnv(), control_interval=3)
    env.reset()
    action = torch.ones(2, 3)
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation[:, 0].tolist() == [3.0, 3.0]
    # Row 0 is terminal at tick 2: 1 + 2. Row 1 runs all 3 ticks: 1 + 2 + 3.
    torch.testing.assert_close(reward, torch.tensor([3.0, 6.0]))
    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, False]
    assert info["macro_control_interval"] == 3
    assert info["elapsed_low_level_steps"].tolist() == [2, 3]
    assert len(env.env.actions) == 3
    torch.testing.assert_close(env.env.actions[0], action)
    torch.testing.assert_close(env.env.actions[1], action)
    torch.testing.assert_close(env.env.actions[2][0], torch.zeros(3))
    torch.testing.assert_close(env.env.actions[2][1], action[1])


def test_macro_wrapper_requires_positive_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        macro.MacroActionWrapper(FakeManagerEnv(), control_interval=0)
