"""Tests for simulator-independent frozen skill-policy inference."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
    / "policies"
    / "frozen_low_level.py"
)
SPEC = importlib.util.spec_from_file_location("dribblebot_frozen_policy", POLICY_PATH)
policy_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = policy_module
SPEC.loader.exec_module(policy_module)


def test_load_and_infer_clips_actor_output(tmp_path: Path) -> None:
    spec = _write_policy(tmp_path, "walk", history_dim=6, latent_dim=2, action_bias=3.0, action_clip=0.5)
    policy = policy_module.FrozenLowLevelPolicy.load(spec)
    action = policy(torch.zeros(4, 6))
    torch.testing.assert_close(action, torch.full((4, 12), 0.5))
    assert policy.history_dim == 6
    assert policy.latent_dim == 2


def test_non_finite_history_fails_safe_to_zero_action(tmp_path: Path) -> None:
    spec = _write_policy(tmp_path, "walk", history_dim=6, latent_dim=2, action_bias=0.25)
    policy = policy_module.FrozenLowLevelPolicy.load(spec)
    history = torch.zeros(2, 6)
    history[1, 3] = float("nan")
    action = policy(history)
    torch.testing.assert_close(action[0], torch.full((12,), 0.25))
    torch.testing.assert_close(action[1], torch.zeros(12))


def test_ball_history_is_adapted_for_walking() -> None:
    history = torch.arange(2 * 3 * 5, dtype=torch.float).view(2, 15)
    adapted = policy_module.adapt_history_for_policy(
        history,
        expected_history_dim=9,
        full_observation_dim=5,
        history_length=3,
        object_sensor_width=2,
    )
    expected = history.view(2, 3, 5)[:, :, 2:].reshape(2, 9)
    torch.testing.assert_close(adapted, expected)


def test_policy_set_routes_rows_to_selected_skill(tmp_path: Path) -> None:
    policies = {
        0: policy_module.FrozenLowLevelPolicy.load(
            _write_policy(tmp_path, "walk", history_dim=9, latent_dim=2, action_bias=0.1)
        ),
        1: policy_module.FrozenLowLevelPolicy.load(
            _write_policy(tmp_path, "dribble", history_dim=15, latent_dim=2, action_bias=0.2)
        ),
        2: policy_module.FrozenLowLevelPolicy.load(
            _write_policy(tmp_path, "shoot", history_dim=15, latent_dim=2, action_bias=0.3)
        ),
    }
    policy_set = policy_module.FrozenSkillPolicySet(
        policies,
        full_observation_dim=5,
        history_length=3,
        object_sensor_width=2,
    )
    action = policy_set.route(torch.zeros(3, 15), torch.tensor([0, 2, 1]))
    torch.testing.assert_close(action[0], torch.full((12,), 0.1))
    torch.testing.assert_close(action[1], torch.full((12,), 0.3))
    torch.testing.assert_close(action[2], torch.full((12,), 0.2))


def test_policy_set_rejects_unknown_skill_id(tmp_path: Path) -> None:
    policies = {
        skill_id: policy_module.FrozenLowLevelPolicy.load(
            _write_policy(tmp_path, name, history_dim=15, latent_dim=2, action_bias=0.0)
        )
        for skill_id, name in enumerate(policy_module.SKILL_NAMES)
    }
    policy_set = policy_module.FrozenSkillPolicySet(
        policies,
        full_observation_dim=5,
        history_length=3,
        object_sensor_width=2,
    )
    with pytest.raises(ValueError, match="Unknown skill ids"):
        policy_set.route(torch.zeros(1, 15), torch.tensor([3]))


def test_decode_high_level_action_scales_commands_and_zeroes_shoot_yaw() -> None:
    action = torch.tensor(
        [
            [3.0, 1.0, 0.0, 0.5, -0.5, 0.25],
            [0.0, 0.0, 2.0, 0.5, -0.5, 0.25],
            [float("nan"), 0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    scales = torch.tensor([[1.2, 0.6, 1.0], [1.5, 1.5, 1.0], [3.0, 3.0, 0.0]])
    skill_ids, commands, invalid = policy_module.decode_high_level_action(action, scales)
    assert skill_ids.tolist() == [0, 2, 0]
    torch.testing.assert_close(commands[0], torch.tanh(action[0, 3:6]) * scales[0])
    assert commands[1, 2].item() == 0.0
    torch.testing.assert_close(commands[2], torch.zeros(3))
    assert invalid.tolist() == [False, False, True]


def test_decode_fixed_shoot_action_uses_three_value_interface() -> None:
    action = torch.tensor([[0.5, -0.5, 0.75], [float("nan"), 0.0, 0.0]])
    scales = torch.tensor(policy_module.REPRODUCTION_COMMAND_SCALES)
    skill_ids, commands, invalid = policy_module.decode_fixed_skill_action(
        action,
        policy_module.SHOOT_SKILL_ID,
        scales,
    )
    assert skill_ids.tolist() == [policy_module.SHOOT_SKILL_ID, policy_module.SHOOT_SKILL_ID]
    torch.testing.assert_close(commands[0, :2], torch.tanh(action[0, :2]) * scales[2, :2])
    assert commands[0, 2].item() == 0.0
    torch.testing.assert_close(commands[1], torch.zeros(3))
    assert invalid.tolist() == [False, True]


def test_reproduction_command_scales_match_archived_coordinator_contract() -> None:
    assert policy_module.REPRODUCTION_COMMAND_SCALES == (
        (1.5, 1.5, 1.0),
        (1.5, 1.5, 1.0),
        (3.0, 3.0, 0.0),
    )


def test_actor_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    spec = _write_policy(
        tmp_path,
        "broken",
        history_dim=6,
        latent_dim=2,
        action_bias=0.0,
        body_input_dim=7,
    )
    with pytest.raises(ValueError, match=r"history\+latent"):
        policy_module.FrozenLowLevelPolicy.load(spec)


def test_legacy_wrapper_action_history_duplicates_current_policy_output() -> None:
    current = torch.tensor([[1.0, 2.0]])
    previous = torch.tensor([[3.0, 4.0]])
    observed_current, observed_previous = policy_module.observation_action_pair(
        current,
        previous,
        "duplicate_previous_policy_output",
    )
    torch.testing.assert_close(observed_current, current)
    torch.testing.assert_close(observed_previous, current)


def test_successive_action_history_retains_previous_output() -> None:
    current = torch.tensor([[1.0, 2.0]])
    previous = torch.tensor([[3.0, 4.0]])
    observed_current, observed_previous = policy_module.observation_action_pair(
        current,
        previous,
        "successive_policy_outputs",
    )
    torch.testing.assert_close(observed_current, current)
    torch.testing.assert_close(observed_previous, previous)


def _write_policy(
    root: Path,
    name: str,
    history_dim: int,
    latent_dim: int,
    action_bias: float,
    action_clip: float = 1.0,
    body_input_dim: int = None,
):
    directory = root / name
    directory.mkdir(exist_ok=True)
    adaptation = torch.nn.Sequential(torch.nn.Linear(history_dim, latent_dim))
    body = torch.nn.Sequential(torch.nn.Linear(body_input_dim or history_dim + latent_dim, 12))
    with torch.no_grad():
        adaptation[0].weight.zero_()
        adaptation[0].bias.zero_()
        body[0].weight.zero_()
        body[0].bias.fill_(action_bias)
    adaptation_script = torch.jit.trace(adaptation.eval(), torch.zeros(1, history_dim))
    resolved_body_dim = body_input_dim or history_dim + latent_dim
    body_script = torch.jit.trace(body.eval(), torch.zeros(1, resolved_body_dim))
    adaptation_path = directory / "adaptation.jit"
    body_path = directory / "body.jit"
    torch.jit.save(adaptation_script, str(adaptation_path))
    torch.jit.save(body_script, str(body_path))
    return policy_module.FrozenPolicySpec(
        name=name,
        body_path=body_path,
        adaptation_path=adaptation_path,
        action_clip=action_clip,
    )
