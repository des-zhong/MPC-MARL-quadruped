"""Simulator-free tests for the legacy low-level observation contract."""

import importlib.util
from pathlib import Path

import torch


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
    / "tasks"
    / "manager_based"
    / "football"
    / "mdp"
    / "legacy_contract.py"
)
SPEC = importlib.util.spec_from_file_location("dribblebot_legacy_contract", CONTRACT_PATH)
legacy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(legacy)


def test_legacy_command_uses_exact_15d_scaling() -> None:
    base = torch.tensor([[1.0, 2.0, 4.0]])
    gait = torch.arange(1.0, 13.0).unsqueeze(0)
    result = legacy.compose_legacy_command(base, gait)
    raw = torch.cat((base, gait), dim=-1)
    expected = raw * torch.tensor(legacy.LEGACY_COMMAND_SCALE)
    torch.testing.assert_close(result, expected)


def test_walking_observation_has_legacy_order_and_width() -> None:
    observation = legacy.assemble_legacy_observation(
        projected_gravity=_filled(3, 1.0),
        scaled_command=_filled(15, 2.0),
        joint_position=_filled(12, 3.0),
        scaled_joint_velocity=_filled(12, 4.0),
        current_action=_filled(12, 5.0),
        previous_action=_filled(12, 6.0),
        gait_clock=_filled(4, 7.0),
        heading=_filled(1, 8.0),
        gait_phase=_filled(1, 9.0),
    )
    assert observation.shape == (1, legacy.LEGACY_WALK_OBSERVATION_DIM)
    torch.testing.assert_close(observation[0, :3], torch.full((3,), 1.0))
    torch.testing.assert_close(observation[0, 3:18], torch.full((15,), 2.0))
    torch.testing.assert_close(observation[0, 18:30], torch.full((12,), 3.0))
    torch.testing.assert_close(observation[0, 30:42], torch.full((12,), 4.0))
    torch.testing.assert_close(observation[0, 42:54], torch.full((12,), 5.0))
    torch.testing.assert_close(observation[0, 54:66], torch.full((12,), 6.0))
    torch.testing.assert_close(observation[0, 66:70], torch.full((4,), 7.0))
    assert observation[0, 70:].tolist() == [8.0, 9.0]


def test_ball_observation_prepends_object_sensor() -> None:
    observation = legacy.assemble_legacy_observation(
        ball_position=_filled(3, 10.0),
        projected_gravity=_filled(3, 1.0),
        scaled_command=_filled(15, 2.0),
        joint_position=_filled(12, 3.0),
        scaled_joint_velocity=_filled(12, 4.0),
        current_action=_filled(12, 5.0),
        previous_action=_filled(12, 6.0),
        gait_clock=_filled(4, 7.0),
        heading=_filled(1, 8.0),
        gait_phase=_filled(1, 9.0),
    )
    assert observation.shape == (1, legacy.LEGACY_BALL_OBSERVATION_DIM)
    torch.testing.assert_close(observation[0, :3], torch.full((3,), 10.0))
    torch.testing.assert_close(observation[0, 3:6], torch.full((3,), 1.0))


def test_noise_only_changes_original_noisy_sensor_segments() -> None:
    torch.manual_seed(7)
    observation = torch.zeros(2, legacy.LEGACY_BALL_OBSERVATION_DIM)
    noisy = legacy.add_legacy_observation_noise(observation, include_ball=True)
    noisy_indices = torch.zeros(legacy.LEGACY_BALL_OBSERVATION_DIM, dtype=torch.bool)
    noisy_indices[:6] = True
    noisy_indices[21:45] = True
    assert torch.count_nonzero(noisy[:, noisy_indices]) > 0
    assert torch.count_nonzero(noisy[:, ~noisy_indices]) == 0
    assert torch.max(torch.abs(noisy[:, :6])) <= 0.05
    assert torch.max(torch.abs(noisy[:, 21:33])) <= 0.01
    assert torch.max(torch.abs(noisy[:, 33:45])) <= 0.075


def test_trot_clock_matches_legacy_phase_mapping() -> None:
    gait = torch.tensor(
        [[0.0, 3.0, 0.5, 0.0, 0.0, 0.5, 0.09, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    clock = legacy.legacy_gait_clock(torch.tensor([0.25]), gait)
    torch.testing.assert_close(clock, torch.tensor([[-1.0, 1.0, 1.0, -1.0]]), atol=1.0e-6, rtol=0.0)


def test_gait_phase_advances_at_policy_dt_and_wraps() -> None:
    phase = torch.tensor([0.90, 0.10])
    frequency = torch.tensor([3.0, 2.0])
    result = legacy.advance_gait_phase(phase, frequency, dt=0.05)
    torch.testing.assert_close(result, torch.tensor([0.05, 0.20]))


def test_history_reset_then_append_matches_wrapper_semantics() -> None:
    history = torch.tensor(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ]
    )
    observation = torch.tensor([[7.0], [8.0]])
    result = legacy.update_zero_padded_history(
        history,
        observation,
        append_mask=torch.tensor([True, True]),
        reset_mask=torch.tensor([False, True]),
    )
    torch.testing.assert_close(result[0, :, 0], torch.tensor([2.0, 3.0, 7.0]))
    torch.testing.assert_close(result[1, :, 0], torch.tensor([0.0, 0.0, 8.0]))
    torch.testing.assert_close(legacy.flatten_history(result), torch.tensor([[2.0, 3.0, 7.0], [0.0, 0.0, 8.0]]))


def test_initial_history_can_remain_all_zero() -> None:
    history = torch.zeros(2, 15, 72)
    observation = torch.ones(2, 72)
    result = legacy.update_zero_padded_history(
        history,
        observation,
        append_mask=torch.tensor([False, False]),
    )
    assert torch.count_nonzero(result) == 0


def _filled(width: int, value: float) -> torch.Tensor:
    return torch.full((1, width), value)
