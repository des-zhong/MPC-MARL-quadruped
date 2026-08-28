"""Tests for deterministic and name-stable rollout excitation."""

import numpy as np
import pytest

from parity.excitation import ExcitationConfig, generate_action, generate_action_sequence


def test_zero_excitation_has_expected_shape_and_dtype() -> None:
    action = generate_action(3, 0.02, 4, ["joint_a", "joint_b"], ExcitationConfig(kind="zero"))
    assert action.shape == (4, 2)
    assert action.dtype == np.float32
    assert np.count_nonzero(action) == 0


def test_sine_excitation_follows_names_across_reordering() -> None:
    config = ExcitationConfig(kind="sine", amplitude=0.4, frequency_hz=1.2, phase_stride_rad=0.5)
    names_a = ["joint_c", "joint_a", "joint_b"]
    names_b = ["joint_b", "joint_c", "joint_a"]
    action_a = generate_action(7, 0.02, 1, names_a, config)[0]
    action_b = generate_action(7, 0.02, 1, names_b, config)[0]

    by_name_a = dict(zip(names_a, action_a))
    by_name_b = dict(zip(names_b, action_b))
    assert by_name_a == pytest.approx(by_name_b)


def test_action_sequence_is_repeatable() -> None:
    config = ExcitationConfig(kind="sine")
    first = generate_action_sequence(8, 0.02, 2, ["a", "b"], config)
    second = generate_action_sequence(8, 0.02, 2, ["a", "b"], config)
    np.testing.assert_array_equal(first, second)
