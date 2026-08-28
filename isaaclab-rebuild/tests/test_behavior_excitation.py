"""Tests for deterministic frozen-skill coordinator inputs."""

import numpy as np

from parity.behavior_excitation import generate_coordinator_sequence


def test_switch_sequence_cycles_skills_at_fixed_interval() -> None:
    sequence = generate_coordinator_sequence(7, 2, "switch", switch_interval=2)
    skill_ids = np.argmax(sequence[:, 0, :3], axis=-1)
    assert skill_ids.tolist() == [0, 0, 1, 1, 2, 2, 0]
    np.testing.assert_array_equal(sequence[:, 0, 3:6], sequence[:, 1, 3:6])


def test_fixed_skill_sequence_uses_requested_logit() -> None:
    sequence = generate_coordinator_sequence(3, 1, "shoot")
    assert np.argmax(sequence[..., :3], axis=-1).tolist() == [[2], [2], [2]]
