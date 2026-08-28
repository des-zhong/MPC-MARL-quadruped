"""Tests for the versioned parity rollout archive."""

from __future__ import annotations

import numpy as np
import pytest

from parity.physics_profile import make_physics_profile
from parity.schema import load_rollout, save_rollout, validate_rollout
from parity_fixtures import add_policy_contract, clone_rollout, make_rollout


def test_rollout_round_trip_without_pickle(tmp_path) -> None:
    metadata, arrays = make_rollout()
    output = save_rollout(tmp_path / "rollout.npz", metadata, arrays)
    loaded_metadata, loaded_arrays = load_rollout(output)

    assert loaded_metadata == metadata
    assert loaded_arrays.keys() == arrays.keys()
    for name in arrays:
        np.testing.assert_array_equal(loaded_arrays[name], arrays[name])


def test_schema_rejects_wrong_joint_width() -> None:
    metadata, arrays = make_rollout()
    _, invalid = clone_rollout(metadata, arrays)
    invalid["joint_pos"] = invalid["joint_pos"][..., :1]

    with pytest.raises(ValueError, match="joint_pos has shape"):
        validate_rollout(metadata, invalid)


def test_schema_requires_complete_ball_state() -> None:
    metadata, arrays = make_rollout()
    arrays["ball_pos_w"] = arrays["root_pos_w"].copy()

    with pytest.raises(ValueError, match="ball rollout is incomplete"):
        validate_rollout(metadata, arrays)


def test_schema_requires_ball_profile_to_match_signals() -> None:
    metadata, arrays = make_rollout()
    metadata["physics_profile"] = make_physics_profile(include_ball=True)

    with pytest.raises(ValueError, match="must agree with the presence of ball state arrays"):
        validate_rollout(metadata, arrays)


def test_schema_rejects_invalid_physics_profile() -> None:
    metadata, arrays = make_rollout()
    metadata["physics_profile"]["solver"]["position_iterations"] = -1

    with pytest.raises(ValueError, match="position_iterations"):
        validate_rollout(metadata, arrays)


def test_schema_accepts_complete_policy_contract() -> None:
    metadata, arrays = make_rollout()
    add_policy_contract(metadata, arrays)

    validate_rollout(metadata, arrays)


def test_schema_requires_both_policy_contract_arrays() -> None:
    metadata, arrays = make_rollout()
    add_policy_contract(metadata, arrays)
    del arrays["legacy_policy_history"]

    with pytest.raises(ValueError, match="policy-contract rollout is incomplete"):
        validate_rollout(metadata, arrays)


def test_schema_rejects_wrong_policy_observation_width() -> None:
    metadata, arrays = make_rollout()
    add_policy_contract(metadata, arrays)
    arrays["legacy_policy_observation"] = arrays["legacy_policy_observation"][..., :-1]

    with pytest.raises(ValueError, match="legacy_policy_observation has shape"):
        validate_rollout(metadata, arrays)
