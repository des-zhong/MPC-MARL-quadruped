"""Tests for frozen-skill behavior trace validation."""

import pytest

from behavior_fixtures import clone_behavior, make_behavior_trace
from parity.behavior_schema import load_behavior_trace, save_behavior_trace, validate_behavior_trace


def test_behavior_trace_round_trip(tmp_path) -> None:
    metadata, arrays = make_behavior_trace()
    path = save_behavior_trace(tmp_path / "behavior.npz", metadata, arrays)
    loaded_metadata, loaded_arrays = load_behavior_trace(path)
    assert loaded_metadata == metadata
    assert loaded_arrays.keys() == arrays.keys()


def test_behavior_trace_requires_observation_to_match_history_tail() -> None:
    metadata, arrays = make_behavior_trace()
    arrays["policy_observation"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="newest frame"):
        validate_behavior_trace(metadata, arrays)


def test_behavior_trace_rejects_incomplete_checkpoint_identity() -> None:
    metadata, arrays = make_behavior_trace()
    del metadata["checkpoint_contract"]["walk"]["body_sha256"]
    with pytest.raises(ValueError, match="must contain"):
        validate_behavior_trace(metadata, arrays)
