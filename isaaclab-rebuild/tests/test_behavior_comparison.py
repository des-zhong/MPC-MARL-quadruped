"""Tests for frozen-skill behavior comparison gates."""

from behavior_fixtures import clone_behavior, make_behavior_trace
from parity.behavior_comparison import compare_behavior_traces


def test_behavior_comparison_aligns_joint_names_and_origins() -> None:
    reference_metadata, reference = make_behavior_trace("isaacgym")
    candidate_metadata, candidate = clone_behavior(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate_metadata["joint_names"] = list(reversed(candidate_metadata["joint_names"]))
    for name in ("policy_action", "processed_joint_target", "joint_pos"):
        candidate[name] = candidate[name][..., ::-1]
    shift = candidate["env_origin_w"].copy()
    shift[:] = (3.0, -2.0, 0.0)
    candidate["env_origin_w"] += shift
    candidate["root_pos_w"] += shift[None]
    candidate["ball_pos_w"] += shift[None]

    report = compare_behavior_traces(reference_metadata, reference, candidate_metadata, candidate)

    assert report["passed"]
    assert report["metrics"]["root_position_rmse_m"] == 0.0


def test_behavior_comparison_hard_gates_action_history_semantics() -> None:
    reference_metadata, reference = make_behavior_trace("isaacgym")
    candidate_metadata, candidate = clone_behavior(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate_metadata["action_history_semantics"] = "successive_policy_outputs"

    report = compare_behavior_traces(reference_metadata, reference, candidate_metadata, candidate)

    assert not report["passed"]
    assert not report["gates"]["action_history_semantics_match"]["passed"]
