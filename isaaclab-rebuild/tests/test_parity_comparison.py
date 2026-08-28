"""Tests for named alignment and parity acceptance gates."""

from __future__ import annotations

import numpy as np
import pytest

from parity.comparison import compare_rollouts
from parity.physics_profile import make_physics_profile
from parity_fixtures import add_policy_contract, clone_rollout, make_rollout


def test_comparison_aligns_names_origins_and_quaternion_signs() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate_metadata["joint_names"] = list(reversed(candidate_metadata["joint_names"]))
    candidate_metadata["action_names"] = list(reversed(candidate_metadata["action_names"]))
    candidate_metadata["foot_names"] = list(reversed(candidate_metadata["foot_names"]))
    candidate["action"] = candidate["action"][..., ::-1]
    for name in ("joint_pos", "joint_vel", "joint_pos_target", "applied_torque"):
        candidate[name] = candidate[name][..., ::-1]
    for name in ("foot_pos_w", "foot_force_w", "foot_contact"):
        candidate[name] = candidate[name][:, :, ::-1]

    origin_shift = np.asarray([[3.0, -2.0, 0.0], [3.0, -2.0, 0.0]], dtype=np.float32)
    candidate["env_origin_w"] += origin_shift
    candidate["root_pos_w"] += origin_shift[None]
    candidate["foot_pos_w"] += origin_shift[None, :, None, :]
    candidate["root_quat_w"] *= -1.0

    report = compare_rollouts(reference_metadata, reference, candidate_metadata, candidate)

    assert report["passed"]
    assert report["metrics"]["root_position_rmse_m"] == 0.0
    assert report["metrics"]["root_orientation_rmse_rad"] == 0.0
    assert report["metrics"]["joint_position_rmse_rad"] == 0.0
    assert report["metrics"]["foot_position_rmse_m"] == 0.0


def test_comparison_reports_failed_gate() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate["root_pos_w"][..., 0] += 0.1

    report = compare_rollouts(
        reference_metadata,
        reference,
        candidate_metadata,
        candidate,
        thresholds={"root_position_rmse_m": 0.05},
    )

    assert not report["passed"]
    assert not report["gates"]["root_position_rmse_m"]["passed"]


def test_comparison_warns_about_contact_semantics() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate_metadata["contact_force_semantics"] = "normal contact force only"

    report = compare_rollouts(reference_metadata, reference, candidate_metadata, candidate)

    assert report["warnings"]


def test_comparison_includes_optional_ball_metrics() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    reference_metadata["physics_profile"] = make_physics_profile(include_ball=True)
    candidate_metadata["physics_profile"] = make_physics_profile(include_ball=True)
    frames, num_envs = reference["root_pos_w"].shape[:2]
    ball_quat = np.zeros((frames, num_envs, 4), dtype=np.float32)
    ball_quat[..., 0] = 1.0
    for arrays in (reference, candidate):
        arrays["ball_pos_w"] = arrays["root_pos_w"].copy()
        arrays["ball_quat_w"] = ball_quat.copy()
        arrays["ball_lin_vel_w"] = arrays["root_lin_vel_w"].copy()
        arrays["ball_ang_vel_w"] = arrays["root_ang_vel_w"].copy()

    report = compare_rollouts(reference_metadata, reference, candidate_metadata, candidate)

    assert report["metrics"]["ball_position_rmse_m"] == 0.0
    assert report["metrics"]["ball_linear_velocity_rmse_mps"] == 0.0


def test_comparison_rejects_different_joint_sets() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate_metadata["joint_names"] = ["joint_b", "different_joint"]
    candidate_metadata["action_names"] = ["joint_b", "different_joint"]

    with pytest.raises(ValueError, match="joint names differ"):
        compare_rollouts(reference_metadata, reference, candidate_metadata, candidate)


def test_comparison_rejects_different_physics_profiles() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate_metadata["physics_profile"]["ground"]["static_friction"] = 0.8

    with pytest.raises(ValueError, match="different deterministic physics profiles"):
        compare_rollouts(reference_metadata, reference, candidate_metadata, candidate)


def test_comparison_includes_policy_observation_and_history_metrics() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    add_policy_contract(reference_metadata, reference)
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate["legacy_policy_observation"][..., 18:30] += 0.01
    candidate["legacy_policy_history"] += 0.01

    report = compare_rollouts(reference_metadata, reference, candidate_metadata, candidate)

    assert report["passed"]
    assert 0.0 < report["metrics"]["legacy_policy_observation_rmse_normalized"] < 0.01
    assert report["metrics"]["legacy_policy_history_rmse_normalized"] == pytest.approx(0.01, abs=2.0e-8)
    assert "legacy_policy_joint_position_rmse_normalized" in report["metrics"]


def test_comparison_rejects_different_policy_contracts() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    add_policy_contract(reference_metadata, reference)
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate_metadata["policy_contract"]["noise_enabled"] = True

    with pytest.raises(ValueError, match="canonical legacy sensor"):
        compare_rollouts(reference_metadata, reference, candidate_metadata, candidate)


def test_comparison_hard_gates_deterministic_command_slice() -> None:
    reference_metadata, reference = make_rollout("isaacgym")
    add_policy_contract(reference_metadata, reference)
    candidate_metadata, candidate = clone_rollout(reference_metadata, reference)
    candidate_metadata["source"] = "isaaclab"
    candidate["legacy_policy_observation"][..., 3:18] += 1.0e-3

    report = compare_rollouts(reference_metadata, reference, candidate_metadata, candidate)

    assert not report["passed"]
    assert not report["gates"]["legacy_policy_command_rmse_normalized"]["passed"]
