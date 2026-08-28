"""Canonical metadata for the legacy low-level policy observation contract."""

from __future__ import annotations

from typing import Any, Mapping


LEGACY_HISTORY_LENGTH = 15
LEGACY_WALK_OBSERVATION_DIM = 72
LEGACY_BALL_OBSERVATION_DIM = 75
LEGACY_PARITY_COMMAND = (
    0.0,  # linear velocity x
    0.0,  # linear velocity y
    0.0,  # yaw velocity
    0.0,  # body height
    3.0,  # gait frequency
    0.5,  # gait phase
    0.0,  # gait offset
    0.0,  # gait bound
    0.5,  # stance duration
    0.09,  # foot swing height
    0.0,  # body pitch
    0.0,  # body roll
    0.0,  # stance width
    0.0,  # stance length
    0.0,  # auxiliary reward coefficient
)

_COMMON_SEGMENTS = (
    ("projected_gravity", 3),
    ("command", 15),
    ("joint_position", 12),
    ("joint_velocity", 12),
    ("current_action", 12),
    ("previous_action", 12),
    ("gait_clock", 4),
    ("heading", 1),
    ("gait_phase", 1),
)


def make_legacy_policy_contract(
    include_ball: bool,
    history_length: int = LEGACY_HISTORY_LENGTH,
) -> dict[str, Any]:
    """Return one name-stable description shared by both simulator adapters."""

    history_length = int(history_length)
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    segment_widths = (("ball_position", 3), *_COMMON_SEGMENTS) if include_ball else _COMMON_SEGMENTS
    start = 0
    segments = []
    for name, width in segment_widths:
        stop = start + width
        segments.append({"name": name, "start": start, "stop": stop})
        start = stop
    expected_dim = LEGACY_BALL_OBSERVATION_DIM if include_ball else LEGACY_WALK_OBSERVATION_DIM
    if start != expected_dim:
        raise RuntimeError(f"Policy contract built {start} values, expected {expected_dim}")
    return {
        "include_ball": bool(include_ball),
        "noise_enabled": False,
        "observation_dim": expected_dim,
        "history_length": history_length,
        "history_dim": expected_dim * history_length,
        "history_order": "oldest_to_newest_zero_padded",
        "unscaled_command": list(LEGACY_PARITY_COMMAND),
        "observation_segments": segments,
    }


def validate_legacy_policy_contract(profile: Mapping[str, Any]) -> None:
    """Reject incomplete or reordered policy-contract metadata."""

    if not isinstance(profile, Mapping):
        raise ValueError("policy_contract must be a mapping")
    include_ball = profile.get("include_ball")
    if not isinstance(include_ball, bool):
        raise ValueError("policy_contract.include_ball must be boolean")
    try:
        history_length = int(profile["history_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("policy_contract.history_length must be a positive integer") from exc
    expected = make_legacy_policy_contract(include_ball, history_length=history_length)
    if dict(profile) != expected:
        raise ValueError("policy_contract does not match the canonical legacy sensor/history layout")
