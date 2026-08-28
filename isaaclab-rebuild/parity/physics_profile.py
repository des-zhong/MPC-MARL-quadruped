"""Deterministic physics parameters shared by both parity recorders."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np


PARITY_PHYSICS_PROFILE: dict[str, Any] = {
    "gravity_mps2": [0.0, 0.0, -9.81],
    "solver": {
        "type": "tgs",
        "position_iterations": 4,
        "velocity_iterations": 0,
        "contact_offset_m": 0.01,
        "rest_offset_m": 0.0,
        "bounce_threshold_velocity_mps": 0.5,
        "max_depenetration_velocity_mps": 1.0,
    },
    "ground": {
        "static_friction": 1.0,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
    },
    "ball": {
        "radius_m": 0.0889,
        "mass_kg": 0.318,
        # These are the defaults of Isaac Gym Preview 4 AssetOptions used by
        # the legacy sphere asset.  They are made explicit on both sides so a
        # simulator release cannot silently change the comparison contract.
        "linear_damping": 0.0,
        "angular_damping": 0.5,
        "max_linear_velocity_mps": 1000.0,
        "max_angular_velocity_radps": 64.0,
        "enable_gyroscopic_forces": True,
        "static_friction": 1.0,
        "dynamic_friction": 1.0,
        "restitution": 0.0,
    },
}


def make_physics_profile(include_ball: bool) -> dict[str, Any]:
    """Return a JSON-safe copy of the deterministic parity profile."""

    profile = copy.deepcopy(PARITY_PHYSICS_PROFILE)
    if not include_ball:
        profile["ball"] = None
    return profile


def validate_physics_profile(profile: object) -> None:
    """Validate the small, versioned profile stored in rollout metadata."""

    if not isinstance(profile, Mapping):
        raise ValueError("physics_profile must be a mapping")
    expected_keys = {"gravity_mps2", "solver", "ground", "ball"}
    if set(profile) != expected_keys:
        raise ValueError(f"physics_profile keys must be {sorted(expected_keys)}")

    gravity = profile["gravity_mps2"]
    if not isinstance(gravity, list) or len(gravity) != 3:
        raise ValueError("physics_profile.gravity_mps2 must contain three values")
    _finite_values("physics_profile.gravity_mps2", gravity)

    solver = _require_mapping(profile["solver"], "physics_profile.solver")
    _require_exact_keys(
        solver,
        {
            "type",
            "position_iterations",
            "velocity_iterations",
            "contact_offset_m",
            "rest_offset_m",
            "bounce_threshold_velocity_mps",
            "max_depenetration_velocity_mps",
        },
        "physics_profile.solver",
    )
    if solver["type"] not in {"tgs", "pgs"}:
        raise ValueError("physics_profile.solver.type must be 'tgs' or 'pgs'")
    for name in ("position_iterations", "velocity_iterations"):
        value = solver[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"physics_profile.solver.{name} must be a non-negative integer")
    _finite_values(
        "physics_profile.solver",
        [
            solver["contact_offset_m"],
            solver["rest_offset_m"],
            solver["bounce_threshold_velocity_mps"],
            solver["max_depenetration_velocity_mps"],
        ],
        non_negative=True,
    )

    ground = _require_mapping(profile["ground"], "physics_profile.ground")
    _require_exact_keys(ground, {"static_friction", "dynamic_friction", "restitution"}, "physics_profile.ground")
    _finite_values("physics_profile.ground", ground.values(), non_negative=True)

    ball = profile["ball"]
    if ball is None:
        return
    ball = _require_mapping(ball, "physics_profile.ball")
    _require_exact_keys(
        ball,
        {
            "radius_m",
            "mass_kg",
            "linear_damping",
            "angular_damping",
            "max_linear_velocity_mps",
            "max_angular_velocity_radps",
            "enable_gyroscopic_forces",
            "static_friction",
            "dynamic_friction",
            "restitution",
        },
        "physics_profile.ball",
    )
    if not isinstance(ball["enable_gyroscopic_forces"], bool):
        raise ValueError("physics_profile.ball.enable_gyroscopic_forces must be boolean")
    _finite_values(
        "physics_profile.ball",
        [value for name, value in ball.items() if name != "enable_gyroscopic_forces"],
        non_negative=True,
    )
    if float(ball["radius_m"]) <= 0.0 or float(ball["mass_kg"]) <= 0.0:
        raise ValueError("physics_profile ball radius and mass must be positive")


def profiles_equal(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Compare nested JSON profiles while tolerating harmless float encoding."""

    validate_physics_profile(reference)
    validate_physics_profile(candidate)
    return _nested_equal(reference, candidate)


def _nested_equal(reference: object, candidate: object) -> bool:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        return set(reference) == set(candidate) and all(
            _nested_equal(reference[key], candidate[key]) for key in reference
        )
    if isinstance(reference, list) and isinstance(candidate, list):
        return len(reference) == len(candidate) and all(
            _nested_equal(ref_value, cand_value) for ref_value, cand_value in zip(reference, candidate)
        )
    if isinstance(reference, bool) or isinstance(candidate, bool):
        return reference is candidate
    if isinstance(reference, (int, float)) and isinstance(candidate, (int, float)):
        return bool(np.isclose(float(reference), float(candidate), rtol=1.0e-9, atol=1.0e-12))
    return reference == candidate


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} keys must be {sorted(expected)}")


def _finite_values(name: str, values: object, non_negative: bool = False) -> None:
    try:
        numeric = np.asarray(list(values), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values") from error
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if non_negative and np.any(numeric < 0.0):
        raise ValueError(f"{name} values must be non-negative")
