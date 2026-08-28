"""Comparison gates for frozen-skill behavior traces."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .behavior_schema import load_behavior_trace, validate_behavior_trace


DEFAULT_BEHAVIOR_THRESHOLDS = {
    "coordinator_action_rmse": 1.0e-6,
    "skill_id_disagreement_rate": 0.0,
    "skill_command_rmse": 1.0e-6,
    "contract_policy_observation_rmse": 0.06,
    "contract_policy_history_rmse": 0.03,
    "first_policy_history_rmse": 0.01,
    "contract_policy_action_rmse": 0.13,
    "contract_processed_joint_target_rmse_rad": 0.03,
    "joint_position_rmse_rad": 0.10,
    "root_position_rmse_m": 0.05,
    "ball_position_rmse_m": 0.12,
    "done_disagreement_rate": 0.0,
}


def compare_behavior_traces(
    reference_metadata: Mapping[str, Any],
    reference_arrays: Mapping[str, np.ndarray],
    candidate_metadata: Mapping[str, Any],
    candidate_arrays: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    validate_behavior_trace(reference_metadata, reference_arrays)
    validate_behavior_trace(candidate_metadata, candidate_arrays)
    if reference_metadata["schema_version"] != candidate_metadata["schema_version"]:
        raise ValueError("behavior traces use different schema versions")
    if reference_metadata["policy_contract"] != candidate_metadata["policy_contract"]:
        raise ValueError("behavior traces use different policy contracts")
    if reference_metadata["checkpoint_contract"] != candidate_metadata["checkpoint_contract"]:
        raise ValueError("behavior traces use different frozen checkpoints")
    if not np.isclose(reference_metadata["step_dt"], candidate_metadata["step_dt"], atol=1.0e-9):
        raise ValueError("behavior traces use different step times")
    if reference_arrays["coordinator_action"].shape[:2] != candidate_arrays["coordinator_action"].shape[:2]:
        raise ValueError("behavior traces use different step or environment counts")
    reference_horizon = int(reference_metadata.get("contract_horizon_steps", 10))
    candidate_horizon = int(candidate_metadata.get("contract_horizon_steps", 10))
    if reference_horizon != candidate_horizon:
        raise ValueError("behavior traces use different contract horizons")

    reference = {name: np.asarray(value) for name, value in reference_arrays.items()}
    candidate = _align_candidate(reference_metadata, candidate_metadata, candidate_arrays)
    horizon = min(reference_horizon, reference["coordinator_action"].shape[0])
    prefix = slice(0, horizon)
    metrics = {
        "coordinator_action_rmse": _rmse(reference["coordinator_action"], candidate["coordinator_action"]),
        "skill_id_disagreement_rate": float(np.mean(reference["skill_id"] != candidate["skill_id"])),
        "skill_command_rmse": _rmse(reference["skill_command"], candidate["skill_command"]),
        "policy_observation_rmse": _rmse(reference["policy_observation"], candidate["policy_observation"]),
        "policy_history_rmse": _rmse(reference["policy_history"], candidate["policy_history"]),
        "contract_policy_observation_rmse": _rmse(
            reference["policy_observation"][prefix], candidate["policy_observation"][prefix]
        ),
        "contract_policy_history_rmse": _rmse(
            reference["policy_history"][prefix], candidate["policy_history"][prefix]
        ),
        "first_policy_history_rmse": _rmse(reference["policy_history"][0], candidate["policy_history"][0]),
        "policy_action_rmse": _rmse(reference["policy_action"], candidate["policy_action"]),
        "contract_policy_action_rmse": _rmse(
            reference["policy_action"][prefix], candidate["policy_action"][prefix]
        ),
        "processed_joint_target_rmse_rad": _rmse(
            reference["processed_joint_target"], candidate["processed_joint_target"]
        ),
        "contract_processed_joint_target_rmse_rad": _rmse(
            reference["processed_joint_target"][prefix], candidate["processed_joint_target"][prefix]
        ),
        "joint_position_rmse_rad": _rmse(reference["joint_pos"], candidate["joint_pos"]),
        "root_position_rmse_m": _relative_position_rmse(reference, candidate, "root_pos_w"),
        "ball_position_rmse_m": _relative_position_rmse(reference, candidate, "ball_pos_w"),
        "done_disagreement_rate": float(np.mean(reference["done"] != candidate["done"])),
    }
    segments = {segment["name"]: segment for segment in reference_metadata["policy_contract"]["observation_segments"]}
    for name in ("current_action", "previous_action"):
        segment = segments[name]
        metrics[f"observed_{name}_rmse"] = _rmse(
            reference["policy_observation"][..., segment["start"] : segment["stop"]],
            candidate["policy_observation"][..., segment["start"] : segment["stop"]],
        )

    selected = dict(DEFAULT_BEHAVIOR_THRESHOLDS if thresholds is None else thresholds)
    gates = {}
    for name, limit in selected.items():
        actual = metrics[name]
        gates[name] = {"actual": actual, "limit": float(limit), "passed": actual <= float(limit)}
    semantics_match = reference_metadata["action_history_semantics"] == candidate_metadata["action_history_semantics"]
    gates["action_history_semantics_match"] = {
        "actual": semantics_match,
        "expected": True,
        "passed": semantics_match,
    }
    return {
        "schema_version": reference_metadata["schema_version"],
        "reference": {"source": reference_metadata["source"], "task": reference_metadata["task"]},
        "candidate": {"source": candidate_metadata["source"], "task": candidate_metadata["task"]},
        "metrics": metrics,
        "gates": gates,
        "semantics": {
            "reference": reference_metadata["action_history_semantics"],
            "candidate": candidate_metadata["action_history_semantics"],
        },
        "contract_horizon_steps": horizon,
        "passed": all(bool(gate["passed"]) for gate in gates.values()),
    }


def compare_behavior_files(reference_path: str, candidate_path: str) -> dict[str, Any]:
    reference_metadata, reference_arrays = load_behavior_trace(reference_path)
    candidate_metadata, candidate_arrays = load_behavior_trace(candidate_path)
    return compare_behavior_traces(reference_metadata, reference_arrays, candidate_metadata, candidate_arrays)


def _align_candidate(
    reference_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    reference_names = list(reference_metadata["joint_names"])
    candidate_names = list(candidate_metadata["joint_names"])
    if set(reference_names) != set(candidate_names):
        raise ValueError("behavior traces use different joint sets")
    index = {name: position for position, name in enumerate(candidate_names)}
    joint_indices = [index[name] for name in reference_names]
    aligned = {name: np.asarray(value) for name, value in arrays.items()}
    for name in ("policy_action", "processed_joint_target", "joint_pos"):
        aligned[name] = aligned[name][..., joint_indices]
    return aligned


def _rmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(reference - candidate))))


def _relative_position_rmse(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    name: str,
) -> float:
    reference_relative = reference[name] - reference["env_origin_w"][None]
    candidate_relative = candidate[name] - candidate["env_origin_w"][None]
    return _rmse(reference_relative, candidate_relative)
