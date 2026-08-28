"""Named-signal alignment, physics metrics, and parity acceptance gates."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .physics_profile import profiles_equal
from .schema import BALL_SIGNALS, POLICY_CONTRACT_SIGNALS, load_rollout, validate_rollout


DEFAULT_THRESHOLDS: dict[str, float] = {
    "action_rmse_normalized": 1.0e-6,
    "done_disagreement_rate": 0.0,
    "root_position_rmse_m": 0.05,
    "root_orientation_rmse_rad": 0.15,
    "root_linear_velocity_world_rmse_mps": 0.30,
    "root_angular_velocity_world_rmse_radps": 0.60,
    "root_linear_velocity_body_rmse_mps": 0.30,
    "root_angular_velocity_body_rmse_radps": 0.60,
    "joint_position_rmse_rad": 0.10,
    "joint_velocity_rmse_radps": 1.50,
    "joint_target_rmse_rad": 0.01,
    "applied_torque_rmse_nm": 12.0,
    "foot_position_rmse_m": 0.08,
    "foot_contact_disagreement_rate": 0.20,
    "ball_position_rmse_m": 0.12,
    "ball_linear_velocity_rmse_mps": 0.75,
    "legacy_policy_observation_rmse_normalized": 0.05,
    "legacy_policy_history_rmse_normalized": 0.05,
    "legacy_policy_command_rmse_normalized": 1.0e-6,
    "legacy_policy_current_action_rmse_normalized": 1.0e-6,
    "legacy_policy_previous_action_rmse_normalized": 1.0e-6,
    "legacy_policy_gait_clock_rmse_normalized": 1.0e-6,
    "legacy_policy_gait_phase_rmse_normalized": 1.0e-6,
}


def _name_indices(source_names: list[str], target_names: list[str], label: str) -> list[int]:
    if set(source_names) != set(target_names):
        missing = sorted(set(target_names).difference(source_names))
        extra = sorted(set(source_names).difference(target_names))
        raise ValueError(f"{label} names differ; missing={missing}, extra={extra}")
    index_by_name = {name: index for index, name in enumerate(source_names)}
    return [index_by_name[name] for name in target_names]


def _l2_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    error = np.linalg.norm(reference - candidate, axis=-1)
    return float(np.sqrt(np.mean(np.square(error)))), float(np.max(error))


def _scalar_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    error = reference - candidate
    return float(np.sqrt(np.mean(np.square(error)))), float(np.max(np.abs(error)))


def _integrate(signal: np.ndarray, time_s: np.ndarray, axis: int) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(signal, x=time_s, axis=axis)
    return np.trapz(signal, x=time_s, axis=axis)


def _quaternion_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    reference_unit = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
    candidate_unit = candidate / np.linalg.norm(candidate, axis=-1, keepdims=True)
    cosine = np.abs(np.sum(reference_unit * candidate_unit, axis=-1))
    angle = 2.0 * np.arccos(np.clip(cosine, 0.0, 1.0))
    return float(np.sqrt(np.mean(np.square(angle)))), float(np.max(angle))


def _relative_position(signal: np.ndarray, env_origin_w: np.ndarray) -> np.ndarray:
    if signal.ndim == 3:
        return signal - env_origin_w[None, :, :]
    if signal.ndim == 4:
        return signal - env_origin_w[None, :, None, :]
    raise ValueError(f"unsupported position signal rank {signal.ndim}")


def _put_pair(metrics: dict[str, float], prefix: str, units: str, values: tuple[float, float]) -> None:
    rmse, maximum = values
    metrics[f"{prefix}_rmse_{units}"] = rmse
    metrics[f"{prefix}_max_{units}"] = maximum


def _align_candidate(
    reference_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    candidate_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    aligned = {name: np.asarray(value) for name, value in candidate_arrays.items()}
    joint_indices = _name_indices(
        list(candidate_metadata["joint_names"]),
        list(reference_metadata["joint_names"]),
        "joint",
    )
    action_indices = _name_indices(
        list(candidate_metadata["action_names"]),
        list(reference_metadata["action_names"]),
        "action",
    )
    foot_indices = _name_indices(
        list(candidate_metadata["foot_names"]),
        list(reference_metadata["foot_names"]),
        "foot",
    )
    aligned["action"] = aligned["action"][..., action_indices]
    for name in ("joint_pos", "joint_vel", "joint_pos_target", "applied_torque"):
        aligned[name] = aligned[name][..., joint_indices]
    for name in ("foot_pos_w", "foot_force_w", "foot_contact"):
        aligned[name] = np.take(aligned[name], foot_indices, axis=2)
    return aligned


def compare_rollouts(
    reference_metadata: Mapping[str, Any],
    reference_arrays: Mapping[str, np.ndarray],
    candidate_metadata: Mapping[str, Any],
    candidate_arrays: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare two validated rollouts after aligning named actions, joints, and feet."""

    validate_rollout(reference_metadata, reference_arrays)
    validate_rollout(candidate_metadata, candidate_arrays)
    if reference_metadata["schema_version"] != candidate_metadata["schema_version"]:
        raise ValueError("rollouts use different schema versions")
    if not np.isclose(reference_metadata["step_dt"], candidate_metadata["step_dt"], atol=1.0e-9):
        raise ValueError("rollouts use different policy step times")
    if reference_arrays["action"].shape[:2] != candidate_arrays["action"].shape[:2]:
        raise ValueError("rollouts use different step or environment counts")
    if not np.allclose(reference_arrays["time_s"], candidate_arrays["time_s"], atol=1.0e-8):
        raise ValueError("rollout time axes differ")
    if not profiles_equal(reference_metadata["physics_profile"], candidate_metadata["physics_profile"]):
        raise ValueError("rollouts use different deterministic physics profiles")

    reference = {name: np.asarray(value) for name, value in reference_arrays.items()}
    candidate = _align_candidate(reference_metadata, candidate_metadata, candidate_arrays)
    reference_has_ball = all(name in reference for name in BALL_SIGNALS)
    candidate_has_ball = all(name in candidate for name in BALL_SIGNALS)
    if reference_has_ball != candidate_has_ball:
        raise ValueError("one rollout contains a ball while the other does not")
    reference_has_policy = all(name in reference for name in POLICY_CONTRACT_SIGNALS)
    candidate_has_policy = all(name in candidate for name in POLICY_CONTRACT_SIGNALS)
    if reference_has_policy != candidate_has_policy:
        raise ValueError("one rollout contains the legacy policy contract while the other does not")
    if reference_has_policy and reference_metadata["policy_contract"] != candidate_metadata["policy_contract"]:
        raise ValueError("rollouts use different legacy policy contracts")

    metrics: dict[str, float] = {}
    _put_pair(metrics, "action", "normalized", _scalar_metrics(reference["action"], candidate["action"]))
    metrics["done_disagreement_rate"] = float(np.mean(reference["done"] != candidate["done"]))

    ref_root_pos = _relative_position(reference["root_pos_w"], reference["env_origin_w"])
    cand_root_pos = _relative_position(candidate["root_pos_w"], candidate["env_origin_w"])
    _put_pair(metrics, "root_position", "m", _l2_metrics(ref_root_pos, cand_root_pos))
    _put_pair(
        metrics,
        "root_orientation",
        "rad",
        _quaternion_metrics(reference["root_quat_w"], candidate["root_quat_w"]),
    )
    vector_metrics = (
        ("root_lin_vel_w", "root_linear_velocity_world", "mps"),
        ("root_ang_vel_w", "root_angular_velocity_world", "radps"),
        ("root_lin_vel_b", "root_linear_velocity_body", "mps"),
        ("root_ang_vel_b", "root_angular_velocity_body", "radps"),
    )
    for signal_name, metric_name, units in vector_metrics:
        _put_pair(metrics, metric_name, units, _l2_metrics(reference[signal_name], candidate[signal_name]))

    joint_metrics = (
        ("joint_pos", "joint_position", "rad"),
        ("joint_vel", "joint_velocity", "radps"),
        ("joint_pos_target", "joint_target", "rad"),
        ("applied_torque", "applied_torque", "nm"),
    )
    for signal_name, metric_name, units in joint_metrics:
        _put_pair(metrics, metric_name, units, _scalar_metrics(reference[signal_name], candidate[signal_name]))

    ref_foot_pos = _relative_position(reference["foot_pos_w"], reference["env_origin_w"])
    cand_foot_pos = _relative_position(candidate["foot_pos_w"], candidate["env_origin_w"])
    _put_pair(metrics, "foot_position", "m", _l2_metrics(ref_foot_pos, cand_foot_pos))
    metrics["foot_contact_disagreement_rate"] = float(
        np.mean(reference["foot_contact"] != candidate["foot_contact"])
    )

    time_s = reference["time_s"]
    ref_normal_force = np.maximum(reference["foot_force_w"][..., 2], 0.0)
    cand_normal_force = np.maximum(candidate["foot_force_w"][..., 2], 0.0)
    ref_impulse = _integrate(ref_normal_force, time_s, axis=0)
    cand_impulse = _integrate(cand_normal_force, time_s, axis=0)
    _put_pair(metrics, "foot_normal_impulse", "ns", _scalar_metrics(ref_impulse, cand_impulse))
    metrics["reference_foot_normal_impulse_mean_ns"] = float(np.mean(ref_impulse))
    metrics["candidate_foot_normal_impulse_mean_ns"] = float(np.mean(cand_impulse))

    ref_power = np.sum(np.abs(reference["applied_torque"] * reference["joint_vel"]), axis=-1)
    cand_power = np.sum(np.abs(candidate["applied_torque"] * candidate["joint_vel"]), axis=-1)
    ref_work = _integrate(ref_power, time_s, axis=0)
    cand_work = _integrate(cand_power, time_s, axis=0)
    metrics["reference_absolute_actuator_work_mean_j"] = float(np.mean(ref_work))
    metrics["candidate_absolute_actuator_work_mean_j"] = float(np.mean(cand_work))
    work_denominator = max(float(np.mean(np.abs(ref_work))), 1.0e-9)
    metrics["absolute_actuator_work_relative_error"] = float(
        np.mean(np.abs(cand_work - ref_work)) / work_denominator
    )

    if reference_has_ball:
        ref_ball_pos = _relative_position(reference["ball_pos_w"], reference["env_origin_w"])
        cand_ball_pos = _relative_position(candidate["ball_pos_w"], candidate["env_origin_w"])
        _put_pair(metrics, "ball_position", "m", _l2_metrics(ref_ball_pos, cand_ball_pos))
        _put_pair(
            metrics,
            "ball_orientation",
            "rad",
            _quaternion_metrics(reference["ball_quat_w"], candidate["ball_quat_w"]),
        )
        _put_pair(
            metrics,
            "ball_linear_velocity",
            "mps",
            _l2_metrics(reference["ball_lin_vel_w"], candidate["ball_lin_vel_w"]),
        )
        _put_pair(
            metrics,
            "ball_angular_velocity",
            "radps",
            _l2_metrics(reference["ball_ang_vel_w"], candidate["ball_ang_vel_w"]),
        )

    if reference_has_policy:
        _put_pair(
            metrics,
            "legacy_policy_observation",
            "normalized",
            _scalar_metrics(
                reference["legacy_policy_observation"],
                candidate["legacy_policy_observation"],
            ),
        )
        _put_pair(
            metrics,
            "legacy_policy_history",
            "normalized",
            _scalar_metrics(
                reference["legacy_policy_history"],
                candidate["legacy_policy_history"],
            ),
        )
        for segment in reference_metadata["policy_contract"]["observation_segments"]:
            name = segment["name"]
            start = int(segment["start"])
            stop = int(segment["stop"])
            _put_pair(
                metrics,
                f"legacy_policy_{name}",
                "normalized",
                _scalar_metrics(
                    reference["legacy_policy_observation"][..., start:stop],
                    candidate["legacy_policy_observation"][..., start:stop],
                ),
            )

    selected_thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    gates: dict[str, dict[str, float | bool]] = {}
    for metric_name, limit in selected_thresholds.items():
        if metric_name not in metrics:
            continue
        numeric_limit = float(limit)
        if not np.isfinite(numeric_limit) or numeric_limit < 0.0:
            raise ValueError(f"threshold for {metric_name} must be finite and non-negative")
        actual = metrics[metric_name]
        gates[metric_name] = {
            "actual": actual,
            "limit": numeric_limit,
            "passed": actual <= numeric_limit,
        }

    warnings = []
    if reference_metadata["contact_force_semantics"] != candidate_metadata["contact_force_semantics"]:
        warnings.append(
            "Contact-force semantics differ; use contact state and normal impulse as the primary contact checks."
        )
    return {
        "schema_version": reference_metadata["schema_version"],
        "reference": {
            "source": reference_metadata["source"],
            "task": reference_metadata["task"],
        },
        "candidate": {
            "source": candidate_metadata["source"],
            "task": candidate_metadata["task"],
        },
        "metrics": metrics,
        "gates": gates,
        "warnings": warnings,
        "passed": all(bool(gate["passed"]) for gate in gates.values()),
    }


def compare_rollout_files(
    reference_path: str,
    candidate_path: str,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Load two archives and return their comparison report."""

    reference_metadata, reference_arrays = load_rollout(reference_path)
    candidate_metadata, candidate_arrays = load_rollout(candidate_path)
    return compare_rollouts(
        reference_metadata,
        reference_arrays,
        candidate_metadata,
        candidate_arrays,
        thresholds=thresholds,
    )
