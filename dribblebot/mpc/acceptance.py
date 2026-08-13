"""World-model evaluation and acceptance gates for iterative MPC collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class ModelAcceptanceConfig:
    max_original_validation_degradation_fraction: float = 0.05
    require_recent_validation_improvement: bool = True
    max_ball_rollout_error_degradation_fraction: float = 0.03
    max_reward_error_degradation_fraction: float = 0.05
    max_termination_brier_degradation_fraction: float = 0.05
    minimum_uncertainty_ratio: float = 0.10
    maximum_uncertainty_ratio: float = 10.0
    require_finite_horizon_20_rollout: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]):
        unknown = sorted(set(mapping) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown model-acceptance settings: {unknown}")
        result = cls(**dict(mapping))
        for name, value in asdict(result).items():
            if isinstance(value, float) and value < 0:
                raise ValueError(f"model_acceptance.{name} cannot be negative")
        return result


def _allowed(new: float, old: float, fraction: float) -> bool:
    scale = max(abs(float(old)), 1.0e-8)
    return float(new) <= float(old) + fraction * scale


class ModelAcceptanceGate:
    def __init__(self, config: ModelAcceptanceConfig):
        self.config = config

    def compare(
        self,
        old_metrics: Mapping[str, Mapping[str, float]],
        new_metrics: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, object]:
        old_original, new_original = old_metrics["original"], new_metrics["original"]
        old_recent, new_recent = old_metrics["recent"], new_metrics["recent"]
        checks = {
            "original_validation": _allowed(
                new_original["normalized_state_rmse"],
                old_original["normalized_state_rmse"],
                self.config.max_original_validation_degradation_fraction,
            ),
            "recent_validation": (
                new_recent["normalized_state_rmse"]
                <= old_recent["normalized_state_rmse"]
                if self.config.require_recent_validation_improvement
                else True
            ),
            "ball_rollout": _allowed(
                new_original["ball_rollout_rmse"],
                old_original["ball_rollout_rmse"],
                self.config.max_ball_rollout_error_degradation_fraction,
            ),
            "reward": _allowed(
                new_original["reward_rmse"],
                old_original["reward_rmse"],
                self.config.max_reward_error_degradation_fraction,
            ),
            "termination_calibration": _allowed(
                new_original["termination_brier"],
                old_original["termination_brier"],
                self.config.max_termination_brier_degradation_fraction,
            ),
            "finite_rollout": (
                bool(new_original["finite_horizon_rollout"])
                if self.config.require_finite_horizon_20_rollout
                else True
            ),
        }
        old_uncertainty = max(old_original["mean_state_uncertainty"], 1.0e-12)
        ratio = new_original["mean_state_uncertainty"] / old_uncertainty
        checks["uncertainty_not_collapsed"] = (
            self.config.minimum_uncertainty_ratio
            <= ratio
            <= self.config.maximum_uncertainty_ratio
        )
        finite_metrics = all(
            np.isfinite(float(value))
            for group in new_metrics.values()
            for value in group.values()
            if not isinstance(value, bool)
        )
        checks["all_metrics_finite"] = finite_metrics
        failed = [name for name, passed in checks.items() if not passed]
        return {
            "accepted": not failed,
            "checks": checks,
            "failed_criteria": failed,
            "uncertainty_ratio": float(ratio),
            "old_metrics": dict(old_metrics),
            "new_metrics": dict(new_metrics),
        }


@torch.no_grad()
def evaluate_model_for_acceptance(
    model,
    dataset,
    device,
    batch_size: int = 2048,
    rollout_horizon: int = 20,
    max_sequences: int = 256,
) -> Dict[str, float]:
    model.eval()
    dynamic = model.schema.continuous_dynamic_indices
    ball = [
        index
        for feature in model.schema.features
        if feature.group == "ball_position"
        for index in range(feature.start, feature.stop)
    ]
    state_sse = reward_sse = done_sse = uncertainty_sum = 0.0
    state_count = reward_count = done_count = uncertainty_count = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        state = batch["state"].float().to(device)
        action = batch["joint_action"].float().to(device)
        target = batch["next_state"].float().to(device)
        next_state, reward, done, _, uncertainty = model.predict_next(
            state, action, deterministic=True
        )
        normalized_error = (
            next_state[:, dynamic] - target[:, dynamic]
        ) / model.normalizer.state_std.to(device)[dynamic].clamp(min=1.0e-6)
        state_sse += float(normalized_error.square().sum())
        state_count += normalized_error.numel()
        reward_error = reward - batch["reward"].float().to(device)
        reward_sse += float(reward_error.square().sum())
        reward_count += reward_error.numel()
        done_target = (
            batch["terminated"].bool() | batch["truncated"].bool()
        ).float().to(device)
        done_sse += float((done - done_target).square().sum())
        done_count += done.numel()
        values = uncertainty["mean_state_uncertainty"]
        uncertainty_sum += float(values.sum())
        uncertainty_count += values.numel()

    available = dataset.sequences(rollout_horizon)
    if not available:
        shorter = max(
            [length for length in range(rollout_horizon - 1, 0, -1) if dataset.sequences(length)],
            default=0,
        )
        rollout_horizon = shorter
        available = dataset.sequences(shorter) if shorter else []
    ball_errors = []
    finite = True
    for episode_index, start in available[:max_sequences]:
        sequence = dataset.get_sequence(episode_index, start, rollout_horizon)
        initial = sequence["state"][0:1].float().to(device)
        actions = sequence["joint_action"][None, None].float().to(device)
        rollout = model.rollout(initial, actions, deterministic=True)
        prediction = rollout["predicted_states"][0, 0, 1:, ball]
        target = sequence["next_state"].float().to(device)[:, ball]
        finite &= bool(torch.isfinite(prediction).all().item())
        ball_errors.append(float((prediction - target).square().mean().sqrt()))
    return {
        "normalized_state_rmse": float(np.sqrt(state_sse / max(state_count, 1))),
        "reward_rmse": float(np.sqrt(reward_sse / max(reward_count, 1))),
        "termination_brier": float(done_sse / max(done_count, 1)),
        "mean_state_uncertainty": float(
            uncertainty_sum / max(uncertainty_count, 1)
        ),
        "ball_rollout_rmse": float(np.mean(ball_errors)) if ball_errors else float("inf"),
        "finite_horizon_rollout": bool(finite and bool(ball_errors)),
        "rollout_horizon": int(rollout_horizon),
        "rollout_sequence_count": int(len(ball_errors)),
    }
