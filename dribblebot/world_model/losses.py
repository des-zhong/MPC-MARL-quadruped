"""One-step and scheduled multi-step world-model losses."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn.functional as F

from .ensemble import WorldModelEnsemble


def gaussian_nll(mean: torch.Tensor, log_variance: torch.Tensor, target: torch.Tensor, weights=None) -> torch.Tensor:
    loss = 0.5 * ((target - mean).square() * torch.exp(-log_variance) + log_variance)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def one_step_member_loss(
    model: WorldModelEnsemble,
    member_index: int,
    batch: Mapping[str, torch.Tensor],
    feature_weights: torch.Tensor,
    reward_weight: float = 1.0,
    termination_weight: float = 1.0,
    event_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    member = model.members[member_index]
    states = batch["state"].float()
    actions = batch["joint_action"].float()
    next_states = batch["next_state"].float()
    outputs = member(model.normalizer.normalize_state(states), actions)
    dynamic = model.schema.continuous_dynamic_indices
    delta_target = model.normalizer.normalize_delta_target(next_states[:, dynamic] - states[:, dynamic])
    state_loss = gaussian_nll(outputs["delta_mean"], outputs["delta_log_variance"], delta_target, feature_weights)
    reward_target = model.normalizer.normalize_reward(batch["reward"].float().reshape(-1))
    reward_loss = gaussian_nll(outputs["reward_mean"], outputs["reward_log_variance"], reward_target)
    binary_target = next_states[:, model.schema.binary_dynamic_indices]
    binary_loss = F.binary_cross_entropy_with_logits(outputs["binary_logits"], binary_target)
    terminated = batch["terminated"].float().reshape(-1)
    truncated = batch["truncated"].float().reshape(-1)
    termination_loss = F.binary_cross_entropy_with_logits(outputs["termination_logit"], terminated)
    truncation_loss = F.binary_cross_entropy_with_logits(outputs["truncation_logit"], truncated)
    event_loss = F.binary_cross_entropy_with_logits(outputs["event_logits"], batch["event_labels"].float())
    total = state_loss + binary_loss + reward_weight * reward_loss + termination_weight * (termination_loss + truncation_loss) + event_weight * event_loss
    return {
        "loss": total, "state_nll": state_loss, "binary_bce": binary_loss,
        "reward_nll": reward_loss, "termination_bce": termination_loss,
        "truncation_bce": truncation_loss, "event_bce": event_loss,
    }


def feature_group_weights(schema, config: Mapping[str, float], device) -> torch.Tensor:
    weights = torch.ones(len(schema.continuous_dynamic_indices), device=device)
    mapping = {
        "robot_position": "robot_position_weight", "robot_velocity": "robot_velocity_weight",
        "robot_orientation": "robot_orientation_weight", "ball_position": "ball_position_weight",
        "ball_velocity": "ball_velocity_weight",
    }
    for group, key in mapping.items():
        positions = schema.group_positions(group)
        if positions:
            weights[positions] = float(config.get(key, 1.0))
    return weights


def multi_step_member_loss(
    model: WorldModelEnsemble,
    member_index: int,
    sequence: Mapping[str, torch.Tensor],
    teacher_forcing_probability: float,
    discount: float = 0.8,
) -> torch.Tensor:
    """Differentiable autoregressive loss on real contiguous subsequences."""

    member = model.members[member_index]
    dynamic = model.schema.continuous_dynamic_indices
    current = sequence["state"][:, 0].float()
    alive = torch.ones(current.shape[0], device=current.device)
    total = current.new_tensor(0.0)
    total_weight = current.new_tensor(0.0)
    horizon = sequence["state"].shape[1]
    for step in range(horizon):
        action = sequence["joint_action"][:, step].float()
        target_next = sequence["next_state"][:, step].float()
        outputs = member(model.normalizer.normalize_state(current), action)
        delta = model.normalizer.denormalize_delta_prediction(outputs["delta_mean"])
        binary_probability = torch.sigmoid(outputs["binary_logits"])
        predicted_next = model.state_adapter.apply_predicted_delta(current, delta, binary_probability, action, True)
        normalized_error = (predicted_next[:, dynamic] - target_next[:, dynamic]) / model.normalizer.delta_std
        reward = model.normalizer.denormalize_reward(outputs["reward_mean"])
        reward_error = reward - sequence["reward"][:, step].float()
        step_weight = (float(discount) ** step) * alive
        step_error = normalized_error.square().mean(-1) + reward_error.square()
        total = total + (step_error * step_weight).sum()
        total_weight = total_weight + step_weight.sum()
        terminal = sequence["terminated"][:, step].bool() | sequence["truncated"][:, step].bool()
        alive = alive * (~terminal).float()
        teacher = torch.rand(current.shape[0], device=current.device) < float(teacher_forcing_probability)
        current = torch.where(teacher[:, None], target_next, predicted_next)
    return total / total_weight.clamp(min=1.0)
