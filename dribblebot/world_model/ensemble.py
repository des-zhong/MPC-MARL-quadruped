"""Ensemble prediction, uncertainty, and MPC-ready vectorized rollout."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch
from torch import nn

from .action_adapter import JointActionAdapter
from .model import WorldModelMember
from .normalizer import WorldModelNormalizer
from .schema import EVENT_NAMES, StateSchema
from .state_adapter import FootballWorldModelStateAdapter


class WorldModelEnsemble(nn.Module):
    """Probabilistic ensemble at one high-level macro-step timescale."""

    def __init__(
        self,
        schema: StateSchema,
        action_adapter: JointActionAdapter,
        normalizer: WorldModelNormalizer,
        ensemble_size: int = 5,
        hidden_dims: Sequence[int] = (512, 512, 512),
        skill_embedding_dim: int = 16,
        cylinder_embedding_dim: int = 64,
        activation: str = "silu",
        layer_norm: bool = True,
        min_log_variance: float = -10.0,
        max_log_variance: float = 2.0,
        num_events: int = len(EVENT_NAMES),
    ):
        super().__init__()
        self.schema = schema
        self.action_adapter = action_adapter
        self.normalizer = normalizer
        self.state_adapter = FootballWorldModelStateAdapter(schema=schema, max_obstacles=len([f for f in schema.features if f.group == "obstacle"]))
        self.state_adapter.action_adapter = action_adapter
        self.members = nn.ModuleList([
            WorldModelMember(
                schema, action_adapter, num_events, hidden_dims, skill_embedding_dim,
                cylinder_embedding_dim, activation, layer_norm, min_log_variance, max_log_variance,
            )
            for _ in range(int(ensemble_size))
        ])
        self.num_events = int(num_events)
        if self.num_events > len(EVENT_NAMES):
            raise ValueError(f"num_events={self.num_events} exceeds known event schema size {len(EVENT_NAMES)}")
        self.event_names = EVENT_NAMES[: self.num_events]
        self.config = {
            "ensemble_size": ensemble_size, "hidden_dims": list(hidden_dims),
            "skill_embedding_dim": skill_embedding_dim, "cylinder_embedding_dim": cylinder_embedding_dim,
            "activation": activation, "layer_norm": layer_norm,
            "min_log_variance": min_log_variance, "max_log_variance": max_log_variance,
            "num_events": num_events,
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.normalizer.to(device)
        return result

    def forward_members(self, states: torch.Tensor, joint_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.action_adapter.assert_within_bounds(joint_actions)
        normalized_states = self.normalizer.normalize_state(states)
        outputs = [member(normalized_states, joint_actions) for member in self.members]
        return {key: torch.stack([output[key] for output in outputs], dim=0) for key in outputs[0]}

    def predict_next(self, states: torch.Tensor, joint_actions: torch.Tensor, deterministic: bool = True):
        """Return next state, reward, done probability, event probabilities, uncertainty."""

        outputs = self.forward_members(states, joint_actions)
        member_delta_means = self.normalizer.denormalize_delta_prediction(outputs["delta_mean"])
        scale = self.normalizer.delta_std.to(states.device)
        member_delta_vars = outputs["delta_log_variance"].exp() * scale.square()
        delta_mean = member_delta_means.mean(0)
        epistemic = member_delta_means.var(0, unbiased=False)
        aleatoric = member_delta_vars.mean(0)
        total_variance = epistemic + aleatoric
        if deterministic:
            delta = delta_mean
        else:
            delta = delta_mean + torch.randn_like(delta_mean) * total_variance.sqrt()

        binary_prob = torch.sigmoid(outputs["binary_logits"]).mean(0)
        reward_members = self.normalizer.denormalize_reward(outputs["reward_mean"])
        reward_aleatoric = (outputs["reward_log_variance"].exp() * self.normalizer.reward_std.square()).mean(0)
        reward_epistemic = reward_members.var(0, unbiased=False)
        reward_mean = reward_members.mean(0)
        reward = reward_mean if deterministic else reward_mean + torch.randn_like(reward_mean) * (reward_aleatoric + reward_epistemic).sqrt()
        termination = torch.sigmoid(outputs["termination_logit"]).mean(0)
        truncation = torch.sigmoid(outputs["truncation_logit"]).mean(0)
        done_probability = 1.0 - (1.0 - termination) * (1.0 - truncation)
        events = torch.sigmoid(outputs["event_logits"]).mean(0)
        next_state = self.state_adapter.apply_predicted_delta(states, delta, binary_prob, joint_actions, deterministic)

        ball_positions = self.schema.group_positions("ball_position")
        uncertainty = {
            "state_epistemic_variance": epistemic,
            "state_aleatoric_variance": aleatoric,
            "state_total_variance": total_variance,
            "mean_state_uncertainty": total_variance.mean(-1),
            "max_state_uncertainty": total_variance.max(-1).values,
            "ball_state_uncertainty": total_variance[..., ball_positions].mean(-1),
            "reward_epistemic_variance": reward_epistemic,
            "reward_aleatoric_variance": reward_aleatoric,
            "reward_uncertainty": reward_epistemic + reward_aleatoric,
            "termination_probability": termination,
            "truncation_probability": truncation,
        }
        return next_state, reward, done_probability, events, uncertainty

    def rollout(
        self,
        initial_states: torch.Tensor,
        action_sequences: torch.Tensor,
        deterministic: bool = True,
        stop_on_done: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Vectorized rollout over [batch, candidates, horizon, action_dim]."""

        if initial_states.ndim != 2 or action_sequences.ndim != 4:
            raise ValueError("Expected initial_states [B,D] and action_sequences [B,C,H,A]")
        batch, candidates, horizon, action_dim = action_sequences.shape
        if initial_states.shape != (batch, self.schema.state_dim) or action_dim != self.action_adapter.action_dim:
            raise ValueError("Initial state or action sequence dimensions do not match the model schema")
        current = initial_states[:, None, :].expand(-1, candidates, -1).reshape(batch * candidates, -1)
        states = [current.reshape(batch, candidates, -1)]
        rewards, dones, state_uncertainty, reward_uncertainty, events = [], [], [], [], []
        alive = torch.ones(batch * candidates, dtype=torch.bool, device=current.device)
        for step in range(horizon):
            action = action_sequences[:, :, step].reshape(batch * candidates, action_dim)
            next_state, reward, done, event, uncertainty = self.predict_next(current, action, deterministic)
            if stop_on_done:
                reward = reward * alive.to(reward.dtype)
                next_state = torch.where(alive[:, None], next_state, current)
                event = event * alive[:, None].to(event.dtype)
                alive = alive & (done < 0.5)
            current = next_state
            states.append(current.reshape(batch, candidates, -1))
            rewards.append(reward.reshape(batch, candidates))
            dones.append(done.reshape(batch, candidates))
            state_uncertainty.append(uncertainty["mean_state_uncertainty"].reshape(batch, candidates))
            reward_uncertainty.append(uncertainty["reward_uncertainty"].reshape(batch, candidates))
            events.append(event.reshape(batch, candidates, self.num_events))
        return {
            "predicted_states": torch.stack(states, dim=2),
            "predicted_rewards": torch.stack(rewards, dim=2),
            "predicted_done_probabilities": torch.stack(dones, dim=2),
            "state_uncertainty": torch.stack(state_uncertainty, dim=2),
            "reward_uncertainty": torch.stack(reward_uncertainty, dim=2),
            "event_probabilities": torch.stack(events, dim=2),
        }

    def _vmap_member_forward(
        self,
        normalized_states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate matching member/state rows without a Python member loop."""

        try:
            from torch.func import functional_call, stack_module_state, vmap
        except ImportError as exc:
            raise RuntimeError(
                "Per-member pessimistic rollouts require torch.func (PyTorch 2.x)"
            ) from exc
        parameters, buffers = stack_module_state(list(self.members))
        base = self.members[0]

        def call_one(member_parameters, member_buffers, state, action):
            return functional_call(
                base,
                (member_parameters, member_buffers),
                (state, action),
            )

        return vmap(call_one, in_dims=(0, 0, 0, 0))(
            parameters,
            buffers,
            normalized_states,
            actions,
        )

    def rollout_members(
        self,
        initial_states: torch.Tensor,
        action_sequences: torch.Tensor,
        deterministic: bool = True,
        stop_on_done: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Propagate every ensemble member's own trajectory.

        Member rewards/done probabilities have shape ``[M,B,C,H]``.  Aggregate
        fields retain the standard :meth:`rollout` shapes so planners can use
        exact minimum-return or mean-minus-return-standard-deviation scoring.
        """

        if initial_states.ndim != 2 or action_sequences.ndim != 4:
            raise ValueError("Expected initial_states [B,D] and action_sequences [B,C,H,A]")
        self.action_adapter.assert_within_bounds(action_sequences)
        batch, candidates, horizon, action_dim = action_sequences.shape
        if initial_states.shape != (batch, self.schema.state_dim):
            raise ValueError("Initial state dimensions do not match the model schema")
        if action_dim != self.action_adapter.action_dim or horizon < 1:
            raise ValueError("Action sequence dimensions do not match the model schema")
        members = len(self.members)
        flat_count = batch * candidates
        initial_flat = initial_states[:, None, :].expand(-1, candidates, -1).reshape(flat_count, -1)
        current = initial_flat.unsqueeze(0).expand(members, -1, -1).clone()
        mean_states = [initial_flat.reshape(batch, candidates, -1)]
        member_rewards, member_dones = [], []
        mean_events, state_uncertainties, reward_uncertainties = [], [], []
        alive = torch.ones(
            members, flat_count, dtype=torch.bool, device=initial_states.device
        )
        dynamic = self.schema.continuous_dynamic_indices
        for step in range(horizon):
            action = action_sequences[:, :, step].reshape(flat_count, action_dim)
            member_action = action.unsqueeze(0).expand(members, -1, -1)
            normalized = self.normalizer.normalize_state(current)
            outputs = self._vmap_member_forward(normalized, member_action)
            delta_mean = self.normalizer.denormalize_delta_prediction(outputs["delta_mean"])
            delta_variance = (
                outputs["delta_log_variance"].exp()
                * self.normalizer.delta_std.to(initial_states.device).square()
            )
            reward_mean = self.normalizer.denormalize_reward(outputs["reward_mean"])
            reward_variance = (
                outputs["reward_log_variance"].exp()
                * self.normalizer.reward_std.to(initial_states.device).square()
            )
            if deterministic:
                delta = delta_mean
                reward = reward_mean
            else:
                delta = delta_mean + torch.randn_like(delta_mean) * delta_variance.sqrt()
                reward = reward_mean + torch.randn_like(reward_mean) * reward_variance.sqrt()
            binary = torch.sigmoid(outputs["binary_logits"])
            termination = torch.sigmoid(outputs["termination_logit"])
            truncation = torch.sigmoid(outputs["truncation_logit"])
            done = 1.0 - (1.0 - termination) * (1.0 - truncation)
            event = torch.sigmoid(outputs["event_logits"])
            next_state = self.state_adapter.apply_predicted_delta(
                current.reshape(members * flat_count, -1),
                delta.reshape(members * flat_count, -1),
                binary.reshape(members * flat_count, -1),
                member_action.reshape(members * flat_count, -1),
                deterministic,
            ).reshape(members, flat_count, -1)
            if stop_on_done:
                reward = reward * alive.to(reward.dtype)
                next_state = torch.where(alive[..., None], next_state, current)
                event = event * alive[..., None].to(event.dtype)
                alive = alive & (done < 0.5)
            current = next_state
            mean_states.append(current.mean(0).reshape(batch, candidates, -1))
            member_rewards.append(reward.reshape(members, batch, candidates))
            member_dones.append(done.reshape(members, batch, candidates))
            mean_events.append(event.mean(0).reshape(batch, candidates, self.num_events))
            state_epistemic = current[..., dynamic].var(0, unbiased=False).mean(-1)
            state_aleatoric = delta_variance.mean(0).mean(-1)
            state_uncertainties.append(
                (state_epistemic + state_aleatoric).reshape(batch, candidates)
            )
            reward_uncertainties.append(
                (
                    reward_mean.var(0, unbiased=False)
                    + reward_variance.mean(0)
                ).reshape(batch, candidates)
            )
        member_reward_tensor = torch.stack(member_rewards, dim=-1)
        member_done_tensor = torch.stack(member_dones, dim=-1)
        return {
            "predicted_states": torch.stack(mean_states, dim=2),
            "predicted_rewards": member_reward_tensor.mean(0),
            "predicted_done_probabilities": member_done_tensor.mean(0),
            "state_uncertainty": torch.stack(state_uncertainties, dim=2),
            "reward_uncertainty": torch.stack(reward_uncertainties, dim=2),
            "event_probabilities": torch.stack(mean_events, dim=2),
            "member_predicted_rewards": member_reward_tensor,
            "member_predicted_done_probabilities": member_done_tensor,
        }

    def evaluate_action_sequences(
        self,
        initial_states: torch.Tensor,
        action_sequences: torch.Tensor,
        gamma: float,
        uncertainty_penalty: float = 0.0,
    ) -> torch.Tensor:
        rollout = self.rollout(initial_states, action_sequences, deterministic=True)
        horizon = action_sequences.shape[2]
        discount = torch.pow(torch.as_tensor(gamma, device=initial_states.device), torch.arange(horizon, device=initial_states.device))
        utility = rollout["predicted_rewards"] - float(uncertainty_penalty) * rollout["state_uncertainty"]
        return (utility * discount).sum(dim=-1)
