"""Training-split normalization statistics."""

from __future__ import annotations

from typing import Dict, Mapping

import torch


class WorldModelNormalizer:
    def __init__(self, state_mean, state_std, delta_mean, delta_std, reward_mean=0.0, reward_std=1.0):
        self.state_mean = torch.as_tensor(state_mean).float()
        self.state_std = torch.as_tensor(state_std).float().clamp(min=1e-6)
        self.delta_mean = torch.as_tensor(delta_mean).float()
        self.delta_std = torch.as_tensor(delta_std).float().clamp(min=1e-6)
        self.reward_mean = torch.as_tensor(reward_mean).float()
        self.reward_std = torch.as_tensor(reward_std).float().clamp(min=1e-6)

    @classmethod
    def fit(
        cls,
        states: torch.Tensor,
        deltas: torch.Tensor,
        rewards: torch.Tensor,
        continuous_state_indices=None,
        normalize_reward: bool = False,
    ) -> "WorldModelNormalizer":
        state_mean = torch.zeros(states.shape[-1], dtype=states.dtype, device=states.device)
        state_std = torch.ones_like(state_mean)
        indices = list(range(states.shape[-1])) if continuous_state_indices is None else continuous_state_indices
        state_mean[indices] = states[:, indices].mean(0)
        state_std[indices] = states[:, indices].std(0, unbiased=False).clamp(min=1e-6)
        reward_mean = rewards.mean() if normalize_reward else rewards.new_tensor(0.0)
        reward_std = rewards.std(unbiased=False).clamp(min=1e-6) if normalize_reward else rewards.new_tensor(1.0)
        return cls(
            state_mean, state_std,
            deltas.mean(0), deltas.std(0, unbiased=False),
            reward_mean, reward_std,
        )

    def to(self, device) -> "WorldModelNormalizer":
        for name in ("state_mean", "state_std", "delta_mean", "delta_std", "reward_mean", "reward_std"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def normalize_state(self, value): return (value - self.state_mean) / self.state_std
    def denormalize_state(self, value): return value * self.state_std + self.state_mean
    def normalize_action(self, value): return value  # Hybrid actions are normalized by JointActionAdapter.
    def normalize_delta_target(self, value): return (value - self.delta_mean) / self.delta_std
    def denormalize_delta_prediction(self, value): return value * self.delta_std + self.delta_mean
    def normalize_reward(self, value): return (value - self.reward_mean) / self.reward_std
    def denormalize_reward(self, value): return value * self.reward_std + self.reward_mean

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: getattr(self, name).detach().cpu() for name in ("state_mean", "state_std", "delta_mean", "delta_std", "reward_mean", "reward_std")}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, torch.Tensor]) -> "WorldModelNormalizer":
        return cls(**payload)
