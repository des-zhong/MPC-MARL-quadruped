"""Probabilistic neural transition member."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from .action_adapter import JointActionAdapter
from .schema import StateSchema


def activation(name: str) -> nn.Module:
    if name.lower() == "silu":
        return nn.SiLU()
    if name.lower() == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation {name!r}")


class ResidualBlock(nn.Module):
    def __init__(self, width: int, activation_name: str = "silu", layer_norm: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), activation(activation_name),
            nn.Linear(width, width),
            nn.LayerNorm(width) if layer_norm else nn.Identity(),
        )
        self.output_activation = activation(activation_name)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output_activation(value + self.net(value))


class WorldModelMember(nn.Module):
    """One independently initialized probabilistic ensemble member."""

    def __init__(
        self,
        schema: StateSchema,
        action_adapter: JointActionAdapter,
        num_events: int,
        hidden_dims: Sequence[int] = (512, 512, 512),
        skill_embedding_dim: int = 16,
        cylinder_embedding_dim: int = 64,
        activation_name: str = "silu",
        layer_norm: bool = True,
        min_log_variance: float = -10.0,
        max_log_variance: float = 2.0,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims cannot be empty")
        self.schema = schema
        self.action_adapter = action_adapter
        self.num_events = int(num_events)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)
        self.dynamic_dim = len(schema.continuous_dynamic_indices)
        self.binary_dim = len(schema.binary_dynamic_indices)
        self.obstacle_features = [f for f in schema.features if f.group == "obstacle"]
        obstacle_indices = [i for f in self.obstacle_features for i in range(f.start, f.stop)]
        self.register_buffer("obstacle_indices", torch.tensor(obstacle_indices, dtype=torch.long), persistent=False)
        non_obstacle = [i for i in range(schema.state_dim) if i not in set(obstacle_indices)]
        self.register_buffer("non_obstacle_indices", torch.tensor(non_obstacle, dtype=torch.long), persistent=False)

        obstacle_input_dim = self.obstacle_features[0].size if self.obstacle_features else 6
        self.obstacle_phi = nn.Sequential(
            nn.Linear(obstacle_input_dim - 1, cylinder_embedding_dim), activation(activation_name),
            nn.Linear(cylinder_embedding_dim, cylinder_embedding_dim), activation(activation_name),
        )
        self.obstacle_rho = nn.Sequential(
            nn.Linear(cylinder_embedding_dim, cylinder_embedding_dim), activation(activation_name),
        )
        self.skill_embedding = nn.Embedding(3, skill_embedding_dim)
        per_robot_action_dim = skill_embedding_dim + 3 + 3
        self.action_encoder = nn.Sequential(
            nn.Linear(
                action_adapter.num_robots * per_robot_action_dim,
                hidden_dims[0] // 2,
            ),
            activation(activation_name),
        )
        input_dim = len(non_obstacle) + cylinder_embedding_dim + hidden_dims[0] // 2
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation(activation_name)]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dims[0]))
        for index in range(len(hidden_dims)):
            width = hidden_dims[index]
            if index and hidden_dims[index - 1] != width:
                layers.append(nn.Linear(hidden_dims[index - 1], width))
            layers.append(ResidualBlock(width, activation_name, layer_norm))
        self.trunk = nn.Sequential(*layers)
        width = hidden_dims[-1]
        self.delta_mean = nn.Linear(width, self.dynamic_dim)
        self.delta_log_variance = nn.Linear(width, self.dynamic_dim)
        self.reward_mean = nn.Linear(width, 1)
        self.reward_log_variance = nn.Linear(width, 1)
        self.binary_logits = nn.Linear(width, self.binary_dim)
        self.termination_logit = nn.Linear(width, 1)
        self.truncation_logit = nn.Linear(width, 1)
        self.event_logits = nn.Linear(width, self.num_events)

    def _encode_obstacles(self, states: torch.Tensor) -> torch.Tensor:
        if not self.obstacle_features:
            return states.new_zeros(states.shape[:-1] + (self.obstacle_rho[0].out_features,))
        obstacles = torch.stack([states[..., f.start : f.stop] for f in self.obstacle_features], dim=-2)
        features, mask = obstacles[..., :-1], obstacles[..., -1:]
        return self.obstacle_rho((self.obstacle_phi(features) * mask).sum(dim=-2))

    def _encode_action(self, actions: torch.Tensor) -> torch.Tensor:
        # Ensemble.forward_members validates canonical actions once before
        # dispatch.  Avoid tensor-to-Python validation here so this pure member
        # function remains compatible with torch.func.vmap for pessimistic
        # per-member rollouts.
        skills, parameters = self.action_adapter._unpack_unchecked(actions)
        normalized = self.action_adapter.normalize_parameters(skills, parameters)
        mask = self.action_adapter._selected(skills, "mask", parameters.dtype)
        encoded = torch.cat((self.skill_embedding(skills), normalized, mask), dim=-1).flatten(-2)
        return self.action_encoder(encoded)

    def forward(self, normalized_states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        if normalized_states.shape[-1] != self.schema.state_dim:
            raise ValueError(f"Expected state dimension {self.schema.state_dim}, got {normalized_states.shape[-1]}")
        encoded = torch.cat(
            (
                normalized_states.index_select(-1, self.non_obstacle_indices),
                self._encode_obstacles(normalized_states),
                self._encode_action(actions),
            ),
            dim=-1,
        )
        latent = self.trunk(encoded)
        return {
            "delta_mean": self.delta_mean(latent),
            "delta_log_variance": self.delta_log_variance(latent).clamp(self.min_log_variance, self.max_log_variance),
            "reward_mean": self.reward_mean(latent).squeeze(-1),
            "reward_log_variance": self.reward_log_variance(latent).squeeze(-1).clamp(self.min_log_variance, self.max_log_variance),
            "binary_logits": self.binary_logits(latent),
            "termination_logit": self.termination_logit(latent).squeeze(-1),
            "truncation_logit": self.truncation_logit(latent).squeeze(-1),
            "event_logits": self.event_logits(latent),
        }
