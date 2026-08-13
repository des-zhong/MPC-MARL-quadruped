from typing import Dict, Sequence

import torch
import torch.nn as nn

from .dataset import BINARY_TARGET_KEYS


REGRESSION_TARGET_DIMS = {
    "delta_robot": 2,
    "delta_yaw": 1,
    "delta_ball": 2,
}


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: nn.Module = nn.ELU,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class AffordanceMLP(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_skills: int,
        num_robots: int,
        command_dim: int = 3,
        skill_embedding_dim: int = 8,
        robot_embedding_dim: int = 4,
        hidden_dims: Sequence[int] = (256, 256, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.command_dim = command_dim
        self.num_skills = max(1, int(num_skills))
        self.num_robots = max(1, int(num_robots))

        self.skill_embedding = nn.Embedding(self.num_skills, skill_embedding_dim)
        self.robot_embedding = nn.Embedding(self.num_robots, robot_embedding_dim)

        trunk_input_dim = (
            state_dim
            + command_dim
            + 1
            + skill_embedding_dim
            + robot_embedding_dim
        )
        trunk_output_dim = hidden_dims[-1] if hidden_dims else trunk_input_dim
        if hidden_dims:
            self.trunk = build_mlp(
                trunk_input_dim,
                hidden_dims[:-1],
                trunk_output_dim,
                dropout=dropout,
            )
        else:
            self.trunk = nn.Identity()

        self.binary_head = nn.Linear(trunk_output_dim, len(BINARY_TARGET_KEYS))
        self.delta_robot_head = nn.Linear(trunk_output_dim, 2)
        self.delta_yaw_head = nn.Linear(trunk_output_dim, 1)
        self.delta_ball_head = nn.Linear(trunk_output_dim, 2)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state_t = inputs["state_t"].float().flatten(start_dim=1)
        command = inputs["command"].float()
        duration = inputs["duration"].float().view(command.shape[0], -1)
        robot_id = inputs["robot_id"].long().view(-1).clamp(0, self.num_robots - 1)
        skill_id = inputs["skill_id"].long().view(-1).clamp(0, self.num_skills - 1)

        features = torch.cat(
            [
                state_t,
                command,
                duration,
                self.robot_embedding(robot_id),
                self.skill_embedding(skill_id),
            ],
            dim=-1,
        )
        latent = self.trunk(features)
        return {
            "binary_logits": self.binary_head(latent),
            "delta_robot": self.delta_robot_head(latent),
            "delta_yaw": self.delta_yaw_head(latent),
            "delta_ball": self.delta_ball_head(latent),
        }

    def predict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = self.forward(inputs)
        probabilities = torch.sigmoid(outputs["binary_logits"])
        return {
            **{
                key: probabilities[:, idx:idx + 1]
                for idx, key in enumerate(BINARY_TARGET_KEYS)
            },
            "delta_robot": outputs["delta_robot"],
            "delta_yaw": outputs["delta_yaw"],
            "delta_ball": outputs["delta_ball"],
        }
