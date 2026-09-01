"""Warm-start state carried between receding-horizon MPC calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class MPCPlannerState:
    """Final CEM distribution and execution context from the previous call.

    Tensor shapes are ``skill_probabilities [B,H,N,3]``,
    ``parameter_means/stds [B,H,N,3,3]``, and ``valid [B]``.
    """

    skill_probabilities: torch.Tensor
    parameter_means: torch.Tensor
    parameter_stds: torch.Tensor
    valid: torch.Tensor

    def to(self, device: Union[str, torch.device]) -> "MPCPlannerState":
        return MPCPlannerState(
            self.skill_probabilities.to(device),
            self.parameter_means.to(device),
            self.parameter_stds.to(device),
            self.valid.to(device),
        )

    def detach(self) -> "MPCPlannerState":
        return MPCPlannerState(
            self.skill_probabilities.detach(),
            self.parameter_means.detach(),
            self.parameter_stds.detach(),
            self.valid.detach(),
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.valid.zero_()
        else:
            self.valid[env_ids] = False
