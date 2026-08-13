"""Warm-start state carried between receding-horizon MPC calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class MPCPlannerState:
    """Final CEM distribution and execution context from the previous call.

    Tensor shapes are ``skill_probabilities [B,H,2,3]``,
    ``parameter_means/stds [B,H,2,3,3]``, ``valid [B]``,
    ``previous_action [B,8]``, and ``previous_state [B,D]``.
    """

    skill_probabilities: torch.Tensor
    parameter_means: torch.Tensor
    parameter_stds: torch.Tensor
    valid: torch.Tensor
    previous_action: Optional[torch.Tensor] = None
    previous_state: Optional[torch.Tensor] = None
    last_plan_uncertainty: Optional[torch.Tensor] = None

    def to(self, device: Union[str, torch.device]) -> "MPCPlannerState":
        return MPCPlannerState(
            self.skill_probabilities.to(device),
            self.parameter_means.to(device),
            self.parameter_stds.to(device),
            self.valid.to(device),
            None if self.previous_action is None else self.previous_action.to(device),
            None if self.previous_state is None else self.previous_state.to(device),
            None if self.last_plan_uncertainty is None else self.last_plan_uncertainty.to(device),
        )

    def detach(self) -> "MPCPlannerState":
        return MPCPlannerState(
            self.skill_probabilities.detach(),
            self.parameter_means.detach(),
            self.parameter_stds.detach(),
            self.valid.detach(),
            None if self.previous_action is None else self.previous_action.detach(),
            None if self.previous_state is None else self.previous_state.detach(),
            None if self.last_plan_uncertainty is None else self.last_plan_uncertainty.detach(),
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.valid.zero_()
        else:
            self.valid[env_ids] = False
