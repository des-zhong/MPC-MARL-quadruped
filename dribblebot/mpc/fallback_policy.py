"""Explicit fallback actions used when every candidate plan is invalid."""

from __future__ import annotations

from typing import Optional

import torch


class SafeRepositionFallback:
    """Select zero-speed reposition for every robot."""

    def __init__(self, action_adapter):
        self.action_adapter = action_adapter

    def __call__(
        self,
        states: torch.Tensor,
        previous_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        count = self.action_adapter.num_robots
        skills = torch.zeros(states.shape[0], count, dtype=torch.long, device=states.device)
        parameters = torch.zeros(states.shape[0], count, 3, dtype=states.dtype, device=states.device)
        return self.action_adapter.pack(skills, parameters)


class PreviousActionFallback(SafeRepositionFallback):
    """Reuse a valid previous action, otherwise use safe reposition."""

    def __call__(
        self,
        states: torch.Tensor,
        previous_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if previous_action is None:
            return super().__call__(states, previous_action)
        try:
            self.action_adapter.assert_within_bounds(previous_action)
        except (ValueError, RuntimeError):
            return super().__call__(states, previous_action)
        return previous_action.clone()
