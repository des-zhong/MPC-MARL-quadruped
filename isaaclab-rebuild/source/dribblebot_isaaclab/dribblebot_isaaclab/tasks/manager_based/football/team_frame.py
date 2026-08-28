"""Canonical team-frame transforms for the two-team coordinator contract.

The archived high-level policy is trained from the perspective of a team
attacking ``+x``. Team 1 attacks ``-x`` in the match world, so observations
must be rotated by pi before they are sent to that policy and field-frame
dribble/shoot commands must be rotated back before entering an action term.
"""

from __future__ import annotations

import torch


WALK_SKILL_ID = 0


def mirror_high_level_commands(commands: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
    """Rotate field-frame ball commands while preserving body-frame walking."""

    if commands.shape[:-1] != skill_ids.shape or commands.shape[-1] != 3:
        raise ValueError(
            "Expected commands [..., 3] and matching skill_ids [...], got "
            f"{tuple(commands.shape)} and {tuple(skill_ids.shape)}"
        )
    mirrored = commands.clone()
    field_frame = (skill_ids != WALK_SKILL_ID).unsqueeze(-1)
    mirrored[..., :2] = torch.where(field_frame, -mirrored[..., :2], mirrored[..., :2])
    return mirrored


def mirror_high_level_policy_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert canonical opponent policy outputs to world-executable actions."""

    if actions.shape[-1] != 6:
        raise ValueError(f"Expected high-level policy actions [..., 6], got {tuple(actions.shape)}")
    mirrored = actions.clone()
    skill_ids = actions[..., :3].argmax(dim=-1)
    field_frame = (skill_ids != WALK_SKILL_ID).unsqueeze(-1)
    # Commands are decoded through tanh; tanh is odd, so negating raw values
    # gives the exact inverse transform without touching skill logits/yaw.
    mirrored[..., 3:5] = torch.where(field_frame, -mirrored[..., 3:5], mirrored[..., 3:5])
    return mirrored


def team_signs(num_robots: int, team_size: int, *, device=None, dtype=None) -> torch.Tensor:
    """Return ``+1`` for team 0 and ``-1`` for team 1, per robot slot."""

    if int(num_robots) != 2 * int(team_size) or int(team_size) <= 0:
        raise ValueError("num_robots must equal 2*team_size with team_size > 0")
    signs = torch.ones(int(num_robots), device=device, dtype=dtype)
    signs[int(team_size) :] = -1
    return signs


__all__ = [
    "WALK_SKILL_ID",
    "mirror_high_level_commands",
    "mirror_high_level_policy_actions",
    "team_signs",
]
