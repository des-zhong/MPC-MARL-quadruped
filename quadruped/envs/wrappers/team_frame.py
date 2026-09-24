"""Coordinate transforms shared by the two-team high-level wrappers."""

import torch


WALK_SKILL_ID = 0


def mirror_high_level_commands(
    commands: torch.Tensor, skill_ids: torch.Tensor
) -> torch.Tensor:
    """Rotate field-frame ball commands by pi while preserving body commands.

    Walking x/y commands are expressed in the robot body frame. Dribbling and
    shooting x/y commands are desired ball velocities in the fixed field frame.
    A team-perspective rotation therefore negates only the latter two skills'
    planar command components. Yaw rate is invariant under the rotation.

    This transform is its own inverse, so it is used both when converting an
    opponent policy's canonical command to the world frame and when encoding a
    previously executed world-frame command back into canonical observations.
    """

    if commands.shape[:-1] != skill_ids.shape or commands.shape[-1] != 3:
        raise ValueError(
            "Expected commands [..., 3] and matching skill_ids [...], got "
            f"{tuple(commands.shape)} and {tuple(skill_ids.shape)}"
        )
    mirrored = commands.clone()
    field_frame = (skill_ids != WALK_SKILL_ID).unsqueeze(-1)
    mirrored[..., :2] = torch.where(
        field_frame, -mirrored[..., :2], mirrored[..., :2]
    )
    return mirrored


def mirror_high_level_policy_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert canonical opponent policy outputs to executable world actions.

    Legacy policy actions contain three skill logits followed by three raw
    command values. Discrete policies contain four skill one-hot values and
    eight direction one-hot values; ball-skill directions are rotated by pi
    for the opponent while Walk directions remain body-relative.
    """

    if actions.shape[-1] == 12:
        mirrored = actions.clone()
        skill_ids = actions[..., :4].argmax(dim=-1)
        field_frame = ((skill_ids == 1) | (skill_ids == 2))
        # Rotating by pi maps direction d to d+4. Reordering logits also
        # handles deterministic raw actor outputs; one-hot samples remain
        # one-hot, and Stop's direction block remains irrelevant/unchanged.
        opposite_order = torch.tensor(
            [4, 5, 6, 7, 0, 1, 2, 3],
            dtype=torch.long,
            device=actions.device,
        )
        rotated_directions = actions[..., 4:12].index_select(-1, opposite_order)
        mirrored[..., 4:12] = torch.where(
            field_frame.unsqueeze(-1), rotated_directions, actions[..., 4:12]
        )
        return mirrored
    if actions.shape[-1] != 6:
        raise ValueError(
            f"Expected high-level policy actions [..., 6] or [..., 12], got {tuple(actions.shape)}"
        )
    mirrored = actions.clone()
    skill_ids = actions[..., :3].argmax(dim=-1)
    field_frame = (skill_ids != WALK_SKILL_ID).unsqueeze(-1)
    mirrored[..., 3:5] = torch.where(
        field_frame, -mirrored[..., 3:5], mirrored[..., 3:5]
    )
    return mirrored
