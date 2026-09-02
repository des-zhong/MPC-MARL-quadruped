"""Planar command transforms for Isaac Lab's ``wxyz`` quaternions.

Ball skills operate in the horizontal plane. Only yaw is used so robot roll
and pitch do not distort a dribble or shooting command.
"""

from __future__ import annotations

import torch


def _yaw_cos_sin_wxyz(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quaternion [..., 4], got {tuple(quat_wxyz.shape)}")
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    norm_sq = quat_wxyz.square().sum(dim=-1).clamp_min(torch.finfo(quat_wxyz.dtype).eps)
    cos_yaw = (w.square() + x.square() - y.square() - z.square()) / norm_sq
    sin_yaw = 2.0 * (w * z + x * y) / norm_sq
    return cos_yaw, sin_yaw


def body_xy_to_world_xy(body_xy: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors from the robot body frame into the world frame."""

    if body_xy.shape[-1] != 2:
        raise ValueError(f"Expected planar vectors [..., 2], got {tuple(body_xy.shape)}")
    cos_yaw, sin_yaw = _yaw_cos_sin_wxyz(quat_wxyz)
    x, y = body_xy.unbind(dim=-1)
    return torch.stack((cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y), dim=-1)


def world_xy_to_body_xy(world_xy: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors from the world frame into the robot body frame."""

    if world_xy.shape[-1] != 2:
        raise ValueError(f"Expected planar vectors [..., 2], got {tuple(world_xy.shape)}")
    cos_yaw, sin_yaw = _yaw_cos_sin_wxyz(quat_wxyz)
    x, y = world_xy.unbind(dim=-1)
    return torch.stack((cos_yaw * x + sin_yaw * y, -sin_yaw * x + cos_yaw * y), dim=-1)


__all__ = ["body_xy_to_world_xy", "world_xy_to_body_xy"]
