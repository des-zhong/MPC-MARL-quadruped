"""Planar command transforms for robot and field coordinate frames.

The soccer skills are planar even when the robot is pitched or rolled. These
helpers therefore use only the quaternion's yaw, avoiding z/xy distortion from
a full three-dimensional inverse rotation.
"""

import torch


def _yaw_cos_sin(quat: torch.Tensor):
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion [..., 4], got {tuple(quat.shape)}")

    x, y, z, w = quat.unbind(dim=-1)
    norm_sq = quat.square().sum(dim=-1).clamp_min(torch.finfo(quat.dtype).eps)
    cos_yaw = (w.square() + x.square() - y.square() - z.square()) / norm_sq
    sin_yaw = 2.0 * (w * z + x * y) / norm_sq
    return cos_yaw, sin_yaw


def body_xy_to_world_xy(body_xy: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors from a robot's body frame into the world frame."""

    if body_xy.shape[-1] != 2:
        raise ValueError(f"Expected planar vectors [..., 2], got {tuple(body_xy.shape)}")
    cos_yaw, sin_yaw = _yaw_cos_sin(quat)
    x, y = body_xy.unbind(dim=-1)
    return torch.stack(
        (cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y),
        dim=-1,
    )


def world_xy_to_body_xy(world_xy: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate planar vectors from the world frame into a robot's body frame."""

    if world_xy.shape[-1] != 2:
        raise ValueError(f"Expected planar vectors [..., 2], got {tuple(world_xy.shape)}")
    cos_yaw, sin_yaw = _yaw_cos_sin(quat)
    x, y = world_xy.unbind(dim=-1)
    return torch.stack(
        (cos_yaw * x + sin_yaw * y, -sin_yaw * x + cos_yaw * y),
        dim=-1,
    )
