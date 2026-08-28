"""Football termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .geometry import out_of_bounds_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_out_of_bounds(
    env: ManagerBasedRLEnv,
    half_extent_xy: tuple[float, float],
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Terminate when the ball leaves the local rectangular field."""

    ball: RigidObject = env.scene[ball_cfg.name]
    local_xy = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return out_of_bounds_mask(local_xy, half_extent_xy)


def ball_too_far_from_robot(
    env: ManagerBasedRLEnv,
    max_distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Terminate unrecoverable rollouts where robot and ball separate."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    distance = torch.linalg.vector_norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    return distance > float(max_distance)
