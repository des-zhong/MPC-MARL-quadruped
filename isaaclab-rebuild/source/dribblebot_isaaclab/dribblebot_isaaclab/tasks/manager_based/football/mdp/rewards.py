"""Dense football reward terms shared by low-level skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .geometry import direction_alignment_score, dribbling_setup_score, planar_velocity_tracking_exp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_ball_velocity_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    scale_xy: tuple[float, float],
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Track the commanded world-frame velocity with the ball."""

    ball: RigidObject = env.scene[ball_cfg.name]
    command = env.command_manager.get_command(command_name)
    return planar_velocity_tracking_exp(ball.data.root_lin_vel_w[:, :2], command[:, :2], scale_xy, std)


def track_ball_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward ball travel in the commanded world-frame direction."""

    ball: RigidObject = env.scene[ball_cfg.name]
    command = env.command_manager.get_command(command_name)
    return direction_alignment_score(ball.data.root_lin_vel_w[:, :2], command[:, :2])


def ball_setup_position_exp(
    env: ManagerBasedRLEnv,
    target_forward: float,
    target_lateral: float,
    position_gain: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Keep the ball at a controllable pose ahead of the chassis."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    position_b = quat_apply_inverse(robot.data.root_quat_w, ball.data.root_pos_w - robot.data.root_pos_w)
    return dribbling_setup_score(position_b[:, :2], target_forward, target_lateral, position_gain)


def robot_ball_command_alignment_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    gain: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Align robot-to-ball and robot-forward directions with the ball command."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_dir = command_xy / torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)

    robot_to_ball = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    ball_dir = robot_to_ball / torch.linalg.vector_norm(robot_to_ball, dim=-1, keepdim=True).clamp_min(1.0e-6)
    forward_seed = torch.zeros_like(robot.data.root_pos_w)
    forward_seed[:, 0] = 1.0
    forward_dir = quat_apply(robot.data.root_quat_w, forward_seed)[:, :2]
    forward_dir = forward_dir / torch.linalg.vector_norm(forward_dir, dim=-1, keepdim=True).clamp_min(1.0e-6)

    error = (1.0 - torch.sum(ball_dir * command_dir, dim=-1)) + (
        1.0 - torch.sum(ball_dir * forward_dir, dim=-1)
    )
    active = torch.linalg.vector_norm(command_xy, dim=-1) > 1.0e-3
    return torch.exp(-float(gain) * error.clamp_min(0.0)) * active
