"""Minimal match-level manager terms used while the four-robot scene is staged."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def match_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a finite per-step survival signal for match smoke tests."""

    return torch.ones(env.num_envs, device=env.device)


def match_ball_out_of_bounds(
    env: ManagerBasedRLEnv,
    half_extent_xy: tuple[float, float] = (4.0, 2.5),
) -> torch.Tensor:
    """Terminate a match smoke episode when the ball leaves the field."""

    ball = env.scene["ball"]
    local_xy = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (torch.abs(local_xy[:, 0]) > float(half_extent_xy[0])) | (
        torch.abs(local_xy[:, 1]) > float(half_extent_xy[1])
    )


def _field_ball_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["ball"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


def match_goal(
    env: ManagerBasedRLEnv,
    goal_x: float = 4.0,
    goal_half_width: float = 1.0,
) -> torch.Tensor:
    ball_xy = _field_ball_xy(env)
    return (ball_xy[:, 0] >= float(goal_x)) & (torch.abs(ball_xy[:, 1]) <= float(goal_half_width))


def match_opponent_goal(
    env: ManagerBasedRLEnv,
    goal_x: float = 4.0,
    goal_half_width: float = 1.0,
) -> torch.Tensor:
    ball_xy = _field_ball_xy(env)
    return (ball_xy[:, 0] <= -float(goal_x)) & (torch.abs(ball_xy[:, 1]) <= float(goal_half_width))


def match_robot_fallen(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
    min_height: float = 0.20,
) -> torch.Tensor:
    heights = torch.stack([env.scene[name].data.root_pos_w[:, 2] for name in robot_names], dim=1)
    return torch.any(heights < float(min_height), dim=1)


def match_goal_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.get_term("goal").float()


def match_opponent_goal_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.get_term("opponent_goal").float()


def match_possession(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
) -> torch.Tensor:
    ball_xy = env.scene["ball"].data.root_pos_w[:, :2]
    robot_xy = torch.stack([env.scene[name].data.root_pos_w[:, :2] for name in robot_names], dim=1)
    distance = torch.linalg.vector_norm(robot_xy - ball_xy[:, None, :], dim=-1).amin(dim=1)
    return torch.exp(-2.0 * torch.square(distance))
