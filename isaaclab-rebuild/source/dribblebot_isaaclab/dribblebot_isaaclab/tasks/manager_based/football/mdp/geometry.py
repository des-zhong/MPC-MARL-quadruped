"""Pure football geometry shared by manager terms and simulator-free tests."""

from __future__ import annotations

import torch


def planar_velocity_tracking_exp(
    actual_velocity_xy: torch.Tensor,
    target_velocity_xy: torch.Tensor,
    scale_xy: tuple[float, float],
    std: float,
) -> torch.Tensor:
    """Track a planar world-frame velocity using normalized squared error."""

    scale = actual_velocity_xy.new_tensor(scale_xy).clamp_min(1.0e-6)
    error = torch.sum(torch.square((target_velocity_xy - actual_velocity_xy) / scale), dim=-1)
    return torch.exp(-error / max(float(std) ** 2, 1.0e-6))


def dribbling_setup_score(
    ball_position_b_xy: torch.Tensor,
    target_forward: float,
    target_lateral: float = 0.0,
    position_gain: float = 10.0,
) -> torch.Tensor:
    """Reward a controllable ball position just ahead of the robot base."""

    target = ball_position_b_xy.new_tensor([target_forward, target_lateral])
    error = torch.sum(torch.square(ball_position_b_xy - target), dim=-1)
    return torch.exp(-float(position_gain) * error)


def direction_alignment_score(actual_velocity_xy: torch.Tensor, target_velocity_xy: torch.Tensor) -> torch.Tensor:
    """Return directional alignment in [0, 1], with zero for a zero command."""

    target_speed = torch.linalg.vector_norm(target_velocity_xy, dim=-1)
    actual_speed = torch.linalg.vector_norm(actual_velocity_xy, dim=-1)
    dot = torch.sum(actual_velocity_xy * target_velocity_xy, dim=-1)
    cosine = dot / (actual_speed * target_speed).clamp_min(1.0e-6)
    return (0.5 * (cosine.clamp(-1.0, 1.0) + 1.0)) * (target_speed > 1.0e-3)


def out_of_bounds_mask(position_xy: torch.Tensor, half_extent_xy: tuple[float, float]) -> torch.Tensor:
    """Return true where a position leaves the rectangular training field."""

    half_extent = position_xy.new_tensor(half_extent_xy)
    return torch.any(torch.abs(position_xy) > half_extent, dim=-1)


def shooting_setup_geometry(
    base_xy: torch.Tensor,
    ball_xy: torch.Tensor,
    command_xy: torch.Tensor,
    setup_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return command direction, desired behind-ball pose, and setup error."""

    command_norm = torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    command_direction = command_xy / command_norm
    target_base_xy = ball_xy - float(setup_distance) * command_direction
    setup_error = torch.linalg.vector_norm(target_base_xy - base_xy, dim=-1)
    return command_direction, target_base_xy, setup_error


def shooting_setup_progress(
    previous_base_xy: torch.Tensor,
    base_xy: torch.Tensor,
    ball_xy: torch.Tensor,
    command_xy: torch.Tensor,
    setup_distance: float,
    dt: float,
    speed_scale: float,
) -> torch.Tensor:
    """Return signed normalized progress toward the desired behind-ball pose."""

    _, target_base_xy, current_error = shooting_setup_geometry(base_xy, ball_xy, command_xy, setup_distance)
    previous_error = torch.linalg.vector_norm(target_base_xy - previous_base_xy, dim=-1)
    denominator = max(float(dt) * float(speed_scale), 1.0e-6)
    return ((previous_error - current_error) / denominator).clamp(min=-1.0, max=1.0)


def shooting_forward_velocity_score(
    ball_velocity_xy: torch.Tensor,
    command_xy: torch.Tensor,
    min_command_speed: float = 0.2,
) -> torch.Tensor:
    """Score a useful strike before it crosses the discrete launch threshold."""

    target_speed = torch.linalg.vector_norm(command_xy, dim=-1)
    command_direction = command_xy / target_speed.clamp_min(1.0e-6).unsqueeze(-1)
    forward_speed = torch.sum(ball_velocity_xy * command_direction, dim=-1)
    ball_speed = torch.linalg.vector_norm(ball_velocity_xy, dim=-1)

    speed_fraction = (forward_speed / target_speed.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    alignment = (forward_speed / ball_speed.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    active_command = (target_speed > float(min_command_speed)).to(command_xy.dtype)
    return speed_fraction * alignment * active_command


def compute_shooting_phase(
    base_xy: torch.Tensor,
    ball_xy: torch.Tensor,
    ball_velocity_xy: torch.Tensor,
    command_xy: torch.Tensor,
    previously_launched: torch.Tensor,
    elapsed_s: torch.Tensor,
    min_command_speed: float,
    launch_speed_fraction: float,
    launch_alignment: float,
    success_distance: float,
    success_speed_fraction: float,
    success_alignment: float,
    max_attempt_time_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the launch/success/failure state machine for one policy step."""

    target_speed = torch.linalg.vector_norm(command_xy, dim=-1)
    active_command = target_speed > float(min_command_speed)
    command_direction = command_xy / target_speed.clamp_min(1.0e-6).unsqueeze(-1)

    ball_speed = torch.linalg.vector_norm(ball_velocity_xy, dim=-1)
    speed_along_command = torch.sum(ball_velocity_xy * command_direction, dim=-1)
    velocity_alignment = speed_along_command / ball_speed.clamp_min(1.0e-6)

    launched_now = (
        active_command
        & (speed_along_command > float(launch_speed_fraction) * target_speed)
        & (velocity_alignment > float(launch_alignment))
    )
    launch_event = launched_now & ~previously_launched
    launched = previously_launched | launched_now

    robot_to_ball = ball_xy - base_xy
    ball_forward_distance = torch.sum(robot_to_ball * command_direction, dim=-1)
    success = (
        active_command
        & launched
        & (ball_forward_distance > float(success_distance))
        & (speed_along_command > float(success_speed_fraction) * target_speed)
        & (velocity_alignment > float(success_alignment))
    )
    failure = active_command & ~success & (elapsed_s > float(max_attempt_time_s))
    return launch_event, launched, success, failure
