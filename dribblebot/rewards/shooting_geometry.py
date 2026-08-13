"""Geometry helpers shared by shooting rewards and their unit tests."""

import torch


def shooting_setup_geometry(base_xy, ball_xy, command_xy, setup_distance):
    """Return command direction, desired behind-ball position, and position error."""
    command_norm = torch.norm(command_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    command_direction = command_xy / command_norm
    target_base_xy = ball_xy - float(setup_distance) * command_direction
    setup_error = torch.norm(target_base_xy - base_xy, dim=-1)
    return command_direction, target_base_xy, setup_error


def shooting_setup_progress(
    previous_base_xy,
    base_xy,
    ball_xy,
    command_xy,
    setup_distance,
    dt,
    speed_scale,
):
    """Signed, normalized progress toward the desired behind-ball position."""
    _, target_base_xy, current_error = shooting_setup_geometry(
        base_xy,
        ball_xy,
        command_xy,
        setup_distance,
    )
    previous_error = torch.norm(target_base_xy - previous_base_xy, dim=-1)
    normalized_progress = (previous_error - current_error) / max(float(dt) * float(speed_scale), 1e-6)
    return normalized_progress.clamp(min=-1.0, max=1.0)


def shooting_forward_velocity_score(
    ball_velocity_xy,
    command_xy,
    min_command_speed=0.2,
):
    """Score partial strikes before they satisfy the discrete launch condition.

    The score is one when the ball reaches the requested speed in the requested
    direction, decreases smoothly for weaker or oblique strikes, and is zero
    for stationary, perpendicular, or backwards ball motion. Unlike the
    post-launch shooting rewards, this deliberately has no launch or separation
    gate so PPO receives credit for the first useful contact with the ball.
    """

    target_speed = torch.norm(command_xy, dim=-1)
    command_direction = command_xy / target_speed.clamp_min(1.0e-6).unsqueeze(-1)
    forward_speed = torch.sum(ball_velocity_xy * command_direction, dim=-1)
    ball_speed = torch.norm(ball_velocity_xy, dim=-1)

    speed_fraction = (forward_speed / target_speed.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    alignment = (forward_speed / ball_speed.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    active_command = (target_speed > float(min_command_speed)).to(command_xy.dtype)
    return speed_fraction * alignment * active_command
