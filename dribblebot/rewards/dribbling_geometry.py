"""Geometry helpers shared by dribbling rewards and unit tests."""

import torch


def dribbling_setup_score(
    ball_local_xy,
    target_forward,
    target_lateral=0.0,
    position_gain=10.0,
):
    """Reward a controllable ball position ahead of the robot base."""

    target = ball_local_xy.new_tensor(
        [float(target_forward), float(target_lateral)]
    )
    error = torch.sum(torch.square(ball_local_xy - target), dim=-1)
    return torch.exp(-float(position_gain) * error)


def dribbling_backward_motion_penalty(
    base_velocity_xy,
    body_forward_xy,
    speed_scale=1.0,
):
    """Return zero for forward motion and a signed penalty for backing up."""

    forward = body_forward_xy / torch.norm(
        body_forward_xy, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    forward_speed = torch.sum(base_velocity_xy * forward, dim=-1)
    return torch.clamp(
        forward_speed / max(float(speed_scale), 1.0e-6), min=-1.0, max=0.0
    )
