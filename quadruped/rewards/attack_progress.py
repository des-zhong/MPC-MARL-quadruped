"""Shared simulator/MPC shaping for getting behind a slow ball."""
import torch


def attack_position_progress(robot_before, robot_after, ball_before, ball_after,
                             goal, dt, attacker, max_ball_speed=.5):
    direction0 = goal-ball_before
    direction1 = goal-ball_after
    direction0 = direction0 / direction0.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction1 = direction1 / direction1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    target0, target1 = ball_before-.55*direction0, ball_after-.55*direction1
    d0 = (robot_before-target0.unsqueeze(-2)).norm(dim=-1)
    d1 = (robot_after-target1.unsqueeze(-2)).norm(dim=-1)
    progress = torch.gather(d0-d1, -1, attacker.long().unsqueeze(-1)).squeeze(-1)
    slow = (ball_after-ball_before).norm(dim=-1) / dt <= max_ball_speed
    # A signed rate: stationary positioning cannot collect a living bonus.
    return (progress / dt).clamp(-1., 1.) * slow
