"""State-based scoring signals shared by real and imagined football."""
import torch


def dribble_score(velocity, direction, commands, target_speed=1.):
    speed = velocity.norm(dim=-1)
    command_speed = commands.norm(dim=-1)
    tracking = ((velocity[..., None, :] * commands).sum(-1)
                / command_speed.square().clamp_min(1e-6)).clamp(0., 1.)
    toward = (velocity * direction).sum(-1)
    # Match the simulator: cap total speed before projecting alignment.
    aligned = (toward / speed.clamp_min(1e-6)).clamp_min(0) * (speed / target_speed).clamp(max=1.)
    return tracking * aligned[..., None] + (toward / target_speed).clamp(-1., 0.)[..., None]


def launch_score(previous_velocity, velocity, direction):
    before = (previous_velocity * direction).sum(-1)
    after = (velocity * direction).sum(-1)
    alignment = after / velocity.norm(dim=-1).clamp_min(1e-6)
    # A directed launch is a threshold-crossing event, not a velocity-rate bonus.
    return ((before < .8) & (after >= .8) & (alignment >= .6)).to(velocity.dtype)
