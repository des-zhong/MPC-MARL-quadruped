"""Geometry for keeping support robots out of teammates' paths."""
import torch


def avoid_teammate_path(position, target, teammate, clearance=.8):
    """Use a lateral waypoint when the straight path crosses a teammate.

    Coordinates end in two world axes; leading dimensions are arbitrary.
    A robot already inside the clearance radius first moves directly away.
    """
    travel = target - position
    relative = teammate - position
    length = travel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = travel / length
    along = (relative * direction).sum(-1)
    lateral = torch.stack((-direction[..., 1], direction[..., 0]), -1)
    offset = (relative * lateral).sum(-1)
    crosses = (along > 0) & (along < length.squeeze(-1)) & (offset.abs() < clearance)
    side = torch.where(offset >= 0, -1., 1.)
    waypoint = teammate + side[..., None] * clearance * lateral
    target = torch.where(crosses[..., None], waypoint, target)
    distance = relative.norm(dim=-1, keepdim=True)
    away = -relative / distance.clamp_min(1e-6)
    # Deterministic finite fallback for coincident centres.
    away = torch.where(distance > 1e-6, away, lateral)
    return torch.where((distance < clearance), position + clearance * away, target)


def shot_lane_obstruction(positions, ball, goal, attacker):
    """Bounded cost for a support robot occupying its team's shooting lane."""
    direction = goal - ball
    length = direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    direction = direction / length
    relative = positions - ball[..., None, :]
    along = (relative * direction[..., None, :]).sum(-1)
    lateral = (relative - along[..., None] * direction[..., None, :]).norm(dim=-1)
    slots = torch.arange(positions.shape[-2], device=positions.device)
    support = slots != attacker[..., None]
    obstruction = (1 - lateral / .65).clamp(0, 1)
    obstruction *= support & (along > 0) & (along < length)
    return obstruction.amax(-1)
