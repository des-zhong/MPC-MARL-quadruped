"""Shared shooting execution contract for the simulator and imagined actions.

Shoot means prepare behind the ball, strike toward the opponent goal, then
release on launch, lost ball, danger, or a bounded deadline. Timer is observable.
"""
import torch

MAX_SECONDS = 2.4
MACRO_SECONDS = .2
COOLDOWN_SECONDS = .4


def next_timer(previous, executed_shoot, dt=MACRO_SECONDS):
    remaining = torch.where(previous > 0, (previous - dt).clamp_min(0),
                            (previous + dt).clamp_max(0))
    remaining = torch.where(executed_shoot & (previous <= 0),
                            torch.full_like(previous, MAX_SECONDS - dt), remaining)
    ended = (previous > 0) & (~executed_shoot | (remaining <= 1e-5))
    return torch.where(ended, torch.full_like(previous, -COOLDOWN_SECONDS), remaining)


def resolve_shooting_option(skills, commands, timer, previous_commands,
                            positions, yaw_sin_cos, ball, velocity, goal, heights):
    """All robot arrays end in [robots,...]; coordinates are global metres."""
    delta = ball[..., None, :] - positions
    distance = delta.norm(dim=-1)
    direction = goal - ball[..., None, :]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sin, cos = yaw_sin_cos.unbind(-1)
    local_x = cos * delta[..., 0] + sin * delta[..., 1]
    local_y = -sin * delta[..., 0] + cos * delta[..., 1]
    behind = (delta * direction).sum(-1) / distance.clamp_min(1e-6)
    facing = cos * direction[..., 0] + sin * direction[..., 1]
    relative = positions[..., :, None, :] - positions[..., None, :, :]
    separation = relative.norm(dim=-1)
    diagonal = torch.eye(skills.shape[-1], device=skills.device, dtype=torch.bool)
    nearest = separation.masked_fill(diagonal, float('inf')).amin(-1)
    safe = (heights >= .22) & (nearest > .45)
    launch = (velocity[..., None, :] * direction).sum(-1) >= .8
    active = ((timer > 1e-5) & safe & (distance < .95) & ~launch
              & (behind > .3) & (facing > .3) & (local_y.abs() < .45))
    ready = ((distance <= .65) & (local_x >= .25) & (local_y.abs() <= .2)
             & (behind >= .8) & (facing >= .8) & safe & ~launch)
    requested = (skills == 2) & (timer >= 0)
    begin = requested & ready
    strike = active | begin
    prepare = ((skills == 2) | (timer > 0)) & ~strike
    result_skills = torch.where(prepare, torch.zeros_like(skills), skills)
    result_skills = torch.where(strike, torch.full_like(skills, 2), result_skills)
    result_commands = commands.clone()
    target = ball[..., None, :] - .45 * direction
    travel = target - positions
    # Route around the ball when on its far side instead of walking through it.
    lateral = torch.stack((-direction[..., 1], direction[..., 0]), -1)
    side = torch.where((delta * lateral).sum(-1) >= 0, -1., 1.)
    travel = torch.where(((behind < .3) & (distance < .9))[..., None],
                         target + side[..., None] * .55 * lateral - positions, travel)
    world_walk = travel / travel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    world_walk *= (.9 * travel.norm(dim=-1).clamp(max=1.))[..., None]
    # Yield away from the nearest robot instead of freezing both robots when
    # they crowd the ball. Keep striking disabled until separation recovers.
    masked_separation = separation.masked_fill(diagonal, float('inf'))
    neighbor = masked_separation.argmin(-1)
    away = relative.gather(-2, neighbor[..., None, None].expand(*neighbor.shape, 1, 2)).squeeze(-2)
    away = away / away.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    world_walk = torch.where((nearest <= .45)[..., None], .35 * away, world_walk)
    body_walk = torch.stack((cos * world_walk[..., 0] + sin * world_walk[..., 1],
                             -sin * world_walk[..., 0] + cos * world_walk[..., 1]), -1)
    angle = torch.atan2(direction[..., 1], direction[..., 0]) - torch.atan2(sin, cos)
    yaw_command = torch.atan2(angle.sin(), angle.cos()).clamp(-1., 1.)
    prepare_command = torch.cat((body_walk, yaw_command[..., None]), -1)
    prepare_command = torch.where((heights >= .22)[..., None], prepare_command, torch.zeros_like(prepare_command))
    strike_command = torch.cat((3. * direction, torch.zeros_like(direction[..., :1])), -1)
    strike_command = torch.where(active[..., None], previous_commands, strike_command)
    result_commands = torch.where(prepare[..., None], prepare_command, result_commands)
    result_commands = torch.where(strike[..., None], strike_command, result_commands)
    return result_skills, result_commands, next_timer(timer, strike), begin
