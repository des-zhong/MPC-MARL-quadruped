"""Quality filters and bounded policy updates for an advisory MPC teacher."""
import torch
import torch.nn.functional as F


def improvement_mask(teacher_return, student_return, valid, margin):
    advantage = teacher_return - student_return
    accept = valid & torch.isfinite(advantage) & (advantage > margin)
    # Never let invalid scores contaminate weighted losses through 0 * inf.
    weight = torch.where(accept, (advantage - margin).clamp(0., 1.), torch.zeros_like(advantage))
    return accept, weight, advantage


def policy_kl(old_mean, old_std, new_mean, new_std):
    """Old-to-new KL of the hybrid policy; categorical noise slots are ignored."""
    old_log = F.log_softmax(old_mean[:, :3], -1)
    new_log = F.log_softmax(new_mean[:, :3], -1)
    categorical = (old_log.exp() * (old_log-new_log)).sum(-1)
    a, b = old_std[:, 3:].clamp_min(1e-6), new_std[:, 3:].clamp_min(1e-6)
    normal = (b.log()-a.log() + (a.square()+(old_mean[:, 3:]-new_mean[:, 3:]).square())/(2*b.square())-.5).sum(-1)
    return (categorical + normal).mean()


def bounded_mean_loss(policy_mean, targets, target_probs, masks):
    """Imitate commands, not CEM search variance (which is not policy exploration)."""
    skill = target_probs.argmax(-1)
    row = torch.arange(len(skill), device=skill.device)
    selected = targets[row, skill].tanh()
    mask = masks[row, skill]
    return (((policy_mean[:, 3:].tanh()-selected).square()*mask).sum(-1)
            / mask.sum(-1).clamp_min(1.))


def selected_action_targets(plan, adapter, team):
    skills, commands = adapter.unpack(plan.first_joint_action)
    skills, commands = skills[:, :team], commands[:, :team]
    normalized = adapter.normalize_parameters(skills, commands).clamp(-.95, .95)
    raw = torch.atanh(normalized).reshape(-1, 3)
    probs = F.one_hot(skills.long(), 3).float().reshape(-1, 3)
    means = raw[:, None, :].expand(-1, 3, -1).clone()
    masks = torch.tensor([adapter.bounds[i].mask for i in range(3)], device=raw.device, dtype=raw.dtype)
    return means, torch.full_like(means, .1), probs, masks
