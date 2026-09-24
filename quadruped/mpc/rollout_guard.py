"""Check teacher proposals against advantages measured in the real simulator."""
import torch
from quadruped.mpc.teacher_quality import policy_kl


def surrogate(log_prob, old_log_prob, advantages, clip=.2):
    ratio = (log_prob.flatten() - old_log_prob.flatten()).clamp(-20, 20).exp()
    advantage = advantages.flatten()
    return torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).mean()


class RealRolloutGuard:
    """A fixed post-PPO reference: repeated teacher steps share one KL budget.

    This is an empirical PPO-surrogate check, not a guarantee about future games.
    The anchor data never participates in the distillation loss.
    """
    def __init__(self, policy, batch, clip=.2):
        self.batch = batch
        self.clip = clip
        with torch.no_grad():
            policy.update_distribution(batch['histories'])
            self.mean = policy.action_mean.detach().clone()
            self.std = policy.action_std.detach().clone()
            self.score = surrogate(policy.get_actions_log_prob(batch['actions']),
                                   batch['log_prob'], batch['advantages'], clip)

    def accepts(self, policy, kl_limit):
        with torch.no_grad():
            policy.update_distribution(self.batch['histories'])
            score = surrogate(policy.get_actions_log_prob(self.batch['actions']),
                              self.batch['log_prob'], self.batch['advantages'], self.clip)
            kl = policy_kl(self.mean, self.std, policy.action_mean, policy.action_std)
            valid = torch.isfinite(score) & torch.isfinite(kl)
            return bool(valid & (score >= self.score - 1e-6) & (kl <= kl_limit))
