"""Regression tests for high-level hybrid PPO safeguards."""

import math
from types import SimpleNamespace

import torch

from dribblebot_learn.ppo_cse.ppo import (
    gaussian_kl_mean,
    hybrid_policy_kl_mean,
)
from dribblebot_learn.ppo_cse.actor_critic import (
    ActorCritic,
    HybridSkillDistribution,
)


def test_gaussian_kl_is_zero_for_identical_policies_and_never_negative():
    old_mu = torch.tensor([[0.0, 0.5, -0.5]])
    old_sigma = torch.tensor([[0.2, 0.3, 0.4]])

    identical = gaussian_kl_mean(old_mu, old_sigma, old_mu, old_sigma)
    shifted = gaussian_kl_mean(old_mu, old_sigma, old_mu + 0.25, old_sigma * 1.2)

    assert float(identical) == 0.0
    assert float(shifted) > 0.0


def test_true_categorical_skill_entropy_detects_collapse():
    uniform = torch.zeros(4, 6)
    collapsed = uniform.clone()
    collapsed[:, 0] = 10.0
    std = torch.full((6,), 0.2)

    uniform_entropy = HybridSkillDistribution(uniform, std).skill_entropy.mean()
    collapsed_entropy = HybridSkillDistribution(collapsed, std).skill_entropy.mean()

    assert torch.isclose(uniform_entropy, torch.tensor(math.log(3.0)))
    assert float(collapsed_entropy) < 0.01
    assert float(uniform_entropy) > float(collapsed_entropy)


def test_categorical_skill_entropy_is_independent_of_command_noise():
    parameters = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0]])
    low_std = torch.full_like(parameters, 0.1)
    high_std = torch.full_like(parameters, 1.0)

    low_noise_entropy = HybridSkillDistribution(
        parameters, low_std
    ).skill_entropy.mean()
    high_noise_entropy = HybridSkillDistribution(
        parameters, high_std
    ).skill_entropy.mean()

    torch.testing.assert_close(low_noise_entropy, high_noise_entropy)


def test_hybrid_distribution_samples_one_hot_skill_and_exact_log_prob():
    parameters = torch.tensor([[1.5, 0.0, -0.5, 0.2, -0.1, 0.4]])
    distribution = HybridSkillDistribution(parameters, torch.full((6,), 0.2))

    actions = distribution.sample()

    assert actions.shape == parameters.shape
    torch.testing.assert_close(actions[:, :3].sum(dim=-1), torch.ones(1))
    assert torch.all((actions[:, :3] == 0.0) | (actions[:, :3] == 1.0))
    assert distribution.log_prob(actions).shape == (1,)


def test_actor_critic_keeps_one_hybrid_log_probability_per_sample():
    parameters = torch.randn(7, 6)
    distribution = HybridSkillDistribution(parameters, torch.full((6,), 0.2))
    policy = SimpleNamespace(
        distribution=distribution,
        hybrid_skill_policy=True,
    )
    actions = distribution.sample()

    log_prob = ActorCritic.get_actions_log_prob(policy, actions)

    assert log_prob.shape == (7,)
    torch.testing.assert_close(log_prob, distribution.log_prob(actions))


def test_hybrid_kl_includes_categorical_and_continuous_changes():
    old = torch.zeros(2, 6)
    std = torch.full_like(old, 0.2)

    identical = hybrid_policy_kl_mean(old, std, old, std)
    changed_skill = old.clone()
    changed_skill[:, 0] = 1.0
    changed_command = old.clone()
    changed_command[:, 3] = 0.25

    assert float(identical) == 0.0
    assert float(hybrid_policy_kl_mean(old, std, changed_skill, std)) > 0.0
    assert float(hybrid_policy_kl_mean(old, std, changed_command, std)) > 0.0
