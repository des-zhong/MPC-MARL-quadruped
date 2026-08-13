"""Regression tests for high-level hybrid PPO safeguards."""

import math

import torch

from dribblebot_learn.ppo_cse.ppo import (
    categorical_skill_entropy,
    gaussian_kl_mean,
)


def test_gaussian_kl_is_zero_for_identical_policies_and_never_negative():
    old_mu = torch.tensor([[0.0, 0.5, -0.5]])
    old_sigma = torch.tensor([[0.2, 0.3, 0.4]])

    identical = gaussian_kl_mean(old_mu, old_sigma, old_mu, old_sigma)
    shifted = gaussian_kl_mean(old_mu, old_sigma, old_mu + 0.25, old_sigma * 1.2)

    assert float(identical) == 0.0
    assert float(shifted) > 0.0


def test_skill_entropy_detects_argmax_collapse_for_each_robot():
    uniform_logits = torch.zeros(4, 12)
    collapsed_logits = uniform_logits.clone()
    collapsed_logits[:, 0] = 10.0
    collapsed_logits[:, 6] = 10.0

    uniform_entropy = categorical_skill_entropy(uniform_logits)
    collapsed_entropy = categorical_skill_entropy(collapsed_logits)

    assert torch.isclose(uniform_entropy, torch.tensor(math.log(3.0)))
    assert float(collapsed_entropy) < 0.01
    assert float(uniform_entropy) > float(collapsed_entropy)


def test_skill_entropy_accounts_for_gaussian_exploration_scale():
    means = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0]])
    low_std = torch.full_like(means, 0.1)
    high_std = torch.full_like(means, 1.0)

    low_noise_entropy = categorical_skill_entropy(means, action_std=low_std)
    high_noise_entropy = categorical_skill_entropy(means, action_std=high_std)

    assert float(low_noise_entropy) < 0.2
    assert float(high_noise_entropy) > 1.0
