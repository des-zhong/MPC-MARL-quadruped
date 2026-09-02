"""Hybrid categorical/continuous actor-critic for the MARL coordinator."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


NUM_SKILLS = 3
NUM_SKILL_PARAMETERS = 3
HYBRID_ACTION_DIM = 1 + NUM_SKILL_PARAMETERS


def actor_output_to_hybrid_action(actor_output: torch.Tensor) -> torch.Tensor:
    """Decode deterministic actor output into ``[skill_index, parameters]``."""

    if actor_output.ndim != 2 or actor_output.shape[-1] != NUM_SKILLS + NUM_SKILL_PARAMETERS:
        raise ValueError(
            "Hybrid actor output must have shape "
            f"(N, {NUM_SKILLS + NUM_SKILL_PARAMETERS}), got {tuple(actor_output.shape)}"
        )
    skill_index = actor_output[:, :NUM_SKILLS].argmax(dim=-1, keepdim=True)
    return torch.cat((skill_index.to(actor_output.dtype), actor_output[:, NUM_SKILLS:]), dim=-1)


class HybridActorCritic(nn.Module):
    """PPO policy with a categorical skill and Gaussian skill parameters.

    RSL-RL transports actions in one dense tensor. The first scalar is an
    integer-valued sample from ``Categorical(skill_logits)`` and the remaining
    three scalars are sampled from a diagonal Normal distribution. Their joint
    log probability and entropy are used by PPO.
    """

    is_recurrent = False
    hybrid_action = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "HybridActorCritic.__init__ ignored arguments: "
                + str(sorted(kwargs))
            )
        super().__init__()
        if int(num_actions) != HYBRID_ACTION_DIM:
            raise ValueError(
                f"HybridActorCritic requires {HYBRID_ACTION_DIM} transported actions, got {num_actions}"
            )
        if state_dependent_std:
            raise ValueError("HybridActorCritic does not support state-dependent parameter std")

        self.obs_groups = obs_groups
        num_actor_obs = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        num_critic_obs = sum(obs[name].shape[-1] for name in obs_groups["critic"])
        self.actor = MLP(
            num_actor_obs,
            NUM_SKILLS + NUM_SKILL_PARAMETERS,
            list(actor_hidden_dims),
            activation,
        )
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs)
            if self.actor_obs_normalization
            else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs)
            if self.critic_obs_normalization
            else nn.Identity()
        )
        self.noise_std_type = str(noise_std_type)
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(float(init_noise_std) * torch.ones(NUM_SKILL_PARAMETERS))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(float(init_noise_std) * torch.ones(NUM_SKILL_PARAMETERS))
            )
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")

        self.skill_distribution: Categorical | None = None
        self.parameter_distribution: Normal | None = None
        Categorical.set_default_validate_args(False)
        Normal.set_default_validate_args(False)
        print(f"Hybrid actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def reset(self, dones=None) -> None:
        del dones

    def forward(self):
        raise NotImplementedError

    def get_actor_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["critic"]], dim=-1)

    def _parameter_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.abs().clamp_min(1.0e-4).expand_as(mean)
        return torch.exp(self.log_std).expand_as(mean)

    def update_distribution(self, actor_obs: torch.Tensor) -> None:
        output = self.actor(actor_obs)
        logits = output[:, :NUM_SKILLS]
        parameter_mean = output[:, NUM_SKILLS:]
        self.skill_distribution = Categorical(logits=logits)
        self.parameter_distribution = Normal(parameter_mean, self._parameter_std(parameter_mean))

    def _require_distribution(self) -> tuple[Categorical, Normal]:
        if self.skill_distribution is None or self.parameter_distribution is None:
            raise RuntimeError("Action distribution has not been initialized")
        return self.skill_distribution, self.parameter_distribution

    def act(self, obs, **kwargs) -> torch.Tensor:
        del kwargs
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(actor_obs)
        skill_distribution, parameter_distribution = self._require_distribution()
        skill = skill_distribution.sample().to(parameter_distribution.mean.dtype).unsqueeze(-1)
        return torch.cat((skill, parameter_distribution.sample()), dim=-1)

    def act_inference(self, obs) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return actor_output_to_hybrid_action(self.actor(actor_obs))

    def evaluate(self, obs, **kwargs) -> torch.Tensor:
        del kwargs
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != HYBRID_ACTION_DIM:
            raise ValueError(f"Expected hybrid actions (N,{HYBRID_ACTION_DIM}), got {tuple(actions.shape)}")
        skill_distribution, parameter_distribution = self._require_distribution()
        skill = actions[:, 0].round().clamp(0, NUM_SKILLS - 1).long()
        return skill_distribution.log_prob(skill) + parameter_distribution.log_prob(actions[:, 1:]).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        skill_distribution, parameter_distribution = self._require_distribution()
        indices = torch.arange(NUM_SKILLS, device=parameter_distribution.mean.device, dtype=parameter_distribution.mean.dtype)
        expected_skill = torch.sum(skill_distribution.probs * indices, dim=-1, keepdim=True)
        return torch.cat((expected_skill, parameter_distribution.mean), dim=-1)

    @property
    def action_std(self) -> torch.Tensor:
        skill_distribution, parameter_distribution = self._require_distribution()
        indices = torch.arange(NUM_SKILLS, device=parameter_distribution.mean.device, dtype=parameter_distribution.mean.dtype)
        expected_skill = torch.sum(skill_distribution.probs * indices, dim=-1, keepdim=True)
        skill_variance = torch.sum(
            skill_distribution.probs * (indices - expected_skill).square(), dim=-1, keepdim=True
        )
        return torch.cat((skill_variance.clamp_min(1.0e-8).sqrt(), parameter_distribution.stddev), dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        skill_distribution, parameter_distribution = self._require_distribution()
        return skill_distribution.entropy() + parameter_distribution.entropy().sum(dim=-1)

    def update_normalization(self, obs) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        # Existing six-output Gaussian coordinators share the same actor head,
        # but stored six noise values (three former logit noises plus three
        # parameter noises). Reuse only the parameter part when opening one as
        # a hybrid policy; optimizer state is intentionally not migrated.
        state_dict = dict(state_dict)
        for key in ("std", "log_std"):
            value = state_dict.get(key)
            if value is not None and value.numel() == NUM_SKILLS + NUM_SKILL_PARAMETERS:
                state_dict[key] = value[-NUM_SKILL_PARAMETERS:]
        super().load_state_dict(state_dict, strict=strict)
        return True


__all__ = [
    "HYBRID_ACTION_DIM",
    "HybridActorCritic",
    "NUM_SKILLS",
    "NUM_SKILL_PARAMETERS",
    "actor_output_to_hybrid_action",
]
