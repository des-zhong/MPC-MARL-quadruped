import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from quadruped_learn.ppo_cse import ActorCritic
from quadruped_learn.ppo_cse.actor_critic import AC_Args
from quadruped_learn.ppo_cse import RolloutStorage
from quadruped_learn.ppo_cse import caches


def gaussian_kl_mean(old_mu, old_sigma, new_mu, new_sigma):
    """Numerically stable KL(old policy || new policy)."""

    old_sigma = old_sigma.clamp_min(1e-6)
    new_sigma = new_sigma.clamp_min(1e-6)
    per_dimension = (
        torch.log(new_sigma) - torch.log(old_sigma)
        + (old_sigma.square() + (old_mu - new_mu).square())
        / (2.0 * new_sigma.square())
        - 0.5
    )
    return per_dimension.sum(dim=-1).mean().clamp_min(0.0)


def hybrid_policy_kl_mean(
    old_parameters,
    old_std,
    new_parameters,
    new_std,
    action_stride=6,
    num_skill_logits=3,
):
    """Exact KL for categorical skill choices and Gaussian commands."""

    if action_stride <= 0 or num_skill_logits <= 1 or num_skill_logits > action_stride:
        raise ValueError("Invalid skill-action layout")
    if old_parameters.shape != new_parameters.shape:
        raise ValueError("Old and new hybrid policy parameters must have the same shape")
    if old_parameters.shape[-1] % action_stride != 0:
        raise ValueError(
            f"Action width {old_parameters.shape[-1]} is not divisible by stride {action_stride}"
        )
    old_grouped = old_parameters.reshape(
        *old_parameters.shape[:-1], -1, action_stride
    )
    new_grouped = new_parameters.reshape_as(old_grouped)
    old_log_probs = torch.log_softmax(old_grouped[..., :num_skill_logits], dim=-1)
    new_log_probs = torch.log_softmax(new_grouped[..., :num_skill_logits], dim=-1)
    old_probs = torch.exp(old_log_probs)
    categorical_kl = (
        old_probs * (old_log_probs - new_log_probs)
    ).sum(dim=-1).sum(dim=-1)

    old_std_grouped = torch.broadcast_to(old_std, old_parameters.shape).reshape_as(
        old_grouped
    )
    new_std_grouped = torch.broadcast_to(new_std, new_parameters.shape).reshape_as(
        new_grouped
    )
    old_command_mean = old_grouped[..., num_skill_logits:].reshape(
        *old_parameters.shape[:-1], -1
    )
    new_command_mean = new_grouped[..., num_skill_logits:].reshape_as(
        old_command_mean
    )
    old_command_std = old_std_grouped[..., num_skill_logits:].reshape_as(
        old_command_mean
    )
    new_command_std = new_std_grouped[..., num_skill_logits:].reshape_as(
        old_command_mean
    )
    command_kl = gaussian_kl_mean(
        old_command_mean,
        old_command_std,
        new_command_mean,
        new_command_std,
    )
    return (categorical_kl.mean() + command_kl).clamp_min(0.0)


def discrete_skill_direction_kl_mean(
    old_parameters,
    new_parameters,
    action_stride=12,
    num_skill_logits=4,
    num_direction_logits=8,
    stop_skill_id=3,
):
    """Exact KL for Stop-or-(skill, direction) categorical decisions."""

    if num_skill_logits + num_direction_logits != action_stride:
        raise ValueError(
            "Discrete action stride must equal skill logits plus direction logits"
        )
    if not 0 <= stop_skill_id < num_skill_logits:
        raise ValueError("stop_skill_id is outside the skill-logit block")
    if old_parameters.shape != new_parameters.shape:
        raise ValueError("Old and new discrete policy parameters must have the same shape")
    if old_parameters.shape[-1] % action_stride != 0:
        raise ValueError(
            f"Action width {old_parameters.shape[-1]} is not divisible by stride "
            f"{action_stride}"
        )

    old_grouped = old_parameters.reshape(
        *old_parameters.shape[:-1], -1, action_stride
    )
    new_grouped = new_parameters.reshape_as(old_grouped)
    old_skill_log_probs = torch.log_softmax(
        old_grouped[..., :num_skill_logits], dim=-1
    )
    new_skill_log_probs = torch.log_softmax(
        new_grouped[..., :num_skill_logits], dim=-1
    )
    old_skill_probs = torch.exp(old_skill_log_probs)
    skill_kl = (
        old_skill_probs * (old_skill_log_probs - new_skill_log_probs)
    ).sum(dim=-1)

    old_direction_log_probs = torch.log_softmax(
        old_grouped[..., num_skill_logits:], dim=-1
    )
    new_direction_log_probs = torch.log_softmax(
        new_grouped[..., num_skill_logits:], dim=-1
    )
    old_direction_probs = torch.exp(old_direction_log_probs)
    direction_kl = (
        old_direction_probs
        * (old_direction_log_probs - new_direction_log_probs)
    ).sum(dim=-1)
    active_probability = 1.0 - old_skill_probs[..., stop_skill_id]
    return (skill_kl + active_probability * direction_kl).sum(dim=-1).mean().clamp_min(0.0)


class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.
    min_learning_rate = 1e-5
    max_learning_rate = 1e-3
    # Optional safeguards for hybrid policies whose first action coordinates
    # encode an argmax-selected discrete skill. They are disabled for existing
    # low-level continuous-control jobs and enabled by train_high_level.py.
    skill_entropy_coef = 0.0
    skill_action_stride = 6
    num_skill_logits = 3
    num_direction_logits = 0
    stop_skill_id = 3
    stop_on_excessive_kl = False
    max_kl_factor = 4.0
    # Prevent an extreme likelihood-ratio exponent from overflowing before
    # the KL-based minibatch guard has a chance to reject the update.
    max_log_ratio = 20.0

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        
        PPO_Args.adaptation_labels = self.actor_critic.adaptation_labels
        PPO_Args.adaptation_dims = self.actor_critic.adaptation_dims
        PPO_Args.adaptation_weights = self.actor_critic.adaptation_weights
        
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPO_Args.learning_rate)
        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=PPO_Args.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=PPO_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPO_Args.learning_rate
        self.last_kl_mean = 0.0
        self.last_action_mean_abs = 0.0
        self.last_action_abs_max = 0.0
        self.last_skill_entropy = 0.0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(obs_history, privileged_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.last_action_mean_abs = self.transition.action_mean.abs().mean().item()
        self.last_action_abs_max = self.transition.actions.abs().max().item()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            time_outs = torch.as_tensor(infos['time_outs'], device=self.device)
            self.transition.rewards += PPO_Args.gamma * torch.squeeze(
                self.transition.values * time_outs.unsqueeze(1), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        
        mean_adaptation_losses = {}
        performed_updates = 0
        adaptation_updates = 0
        label_start_end = {}
        si = 0
        for idx, (label, length) in enumerate(zip(PPO_Args.adaptation_labels, PPO_Args.adaptation_dims)):
            label_start_end[label] = (si, si + length)
            si = si + length
            mean_adaptation_losses[label] = 0
        
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            self.actor_critic.act(obs_history_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_history_batch, privileged_obs_batch, masks=masks_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if PPO_Args.desired_kl is not None:
                with torch.inference_mode():
                    if self.actor_critic.discrete_skill_direction_policy:
                        kl_mean = discrete_skill_direction_kl_mean(
                            old_mu_batch,
                            mu_batch,
                            PPO_Args.skill_action_stride,
                            PPO_Args.num_skill_logits,
                            PPO_Args.num_direction_logits,
                            PPO_Args.stop_skill_id,
                        )
                    elif self.actor_critic.hybrid_skill_policy:
                        kl_mean = hybrid_policy_kl_mean(
                            old_mu_batch,
                            old_sigma_batch,
                            mu_batch,
                            sigma_batch,
                            PPO_Args.skill_action_stride,
                            PPO_Args.num_skill_logits,
                        )
                    else:
                        kl_mean = gaussian_kl_mean(
                            old_mu_batch,
                            old_sigma_batch,
                            mu_batch,
                            sigma_batch,
                        )
                    self.last_kl_mean = kl_mean.item()

                    if PPO_Args.schedule == 'adaptive':
                        if kl_mean > PPO_Args.desired_kl * 2.0:
                            self.learning_rate = max(PPO_Args.min_learning_rate, self.learning_rate / 1.5)
                        elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(PPO_Args.max_learning_rate, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                if (
                    PPO_Args.stop_on_excessive_kl
                    and PPO_Args.desired_kl is not None
                    and kl_mean > PPO_Args.desired_kl * PPO_Args.max_kl_factor
                ):
                    # This minibatch is already too far from the rollout
                    # policy. Applying another gradient step defeats adaptive
                    # learning-rate control and can irreversibly collapse an
                    # argmax-selected skill.
                    continue

            # Surrogate loss.  A very stale minibatch can produce a large
            # finite log-ratio even when the KL guard is disabled or delayed;
            # clamping keeps exp() finite and avoids turning one bad sample
            # into NaN actor gradients.
            log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            if self.actor_critic.hybrid_skill_policy:
                log_ratio = log_ratio.clamp(
                    min=-float(PPO_Args.max_log_ratio),
                    max=float(PPO_Args.max_log_ratio),
                )
            ratio = torch.exp(log_ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                               1.0 + PPO_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if PPO_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                          PPO_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            skill_entropy = torch.zeros((), device=mu_batch.device)
            if self.actor_critic.hybrid_skill_policy:
                skill_entropy = self.actor_critic.skill_entropy.mean()
                self.last_skill_entropy = skill_entropy.detach().item()
                entropy_bonus = (
                    PPO_Args.entropy_coef
                    * self.actor_critic.continuous_entropy.mean()
                    + PPO_Args.skill_entropy_coef * skill_entropy
                )
            else:
                self.last_skill_entropy = 0.0
                entropy_bonus = PPO_Args.entropy_coef * entropy_batch.mean()
            loss = (
                surrogate_loss
                + PPO_Args.value_loss_coef * value_loss
                - entropy_bonus
            )
            if not bool(torch.isfinite(loss)):
                continue

            # Gradient step.  Never pass non-finite gradients or parameters to
            # Adam: its running moments would otherwise permanently poison a
            # resumed high-level policy checkpoint.
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), PPO_Args.max_grad_norm
            )
            if not bool(torch.isfinite(grad_norm)):
                self.optimizer.zero_grad(set_to_none=True)
                continue
            parameter_backup = None
            if self.actor_critic.hybrid_skill_policy:
                parameter_backup = [
                    parameter.detach().clone()
                    for parameter in self.actor_critic.parameters()
                ]
            self.optimizer.step()
            if (
                parameter_backup is not None
                and not all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in self.actor_critic.parameters()
                )
            ):
                with torch.no_grad():
                    for parameter, backup in zip(
                        self.actor_critic.parameters(), parameter_backup
                    ):
                        parameter.copy_(backup)
                        state = self.optimizer.state.get(parameter)
                        if state is not None:
                            state.clear()
                self.optimizer.zero_grad(set_to_none=True)
                continue
            performed_updates += 1
            with torch.no_grad():
                self.actor_critic.std.clamp_(
                    min=AC_Args.min_action_std,
                    max=AC_Args.max_action_std,
                )

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch.shape[0]
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step, only update concurrent state estimation module, not policy network
            if len(PPO_Args.adaptation_labels) > 0:

                for epoch in range(PPO_Args.num_adaptation_module_substeps):

                    adaptation_pred = self.actor_critic.get_student_latent(obs_history_batch)
                    with torch.no_grad():
                        adaptation_target = privileged_obs_batch
                    adaptation_loss = 0
                    for idx, (label, length, weight) in enumerate(zip(PPO_Args.adaptation_labels, PPO_Args.adaptation_dims, PPO_Args.adaptation_weights)):

                        start, end = label_start_end[label]
                        selection_indices = torch.linspace(start, end - 1, steps=end - start, dtype=torch.long)

                        idx_adaptation_loss = F.mse_loss(adaptation_pred[:, selection_indices] * weight,
                                                        adaptation_target[:, selection_indices] * weight)
                        mean_adaptation_losses[label] += idx_adaptation_loss.item()

                        adaptation_loss += idx_adaptation_loss

                    self.adaptation_module_optimizer.zero_grad()
                    adaptation_loss.backward()
                    self.adaptation_module_optimizer.step()
                    adaptation_updates += 1

                    mean_adaptation_module_loss += adaptation_loss.item()
                    mean_adaptation_module_test_loss += 0  # adaptation_test_loss.item()

        num_updates = max(performed_updates, 1)
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        auxiliary_updates = max(adaptation_updates, 1)
        mean_adaptation_module_loss /= auxiliary_updates
        mean_decoder_loss /= auxiliary_updates
        mean_decoder_loss_student /= auxiliary_updates
        mean_adaptation_module_test_loss /= auxiliary_updates
        mean_decoder_test_loss /= auxiliary_updates
        mean_decoder_test_loss_student /= auxiliary_updates
        for label in PPO_Args.adaptation_labels:
            mean_adaptation_losses[label] /= auxiliary_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses
