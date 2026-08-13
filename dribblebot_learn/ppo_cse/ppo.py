import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from dribblebot_learn.ppo_cse import ActorCritic
from dribblebot_learn.ppo_cse.actor_critic import AC_Args
from dribblebot_learn.ppo_cse import RolloutStorage
from dribblebot_learn.ppo_cse import caches


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


def categorical_skill_entropy(
    action_mean,
    action_stride=6,
    num_skill_logits=3,
    action_std=None,
):
    """Approximate entropy of argmax skills from Gaussian action coordinates.

    Skill choice is made by taking an argmax after Gaussian sampling.  Raw
    means are therefore not calibrated logits: the same mean gap is exploratory
    at a large standard deviation and effectively deterministic at a small
    one.  Scaling by the sampling standard deviation makes this regularizer
    detect the collapse that actually reaches the environment.
    """

    if action_stride <= 0 or num_skill_logits <= 1 or num_skill_logits > action_stride:
        raise ValueError("Invalid skill-action layout")
    if action_mean.shape[-1] % action_stride != 0:
        raise ValueError(
            f"Action width {action_mean.shape[-1]} is not divisible by stride {action_stride}"
        )
    logits = action_mean.reshape(*action_mean.shape[:-1], -1, action_stride)[
        ..., :num_skill_logits
    ]
    if action_std is not None:
        if action_std.shape != action_mean.shape:
            action_std = torch.broadcast_to(action_std, action_mean.shape)
        skill_std = action_std.reshape(
            *action_std.shape[:-1], -1, action_stride
        )[..., :num_skill_logits]
        logits = logits / skill_std.clamp_min(1e-6)
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1).mean()


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
    stop_on_excessive_kl = False
    max_kl_factor = 4.0

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

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
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
            if PPO_Args.skill_entropy_coef > 0.0:
                skill_entropy = categorical_skill_entropy(
                    mu_batch,
                    PPO_Args.skill_action_stride,
                    PPO_Args.num_skill_logits,
                    sigma_batch,
                )
                self.last_skill_entropy = skill_entropy.detach().item()
            loss = (
                surrogate_loss
                + PPO_Args.value_loss_coef * value_loss
                - PPO_Args.entropy_coef * entropy_batch.mean()
                - PPO_Args.skill_entropy_coef * skill_entropy
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPO_Args.max_grad_norm)
            self.optimizer.step()
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
