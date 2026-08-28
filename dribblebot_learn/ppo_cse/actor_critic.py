import torch
import torch.nn as nn
import torch.nn.functional as F
from params_proto import PrefixProto
from torch.distributions import Categorical, Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    min_action_std = 0.05
    max_action_std = 2.0
    # Optional bound applied to the Gaussian mean. Individual training scripts
    # can enable it without changing the checkpoint parameter structure.
    action_mean_bound = None
    # High-level soccer uses a hybrid action: the first three values of every
    # six-value block are categorical skill logits and the remaining values are
    # continuous command means. The actor output width is unchanged so legacy
    # checkpoints remain loadable.
    hybrid_skill_policy = False
    skill_action_stride = 6
    num_skill_logits = 3
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]
    
    adaptation_labels = []
    adaptation_dims = []
    adaptation_weights = []

    use_decoder = False


class ActionMeanBound(nn.Module):
    """Parameter-free output bound that is preserved in exported actor JITs."""

    __constants__ = ["bound"]

    def __init__(self, bound):
        super().__init__()
        self.bound = float(bound)
        if self.bound <= 0.0:
            raise ValueError(f"Action mean bound must be positive, got {bound}")

    def forward(self, actions):
        return self.bound * torch.tanh(actions / self.bound)


class HybridSkillDistribution:
    """Categorical skill choice plus Gaussian command distribution.

    Parameters are packed in the environment's existing action layout. This
    lets rollout storage and old six-output actor checkpoints keep their tensor
    shapes while giving PPO exact categorical log probabilities and entropy.
    """

    def __init__(self, parameters, std, action_stride=6, num_skill_logits=3):
        if action_stride <= 0 or not 1 < num_skill_logits < action_stride:
            raise ValueError("Invalid hybrid skill-action layout")
        if parameters.shape[-1] % action_stride != 0:
            raise ValueError(
                f"Action width {parameters.shape[-1]} is not divisible by "
                f"stride {action_stride}"
            )
        self.parameters = parameters
        self.action_stride = int(action_stride)
        self.num_skill_logits = int(num_skill_logits)
        self.group_count = parameters.shape[-1] // self.action_stride
        grouped = parameters.reshape(
            *parameters.shape[:-1], self.group_count, self.action_stride
        )
        grouped_std = torch.broadcast_to(std, parameters.shape).reshape_as(grouped)
        self.skill_logits = grouped[..., : self.num_skill_logits]
        self.command_mean = grouped[..., self.num_skill_logits :]
        self.command_std = grouped_std[..., self.num_skill_logits :].clamp_min(1e-6)
        self.skill_distribution = Categorical(logits=self.skill_logits)
        self.command_distribution = Normal(self.command_mean, self.command_std)

    @property
    def mean(self):
        """Packed policy parameters used by rollout storage and exact KL."""

        return self.parameters

    @property
    def stddev(self):
        grouped = torch.zeros_like(
            self.parameters.reshape(
                *self.parameters.shape[:-1], self.group_count, self.action_stride
            )
        )
        grouped[..., self.num_skill_logits :] = self.command_std
        return grouped.reshape_as(self.parameters)

    @property
    def skill_entropy(self):
        return self.skill_distribution.entropy().sum(dim=-1)

    @property
    def continuous_entropy(self):
        return self.command_distribution.entropy().sum(dim=(-1, -2))

    def entropy(self):
        return self.skill_entropy + self.continuous_entropy

    def sample(self):
        skill_ids = self.skill_distribution.sample()
        skill_one_hot = F.one_hot(
            skill_ids, num_classes=self.num_skill_logits
        ).to(self.parameters.dtype)
        commands = self.command_distribution.sample()
        return torch.cat((skill_one_hot, commands), dim=-1).reshape_as(self.parameters)

    def mode(self):
        skill_ids = torch.argmax(self.skill_logits, dim=-1)
        skill_one_hot = F.one_hot(
            skill_ids, num_classes=self.num_skill_logits
        ).to(self.parameters.dtype)
        return torch.cat((skill_one_hot, self.command_mean), dim=-1).reshape_as(
            self.parameters
        )

    def log_prob(self, actions):
        grouped_actions = actions.reshape(
            *actions.shape[:-1], self.group_count, self.action_stride
        )
        skill_ids = torch.argmax(
            grouped_actions[..., : self.num_skill_logits], dim=-1
        )
        skill_log_prob = self.skill_distribution.log_prob(skill_ids).sum(dim=-1)
        command_log_prob = self.command_distribution.log_prob(
            grouped_actions[..., self.num_skill_logits :]
        ).sum(dim=(-1, -2))
        return skill_log_prob + command_log_prob


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()
        
        self.adaptation_labels = AC_Args.adaptation_labels
        self.adaptation_dims = AC_Args.adaptation_dims
        self.adaptation_weights = AC_Args.adaptation_weights

        if len(self.adaptation_weights) < len(self.adaptation_labels):
            # pad
            self.adaptation_weights += [1.0] * (len(self.adaptation_labels) - len(self.adaptation_weights))

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(AC_Args.activation)

        # Adaptation module
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)



        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if AC_Args.action_mean_bound is not None and float(AC_Args.action_mean_bound) > 0.0:
            actor_layers.append(ActionMeanBound(AC_Args.action_mean_bound))
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.hybrid_skill_policy = bool(AC_Args.hybrid_skill_policy)
        self.skill_action_stride = int(AC_Args.skill_action_stride)
        self.num_skill_logits = int(AC_Args.num_skill_logits)
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.hybrid_skill_policy:
            return self.distribution.entropy()
        return self.distribution.entropy().sum(dim=-1)

    @property
    def skill_entropy(self):
        if not self.hybrid_skill_policy:
            return self.entropy.new_zeros(self.entropy.shape)
        return self.distribution.skill_entropy

    @property
    def continuous_entropy(self):
        if not self.hybrid_skill_policy:
            return self.entropy
        return self.distribution.continuous_entropy

    def update_distribution(self, observation_history):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        std = self.std.clamp(min=AC_Args.min_action_std, max=AC_Args.max_action_std)
        if self.hybrid_skill_policy:
            self.distribution = HybridSkillDistribution(
                mean,
                std,
                action_stride=self.skill_action_stride,
                num_skill_logits=self.num_skill_logits,
            )
        else:
            self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        log_prob = self.distribution.log_prob(actions)
        # Normal.log_prob returns one value per action coordinate.  The hybrid
        # distribution has already reduced its categorical and command terms
        # to one joint value per sample, so summing it again would incorrectly
        # reduce the entire minibatch to a scalar.
        if self.hybrid_skill_policy:
            return log_prob
        return log_prob.sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_info={}):
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        if self.hybrid_skill_policy:
            std = self.std.clamp(min=AC_Args.min_action_std, max=AC_Args.max_action_std)
            return HybridSkillDistribution(
                actions_mean,
                std,
                action_stride=self.skill_action_stride,
                num_skill_logits=self.num_skill_logits,
            ).mode()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        if self.hybrid_skill_policy:
            std = self.std.clamp(min=AC_Args.min_action_std, max=AC_Args.max_action_std)
            return HybridSkillDistribution(
                actions_mean,
                std,
                action_stride=self.skill_action_stride,
                num_skill_logits=self.num_skill_logits,
            ).mode()
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        value = self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
