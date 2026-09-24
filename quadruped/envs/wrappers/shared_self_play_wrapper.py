import copy
import math

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaacgym.torch_utils import quat_apply
from torch.distributions import Categorical

from .team_frame import (
    mirror_high_level_commands,
    mirror_high_level_policy_actions,
)


class FrozenOpponentPolicy(nn.Module):
    """Frozen copy of the trainable actor used for stochastic self-play.

    ``ActorCritic`` caches a ``Normal`` distribution whose mean is produced by
    the latest forward pass.  Those cached non-leaf tensors make the complete
    training module unsafe to deepcopy.  Snapshot the inference networks and
    the effective action standard deviation, then reconstruct the same hybrid
    or categorical sampling performed by the trainable policy.  This prevents
    a frozen opponent from receiving an unintended deterministic advantage
    over the exploratory PPO policy.
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.adaptation_module = copy.deepcopy(actor_critic.adaptation_module)
        self.actor_body = copy.deepcopy(actor_critic.actor_body)
        self.discrete_skill_direction_policy = bool(
            getattr(actor_critic, "discrete_skill_direction_policy", False)
        )
        self.hybrid_skill_policy = bool(
            getattr(actor_critic, "hybrid_skill_policy", False)
        )
        self.skill_action_stride = int(
            getattr(actor_critic, "skill_action_stride", 6)
        )
        self.num_skill_logits = int(getattr(actor_critic, "num_skill_logits", 3))
        self.num_direction_logits = int(
            getattr(actor_critic, "num_direction_logits", 0)
        )
        self.stop_skill_id = int(getattr(actor_critic, "stop_skill_id", 3))

        action_std = getattr(actor_critic, "std", None)
        if action_std is not None and not self.discrete_skill_direction_policy:
            min_std = float(getattr(actor_critic, "min_action_std", 0.05))
            max_std = float(getattr(actor_critic, "max_action_std", 2.0))
            self.register_buffer(
                "std",
                action_std.detach().clamp(min=min_std, max=max_std).clone(),
            )
        else:
            self.register_buffer("std", None)

    def _action_parameters(self, observation_history):
        latent = self.adaptation_module(observation_history)
        parameters = self.actor_body(
            torch.cat((observation_history, latent), dim=-1)
        )
        if parameters.shape[-1] % self.skill_action_stride != 0:
            raise ValueError(
                f"Opponent action width {parameters.shape[-1]} is not divisible "
                f"by stride {self.skill_action_stride}"
            )
        return parameters

    def act_student(self, observation_history):
        """Return deterministic mode actions for evaluation/deployment."""

        parameters = self._action_parameters(observation_history)
        grouped = parameters.reshape(
            *parameters.shape[:-1],
            parameters.shape[-1] // self.skill_action_stride,
            self.skill_action_stride,
        )
        if self.discrete_skill_direction_policy:
            skill_ids = grouped[..., : self.num_skill_logits].argmax(dim=-1)
            direction_ids = grouped[..., self.num_skill_logits :].argmax(dim=-1)
            skills = F.one_hot(
                skill_ids, num_classes=self.num_skill_logits
            ).to(parameters.dtype)
            directions = F.one_hot(
                direction_ids, num_classes=self.num_direction_logits
            ).to(parameters.dtype)
            active = (skill_ids != self.stop_skill_id).unsqueeze(-1)
            directions = directions * active.to(parameters.dtype)
            return torch.cat((skills, directions), dim=-1).reshape_as(parameters)

        if self.hybrid_skill_policy:
            skill_ids = grouped[..., : self.num_skill_logits].argmax(dim=-1)
            skills = F.one_hot(
                skill_ids, num_classes=self.num_skill_logits
            ).to(parameters.dtype)
            return torch.cat(
                (skills, grouped[..., self.num_skill_logits :]), dim=-1
            ).reshape_as(parameters)
        return parameters

    def act_training(self, observation_history):
        """Sample the frozen policy with the same distribution as PPO."""

        parameters = self._action_parameters(observation_history)

        group_count = parameters.shape[-1] // self.skill_action_stride
        grouped = parameters.reshape(
            *parameters.shape[:-1], group_count, self.skill_action_stride
        )
        if self.discrete_skill_direction_policy:
            skill_ids = Categorical(
                logits=grouped[..., : self.num_skill_logits]
            ).sample()
            direction_ids = Categorical(
                logits=grouped[..., self.num_skill_logits :]
            ).sample()
            skills = F.one_hot(
                skill_ids, num_classes=self.num_skill_logits
            ).to(parameters.dtype)
            directions = F.one_hot(
                direction_ids, num_classes=self.num_direction_logits
            ).to(parameters.dtype)
            active = (skill_ids != self.stop_skill_id).unsqueeze(-1)
            directions = directions * active.to(parameters.dtype)
            return torch.cat((skills, directions), dim=-1).reshape_as(parameters)

        if self.hybrid_skill_policy:
            skill_ids = Categorical(
                logits=grouped[..., : self.num_skill_logits]
            ).sample()
            skills = F.one_hot(
                skill_ids, num_classes=self.num_skill_logits
            ).to(parameters.dtype)
            command_mean = grouped[..., self.num_skill_logits :]
            grouped_std = torch.broadcast_to(self.std, parameters.shape).reshape_as(
                grouped
            )
            command_std = grouped_std[..., self.num_skill_logits :].clamp_min(
                1e-6
            )
            commands = torch.normal(command_mean, command_std)
            return torch.cat((skills, commands), dim=-1).reshape_as(parameters)

        if self.std is None:
            return parameters
        return torch.normal(parameters, torch.broadcast_to(self.std, parameters.shape))


class SharedPolicySelfPlayWrapper(gym.Wrapper):
    """Expose one shared-policy sample per robot and drive an opponent.

    The wrapped high-level environment owns ``2 * team_size`` physical AS2
    actors. Slots ``[0, team_size)`` are the learning team and the remaining
    slots are the opponent team.  PPO sees ``match_count * team_size`` agents,
    each with the same fixed-size, agent-centric observation and six actions
    (or twelve packed categorical values in discrete mode).
    The opponent may be a frozen policy or a simple external action provider.
    """

    LOCAL_OBS_DIM = 34

    def __init__(
        self,
        env,
        team_size,
        opponent_device=None,
        opponent_pool_size=8,
        opponent_latest_probability=0.5,
    ):
        super().__init__(env)
        self.env = env
        self.team_size = int(team_size)
        if self.team_size < 1:
            raise ValueError("team_size must be at least 1")
        if int(env.num_robots) != 2 * self.team_size:
            raise ValueError(
                "Self-play requires exactly two equal teams: "
                f"got {env.num_robots} physical robots for team_size={self.team_size}"
            )

        self.match_count = int(env.num_envs)
        self.num_envs = self.match_count * self.team_size
        self.num_train_envs = int(env.num_train_envs) * self.team_size
        self.num_robots = self.team_size
        self.action_encoding = str(
            getattr(env, "action_encoding", getattr(env.cfg.env, "high_level_action_encoding", "hybrid"))
        )
        self.num_actions = 12 if self.action_encoding == "discrete_skill_direction" else 6
        # Discrete coordinators expose the Stop state explicitly.  Legacy
        # hybrid coordinators retain the original 34-value observation.
        self.num_obs = self.LOCAL_OBS_DIM + (
            1 if self.action_encoding == "discrete_skill_direction" else 0
        )
        self.shooting_options = bool(getattr(env.cfg.env, "high_level_shooting_options", False))
        self.num_obs += int(self.shooting_options)
        self.centralized_critic = bool(getattr(env.cfg.env, "centralized_critic", False))
        self.num_privileged_obs = self.num_obs * (2*self.team_size if self.centralized_critic else 1)
        self.history_length = int(env.history_length)
        self.num_obs_history = self.num_obs * self.history_length
        self.max_episode_length = env.max_episode_length
        self.device = env.device
        self.opponent_device = torch.device(opponent_device or self.device)
        self.opponent_policy = None
        self.opponent_policy_callable = None
        self.opponent_action_provider = None
        self.opponent_snapshot_iteration = -1
        self.opponent_pool_size = int(opponent_pool_size)
        self.opponent_latest_probability = float(opponent_latest_probability)
        if self.opponent_pool_size < 1:
            raise ValueError("opponent_pool_size must be at least 1")
        if not 0.0 <= self.opponent_latest_probability <= 1.0:
            raise ValueError("opponent_latest_probability must be between 0 and 1")
        self.opponent_pool = []
        self.opponent_pool_iterations = []
        self.opponent_assignment = torch.zeros(
            self.match_count, dtype=torch.long, device=self.device
        )
        # Online MPC may preview the opponent action before the environment
        # step. Cache that sampled action so planning and execution use the
        # same stochastic draw.
        self._cached_opponent_actions = None

        self._history = torch.zeros(
            self.match_count,
            2,
            self.team_size,
            self.num_obs_history,
            dtype=torch.float,
            device=self.device,
        )
        self._cached = None

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def actions(self):
        # Runner diagnostics concern only the trainable team.
        return self.env.actions[:, : 12 * self.team_size]

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf.repeat_interleave(self.team_size)

    def randomize_episode_lengths(self):
        self.env.episode_length_buf[:] = torch.randint_like(
            self.env.episode_length_buf,
            high=int(self.max_episode_length),
        )

    def _frozen_snapshot(self, actor_critic):
        snapshot = FrozenOpponentPolicy(actor_critic).to(self.opponent_device)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        return snapshot

    @staticmethod
    def _snapshot_missing_keys(incompatible):
        """Ignore action noise absent from deterministic legacy snapshots."""

        return [key for key in incompatible.missing_keys if key != "std"]

    def _sample_opponent_assignments(self, mask=None):
        if not self.opponent_pool:
            self.opponent_assignment.zero_()
            return
        if mask is None:
            target = torch.arange(self.match_count, device=self.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(-1)
            target = mask.nonzero(as_tuple=False).flatten()
        if target.numel() == 0:
            return

        pool_count = len(self.opponent_pool)
        if pool_count == 1:
            sampled = torch.zeros(target.numel(), dtype=torch.long, device=self.device)
        else:
            weights = torch.full(
                (pool_count,),
                (1.0 - self.opponent_latest_probability) / (pool_count - 1),
                device=self.device,
            )
            weights[-1] = self.opponent_latest_probability
            if float(weights.sum()) <= 0.0:
                weights.fill_(1.0 / pool_count)
            sampled = torch.multinomial(weights, target.numel(), replacement=True)
        self.opponent_assignment[target] = sampled

    def update_opponent_policy(self, actor_critic, iteration=0):
        """Add a frozen snapshot and resample opponents across parallel matches."""
        self._cached_opponent_actions = None
        if self.opponent_action_provider is not None:
            self.opponent_snapshot_iteration = int(iteration)
            return
        snapshot = self._frozen_snapshot(actor_critic)
        had_pool = bool(self.opponent_pool)
        evicted_assignment_mask = None
        if len(self.opponent_pool) >= self.opponent_pool_size:
            # Preserve the initialization policy as a stable weak/anchor
            # opponent and evict the oldest non-anchor snapshot.
            evict = 1 if self.opponent_pool_size > 1 else 0
            evicted_assignment_mask = self.opponent_assignment == evict
            self.opponent_assignment[self.opponent_assignment > evict] -= 1
            self.opponent_pool.pop(evict)
            self.opponent_pool_iterations.pop(evict)
        self.opponent_pool.append(snapshot)
        self.opponent_pool_iterations.append(int(iteration))
        self.opponent_policy = snapshot
        self.opponent_policy_callable = None
        self.opponent_snapshot_iteration = int(iteration)
        from quadruped.envs.soccer_curriculum import get_curriculum
        curriculum = get_curriculum(self.env.env)
        if curriculum is not None:
            curriculum.opponent_window.clear()
            for window in curriculum.windows:
                window.clear()
        if not had_pool:
            self._sample_opponent_assignments()
        elif evicted_assignment_mask is not None:
            self._sample_opponent_assignments(evicted_assignment_mask)

    def load_opponent_policy_state_dict(self, state_dict, actor_critic, iteration=-1):
        self._cached_opponent_actions = None
        if self.opponent_action_provider is not None:
            self.opponent_snapshot_iteration = int(iteration)
            return None
        snapshot = self._frozen_snapshot(actor_critic)
        incompatible = snapshot.load_state_dict(state_dict, strict=False)
        missing_keys = self._snapshot_missing_keys(incompatible)
        if missing_keys:
            raise ValueError(
                "Opponent checkpoint is missing inference weights: "
                f"{missing_keys}"
            )
        self.opponent_pool = [snapshot]
        self.opponent_pool_iterations = [int(iteration)]
        self.opponent_policy = snapshot
        self.opponent_policy_callable = None
        self.opponent_snapshot_iteration = int(iteration)
        self._sample_opponent_assignments()

    def load_opponent_pool_state_dict(self, payload, actor_critic):
        self._cached_opponent_actions = None
        if not isinstance(payload, dict) or not payload.get("policies"):
            raise ValueError("Opponent-pool checkpoint contains no policies")
        from quadruped.envs.soccer_curriculum import get_curriculum
        curriculum = get_curriculum(self.env.env)
        if curriculum is not None and payload.get("soccer_curriculum") is not None:
            curriculum.load_state_dict(payload["soccer_curriculum"])
        policies = payload["policies"]
        iterations = payload.get("iterations", list(range(len(policies))))
        if len(policies) != len(iterations):
            raise ValueError("Opponent-pool policies and iterations have different lengths")
        if len(policies) > self.opponent_pool_size:
            if self.opponent_pool_size == 1:
                selected_indices = [len(policies) - 1]
            else:
                selected_indices = [0] + list(
                    range(len(policies) - self.opponent_pool_size + 1, len(policies))
                )
            policies = [policies[index] for index in selected_indices]
            iterations = [iterations[index] for index in selected_indices]
        self.opponent_pool = []
        self.opponent_pool_iterations = []
        for state_dict, iteration in zip(policies, iterations):
            snapshot = self._frozen_snapshot(actor_critic)
            incompatible = snapshot.load_state_dict(state_dict, strict=False)
            missing_keys = self._snapshot_missing_keys(incompatible)
            if missing_keys:
                raise ValueError(
                    "Opponent-pool checkpoint is missing inference weights: "
                    f"{missing_keys}"
                )
            self.opponent_pool.append(snapshot)
            self.opponent_pool_iterations.append(int(iteration))
        self.opponent_policy = self.opponent_pool[-1]
        self.opponent_policy_callable = None
        self.opponent_snapshot_iteration = self.opponent_pool_iterations[-1]
        self._sample_opponent_assignments()

    def update_training_curriculum(self):
        from quadruped.envs.soccer_curriculum import get_curriculum
        curriculum = get_curriculum(self.env.env)
        if curriculum is None:
            return {}
        curriculum.advance()
        return curriculum.metrics()

    def opponent_update_ready(self):
        from quadruped.envs.soccer_curriculum import get_curriculum
        curriculum = get_curriculum(self.env.env)
        return curriculum is None or curriculum.opponent_ready()

    def opponent_pool_state_dict(self):
        if not self.opponent_pool:
            return None
        from quadruped.envs.soccer_curriculum import get_curriculum
        curriculum = get_curriculum(self.env.env)
        return {
            "soccer_curriculum": curriculum.state_dict() if curriculum is not None else None,
            "format_version": 1,
            "iterations": list(self.opponent_pool_iterations),
            "policies": [policy.state_dict() for policy in self.opponent_pool],
        }

    def opponent_policy_state_dict(self):
        if not self.opponent_pool:
            return None
        return self.opponent_pool[-1].state_dict()

    def configure_opponent_layout(self, action_encoding, history_length):
        """Keep native opponent observations and actions for mixed-policy evaluation."""
        if action_encoding not in ("hybrid", "discrete_skill_direction"):
            raise ValueError(f"Unsupported opponent encoding: {action_encoding}")
        self.opponent_action_encoding = action_encoding
        self.opponent_num_actions = 12 if action_encoding == "discrete_skill_direction" else 6
        self.opponent_num_obs = self.LOCAL_OBS_DIM + int(self.shooting_options) + int(self.opponent_num_actions == 12)
        self._opponent_history = torch.zeros(
            self.match_count, self.team_size, self.opponent_num_obs * history_length,
            device=self.device,
        )

    def set_opponent_callable(self, policy):
        """Install an exported deterministic policy for evaluation."""
        self._cached_opponent_actions = None
        self.opponent_policy = None
        self.opponent_policy_callable = policy
        self.opponent_action_provider = None

    def set_opponent_action_provider(self, provider):
        """Install a callable returning executable opponent actions."""

        self._cached_opponent_actions = None
        self.opponent_policy = None
        self.opponent_policy_callable = None
        self.opponent_action_provider = provider

    def _roots(self):
        raw = self.env.env
        return raw.root_states[raw.robot_actor_idxs_all.reshape(-1)].view(
            self.match_count, 2 * self.team_size, 13
        )

    def _nearest_relative(self, positions, own_slot, candidates, sign, scale):
        if not candidates:
            zeros = positions.new_zeros((self.match_count, 2))
            return zeros, positions.new_zeros((self.match_count, 1)), None
        relative = positions[:, candidates] - positions[:, own_slot : own_slot + 1]
        distance = torch.norm(relative, dim=-1)
        nearest = distance.argmin(dim=1)
        rows = torch.arange(self.match_count, device=self.device)
        selected = relative[rows, nearest]
        return sign * selected / scale, positions.new_ones((self.match_count, 1)), nearest

    def _team_observations(self, team):
        encoding = getattr(self, "opponent_action_encoding", self.action_encoding) if team == 1 else self.action_encoding
        obs_dim = getattr(self, "opponent_num_obs", self.num_obs) if team == 1 else self.num_obs
        raw = self.env.env
        roots = self._roots()
        positions = roots[:, :, :2]
        velocities = roots[:, :, 7:9]
        sign_value = 1.0 if team == 0 else -1.0
        sign = positions.new_tensor(sign_value)
        offset = team * self.team_size
        opponent_offset = (1 - team) * self.team_size
        half_length = max(0.5 * float(getattr(raw.cfg.env, "field_length", 8.0)), 1e-6)
        half_width = max(0.5 * float(getattr(raw.cfg.env, "field_width", 5.0)), 1e-6)
        field_scale = positions.new_tensor([half_length, half_width])
        ball_xy = raw.object_pos_world_frame[:, :2]
        ball_vel = raw.object_lin_vel[:, :2]
        forward_seed = raw.forward_vec[:, None, :].expand(-1, 2 * self.team_size, -1)
        forward = quat_apply(
            roots[:, :, 3:7].reshape(-1, 4), forward_seed.reshape(-1, 3)
        ).view(self.match_count, 2 * self.team_size, 3)
        affordance_data = self.env._skill_affordances(roots)
        affordances = affordance_data["features"]
        attacker_mask = self.env._attacker_mask(affordance_data)
        command_scale = self.env._command_obs_scale()
        goal_x = float(getattr(raw.cfg.env, "team_goal_x", half_length))

        observations = []
        for local_slot in range(self.team_size):
            slot = offset + local_slot
            teammate_slots = [offset + i for i in range(self.team_size) if i != local_slot]
            opponent_slots = [opponent_offset + i for i in range(self.team_size)]
            teammate_rel, _, _ = self._nearest_relative(
                positions, slot, teammate_slots, sign, field_scale
            )
            opponent_rel, opponent_mask, nearest_opponent = self._nearest_relative(
                positions, slot, opponent_slots, sign, field_scale
            )
            rows = torch.arange(self.match_count, device=self.device)
            opponent_vel = velocities.new_zeros((self.match_count, 2))
            if nearest_opponent is not None:
                opponent_vel = sign * velocities[rows, opponent_offset + nearest_opponent] / 3.0

            own_xy = sign * (positions[:, slot] - raw.env_origins[:, :2]) / field_scale
            own_forward = sign * forward[:, slot, :2]
            own_vel = sign * velocities[:, slot] / 3.0
            ball_rel = sign * (ball_xy - positions[:, slot]) / field_scale
            canonical_ball_vel = sign * ball_vel / 5.0
            ball_distance = torch.norm(ball_xy - positions[:, slot], dim=-1, keepdim=True)
            ball_distance = ball_distance / math.hypot(half_length, half_width)
            canonical_goal = raw.env_origins[:, :2].clone()
            canonical_goal[:, 0] += sign_value * goal_x
            goal_rel = sign * (canonical_goal - positions[:, slot]) / field_scale
            own_affordance = affordances[:, slot].clone()
            robot_to_ball = ball_xy - positions[:, slot]
            ball_to_goal = canonical_goal - ball_xy
            own_affordance[:, -1] = torch.sum(
                robot_to_ball / torch.norm(robot_to_ball, dim=-1, keepdim=True).clamp(min=1e-6)
                * ball_to_goal / torch.norm(ball_to_goal, dim=-1, keepdim=True).clamp(min=1e-6),
                dim=-1,
            ).clamp(min=-1.0, max=1.0)
            skill_count = 4 if encoding == "discrete_skill_direction" else 3
            skill_one_hot = torch.nn.functional.one_hot(
                self.env.skill_ids[:, slot].clamp(min=0, max=skill_count - 1),
                num_classes=skill_count,
            ).float()
            command = self.env.skill_commands[:, slot]
            if team == 1:
                # Ball-skill xy commands are stored in the fixed world frame,
                # unlike body-frame walk commands. Rotate them back into the
                # opponent policy's canonical attack-positive frame.
                command = mirror_high_level_commands(
                    command, self.env.skill_ids[:, slot]
                )
            command = command / command_scale
            # This replaces the old teammate-presence bit, which was a
            # constant one in every multi-robot training sample.  The shared
            # actor now receives an explicit role bit without changing the
            # 34-value observation/checkpoint shape: closest robot attacks,
            # the other robot spreads into support.
            attacker_role = attacker_mask[:, slot : slot + 1].float()
            obs = torch.cat(
                (
                    own_xy,
                    own_forward,
                    own_vel,
                    roots[:, slot, 12:13] / 3.0,
                    ball_rel,
                    canonical_ball_vel,
                    ball_distance,
                    goal_rel,
                    teammate_rel,
                    attacker_role,
                    opponent_rel,
                    opponent_vel,
                    opponent_mask,
                    own_affordance,
                    skill_one_hot,
                    command,
                ),
                dim=-1,
            )
            if getattr(self, "shooting_options", False):
                obs = torch.cat((obs, self.env.shoot_option_remaining[:, slot:slot+1] / 2.4), -1)
            if obs.shape[1] != obs_dim:
                raise RuntimeError(f"Local observation has {obs.shape[1]} values, expected {obs_dim}")
            observations.append(obs)
        return torch.stack(observations, dim=1)

    def _update_observations(self, reset_mask=None):
        for team in range(2):
            obs = torch.nan_to_num(self._team_observations(team), nan=0.0, posinf=100.0, neginf=-100.0)
            history = self._opponent_history if team == 1 and hasattr(self, "_opponent_history") else self._history[:, team]
            history[:] = torch.cat((history[:, :, obs.shape[-1] :], obs), dim=-1)
            if reset_mask is not None and bool(torch.any(reset_mask)):
                history[reset_mask] = 0.0
        team_obs = self._team_observations(0).reshape(self.num_envs, self.num_obs)
        team_history = self._history[:, 0].reshape(self.num_envs, self.num_obs_history)
        privileged = team_obs
        if self.centralized_critic:
            global_obs = torch.cat([self._team_observations(t).flatten(1) for t in range(2)], dim=-1)
            privileged = global_obs.repeat_interleave(self.team_size, dim=0)
        self._cached = {
            "obs": team_obs,
            "privileged_obs": privileged,
            "obs_history": team_history,
        }

    def get_observations(self):
        return self._cached

    def reset(self):
        self.env.reset()
        self._cached_opponent_actions = None
        self.env.preserve_external_high_level_actions = torch.zeros(
            self.match_count,
            2 * self.team_size,
            dtype=torch.bool,
            device=self.device,
        )
        self._history.zero_()
        if hasattr(self, "_opponent_history"):
            self._opponent_history.zero_()
        self._update_observations()
        return self._cached

    def _opponent_actions(self, *, cache=False):
        if self.opponent_action_provider is not None:
            actions = self.opponent_action_provider().to(self.device)
            expected = (self.match_count, self.team_size, self.num_actions)
            if tuple(actions.shape) != expected:
                raise ValueError(
                    "Opponent action provider must return executable actions "
                    f"with shape {expected}, got {tuple(actions.shape)}"
                )
            # Action providers (for example a deterministic rules controller
            # or an MPC forecaster) operate in physical world semantics and
            # therefore already account for the opponent's -x attack
            # direction. Learned opponent policies below emit canonical +x
            # actions and are mirrored at the end of this method.
            if cache:
                self._cached_opponent_actions = actions.detach().clone()
            return actions
        if not self.opponent_pool and self.opponent_policy_callable is None:
            actions = torch.zeros(
                self.match_count, self.team_size, self.num_actions, device=self.device
            )
            if cache:
                self._cached_opponent_actions = actions
            return actions
        history = getattr(self, "_opponent_history", self._history[:, 1]).reshape(self.num_envs, -1)
        with torch.inference_mode():
            if self.opponent_policy_callable is not None:
                opponent_obs = {
                    "obs": self._team_observations(1).reshape(self.num_envs, -1),
                    "privileged_obs": self._team_observations(1).reshape(self.num_envs, -1),
                    "obs_history": history,
                }
                actions = self.opponent_policy_callable(opponent_obs)
                actions = actions.to(self.device).view(
                    self.match_count, self.team_size, getattr(self, "opponent_num_actions", self.num_actions)
                )
            else:
                grouped_history = history.view(
                    self.match_count, self.team_size, self.num_obs_history
                )
                actions = torch.zeros(
                    self.match_count,
                    self.team_size,
                    self.num_actions,
                    dtype=history.dtype,
                    device=self.device,
                )
                for pool_idx, policy in enumerate(self.opponent_pool):
                    match_mask = self.opponent_assignment == pool_idx
                    if not bool(torch.any(match_mask)):
                        continue
                    selected_history = grouped_history[match_mask].reshape(
                        -1, self.num_obs_history
                    ).to(self.opponent_device)
                    selected_actions = policy.act_training(selected_history).view(
                        -1, self.team_size, self.num_actions
                    )
                    actions[match_mask] = selected_actions.to(self.device)
        # The opponent observes a world rotated by pi so that its -x target is
        # canonical +x. Walk commands are body-relative and need no change, but
        # field-frame dribble/shoot commands must be rotated back before the
        # shared skill wrapper executes them in the real world.
        actions = mirror_high_level_policy_actions(actions)
        if cache:
            self._cached_opponent_actions = actions.detach().clone()
        return actions

    def _local_role_rewards(self, info):
        """Return per-learning-robot role shaping for the shared actor.

        The raw high-level environment emits one cooperative match reward.
        PPO then repeats that reward for both robots, which gives a passive
        near-ball robot credit for a teammate's dribble. These dense terms are
        deliberately local: only the assigned attacker is rewarded for an
        active ball skill, and only the support robot is charged for crowding
        or taking an attacker-only skill.
        """

        raw = self.env
        attacker = torch.as_tensor(
            info.get("high_level_attacker_mask", torch.zeros(
                self.match_count, 2 * self.team_size, dtype=torch.bool,
                device=self.device,
            )),
            dtype=torch.bool,
            device=self.device,
        )[:, : self.team_size]
        role_conflict = torch.as_tensor(
            info.get("high_level_role_conflict_mask", torch.zeros_like(attacker)),
            dtype=torch.bool,
            device=self.device,
        )[:, : self.team_size]
        assist = torch.as_tensor(
            info.get("high_level_attacker_command_assist_mask", torch.zeros_like(attacker)),
            dtype=torch.bool,
            device=self.device,
        )[:, : self.team_size]
        executed = torch.as_tensor(
            info.get("high_level_skill_ids", torch.zeros(
                self.match_count, 2 * self.team_size, dtype=torch.long,
                device=self.device,
            )),
            dtype=torch.long,
            device=self.device,
        )[:, : self.team_size]
        requested = torch.as_tensor(
            info.get("high_level_requested_skill_ids", torch.zeros_like(executed)),
            dtype=torch.long,
            device=self.device,
        )[:, : self.team_size]
        invalid = torch.as_tensor(
            info.get("high_level_invalid_skill_mask", torch.zeros_like(attacker)),
            dtype=torch.bool,
            device=self.device,
        )[:, : self.team_size]
        commands = torch.as_tensor(
            info.get("high_level_commands", torch.zeros(
                self.match_count, 2 * self.team_size, 3,
                dtype=torch.float, device=self.device,
            )),
            dtype=torch.float,
            device=self.device,
        )[:, : self.team_size, :2]

        previous_distances = getattr(
            raw.env,
            "prev_high_level_robot_ball_distances",
            None,
        )
        decision_distances = info.get("high_level_decision_robot_ball_distances")
        if decision_distances is not None:
            previous_distances = torch.as_tensor(
                decision_distances, dtype=torch.float, device=self.device
            )
        if previous_distances is None:
            previous_distances = raw.env._high_level_robot_ball_distances()
        previous_distances = previous_distances[:, : self.team_size]
        roots = raw.env.root_states[
            raw.env.robot_actor_idxs_all[:, : self.team_size].reshape(-1)
        ].view(self.match_count, self.team_size, 13)
        current_distances = torch.norm(
            roots[:, :, :2] - raw.env.object_pos_world_frame[:, None, :2], dim=-1
        )
        dribble_distance = float(
            getattr(raw.cfg.rewards, "high_level_dribble_skill_distance", 1.0)
        )
        attacker_ready = attacker & (previous_distances <= dribble_distance)
        valid_ball_skill = (
            ((executed == 1) | (executed == 2))
            & (requested == executed)
            & ~invalid
        )
        target_speed = max(
            float(getattr(raw.cfg.rewards, "high_level_skill_command_target_speed", 0.8)),
            1e-6,
        )
        command_fraction = torch.clamp(
            torch.norm(commands, dim=-1) / target_speed, min=0.0, max=1.0
        )
        elapsed_steps = torch.as_tensor(
            info.get(
                "elapsed_low_level_steps",
                torch.full(
                    (self.match_count,),
                    int(raw.control_interval),
                    dtype=torch.long,
                    device=self.device,
                ),
            ),
            dtype=torch.float,
            device=self.device,
        ).reshape(self.match_count, 1)
        elapsed_time = (elapsed_steps * float(raw.env.dt)).clamp(min=1e-6)
        closing_speed = (previous_distances - current_distances) / elapsed_time
        closing_fraction = torch.clamp(
            closing_speed / 0.5, min=0.0, max=1.0
        )
        target_ball_speed = max(
            float(getattr(raw.cfg.rewards, "high_level_dribble_target_ball_speed", 1.0)),
            1e-6,
        )
        command_direction = commands / torch.norm(
            commands, dim=-1, keepdim=True
        ).clamp(min=1e-6)
        command_aligned_ball_speed = torch.sum(
            raw.env.object_lin_vel[:, None, :2] * command_direction, dim=-1
        )
        ball_speed_fraction = torch.clamp(
            command_aligned_ball_speed / target_ball_speed,
            min=0.0,
            max=1.0,
        )
        physical_consequence = torch.maximum(
            closing_fraction, ball_speed_fraction
        )
        active_score = 2.0 * command_fraction * physical_consequence - 1.0
        attacker_score = torch.where(
            valid_ball_skill, active_score, -torch.ones_like(active_score)
        )
        attacker_scale = float(
            getattr(raw.cfg.rewards, "high_level_local_attacker_ball_skill_scale", 0.0)
        )
        assist_scale = float(
            getattr(raw.cfg.rewards, "high_level_local_attacker_command_assist_scale", 0.0)
        )
        conflict_scale = float(
            getattr(raw.cfg.rewards, "high_level_local_role_conflict_scale", 0.0)
        )
        crowding_scale = float(
            getattr(raw.cfg.rewards, "high_level_local_support_ball_crowding_scale", 0.0)
        )
        attacker_ball_skill_reward = (
            torch.where(attacker_ready, attacker_score, torch.zeros_like(attacker_score))
            * attacker.float()
            * attacker_scale
        )
        attacker_command_assist_reward = assist.float() * assist_scale
        role_conflict_reward = role_conflict.float() * conflict_scale
        local = (
            attacker_ball_skill_reward
            + attacker_command_assist_reward
            + role_conflict_reward
        )

        support_min = max(
            float(getattr(raw.cfg.rewards, "high_level_support_min_ball_distance", 1.15)),
            1e-6,
        )
        support = ~attacker
        crowding = torch.clamp(
            (support_min - current_distances) / support_min, min=0.0, max=1.0
        ).square()
        support_ball_crowding_reward = support.float() * crowding * crowding_scale
        local = local + support_ball_crowding_reward
        attacker_ball_skill_reward = attacker_ball_skill_reward * elapsed_time
        attacker_command_assist_reward = attacker_command_assist_reward * elapsed_time
        role_conflict_reward = role_conflict_reward * elapsed_time
        support_ball_crowding_reward = support_ball_crowding_reward * elapsed_time
        local = local * elapsed_time
        self._last_local_role_reward_components = {
            "attacker_ball_skill": torch.nan_to_num(attacker_ball_skill_reward),
            "attacker_command_assist": torch.nan_to_num(attacker_command_assist_reward),
            "role_conflict": torch.nan_to_num(role_conflict_reward),
            "support_ball_crowding": torch.nan_to_num(support_ball_crowding_reward),
        }
        return torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)

    def preview_opponent_actions(self):
        """Return the sampled action that the next step will execute."""

        if getattr(self, "_cached_opponent_actions", None) is None:
            return self._opponent_actions(cache=True)
        return self._cached_opponent_actions

    def _consume_opponent_actions(self):
        if getattr(self, "_cached_opponent_actions", None) is None:
            return self._opponent_actions()
        actions = self._cached_opponent_actions
        self._cached_opponent_actions = None
        return actions

    def step(self, actions):
        team_actions = actions.to(self.device).view(
            self.match_count, self.team_size, self.num_actions
        )
        used_opponent_assignment = self.opponent_assignment.clone()
        opponent_actions = self._consume_opponent_actions()
        if team_actions.shape[-1] != opponent_actions.shape[-1]:
            joint_actions = (team_actions, opponent_actions)
        else:
            joint_actions = torch.cat((team_actions, opponent_actions), dim=1).reshape(self.match_count, -1)
        # A direct provider has already made its own role and safety choices.
        # Preserve those opponent commands while retaining the normal
        # geometric fallback for learned learning-team actions.
        preserve_external = torch.zeros(
            self.match_count,
            2 * self.team_size,
            dtype=torch.bool,
            device=self.device,
        )
        if self.opponent_action_provider is not None and bool(
            getattr(
                self.opponent_action_provider,
                "preserve_high_level_actions",
                False,
            )
        ):
            preserve_external[:, self.team_size :] = True
        self.env.preserve_external_high_level_actions = preserve_external
        _, rewards, dones, info = self.env.step(joint_actions)
        dones = dones.bool()
        from quadruped.envs.soccer_curriculum import get_curriculum
        curriculum = get_curriculum(self.env.env)
        if curriculum is not None and "soccer_start_kind" in info:
            terminal = dones.nonzero(as_tuple=False).flatten()
            def completed(value):
                return torch.as_tensor(value, device=self.device)[terminal].cpu().tolist()
            curriculum.record(
                completed(info["soccer_start_kind"]), completed(info["soccer_start_stage"]),
                completed(info["high_level_goal"]),
                completed(info["high_level_learning_team_failure"]),
                anchor=completed(used_opponent_assignment == 0),
                concessions=completed(info["high_level_opponent_goal"]),
                timeouts=completed(info["time_outs"]),
            )
        self._sample_opponent_assignments(dones)
        self._update_observations(reset_mask=dones)
        agent_rewards = rewards.repeat_interleave(self.team_size)
        local_role_rewards = self._local_role_rewards(info)
        # The raw simulator auto-resets terminal matches before returning, so
        # their root states no longer correspond to the recorded decision.
        # Terminal goals/penalties remain in the shared reward; omit only this
        # state-difference shaping on those rows.
        local_role_rewards[dones] = 0.0
        local_components = getattr(self, "_last_local_role_reward_components", {})
        for component in local_components.values():
            component[dones] = 0.0
        agent_rewards = agent_rewards + local_role_rewards.reshape(-1)
        agent_dones = dones.repeat_interleave(self.team_size)
        info = dict(info)
        # Preserve the cooperative match reward before adding decentralized
        # role shaping. Online dynamics/value learning consumes this reward,
        # while PPO continues to receive the per-agent shaped reward below.
        info["high_level_match_rewards"] = rewards.detach()
        info["high_level_local_role_rewards"] = local_role_rewards.detach().cpu().numpy()
        for name, component in local_components.items():
            info[f"high_level_local_{name}_rewards"] = component.detach().cpu().numpy()
        for key in ("env_bins", "time_outs"):
            if key in info:
                value = torch.as_tensor(info[key], device=self.device)
                info[key] = value.repeat_interleave(self.team_size, dim=0)
        info["opponent_snapshot_iteration"] = self.opponent_snapshot_iteration
        info["opponent_pool_size"] = len(self.opponent_pool)
        if self.opponent_pool_iterations:
            selected_iterations = torch.as_tensor(
                self.opponent_pool_iterations, device=self.device, dtype=torch.long
            )[used_opponent_assignment]
            info["opponent_pool_selected_iterations"] = selected_iterations.detach().cpu().numpy()
        return self._cached, agent_rewards, agent_dones, info
