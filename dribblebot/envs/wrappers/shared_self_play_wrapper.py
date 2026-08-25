import copy
import math

import gym
import torch
import torch.nn as nn
from isaacgym.torch_utils import quat_apply

from .team_frame import (
    mirror_high_level_commands,
    mirror_high_level_policy_actions,
)


class FrozenOpponentPolicy(nn.Module):
    """Inference-only copy of the trainable actor.

    ``ActorCritic`` caches a ``Normal`` distribution whose mean is produced by
    the latest forward pass.  Those cached non-leaf tensors make the complete
    training module unsafe to deepcopy.  Self-play only needs the adaptation
    and actor networks, so snapshot exactly those stateful components.
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.adaptation_module = copy.deepcopy(actor_critic.adaptation_module)
        self.actor_body = copy.deepcopy(actor_critic.actor_body)

    def act_student(self, observation_history):
        latent = self.adaptation_module(observation_history)
        return self.actor_body(torch.cat((observation_history, latent), dim=-1))


class SharedPolicySelfPlayWrapper(gym.Wrapper):
    """Expose one shared-policy sample per robot and drive an opponent.

    The wrapped high-level environment owns ``2 * team_size`` physical AS2
    actors. Slots ``[0, team_size)`` are the learning team and the remaining
    slots are the opponent team.  PPO sees ``match_count * team_size`` agents,
    each with the same fixed-size, agent-centric observation and six actions.
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
        self.num_actions = 6
        self.num_obs = self.LOCAL_OBS_DIM
        self.num_privileged_obs = self.num_obs
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
        if not had_pool:
            self._sample_opponent_assignments()
        elif evicted_assignment_mask is not None:
            self._sample_opponent_assignments(evicted_assignment_mask)

    def load_opponent_policy_state_dict(self, state_dict, actor_critic, iteration=-1):
        if self.opponent_action_provider is not None:
            self.opponent_snapshot_iteration = int(iteration)
            return None
        snapshot = self._frozen_snapshot(actor_critic)
        incompatible = snapshot.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise ValueError(
                "Opponent checkpoint is missing inference weights: "
                f"{incompatible.missing_keys}"
            )
        self.opponent_pool = [snapshot]
        self.opponent_pool_iterations = [int(iteration)]
        self.opponent_policy = snapshot
        self.opponent_policy_callable = None
        self.opponent_snapshot_iteration = int(iteration)
        self._sample_opponent_assignments()

    def load_opponent_pool_state_dict(self, payload, actor_critic):
        if not isinstance(payload, dict) or not payload.get("policies"):
            raise ValueError("Opponent-pool checkpoint contains no policies")
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
            if incompatible.missing_keys:
                raise ValueError(
                    "Opponent-pool checkpoint is missing inference weights: "
                    f"{incompatible.missing_keys}"
                )
            self.opponent_pool.append(snapshot)
            self.opponent_pool_iterations.append(int(iteration))
        self.opponent_policy = self.opponent_pool[-1]
        self.opponent_policy_callable = None
        self.opponent_snapshot_iteration = self.opponent_pool_iterations[-1]
        self._sample_opponent_assignments()

    def opponent_pool_state_dict(self):
        if not self.opponent_pool:
            return None
        return {
            "format_version": 1,
            "iterations": list(self.opponent_pool_iterations),
            "policies": [policy.state_dict() for policy in self.opponent_pool],
        }

    def opponent_policy_state_dict(self):
        if not self.opponent_pool:
            return None
        return self.opponent_pool[-1].state_dict()

    def set_opponent_callable(self, policy):
        """Install an exported deterministic policy for evaluation."""
        self.opponent_policy = None
        self.opponent_policy_callable = policy
        self.opponent_action_provider = None

    def set_opponent_action_provider(self, provider):
        """Install a callable returning executable opponent actions."""

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
            skill_one_hot = torch.nn.functional.one_hot(
                self.env.skill_ids[:, slot], num_classes=3
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
            if obs.shape[1] != self.num_obs:
                raise RuntimeError(f"Local observation has {obs.shape[1]} values, expected {self.num_obs}")
            observations.append(obs)
        return torch.stack(observations, dim=1)

    def _update_observations(self, reset_mask=None):
        for team in range(2):
            obs = torch.nan_to_num(self._team_observations(team), nan=0.0, posinf=100.0, neginf=-100.0)
            history = self._history[:, team]
            history[:] = torch.cat((history[:, :, self.num_obs :], obs), dim=-1)
            if reset_mask is not None and bool(torch.any(reset_mask)):
                history[reset_mask] = 0.0
        team_obs = self._team_observations(0).reshape(self.num_envs, self.num_obs)
        team_history = self._history[:, 0].reshape(self.num_envs, self.num_obs_history)
        self._cached = {
            "obs": team_obs,
            "privileged_obs": team_obs,
            "obs_history": team_history,
        }

    def get_observations(self):
        return self._cached

    def reset(self):
        self.env.reset()
        self._history.zero_()
        self._update_observations()
        return self._cached

    def _opponent_actions(self):
        if self.opponent_action_provider is not None:
            actions = self.opponent_action_provider().to(self.device)
            expected = (self.match_count, self.team_size, self.num_actions)
            if tuple(actions.shape) != expected:
                raise ValueError(
                    "Opponent action provider must return executable actions "
                    f"with shape {expected}, got {tuple(actions.shape)}"
                )
            return actions
        if not self.opponent_pool and self.opponent_policy_callable is None:
            return torch.zeros(
                self.match_count, self.team_size, self.num_actions, device=self.device
            )
        history = self._history[:, 1].reshape(self.num_envs, self.num_obs_history)
        with torch.inference_mode():
            if self.opponent_policy_callable is not None:
                opponent_obs = {
                    "obs": self._team_observations(1).reshape(self.num_envs, self.num_obs),
                    "privileged_obs": self._team_observations(1).reshape(self.num_envs, self.num_obs),
                    "obs_history": history,
                }
                actions = self.opponent_policy_callable(opponent_obs)
                actions = actions.to(self.device).view(
                    self.match_count, self.team_size, self.num_actions
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
                    selected_actions = policy.act_student(selected_history).view(
                        -1, self.team_size, self.num_actions
                    )
                    actions[match_mask] = selected_actions.to(self.device)
        # The opponent observes a world rotated by pi so that its -x target is
        # canonical +x. Walk commands are body-relative and need no change, but
        # field-frame dribble/shoot commands must be rotated back before the
        # shared skill wrapper executes them in the real world.
        return mirror_high_level_policy_actions(actions)

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
            getattr(raw.cfg.rewards, "high_level_local_attacker_ball_skill_scale", 2.0)
        )
        assist_scale = float(
            getattr(raw.cfg.rewards, "high_level_local_attacker_command_assist_scale", -4.0)
        )
        conflict_scale = float(
            getattr(raw.cfg.rewards, "high_level_local_role_conflict_scale", -3.0)
        )
        crowding_scale = float(
            getattr(raw.cfg.rewards, "high_level_local_support_ball_crowding_scale", -3.0)
        )
        local = torch.where(attacker_ready, attacker_score, torch.zeros_like(attacker_score))
        local = local * attacker.float() * attacker_scale
        local = local + assist.float() * assist_scale
        local = local + role_conflict.float() * conflict_scale

        support_min = max(
            float(getattr(raw.cfg.rewards, "high_level_support_min_ball_distance", 1.15)),
            1e-6,
        )
        support = ~attacker
        crowding = torch.clamp(
            (support_min - current_distances) / support_min, min=0.0, max=1.0
        ).square()
        local = local + support.float() * crowding * crowding_scale
        local = local * elapsed_time
        return torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)

    def preview_opponent_actions(self):
        """Return deterministic opponent actions in executable world semantics."""

        return self._opponent_actions()

    def step(self, actions):
        team_actions = actions.to(self.device).view(
            self.match_count, self.team_size, self.num_actions
        )
        used_opponent_assignment = self.opponent_assignment.clone()
        opponent_actions = self._opponent_actions()
        joint_actions = torch.cat((team_actions, opponent_actions), dim=1).reshape(
            self.match_count, -1
        )
        _, rewards, dones, info = self.env.step(joint_actions)
        dones = dones.bool()
        self._sample_opponent_assignments(dones)
        self._update_observations(reset_mask=dones)
        agent_rewards = rewards.repeat_interleave(self.team_size)
        local_role_rewards = self._local_role_rewards(info)
        # The raw simulator auto-resets terminal matches before returning, so
        # their root states no longer correspond to the recorded decision.
        # Terminal goals/penalties remain in the shared reward; omit only this
        # state-difference shaping on those rows.
        local_role_rewards[dones] = 0.0
        agent_rewards = agent_rewards + local_role_rewards.reshape(-1)
        agent_dones = dones.repeat_interleave(self.team_size)
        info = dict(info)
        info["high_level_local_role_rewards"] = local_role_rewards.detach().cpu().numpy()
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
