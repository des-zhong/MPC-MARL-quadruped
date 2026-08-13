import copy
import math

import gym
import torch
import torch.nn as nn
from isaacgym.torch_utils import quat_apply


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
    """Expose one shared-policy sample per robot and drive a frozen opponent.

    The wrapped high-level environment owns ``2 * team_size`` physical AS2
    actors. Slots ``[0, team_size)`` are the learning team and the remaining
    slots are the opponent team.  PPO sees ``match_count * team_size`` agents,
    each with the same fixed-size, agent-centric observation and six actions.
    """

    LOCAL_OBS_DIM = 34

    def __init__(self, env, team_size, opponent_device=None):
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
        self.opponent_snapshot_iteration = -1

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

    def update_opponent_policy(self, actor_critic, iteration=0):
        """Atomically replace the opponent with a detached policy snapshot."""
        snapshot = FrozenOpponentPolicy(actor_critic).to(self.opponent_device)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)
        self.opponent_policy = snapshot
        self.opponent_policy_callable = None
        self.opponent_snapshot_iteration = int(iteration)

    def load_opponent_policy_state_dict(self, state_dict, actor_critic, iteration=-1):
        self.update_opponent_policy(actor_critic, iteration=iteration)
        incompatible = self.opponent_policy.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise ValueError(
                "Opponent checkpoint is missing inference weights: "
                f"{incompatible.missing_keys}"
            )

    def opponent_policy_state_dict(self):
        if self.opponent_policy is None:
            return None
        return self.opponent_policy.state_dict()

    def set_opponent_callable(self, policy):
        """Install an exported deterministic policy for evaluation."""
        self.opponent_policy = None
        self.opponent_policy_callable = policy

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
        affordances = self.env._skill_affordances(roots)["features"]
        command_scale = self.env._command_obs_scale()
        goal_x = float(getattr(raw.cfg.env, "team_goal_x", half_length))

        observations = []
        for local_slot in range(self.team_size):
            slot = offset + local_slot
            teammate_slots = [offset + i for i in range(self.team_size) if i != local_slot]
            opponent_slots = [opponent_offset + i for i in range(self.team_size)]
            teammate_rel, teammate_mask, _ = self._nearest_relative(
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
            command = self.env.skill_commands[:, slot] / command_scale
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
                    teammate_mask,
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
        if self.opponent_policy is None and self.opponent_policy_callable is None:
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
            else:
                actions = self.opponent_policy.act_student(history.to(self.opponent_device))
        return actions.to(self.device).view(self.match_count, self.team_size, self.num_actions)

    def preview_opponent_actions(self):
        """Return the deterministic frozen-policy action without stepping the env."""

        return self._opponent_actions()

    def step(self, actions):
        team_actions = actions.to(self.device).view(
            self.match_count, self.team_size, self.num_actions
        )
        opponent_actions = self._opponent_actions()
        joint_actions = torch.cat((team_actions, opponent_actions), dim=1).reshape(
            self.match_count, -1
        )
        _, rewards, dones, info = self.env.step(joint_actions)
        dones = dones.bool()
        self._update_observations(reset_mask=dones)
        agent_rewards = rewards.repeat_interleave(self.team_size)
        agent_dones = dones.repeat_interleave(self.team_size)
        info = dict(info)
        for key in ("env_bins", "time_outs"):
            if key in info:
                value = torch.as_tensor(info[key], device=self.device)
                info[key] = value.repeat_interleave(self.team_size, dim=0)
        info["opponent_snapshot_iteration"] = self.opponent_snapshot_iteration
        return self._cached, agent_rewards, agent_dones, info
