"""Thin self-play adapter for the four-robot manager-based match scene."""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

from .macro import MacroActionWrapper
from .team_frame import legacy_policy_action_to_hybrid, mirror_high_level_policy_actions


class MatchSelfPlayWrapper(gym.Wrapper):
    """Expose one shared-policy sample per learning-team robot.

    Each sample uses the hybrid action ``[skill_index, parameter_x,
    parameter_y, parameter_yaw]``. The first value is integer-valued after
    sampling; the remaining values are continuous pre-``tanh`` parameters.
    """

    num_obs = 34
    num_actions = 4

    def __init__(
        self,
        env: MacroActionWrapper,
        team_size: int = 2,
        history_length: int = 4,
        opponent_snapshot_interval: int = 500,
        opponent_pool_size: int = 8,
        opponent_latest_probability: float = 0.5,
    ):
        super().__init__(env)
        if team_size <= 0 or history_length <= 0:
            raise ValueError("team_size and history_length must be positive")
        self.env = env
        self.match_count = int(env.env.num_envs)
        self.team_size = int(team_size)
        self.num_envs = self.match_count * self.team_size
        self.num_train_envs = self.num_envs
        self.num_robots = self.team_size
        self.max_episode_length = int(env.env.max_episode_length // max(env.control_interval, 1))
        self.history_length = int(history_length)
        self.num_obs_history = self.num_obs * self.history_length
        self.device = env.device
        if int(opponent_snapshot_interval) <= 0:
            raise ValueError("opponent_snapshot_interval must be positive")
        self.opponent_snapshot_interval = int(opponent_snapshot_interval)
        self.opponent_pool_size = int(opponent_pool_size)
        self.opponent_latest_probability = float(opponent_latest_probability)
        if self.opponent_pool_size < 1:
            raise ValueError("opponent_pool_size must be at least 1")
        if not 0.0 <= self.opponent_latest_probability <= 1.0:
            raise ValueError("opponent_latest_probability must be in [0, 1]")
        action_low = np.tile(np.asarray([0.0, -np.inf, -np.inf, -np.inf], dtype=np.float32), (self.num_envs, 1))
        action_high = np.tile(np.asarray([2.0, np.inf, np.inf, np.inf], dtype=np.float32), (self.num_envs, 1))
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(-float("inf"), float("inf"), (self.num_envs, self.num_obs), dtype=float),
                "privileged_obs": gym.spaces.Box(
                    -float("inf"), float("inf"), (self.num_envs, self.num_obs), dtype=float
                ),
                "obs_history": gym.spaces.Box(
                    -float("inf"),
                    float("inf"),
                    (self.num_envs, self.num_obs_history),
                    dtype=float,
                ),
            }
        )
        self.opponent_snapshot_iteration = -1
        self.opponent_policy_callable: Callable[[dict[str, torch.Tensor]], torch.Tensor] | None = None
        self.opponent_action_provider: Callable[[], torch.Tensor] | None = None
        self.opponent_pool: list[Callable[[dict[str, torch.Tensor]], torch.Tensor]] = []
        self.opponent_pool_iterations: list[int] = []
        self.opponent_assignment = torch.zeros(self.match_count, dtype=torch.long, device=self.device)
        self._history = torch.zeros(
            self.match_count,
            2,
            self.team_size,
            self.num_obs_history,
            device=self.device,
        )
        self._cached: dict[str, torch.Tensor] | None = None
        self._last_team_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)

    @property
    def cfg(self):
        return self.env.env.cfg

    @property
    def episode_length_buf(self):
        return self.env.env.episode_length_buf.repeat_interleave(self.team_size)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        values = value.to(self.device).reshape(self.match_count, self.team_size)
        self.env.env.episode_length_buf[:] = values[:, 0]

    @property
    def actions(self) -> torch.Tensor:
        return self._last_team_actions

    def get_observations(self):
        return self._cached

    def set_opponent_callable(self, policy: Callable[[dict[str, torch.Tensor]], torch.Tensor]) -> None:
        if not callable(policy):
            raise TypeError("opponent policy must be callable")
        self.opponent_policy_callable = policy
        self.opponent_action_provider = None
        self.opponent_pool = [policy]
        self.opponent_pool_iterations = [self.opponent_snapshot_iteration]
        self._sample_opponent_assignments()

    def set_opponent_action_provider(self, provider: Callable[[], torch.Tensor]) -> None:
        """Install a provider returning executable world-semantic actions."""

        if not callable(provider):
            raise TypeError("opponent action provider must be callable")
        self.opponent_action_provider = provider
        self.opponent_policy_callable = None
        self.opponent_pool.clear()
        self.opponent_pool_iterations.clear()
        self.opponent_assignment.zero_()

    def load_opponent_checkpoint(self, root=None, device: str = "cpu", iteration: int = -1) -> None:
        """Install the archived TorchScript coordinator as an opponent snapshot."""

        from ....policies.high_level import FrozenCoordinatorPolicy

        self.opponent_policy_callable = FrozenCoordinatorPolicy.load(root=root, device=device)
        self.opponent_snapshot_iteration = int(iteration)
        self.opponent_pool = [self.opponent_policy_callable]
        self.opponent_pool_iterations = [int(iteration)]
        self._sample_opponent_assignments()

    def update_opponent_snapshot(
        self,
        policy: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        iteration: int,
        force: bool = False,
    ) -> bool:
        """Atomically replace the inference callable at the configured cadence."""

        iteration = int(iteration)
        if self.opponent_action_provider is not None:
            self.opponent_snapshot_iteration = iteration
            return False
        if not force and iteration % self.opponent_snapshot_interval != 0:
            return False
        had_pool = bool(self.opponent_pool)
        evicted_assignment_mask = None
        if len(self.opponent_pool) >= self.opponent_pool_size:
            # Keep the initial anchor and evict the oldest non-anchor policy.
            evict = 1 if self.opponent_pool_size > 1 else 0
            evicted_assignment_mask = self.opponent_assignment == evict
            self.opponent_pool.pop(evict)
            self.opponent_pool_iterations.pop(evict)
            self.opponent_assignment[self.opponent_assignment > evict] -= 1
        self.opponent_pool.append(policy)
        self.opponent_pool_iterations.append(iteration)
        self.opponent_policy_callable = policy
        self.opponent_snapshot_iteration = iteration
        if not had_pool:
            self._sample_opponent_assignments()
        elif evicted_assignment_mask is not None:
            self._sample_opponent_assignments(evicted_assignment_mask)
        return True

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._history.zero_()
        self._last_team_actions.zero_()
        self._sample_opponent_assignments()
        if self.opponent_action_provider is not None and hasattr(self.opponent_action_provider, "reset"):
            self.opponent_action_provider.reset()
        self._update_observations(observation, reset_mask=None)
        return self._cached, info

    def seed(self, seed: int = -1) -> int:
        raw = self.env.env
        if hasattr(raw, "seed"):
            result = raw.seed(seed)
            return int(seed if result is None else result)
        return int(seed)

    def step(self, actions: torch.Tensor):
        team_actions = actions.to(self.device).view(self.match_count, self.team_size, self.num_actions)
        self._last_team_actions[:] = team_actions.reshape(self.num_envs, self.num_actions)
        previous_distances = self._learning_team_ball_distances()
        opponent_actions = self._opponent_actions()
        joint_actions = torch.cat((team_actions, opponent_actions), dim=1).reshape(
            self.match_count,
            2 * self.team_size * self.num_actions,
        )
        preserve_provider = bool(
            self.opponent_action_provider is not None
            and getattr(self.opponent_action_provider, "preserve_high_level_actions", False)
        )
        for slot in range(2 * self.team_size):
            term = self.env.env.action_manager.get_term(f"skill_policy_{slot}")
            term.preserve_external_actions = preserve_provider and slot >= self.team_size
        observation, rewards, terminated, truncated, info = self.env.step(joint_actions)
        done = terminated | truncated
        self._sample_opponent_assignments(done)
        if self.opponent_action_provider is not None and hasattr(self.opponent_action_provider, "reset"):
            self.opponent_action_provider.reset(done.nonzero(as_tuple=False).flatten())
        self._update_observations(observation, reset_mask=done)
        local_role_rewards = self._local_role_rewards(info, done, previous_distances)
        info = dict(info)
        info["opponent_snapshot_iteration"] = self.opponent_snapshot_iteration
        info["opponent_snapshot_interval"] = self.opponent_snapshot_interval
        info["opponent_pool_size"] = len(self.opponent_pool)
        if self.opponent_pool_iterations:
            info["opponent_pool_selected_iterations"] = torch.as_tensor(
                self.opponent_pool_iterations, dtype=torch.long, device=self.device
            )[self.opponent_assignment]
        info["match_done"] = done
        info["match_rewards"] = rewards.detach()
        info["local_role_rewards"] = local_role_rewards.detach()
        agent_rewards = rewards.repeat_interleave(self.team_size) + local_role_rewards.reshape(-1)
        return self._cached, agent_rewards, terminated.repeat_interleave(self.team_size), truncated.repeat_interleave(self.team_size), info

    def _learning_team_ball_distances(self) -> torch.Tensor:
        robot_xy = torch.stack(
            [self.env.env.scene[f"robot_{slot}"].data.root_pos_w[:, :2] for slot in range(self.team_size)],
            dim=1,
        )
        ball_xy = self.env.env.scene["ball"].data.root_pos_w[:, None, :2]
        return torch.linalg.vector_norm(robot_xy - ball_xy, dim=-1)

    def _local_role_rewards(
        self,
        info: dict[str, Any],
        done: torch.Tensor,
        previous_distances: torch.Tensor,
    ) -> torch.Tensor:
        """Add current Isaac Gym per-agent attacker/support credit shaping."""

        terms = [
            self.env.env.action_manager.get_term(f"skill_policy_{slot}")
            for slot in range(self.team_size)
        ]
        attacker = torch.stack([term.attacker_mask for term in terms], dim=1)
        executed = torch.stack([term.skill_ids for term in terms], dim=1)
        requested = torch.stack([term.requested_skill_ids for term in terms], dim=1)
        invalid = torch.stack([term.invalid_skill_mask for term in terms], dim=1)
        commands = torch.stack([term.skill_commands[:, :2] for term in terms], dim=1)
        robot_xy = torch.stack(
            [self.env.env.scene[f"robot_{slot}"].data.root_pos_w[:, :2] for slot in range(self.team_size)],
            dim=1,
        )
        ball = self.env.env.scene["ball"]
        distances = torch.linalg.vector_norm(robot_xy - ball.data.root_pos_w[:, None, :2], dim=-1)
        attacker_ready = attacker & (previous_distances <= 1.0)
        valid_ball_skill = ((executed == 1) | (executed == 2)) & (requested == executed) & ~invalid
        command_fraction = (torch.linalg.vector_norm(commands, dim=-1) / 0.8).clamp(0.0, 1.0)
        command_direction = commands / torch.linalg.vector_norm(
            commands, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        aligned_ball_speed = torch.sum(
            ball.data.root_lin_vel_w[:, None, :2] * command_direction, dim=-1
        )
        elapsed = torch.as_tensor(
            info.get("elapsed_low_level_steps", self.env.control_interval),
            dtype=torch.float,
            device=self.device,
        ).reshape(self.match_count, 1)
        elapsed_time = (elapsed * float(self.env.env.step_dt)).clamp_min(1.0e-6)
        closing_fraction = (
            ((previous_distances - distances) / elapsed_time) / 0.5
        ).clamp(0.0, 1.0)
        ball_speed_fraction = aligned_ball_speed.clamp(0.0, 1.0)
        physical_consequence = torch.maximum(closing_fraction, ball_speed_fraction)
        active_score = 2.0 * command_fraction * physical_consequence - 1.0
        attacker_score = torch.where(valid_ball_skill, active_score, -torch.ones_like(active_score))
        local = torch.where(attacker_ready, attacker_score, torch.zeros_like(attacker_score))
        local *= attacker.float() * 2.0

        attacker_assist = attacker & (requested == 0) & ((executed == 1) | (executed == 2))
        role_conflict = ((~attacker) & (requested != 0)) | attacker_assist
        local += attacker_assist.float() * -4.0
        local += role_conflict.float() * -3.0
        support_crowding = ((1.15 - distances) / 1.15).clamp(0.0, 1.0).square()
        local += (~attacker).float() * support_crowding * -3.0
        local *= elapsed_time
        local[done] = 0.0
        return torch.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)

    def _opponent_actions(self) -> torch.Tensor:
        if self.opponent_action_provider is not None:
            actions = self.opponent_action_provider()
            expected = (self.match_count, self.team_size, self.num_actions)
            if not isinstance(actions, torch.Tensor) or actions.shape != expected:
                shape = tuple(actions.shape) if isinstance(actions, torch.Tensor) else type(actions).__name__
                raise RuntimeError(f"opponent action provider returned {shape}, expected {expected}")
            return torch.nan_to_num(actions.to(self.device), nan=0.0, posinf=10.0, neginf=-10.0)
        if not self.opponent_pool and self.opponent_policy_callable is None:
            return torch.zeros(
                self.match_count,
                self.team_size,
                self.num_actions,
                device=self.device,
            )
        grouped_history = self._history[:, 1]
        actions = torch.zeros(
            self.match_count, self.team_size, self.num_actions, device=self.device
        )
        policies = self.opponent_pool or [self.opponent_policy_callable]
        with torch.inference_mode():
            for pool_index, policy in enumerate(policies):
                match_mask = self.opponent_assignment == pool_index
                if not torch.any(match_mask):
                    continue
                history = grouped_history[match_mask].reshape(-1, self.num_obs_history)
                selected = policy(
                    {
                        "obs": history[:, -self.num_obs :],
                        "privileged_obs": history[:, -self.num_obs :],
                        "obs_history": history,
                    }
                )
                if not isinstance(selected, torch.Tensor):
                    raise TypeError(
                        f"opponent policy must return torch.Tensor, got {type(selected).__name__}"
                    )
                selected = legacy_policy_action_to_hybrid(selected)
                expected = (int(match_mask.sum()) * self.team_size, self.num_actions)
                if selected.shape != expected:
                    raise RuntimeError(
                        f"opponent policy returned shape {tuple(selected.shape)}, expected {expected}"
                    )
                actions[match_mask] = torch.nan_to_num(
                    selected.to(self.device), nan=0.0, posinf=10.0, neginf=-10.0
                ).view(-1, self.team_size, self.num_actions)
        # The checkpoint emits commands in its canonical +x frame. Team 1
        # attacks -x in the world, so rotate dribble/shoot planar commands back
        # before concatenating them with team-0 actions.
        return mirror_high_level_policy_actions(actions)

    def _sample_opponent_assignments(self, mask: torch.Tensor | None = None) -> None:
        if not self.opponent_pool:
            self.opponent_assignment.zero_()
            return
        if mask is None:
            targets = torch.arange(self.match_count, device=self.device)
        else:
            targets = torch.as_tensor(mask, dtype=torch.bool, device=self.device).nonzero().flatten()
        if targets.numel() == 0:
            return
        pool_count = len(self.opponent_pool)
        if pool_count == 1:
            sampled = torch.zeros(len(targets), dtype=torch.long, device=self.device)
        else:
            weights = torch.full(
                (pool_count,),
                (1.0 - self.opponent_latest_probability) / (pool_count - 1),
                device=self.device,
            )
            weights[-1] = self.opponent_latest_probability
            sampled = torch.multinomial(weights, len(targets), replacement=True)
        self.opponent_assignment[targets] = sampled

    def opponent_pool_state_dict(self) -> dict[str, Any] | None:
        """Return serializable learned-policy snapshots, or None for external providers."""

        if not self.opponent_pool:
            return None
        policies = []
        iterations = []
        for policy, iteration in zip(self.opponent_pool, self.opponent_pool_iterations, strict=True):
            if not hasattr(policy, "state_dict"):
                # Archived TorchScript/external callables are restored from
                # their configured root rather than embedded in PPO files.
                continue
            policies.append(policy.state_dict())
            iterations.append(int(iteration))
        if not policies:
            return None
        return {"format_version": 1, "iterations": iterations, "policies": policies}

    def restore_opponent_pool(
        self,
        policies: list[Callable[[dict[str, torch.Tensor]], torch.Tensor]],
        iterations: list[int],
    ) -> None:
        """Replace the runtime pool with validated reconstructed policies."""

        if not policies or len(policies) != len(iterations):
            raise ValueError("Restored opponent pool requires equally sized non-empty lists")
        if len(policies) > self.opponent_pool_size:
            if self.opponent_pool_size == 1:
                selected = [len(policies) - 1]
            else:
                selected = [0] + list(range(len(policies) - self.opponent_pool_size + 1, len(policies)))
            policies = [policies[index] for index in selected]
            iterations = [iterations[index] for index in selected]
        self.opponent_action_provider = None
        self.opponent_pool = list(policies)
        self.opponent_pool_iterations = [int(value) for value in iterations]
        self.opponent_policy_callable = self.opponent_pool[-1]
        self.opponent_snapshot_iteration = self.opponent_pool_iterations[-1]
        self._sample_opponent_assignments()

    def _update_observations(self, observation: dict[str, torch.Tensor], reset_mask: torch.Tensor | None) -> None:
        match_state = observation["policy"]
        if match_state.shape != (self.match_count, 2 * self.team_size, self.num_obs):
            raise RuntimeError(
                f"Match policy observation has shape {tuple(match_state.shape)}, expected "
                f"({self.match_count}, {2 * self.team_size}, {self.num_obs})"
            )
        for team in range(2):
            team_state = match_state[:, team * self.team_size : (team + 1) * self.team_size]
            history = self._history[:, team]
            history[:] = torch.cat((history[:, :, self.num_obs :], team_state), dim=-1)
            if reset_mask is not None:
                history[reset_mask] = 0.0
        team_state = match_state[:, : self.team_size].reshape(self.num_envs, self.num_obs)
        team_history = self._history[:, 0].reshape(self.num_envs, self.num_obs_history)
        self._cached = {
            "obs": team_state,
            "privileged_obs": team_state,
            "obs_history": team_history,
        }


def make_match_self_play_env(cfg=None, render_mode=None, **kwargs):
    """Gym entry point for the manager-based match plus self-play adapter."""

    if cfg is None:
        raise ValueError("make_match_self_play_env requires an Isaac Lab cfg")
    from isaaclab.envs import ManagerBasedRLEnv

    raw_env = ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    macro_env = MacroActionWrapper(raw_env, control_interval=int(getattr(cfg, "macro_control_interval", 10)))
    wrapper = MatchSelfPlayWrapper(
        macro_env,
        team_size=int(getattr(cfg, "team_size", 2)),
        history_length=int(getattr(cfg, "coordinator_history_length", 4)),
        opponent_snapshot_interval=int(getattr(cfg, "opponent_snapshot_interval", 500)),
        opponent_pool_size=int(getattr(cfg, "opponent_pool_size", 8)),
        opponent_latest_probability=float(getattr(cfg, "opponent_latest_probability", 0.5)),
    )
    opponent_root = getattr(cfg, "opponent_checkpoint_root", None)
    if opponent_root:
        wrapper.load_opponent_checkpoint(root=opponent_root, device=str(getattr(cfg, "opponent_policy_device", "cpu")))
    elif str(getattr(cfg, "opponent_mode", "zero")) == "rule_based":
        from .rule_based_opponent import RuleBasedOpponent

        wrapper.set_opponent_action_provider(RuleBasedOpponent(wrapper))
    return wrapper


__all__ = ["MatchSelfPlayWrapper", "make_match_self_play_env"]
