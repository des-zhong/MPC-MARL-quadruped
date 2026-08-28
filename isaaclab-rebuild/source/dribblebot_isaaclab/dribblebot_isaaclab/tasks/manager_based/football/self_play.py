"""Thin self-play adapter for the four-robot manager-based match scene."""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import torch

from .macro import MacroActionWrapper
from .team_frame import mirror_high_level_policy_actions


class MatchSelfPlayWrapper(gym.Wrapper):
    """Expose one shared-policy sample per learning-team robot."""

    num_obs = 34
    num_actions = 6

    def __init__(
        self,
        env: MacroActionWrapper,
        team_size: int = 2,
        history_length: int = 4,
        opponent_snapshot_interval: int = 500,
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
        self.action_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.num_envs, self.num_actions),
            dtype=float,
        )
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

    def load_opponent_checkpoint(self, root=None, device: str = "cpu", iteration: int = -1) -> None:
        """Install the archived TorchScript coordinator as an opponent snapshot."""

        from ....policies.high_level import FrozenCoordinatorPolicy

        self.opponent_policy_callable = FrozenCoordinatorPolicy.load(root=root, device=device)
        self.opponent_snapshot_iteration = int(iteration)

    def update_opponent_snapshot(
        self,
        policy: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        iteration: int,
        force: bool = False,
    ) -> bool:
        """Atomically replace the inference callable at the configured cadence."""

        iteration = int(iteration)
        if not force and iteration % self.opponent_snapshot_interval != 0:
            return False
        self.opponent_policy_callable = policy
        self.opponent_snapshot_iteration = iteration
        return True

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self._history.zero_()
        self._last_team_actions.zero_()
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
        opponent_actions = self._opponent_actions()
        joint_actions = torch.cat((team_actions, opponent_actions), dim=1).reshape(
            self.match_count,
            2 * self.team_size * self.num_actions,
        )
        observation, rewards, terminated, truncated, info = self.env.step(joint_actions)
        done = terminated | truncated
        self._update_observations(observation, reset_mask=done)
        info = dict(info)
        info["opponent_snapshot_iteration"] = self.opponent_snapshot_iteration
        info["opponent_snapshot_interval"] = self.opponent_snapshot_interval
        info["match_done"] = done
        return self._cached, rewards.repeat_interleave(self.team_size), terminated.repeat_interleave(self.team_size), truncated.repeat_interleave(self.team_size), info

    def _opponent_actions(self) -> torch.Tensor:
        if self.opponent_policy_callable is None:
            return torch.zeros(
                self.match_count,
                self.team_size,
                self.num_actions,
                device=self.device,
            )
        history = self._history[:, 1].reshape(self.match_count * self.team_size, self.num_obs_history)
        with torch.inference_mode():
            actions = self.opponent_policy_callable(
                {
                    "obs": history[:, -self.num_obs :],
                    "privileged_obs": history[:, -self.num_obs :],
                    "obs_history": history,
                }
            )
        if not isinstance(actions, torch.Tensor):
            raise TypeError(f"opponent policy must return torch.Tensor, got {type(actions).__name__}")
        expected = (self.match_count * self.team_size, self.num_actions)
        if actions.shape != expected:
            raise RuntimeError(f"opponent policy returned shape {tuple(actions.shape)}, expected {expected}")
        actions = torch.nan_to_num(actions.to(self.device), nan=0.0, posinf=10.0, neginf=-10.0)
        actions = actions.view(self.match_count, self.team_size, self.num_actions)
        # The checkpoint emits commands in its canonical +x frame. Team 1
        # attacks -x in the world, so rotate dribble/shoot planar commands back
        # before concatenating them with team-0 actions.
        return mirror_high_level_policy_actions(actions)

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
    )
    opponent_root = getattr(cfg, "opponent_checkpoint_root", None)
    if opponent_root:
        wrapper.load_opponent_checkpoint(root=opponent_root, device=str(getattr(cfg, "opponent_policy_device", "cpu")))
    return wrapper


__all__ = ["MatchSelfPlayWrapper", "make_match_self_play_env"]
