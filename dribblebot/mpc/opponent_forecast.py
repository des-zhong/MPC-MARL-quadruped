"""Frozen opponent-policy forecasts for joint-team MPC execution."""

from __future__ import annotations

from typing import Optional

import torch


class FrozenPolicyOpponentForecaster:
    """Hold a frozen high-level opponent's current decision over the horizon.

    The self-play observation wrapper is reused only as an observation/history
    encoder. Simulator stepping remains owned by ``MPCSimulatorController`` so
    the planner's complete joint action is executed exactly once.
    """

    def __init__(
        self,
        match_env,
        team_size: int,
        action_adapter,
        policy_record,
        opponent_device=None,
    ):
        from dribblebot.envs.wrappers.shared_self_play_wrapper import (
            SharedPolicySelfPlayWrapper,
        )

        self.match_env = match_env
        self.team_size = int(team_size)
        self.action_adapter = action_adapter
        self.observer = SharedPolicySelfPlayWrapper(
            match_env,
            team_size=self.team_size,
            opponent_device=opponent_device,
        )
        expected_history_dim = policy_record.get("expected_history_dim")
        if (
            expected_history_dim is not None
            and int(expected_history_dim) != self.observer.num_obs_history
        ):
            raise ValueError(
                "Opponent high-level policy expects obs_history dim "
                f"{expected_history_dim}, but MPC provides "
                f"{self.observer.num_obs_history} "
                f"({self.observer.num_obs} obs x "
                f"{self.observer.history_length} history)."
            )
        self.observer.set_opponent_callable(policy_record["policy"])

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.observer._history.zero_()
            self.observer._update_observations()
            return
        ids = torch.as_tensor(
            env_ids, device=self.observer.device, dtype=torch.long
        ).flatten()
        if ids.numel():
            self.observer._history[ids] = 0.0

    def fixed_action_sequence(self, horizon: int):
        # preview_opponent_actions has already converted canonical policy output
        # to executable world semantics. This matches the joint world model's
        # fixed global field-frame action schema.
        raw = self.observer.preview_opponent_actions()
        skills = raw[..., :3].argmax(dim=-1)
        commands = (
            torch.tanh(raw[..., 3:6])
            * self.match_env._command_scales(skills)
        )
        commands[..., 2] = torch.where(
            skills == 2,
            torch.zeros_like(commands[..., 2]),
            commands[..., 2],
        )
        batch = raw.shape[0]
        all_skills = torch.zeros(
            batch,
            2 * self.team_size,
            dtype=torch.long,
            device=raw.device,
        )
        all_commands = torch.zeros(
            batch,
            2 * self.team_size,
            3,
            dtype=raw.dtype,
            device=raw.device,
        )
        all_skills[:, self.team_size :] = skills
        all_commands[:, self.team_size :] = commands
        joint = self.action_adapter.pack(all_skills, all_commands)
        fixed = joint[:, None].expand(-1, int(horizon), -1).clone()
        mask = torch.zeros(
            self.action_adapter.num_robots,
            dtype=torch.bool,
            device=raw.device,
        )
        mask[self.team_size :] = True
        return fixed, mask

    def observe(self, dones: torch.Tensor) -> None:
        self.observer._update_observations(reset_mask=dones.bool())


class ZeroOpponentForecaster:
    """Keep the opponent on zero-command reposition actions when disabled."""

    def __init__(self, match_env, team_size: int, action_adapter):
        self.match_env = match_env
        self.team_size = int(team_size)
        self.action_adapter = action_adapter

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        return None

    def fixed_action_sequence(self, horizon: int):
        batch = int(self.match_env.num_envs)
        device = self.match_env.device
        skills = torch.zeros(
            batch,
            self.action_adapter.num_robots,
            dtype=torch.long,
            device=device,
        )
        commands = torch.zeros(
            batch,
            self.action_adapter.num_robots,
            3,
            dtype=torch.float,
            device=device,
        )
        joint = self.action_adapter.pack(skills, commands)
        fixed = joint[:, None].expand(-1, int(horizon), -1).clone()
        mask = torch.zeros(
            self.action_adapter.num_robots, dtype=torch.bool, device=device
        )
        mask[self.team_size :] = True
        return fixed, mask

    def observe(self, dones: torch.Tensor) -> None:
        return None
