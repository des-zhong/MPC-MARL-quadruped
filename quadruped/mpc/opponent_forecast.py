"""Simple deterministic opponent actions for joint-team MPC."""

from __future__ import annotations

from typing import Optional

import torch


class SlowWalkToBallOpponentForecaster:
    """Make every opponent use the walk skill and approach the ball slowly.

    Walk commands are body-frame velocities. The current measured robot-to-ball
    direction is held over the MPC horizon, matching the action that is sent to
    the simulator for the next receding-horizon step.
    """

    def __init__(
        self,
        match_env,
        team_size: int,
        action_adapter,
        speed_mps: float = 0.35,
        stop_distance_m: float = 0.25,
        slow_distance_m: float = 1.0,
        yaw_gain: float = 1.0,
    ):
        self.match_env = match_env
        self.team_size = int(team_size)
        self.action_adapter = action_adapter
        self.speed_mps = float(speed_mps)
        self.stop_distance_m = float(stop_distance_m)
        self.slow_distance_m = float(slow_distance_m)
        self.yaw_gain = float(yaw_gain)
        if self.team_size < 1:
            raise ValueError("team_size must be at least 1")
        if action_adapter.num_robots != 2 * self.team_size:
            raise ValueError(
                "Simple opponent requires two equal teams: "
                f"team_size={self.team_size}, robots={action_adapter.num_robots}"
            )
        if self.speed_mps <= 0.0:
            raise ValueError("opponent speed_mps must be positive")
        if self.stop_distance_m < 0.0:
            raise ValueError("opponent stop_distance_m cannot be negative")
        if self.slow_distance_m <= self.stop_distance_m:
            raise ValueError(
                "opponent slow_distance_m must exceed stop_distance_m"
            )
        if self.yaw_gain < 0.0:
            raise ValueError("opponent yaw_gain cannot be negative")

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        del env_ids

    def _joint_action(self) -> torch.Tensor:
        affordances = self.match_env._skill_affordances()
        local_ball = affordances["local_ball_xy"][:, self.team_size :]
        distance = torch.linalg.vector_norm(local_ball, dim=-1)
        direction = local_ball / distance.clamp(min=1.0e-6).unsqueeze(-1)
        speed_fraction = (
            (distance - self.stop_distance_m)
            / (self.slow_distance_m - self.stop_distance_m)
        ).clamp(min=0.0, max=1.0)

        batch = local_ball.shape[0]
        device = local_ball.device
        dtype = local_ball.dtype
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
            dtype=dtype,
            device=device,
        )
        opponent_commands = commands[:, self.team_size :]
        opponent_commands[..., :2] = (
            direction * (self.speed_mps * speed_fraction).unsqueeze(-1)
        )
        opponent_commands[..., 2] = self.yaw_gain * torch.atan2(
            local_ball[..., 1], local_ball[..., 0]
        )

        bounds = self.action_adapter.bounds[0]
        low = torch.as_tensor(bounds.low, dtype=dtype, device=device)
        high = torch.as_tensor(bounds.high, dtype=dtype, device=device)
        mask = torch.as_tensor(bounds.mask, dtype=dtype, device=device)
        opponent_commands[:] = torch.maximum(
            torch.minimum(opponent_commands, high), low
        ) * mask
        return self.action_adapter.pack(skills, commands)

    def current_joint_action(self) -> torch.Tensor:
        """Return the current canonical action for both teams.

        Learning-team entries are zero placeholders; opponent entries contain
        the slow walk command.
        """

        return self._joint_action()

    def fixed_action_sequence(self, horizon: int):
        joint = self.current_joint_action()
        fixed = joint[:, None].expand(-1, int(horizon), -1).clone()
        mask = torch.zeros(
            self.action_adapter.num_robots,
            dtype=torch.bool,
            device=joint.device,
        )
        mask[self.team_size :] = True
        return fixed, mask

    def wrapper_actions(self) -> torch.Tensor:
        """Return opponent-only raw actions for SharedPolicySelfPlayWrapper."""

        raw = self.action_adapter.to_wrapper_action(self.current_joint_action())
        raw = raw.reshape(
            raw.shape[0], self.action_adapter.num_robots, 6
        )
        return raw[:, self.team_size :]

    def observe(self, dones: torch.Tensor) -> None:
        del dones


class ZeroOpponentForecaster:
    """Keep the opponent on zero-command walk actions when requested."""

    def __init__(self, match_env, team_size: int, action_adapter):
        self.match_env = match_env
        self.team_size = int(team_size)
        self.action_adapter = action_adapter

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        del env_ids

    def _joint_action(self) -> torch.Tensor:
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
        return self.action_adapter.pack(skills, commands)

    def current_joint_action(self) -> torch.Tensor:
        return self._joint_action()

    def fixed_action_sequence(self, horizon: int):
        joint = self.current_joint_action()
        fixed = joint[:, None].expand(-1, int(horizon), -1).clone()
        mask = torch.zeros(
            self.action_adapter.num_robots, dtype=torch.bool, device=joint.device
        )
        mask[self.team_size :] = True
        return fixed, mask

    def wrapper_actions(self) -> torch.Tensor:
        raw = self.action_adapter.to_wrapper_action(self.current_joint_action())
        raw = raw.reshape(
            raw.shape[0], self.action_adapter.num_robots, 6
        )
        return raw[:, self.team_size :]

    def observe(self, dones: torch.Tensor) -> None:
        del dones
