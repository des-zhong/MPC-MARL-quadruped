"""Command generators for football skills."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from .legacy_contract import advance_gait_phase, legacy_gait_clock

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GaitCommand(CommandTerm):
    """Generate the 12 non-velocity values in the legacy 15D command."""

    cfg: GaitCommandCfg

    def __init__(self, cfg: GaitCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.gait_parameters = torch.zeros(self.num_envs, 12, device=self.device)
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.last_update_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._observation_revision = 0
        self._validate_cfg()
        self._set_range_midpoints()

    @property
    def command(self) -> torch.Tensor:
        """Unscaled gait parameters in the old command order."""

        return self.gait_parameters

    @property
    def clock(self) -> torch.Tensor:
        """Four phase clocks in FL, FR, RL, RR order."""

        return legacy_gait_clock(self.gait_phase, self.gait_parameters, self.cfg.pacing_offset)

    @property
    def observation_revision(self) -> int:
        """Revision used to invalidate observation caches after command updates."""

        return self._observation_revision

    def set_parameters(self, parameters: torch.Tensor, env_ids: Sequence[int] | None = None) -> None:
        """Inject externally selected gait parameters without changing phase."""

        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        tensor_ids = self._as_index_tensor(ids)
        if parameters.ndim != 2 or parameters.shape != (tensor_ids.numel(), 12):
            raise ValueError(f"Gait parameters must have shape ({tensor_ids.numel()}, 12).")
        parameters = parameters.to(device=self.device, dtype=self.gait_parameters.dtype)
        duration = parameters[:, 5]
        if torch.any((duration <= 0.0) | (duration >= 1.0)):
            raise ValueError("Gait duration must be strictly between zero and one.")
        self.gait_parameters[tensor_ids] = parameters
        self._observation_revision += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        extras = super().reset(ids)
        tensor_ids = self._as_index_tensor(ids)

        # The old external reset performed one implicit zero-action step.  Its
        # first returned observation therefore contained one phase increment.
        # Internal episode resets did not perform that extra step.
        if self._env.common_step_counter == 0:
            frequency = self.gait_parameters[tensor_ids, 1]
            initial = torch.zeros_like(frequency)
            self.gait_phase[tensor_ids] = advance_gait_phase(initial, frequency, self._env.step_dt)
        else:
            self.gait_phase[tensor_ids] = 0.0
        self.last_update_step[tensor_ids] = int(self._env.common_step_counter)
        self._observation_revision += 1
        return extras

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        # Episode length is zero for environments reset inside the current
        # step.  Their reset observation must retain phase zero.
        active = self._env.episode_length_buf > 0
        advanced = advance_gait_phase(
            self.gait_phase,
            self.gait_parameters[:, 1],
            self._env.step_dt,
        )
        self.gait_phase[:] = torch.where(active, advanced, self.gait_phase)
        self.last_update_step[:] = int(self._env.common_step_counter)
        self._observation_revision += 1

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = self._as_index_tensor(env_ids)
        if ids.numel() == 0:
            return
        for column, value_range in enumerate(self._ranges()):
            self.gait_parameters[ids, column] = self._uniform(value_range, ids.numel())

    def _set_range_midpoints(self) -> None:
        for column, value_range in enumerate(self._ranges()):
            self.gait_parameters[:, column] = 0.5 * (value_range[0] + value_range[1])

    def _ranges(self) -> tuple[tuple[float, float], ...]:
        ranges = self.cfg.ranges
        return (
            ranges.body_height,
            ranges.frequency,
            ranges.phase,
            ranges.offset,
            ranges.bound,
            ranges.duration,
            ranges.foot_swing_height,
            ranges.body_pitch,
            ranges.body_roll,
            ranges.stance_width,
            ranges.stance_length,
            ranges.aux_reward,
        )

    def _uniform(self, value_range: tuple[float, float], count: int) -> torch.Tensor:
        return torch.empty(count, device=self.device).uniform_(*value_range)

    def _as_index_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _validate_cfg(self) -> None:
        ranges = self._ranges()
        if any(low > high for low, high in ranges):
            raise ValueError("Gait command ranges must be ordered as (minimum, maximum).")
        if self.cfg.ranges.frequency[0] < 0.0:
            raise ValueError("Gait frequency must be non-negative.")
        duration = self.cfg.ranges.duration
        if duration[0] <= 0.0 or duration[1] >= 1.0:
            raise ValueError("Gait duration range must stay strictly between zero and one.")


@configclass
class GaitCommandCfg(CommandTermCfg):
    """Configuration for the legacy gait-parameter command."""

    class_type: type = GaitCommand
    pacing_offset: bool = False

    @configclass
    class Ranges:
        """Uniform ranges in legacy command indices 3 through 14."""

        body_height: tuple[float, float] = MISSING
        frequency: tuple[float, float] = MISSING
        phase: tuple[float, float] = MISSING
        offset: tuple[float, float] = MISSING
        bound: tuple[float, float] = MISSING
        duration: tuple[float, float] = MISSING
        foot_swing_height: tuple[float, float] = MISSING
        body_pitch: tuple[float, float] = MISSING
        body_roll: tuple[float, float] = MISSING
        stance_width: tuple[float, float] = MISSING
        stance_length: tuple[float, float] = MISSING
        aux_reward: tuple[float, float] = MISSING

    ranges: Ranges = MISSING


class ShootingCommand(CommandTerm):
    """Sample a world-frame ball velocity and create its attainable setup."""

    cfg: ShootingCommandCfg

    def __init__(self, cfg: ShootingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.robot_name]
        self.ball: RigidObject = env.scene[cfg.ball_name]
        self.target_velocity_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._validate_cfg()

        # The progress reward reads this cache on the first step after reset.
        env._shooting_reset_base_xy = self.robot.data.root_pos_w[:, :2].clone()

    @property
    def command(self) -> torch.Tensor:
        """Desired post-strike ball velocity in the world frame."""

        return self.target_velocity_w

    def _update_metrics(self) -> None:
        pass

    def _update_command(self) -> None:
        pass

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = self._as_index_tensor(env_ids)
        if ids.numel() == 0:
            return

        self.target_velocity_w[ids, :2] = self._sample_velocity_xy(ids.numel())
        self.target_velocity_w[ids, 2] = 0.0

        # A command can also resample because its timer elapsed. Only a manager
        # reset should teleport the robot and ball into a new shooting setup.
        reset_ids = ids[self.command_counter[ids] == 0]
        if reset_ids.numel() > 0:
            self._reset_shooting_scenario(reset_ids)

    def _reset_shooting_scenario(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        command_xy = self.target_velocity_w[env_ids, :2]
        command_direction = command_xy / torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
        command_left = torch.stack((-command_direction[:, 1], command_direction[:, 0]), dim=-1)

        yaw_error = self._uniform(self.cfg.yaw_error_range, count)
        command_yaw = torch.atan2(command_direction[:, 1], command_direction[:, 0])
        zeros = torch.zeros(count, device=self.device)

        robot_pose = self.robot.data.root_state_w[env_ids, :7].clone()
        robot_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, command_yaw + yaw_error)
        robot_velocity = torch.zeros(count, 6, device=self.device)
        self.robot.write_root_pose_to_sim(robot_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(robot_velocity, env_ids=env_ids)

        longitudinal = self._uniform(self.cfg.longitudinal_range, count)
        lateral = self._uniform(self.cfg.lateral_range, count)
        ball_offset = longitudinal.unsqueeze(-1) * command_direction + lateral.unsqueeze(-1) * command_left

        ball_pose = self.ball.data.default_root_state[env_ids, :7].clone()
        ball_pose[:, :2] = robot_pose[:, :2] + ball_offset
        ball_pose[:, 2] = self._env.scene.env_origins[env_ids, 2] + float(self.cfg.ball_height)
        ball_velocity = torch.zeros(count, 6, device=self.device)
        self.ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(ball_velocity, env_ids=env_ids)

        self._env._shooting_reset_base_xy[env_ids] = robot_pose[:, :2]

    def _sample_velocity_xy(self, count: int) -> torch.Tensor:
        command_xy = torch.empty(count, 2, device=self.device)
        command_xy[:, 0] = self._uniform(self.cfg.ranges.lin_vel_x, count)
        command_xy[:, 1] = self._uniform(self.cfg.ranges.lin_vel_y, count)

        for _ in range(16):
            invalid = torch.linalg.vector_norm(command_xy, dim=-1) <= float(self.cfg.min_command_speed)
            invalid_count = int(invalid.sum().item())
            if invalid_count == 0:
                break
            command_xy[invalid, 0] = self._uniform(self.cfg.ranges.lin_vel_x, invalid_count)
            command_xy[invalid, 1] = self._uniform(self.cfg.ranges.lin_vel_y, invalid_count)

        invalid = torch.linalg.vector_norm(command_xy, dim=-1) <= float(self.cfg.min_command_speed)
        if torch.any(invalid):
            fallback_x = max(self.cfg.ranges.lin_vel_x, key=abs)
            fallback_y = max(self.cfg.ranges.lin_vel_y, key=abs)
            command_xy[invalid] = command_xy.new_tensor((fallback_x, fallback_y))
        return command_xy

    def _uniform(self, value_range: tuple[float, float], count: int) -> torch.Tensor:
        return torch.empty(count, device=self.device).uniform_(*value_range)

    def _as_index_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _validate_cfg(self) -> None:
        ranges = (
            self.cfg.ranges.lin_vel_x,
            self.cfg.ranges.lin_vel_y,
            self.cfg.longitudinal_range,
            self.cfg.lateral_range,
            self.cfg.yaw_error_range,
        )
        if any(low > high for low, high in ranges):
            raise ValueError("Shooting command ranges must be ordered as (minimum, maximum).")
        if self.cfg.min_command_speed < 0.0:
            raise ValueError("min_command_speed must be non-negative.")
        max_x = max(abs(value) for value in self.cfg.ranges.lin_vel_x)
        max_y = max(abs(value) for value in self.cfg.ranges.lin_vel_y)
        if (max_x**2 + max_y**2) ** 0.5 <= self.cfg.min_command_speed:
            raise ValueError("Velocity ranges cannot satisfy min_command_speed.")


@configclass
class ShootingCommandCfg(CommandTermCfg):
    """Configuration for a shooting target and command-relative reset."""

    class_type: type = ShootingCommand
    robot_name: str = "robot"
    ball_name: str = "ball"
    min_command_speed: float = 0.3
    longitudinal_range: tuple[float, float] = (0.35, 0.60)
    lateral_range: tuple[float, float] = (-0.20, 0.20)
    yaw_error_range: tuple[float, float] = (-0.25, 0.25)
    ball_height: float = 0.10

    @configclass
    class Ranges:
        """Uniform ranges for desired world-frame ball velocity."""

        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
