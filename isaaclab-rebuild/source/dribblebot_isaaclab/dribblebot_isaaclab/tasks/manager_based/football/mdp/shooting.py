"""State machine and reward terms for the low-level shooting skill."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .geometry import (
    compute_shooting_phase,
    shooting_forward_velocity_score,
    shooting_setup_geometry,
    shooting_setup_progress,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg, TerminationTermCfg


def _command_geometry(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    target_speed = torch.linalg.vector_norm(command_xy, dim=-1)
    command_direction = command_xy / target_speed.clamp_min(1.0e-6).unsqueeze(-1)
    return command_xy, target_speed, command_direction


def _phase_buffer(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return value


def _pre_kick_gate(env: ManagerBasedRLEnv, target_speed: torch.Tensor, min_command_speed: float) -> torch.Tensor:
    active = target_speed > float(min_command_speed)
    launched = _phase_buffer(env, "_shooting_launched")
    return (active & ~launched).float()


def _post_kick_gate(env: ManagerBasedRLEnv, target_speed: torch.Tensor, min_command_speed: float) -> torch.Tensor:
    active = target_speed > float(min_command_speed)
    launched = _phase_buffer(env, "_shooting_launched")
    return (active & launched).float()


def _separation_gate(robot: Articulation, ball: RigidObject, min_separation: float) -> torch.Tensor:
    distance = torch.linalg.vector_norm(ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    return (distance > float(min_separation)).float()


def _setup_score(
    robot: Articulation,
    ball: RigidObject,
    command_xy: torch.Tensor,
    setup_distance: float,
    position_gain: float,
) -> torch.Tensor:
    _, _, setup_error = shooting_setup_geometry(
        robot.data.root_pos_w[:, :2],
        ball.data.root_pos_w[:, :2],
        command_xy,
        setup_distance,
    )
    return torch.exp(-float(position_gain) * torch.square(setup_error))


def shooting_ball_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale_xy: tuple[float, float],
    tracking_sigma: float,
    min_command_speed: float,
    min_separation: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Track the full desired ball velocity after a separated launch."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    command_xy, target_speed, _ = _command_geometry(env, command_name)
    scale = command_xy.new_tensor(scale_xy).clamp_min(1.0e-6)
    velocity_error = torch.sum(torch.square((command_xy - ball.data.root_lin_vel_w[:, :2]) / scale), dim=-1)
    tracking = torch.exp(-velocity_error / max(float(tracking_sigma) * 2.0, 1.0e-6))
    return (
        tracking
        * _post_kick_gate(env, target_speed, min_command_speed)
        * _separation_gate(robot, ball, min_separation)
    )


def shooting_ball_forward_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_command_speed: float,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Give dense credit for the first weak but correctly directed strike."""

    ball: RigidObject = env.scene[ball_cfg.name]
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    return shooting_forward_velocity_score(ball.data.root_lin_vel_w[:, :2], command_xy, min_command_speed)


def shooting_ball_speed_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale_xy: tuple[float, float],
    min_command_speed: float,
    min_separation: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Track desired speed magnitude after launch."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    _, target_speed, _ = _command_geometry(env, command_name)
    ball_speed = torch.linalg.vector_norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
    speed_scale = torch.linalg.vector_norm(ball_speed.new_tensor(scale_xy)).clamp_min(1.0e-6)
    speed_error = torch.square((target_speed - ball_speed) / speed_scale)
    return (
        torch.exp(-2.0 * speed_error)
        * _post_kick_gate(env, target_speed, min_command_speed)
        * _separation_gate(robot, ball, min_separation)
    )


def shooting_ball_direction(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_command_speed: float,
    min_separation: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Align post-launch ball motion with the requested direction."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    _, target_speed, command_direction = _command_geometry(env, command_name)
    ball_velocity = ball.data.root_lin_vel_w[:, :2]
    ball_speed = torch.linalg.vector_norm(ball_velocity, dim=-1)
    ball_direction = ball_velocity / ball_speed.clamp_min(1.0e-6).unsqueeze(-1)
    alignment = torch.sum(command_direction * ball_direction, dim=-1).clamp(-1.0, 1.0)
    moving = (ball_speed > 0.10).float()
    return (
        0.5
        * (alignment + 1.0)
        * moving
        * _post_kick_gate(env, target_speed, min_command_speed)
        * _separation_gate(robot, ball, min_separation)
    )


def shooting_ball_out(
    env: ManagerBasedRLEnv,
    command_name: str,
    scale_xy: tuple[float, float],
    setup_distance: float,
    min_command_speed: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Reward an aligned ball separating from the robot after launch."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    _, target_speed, command_direction = _command_geometry(env, command_name)
    robot_to_ball = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    ball_distance = torch.linalg.vector_norm(robot_to_ball, dim=-1).clamp_min(1.0e-6)
    ball_direction = robot_to_ball / ball_distance.unsqueeze(-1)
    position_alignment = torch.sum(ball_direction * command_direction, dim=-1).clamp(0.0, 1.0)
    speed_along_command = torch.sum(ball.data.root_lin_vel_w[:, :2] * command_direction, dim=-1).clamp_min(0.0)
    speed_scale = torch.linalg.vector_norm(robot_to_ball.new_tensor(scale_xy)).clamp_min(1.0e-6)
    separation = (ball_distance - float(setup_distance)).clamp_min(0.0)
    return (
        torch.tanh(2.0 * separation)
        * position_alignment
        * torch.tanh(speed_along_command / speed_scale)
        * _post_kick_gate(env, target_speed, min_command_speed)
    )


def shooting_setup_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    setup_distance: float,
    position_gain: float,
    min_command_speed: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Place the robot at the desired setup pose behind the ball."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    command_xy, target_speed, _ = _command_geometry(env, command_name)
    score = _setup_score(robot, ball, command_xy, setup_distance, position_gain)
    return score * _pre_kick_gate(env, target_speed, min_command_speed)


def shooting_robot_heading(
    env: ManagerBasedRLEnv,
    command_name: str,
    setup_distance: float,
    position_gain: float,
    min_command_speed: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Align body forward with the command only near the setup pose."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    command_xy, target_speed, command_direction = _command_geometry(env, command_name)
    forward_seed = torch.zeros_like(robot.data.root_pos_w)
    forward_seed[:, 0] = 1.0
    body_forward = quat_apply(robot.data.root_quat_w, forward_seed)[:, :2]
    heading_alignment = torch.sum(body_forward * command_direction, dim=-1).clamp(-1.0, 1.0)
    setup_score = _setup_score(robot, ball, command_xy, setup_distance, position_gain)
    return (
        0.5
        * (heading_alignment + 1.0)
        * setup_score
        * _pre_kick_gate(env, target_speed, min_command_speed)
    )


def shooting_ball_in_front(
    env: ManagerBasedRLEnv,
    command_name: str,
    setup_distance: float,
    position_gain: float,
    min_command_speed: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Keep the ball in front while the robot occupies its setup pose."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    command_xy, target_speed, _ = _command_geometry(env, command_name)
    forward_seed = torch.zeros_like(robot.data.root_pos_w)
    forward_seed[:, 0] = 1.0
    body_forward = quat_apply(robot.data.root_quat_w, forward_seed)[:, :2]
    robot_to_ball = ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    robot_to_ball_direction = robot_to_ball / torch.linalg.vector_norm(
        robot_to_ball, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    front_alignment = torch.sum(body_forward * robot_to_ball_direction, dim=-1).clamp(0.0, 1.0)
    setup_score = _setup_score(robot, ball, command_xy, setup_distance, position_gain)
    return front_alignment * setup_score * _pre_kick_gate(env, target_speed, min_command_speed)


class ShootingSetupProgressReward(ManagerTermBase):
    """Signed progress toward the moving setup pose behind the ball."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot_cfg = cfg.params.get("robot_cfg", SceneEntityCfg("robot"))
        self.robot: Articulation = env.scene[robot_cfg.name]
        self.previous_base_xy = self.robot.data.root_pos_w[:, :2].clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.previous_base_xy[env_ids] = self.robot.data.root_pos_w[env_ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        setup_distance: float,
        speed_scale: float,
        min_command_speed: float,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        ball: RigidObject = env.scene[ball_cfg.name]
        command_xy, target_speed, _ = _command_geometry(env, command_name)
        current_base_xy = robot.data.root_pos_w[:, :2]

        reset_base_xy = getattr(env, "_shooting_reset_base_xy", self.previous_base_xy)
        first_step = env.episode_length_buf <= 1
        previous_base_xy = torch.where(first_step.unsqueeze(-1), reset_base_xy, self.previous_base_xy)
        progress = shooting_setup_progress(
            previous_base_xy,
            current_base_xy,
            ball.data.root_pos_w[:, :2],
            command_xy,
            setup_distance,
            env.step_dt,
            speed_scale,
        )
        self.previous_base_xy[:] = current_base_xy
        return progress * _pre_kick_gate(env, target_speed, min_command_speed)


def shooting_excess_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,
    free_yaw_rate: float,
    yaw_rate_scale: float,
    min_command_speed: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only excessive pre-kick yaw rate."""

    robot: Articulation = env.scene[robot_cfg.name]
    _, target_speed, _ = _command_geometry(env, command_name)
    excess = (torch.abs(robot.data.root_ang_vel_b[:, 2]) - float(free_yaw_rate)).clamp_min(0.0)
    penalty = torch.square(excess / max(float(yaw_rate_scale), 1.0e-6))
    return penalty * _pre_kick_gate(env, target_speed, min_command_speed)


def shooting_launch_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step event emitted when a strike first crosses the launch gate."""

    return _phase_buffer(env, "_shooting_launch_event").float()


def shooting_success_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step terminal shooting success event."""

    return _phase_buffer(env, "_shooting_success").float()


def shooting_failure_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step terminal shooting timeout event."""

    return _phase_buffer(env, "_shooting_failure").float()


class ShootingPhaseTermination(ManagerTermBase):
    """Own persistent launch state and emit terminal success or failure."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.launched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.launch_event = torch.zeros_like(self.launched)
        self.success = torch.zeros_like(self.launched)
        self.failure = torch.zeros_like(self.launched)
        self.launch_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.completed_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.success_count = torch.zeros_like(self.completed_count)
        self.failure_count = torch.zeros_like(self.completed_count)
        self.last_terminal_step = torch.full_like(self.completed_count, -1)
        self.last_terminal_launch_step = torch.full_like(self.completed_count, -1)
        self.last_terminal_success = torch.zeros_like(self.launched)
        self.last_terminal_failure = torch.zeros_like(self.launched)
        self.last_terminal_ball_speed = torch.zeros(env.num_envs, device=env.device)
        self.last_terminal_separation = torch.zeros(env.num_envs, device=env.device)
        self.last_terminal_alignment = torch.zeros(env.num_envs, device=env.device)
        self._publish()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.launched[env_ids] = False
        self.launch_event[env_ids] = False
        self.success[env_ids] = False
        self.failure[env_ids] = False
        self.launch_step[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        min_command_speed: float,
        launch_speed_fraction: float,
        launch_alignment: float,
        success_distance: float,
        success_speed_fraction: float,
        success_alignment: float,
        max_attempt_time_s: float,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[robot_cfg.name]
        ball: RigidObject = env.scene[ball_cfg.name]
        command_xy = env.command_manager.get_command(command_name)[:, :2]
        elapsed_s = env.episode_length_buf.float() * env.step_dt
        launch_event, launched, success, failure = compute_shooting_phase(
            robot.data.root_pos_w[:, :2],
            ball.data.root_pos_w[:, :2],
            ball.data.root_lin_vel_w[:, :2],
            command_xy,
            self.launched,
            elapsed_s,
            min_command_speed,
            launch_speed_fraction,
            launch_alignment,
            success_distance,
            success_speed_fraction,
            success_alignment,
            max_attempt_time_s,
        )
        self.launch_event[:] = launch_event
        self.launched[:] = launched
        self.success[:] = success
        self.failure[:] = failure
        self.launch_step[launch_event] = env.episode_length_buf[launch_event]

        terminal = success | failure
        if torch.any(terminal):
            target_speed = torch.linalg.vector_norm(command_xy, dim=-1)
            command_direction = command_xy / target_speed.clamp_min(1.0e-6).unsqueeze(-1)
            ball_velocity_xy = ball.data.root_lin_vel_w[:, :2]
            ball_speed = torch.linalg.vector_norm(ball_velocity_xy, dim=-1)
            alignment = torch.sum(ball_velocity_xy * command_direction, dim=-1) / ball_speed.clamp_min(1.0e-6)
            separation = torch.linalg.vector_norm(
                ball.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2],
                dim=-1,
            )
            self.completed_count[terminal] += 1
            self.success_count[success] += 1
            self.failure_count[failure] += 1
            self.last_terminal_step[terminal] = env.episode_length_buf[terminal]
            self.last_terminal_launch_step[terminal] = self.launch_step[terminal]
            self.last_terminal_success[terminal] = success[terminal]
            self.last_terminal_failure[terminal] = failure[terminal]
            self.last_terminal_ball_speed[terminal] = ball_speed[terminal]
            self.last_terminal_separation[terminal] = separation[terminal]
            self.last_terminal_alignment[terminal] = alignment[terminal]
        return success | failure

    def _publish(self) -> None:
        self._env._shooting_launched = self.launched
        self._env._shooting_launch_event = self.launch_event
        self._env._shooting_success = self.success
        self._env._shooting_failure = self.failure
