"""Football observation terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse

from .legacy_contract import (
    LEGACY_BALL_OBSERVATION_DIM,
    LEGACY_HISTORY_LENGTH,
    LEGACY_WALK_OBSERVATION_DIM,
    add_legacy_observation_noise,
    assemble_legacy_observation,
    compose_legacy_command,
    flatten_history,
    update_zero_padded_history,
)
from ..team_frame import mirror_high_level_commands, team_signs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_OBSERVATION_CACHE = "_dribblebot_legacy_observation_cache"
_RESET_ACTIONS = "_dribblebot_legacy_reset_actions"


class LegacyPolicyObservation(ManagerTermBase):
    """Build one 72D/75D observation compatible with old low-level policies."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._allow_noise = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._allow_noise = True
        _capture_pre_action_reset(self._env, env_ids, self.cfg.params.get("action_name"))
        invalidate_legacy_observation_cache(self._env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        include_ball: bool,
        enable_noise: bool = True,
        base_command_name: str = "base_velocity",
        gait_command_name: str = "gait_parameters",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        action_name: str | None = None,
        action_clip: float = 1.0,
        observation_clip: float = 100.0,
    ) -> torch.Tensor:
        return _legacy_policy_observation(
            env,
            include_ball=include_ball,
            enable_noise=enable_noise and self._allow_noise,
            base_command_name=base_command_name,
            gait_command_name=gait_command_name,
            robot_cfg=robot_cfg,
            ball_cfg=ball_cfg,
            action_name=action_name,
            action_clip=action_clip,
            observation_clip=observation_clip,
        )


class LegacyHistoryObservation(ManagerTermBase):
    """Maintain old zero-padded history without Isaac Lab's first-frame fill."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        include_ball = bool(cfg.params["include_ball"])
        self._history_length = int(cfg.params.get("history_length", LEGACY_HISTORY_LENGTH))
        if self._history_length <= 0:
            raise ValueError("Legacy history length must be positive.")
        observation_dim = LEGACY_BALL_OBSERVATION_DIM if include_ball else LEGACY_WALK_OBSERVATION_DIM
        self._history = torch.zeros(
            self.num_envs,
            self._history_length,
            observation_dim,
            device=self.device,
        )
        self._last_append_step = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._skip_next_append = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._allow_noise = False

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._allow_noise = True
        ids = _as_index_tensor(env_ids, self.num_envs, self.device)
        self._history[ids] = 0.0
        self._last_append_step[ids] = -1
        self._skip_next_append[ids] = self._env.common_step_counter == 0
        _capture_pre_action_reset(self._env, ids, self.cfg.params.get("action_name"))
        invalidate_legacy_observation_cache(self._env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        include_ball: bool,
        enable_noise: bool = True,
        history_length: int = LEGACY_HISTORY_LENGTH,
        base_command_name: str = "base_velocity",
        gait_command_name: str = "gait_parameters",
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        action_name: str | None = None,
        action_clip: float = 1.0,
        observation_clip: float = 100.0,
    ) -> torch.Tensor:
        if int(history_length) != self._history_length:
            raise ValueError("Configured history length changed after term initialization.")

        observation = _legacy_policy_observation(
            env,
            include_ball=include_ball,
            enable_noise=enable_noise and self._allow_noise,
            base_command_name=base_command_name,
            gait_command_name=gait_command_name,
            robot_cfg=robot_cfg,
            ball_cfg=ball_cfg,
            action_name=action_name,
            action_clip=action_clip,
            observation_clip=observation_clip,
        )
        gait_term = env.command_manager.get_term(gait_command_name)
        step = int(env.common_step_counter)
        ready = gait_term.last_update_step == step

        skipped = ready & self._skip_next_append
        self._skip_next_append[skipped] = False
        self._last_append_step[skipped] = step

        append_mask = ready & ~self._skip_next_append & (self._last_append_step != step)
        if torch.any(append_mask):
            self._history = update_zero_padded_history(
                self._history,
                observation,
                append_mask=append_mask,
            )
            self._last_append_step[append_mask] = step
        return flatten_history(self._history)


def _legacy_policy_observation(
    env: ManagerBasedRLEnv,
    include_ball: bool,
    enable_noise: bool,
    base_command_name: str,
    gait_command_name: str,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    action_name: str | None,
    action_clip: float,
    observation_clip: float,
) -> torch.Tensor:
    gait_term = env.command_manager.get_term(gait_command_name)
    action_revision = 0
    if action_name is not None:
        action_revision = int(getattr(env.action_manager.get_term(action_name), "observation_revision", 0))
    cache_key = (
        int(env.common_step_counter),
        int(gait_term.observation_revision),
        include_ball,
        enable_noise,
        base_command_name,
        gait_command_name,
        action_name,
        action_revision,
        float(action_clip),
        float(observation_clip),
    )
    cache = getattr(env, _OBSERVATION_CACHE, None)
    if cache is not None and cache.get("key") == cache_key:
        return cache["value"]

    robot: Articulation = env.scene[robot_cfg.name]
    base_command = env.command_manager.get_command(base_command_name)
    gait_parameters = gait_term.command
    scaled_command = compose_legacy_command(base_command, gait_parameters)

    joint_position = (
        robot.data.joint_pos[:, robot_cfg.joint_ids]
        - robot.data.default_joint_pos[:, robot_cfg.joint_ids]
    )
    scaled_joint_velocity = robot.data.joint_vel[:, robot_cfg.joint_ids] * 0.05
    current_action, previous_action = _legacy_action_pair(env, action_name)
    current_action = current_action.clamp(-float(action_clip), float(action_clip))
    previous_action = previous_action.clamp(-float(action_clip), float(action_clip))
    current_action = _apply_reset_action_override(env, current_action, action_clip, action_name)
    ball = ball_position_b(env, robot_cfg, ball_cfg) if include_ball else None
    observation = assemble_legacy_observation(
        projected_gravity=robot.data.projected_gravity_b,
        scaled_command=scaled_command,
        joint_position=joint_position,
        scaled_joint_velocity=scaled_joint_velocity,
        current_action=current_action,
        previous_action=previous_action,
        gait_clock=gait_term.clock,
        heading=robot_heading_w(env, robot_cfg),
        gait_phase=gait_term.gait_phase.unsqueeze(-1),
        ball_position=ball,
    )
    if enable_noise:
        observation = add_legacy_observation_noise(observation, include_ball)
    observation = observation.clamp(-float(observation_clip), float(observation_clip))
    setattr(env, _OBSERVATION_CACHE, {"key": cache_key, "value": observation})
    return observation


def _capture_pre_action_reset(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    action_name: str | None,
) -> None:
    ids = _as_index_tensor(env_ids, env.num_envs, env.device)
    current_action, _ = _legacy_action_pair(env, action_name)
    if not hasattr(env, _RESET_ACTIONS):
        setattr(env, _RESET_ACTIONS, {})
    state_by_source = getattr(env, _RESET_ACTIONS)
    source_key = action_name or "__action_manager__"
    if source_key not in state_by_source:
        state_by_source[source_key] = {
            "action": torch.zeros_like(current_action),
            "mask": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
    state_by_source[source_key]["action"][ids] = current_action[ids]
    state_by_source[source_key]["mask"][ids] = True


def _apply_reset_action_override(
    env: ManagerBasedRLEnv,
    current_action: torch.Tensor,
    action_clip: float,
    action_name: str | None,
) -> torch.Tensor:
    if not hasattr(env, _RESET_ACTIONS):
        return current_action
    source_key = action_name or "__action_manager__"
    state = getattr(env, _RESET_ACTIONS).get(source_key)
    if state is None:
        return current_action
    active = state["mask"] & (env.episode_length_buf == 0)
    result = current_action.clone()
    reset_action = state["action"].clamp(-float(action_clip), float(action_clip))
    result[active] = reset_action[active]
    return result


def invalidate_legacy_observation_cache(env: ManagerBasedRLEnv) -> None:
    """Invalidate cached noisy observations before an external command/action update."""

    setattr(env, _OBSERVATION_CACHE, None)


def compute_legacy_policy_observation(
    env: ManagerBasedRLEnv,
    include_ball: bool,
    enable_noise: bool,
    base_command_name: str,
    gait_command_name: str,
    robot_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    action_name: str | None,
    action_clip: float,
    observation_clip: float,
) -> torch.Tensor:
    """Public adapter used by policy-driven manager action terms."""

    return _legacy_policy_observation(
        env,
        include_ball=include_ball,
        enable_noise=enable_noise,
        base_command_name=base_command_name,
        gait_command_name=gait_command_name,
        robot_cfg=robot_cfg,
        ball_cfg=ball_cfg,
        action_name=action_name,
        action_clip=action_clip,
        observation_clip=observation_clip,
    )


def _legacy_action_pair(
    env: ManagerBasedRLEnv,
    action_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_name is None:
        return env.action_manager.action, env.action_manager.prev_action
    action_term = env.action_manager.get_term(action_name)
    return action_term.policy_action, action_term.prev_policy_action


def _as_index_tensor(
    env_ids: Sequence[int] | None,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def ball_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
):
    """Ball position relative to the robot, expressed in the body frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    position_b = quat_apply_inverse(robot.data.root_quat_w, ball.data.root_pos_w - robot.data.root_pos_w)
    position_b[:, 2] = 0.0
    return position_b


def ball_linear_velocity_w(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")):
    """Ball linear velocity in the fixed world frame used by skill commands."""

    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_lin_vel_w


def ball_angular_velocity_w(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg = SceneEntityCfg("ball")):
    """Ball angular velocity in world frame for critic and state-model inputs."""

    ball: RigidObject = env.scene[ball_cfg.name]
    return ball.data.root_ang_vel_w


def robot_heading_w(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Robot yaw in world frame, matching the original one-value YawSensor."""

    robot: Articulation = env.scene[robot_cfg.name]
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    return yaw.unsqueeze(-1)


def match_local_observation(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
    team_size: int = 2,
    field_length: float = 8.0,
    field_width: float = 5.0,
    team_goal_x: float = 4.0,
    command_obs_scale: tuple[float, float, float] = (3.0, 3.0, 1.0),
    dribble_distance: float = 1.0,
    shoot_distance: float = 0.75,
    shoot_min_forward: float = -0.1,
    shoot_lateral_reach: float = 0.45,
) -> torch.Tensor:
    """Build the archived 34D agent-centric observation for each AS2 actor."""

    if len(robot_names) != 2 * int(team_size):
        raise ValueError("robot_names must contain two equally sized teams")
    robot_data = [env.scene[name] for name in robot_names]
    roots = torch.stack([robot.data.root_pos_w for robot in robot_data], dim=1)
    quats = torch.stack([robot.data.root_quat_w for robot in robot_data], dim=1)
    velocities = torch.stack([robot.data.root_lin_vel_w[:, :2] for robot in robot_data], dim=1)
    yaw_rates = torch.stack([robot.data.root_ang_vel_w[:, 2] for robot in robot_data], dim=1)
    ball = env.scene["ball"]
    ball_xy = ball.data.root_pos_w[:, :2]
    ball_vel = ball.data.root_lin_vel_w[:, :2]
    field_scale = roots.new_tensor((max(float(field_length) * 0.5, 1.0e-6), max(float(field_width) * 0.5, 1.0e-6)))
    field_diag = torch.linalg.vector_norm(field_scale).clamp_min(1.0e-6)
    forward_seed = roots.new_zeros((roots.shape[0], roots.shape[1], 3))
    forward_seed[..., 0] = 1.0
    forward = quat_apply(quats.reshape(-1, 4), forward_seed.reshape(-1, 3)).view_as(forward_seed)[..., :2]
    signs = team_signs(roots.shape[1], team_size, device=roots.device, dtype=roots.dtype)
    signs_xy = signs.view(1, -1, 1)
    forward = signs_xy * forward
    own_xy = signs_xy * (roots[..., :2] - env.scene.env_origins[:, None, :2]) / field_scale
    own_vel = signs_xy * velocities / 3.0
    own_yaw_rate = yaw_rates.unsqueeze(-1) / 3.0
    ball_delta_world = ball_xy[:, None, :] - roots[..., :2]
    ball_delta_local = quat_apply_inverse(
        quats.reshape(-1, 4),
        torch.cat((ball_delta_world, roots.new_zeros((*ball_delta_world.shape[:2], 1))), dim=-1).reshape(-1, 3),
    ).view(roots.shape[0], roots.shape[1], 3)[..., :2]
    # Team 1 attacks -x in the world; expose the same canonical +x frame to
    # both teams' high-level policies.
    ball_rel = signs_xy * ball_delta_world / field_scale
    ball_distance = torch.linalg.vector_norm(ball_delta_world, dim=-1)
    ball_distance_norm = (ball_distance / field_diag).unsqueeze(-1)

    goal_world = env.scene.env_origins[:, None, :2].expand(-1, roots.shape[1], -1).clone()
    team_ids = torch.arange(roots.shape[1], device=roots.device) // int(team_size)
    goal_world[:, team_ids == 0, 0] += float(team_goal_x)
    goal_world[:, team_ids == 1, 0] -= float(team_goal_x)
    goal_rel = signs_xy * (goal_world - roots[..., :2]) / field_scale
    ball_to_goal = goal_world - ball_xy[:, None, :]
    robot_dir = ball_delta_world / ball_distance.clamp_min(1.0e-6).unsqueeze(-1)
    goal_dir = ball_to_goal / torch.linalg.vector_norm(ball_to_goal, dim=-1).clamp_min(1.0e-6).unsqueeze(-1)
    behind_alignment = torch.sum(robot_dir * goal_dir, dim=-1).clamp(-1.0, 1.0)
    local_x = ball_delta_local[..., 0]
    local_y = ball_delta_local[..., 1]
    can_dribble = ball_distance <= float(dribble_distance)
    ball_strikeable = (local_x >= float(shoot_min_forward)) & (torch.abs(local_y) <= float(shoot_lateral_reach))
    can_shoot = (ball_distance <= float(shoot_distance)) & ball_strikeable
    affordance = torch.cat(
        (
            # The legacy affordance uses the ball vector in the robot-local
            # frame, unlike the world-frame ball_rel above.
            ball_delta_local / field_scale,
            ball_distance_norm,
            can_dribble.float().unsqueeze(-1),
            can_shoot.float().unsqueeze(-1),
            behind_alignment.unsqueeze(-1),
        ),
        dim=-1,
    )

    teammate_rel = roots.new_zeros((roots.shape[0], roots.shape[1], 2))
    teammate_mask = roots.new_zeros((roots.shape[0], roots.shape[1], 1))
    opponent_rel = roots.new_zeros((roots.shape[0], roots.shape[1], 2))
    opponent_vel = roots.new_zeros((roots.shape[0], roots.shape[1], 2))
    opponent_mask = roots.new_zeros((roots.shape[0], roots.shape[1], 1))
    for slot in range(roots.shape[1]):
        team = slot // int(team_size)
        teammate_slots = [index for index in range(roots.shape[1]) if index // int(team_size) == team and index != slot]
        opponent_slots = [index for index in range(roots.shape[1]) if index // int(team_size) != team]
        if teammate_slots:
            delta = roots[:, teammate_slots, :2] - roots[:, slot : slot + 1, :2]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            nearest = torch.argmin(distance, dim=-1)
            rows = torch.arange(roots.shape[0], device=roots.device)
            teammate_rel[:, slot] = signs[slot] * delta[rows, nearest] / field_scale
            teammate_mask[:, slot, 0] = 1.0
        if opponent_slots:
            delta = roots[:, opponent_slots, :2] - roots[:, slot : slot + 1, :2]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            nearest = torch.argmin(distance, dim=-1)
            rows = torch.arange(roots.shape[0], device=roots.device)
            opponent_rel[:, slot] = signs[slot] * delta[rows, nearest] / field_scale
            opponent_vel[:, slot] = signs[slot] * velocities[:, opponent_slots, :][rows, nearest] / 3.0
            opponent_mask[:, slot, 0] = 1.0

    skill_ids = []
    commands = []
    stored_attacker = []
    for slot in range(roots.shape[1]):
        term = env.action_manager.get_term(f"skill_policy_{slot}")
        skill_ids.append(term.skill_ids)
        commands.append(term.skill_commands)
        stored_attacker.append(term.attacker_mask)
    skill_ids_tensor = torch.stack(skill_ids, dim=1)
    stored_attacker_tensor = torch.stack(stored_attacker, dim=1)
    attacker_role = torch.zeros_like(teammate_mask)
    # The current Isaac Gym observation reuses the old constant teammate bit
    # as a role bit without changing the archived 34D interface. During the
    # first observation after reset, before action terms have selected roles,
    # nearest-to-ball is the deterministic fallback.
    for team in range(2):
        start = team * int(team_size)
        end = start + int(team_size)
        stored = stored_attacker_tensor[:, start:end]
        valid = stored.long().sum(dim=1) == 1
        nearest = torch.nn.functional.one_hot(
            ball_distance[:, start:end].argmin(dim=1), num_classes=int(team_size)
        ).bool()
        selected = torch.where(valid[:, None], stored, nearest)
        attacker_role[:, start:end, 0] = selected.float()
    command_tensor = torch.stack(commands, dim=1) / roots.new_tensor(command_obs_scale).clamp_min(1.0e-6)
    skill_one_hot = torch.nn.functional.one_hot(skill_ids_tensor, num_classes=3).float()
    observation = torch.cat(
        (
            own_xy,
            forward,
            own_vel,
            own_yaw_rate,
            ball_rel,
            signs_xy * ball_vel[:, None, :].expand(-1, roots.shape[1], -1) / 5.0,
            ball_distance_norm,
            goal_rel,
            teammate_rel,
            attacker_role,
            opponent_rel,
            opponent_vel,
            opponent_mask,
            affordance,
            skill_one_hot,
            mirror_high_level_commands(command_tensor, skill_ids_tensor),
        ),
        dim=-1,
    )
    if observation.shape[-1] != 34:
        raise RuntimeError(f"Match local observation has {observation.shape[-1]} values, expected 34")
    return torch.nan_to_num(observation, nan=0.0, posinf=100.0, neginf=-100.0)
