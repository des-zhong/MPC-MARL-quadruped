"""Manager action terms for composing frozen low-level football skills."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from .....policies import (
    DRIBBLE_SKILL_ID,
    SHOOT_SKILL_ID,
    WALK_SKILL_ID,
    FrozenPolicySpec,
    FrozenSkillPolicySet,
    HYBRID_COORDINATOR_ACTION_DIM,
    LEGACY_WRAPPER_COMMAND_SCALES,
    decode_hybrid_action,
    decode_fixed_skill_action,
    observation_action_pair,
)
from .legacy_contract import LEGACY_BALL_OBSERVATION_DIM, LEGACY_HISTORY_LENGTH
from .legacy_contract import flatten_history, update_zero_padded_history
from .observations import compute_legacy_policy_observation, invalidate_legacy_observation_cache
from ..command_frames import world_xy_to_body_xy

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FrozenSkillPolicyAction(ActionTerm):
    """Map one hybrid action through frozen skills to 12 joint targets.

    The trainable coordinator action is
    ``[skill_index, parameter_x, parameter_y, parameter_yaw]``. Fixed-skill
    low-level fixtures retain their three continuous command parameters.
    """

    cfg: FrozenSkillPolicyActionCfg

    def __init__(self, cfg: FrozenSkillPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names,
            preserve_order=cfg.preserve_order,
        )
        if len(self._joint_ids) != int(cfg.expected_joint_count):
            raise ValueError(
                f"Frozen skill action resolved {len(self._joint_ids)} joints, "
                f"expected {cfg.expected_joint_count}."
            )

        if cfg.fixed_skill_id is not None and not WALK_SKILL_ID <= int(cfg.fixed_skill_id) <= SHOOT_SKILL_ID:
            raise ValueError(f"Unknown fixed skill id: {cfg.fixed_skill_id}")
        valid_frames = {"body", "world"}
        if cfg.ball_command_input_frame not in valid_frames:
            raise ValueError(
                f"ball_command_input_frame must be one of {sorted(valid_frames)}, "
                f"got {cfg.ball_command_input_frame!r}"
            )
        if cfg.ball_skill_command_frame not in valid_frames:
            raise ValueError(
                f"ball_skill_command_frame must be one of {sorted(valid_frames)}, "
                f"got {cfg.ball_skill_command_frame!r}"
            )
        if cfg.ball_command_input_frame == "body" and cfg.ball_skill_command_frame == "world":
            raise ValueError("body input with a world-frame ball checkpoint is not supported")
        self._action_dim = 3 if cfg.fixed_skill_id is not None else HYBRID_COORDINATOR_ACTION_DIM
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._policy_action = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._prev_policy_action = torch.zeros_like(self._policy_action)
        self._processed_actions = torch.zeros_like(self._policy_action)
        self._history = torch.zeros(
            self.num_envs,
            cfg.history_length,
            cfg.full_observation_dim,
            device=self.device,
        )
        self.skill_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.requested_skill_ids = torch.zeros_like(self.skill_ids)
        self.invalid_skill_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.skill_transition_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.skill_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.invalid_input_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.attacker_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.preserve_external_actions = False
        self._observation_revision = 0
        observation_action_pair(self._policy_action, self._prev_policy_action, cfg.action_history_semantics)

        self._scale = self._resolve_joint_scale(cfg.joint_scale)
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._robot_cfg = SceneEntityCfg(cfg.asset_name, joint_ids=self._joint_ids)
        self._ball_cfg = SceneEntityCfg(cfg.ball_name)
        self._command_scales = torch.tensor(cfg.command_scales, dtype=torch.float, device=self.device)
        self._fixed_gait_parameters = torch.tensor(
            cfg.fixed_gait_parameters,
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self._policies = FrozenSkillPolicySet.load(
            {
                WALK_SKILL_ID: FrozenPolicySpec(
                    "walk",
                    body_path=cfg.walk_body_path,
                    adaptation_path=cfg.walk_adaptation_path,
                    action_clip=cfg.walk_action_clip,
                    expected_action_dim=len(self._joint_ids),
                ),
                DRIBBLE_SKILL_ID: FrozenPolicySpec(
                    "dribble",
                    body_path=cfg.dribble_body_path,
                    adaptation_path=cfg.dribble_adaptation_path,
                    action_clip=cfg.dribble_action_clip,
                    expected_action_dim=len(self._joint_ids),
                ),
                SHOOT_SKILL_ID: FrozenPolicySpec(
                    "shoot",
                    body_path=cfg.shoot_body_path,
                    adaptation_path=cfg.shoot_adaptation_path,
                    action_clip=cfg.shoot_action_clip,
                    expected_action_dim=len(self._joint_ids),
                ),
            },
            device=self.device,
            full_observation_dim=cfg.full_observation_dim,
            history_length=cfg.history_length,
            object_sensor_width=cfg.object_sensor_width,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def policy_action(self) -> torch.Tensor:
        """Most recent 12D low-level action exposed to legacy observations."""

        return self._policy_action

    @property
    def prev_policy_action(self) -> torch.Tensor:
        """Action exposed in the legacy previous-action observation slice."""

        return self._prev_policy_action

    @property
    def observation_revision(self) -> int:
        return self._observation_revision

    @property
    def policy_history(self) -> torch.Tensor:
        """Flattened 75D history consumed by the policy router."""

        return flatten_history(self._history)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = torch.nan_to_num(
            actions.to(self.device),
            nan=0.0,
            posinf=self.cfg.input_clip,
            neginf=-self.cfg.input_clip,
        ).clamp(-self.cfg.input_clip, self.cfg.input_clip)
        if self.cfg.fixed_skill_id is None:
            skill_ids, commands, invalid = decode_hybrid_action(
                actions.to(self.device),
                self._command_scales,
                input_clip=self.cfg.input_clip,
            )
            requested_skill_ids = skill_ids.clone()
        else:
            requested_skill_ids = torch.full(
                (self.num_envs,), int(self.cfg.fixed_skill_id), dtype=torch.long, device=self.device
            )
            skill_ids, commands, invalid = decode_fixed_skill_action(
                actions.to(self.device),
                int(self.cfg.fixed_skill_id),
                self._command_scales,
                input_clip=self.cfg.input_clip,
            )
        invalid_skill_mask = invalid.clone() if self.cfg.fixed_skill_id is None else torch.zeros_like(invalid)
        if self.cfg.geometric_skill_fallback and self.cfg.fixed_skill_id is None:
            skill_ids, commands, geometric_invalid = self._apply_geometric_skill_fallback(
                requested_skill_ids,
                skill_ids,
                commands,
            )
            invalid_skill_mask |= geometric_invalid
        self.skill_transition_mask[:] = requested_skill_ids != self.requested_skill_ids
        self.requested_skill_ids[:] = requested_skill_ids
        self.invalid_skill_mask[:] = invalid_skill_mask
        self.skill_ids[:] = skill_ids
        self.skill_commands[:] = commands
        self.invalid_input_mask[:] = invalid
        self._inject_commands(self._execution_commands(commands, skill_ids))

        # HighLevelSkillWrapper copies ``last_low_level_actions`` from
        # ``low_level_actions`` immediately before it builds each low-level
        # observation.  Existing coordinator checkpoints therefore saw the
        # preceding policy output in both the current-action and
        # previous-action slices.  Keep the conventional successive-output
        # mode available for new experiments, but make the compatibility mode
        # explicit and testable.
        _, observed_previous_action = observation_action_pair(
            self._policy_action,
            self._prev_policy_action,
            self.cfg.action_history_semantics,
        )
        self._prev_policy_action[:] = observed_previous_action

        invalidate_legacy_observation_cache(self._env)
        observation = compute_legacy_policy_observation(
            self._env,
            include_ball=True,
            enable_noise=False,
            base_command_name=self.cfg.base_command_name,
            gait_command_name=self.cfg.gait_command_name,
            robot_cfg=self._robot_cfg,
            ball_cfg=self._ball_cfg,
            action_name=self.cfg.action_term_name,
            action_clip=max(
                self.cfg.walk_action_clip,
                self.cfg.dribble_action_clip,
                self.cfg.shoot_action_clip,
            ),
            observation_clip=self.cfg.observation_clip,
        )
        self._history = update_zero_padded_history(self._history, observation)
        next_policy_action = self._policies.route(flatten_history(self._history), self.skill_ids)
        next_policy_action[self.invalid_input_mask] = 0.0
        if self.cfg.action_history_semantics == "successive_policy_outputs":
            self._prev_policy_action[:] = self._policy_action
        self._policy_action[:] = next_policy_action
        self._processed_actions[:] = self._policy_action * self._scale + self._offset
        self._observation_revision += 1
        invalidate_legacy_observation_cache(self._env)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._policy_action[ids] = 0.0
        self._prev_policy_action[ids] = 0.0
        self._processed_actions[ids] = self._offset[ids]
        self._history[ids] = 0.0
        reset_skill_id = WALK_SKILL_ID if self.cfg.fixed_skill_id is None else int(self.cfg.fixed_skill_id)
        self.skill_ids[ids] = reset_skill_id
        self.requested_skill_ids[ids] = reset_skill_id
        self.invalid_skill_mask[ids] = False
        self.skill_transition_mask[ids] = False
        self.skill_commands[ids] = 0.0
        self.invalid_input_mask[ids] = False
        self.attacker_mask[ids] = False
        self._clear_attacker_assignment(ids)
        self._observation_revision += 1
        invalidate_legacy_observation_cache(self._env)

    def _apply_geometric_skill_fallback(
        self,
        requested_skill_ids: torch.Tensor,
        skill_ids: torch.Tensor,
        commands: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mirror the legacy strikeability fallback for one robot term."""

        if self.preserve_external_actions:
            self.attacker_mask[:] = self._stable_attacker_mask() if self.cfg.team_robot_names else False
            return skill_ids, commands, torch.zeros_like(skill_ids, dtype=torch.bool)

        robot: Articulation = self._asset
        ball = self._env.scene[self.cfg.ball_name]
        ball_delta_local = quat_apply_inverse(
            robot.data.root_quat_w,
            ball.data.root_pos_w - robot.data.root_pos_w,
        )[:, :2]
        distance = torch.linalg.vector_norm(ball_delta_local, dim=-1)
        can_dribble = distance <= float(self.cfg.dribble_skill_distance)
        can_shoot = (distance <= float(self.cfg.shoot_skill_distance)) & (
            ball_delta_local[:, 0] >= float(self.cfg.shoot_min_forward)
        ) & (torch.abs(ball_delta_local[:, 1]) <= float(self.cfg.shoot_lateral_reach))
        dribble_invalid = (requested_skill_ids == DRIBBLE_SKILL_ID) & ~can_dribble
        shoot_invalid = (requested_skill_ids == SHOOT_SKILL_ID) & ~can_shoot
        shoot_to_dribble = shoot_invalid & can_dribble
        role_conflict = torch.zeros_like(dribble_invalid)
        support_override = torch.zeros_like(dribble_invalid)
        attacker_walk_conflict = torch.zeros_like(dribble_invalid)
        if self.cfg.role_aware_fallback and self.cfg.team_robot_names:
            self.attacker_mask[:] = self._stable_attacker_mask()
            support_override = ~self.attacker_mask
            role_conflict = support_override & (requested_skill_ids != WALK_SKILL_ID)
            attacker_walk_conflict = (
                self.attacker_mask & (requested_skill_ids == WALK_SKILL_ID) & can_dribble
            )
            role_conflict |= attacker_walk_conflict
        force_walk = dribble_invalid | (shoot_invalid & ~can_dribble) | support_override
        final_skill_ids = skill_ids.clone()
        final_skill_ids[shoot_to_dribble] = DRIBBLE_SKILL_ID
        final_skill_ids[attacker_walk_conflict] = DRIBBLE_SKILL_ID
        final_skill_ids[force_walk] = WALK_SKILL_ID

        final_scales = self._command_scales[final_skill_ids]
        commands = torch.maximum(torch.minimum(commands, final_scales), -final_scales)
        walk_scale = self._command_scales[WALK_SKILL_ID]
        direction = ball_delta_local / distance.clamp_min(1.0e-6).unsqueeze(-1)
        walk_commands = torch.zeros_like(commands)
        walk_commands[:, :2] = torch.clamp(
            direction * float(self.cfg.approach_walk_speed),
            -walk_scale[:2],
            walk_scale[:2],
        )
        walk_commands[:, 2] = torch.clamp(
            1.5 * torch.atan2(ball_delta_local[:, 1], ball_delta_local[:, 0]),
            -walk_scale[2],
            walk_scale[2],
        )
        commands[force_walk] = walk_commands[force_walk]
        if torch.any(support_override):
            support_commands = self._support_walk_commands()
            commands[support_override] = support_commands[support_override]
        if torch.any(attacker_walk_conflict):
            goalward_commands = self._goalward_ball_commands(final_skill_ids)
            commands[attacker_walk_conflict] = goalward_commands[attacker_walk_conflict]
        commands[:, 2] = torch.where(
            final_skill_ids == SHOOT_SKILL_ID,
            torch.zeros_like(commands[:, 2]),
            commands[:, 2],
        )
        return final_skill_ids, commands, dribble_invalid | shoot_invalid | role_conflict

    def _execution_commands(self, commands: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        """Adapt coordinator commands to the frozen checkpoint frame contract."""

        if self.cfg.ball_command_input_frame == self.cfg.ball_skill_command_frame:
            return commands
        converted = commands.clone()
        ball_skill = skill_ids != WALK_SKILL_ID
        if torch.any(ball_skill):
            converted[ball_skill, :2] = world_xy_to_body_xy(
                commands[ball_skill, :2], self._asset.data.root_quat_w[ball_skill]
            )
        return converted

    def _stable_attacker_mask(self) -> torch.Tensor:
        """Select one team attacker with the Isaac Gym distance hysteresis."""

        names = tuple(self.cfg.team_robot_names)
        if self.cfg.team_slot < 0 or self.cfg.team_slot >= len(names):
            raise ValueError("team_slot must index team_robot_names")
        ball_xy = self._env.scene[self.cfg.ball_name].data.root_pos_w[:, :2]
        positions = torch.stack([self._env.scene[name].data.root_pos_w[:, :2] for name in names], dim=1)
        distances = torch.linalg.vector_norm(positions - ball_xy[:, None, :], dim=-1)
        nearest = distances.argmin(dim=1)
        state_name = "_football_attacker_slots"
        assignments = getattr(self._env, state_name, None)
        if assignments is None:
            assignments = torch.full((self.num_envs, 2), -1, dtype=torch.long, device=self.device)
            setattr(self._env, state_name, assignments)
        rows = torch.arange(self.num_envs, device=self.device)
        previous = assignments[:, int(self.cfg.team_index)]
        previous_valid = (previous >= 0) & (previous < len(names))
        previous_safe = previous.clamp(0, len(names) - 1)
        switch = (~previous_valid) | (
            distances[rows, nearest] + float(self.cfg.attacker_switch_margin)
            < distances[rows, previous_safe]
        )
        chosen = torch.where(switch, nearest, previous_safe)
        assignments[:, int(self.cfg.team_index)] = chosen
        return chosen == int(self.cfg.team_slot)

    def _clear_attacker_assignment(self, env_ids) -> None:
        assignments = getattr(self._env, "_football_attacker_slots", None)
        if assignments is not None and self.cfg.team_robot_names:
            assignments[env_ids, int(self.cfg.team_index)] = -1

    def _support_walk_commands(self) -> torch.Tensor:
        robot_xy = self._asset.data.root_pos_w[:, :2]
        ball_xy = self._env.scene[self.cfg.ball_name].data.root_pos_w[:, :2]
        origins = self._env.scene.env_origins[:, :2]
        sign = 1.0 if int(self.cfg.team_index) == 0 else -1.0
        goal = origins.clone()
        goal[:, 0] += sign * float(self.cfg.team_goal_x)
        goal_direction = goal - ball_xy
        goal_direction /= torch.linalg.vector_norm(goal_direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
        lateral = torch.stack((-goal_direction[:, 1], goal_direction[:, 0]), dim=-1)
        side_value = torch.sum((robot_xy - ball_xy) * lateral, dim=-1)
        default_side = 1.0 if int(self.cfg.team_slot) % 2 == 0 else -1.0
        side = torch.where(
            torch.abs(side_value) > 0.1,
            torch.sign(side_value),
            torch.full_like(side_value, default_side),
        )
        target = (
            ball_xy - float(self.cfg.support_depth) * goal_direction
            + side[:, None] * float(self.cfg.support_lateral) * lateral
        )
        lower = origins + target.new_tensor(
            (-0.5 * self.cfg.field_length + self.cfg.field_margin,
             -0.5 * self.cfg.field_width + self.cfg.field_margin)
        )
        upper = origins + target.new_tensor(
            (0.5 * self.cfg.field_length - self.cfg.field_margin,
             0.5 * self.cfg.field_width - self.cfg.field_margin)
        )
        target = torch.minimum(torch.maximum(target, lower), upper)
        body_delta = world_xy_to_body_xy(target - robot_xy, self._asset.data.root_quat_w)
        distance = torch.linalg.vector_norm(body_delta, dim=-1)
        direction = body_delta / distance.clamp_min(1.0e-6).unsqueeze(-1)
        commands = torch.zeros_like(self.skill_commands)
        speed = torch.minimum(distance, torch.full_like(distance, float(self.cfg.support_walk_speed)))
        commands[:, :2] = direction * speed.unsqueeze(-1)
        ball_body = world_xy_to_body_xy(ball_xy - robot_xy, self._asset.data.root_quat_w)
        commands[:, 2] = 1.5 * torch.atan2(ball_body[:, 1], ball_body[:, 0])
        commands[distance <= float(self.cfg.support_command_deadband)] = 0.0
        scale = self._command_scales[WALK_SKILL_ID]
        return torch.maximum(torch.minimum(commands, scale), -scale)

    def _goalward_ball_commands(self, skill_ids: torch.Tensor) -> torch.Tensor:
        ball_xy = self._env.scene[self.cfg.ball_name].data.root_pos_w[:, :2]
        goal = self._env.scene.env_origins[:, :2].clone()
        goal[:, 0] += (1.0 if int(self.cfg.team_index) == 0 else -1.0) * float(self.cfg.team_goal_x)
        direction = goal - ball_xy
        direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
        commands = torch.zeros_like(self.skill_commands)
        scale = self._command_scales[skill_ids]
        commands[:, :2] = direction * scale[:, :2]
        return commands

    def _inject_commands(self, commands: torch.Tensor) -> None:
        base_term = self._env.command_manager.get_term(self.cfg.base_command_name)
        base_term.command[:] = commands
        if hasattr(base_term, "is_standing_env"):
            base_term.is_standing_env[:] = False
        gait_term = self._env.command_manager.get_term(self.cfg.gait_command_name)
        gait_term.set_parameters(self._fixed_gait_parameters)

    def _resolve_joint_scale(self, scale: float | dict[str, float]) -> torch.Tensor:
        if isinstance(scale, (float, int)):
            return torch.full(
                (self.num_envs, len(self._joint_ids)),
                float(scale),
                device=self.device,
            )
        if isinstance(scale, dict):
            result = torch.ones(self.num_envs, len(self._joint_ids), device=self.device)
            indices, _, values = string_utils.resolve_matching_names_values(scale, self._joint_names)
            result[:, indices] = torch.tensor(values, dtype=torch.float, device=self.device)
            return result
        raise ValueError(f"Unsupported joint scale type: {type(scale)}")


@configclass
class FrozenSkillPolicyActionCfg(ActionTermCfg):
    """Configuration for a trusted three-skill TorchScript action term."""

    class_type: type = FrozenSkillPolicyAction
    joint_names: list[str] = MISSING
    preserve_order: bool = True
    expected_joint_count: int = 12
    joint_scale: float | dict[str, float] = 1.0
    ball_name: str = "ball"
    # Walking is always body-frame. These fields apply only to dribble/shoot.
    # The curated reproduction bundle predates metadata and stays "world";
    # newly trained Isaac Gym ball skills should set the checkpoint side to
    # "body" while match/coordinator input remains "world".
    ball_command_input_frame: str = "world"
    ball_skill_command_frame: str = "world"
    action_term_name: str = "skill_policy"
    base_command_name: str = "base_velocity"
    gait_command_name: str = "gait_parameters"
    history_length: int = LEGACY_HISTORY_LENGTH
    full_observation_dim: int = LEGACY_BALL_OBSERVATION_DIM
    object_sensor_width: int = 3
    input_clip: float = 10.0
    observation_clip: float = 100.0
    action_history_semantics: str = "duplicate_previous_policy_output"
    fixed_skill_id: int | None = None
    geometric_skill_fallback: bool = False
    role_aware_fallback: bool = False
    team_robot_names: tuple[str, ...] = ()
    team_index: int = 0
    team_slot: int = 0
    attacker_switch_margin: float = 0.15
    support_command_deadband: float = 0.08
    support_depth: float = 0.5
    support_lateral: float = 1.2
    support_walk_speed: float = 0.75
    field_length: float = 8.0
    field_width: float = 5.0
    field_margin: float = 0.35
    team_goal_x: float = 4.0
    dribble_skill_distance: float = 1.0
    shoot_skill_distance: float = 0.75
    shoot_min_forward: float = -0.1
    shoot_lateral_reach: float = 0.45
    approach_walk_speed: float = 0.9
    command_scales: tuple[tuple[float, float, float], ...] = LEGACY_WRAPPER_COMMAND_SCALES
    fixed_gait_parameters: tuple[float, ...] = (
        0.0,
        3.0,
        0.5,
        0.0,
        0.0,
        0.5,
        0.09,
        0.0,
        0.0,
        0.05,
        0.05,
        0.005,
    )
    walk_body_path: str = MISSING
    walk_adaptation_path: str = MISSING
    walk_action_clip: float = 1.0
    dribble_body_path: str = MISSING
    dribble_adaptation_path: str = MISSING
    dribble_action_clip: float = 1.0
    shoot_body_path: str = MISSING
    shoot_adaptation_path: str = MISSING
    shoot_action_clip: float = 1.0
