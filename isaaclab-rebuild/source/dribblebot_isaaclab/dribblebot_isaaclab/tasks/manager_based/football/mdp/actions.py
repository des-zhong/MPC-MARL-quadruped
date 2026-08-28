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
    LEGACY_WRAPPER_COMMAND_SCALES,
    decode_high_level_action,
    decode_fixed_skill_action,
    observation_action_pair,
)
from .legacy_contract import LEGACY_BALL_OBSERVATION_DIM, LEGACY_HISTORY_LENGTH
from .legacy_contract import flatten_history, update_zero_padded_history
from .observations import compute_legacy_policy_observation, invalidate_legacy_observation_cache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FrozenSkillPolicyAction(ActionTerm):
    """Map one 6D coordinator action through frozen skills to 12 joint targets."""

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
        self._action_dim = 3 if cfg.fixed_skill_id is not None else 6
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
        self.skill_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.invalid_input_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
            safe_action = torch.nan_to_num(
                actions.to(self.device),
                nan=0.0,
                posinf=self.cfg.input_clip,
                neginf=-self.cfg.input_clip,
            ).clamp(-self.cfg.input_clip, self.cfg.input_clip)
            requested_skill_ids = torch.argmax(safe_action[:, :3], dim=-1)
            skill_ids, commands, invalid = decode_high_level_action(
                actions.to(self.device),
                self._command_scales,
                input_clip=self.cfg.input_clip,
            )
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
        invalid_skill_mask = torch.zeros_like(invalid)
        if self.cfg.geometric_skill_fallback and self.cfg.fixed_skill_id is None:
            skill_ids, commands, invalid_skill_mask = self._apply_geometric_skill_fallback(
                requested_skill_ids,
                skill_ids,
                commands,
            )
        self.requested_skill_ids[:] = requested_skill_ids
        self.invalid_skill_mask[:] = invalid_skill_mask
        self.skill_ids[:] = skill_ids
        self.skill_commands[:] = commands
        self.invalid_input_mask[:] = invalid
        self._inject_commands(commands)

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
        self.skill_commands[ids] = 0.0
        self.invalid_input_mask[ids] = False
        self._observation_revision += 1
        invalidate_legacy_observation_cache(self._env)

    def _apply_geometric_skill_fallback(
        self,
        requested_skill_ids: torch.Tensor,
        skill_ids: torch.Tensor,
        commands: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mirror the legacy strikeability fallback for one robot term."""

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
        force_walk = dribble_invalid | (shoot_invalid & ~can_dribble)
        final_skill_ids = skill_ids.clone()
        final_skill_ids[shoot_to_dribble] = DRIBBLE_SKILL_ID
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
        commands[:, 2] = torch.where(
            final_skill_ids == SHOOT_SKILL_ID,
            torch.zeros_like(commands[:, 2]),
            commands[:, 2],
        )
        return final_skill_ids, commands, dribble_invalid | shoot_invalid

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
