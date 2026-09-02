"""Reusable manager configuration for trusted frozen AS2 skill policies."""

from __future__ import annotations

from isaaclab.utils import configclass

from ....policies import (
    DRIBBLE_SKILL_ID,
    SHOOT_SKILL_ID,
    LEGACY_WRAPPER_COMMAND_SCALES,
    ball_skill_command_frame,
    checkpoint_root,
)
from ..as2_dribble.dribble_env_cfg import AS2DribbleObservationsCfg
from . import mdp


def make_frozen_skill_action_cfg(
    *,
    asset_name: str = "robot",
    action_term_name: str = "skill_policy",
    base_command_name: str = "base_velocity",
    gait_command_name: str = "gait_parameters",
    fixed_skill_id: int | None = None,
    command_scales: tuple[tuple[float, float, float], ...] = LEGACY_WRAPPER_COMMAND_SCALES,
    action_history_semantics: str = "duplicate_previous_policy_output",
    geometric_skill_fallback: bool = False,
    ball_command_input_frame: str | None = None,
    ball_skill_checkpoint_frame: str | None = None,
    role_aware_fallback: bool = False,
    team_robot_names: tuple[str, ...] = (),
    team_index: int = 0,
    team_slot: int = 0,
) -> mdp.FrozenSkillPolicyActionCfg:
    """Build one frozen-policy action term from the curated checkpoint bundle."""

    root = checkpoint_root()
    checkpoint_frame = ball_skill_checkpoint_frame or ball_skill_command_frame()
    input_frame = ball_command_input_frame
    if input_frame is None:
        input_frame = (
            checkpoint_frame
            if fixed_skill_id in (DRIBBLE_SKILL_ID, SHOOT_SKILL_ID)
            else "world"
        )
    return mdp.FrozenSkillPolicyActionCfg(
        asset_name=asset_name,
        action_term_name=action_term_name,
        base_command_name=base_command_name,
        gait_command_name=gait_command_name,
        joint_names=list(mdp.LEGACY_JOINT_NAMES),
        preserve_order=True,
        joint_scale={
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
        },
        fixed_skill_id=fixed_skill_id,
        command_scales=command_scales,
        action_history_semantics=action_history_semantics,
        geometric_skill_fallback=geometric_skill_fallback,
        ball_command_input_frame=input_frame,
        ball_skill_command_frame=checkpoint_frame,
        role_aware_fallback=role_aware_fallback,
        team_robot_names=team_robot_names,
        team_index=team_index,
        team_slot=team_slot,
        walk_body_path=str(root / "walk" / "body_latest.jit"),
        walk_adaptation_path=str(root / "walk" / "adaptation_module_latest.jit"),
        dribble_body_path=str(root / "dribble" / "body_latest.jit"),
        dribble_adaptation_path=str(root / "dribble" / "adaptation_module_latest.jit"),
        shoot_body_path=str(root / "shoot" / "body_latest.jit"),
        shoot_adaptation_path=str(root / "shoot" / "adaptation_module_latest.jit"),
    )


@configclass
class AS2FrozenSkillActionsCfg:
    """Hybrid skill index and parameters routed through the frozen bundle."""

    skill_policy: mdp.FrozenSkillPolicyActionCfg = make_frozen_skill_action_cfg()


@configclass
class AS2FrozenSkillObservationsCfg(AS2DribbleObservationsCfg):
    """Expose low-level policy actions, rather than coordinator inputs, to legacy groups."""

    def __post_init__(self) -> None:
        self.legacy_policy.legacy_observation.params["action_name"] = "skill_policy"
        self.legacy_history.legacy_history.params["action_name"] = "skill_policy"


__all__ = [
    "AS2FrozenSkillActionsCfg",
    "AS2FrozenSkillObservationsCfg",
    "make_frozen_skill_action_cfg",
]
