"""Single-AS2 smoke task for frozen walking, dribbling, and shooting policies."""

from isaaclab.utils import configclass

from ....policies import REPRODUCTION_COMMAND_SCALES
from ..as2_dribble.dribble_env_cfg import AS2DribbleFlatEnvCfg
from ..football.frozen_skill_cfg import (
    AS2FrozenSkillActionsCfg,
    AS2FrozenSkillObservationsCfg,
    make_frozen_skill_action_cfg,
)


AS2SkillActionsCfg = AS2FrozenSkillActionsCfg
AS2SkillObservationsCfg = AS2FrozenSkillObservationsCfg


@configclass
class AS2CoordinatorActionsCfg:
    """Coordinator action term using the archived high-level command scales."""

    skill_policy = make_frozen_skill_action_cfg(
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
    )


@configclass
class AS2SkillFlatEnvCfg(AS2DribbleFlatEnvCfg):
    """Checkpoint-inference integration task; not yet a coordinator training task."""

    actions: AS2SkillActionsCfg = AS2SkillActionsCfg()
    observations: AS2SkillObservationsCfg = AS2SkillObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 256
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.gait_parameters.resampling_time_range = (1.0e6, 1.0e6)


@configclass
class AS2SkillFlatEnvCfg_PLAY(AS2SkillFlatEnvCfg):
    """Small deterministic scene for frozen-policy integration smoke tests."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.observations.legacy_policy.legacy_observation.params["enable_noise"] = False
        self.observations.legacy_history.legacy_history.params["enable_noise"] = False
        self.events.physics_material = None
        self.events.ball_physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None


@configclass
class AS2CoordinatorFlatEnvCfg(AS2DribbleFlatEnvCfg):
    """Low-level manager scene with the reproduction coordinator contract."""

    actions: AS2CoordinatorActionsCfg = AS2CoordinatorActionsCfg()
    observations: AS2FrozenSkillObservationsCfg = AS2FrozenSkillObservationsCfg()
    macro_control_interval: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 256
        self.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.gait_parameters.resampling_time_range = (1.0e6, 1.0e6)


@configclass
class AS2CoordinatorFlatEnvCfg_PLAY(AS2CoordinatorFlatEnvCfg):
    """Small deterministic coordinator-rate smoke scene."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.observations.legacy_policy.legacy_observation.params["enable_noise"] = False
        self.observations.legacy_history.legacy_history.params["enable_noise"] = False
        self.events.physics_material = None
        self.events.ball_physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
