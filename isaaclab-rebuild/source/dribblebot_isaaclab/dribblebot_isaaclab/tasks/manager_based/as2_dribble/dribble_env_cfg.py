"""Manager-based AS2 dribbling task with an explicit football scene."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    MySceneCfg as VelocitySceneCfg,
    RewardsCfg as VelocityRewardsCfg,
    TerminationsCfg as VelocityTerminationsCfg,
)

from ....assets import SOCCER_BALL_CFG
from ..as2_velocity.velocity_env_cfg import (
    AS2VelocityEventCfg,
    AS2VelocityFlatEnvCfg,
    AS2VelocityObservationsCfg,
    legacy_observation_params,
)
from ..football import mdp


@configclass
class AS2DribbleSceneCfg(VelocitySceneCfg):
    """Locomotion scene extended with a dynamic soccer ball."""

    ball: RigidObjectCfg = SOCCER_BALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Ball")


@configclass
class AS2DribbleObservationsCfg(AS2VelocityObservationsCfg):
    """Policy observations for the dribbling skill."""

    @configclass
    class PolicyCfg(AS2VelocityObservationsCfg.PolicyCfg):
        # These were privileged sensors in the Isaac Gym training contract.
        base_lin_vel = None
        ball_position_b = ObsTerm(
            func=mdp.ball_position_b,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        robot_heading_w = ObsTerm(func=mdp.robot_heading_w)

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        ball_linear_velocity_w = ObsTerm(
            func=mdp.ball_linear_velocity_w,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LegacyPolicyCfg(ObsGroup):
        legacy_observation = ObsTerm(
            func=mdp.LegacyPolicyObservation,
            params={
                **legacy_observation_params(include_ball=True),
                "ball_cfg": SceneEntityCfg("ball"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LegacyHistoryCfg(ObsGroup):
        legacy_history = ObsTerm(
            func=mdp.LegacyHistoryObservation,
            params={
                **legacy_observation_params(include_ball=True),
                "ball_cfg": SceneEntityCfg("ball"),
                "history_length": mdp.LEGACY_HISTORY_LENGTH,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    legacy_policy: LegacyPolicyCfg = LegacyPolicyCfg()
    legacy_history: LegacyHistoryCfg = LegacyHistoryCfg()


@configclass
class AS2DribbleEventCfg(AS2VelocityEventCfg):
    """Robot randomization plus ball material and relative reset events."""

    ball_collider_offsets = EventTerm(
        func=mdp.set_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "rest_offset": 0.0,
            "contact_offset": 0.01,
        },
    )

    ball_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.7, 4.0),
            "dynamic_friction_range": (0.7, 4.0),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 64,
        },
    )
    reset_ball = EventTerm(
        func=mdp.reset_ball_in_front,
        mode="reset",
        params={
            "forward_range": (0.35, 0.80),
            "lateral_range": (-0.35, 0.35),
            "ball_height": 0.10,
            "velocity_range": (-0.15, 0.15),
        },
    )


@configclass
class AS2DribbleRewardsCfg(VelocityRewardsCfg):
    """Locomotion regularizers plus ball-control objectives."""

    track_ball_velocity_xy_exp = RewTerm(
        func=mdp.track_ball_velocity_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.5, "scale_xy": (1.5, 1.5)},
    )
    track_ball_direction = RewTerm(
        func=mdp.track_ball_direction,
        weight=4.0,
        params={"command_name": "base_velocity"},
    )
    ball_setup_position_exp = RewTerm(
        func=mdp.ball_setup_position_exp,
        weight=4.0,
        params={"target_forward": 0.35, "target_lateral": 0.0, "position_gain": 10.0},
    )
    robot_ball_command_alignment_exp = RewTerm(
        func=mdp.robot_ball_command_alignment_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "gain": 2.0},
    )


@configclass
class AS2DribbleTerminationsCfg(VelocityTerminationsCfg):
    """Base-contact termination plus football recovery limits."""

    ball_out_of_bounds = DoneTerm(
        func=mdp.ball_out_of_bounds,
        params={"half_extent_xy": (2.5, 2.5)},
    )
    ball_too_far = DoneTerm(
        func=mdp.ball_too_far_from_robot,
        params={"max_distance": 2.5},
    )


@configclass
class AS2DribbleFlatEnvCfg(AS2VelocityFlatEnvCfg):
    """First migrated low-level football skill."""

    scene: AS2DribbleSceneCfg = AS2DribbleSceneCfg(num_envs=4096, env_spacing=5.5)
    observations: AS2DribbleObservationsCfg = AS2DribbleObservationsCfg()
    events: AS2DribbleEventCfg = AS2DribbleEventCfg()
    rewards: AS2DribbleRewardsCfg = AS2DribbleRewardsCfg()
    terminations: AS2DribbleTerminationsCfg = AS2DribbleTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # For this skill, xy is desired ball world velocity and z is desired
        # robot yaw rate. This matches the old SoccerRewards command contract.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (7.0, 7.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.5, 1.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.gait_parameters.resampling_time_range = (7.0, 7.0)

        # The ball owns planar tracking; only yaw tracking remains on the base.
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.feet_air_time.weight = 0.0
        self.episode_length_s = 40.0


@configclass
class AS2DribbleFlatEnvCfg_PLAY(AS2DribbleFlatEnvCfg):
    """Small deterministic dribbling scene for smoke testing."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 5.5
        self.observations.policy.enable_corruption = False
        self.observations.legacy_policy.legacy_observation.params["enable_noise"] = False
        self.observations.legacy_history.legacy_history.params["enable_noise"] = False
        self.events.physics_material = None
        self.events.ball_physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
