"""Manager-based AS2 shooting task with explicit phase transitions."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ....policies import REPRODUCTION_COMMAND_SCALES
from ..as2_dribble.dribble_env_cfg import (
    AS2DribbleEventCfg,
    AS2DribbleFlatEnvCfg,
    AS2DribbleRewardsCfg,
    AS2DribbleSceneCfg,
    AS2DribbleTerminationsCfg,
)
from ..football import mdp
from ..football.frozen_skill_cfg import AS2FrozenSkillObservationsCfg, make_frozen_skill_action_cfg


@configclass
class AS2ShootingEventCfg(AS2DribbleEventCfg):
    """Keep reset state deterministic until shooting contact is reproduced."""

    physics_material = None
    ball_physics_material = None
    add_base_mass = None
    base_external_force_torque = None
    reset_ball = None


@configclass
class AS2ShootingRewardsCfg(AS2DribbleRewardsCfg):
    """Dense setup, strike, launch, and terminal shooting objectives."""

    shooting_ball_vel = RewTerm(
        func=mdp.shooting_ball_velocity_exp,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "scale_xy": (3.0, 3.0),
            "tracking_sigma": 0.25,
            "min_command_speed": 0.3,
            "min_separation": 0.55,
        },
    )
    shooting_ball_forward = RewTerm(
        func=mdp.shooting_ball_forward_velocity,
        weight=8.0,
        params={"command_name": "base_velocity", "min_command_speed": 0.3},
    )
    shooting_ball_vel_norm = RewTerm(
        func=mdp.shooting_ball_speed_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "scale_xy": (3.0, 3.0),
            "min_command_speed": 0.3,
            "min_separation": 0.55,
        },
    )
    shooting_ball_vel_angle = RewTerm(
        func=mdp.shooting_ball_direction,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "min_command_speed": 0.3,
            "min_separation": 0.55,
        },
    )
    shooting_ball_out = RewTerm(
        func=mdp.shooting_ball_out,
        weight=7.0,
        params={
            "command_name": "base_velocity",
            "scale_xy": (3.0, 3.0),
            "setup_distance": 0.45,
            "min_command_speed": 0.3,
        },
    )
    shooting_robot_ball_behind = RewTerm(
        func=mdp.shooting_setup_position_exp,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "setup_distance": 0.45,
            "position_gain": 6.0,
            "min_command_speed": 0.3,
        },
    )
    shooting_robot_forward_cmd = RewTerm(
        func=mdp.shooting_robot_heading,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "setup_distance": 0.45,
            "position_gain": 6.0,
            "min_command_speed": 0.3,
        },
    )
    shooting_ball_in_front = RewTerm(
        func=mdp.shooting_ball_in_front,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "setup_distance": 0.45,
            "position_gain": 6.0,
            "min_command_speed": 0.3,
        },
    )
    shooting_robot_approach_ball = RewTerm(
        func=mdp.ShootingSetupProgressReward,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "setup_distance": 0.45,
            "speed_scale": 1.0,
            "min_command_speed": 0.3,
        },
    )
    shooting_excess_yaw = RewTerm(
        func=mdp.shooting_excess_yaw,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "free_yaw_rate": 0.75,
            "yaw_rate_scale": 2.0,
            "min_command_speed": 0.3,
        },
    )
    shooting_launch = RewTerm(func=mdp.shooting_launch_event, weight=50.0)
    shooting_success = RewTerm(func=mdp.shooting_success_event, weight=200.0)
    shooting_failure = RewTerm(func=mdp.shooting_failure_event, weight=-25.0)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-4)


@configclass
class AS2ShootingTerminationsCfg(AS2DribbleTerminationsCfg):
    """Terminate on robot failure or a completed shooting attempt."""

    ball_out_of_bounds = None
    ball_too_far = None
    shooting_phase = DoneTerm(
        func=mdp.ShootingPhaseTermination,
        params={
            "command_name": "base_velocity",
            "min_command_speed": 0.3,
            "launch_speed_fraction": 0.20,
            "launch_alignment": 0.50,
            "success_distance": 0.70,
            "success_speed_fraction": 0.25,
            "success_alignment": 0.65,
            "max_attempt_time_s": 5.0,
        },
    )


@configclass
class AS2ShootingFlatEnvCfg(AS2DribbleFlatEnvCfg):
    """Command-relative flat-ground shooting environment."""

    scene: AS2DribbleSceneCfg = AS2DribbleSceneCfg(num_envs=1000, env_spacing=5.5)
    events: AS2ShootingEventCfg = AS2ShootingEventCfg()
    rewards: AS2ShootingRewardsCfg = AS2ShootingRewardsCfg()
    terminations: AS2ShootingTerminationsCfg = AS2ShootingTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.commands.base_velocity = mdp.ShootingCommandCfg(
            resampling_time_range=(10.0, 10.0),
            min_command_speed=0.3,
            longitudinal_range=(0.35, 0.60),
            lateral_range=(-0.20, 0.20),
            yaw_error_range=(-0.25, 0.25),
            ball_height=0.10,
            ranges=mdp.ShootingCommandCfg.Ranges(
                lin_vel_x=(-3.0, 3.0),
                lin_vel_y=(-3.0, 3.0),
            ),
        )
        self.commands.gait_parameters.resampling_time_range = (10.0, 10.0)

        # Shooting owns the task signal; locomotion managers retain only the
        # actuator, posture, collision, and smoothness regularizers.
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.track_ball_velocity_xy_exp.weight = 0.0
        self.rewards.track_ball_direction.weight = 0.0
        self.rewards.ball_setup_position_exp.weight = 0.0
        self.rewards.robot_ball_command_alignment_exp.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.undesired_contacts.weight = -5.0
        self.episode_length_s = 10.0


@configclass
class AS2ShootingFlatEnvCfg_PLAY(AS2ShootingFlatEnvCfg):
    """Small noise-free scene for visual shooting smoke tests."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.observations.legacy_policy.legacy_observation.params["enable_noise"] = False
        self.observations.legacy_history.legacy_history.params["enable_noise"] = False


@configclass
class AS2FrozenShootingActionsCfg:
    """Three normalized command values executed by the frozen shooting policy."""

    skill_policy: mdp.FrozenSkillPolicyActionCfg = make_frozen_skill_action_cfg(
        fixed_skill_id=mdp.SHOOT_SKILL_ID,
        command_scales=REPRODUCTION_COMMAND_SCALES,
        action_history_semantics="successive_policy_outputs",
    )


@configclass
class AS2FrozenShootingFlatEnvCfg(AS2ShootingFlatEnvCfg):
    """Compose the trusted shoot checkpoint with the shooting task state machine."""

    actions: AS2FrozenShootingActionsCfg = AS2FrozenShootingActionsCfg()
    observations: AS2FrozenSkillObservationsCfg = AS2FrozenSkillObservationsCfg()


@configclass
class AS2FrozenShootingFlatEnvCfg_PLAY(AS2FrozenShootingFlatEnvCfg):
    """Small deterministic frozen-shoot scene for closed-loop validation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.observations.policy.enable_corruption = False
        self.observations.legacy_policy.legacy_observation.params["enable_noise"] = False
        self.observations.legacy_history.legacy_history.params["enable_noise"] = False
