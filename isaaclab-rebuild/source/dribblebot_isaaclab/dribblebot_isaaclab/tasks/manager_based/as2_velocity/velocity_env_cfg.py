"""Manager-based AS2 locomotion task.

This module is the first executable migration slice. The manager terms replace
the old ``LeggedRobot`` lifecycle while retaining its 5 ms physics step, four
physics steps per action, joint offsets, PD gains, and primary reward scales.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CommandsCfg as VelocityCommandsCfg,
    EventCfg as VelocityEventCfg,
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg as VelocityObservationsCfg,
)

from ....assets import AS2_CFG
from ..football import mdp


FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=1,
    num_cols=1,
    border_width=0.0,
    sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
)


def legacy_observation_params(include_ball: bool, enable_noise: bool = True) -> dict:
    return {
        "include_ball": include_ball,
        "enable_noise": enable_noise,
        "robot_cfg": SceneEntityCfg(
            "robot",
            joint_names=list(mdp.LEGACY_JOINT_NAMES),
            preserve_order=True,
        ),
    }


@configclass
class AS2CommandsCfg(VelocityCommandsCfg):
    """Velocity command plus the remaining legacy gait parameters."""

    gait_parameters: mdp.GaitCommandCfg = mdp.GaitCommandCfg(
        resampling_time_range=(7.0, 7.0),
        ranges=mdp.GaitCommandCfg.Ranges(
            body_height=(-0.05, 0.05),
            frequency=(3.0, 3.0),
            phase=(0.5, 0.5),
            offset=(0.0, 0.0),
            bound=(0.0, 0.0),
            duration=(0.5, 0.5),
            foot_swing_height=(0.09, 0.09),
            body_pitch=(0.0, 0.0),
            body_roll=(0.0, 0.0),
            stance_width=(0.0, 0.1),
            stance_length=(0.0, 0.1),
            aux_reward=(0.0, 0.01),
        ),
    )


@configclass
class AS2VelocityObservationsCfg(VelocityObservationsCfg):
    """Current training observations plus checkpoint-compatible groups."""

    @configclass
    class LegacyPolicyCfg(ObsGroup):
        legacy_observation = ObsTerm(
            func=mdp.LegacyPolicyObservation,
            params=legacy_observation_params(include_ball=False),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class LegacyHistoryCfg(ObsGroup):
        legacy_history = ObsTerm(
            func=mdp.LegacyHistoryObservation,
            params={
                **legacy_observation_params(include_ball=False),
                "history_length": mdp.LEGACY_HISTORY_LENGTH,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    legacy_policy: LegacyPolicyCfg = LegacyPolicyCfg()
    legacy_history: LegacyHistoryCfg = LegacyHistoryCfg()


@configclass
class AS2VelocityEventCfg(VelocityEventCfg):
    """Locomotion events plus deterministic PhysX collider offsets."""

    legacy_base_com = EventTerm(
        func=mdp.set_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com": (0.0, 0.0, 0.0),
        },
    )

    robot_collider_offsets = EventTerm(
        func=mdp.set_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "rest_offset": 0.0,
            "contact_offset": 0.01,
        },
    )


@configclass
class AS2VelocityRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """AS2 velocity task built from Isaac Lab's standard locomotion managers."""

    commands: AS2CommandsCfg = AS2CommandsCfg()
    observations: AS2VelocityObservationsCfg = AS2VelocityObservationsCfg()
    events: AS2VelocityEventCfg = AS2VelocityEventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Scene and control adapter.
        self.scene.robot = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
        # Isaac Lab may expose URDF DOFs grouped by joint type (hips, then
        # thighs, then calves).  Legacy checkpoints and the Isaac Gym task use
        # the leg-major FL/FR/RL/RR order, so resolve both the action and
        # compatibility observations against the explicit canonical list.
        # The Skill task intentionally replaces the regular joint action group
        # with a hybrid coordinator term, so there is no
        # ``joint_pos`` member to configure in that subclass.
        joint_pos_action = getattr(self.actions, "joint_pos", None)
        if joint_pos_action is not None:
            joint_pos_action.joint_names = list(mdp.LEGACY_JOINT_NAMES)
            joint_pos_action.preserve_order = True
            joint_pos_action.scale = {
                ".*_hip_joint": 0.125,
                ".*_thigh_joint": 0.25,
                ".*_calf_joint": 0.25,
            }

        # Original simulator timing: 200 Hz physics, 50 Hz policy.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 40.0

        # Velocity command contract used by the walking skill.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (7.0, 7.0)
        self.commands.gait_parameters.resampling_time_range = (7.0, 7.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Domain randomization translated from config_as2/train_walking.py.
        self.events.push_robot = None
        if self.events.add_base_mass is not None:
            self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
            self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        if self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"].body_names = "base_link"
        self.events.base_com = None
        if self.events.physics_material is not None:
            self.events.physics_material.params["static_friction_range"] = (0.7, 1.5)
            self.events.physics_material.params["dynamic_friction_range"] = (0.7, 1.5)
            self.events.physics_material.params["restitution_range"] = (0.0, 0.4)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Body names and first-pass reward parity.
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh", ".*_calf"]
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.dof_torques_l2.weight = -3.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_pos_limits.weight = -10.0
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class AS2VelocityFlatEnvCfg(AS2VelocityRoughEnvCfg):
    """Flat-ground AS2 walking task used for the initial parity gate."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Keep the smoke/parity scene fully offline.  The stock locomotion
        # config references Nucleus ground/material/sky assets, which are not
        # guaranteed on an air-gapped workstation or CI runner.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_TERRAIN_CFG
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.2),
        )
        self.scene.sky_light.spawn.texture_file = None
        self.commands.base_velocity.debug_vis = False
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class AS2VelocityFlatEnvCfg_PLAY(AS2VelocityFlatEnvCfg):
    """Small deterministic scene for visual and physics smoke tests."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.legacy_policy.legacy_observation.params["enable_noise"] = False
        self.observations.legacy_history.legacy_history.params["enable_noise"] = False
        self.events.base_external_force_torque = None
        self.events.add_base_mass = None
