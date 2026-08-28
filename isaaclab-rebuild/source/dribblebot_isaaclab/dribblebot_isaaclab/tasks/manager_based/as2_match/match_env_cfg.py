"""Four-AS2 manager-based match scene and coordinator contracts."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from ....assets import AS2_CFG
from ....policies import REPRODUCTION_COMMAND_SCALES
from ..as2_dribble.dribble_env_cfg import AS2DribbleSceneCfg
from ..as2_velocity.velocity_env_cfg import FLAT_TERRAIN_CFG
from ..football import mdp
from ..football.frozen_skill_cfg import make_frozen_skill_action_cfg


ROBOT_NAMES = ("robot_0", "robot_1", "robot_2", "robot_3")
TEAM_SIZE = 2


@configclass
class AS2MatchSceneCfg(AS2DribbleSceneCfg):
    """One ball and four independent AS2 articulations per environment."""

    robot = None
    robot_0: ArticulationCfg = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_0")
    robot_1: ArticulationCfg = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
    robot_2: ArticulationCfg = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    robot_3: ArticulationCfg = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_3")


def _velocity_command(asset_name: str):
    return env_mdp.UniformVelocityCommandCfg(
        asset_name=asset_name,
        resampling_time_range=(1.0e6, 1.0e6),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=env_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.0, 3.0),
            lin_vel_y=(-3.0, 3.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )


def _gait_command():
    return mdp.GaitCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
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
            aux_reward=(0.0, 0.005),
        ),
    )


@configclass
class AS2MatchCommandsCfg:
    base_velocity_0 = _velocity_command("robot_0")
    base_velocity_1 = _velocity_command("robot_1")
    base_velocity_2 = _velocity_command("robot_2")
    base_velocity_3 = _velocity_command("robot_3")
    gait_parameters_0 = _gait_command()
    gait_parameters_1 = _gait_command()
    gait_parameters_2 = _gait_command()
    gait_parameters_3 = _gait_command()


@configclass
class AS2MatchActionsCfg:
    skill_policy_0 = make_frozen_skill_action_cfg(
        asset_name="robot_0",
        action_term_name="skill_policy_0",
        base_command_name="base_velocity_0",
        gait_command_name="gait_parameters_0",
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
    )
    skill_policy_1 = make_frozen_skill_action_cfg(
        asset_name="robot_1",
        action_term_name="skill_policy_1",
        base_command_name="base_velocity_1",
        gait_command_name="gait_parameters_1",
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
    )
    skill_policy_2 = make_frozen_skill_action_cfg(
        asset_name="robot_2",
        action_term_name="skill_policy_2",
        base_command_name="base_velocity_2",
        gait_command_name="gait_parameters_2",
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
    )
    skill_policy_3 = make_frozen_skill_action_cfg(
        asset_name="robot_3",
        action_term_name="skill_policy_3",
        base_command_name="base_velocity_3",
        gait_command_name="gait_parameters_3",
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
    )


@configclass
class AS2MatchObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        match_state = ObsTerm(
            func=mdp.match_local_observation,
            params={"robot_names": ROBOT_NAMES, "team_size": TEAM_SIZE},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        match_state = ObsTerm(
            func=mdp.match_local_observation,
            params={"robot_names": ROBOT_NAMES, "team_size": TEAM_SIZE},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class AS2MatchEventsCfg:
    reset_match = EventTerm(
        func=mdp.reset_match_scene,
        mode="reset",
        params={"robot_names": ROBOT_NAMES, "team_size": TEAM_SIZE, "ball_name": "ball"},
    )
    robot_0_com = EventTerm(
        func=mdp.set_rigid_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot_0", body_names="base_link"), "com": (0.0, 0.0, 0.0)},
    )
    robot_1_com = EventTerm(
        func=mdp.set_rigid_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot_1", body_names="base_link"), "com": (0.0, 0.0, 0.0)},
    )
    robot_2_com = EventTerm(
        func=mdp.set_rigid_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot_2", body_names="base_link"), "com": (0.0, 0.0, 0.0)},
    )
    robot_3_com = EventTerm(
        func=mdp.set_rigid_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot_3", body_names="base_link"), "com": (0.0, 0.0, 0.0)},
    )
    robot_0_collider_offsets = EventTerm(
        func=mdp.set_rigid_body_collider_offsets,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot_0"), "rest_offset": 0.0, "contact_offset": 0.01},
    )
    robot_1_collider_offsets = EventTerm(
        func=mdp.set_rigid_body_collider_offsets,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot_1"), "rest_offset": 0.0, "contact_offset": 0.01},
    )
    robot_2_collider_offsets = EventTerm(
        func=mdp.set_rigid_body_collider_offsets,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot_2"), "rest_offset": 0.0, "contact_offset": 0.01},
    )
    robot_3_collider_offsets = EventTerm(
        func=mdp.set_rigid_body_collider_offsets,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot_3"), "rest_offset": 0.0, "contact_offset": 0.01},
    )


@configclass
class AS2MatchRewardsCfg:
    alive = RewTerm(func=mdp.match_alive, weight=1.0)
    goal = RewTerm(func=mdp.match_goal_event, weight=20.0)
    opponent_goal = RewTerm(func=mdp.match_opponent_goal_event, weight=-20.0)
    possession = RewTerm(
        func=mdp.match_possession,
        weight=0.5,
        params={"robot_names": ROBOT_NAMES},
    )


@configclass
class AS2MatchTerminationsCfg:
    time_out = DoneTerm(func=env_mdp.time_out, time_out=True)
    goal = DoneTerm(
        func=mdp.match_goal,
        params={"goal_x": 4.0, "goal_half_width": 1.0},
    )
    opponent_goal = DoneTerm(
        func=mdp.match_opponent_goal,
        params={"goal_x": 4.0, "goal_half_width": 1.0},
    )
    ball_out_of_bounds = DoneTerm(
        func=mdp.match_ball_out_of_bounds,
        params={"half_extent_xy": (4.0, 2.5)},
    )
    robot_fallen = DoneTerm(
        func=mdp.match_robot_fallen,
        params={"robot_names": ROBOT_NAMES, "min_height": 0.20},
    )


@configclass
class AS2MatchRecordersCfg(RecorderManagerBaseCfg):
    """Keep canonical snapshots in memory for collectors by default."""

    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    snapshot = mdp.MatchSnapshotRecorderCfg(robot_names=ROBOT_NAMES)


@configclass
class AS2MatchFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: AS2MatchSceneCfg = AS2MatchSceneCfg(num_envs=128, env_spacing=10.0)
    commands: AS2MatchCommandsCfg = AS2MatchCommandsCfg()
    actions: AS2MatchActionsCfg = AS2MatchActionsCfg()
    observations: AS2MatchObservationsCfg = AS2MatchObservationsCfg()
    events: AS2MatchEventsCfg = AS2MatchEventsCfg()
    rewards: AS2MatchRewardsCfg = AS2MatchRewardsCfg()
    terminations: AS2MatchTerminationsCfg = AS2MatchTerminationsCfg()
    recorders: AS2MatchRecordersCfg = AS2MatchRecordersCfg()
    curriculum = None
    episode_length_s: float = 30.0
    decimation: int = 4
    macro_control_interval: int = 10
    team_size: int = TEAM_SIZE
    coordinator_history_length: int = 4
    opponent_snapshot_interval: int = 500
    opponent_checkpoint_root: str | None = None
    opponent_policy_device: str = "cpu"

    def __post_init__(self) -> None:
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_TERRAIN_CFG
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2))
        self.scene.sky_light.spawn.texture_file = None
        self.scene.height_scanner = None
        self.scene.contact_forces = None
        self.scene.robot = None


@configclass
class AS2MatchFlatEnvCfg_PLAY(AS2MatchFlatEnvCfg):
    """Two-environment four-AS2 match smoke scene."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 2
