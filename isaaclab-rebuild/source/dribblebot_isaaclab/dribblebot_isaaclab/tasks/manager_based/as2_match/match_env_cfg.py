"""Four-AS2 manager-based match scene and coordinator contracts."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RecorderManagerBaseCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from ....assets import (
    AS2_CFG,
    GOAL_VISUAL_CFG,
    GOAL_EAST_CROSSBAR_CFG,
    GOAL_EAST_NORTH_CFG,
    GOAL_EAST_SOUTH_CFG,
    GOAL_WEST_CROSSBAR_CFG,
    GOAL_WEST_NORTH_CFG,
    GOAL_WEST_SOUTH_CFG,
    SOCCER_FIELD_VISUAL_CFG,
)
from ....policies import REPRODUCTION_COMMAND_SCALES, ball_skill_command_frame
from ..as2_dribble.dribble_env_cfg import AS2DribbleSceneCfg
from ..as2_velocity.velocity_env_cfg import FLAT_TERRAIN_CFG
from ..football import mdp
from ..football.frozen_skill_cfg import make_frozen_skill_action_cfg


ROBOT_NAMES = ("robot_0", "robot_1", "robot_2", "robot_3")
TEAM_SIZE = 2
TEAM_0_NAMES = ROBOT_NAMES[:TEAM_SIZE]
TEAM_1_NAMES = ROBOT_NAMES[TEAM_SIZE:]
# The checked-in reproduction dribble/shoot configs have no ball_xy_frame
# metadata and retain the legacy world-frame contract. Set this to "body"
# only when all installed ball-skill checkpoints declare that contract.
BALL_SKILL_CHECKPOINT_FRAME = ball_skill_command_frame()


@configclass
class AS2MatchSceneCfg(AS2DribbleSceneCfg):
    """One ball and four independent AS2 articulations per environment."""

    robot = None
    robot_0: ArticulationCfg = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_0")
    robot_1: ArticulationCfg = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
    robot_2: ArticulationCfg = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
    robot_3: ArticulationCfg = AS2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_3")
    field_visual: AssetBaseCfg = SOCCER_FIELD_VISUAL_CFG
    goal_visual: AssetBaseCfg = GOAL_VISUAL_CFG
    goal_east_north: AssetBaseCfg = GOAL_EAST_NORTH_CFG
    goal_east_south: AssetBaseCfg = GOAL_EAST_SOUTH_CFG
    goal_east_crossbar: AssetBaseCfg = GOAL_EAST_CROSSBAR_CFG
    goal_west_north: AssetBaseCfg = GOAL_WEST_NORTH_CFG
    goal_west_south: AssetBaseCfg = GOAL_WEST_SOUTH_CFG
    goal_west_crossbar: AssetBaseCfg = GOAL_WEST_CROSSBAR_CFG
    # Physical perimeter copied from the current Isaac Gym match. End walls
    # are split around the 2 m goal mouth so scoring remains possible.
    wall_north: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallNorth",
        spawn=sim_utils.CuboidCfg(
            size=(8.34, 0.12, 0.50),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.35, dynamic_friction=0.35, restitution=0.85,
                friction_combine_mode="average", restitution_combine_mode="average",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.88)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.61, 0.25)),
    )
    wall_south: AssetBaseCfg = wall_north.replace(
        prim_path="{ENV_REGEX_NS}/WallSouth",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.61, 0.25)),
    )
    wall_east_north: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallEastNorth",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 1.55, 0.50),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.35, dynamic_friction=0.35, restitution=0.85,
                friction_combine_mode="average", restitution_combine_mode="average",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.88)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.11, 1.775, 0.25)),
    )
    wall_east_south: AssetBaseCfg = wall_east_north.replace(
        prim_path="{ENV_REGEX_NS}/WallEastSouth",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(4.11, -1.775, 0.25)),
    )
    wall_west_north: AssetBaseCfg = wall_east_north.replace(
        prim_path="{ENV_REGEX_NS}/WallWestNorth",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.11, 1.775, 0.25)),
    )
    wall_west_south: AssetBaseCfg = wall_east_north.replace(
        prim_path="{ENV_REGEX_NS}/WallWestSouth",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.11, -1.775, 0.25)),
    )


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
    """Four 4D hybrid skill actions, one per physical robot.

    Each action is ``[skill_index, parameter_x, parameter_y, parameter_yaw]``;
    the self-play wrapper exposes the first two robots as two learning agents.
    """

    skill_policy_0 = make_frozen_skill_action_cfg(
        asset_name="robot_0",
        action_term_name="skill_policy_0",
        base_command_name="base_velocity_0",
        gait_command_name="gait_parameters_0",
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
        ball_skill_checkpoint_frame=BALL_SKILL_CHECKPOINT_FRAME,
        role_aware_fallback=True,
        team_robot_names=TEAM_0_NAMES,
        team_index=0,
        team_slot=0,
    )
    skill_policy_1 = make_frozen_skill_action_cfg(
        asset_name="robot_1",
        action_term_name="skill_policy_1",
        base_command_name="base_velocity_1",
        gait_command_name="gait_parameters_1",
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
        ball_skill_checkpoint_frame=BALL_SKILL_CHECKPOINT_FRAME,
        role_aware_fallback=True,
        team_robot_names=TEAM_0_NAMES,
        team_index=0,
        team_slot=1,
    )
    skill_policy_2 = make_frozen_skill_action_cfg(
        asset_name="robot_2",
        action_term_name="skill_policy_2",
        base_command_name="base_velocity_2",
        gait_command_name="gait_parameters_2",
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
        ball_skill_checkpoint_frame=BALL_SKILL_CHECKPOINT_FRAME,
        role_aware_fallback=True,
        team_robot_names=TEAM_1_NAMES,
        team_index=1,
        team_slot=0,
    )
    skill_policy_3 = make_frozen_skill_action_cfg(
        asset_name="robot_3",
        action_term_name="skill_policy_3",
        base_command_name="base_velocity_3",
        gait_command_name="gait_parameters_3",
        command_scales=REPRODUCTION_COMMAND_SCALES,
        geometric_skill_fallback=True,
        ball_skill_checkpoint_frame=BALL_SKILL_CHECKPOINT_FRAME,
        role_aware_fallback=True,
        team_robot_names=TEAM_1_NAMES,
        team_index=1,
        team_slot=1,
    )

    ball_drag = mdp.MatchBallDragCfg()


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
    # Gym samples this once per physical match, including in play mode.
    ball_mass = EventTerm(
        func=env_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (0.5, 0.8),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    reset_match = EventTerm(
        func=mdp.reset_match_scene,
        mode="reset",
        params={
            "robot_names": ROBOT_NAMES,
            "team_size": TEAM_SIZE,
            "ball_name": "ball",
            "randomize": True,
            "team_0_x_range": (-3.4, 0.0),
            "team_1_x_range": (0.0, 3.4),
            "robot_y_range": (-1.9, 1.9),
            "ball_x_range": (-3.2, 2.8),
            "ball_y_range": (-1.7, 1.7),
            "min_clearance": 0.75,
            # Increase valid dribble/shoot exposure during from-scratch MARL
            # training without removing full-field randomized starts.
            "near_ball_probability": 0.65,
            "near_ball_distance_range": (0.4, 0.95),
            "near_ball_angle_range": (-0.35, 0.35),
        },
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
    # IsaacLab multiplies these weights by the 0.02 s manager step, matching
    # the current Isaac Gym high-level reward scaling convention.
    # Terminal outcome must dominate the dense shaping accumulated during a
    # 30-second episode; the old 500 weight admitted a safe-timeout optimum.
    goal = RewTerm(func=mdp.match_goal_event, weight=3000.0)
    accidental_termination = RewTerm(func=mdp.match_accidental_termination_event, weight=-180.0)
    timeout = RewTerm(func=mdp.match_timeout_event, weight=-180.0)
    time_pressure = RewTerm(func=mdp.match_time_pressure, weight=0.3)
    ball_goal_progress = RewTerm(func=mdp.match_ball_goal_progress, weight=1.0)
    ball_position_progress = RewTerm(func=mdp.MatchBallGoalProgressDelta, weight=6.0)
    robot_spacing = RewTerm(
        func=mdp.match_robot_spacing,
        weight=-0.5,
        params={"robot_names": ROBOT_NAMES},
    )
    robot_collision = RewTerm(
        func=mdp.match_robot_collision,
        weight=-1.5,
        params={"robot_names": ROBOT_NAMES, "collision_distance": 0.65, "lookahead": 0.25},
    )
    invalid_skill = RewTerm(
        func=mdp.match_invalid_skill,
        # Fallback and per-role credit already penalize an invalid request.
        # Keep this term diagnostic/guiding rather than large enough to force
        # the categorical policy into an all-walk local optimum.
        weight=-0.5,
        params={"action_names": ("skill_policy_0", "skill_policy_1")},
    )
    aggressive_command = RewTerm(
        func=mdp.match_aggressive_command,
        weight=-1.0,
        params={"action_names": ("skill_policy_0", "skill_policy_1")},
    )
    pass_ball = RewTerm(
        func=mdp.match_pass,
        weight=2.0,
        params={
            "robot_names": TEAM_0_NAMES,
            "action_names": ("skill_policy_0", "skill_policy_1"),
        },
    )
    approach_ball = RewTerm(
        func=mdp.match_approach_ball,
        weight=0.75,
        params={
            "robot_names": TEAM_0_NAMES,
            "action_names": ("skill_policy_0", "skill_policy_1"),
        },
    )
    walk_command_alignment = RewTerm(
        func=mdp.match_walk_command_alignment,
        weight=0.2,
        params={
            "robot_names": TEAM_0_NAMES,
            "action_names": ("skill_policy_0", "skill_policy_1"),
        },
    )
    face_ball_while_approaching = RewTerm(
        func=mdp.match_face_ball_while_approaching,
        weight=0.1,
        params={
            "robot_names": TEAM_0_NAMES,
            "action_names": ("skill_policy_0", "skill_policy_1"),
            "dribble_distance": 1.0,
            "target_speed": 0.9,
        },
    )
    face_goal_while_moving = RewTerm(
        func=mdp.match_face_goal_while_moving,
        weight=0.4,
        params={
            "robot_names": TEAM_0_NAMES,
            "action_names": ("skill_policy_0", "skill_policy_1"),
            "goal_x": 4.0,
            "target_speed": 0.5,
        },
    )
    dribble_ball_control = RewTerm(
        func=mdp.match_dribble_ball_control,
        weight=3.5,
        params={
            "robot_names": TEAM_0_NAMES,
            "action_names": ("skill_policy_0", "skill_policy_1"),
        },
    )
    shoot_setup = RewTerm(
        func=mdp.match_shoot_setup,
        weight=14.0,
        params={
            "robot_names": TEAM_0_NAMES,
            "action_names": ("skill_policy_0", "skill_policy_1"),
        },
    )
    shoot_launch = RewTerm(
        func=mdp.MatchShootLaunchReward,
        weight=30.0,
        params={
            "robot_names": TEAM_0_NAMES,
            "action_names": ("skill_policy_0", "skill_policy_1"),
            "shoot_distance": 0.75,
            "min_command_speed": 0.2,
            "min_ball_speed": 0.8,
            "min_delta_speed": 0.25,
            "target_delta_speed": 1.5,
            "min_command_alignment": 0.6,
        },
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
        params={"half_extent_xy": (4.0, 2.5), "boundary_walls": True},
    )
    robot_fallen = DoneTerm(
        func=mdp.match_robot_fallen,
        params={"robot_names": ROBOT_NAMES, "min_height": 0.20},
    )


@configclass
class AS2MatchCurriculumCfg:
    """Difficulty schedule for from-scratch coordinator training."""

    training_difficulty = CurrTerm(
        func=mdp.match_training_curriculum,
        params={
            # At 128 matches and a 10-step macro action, these correspond to
            # roughly 500 and 2,000 PPO iterations respectively.
            "warmup_steps": 120_000,
            "expansion_steps": 480_000,
        },
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
    curriculum: AS2MatchCurriculumCfg = AS2MatchCurriculumCfg()
    # Raw manager steps completed before this simulator instance was created.
    # The segmented train/eval launcher fills this from the checkpoint index.
    curriculum_step_offset: int = 0
    episode_length_s: float = 30.0
    decimation: int = 4
    macro_control_interval: int = 10
    team_size: int = TEAM_SIZE
    coordinator_history_length: int = 4
    opponent_snapshot_interval: int = 500
    opponent_pool_size: int = 8
    opponent_latest_probability: float = 0.5
    opponent_checkpoint_root: str | None = None
    opponent_policy_device: str = "cpu"
    opponent_mode: str = "zero"

    def __post_init__(self) -> None:
        # Scene/material parity with the default Gym 2v2 match. Keep these
        # overrides local to match; low-level parity fixtures use fixed balls.
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            friction_combine_mode="average", restitution_combine_mode="average",
        )
        self.scene.terrain.physics_material = self.sim.physics_material.copy()
        self.scene.ball.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.85,
            friction_combine_mode="average", restitution_combine_mode="average",
        )
        self.scene.ball.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))
        for index in range(TEAM_SIZE, 4):
            # Preserve both original AS2 body/foot materials on the learner.
            # Gym paints every opponent link red at actor creation.
            color = (0.85, 0.10, 0.10)
            getattr(self.scene, f"robot_{index}").spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
            )
        self.sim.gravity = (0.0, 0.0, -9.8)  # Gym _randomize_gravity baseline
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # A single flat terrain tile otherwise assigns (0, 0, 0) to EVERY
        # match, while static goals/walls follow GridCloner's spaced origins.
        self.scene.terrain.use_terrain_origins = False
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
        self.events.reset_match.params["randomize"] = False
        # Evaluation always uses the mature objective; training curricula are
        # driven by optimizer-time steps and must not alter fixed PLAY resets.
        self.curriculum = None
