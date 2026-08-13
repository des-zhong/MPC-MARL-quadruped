from typing import Union
from pathlib import Path

from params_proto import Meta

from dribblebot.envs.base.legged_robot_config import Cfg


AS2_URDF_PATH = Path(__file__).resolve().parents[3] / "resources" / "robots" / "as2" / "urdf" / "as2.urdf"


def config_as2(Cnfg: Union[Cfg, Meta]):
    """Apply AS2-specific asset, pose, controller, and randomization defaults."""

    state = Cnfg.init_state
    state.pos = [0.0, 0.0, 0.34]
    state.default_joint_angles = {
        "FL_hip_joint": 0.1,
        "FR_hip_joint": -0.1,
        "RL_hip_joint": 0.1,
        "RR_hip_joint": -0.1,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    }

    control = Cnfg.control
    control.control_type = "P"
    control.stiffness = {
        "hip_joint": 30.0,
        "thigh_joint": 30.0,
        "calf_joint": 35.0,
    }
    control.damping = {
        "hip_joint": 0.8,
        "thigh_joint": 0.8,
        "calf_joint": 1.0,
    }
    control.action_scale = 0.25
    control.hip_scale_reduction = 0.5
    control.decimation = 4

    asset = Cnfg.asset
    asset.file = str(AS2_URDF_PATH)
    asset.foot_name = "foot"
    asset.penalize_contacts_on = ["thigh", "calf"]
    asset.terminate_after_contacts_on = ["base_link"]
    asset.self_collisions = 0
    asset.flip_visual_attachments = False
    asset.fix_base_link = False
    asset.collapse_fixed_joints = False

    rewards = Cnfg.rewards
    rewards.soft_dof_pos_limit = 0.9
    rewards.base_height_target = 0.34

    # AS2 has higher torque limits and is about 56% heavier than Go1. Start
    # with a lower raw-torque penalty and tune it from measured rollouts.
    reward_scales = Cnfg.reward_scales
    reward_scales.torques = -3.0e-5
    reward_scales.action_rate = -0.01
    reward_scales.dof_pos_limits = -10.0
    reward_scales.orientation = -5.0
    reward_scales.base_height = -30.0

    terrain = Cnfg.terrain
    terrain.mesh_type = "trimesh"
    terrain.measure_heights = False
    terrain.terrain_noise_magnitude = 0.0
    terrain.teleport_robots = True
    terrain.border_size = 50
    terrain.terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
    terrain.curriculum = False

    env = Cnfg.env
    env.num_observations = 42
    env.num_envs = 4000

    commands = Cnfg.commands
    commands.heading_command = False
    commands.resampling_time = 10.0
    commands.command_curriculum = True
    commands.num_lin_vel_bins = 30
    commands.num_ang_vel_bins = 30
    commands.lin_vel_x = [-0.6, 0.6]
    commands.lin_vel_y = [-0.6, 0.6]
    commands.ang_vel_yaw = [-1.0, 1.0]

    domain_rand = Cnfg.domain_rand
    domain_rand.randomize_base_mass = False
    domain_rand.randomize_com_displacement = False
    domain_rand.randomize_motor_strength = False
    domain_rand.randomize_motor_offset = False
    domain_rand.randomize_Kp_factor = False
    domain_rand.randomize_Kd_factor = False
    domain_rand.randomize_gravity = False
    domain_rand.push_robots = False
    domain_rand.randomize_friction = True
    domain_rand.friction_range = [0.7, 1.5]
    domain_rand.randomize_restitution = False
    domain_rand.restitution_range = [0.0, 0.4]
    domain_rand.rand_interval_s = 6
