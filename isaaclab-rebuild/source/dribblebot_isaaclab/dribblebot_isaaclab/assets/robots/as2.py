"""AS2 articulation configuration for Isaac Lab 2.3.x."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from ..paths import asset_path


AS2_URDF_PATH = asset_path("robots/as2/urdf/as2.urdf")


AS2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(AS2_URDF_PATH),
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        self_collision=True,
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # The Isaac Sim 5.1 URDF importer emits instanceable collider prims.
        # Collider offsets therefore cannot be overridden recursively after
        # the USD is referenced into the scene.  The manager-based startup
        # event sets these values through the PhysX tensor view instead,
        # preserving instancing for large vectorized training scenes.
        collision_props=None,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.34),
        # Isaac Lab 2.x uses wxyz quaternion ordering.
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips_and_thighs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=60.0,
            velocity_limit=24.0,
            stiffness={".*_hip_joint": 30.0, ".*_thigh_joint": 30.0},
            damping={".*_hip_joint": 0.8, ".*_thigh_joint": 0.8},
            friction=0.0,
        ),
        "calves": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=90.0,
            velocity_limit=16.0,
            stiffness=35.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
"""AS2 configuration matching the original P-controller and URDF limits."""
