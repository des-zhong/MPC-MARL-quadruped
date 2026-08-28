"""Soccer ball rigid-body configuration."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


SOCCER_BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.0889,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            # Match the explicit/default Isaac Gym Preview 4 sphere values.
            linear_damping=0.0,
            angular_damping=0.5,
            max_linear_velocity=1000.0,
            max_angular_velocity=64.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.318),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.05)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.6, 0.0, 0.10),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    ),
)
