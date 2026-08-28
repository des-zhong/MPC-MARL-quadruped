"""Simulator-configuration adapters shared by Isaac Lab parity recorders."""

from __future__ import annotations

from .physics_profile import make_physics_profile
from .policy_contract import LEGACY_PARITY_COMMAND


def configure_isaaclab_env_for_parity(env_cfg: object, seed: int, num_envs: int) -> None:
    """Mutate a manager-based config into the deterministic parity profile."""

    profile = make_physics_profile(include_ball=getattr(env_cfg.scene, "ball", None) is not None)
    solver = profile["solver"]
    ground = profile["ground"]
    env_cfg.seed = int(seed)
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.gravity = tuple(profile["gravity_mps2"])
    env_cfg.sim.physx.solver_type = 1 if solver["type"] == "tgs" else 0
    env_cfg.sim.physx.bounce_threshold_velocity = solver["bounce_threshold_velocity_mps"]
    env_cfg.observations.policy.enable_corruption = False
    if getattr(env_cfg.observations, "critic", None) is not None:
        env_cfg.observations.critic.enable_corruption = False
    for group_name in ("legacy_policy", "legacy_history"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is None:
            raise RuntimeError(f"Parity task does not expose the required observation group '{group_name}'")
        group.enable_corruption = False
        term_name = "legacy_observation" if group_name == "legacy_policy" else "legacy_history"
        term_cfg = getattr(group, term_name, None)
        if term_cfg is None:
            raise RuntimeError(f"Observation group '{group_name}' does not contain term '{term_name}'")
        term_cfg.params["enable_noise"] = False

    base_velocity = env_cfg.commands.base_velocity
    base_velocity.heading_command = False
    base_velocity.rel_heading_envs = 0.0
    base_velocity.ranges.lin_vel_x = (LEGACY_PARITY_COMMAND[0], LEGACY_PARITY_COMMAND[0])
    base_velocity.ranges.lin_vel_y = (LEGACY_PARITY_COMMAND[1], LEGACY_PARITY_COMMAND[1])
    base_velocity.ranges.ang_vel_z = (LEGACY_PARITY_COMMAND[2], LEGACY_PARITY_COMMAND[2])
    base_velocity.ranges.heading = (0.0, 0.0)

    gait_ranges = env_cfg.commands.gait_parameters.ranges
    gait_names = (
        "body_height",
        "frequency",
        "phase",
        "offset",
        "bound",
        "duration",
        "foot_swing_height",
        "body_pitch",
        "body_roll",
        "stance_width",
        "stance_length",
        "aux_reward",
    )
    for name, value in zip(gait_names, LEGACY_PARITY_COMMAND[3:]):
        setattr(gait_ranges, name, (value, value))

    for name in (
        "physics_material",
        "ball_physics_material",
        "add_base_mass",
        "base_com",
        "base_external_force_torque",
        "push_robot",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    _disable_term_group(env_cfg.terminations)

    terrain_material = getattr(env_cfg.scene.terrain, "physics_material", None)
    if terrain_material is not None:
        terrain_material.static_friction = ground["static_friction"]
        terrain_material.dynamic_friction = ground["dynamic_friction"]
        terrain_material.restitution = ground["restitution"]
    env_cfg.sim.physics_material = terrain_material

    robot_spawn = env_cfg.scene.robot.spawn
    robot_spawn.rigid_props.max_depenetration_velocity = solver["max_depenetration_velocity_mps"]
    robot_spawn.articulation_props.solver_position_iteration_count = solver["position_iterations"]
    robot_spawn.articulation_props.solver_velocity_iteration_count = solver["velocity_iterations"]
    env_cfg.events.robot_collider_offsets.params["contact_offset"] = solver["contact_offset_m"]
    env_cfg.events.robot_collider_offsets.params["rest_offset"] = solver["rest_offset_m"]

    ball_cfg = getattr(env_cfg.scene, "ball", None)
    if ball_cfg is not None:
        ball = profile["ball"]
        ball_spawn = ball_cfg.spawn
        ball_spawn.mass_props.mass = ball["mass_kg"]
        ball_spawn.rigid_props.linear_damping = ball["linear_damping"]
        ball_spawn.rigid_props.angular_damping = ball["angular_damping"]
        ball_spawn.rigid_props.max_linear_velocity = ball["max_linear_velocity_mps"]
        ball_spawn.rigid_props.max_angular_velocity = ball["max_angular_velocity_radps"]
        ball_spawn.rigid_props.enable_gyroscopic_forces = ball["enable_gyroscopic_forces"]
        ball_spawn.rigid_props.max_depenetration_velocity = solver["max_depenetration_velocity_mps"]
        ball_spawn.rigid_props.solver_position_iteration_count = solver["position_iterations"]
        ball_spawn.rigid_props.solver_velocity_iteration_count = solver["velocity_iterations"]
        ball_spawn.collision_props.contact_offset = solver["contact_offset_m"]
        ball_spawn.collision_props.rest_offset = solver["rest_offset_m"]
        ball_spawn.physics_material.static_friction = ball["static_friction"]
        ball_spawn.physics_material.dynamic_friction = ball["dynamic_friction"]
        ball_spawn.physics_material.restitution = ball["restitution"]
        env_cfg.events.ball_collider_offsets.params["contact_offset"] = solver["contact_offset_m"]
        env_cfg.events.ball_collider_offsets.params["rest_offset"] = solver["rest_offset_m"]


def _disable_term_group(group: object) -> None:
    if group is None:
        return
    if hasattr(group, "to_dict"):
        term_names = group.to_dict().keys()
    else:
        term_names = vars(group).keys()
    for name in list(term_names):
        if not name.startswith("_"):
            setattr(group, name, None)
