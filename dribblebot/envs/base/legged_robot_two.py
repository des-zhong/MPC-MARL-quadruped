import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from dribblebot.envs.base.legged_robot import LeggedRobot
from dribblebot.envs.base.legged_robot_config import Cfg


class TwoRobotLeggedRobot(LeggedRobot):
    """Multi-robot soccer env with a robot-0 compatibility API.

    The existing low-level policies expect one robot observation and one
    12-DoF action. This env spawns a configurable number of physical robot
    actors, keeps robot 0 exposed through the original attributes, and exposes
    the complete team through the ``*_all`` tensors. ``other_*`` aliases robot
    1 when present and robot 0 in the one-robot case.
    """

    num_robots = 2

    def _create_envs(self):
        self.num_robots = int(getattr(self.cfg.env, "num_robots", 2))
        if self.num_robots < 1:
            raise ValueError("cfg.env.num_robots must be at least 1")
        # Static boxes remain available for legacy scenarios, but competitive
        # high-level training sets this to zero and uses the second half of the
        # AS2 actor slots as a policy-controlled opponent team.
        self.cfg.env.num_static_opponents = int(
            getattr(self.cfg.env, "num_static_opponents", 0)
        )
        all_assets = []
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../..")
        )

        from dribblebot.robots.as2 import As2
        from dribblebot.robots.go1 import Go1

        robot_classes = {
            "as2": As2,
            "go1": Go1,
        }


        self.robot = robot_classes[self.cfg.robot.name](self)
        all_assets.append(self.robot)
        self.robot_asset, dof_props_asset, rigid_shape_props_asset = self.robot.initialize()

        object_init_state_list = (
            self.cfg.ball.ball_init_pos
            + self.cfg.ball.ball_init_rot
            + self.cfg.ball.ball_init_lin_vel
            + self.cfg.ball.ball_init_ang_vel
        )
        self.object_init_state = to_torch(object_init_state_list, device=self.device, requires_grad=False)

        from dribblebot.assets.ball import Ball

        asset_classes = {
            "ball": Ball,
        }

        if self.cfg.env.add_balls:
            self.asset = asset_classes[self.cfg.ball.asset](self)
            all_assets.append(self.asset)
            self.ball_asset, ball_rigid_shape_props_asset = self.asset.initialize()
            self.ball_force_feedback = self.asset.get_force_feedback()
            self.num_object_bodies = self.gym.get_asset_rigid_body_count(self.ball_asset)
        else:
            self.ball_force_feedback = None
            self.num_object_bodies = 0

        self.num_static_opponents = int(getattr(self.cfg.env, "num_static_opponents", 0))
        self.static_opponent_size = list(getattr(self.cfg.env, "static_opponent_size", [0.45, 0.45, 0.50]))
        if len(self.static_opponent_size) != 3:
            raise ValueError(f"cfg.env.static_opponent_size must have 3 values, got {self.static_opponent_size}")
        self.static_opponent_size = [float(value) for value in self.static_opponent_size]
        if self.num_static_opponents > 0:
            opponent_asset_options = gymapi.AssetOptions()
            opponent_asset_options.fix_base_link = True
            opponent_asset_options.disable_gravity = True
            opponent_asset_options.density = 1000.0
            self.static_opponent_asset = self.gym.create_box(
                self.sim,
                self.static_opponent_size[0],
                self.static_opponent_size[1],
                self.static_opponent_size[2],
                opponent_asset_options,
            )
            self.num_static_opponent_bodies = (
                self.num_static_opponents * self.gym.get_asset_rigid_body_count(self.static_opponent_asset)
            )
        else:
            self.static_opponent_asset = None
            self.num_static_opponent_bodies = 0

        self.add_goalposts = bool(getattr(self.cfg.env, "add_goalposts", False))
        if self.add_goalposts:
            configured_goalpost_file = getattr(
                self.cfg.env,
                "goalpost_asset_file",
                "resources/objects/goalpost/goalpost.urdf",
            )
            goalpost_file = os.path.expanduser(configured_goalpost_file)
            if not os.path.isabs(goalpost_file):
                goalpost_file = os.path.join(project_root, goalpost_file)
            goalpost_file = os.path.abspath(goalpost_file)
            if not os.path.isfile(goalpost_file):
                raise FileNotFoundError(f"Goalpost asset not found: {goalpost_file}")
            goalpost_options = gymapi.AssetOptions()
            goalpost_options.fix_base_link = True
            goalpost_options.disable_gravity = True
            goalpost_options.collapse_fixed_joints = True
            self.goalpost_asset = self.gym.load_asset(
                self.sim,
                os.path.dirname(goalpost_file),
                os.path.basename(goalpost_file),
                goalpost_options,
            )
            self.num_goalpost_bodies = self.gym.get_asset_rigid_body_count(
                self.goalpost_asset
            )
        else:
            self.goalpost_asset = None
            self.num_goalpost_bodies = 0

        self.add_field_texture = bool(
            getattr(self.cfg.env, "add_field_texture", False)
        )
        if self.add_field_texture:
            configured_texture_file = getattr(
                self.cfg.env,
                "field_texture_file",
                "resources/textures/field.png",
            )
            field_texture_file = os.path.expanduser(configured_texture_file)
            if not os.path.isabs(field_texture_file):
                field_texture_file = os.path.join(project_root, field_texture_file)
            field_texture_file = os.path.abspath(field_texture_file)
            if not os.path.isfile(field_texture_file):
                raise FileNotFoundError(
                    f"Soccer field texture not found: {field_texture_file}"
                )

            configured_surface_file = getattr(
                self.cfg.env,
                "field_surface_asset_file",
                "resources/objects/soccer_field/soccer_field.urdf",
            )
            field_surface_file = os.path.expanduser(configured_surface_file)
            if not os.path.isabs(field_surface_file):
                field_surface_file = os.path.join(project_root, field_surface_file)
            field_surface_file = os.path.abspath(field_surface_file)
            if not os.path.isfile(field_surface_file):
                raise FileNotFoundError(
                    f"Soccer field surface asset not found: {field_surface_file}"
                )

            field_length = float(getattr(self.cfg.env, "field_length", 8.0))
            field_width = float(getattr(self.cfg.env, "field_width", 5.0))
            texture_length_scale = float(
                getattr(self.cfg.env, "field_texture_length_scale", 1.0)
            )
            texture_width_scale = float(
                getattr(self.cfg.env, "field_texture_width_scale", 1.0)
            )
            if texture_length_scale <= 0.0 or texture_width_scale <= 0.0:
                raise ValueError(
                    "Field texture length/width scales must both be positive"
                )
            self.field_surface_thickness = max(
                float(getattr(self.cfg.env, "field_surface_thickness", 0.01)),
                0.001,
            )
            field_surface_options = gymapi.AssetOptions()
            field_surface_options.fix_base_link = True
            field_surface_options.disable_gravity = True
            self.field_surface_asset = self.gym.load_asset(
                self.sim,
                os.path.dirname(field_surface_file),
                os.path.basename(field_surface_file),
                field_surface_options,
            )
            field_surface_props = self.gym.get_asset_rigid_shape_properties(
                self.field_surface_asset
            )
            for shape_prop in field_surface_props:
                shape_prop.friction = float(self.cfg.terrain.static_friction)
                shape_prop.restitution = float(self.cfg.terrain.restitution)
            self.gym.set_asset_rigid_shape_properties(
                self.field_surface_asset, field_surface_props
            )
            self.field_texture_handle = self.gym.create_texture_from_file(
                self.sim, field_texture_file
            )
            if self.field_texture_handle < 0:
                raise RuntimeError(
                    f"Isaac Gym could not load soccer field texture: {field_texture_file}"
                )
            self.num_field_surface_bodies = self.gym.get_asset_rigid_body_count(
                self.field_surface_asset
            )
        else:
            self.field_surface_asset = None
            self.field_texture_handle = None
            self.field_surface_thickness = 0.0
            self.num_field_surface_bodies = 0

        self.add_field_markers = bool(getattr(self.cfg.env, "add_field_markers", False))
        self.num_field_markers = 0
        self.num_field_marker_bodies = 0
        self.field_marker_width = float(getattr(self.cfg.env, "field_marker_width", 0.04))
        self.field_marker_height = float(getattr(self.cfg.env, "field_marker_height", 0.03))
        self.field_marker_width = max(self.field_marker_width, 0.01)
        self.field_marker_height = max(self.field_marker_height, 0.01)
        if self.add_field_markers:
            marker_asset_options = gymapi.AssetOptions()
            marker_asset_options.fix_base_link = True
            marker_asset_options.disable_gravity = True
            marker_asset_options.density = 1000.0
            field_length = float(getattr(self.cfg.env, "field_length", 8.0))
            field_width = float(getattr(self.cfg.env, "field_width", 5.0))
            self.field_marker_long_asset = self.gym.create_box(
                self.sim,
                field_length + 2.0 * self.field_marker_width,
                self.field_marker_width,
                self.field_marker_height,
                marker_asset_options,
            )
            self.field_marker_short_asset = self.gym.create_box(
                self.sim,
                self.field_marker_width,
                field_width + 2.0 * self.field_marker_width,
                self.field_marker_height,
                marker_asset_options,
            )
            if self.add_goalposts:
                self.goal_marker_asset = None
                self.num_field_markers = 4
            else:
                self.goal_marker_asset = self.gym.create_box(
                    self.sim,
                    2.0 * self.field_marker_width,
                    2.0 * self.field_marker_width,
                    4.0 * self.field_marker_height,
                    marker_asset_options,
                )
                self.num_field_markers = 8
            self.num_field_marker_bodies = (
                2 * self.gym.get_asset_rigid_body_count(self.field_marker_long_asset)
                + 2 * self.gym.get_asset_rigid_body_count(self.field_marker_short_asset)
            )
            if self.goal_marker_asset is not None:
                self.num_field_marker_bodies += (
                    4 * self.gym.get_asset_rigid_body_count(self.goal_marker_asset)
                )
        else:
            self.field_marker_long_asset = None
            self.field_marker_short_asset = None
            self.goal_marker_asset = None

        self.num_robot_bodies = self.robot.get_num_bodies()
        self.num_robot_dof = self.robot.get_num_dof()
        self.num_robot_actuated_dof = self.robot.get_num_actuated_dof()
        self.num_bodies = self.num_robot_bodies
        self.num_dof = self.num_robots * self.num_robot_dof
        self.num_total_actuated_dof = self.num_robots * self.num_robot_actuated_dof
        self.num_actuated_dof = (
            self.num_total_actuated_dof
            if bool(getattr(self.cfg.env, "control_all_robots", False))
            else self.num_robot_actuated_dof
        )
        self.total_rigid_body_num = (
            self.num_robots * self.num_robot_bodies
            + self.num_object_bodies
            + self.num_static_opponent_bodies
            + self.num_field_surface_bodies
            + self.num_field_marker_bodies
            + self.num_goalpost_bodies
        )

        if self.cfg.terrain.mesh_type == "boxes":
            self.total_rigid_body_num += self.cfg.terrain.num_cols * self.cfg.terrain.num_rows

        self.robot_rigid_body_offset = 0
        self.other_robot_rigid_body_offset = self.num_robot_bodies
        self.object_rigid_body_offset = self.num_robots * self.num_robot_bodies
        self.static_opponent_rigid_body_offset = self.object_rigid_body_offset + self.num_object_bodies
        self.field_surface_rigid_body_offset = (
            self.static_opponent_rigid_body_offset + self.num_static_opponent_bodies
        )
        self.field_marker_rigid_body_offset = (
            self.field_surface_rigid_body_offset + self.num_field_surface_bodies
        )
        self.goalpost_rigid_body_offset = (
            self.field_marker_rigid_body_offset + self.num_field_marker_bodies
        )

        self.ball_init_pose = gymapi.Transform()
        self.ball_init_pose.p = gymapi.Vec3(*self.object_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        robot_dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.dof_names = list(robot_dof_names) * self.num_robots
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        teammate_init_pos = list(
            getattr(
                self.cfg.env,
                "teammate_init_pos",
                getattr(self.cfg.env, "opponent_init_pos", [-1.0, 0.0, float(self.cfg.init_state.pos[2])]),
            )
        )
        if len(teammate_init_pos) != 3:
            raise ValueError(f"cfg.env.teammate_init_pos must have 3 values, got {teammate_init_pos}")
        teammate_init_yaw = getattr(
            self.cfg.env,
            "teammate_init_yaw",
            getattr(self.cfg.env, "opponent_init_yaw", np.pi),
        )
        teammate_init_state_list = (
            teammate_init_pos
            + [0.0, 0.0, 0.0, 1.0]
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.teammate_base_init_state = to_torch(teammate_init_state_list, device=self.device, requires_grad=False)
        self.opponent_base_init_state = self.teammate_base_init_state
        teammate_start_pose = gymapi.Transform()
        teammate_start_pose.p = gymapi.Vec3(*self.teammate_base_init_state[:3])
        teammate_start_pose.r = gymapi.Quat.from_euler_zyx(teammate_init_yaw, 0.0, 0.0)

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._get_env_origins(torch.arange(self.num_envs, device=self.device), self.cfg)
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.robot_actor_handles = []
        self.other_robot_actor_handles = []
        self.robot_actor_handles_all = []
        self.object_actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []
        self.robot_actor_idxs = []
        self.other_robot_actor_idxs = []
        self.robot_actor_idxs_all = []
        self.object_actor_idxs = []
        self.static_opponent_actor_handles = []
        self.static_opponent_actor_idxs = []
        self.field_marker_actor_handles = []
        self.field_marker_actor_idxs = []
        self.field_surface_actor_handles = []
        self.field_surface_actor_idxs = []
        self.goalpost_actor_handles = []
        self.goalpost_actor_idxs = []

        self.object_rigid_body_idxs = []
        self.static_opponent_rigid_body_idxs = []
        self.feet_rigid_body_idxs = []
        self.robot_rigid_body_idxs = []
        self.robot_rigid_body_idxs_all = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
        self._randomize_gravity()
        self._randomize_ball_drag()

        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos += self.base_init_state[:3]
            pos[0:1] += torch_rand_float(
                -self.cfg.terrain.x_init_range,
                self.cfg.terrain.x_init_range,
                (1, 1),
                device=self.device,
            ).squeeze(1)
            pos[1:2] += torch_rand_float(
                -self.cfg.terrain.y_init_range,
                self.cfg.terrain.y_init_range,
                (1, 1),
                device=self.device,
            ).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            dof_props = self._process_dof_props(dof_props_asset, i)
            env_robot_handles = []
            env_robot_actor_idxs = []
            team_size = int(getattr(self.cfg.env, "num_team_robots", self.num_robots))
            opponent_color_values = list(
                getattr(self.cfg.env, "opponent_team_color", [0.85, 0.10, 0.10])
            )
            if len(opponent_color_values) != 3:
                raise ValueError(
                    "cfg.env.opponent_team_color must contain three RGB values, "
                    f"got {opponent_color_values}"
                )
            opponent_color = gymapi.Vec3(
                *(float(value) for value in opponent_color_values)
            )
            for robot_slot in range(self.num_robots):
                robot_pose = start_pose if robot_slot == 0 else gymapi.Transform()
                if robot_slot > 0:
                    robot_pos = self.env_origins[i].clone() + self.teammate_base_init_state[:3]
                    robot_pos[1] += 0.6 * (robot_slot - 1)
                    robot_pose.p = gymapi.Vec3(*robot_pos)
                    robot_pose.r = gymapi.Quat.from_euler_zyx(
                        teammate_init_yaw, 0.0, 0.0
                    )
                robot_handle = self.gym.create_actor(
                    env_handle,
                    self.robot_asset,
                    robot_pose,
                    f"robot_{robot_slot}",
                    i,
                    self.cfg.asset.self_collisions,
                    0,
                )
                for body_name in body_names:
                    rigid_body_idx = self.gym.find_actor_rigid_body_handle(
                        env_handle, robot_handle, body_name
                    )
                    if robot_slot == 0:
                        self.robot_rigid_body_idxs.append(rigid_body_idx)
                    self.robot_rigid_body_idxs_all.append(rigid_body_idx)
                self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
                body_props = self.gym.get_actor_rigid_body_properties(
                    env_handle, robot_handle
                )
                body_props = self._process_rigid_body_props(body_props, i)
                self.gym.set_actor_rigid_body_properties(
                    env_handle, robot_handle, body_props, recomputeInertia=True
                )
                if robot_slot >= team_size:
                    for body_index in range(self.num_robot_bodies):
                        self.gym.set_rigid_body_color(
                            env_handle,
                            robot_handle,
                            body_index,
                            gymapi.MESH_VISUAL,
                            opponent_color,
                        )
                env_robot_handles.append(robot_handle)
                env_robot_actor_idxs.append(
                    self.gym.get_actor_index(
                        env_handle, robot_handle, gymapi.DOMAIN_SIM
                    )
                )

            self.robot_actor_handles.append(env_robot_handles[0])
            self.other_robot_actor_handles.append(
                env_robot_handles[1] if self.num_robots > 1 else env_robot_handles[0]
            )
            self.robot_actor_handles_all.append(env_robot_handles)
            self.robot_actor_idxs.append(env_robot_actor_idxs[0])
            self.other_robot_actor_idxs.append(
                env_robot_actor_idxs[1] if self.num_robots > 1 else env_robot_actor_idxs[0]
            )
            self.robot_actor_idxs_all.append(env_robot_actor_idxs)

            if self.cfg.env.add_balls:
                ball_rigid_shape_props = self._process_ball_rigid_shape_props(ball_rigid_shape_props_asset, i)
                self.gym.set_asset_rigid_shape_properties(self.ball_asset, ball_rigid_shape_props)
                ball_handle = self.gym.create_actor(env_handle, self.ball_asset, self.ball_init_pose, "ball", i, 0)
                color = gymapi.Vec3(1, 1, 0)
                self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                ball_idx = self.gym.get_actor_rigid_body_index(env_handle, ball_handle, 0, gymapi.DOMAIN_SIM)
                ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
                ball_body_props[0].mass = self.cfg.ball.mass * (np.random.rand() * 0.3 + 0.5)
                self.gym.set_actor_rigid_body_properties(
                    env_handle,
                    ball_handle,
                    ball_body_props,
                    recomputeInertia=True,
                )
                self.object_actor_handles.append(ball_handle)
                self.object_rigid_body_idxs.append(ball_idx)
                self.object_actor_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))

            static_opponent_handles = []
            static_opponent_actor_idxs = []
            if self.num_static_opponents > 0:
                for opponent_idx in range(self.num_static_opponents):
                    opponent_pose = gymapi.Transform()
                    opponent_pose.p = gymapi.Vec3(
                        float(self.env_origins[i, 0].item()) + 0.75 * (opponent_idx + 1),
                        float(self.env_origins[i, 1].item()) + 1.25,
                        0.5 * self.static_opponent_size[2],
                    )
                    opponent_handle = self.gym.create_actor(
                        env_handle,
                        self.static_opponent_asset,
                        opponent_pose,
                        f"static_opponent_{opponent_idx}",
                        i,
                        0,
                        0,
                    )
                    color = gymapi.Vec3(0.85, 0.15, 0.10)
                    self.gym.set_rigid_body_color(
                        env_handle,
                        opponent_handle,
                        0,
                        gymapi.MESH_VISUAL_AND_COLLISION,
                        color,
                    )
                    static_opponent_handles.append(opponent_handle)
                    static_opponent_actor_idxs.append(
                        self.gym.get_actor_index(env_handle, opponent_handle, gymapi.DOMAIN_SIM)
                    )
                    self.static_opponent_rigid_body_idxs.append(
                        self.gym.get_actor_rigid_body_index(env_handle, opponent_handle, 0, gymapi.DOMAIN_SIM)
                    )
            self.static_opponent_actor_handles.append(static_opponent_handles)
            self.static_opponent_actor_idxs.append(static_opponent_actor_idxs)

            if self.add_field_texture:
                field_surface_pose = gymapi.Transform()
                surface_offset = float(
                    getattr(self.cfg.env, "field_surface_offset", 0.002)
                )
                field_surface_pose.p = gymapi.Vec3(
                    float(self.env_origins[i, 0].item()),
                    float(self.env_origins[i, 1].item()),
                    float(self.env_origins[i, 2].item())
                    - 0.5 * self.field_surface_thickness
                    + surface_offset,
                )
                field_surface_handle = self.gym.create_actor(
                    env_handle,
                    self.field_surface_asset,
                    field_surface_pose,
                    "soccer_field",
                    i,
                    0,
                    0,
                )
                self.gym.set_rigid_body_texture(
                    env_handle,
                    field_surface_handle,
                    0,
                    gymapi.MESH_VISUAL,
                    self.field_texture_handle,
                )
                self.field_surface_actor_handles.append(field_surface_handle)
                self.field_surface_actor_idxs.append(
                    self.gym.get_actor_index(
                        env_handle, field_surface_handle, gymapi.DOMAIN_SIM
                    )
                )

            field_marker_handles = []
            field_marker_actor_idxs = []
            if self.add_field_markers:
                field_marker_specs = self._get_field_marker_specs(i)
                for marker_idx, (asset, pose, color) in enumerate(field_marker_specs):
                    marker_handle = self.gym.create_actor(
                        env_handle,
                        asset,
                        pose,
                        f"field_marker_{marker_idx}",
                        i,
                        0,
                        0,
                    )
                    self.gym.set_rigid_body_color(
                        env_handle,
                        marker_handle,
                        0,
                        gymapi.MESH_VISUAL_AND_COLLISION,
                        color,
                    )
                    field_marker_handles.append(marker_handle)
                    field_marker_actor_idxs.append(
                        self.gym.get_actor_index(env_handle, marker_handle, gymapi.DOMAIN_SIM)
                    )
            self.field_marker_actor_handles.append(field_marker_handles)
            self.field_marker_actor_idxs.append(field_marker_actor_idxs)

            if self.add_goalposts:
                goalpost_pose = gymapi.Transform()
                goalpost_pose.p = gymapi.Vec3(
                    float(self.env_origins[i, 0].item()),
                    float(self.env_origins[i, 1].item()),
                    float(self.env_origins[i, 2].item()),
                )
                goalpost_handle = self.gym.create_actor(
                    env_handle,
                    self.goalpost_asset,
                    goalpost_pose,
                    "goalposts",
                    i,
                    0,
                    0,
                )
                self.goalpost_actor_handles.append(goalpost_handle)
                self.goalpost_actor_idxs.append(
                    self.gym.get_actor_index(
                        env_handle, goalpost_handle, gymapi.DOMAIN_SIM
                    )
                )

            self.envs.append(env_handle)

        self.robot_actor_idxs = torch.as_tensor(self.robot_actor_idxs, device=self.device, dtype=torch.long)
        self.other_robot_actor_idxs = torch.as_tensor(self.other_robot_actor_idxs, device=self.device, dtype=torch.long)
        self.robot_actor_idxs_all = torch.as_tensor(self.robot_actor_idxs_all, device=self.device, dtype=torch.long)
        self.all_robot_actor_idxs = self.robot_actor_idxs_all
        self.object_actor_idxs = torch.as_tensor(self.object_actor_idxs, device=self.device, dtype=torch.long)
        self.object_rigid_body_idxs = torch.as_tensor(self.object_rigid_body_idxs, device=self.device, dtype=torch.long)
        self.static_opponent_actor_idxs = torch.as_tensor(
            self.static_opponent_actor_idxs,
            device=self.device,
            dtype=torch.long,
        ).view(self.num_envs, self.num_static_opponents)
        self.static_opponent_rigid_body_idxs = torch.as_tensor(
            self.static_opponent_rigid_body_idxs,
            device=self.device,
            dtype=torch.long,
        )
        self.field_marker_actor_idxs = torch.as_tensor(
            self.field_marker_actor_idxs,
            device=self.device,
            dtype=torch.long,
        ).view(self.num_envs, self.num_field_markers)
        self.field_surface_actor_idxs = torch.as_tensor(
            self.field_surface_actor_idxs,
            device=self.device,
            dtype=torch.long,
        ).view(self.num_envs, int(self.add_field_texture))
        self.goalpost_actor_idxs = torch.as_tensor(
            self.goalpost_actor_idxs,
            device=self.device,
            dtype=torch.long,
        ).view(self.num_envs, int(self.add_goalposts))

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.other_feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, feet_name in enumerate(feet_names):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.robot_actor_handles[0],
                feet_name,
            )
            self.other_feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.other_robot_actor_handles[0],
                feet_name,
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.other_penalised_contact_indices = torch.zeros_like(self.penalised_contact_indices)
        for i, contact_name in enumerate(penalized_contact_names):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.robot_actor_handles[0],
                contact_name,
            )
            self.other_penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.other_robot_actor_handles[0],
                contact_name,
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.other_termination_contact_indices = torch.zeros_like(self.termination_contact_indices)
        for i, contact_name in enumerate(termination_contact_names):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.robot_actor_handles[0],
                contact_name,
            )
            self.other_termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0],
                self.other_robot_actor_handles[0],
                contact_name,
            )

        self.other_feet_indices_full = self.other_feet_indices.clone()
        self.other_penalised_contact_indices_full = self.other_penalised_contact_indices.clone()
        self.other_termination_contact_indices_full = self.other_termination_contact_indices.clone()
        self.other_feet_indices = self._rigid_body_indices_to_robot_local(
            self.other_feet_indices,
            self.other_robot_rigid_body_offset,
            "other_feet_indices",
        )
        self.other_penalised_contact_indices = self._rigid_body_indices_to_robot_local(
            self.other_penalised_contact_indices,
            self.other_robot_rigid_body_offset,
            "other_penalised_contact_indices",
        )
        self.other_termination_contact_indices = self._rigid_body_indices_to_robot_local(
            self.other_termination_contact_indices,
            self.other_robot_rigid_body_offset,
            "other_termination_contact_indices",
        )

        self.initialize_sensors()

        if self.cfg.perception.compute_segmentation or self.cfg.perception.compute_rgb or self.cfg.perception.compute_depth:
            self.initialize_cameras(range(self.num_envs))

        if self.cfg.perception.measure_heights:
            from dribblebot.sensors.heightmap_sensor import HeightmapSensor

            self.heightmap_sensor = HeightmapSensor(self)

        if self.cfg.env.record_video:
            from dribblebot.sensors.floating_camera_sensor import FloatingCameraSensor

            self.rendering_camera = FloatingCameraSensor(self)

        from dribblebot.utils.logger import Logger

        self.logger = Logger(self)

        self.video_writer = None
        self.video_frames = []
        self.complete_video_frames = []

    def _rigid_body_indices_to_robot_local(self, indices, robot_body_offset, label):
        local = indices.clone()
        if int(local.numel()) == 0:
            return local

        local = torch.where(local >= robot_body_offset, local - robot_body_offset, local)
        invalid = torch.logical_or(local < 0, local >= self.num_robot_bodies)
        if bool(torch.any(invalid).detach().cpu().item()):
            raise ValueError(
                f"{label} contains invalid local rigid-body indices after offset conversion: "
                f"{local.detach().cpu().tolist()}"
            )
        return local

    def _get_field_marker_specs(self, env_id):
        origin_x = float(self.env_origins[env_id, 0].item())
        origin_y = float(self.env_origins[env_id, 1].item())
        origin_z = float(self.env_origins[env_id, 2].item())
        half_length = 0.5 * float(getattr(self.cfg.env, "field_length", 8.0))
        half_width = 0.5 * float(getattr(self.cfg.env, "field_width", 5.0))
        goal_x = float(getattr(self.cfg.env, "team_goal_x", half_length))
        goal_half_width = float(getattr(self.cfg.env, "team_goal_half_width", 1.0))
        marker_half_width = 0.5 * self.field_marker_width
        marker_half_height = 0.5 * self.field_marker_height
        ball_radius = float(getattr(self.cfg.ball, "radius", 0.09))
        outside_clearance = ball_radius + 0.02

        boundary_color = gymapi.Vec3(0.92, 0.92, 0.88)
        learning_target_color = gymapi.Vec3(0.0, 0.85, 0.25)
        opponent_target_color = gymapi.Vec3(0.85, 0.10, 0.10)

        def pose_at(x, y, z):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(origin_x + x, origin_y + y, origin_z + z)
            return pose

        boundary_z = marker_half_height
        goal_z = 2.0 * self.field_marker_height
        specs = [
            (
                self.field_marker_long_asset,
                pose_at(0.0, half_width + marker_half_width + outside_clearance, boundary_z),
                boundary_color,
            ),
            (
                self.field_marker_long_asset,
                pose_at(0.0, -half_width - marker_half_width - outside_clearance, boundary_z),
                boundary_color,
            ),
            (
                self.field_marker_short_asset,
                pose_at(half_length + marker_half_width + outside_clearance, 0.0, boundary_z),
                boundary_color,
            ),
            (
                self.field_marker_short_asset,
                pose_at(-half_length - marker_half_width - outside_clearance, 0.0, boundary_z),
                boundary_color,
            ),
        ]
        if self.goal_marker_asset is None:
            return specs
        specs.extend([
            (
                self.goal_marker_asset,
                pose_at(
                    goal_x + marker_half_width + outside_clearance,
                    goal_half_width + marker_half_width + outside_clearance,
                    goal_z,
                ),
                learning_target_color,
            ),
            (
                self.goal_marker_asset,
                pose_at(
                    goal_x + marker_half_width + outside_clearance,
                    -goal_half_width - marker_half_width - outside_clearance,
                    goal_z,
                ),
                learning_target_color,
            ),
            (
                self.goal_marker_asset,
                pose_at(
                    -goal_x - marker_half_width - outside_clearance,
                    goal_half_width + marker_half_width + outside_clearance,
                    goal_z,
                ),
                opponent_target_color,
            ),
            (
                self.goal_marker_asset,
                pose_at(
                    -goal_x - marker_half_width - outside_clearance,
                    -goal_half_width - marker_half_width - outside_clearance,
                    goal_z,
                ),
                opponent_target_color,
            ),
        ])
        return specs

    def _get_high_level_camera_pose(self):
        field_length = float(getattr(self.cfg.env, "field_length", 8.0))
        field_width = float(getattr(self.cfg.env, "field_width", 5.0))
        default_height = 1.2 * max(field_length, field_width)
        camera_height = float(getattr(self.cfg.env, "high_level_camera_height", default_height))
        camera_height = max(camera_height, default_height, 3.0)
        origin = self.env_origins[0]
        target_loc = [
            float(origin[0].item()),
            float(origin[1].item()),
            float(origin[2].item()),
        ]
        cam_distance = [1.0e-3, 0.0, camera_height]
        return target_loc, cam_distance

    def render(self, mode="rgb_array", target_loc=None, cam_distance=None):
        if bool(getattr(self.cfg.env, "high_level_control", False)) and target_loc is None and cam_distance is None:
            target_loc, cam_distance = self._get_high_level_camera_pose()
        return super().render(mode=mode, target_loc=target_loc, cam_distance=cam_distance)

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            if bool(getattr(self.cfg.env, "high_level_control", False)):
                target_loc, cam_distance = self._get_high_level_camera_pose()
            else:
                bx = self.root_states[self.robot_actor_idxs[0], 0]
                by = self.root_states[self.robot_actor_idxs[0], 1]
                bz = self.root_states[self.robot_actor_idxs[0], 2]
                target_loc = [bx, by, bz]
                cam_distance = [0, -1.0, 1.0]
            self.rendering_camera.set_position(target_loc, cam_distance)
            self.video_frame = self.rendering_camera.get_observation()
            self.video_frames.append(self.video_frame)

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            robot_dof_count = len(props)
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for robot_idx in range(self.num_robots):
                offset = robot_idx * robot_dof_count
                for dof_idx in range(robot_dof_count):
                    target_idx = offset + dof_idx
                    self.dof_pos_limits[target_idx, 0] = props["lower"][dof_idx].item()
                    self.dof_pos_limits[target_idx, 1] = props["upper"][dof_idx].item()
                    self.dof_vel_limits[target_idx] = props["velocity"][dof_idx].item()
                    self.torque_limits[target_idx] = props["effort"][dof_idx].item()
                    midpoint = (self.dof_pos_limits[target_idx, 0] + self.dof_pos_limits[target_idx, 1]) / 2
                    radius = self.dof_pos_limits[target_idx, 1] - self.dof_pos_limits[target_idx, 0]
                    self.dof_pos_limits[target_idx, 0] = midpoint - 0.5 * radius * self.cfg.rewards.soft_dof_pos_limit
                    self.dof_pos_limits[target_idx, 1] = midpoint + 0.5 * radius * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _init_buffers(self):
        super()._init_buffers()

        rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        self.rigid_body_state_all = rigid_body_state[: self.num_envs * self.total_rigid_body_num, :].view(
            self.num_envs,
            self.total_rigid_body_num,
            13,
        )
        self.rigid_body_state = self.rigid_body_state_all[:, : self.num_robot_bodies, :]
        self.robot_rigid_body_state_all = self.rigid_body_state_all[
            :, : self.num_robots * self.num_robot_bodies, :
        ].view(self.num_envs, self.num_robots, self.num_robot_bodies, 13)
        compatibility_slot = 1 if self.num_robots > 1 else 0
        self.other_rigid_body_state = self.robot_rigid_body_state_all[
            :, compatibility_slot
        ]
        self.rigid_body_state_object = self.rigid_body_state_all[
            :,
            self.object_rigid_body_offset : self.object_rigid_body_offset + self.num_object_bodies,
            :,
        ]
        self.rigid_body_state_static_opponents = self.rigid_body_state_all[
            :,
            self.static_opponent_rigid_body_offset : self.static_opponent_rigid_body_offset + self.num_static_opponent_bodies,
            :,
        ]

        net_contact_forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        self.contact_forces_all = net_contact_forces[: self.num_envs * self.total_rigid_body_num, :].view(
            self.num_envs,
            self.total_rigid_body_num,
            3,
        )
        self.contact_forces = self.contact_forces_all[:, : self.num_robot_bodies, :]
        self.robot_contact_forces_all = self.contact_forces_all[
            :, : self.num_robots * self.num_robot_bodies, :
        ].view(self.num_envs, self.num_robots, self.num_robot_bodies, 3)
        self.other_contact_forces = self.robot_contact_forces_all[
            :, compatibility_slot
        ]
        self.static_opponent_contact_forces = self.contact_forces_all[
            :,
            self.static_opponent_rigid_body_offset : self.static_opponent_rigid_body_offset + self.num_static_opponent_bodies,
            :,
        ]

        self._refresh_two_robot_views()
        self.prev_object_pos_world_frame = self.object_pos_world_frame.clone()
        self.prev_object_lin_vel = self.object_lin_vel.clone()
        self.high_level_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.high_level_opponent_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.high_level_ball_off_border_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.high_level_obstacle_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.high_level_accidental_termination_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_high_level_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_high_level_opponent_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_high_level_ball_off_border_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_high_level_obstacle_contact_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_high_level_accidental_termination_buf = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.high_level_skill_ids = torch.zeros(self.num_envs, self.num_robots, dtype=torch.long, device=self.device)
        self.high_level_requested_skill_ids = torch.zeros(
            self.num_envs,
            self.num_robots,
            dtype=torch.long,
            device=self.device,
        )
        self.high_level_invalid_skill_mask = torch.zeros(
            self.num_envs,
            self.num_robots,
            dtype=torch.bool,
            device=self.device,
        )
        self.high_level_commands = torch.zeros(self.num_envs, self.num_robots, 3, dtype=torch.float, device=self.device)
        self.prev_high_level_robot_ball_distances = self._high_level_robot_ball_distances()
        if self.cfg.control.control_type == "actuator_net":
            self._install_multi_robot_actuator_network()

    def _install_multi_robot_actuator_network(self):
        actuator_path = (
            f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/"
            "../../resources/actuator_nets/unitree_go1.pt"
        )
        actuator_network = torch.jit.load(actuator_path, map_location=self.device)

        def eval_actuator_network(
            joint_pos,
            joint_pos_last,
            joint_pos_last_last,
            joint_vel,
            joint_vel_last,
            joint_vel_last_last,
        ):
            xs = torch.cat(
                (
                    joint_pos.unsqueeze(-1),
                    joint_pos_last.unsqueeze(-1),
                    joint_pos_last_last.unsqueeze(-1),
                    joint_vel.unsqueeze(-1),
                    joint_vel_last.unsqueeze(-1),
                    joint_vel_last_last.unsqueeze(-1),
                ),
                dim=-1,
            )
            torques = actuator_network(xs.reshape(-1, 6))
            return torques.reshape(self.num_envs, self.num_dof)

        self.actuator_network = eval_actuator_network
        self.joint_pos_err_last_last = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.joint_pos_err_last = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.joint_vel_last_last = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.joint_vel_last = torch.zeros((self.num_envs, self.num_dof), device=self.device)

    def _refresh_two_robot_views(self):
        self.robot_root_states_all = self.root_states[self.robot_actor_idxs_all.reshape(-1)].view(
            self.num_envs,
            self.num_robots,
            13,
        )
        compatibility_slot = 1 if self.num_robots > 1 else 0
        compatibility_roots = self.robot_root_states_all[:, compatibility_slot]
        self.other_base_pos = compatibility_roots[:, 0:3]
        self.other_base_quat = compatibility_roots[:, 3:7]
        self.other_base_lin_vel = quat_rotate_inverse(
            self.other_base_quat,
            compatibility_roots[:, 7:10],
        )
        self.other_base_ang_vel = quat_rotate_inverse(
            self.other_base_quat,
            compatibility_roots[:, 10:13],
        )
        self.other_projected_gravity = quat_rotate_inverse(self.other_base_quat, self.gravity_vec)
        if self.num_static_opponents > 0:
            self.static_opponent_root_states = self.root_states[self.static_opponent_actor_idxs.reshape(-1)].view(
                self.num_envs,
                self.num_static_opponents,
                13,
            )
        else:
            self.static_opponent_root_states = self.root_states.new_zeros((self.num_envs, 0, 13))

    def _high_level_robot_ball_distances(self):
        if not self.cfg.env.add_balls:
            return torch.zeros(self.num_envs, self.num_robots, dtype=torch.float, device=self.device)

        roots = self.root_states[self.robot_actor_idxs_all.reshape(-1)].view(self.num_envs, self.num_robots, 13)
        ball_xy = self.object_pos_world_frame[:, None, :2]
        return torch.norm(roots[:, :, :2] - ball_xy, dim=-1)

    def pre_physics_step(self):
        if hasattr(self, "last_high_level_goal_buf"):
            self.last_high_level_goal_buf[:] = False
            self.last_high_level_opponent_goal_buf[:] = False
            self.last_high_level_ball_off_border_buf[:] = False
            self.last_high_level_obstacle_contact_buf[:] = False
            self.last_high_level_accidental_termination_buf[:] = False
        if self.cfg.env.add_balls and hasattr(self, "object_pos_world_frame"):
            self.prev_object_pos_world_frame = self.object_pos_world_frame.clone()
            self.prev_object_lin_vel = self.object_lin_vel.clone()
            if hasattr(self, "prev_high_level_robot_ball_distances"):
                self.prev_high_level_robot_ball_distances = self._high_level_robot_ball_distances()
        super().pre_physics_step()

    def post_physics_step(self):
        super().post_physics_step()
        self._refresh_two_robot_views()

    def compute_reward(self):
        reward_attrs = {
            "dof_pos": self.dof_pos[:, :self.num_robot_dof],
            "dof_vel": self.dof_vel[:, :self.num_robot_dof],
            "last_dof_vel": self.last_dof_vel[:, :self.num_robot_dof],
            "torques": self.torques[:, :self.num_robot_dof],
            "joint_pos_target": self.joint_pos_target[:, :self.num_robot_dof],
            "last_joint_pos_target": self.last_joint_pos_target[:, :self.num_robot_dof],
            "last_last_joint_pos_target": self.last_last_joint_pos_target[:, :self.num_robot_dof],
            "default_dof_pos": self.default_dof_pos[:, :self.num_robot_dof],
            "dof_pos_limits": self.dof_pos_limits[:self.num_robot_dof],
        }
        originals = {name: getattr(self, name) for name in reward_attrs}
        try:
            for name, value in reward_attrs.items():
                setattr(self, name, value)
            super().compute_reward()
        finally:
            for name, value in originals.items():
                setattr(self, name, value)

    def check_termination(self):
        super().check_termination()
        base_reset_buf = self.reset_buf.clone()

        if int(self.termination_contact_indices.numel()) > 0:
            any_robot_contact_reset = torch.any(
                torch.norm(
                    self.robot_contact_forces_all[
                        :, :, self.termination_contact_indices, :
                    ],
                    dim=-1,
                )
                > 1.0,
                dim=(1, 2),
            )
            self.reset_buf = torch.logical_or(
                self.reset_buf, any_robot_contact_reset
            )

        if self.cfg.rewards.use_terminal_body_height:
            if torch.is_tensor(self.measured_heights):
                robot_heights = self.robot_root_states_all[:, :, 2]
                terrain_height = torch.mean(self.measured_heights, dim=1, keepdim=True)
                other_body_height = torch.any(
                    robot_heights - terrain_height
                    < self.cfg.rewards.terminal_body_height,
                    dim=1,
                )
            else:
                other_body_height = torch.any(
                    self.robot_root_states_all[:, :, 2]
                    < self.cfg.rewards.terminal_body_height,
                    dim=1,
                )
            self.reset_buf = torch.logical_or(self.reset_buf, other_body_height)

        if self.cfg.rewards.use_terminal_roll_pitch:
            other_projected_gravity = quat_rotate_inverse(
                self.robot_root_states_all[:, :, 3:7].reshape(-1, 4),
                self.gravity_vec[:, None, :]
                .expand(-1, self.num_robots, -1)
                .reshape(-1, 3),
            ).view(self.num_envs, self.num_robots, 3)
            other_body_ori = (
                torch.sum(torch.square(other_projected_gravity[:, :, :2]), dim=2)
                > self.cfg.rewards.terminal_body_ori
            ).any(dim=1)
            self.reset_buf = torch.logical_or(self.reset_buf, other_body_ori)

        self.high_level_goal_buf[:] = False
        self.high_level_opponent_goal_buf[:] = False
        self.high_level_ball_off_border_buf[:] = False
        self.high_level_obstacle_contact_buf[:] = False
        self.high_level_accidental_termination_buf[:] = False

        if self.cfg.env.add_balls and getattr(self.cfg.rewards, "use_high_level_match_termination", False):
            field_xy = self.object_pos_world_frame[:, :2] - self.env_origins[:, :2]
            half_length = 0.5 * float(getattr(self.cfg.env, "field_length", 8.0))
            half_width = 0.5 * float(getattr(self.cfg.env, "field_width", 5.0))
            goal_x = float(getattr(self.cfg.env, "team_goal_x", half_length))
            goal_half_width = float(getattr(self.cfg.env, "team_goal_half_width", 1.0))
            border_margin = float(getattr(self.cfg.rewards, "high_level_border_margin", 0.0))

            self.high_level_goal_buf[:] = (field_xy[:, 0] >= goal_x) & (torch.abs(field_xy[:, 1]) <= goal_half_width)
            self.high_level_opponent_goal_buf[:] = (field_xy[:, 0] <= -goal_x) & (
                torch.abs(field_xy[:, 1]) <= goal_half_width
            )
            outside_field = (
                (field_xy[:, 0] < -half_length - border_margin)
                | (field_xy[:, 0] > half_length + border_margin)
                | (torch.abs(field_xy[:, 1]) > half_width + border_margin)
            )
            self.high_level_ball_off_border_buf[:] = (
                outside_field & ~self.high_level_goal_buf & ~self.high_level_opponent_goal_buf
            )
            self.high_level_obstacle_contact_buf[:] = self._ball_touches_static_opponent()

            self.high_level_accidental_termination_buf[:] = (
                self.high_level_ball_off_border_buf
                | self.high_level_opponent_goal_buf
                | self.high_level_obstacle_contact_buf
                | (self.reset_buf & ~self.time_out_buf & ~self.high_level_goal_buf)
            )
            self.reset_buf = torch.logical_or(self.reset_buf, self.high_level_goal_buf)
            self.reset_buf = torch.logical_or(self.reset_buf, self.high_level_opponent_goal_buf)
            self.reset_buf = torch.logical_or(self.reset_buf, self.high_level_ball_off_border_buf)
            self.reset_buf = torch.logical_or(self.reset_buf, self.high_level_obstacle_contact_buf)
            self.high_level_accidental_termination_buf[:] = (
                self.high_level_accidental_termination_buf
                | (base_reset_buf & ~self.time_out_buf & ~self.high_level_goal_buf)
            )

    def _ball_touches_static_opponent(self):
        if self.num_static_opponents <= 0 or not self.cfg.env.add_balls:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        obstacle_states = self.root_states[self.static_opponent_actor_idxs.reshape(-1)].view(
            self.num_envs,
            self.num_static_opponents,
            13,
        )
        ball_xy = self.object_pos_world_frame[:, None, :2]
        delta = ball_xy - obstacle_states[:, :, :2]
        _, _, yaw = get_euler_xyz(obstacle_states[:, :, 3:7].reshape(-1, 4))
        yaw = yaw.view(self.num_envs, self.num_static_opponents)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        local_x = cos_yaw * delta[:, :, 0] + sin_yaw * delta[:, :, 1]
        local_y = -sin_yaw * delta[:, :, 0] + cos_yaw * delta[:, :, 1]
        ball_radius = float(getattr(self.cfg.ball, "radius", 0.0889))
        half_x = 0.5 * self.static_opponent_size[0] + ball_radius
        half_y = 0.5 * self.static_opponent_size[1] + ball_radius
        touch = (torch.abs(local_x) <= half_x) & (torch.abs(local_y) <= half_y)
        return torch.any(touch, dim=1)

    def reset_idx(self, env_ids):
        if len(env_ids) > 0 and hasattr(self, "last_high_level_goal_buf"):
            self.last_high_level_goal_buf[env_ids] = self.high_level_goal_buf[env_ids]
            self.last_high_level_opponent_goal_buf[env_ids] = self.high_level_opponent_goal_buf[env_ids]
            self.last_high_level_ball_off_border_buf[env_ids] = self.high_level_ball_off_border_buf[env_ids]
            self.last_high_level_obstacle_contact_buf[env_ids] = self.high_level_obstacle_contact_buf[env_ids]
            self.last_high_level_accidental_termination_buf[env_ids] = self.high_level_accidental_termination_buf[env_ids]
        super().reset_idx(env_ids)
        if len(env_ids) == 0 or not hasattr(self, "high_level_goal_buf"):
            return

        self.high_level_goal_buf[env_ids] = False
        self.high_level_opponent_goal_buf[env_ids] = False
        self.high_level_ball_off_border_buf[env_ids] = False
        self.high_level_obstacle_contact_buf[env_ids] = False
        self.high_level_accidental_termination_buf[env_ids] = False
        self.high_level_skill_ids[env_ids] = 0
        self.high_level_requested_skill_ids[env_ids] = 0
        self.high_level_invalid_skill_mask[env_ids] = False
        self.high_level_commands[env_ids] = 0.0
        if self.cfg.env.add_balls:
            self.prev_object_pos_world_frame[env_ids] = self.object_pos_world_frame[env_ids]
            self.prev_object_lin_vel[env_ids] = self.object_lin_vel[env_ids]
            self.prev_high_level_robot_ball_distances[env_ids] = self._high_level_robot_ball_distances()[env_ids]

    def _apply_drag_force(self, force_tensor):
        if self.cfg.domain_rand.randomize_ball_drag:
            force_tensor[:, self.object_rigid_body_offset, :2] = (
                -self.ball_drags * torch.square(self.object_lin_vel[:, :2]) * torch.sign(self.object_lin_vel[:, :2])
            )

    def _reset_dofs(self, env_ids, cfg):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5,
            1.5,
            (len(env_ids), self.num_dof),
            device=self.device,
        )
        self.dof_vel[env_ids] = 0.0

        all_subject_env_ids = self.robot_actor_idxs_all[env_ids].reshape(-1).to(device=self.device)
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_subject_env_ids_int32),
            len(all_subject_env_ids_int32),
        )

    def _clamp_range_to_field(self, value_range, axis):
        value_range = [float(value_range[0]), float(value_range[1])]
        half_extent = 0.5 * float(getattr(self.cfg.env, "field_length", 8.0))
        if axis == "y":
            half_extent = 0.5 * float(getattr(self.cfg.env, "field_width", 5.0))
        margin = float(getattr(self.cfg.env, "field_margin", 0.4))
        clamped = [max(value_range[0], -half_extent + margin), min(value_range[1], half_extent - margin)]
        if clamped[0] > clamped[1]:
            raise ValueError(
                f"Invalid {axis}-range {value_range} for field half extent {half_extent} and margin {margin}."
            )
        return clamped

    def _sample_field_xy(self, env_ids, x_range, y_range):
        x_range = self._clamp_range_to_field(x_range, "x")
        y_range = self._clamp_range_to_field(y_range, "y")
        xy_local = torch.cat(
            (
                torch_rand_float(x_range[0], x_range[1], (len(env_ids), 1), device=self.device),
                torch_rand_float(y_range[0], y_range[1], (len(env_ids), 1), device=self.device),
            ),
            dim=1,
        )
        return self.env_origins[env_ids, 0:2] + xy_local

    def _sample_field_xy_with_clearance(self, env_ids, x_range, y_range, protected_xy, min_clearance):
        xy_world = self._sample_field_xy(env_ids, x_range, y_range)
        if not protected_xy or min_clearance <= 0.0:
            return xy_world

        for _ in range(20):
            too_close = torch.zeros(len(env_ids), device=self.device, dtype=torch.bool)
            for protected in protected_xy:
                too_close = torch.logical_or(
                    too_close,
                    torch.norm(xy_world - protected, dim=1) < min_clearance,
                )
            if not bool(torch.any(too_close).detach().cpu().item()):
                break

            resample_env_ids = env_ids[too_close]
            xy_world[too_close] = self._sample_field_xy(resample_env_ids, x_range, y_range)
        return xy_world

    def _apply_high_level_near_ball_init(self, env_ids, default_ball_xy):
        """Place the ball in front of one robot for a subset of match resets."""

        probability = float(
            getattr(self.cfg.env, "high_level_near_ball_init_probability", 0.0)
        )
        if probability <= 0.0 or len(env_ids) == 0:
            return default_ball_xy

        probability = min(probability, 1.0)
        distance_range = getattr(
            self.cfg.env,
            "high_level_near_ball_init_distance_range",
            [0.4, 0.95],
        )
        min_distance = float(distance_range[0])
        max_distance = float(distance_range[1])
        if min_distance <= 0.0 or max_distance < min_distance:
            raise ValueError(
                "high_level_near_ball_init_distance_range must be positive and ordered, "
                f"got {distance_range!r}."
            )

        angle_range = getattr(
            self.cfg.env,
            "high_level_near_ball_init_angle_range",
            [-0.35, 0.35],
        )
        min_angle = float(angle_range[0])
        max_angle = float(angle_range[1])
        if max_angle < min_angle:
            raise ValueError(
                "high_level_near_ball_init_angle_range must be ordered, "
                f"got {angle_range!r}."
            )

        num_envs = len(env_ids)
        robot_actor_ids = self.robot_actor_idxs_all[env_ids]
        robot_states = self.root_states[robot_actor_ids.reshape(-1)].view(num_envs, self.num_robots, 13)
        attacker_slots = torch.randint(
            0,
            self.num_robots,
            (num_envs,),
            device=self.device,
        )
        row_ids = torch.arange(num_envs, device=self.device)
        attacker_states = robot_states[row_ids, attacker_slots]
        _, _, attacker_yaw = get_euler_xyz(attacker_states[:, 3:7])
        sample_device = str(self.device)
        relative_angle = torch_rand_float(
            min_angle,
            max_angle,
            (num_envs, 1),
            device=sample_device,
        ).squeeze(1)
        distance = torch_rand_float(
            min_distance,
            max_distance,
            (num_envs, 1),
            device=sample_device,
        ).squeeze(1)
        world_angle = attacker_yaw + relative_angle
        candidate_ball_xy = attacker_states[:, :2] + distance.unsqueeze(1) * torch.stack(
            (torch.cos(world_angle), torch.sin(world_angle)),
            dim=1,
        )

        local_candidate = candidate_ball_xy - self.env_origins[env_ids, :2]
        half_length = 0.5 * float(getattr(self.cfg.env, "field_length", 8.0))
        half_width = 0.5 * float(getattr(self.cfg.env, "field_width", 5.0))
        margin = float(getattr(self.cfg.env, "field_margin", 0.4))
        inside_field = (
            (torch.abs(local_candidate[:, 0]) <= half_length - margin)
            & (torch.abs(local_candidate[:, 1]) <= half_width - margin)
        )

        teammate_clearance = float(getattr(self.cfg.env, "match_init_min_clearance", 0.75))
        robot_clearances = torch.norm(
            candidate_ball_xy[:, None, :] - robot_states[:, :, :2], dim=2
        )
        non_attacker = torch.ones_like(robot_clearances, dtype=torch.bool)
        non_attacker[row_ids, attacker_slots] = False
        clear_of_teammate = (
            (robot_clearances >= teammate_clearance) | ~non_attacker
        ).all(dim=1)
        use_near_init = (
            (torch.rand(num_envs, device=self.device) < probability)
            & inside_field
            & clear_of_teammate
        )
        return torch.where(use_near_init.unsqueeze(1), candidate_ball_xy, default_ball_xy)

    def _reset_static_opponent_states(self, env_ids, cfg):
        if self.num_static_opponents <= 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        num_envs = len(env_ids)
        actor_ids = self.static_opponent_actor_idxs[env_ids]
        x_range = getattr(cfg.env, "static_opponent_x_range", [-2.0, 2.0])
        y_range = getattr(cfg.env, "static_opponent_y_range", [-1.5, 1.5])
        yaw_range = getattr(cfg.env, "static_opponent_yaw_range", [-np.pi, np.pi])
        min_clearance = float(getattr(cfg.env, "static_opponent_min_clearance", 0.75))
        half_height = 0.5 * float(self.static_opponent_size[2])

        protected_xy = [
            self.root_states[self.robot_actor_idxs_all[env_ids, robot], 0:2]
            for robot in range(self.num_robots)
        ]
        if self.cfg.env.add_balls:
            protected_xy.append(self.root_states[self.object_actor_idxs[env_ids], 0:2])

        for opponent_idx in range(self.num_static_opponents):
            opponent_protected_xy = list(protected_xy)
            if opponent_idx > 0:
                previous_actor_ids = actor_ids[:, :opponent_idx]
                previous_xy = self.root_states[previous_actor_ids.reshape(-1), 0:2].view(
                    num_envs,
                    opponent_idx,
                    2,
                )
                for previous_idx in range(opponent_idx):
                    opponent_protected_xy.append(previous_xy[:, previous_idx, :])

            xy_world = self._sample_field_xy_with_clearance(
                env_ids,
                x_range,
                y_range,
                opponent_protected_xy,
                min_clearance,
            )

            yaw = torch_rand_float(yaw_range[0], yaw_range[1], (num_envs, 1), device=self.device).squeeze(1)
            current_actor_ids = actor_ids[:, opponent_idx]
            self.root_states[current_actor_ids] = 0.0
            self.root_states[current_actor_ids, 0:2] = xy_world
            self.root_states[current_actor_ids, 2] = self.env_origins[env_ids, 2] + half_height
            self.root_states[current_actor_ids, 3:7] = quat_from_euler_xyz(
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                yaw,
            )
            self.root_states[current_actor_ids, 7:13] = 0.0

        return actor_ids.reshape(-1).to(device=self.device)

    def _reset_root_states(self, env_ids, cfg):
        robot_actor_ids = self.robot_actor_idxs_all[env_ids].to(device=self.device)
        randomize_match_init = bool(getattr(cfg.env, "randomize_match_init", False))
        protected_xy = []
        min_clearance = float(getattr(cfg.env, "match_init_min_clearance", 0.75))
        team_size = int(getattr(cfg.env, "num_team_robots", self.num_robots))
        if team_size < 1 or 2 * team_size != self.num_robots:
            team_size = self.num_robots
        for robot_slot in range(self.num_robots):
            actor_ids = robot_actor_ids[:, robot_slot]
            self.root_states[actor_ids] = self.base_init_state
            if randomize_match_init:
                if robot_slot >= team_size and team_size != self.num_robots:
                    prefix = "opponent"
                else:
                    prefix = "robot" if robot_slot == 0 else "teammate"
                xy = self._sample_field_xy_with_clearance(
                    env_ids,
                    getattr(cfg.env, f"{prefix}_init_x_range", [-3.2, -0.8]),
                    getattr(cfg.env, f"{prefix}_init_y_range", [-1.8, 1.8]),
                    protected_xy,
                    min_clearance if protected_xy else 0.0,
                )
                yaw_range = getattr(
                    cfg.env,
                    (
                        "opponent_yaw_range"
                        if robot_slot >= team_size and team_size != self.num_robots
                        else ("robot_yaw_init_range" if robot_slot == 0 else "teammate_yaw_range")
                    ),
                    [-np.pi, np.pi],
                )
                yaw = torch_rand_float(
                    yaw_range[0], yaw_range[1], (len(env_ids), 1), device=self.device
                ).squeeze(1)
                self.root_states[actor_ids, 0:2] = xy
                self.root_states[actor_ids, 2] = (
                    self.env_origins[env_ids, 2] + self.base_init_state[2]
                )
                self.root_states[actor_ids, 7:13] = 0.0
            else:
                offset = (
                    self.base_init_state[:3]
                    if robot_slot == 0
                    else self.teammate_base_init_state[:3]
                ).clone()
                offset[1] += 0.6 * max(robot_slot - 1, 0)
                self.root_states[actor_ids, :3] = self.env_origins[env_ids] + offset
                if robot_slot == 0:
                    yaw = 2 * (
                        torch.rand(len(env_ids), device=self.device) - 0.5
                    ) * float(cfg.terrain.yaw_init_range)
                    self.root_states[actor_ids, 7:13] = torch_rand_float(
                        -0.5, 0.5, (len(env_ids), 6), device=self.device
                    )
                else:
                    base_yaw = float(
                        getattr(
                            cfg.env,
                            "teammate_init_yaw",
                            getattr(cfg.env, "opponent_init_yaw", np.pi),
                        )
                    )
                    yaw_range = float(
                        getattr(
                            cfg.env,
                            "teammate_yaw_init_range",
                            getattr(cfg.env, "opponent_yaw_init_range", 0.0),
                        )
                    )
                    yaw = base_yaw + 2 * (
                        torch.rand(len(env_ids), device=self.device) - 0.5
                    ) * yaw_range
                    self.root_states[actor_ids, 7:13] = 0.0
            self.root_states[actor_ids, 3:7] = quat_from_euler_xyz(
                torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
            )
            protected_xy.append(self.root_states[actor_ids, 0:2])

        if self.cfg.env.add_balls:
            object_env_ids = self.object_actor_idxs[env_ids].to(device=self.device)
            self.root_states[object_env_ids] = self.object_init_state
            self.root_states[object_env_ids, :3] += self.env_origins[env_ids]

            if randomize_match_init:
                ball_xy = self._sample_field_xy_with_clearance(
                    env_ids,
                    getattr(cfg.env, "ball_init_x_range", [-2.0, 2.0]),
                    getattr(cfg.env, "ball_init_y_range", [-1.6, 1.6]),
                    protected_xy,
                    min_clearance,
                )
                ball_xy = self._apply_high_level_near_ball_init(env_ids, ball_xy)
                self.root_states[object_env_ids, 0:2] = ball_xy
                self.root_states[object_env_ids, 2] = self.env_origins[env_ids, 2] + self.object_init_state[2]
                self.root_states[object_env_ids, 7:13] = 0.0
            else:
                self.root_states[object_env_ids, 0:3] += 2 * (
                    torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False) - 0.5
                ) * torch.tensor(cfg.ball.init_pos_range, device=self.device, requires_grad=False)
                self.root_states[object_env_ids, 7:10] += 2 * (
                    torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False) - 0.5
                ) * torch.tensor(cfg.ball.init_vel_range, device=self.device, requires_grad=False)

        static_opponent_env_ids = self._reset_static_opponent_states(env_ids, cfg)

        all_subject_env_ids = robot_actor_ids.reshape(-1)
        if self.cfg.env.add_balls:
            all_subject_env_ids = torch.cat((all_subject_env_ids, object_env_ids))
        if int(static_opponent_env_ids.numel()) > 0:
            all_subject_env_ids = torch.cat((all_subject_env_ids, static_opponent_env_ids))
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(all_subject_env_ids_int32),
            len(all_subject_env_ids_int32),
        )

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []

    def _push_robots(self, env_ids, cfg):
        if cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]
            max_vel = cfg.domain_rand.max_push_vel_xy
            self.root_states[self.robot_actor_idxs_all[env_ids].reshape(-1), 7:9] = torch_rand_float(
                -max_vel,
                max_vel,
                (len(env_ids) * self.num_robots, 2),
                device=self.device,
            )
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            env_int = int(env_id.item()) if torch.is_tensor(env_id) else int(env_id)
            for actor_handle in self.robot_actor_handles_all[env_int]:
                rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_int], actor_handle)
                for i in range(self.num_robot_dof):
                    rigid_shape_props[i].friction = self.friction_coeffs[env_int, 0]
                    rigid_shape_props[i].restitution = self.restitutions[env_int, 0]
                self.gym.set_actor_rigid_shape_properties(self.envs[env_int], actor_handle, rigid_shape_props)
