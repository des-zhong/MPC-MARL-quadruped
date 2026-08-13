import torch
import numpy as np
from dribblebot.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from .rewards import Rewards
from .shooting_geometry import (
    shooting_forward_velocity_score,
    shooting_setup_geometry,
    shooting_setup_progress,
)
from .dribbling_geometry import (
    dribbling_backward_motion_penalty,
    dribbling_setup_score,
)

class SoccerRewards(Rewards):
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        # k_qd = -6e-4
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        # k_q = -0.75
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target - self.env.last_joint_pos_target)
        diff = diff * (self.env.last_actions[:,:12] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target - 2 * self.env.last_joint_pos_target + self.env.last_last_joint_pos_target)
        diff = diff * (self.env.last_actions[:,:12] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:,:12] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _command_scale(self, name, default):
        values = getattr(self.env.cfg.rewards, name, default)
        if len(values) != len(default):
            raise ValueError(f"cfg.rewards.{name} must have {len(default)} values, got {values}")
        return torch.tensor(values, dtype=torch.float, device=self.env.device).abs().clamp(min=1e-6)

    def _walking_command_scale(self):
        return self._command_scale("walking_command_scale", [1.5, 1.5, 1.0])

    def _dribbling_command_scale(self):
        return self._command_scale("dribbling_command_scale", [1.5, 1.5, 1.0])

    def _shooting_command_scale(self):
        return self._command_scale("shooting_command_scale", [1.5, 1.5])

    # encourage robot velocity align vector from robot body to ball
    # r_cv
    def _reward_dribbling_robot_ball_vel(self):
        FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
        FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        FR_HIP_velocities = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,7:10].view(self.env.num_envs,3))
        
        delta_dribbling_robot_ball_vel = 1.0
        robot_ball_vec = self.env.object_local_pos[:,0:2] - FR_HIP_positions[:,0:2]
        d_robot_ball=robot_ball_vec / torch.norm(robot_ball_vec, dim=-1).clamp_min(1e-6).unsqueeze(dim=-1)
        ball_robot_velocity_projection = torch.norm(self.env.commands[:,:2], dim=-1) - torch.sum(d_robot_ball * FR_HIP_velocities[:,0:2], dim=-1) # set approaching speed to velocity command
        speed_scale = torch.norm(self._dribbling_command_scale()[:2]).clamp(min=1e-6)
        ball_robot_velocity_projection = ball_robot_velocity_projection / speed_scale
        velocity_concatenation = torch.cat((torch.zeros(self.env.num_envs,1, device=self.env.device), ball_robot_velocity_projection.unsqueeze(dim=-1)), dim=-1)
        rew_dribbling_robot_ball_vel=torch.exp(-delta_dribbling_robot_ball_vel* torch.pow(torch.max(velocity_concatenation,dim=-1).values, 2) )
        return rew_dribbling_robot_ball_vel

    # encourage robot near ball
    # r_cp
    def _reward_dribbling_robot_ball_pos(self):
        # The old target was the front-right hip, which lies inside the
        # footprint and explicitly taught the policy to keep the ball under the
        # chassis. Target a centered, controllable pose ahead of the base.
        return dribbling_setup_score(
            self.env.object_local_pos[:, :2],
            getattr(self.env.cfg.rewards, "dribbling_target_forward", 0.35),
            getattr(self.env.cfg.rewards, "dribbling_target_lateral", 0.0),
            getattr(self.env.cfg.rewards, "dribbling_position_gain", 10.0),
        )

    def _reward_dribbling_backward_motion(self):
        """Penalize base motion opposite the direction the body faces."""

        _, _, yaw = get_euler_xyz(self.env.base_quat)
        body_forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        return dribbling_backward_motion_penalty(
            self.env.base_lin_vel[:, :2],
            body_forward,
            getattr(self.env.cfg.rewards, "dribbling_forward_speed_scale", 1.0),
        )

    # encourage ball vel align with unit vector between ball target and ball current position
    # r^bv
    def _reward_dribbling_ball_vel(self):
        # target velocity is command input
        command_scale = self._dribbling_command_scale()[:2]
        lin_vel_error = torch.sum(torch.square((self.env.commands[:, :2] - self.env.object_lin_vel[:, :2]) / command_scale), dim=1)
        # rew_dribbling_ball_vel = torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma*2))
        return torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma*2))
        
    def _reward_dribbling_robot_ball_yaw(self):
        robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - self.env.base_pos[:,0:2]
        d_robot_ball=robot_ball_vec / torch.norm(robot_ball_vec, dim=-1).clamp_min(1e-6).unsqueeze(dim=-1)

        unit_command_vel = self.env.commands[:,:2] / torch.norm(self.env.commands[:,:2], dim=-1).clamp_min(1e-6).unsqueeze(dim=-1)
        robot_ball_cmd_yaw_error = torch.norm(unit_command_vel, dim=-1) - torch.sum(d_robot_ball * unit_command_vel, dim=-1)

        # robot ball vector align with body yaw angle
        roll, pitch, yaw = get_euler_xyz(self.env.base_quat)
        body_yaw_vec = torch.zeros(self.env.num_envs, 2, device=self.env.device)
        body_yaw_vec[:,0] = torch.cos(yaw)
        body_yaw_vec[:,1] = torch.sin(yaw)
        robot_ball_body_yaw_error = torch.norm(body_yaw_vec, dim=-1) - torch.sum(d_robot_ball * body_yaw_vec, dim=-1)
        delta_dribbling_robot_ball_cmd_yaw = 2.0
        rew_dribbling_robot_ball_yaw = torch.exp(-delta_dribbling_robot_ball_cmd_yaw * (robot_ball_cmd_yaw_error+robot_ball_body_yaw_error))
        return rew_dribbling_robot_ball_yaw
    
    def _reward_dribbling_ball_vel_norm(self):
        # target velocity is command input
        speed_scale = torch.norm(self._dribbling_command_scale()[:2]).clamp(min=1e-6)
        vel_norm_diff = torch.pow(
            (torch.norm(self.env.commands[:, :2], dim=-1) - torch.norm(self.env.object_lin_vel[:, :2], dim=-1)) / speed_scale,
            2,
        )
        delta_vel_norm = 2.0
        rew_vel_norm_tracking = torch.exp(-delta_vel_norm * vel_norm_diff)
        return rew_vel_norm_tracking

    # def _reward_dribbling_ball_vel_angle(self):
    #     angle_diff = torch.atan2(self.env.commands[:,1], self.env.commands[:,0]) - torch.atan2(self.env.object_lin_vel[:,1], self.env.object_lin_vel[:,0])
    #     angle_diff_in_pi = torch.pow(wrap_to_pi(angle_diff), 2)
    #     rew_vel_angle_tracking = torch.exp(-5.0*angle_diff_in_pi/(torch.pi**2))
    #     # print("angle_diff", angle_diff, " angle_diff_in_pi: ", angle_diff_in_pi, " rew_vel_angle_tracking", rew_vel_angle_tracking, " commands", self.env.commands[:, :2], " object_lin_vel", self.env.object_lin_vel[:, :2])
    #     return rew_vel_angle_tracking

    def _reward_dribbling_ball_vel_angle(self):
        angle_diff = torch.atan2(self.env.commands[:,1], self.env.commands[:,0]) - torch.atan2(self.env.object_lin_vel[:,1], self.env.object_lin_vel[:,0])
        angle_diff_in_pi = torch.pow(wrap_to_pi(angle_diff), 2)
        rew_vel_angle_tracking = 1.0 - angle_diff_in_pi/(torch.pi**2)
        return rew_vel_angle_tracking

    def _reward_walking_vel(self):
        command_scale = self._walking_command_scale()
        lin_vel_error = torch.sum(
            torch.square((self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]) / command_scale[:2]),
            dim=1,
        )
        yaw_vel_error = torch.square(
            (self.env.commands[:, 2] - self.env.base_ang_vel[:, 2]) / command_scale[2]
        )
        sigma = max(float(self.env.cfg.rewards.tracking_sigma) * 2.0, 1e-6)
        lin_reward = torch.exp(-lin_vel_error / sigma)
        yaw_reward = torch.exp(-yaw_vel_error / sigma)

        # A yaw-rate miss must not erase the translational learning signal (or
        # vice versa). The old joint exponential underflowed once either error
        # became large, which made a degraded walking policy unable to recover.
        lin_weight = float(getattr(self.env.cfg.rewards, "walking_linear_reward_weight", 2.0 / 3.0))
        lin_weight = min(max(lin_weight, 0.0), 1.0)
        return lin_weight * lin_reward + (1.0 - lin_weight) * yaw_reward

    def _command_xy(self):
        cmd = self.env.commands[:, :2]
        cmd_norm = torch.norm(cmd, dim=-1).clamp(min=1e-6)
        cmd_dir = cmd / cmd_norm.unsqueeze(-1)
        return cmd, cmd_norm, cmd_dir

    def _ball_xy_vel(self):
        return self.env.object_lin_vel[:, :2]

    def _active_command_gate(self):
        _, target_speed, _ = self._command_xy()
        min_speed = getattr(self.env.cfg.rewards, "shooting_min_command_speed", 0.2)
        return (target_speed > min_speed).float()

    def _pre_kick_gate(self):
        launched = getattr(
            self.env,
            "shooting_launched_buf",
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        )
        return self._active_command_gate() * (~launched).float()

    def _post_kick_gate(self):
        launched = getattr(
            self.env,
            "shooting_launched_buf",
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        )
        return self._active_command_gate() * launched.float()

    def _separation_gate(self):
        robot_to_ball = self.env.object_pos_world_frame[:, :2] - self.env.base_pos[:, :2]
        ball_distance = torch.norm(robot_to_ball, dim=-1)
        min_separation = getattr(self.env.cfg.rewards, "shooting_reward_min_separation", 0.55)
        return (ball_distance > min_separation).float()

    def _body_forward_xy(self):
        _, _, yaw = get_euler_xyz(self.env.base_quat)
        body_forward = torch.zeros(self.env.num_envs, 2, device=self.env.device)
        body_forward[:, 0] = torch.cos(yaw)
        body_forward[:, 1] = torch.sin(yaw)
        return body_forward

    def _shooting_setup_geometry(self):
        setup_distance = getattr(self.env.cfg.rewards, "shooting_setup_distance", 0.45)
        return shooting_setup_geometry(
            self.env.base_pos[:, :2],
            self.env.object_pos_world_frame[:, :2],
            self.env.commands[:, :2],
            setup_distance,
        )

    def _shooting_setup_score(self):
        _, _, setup_error = self._shooting_setup_geometry()
        position_gain = getattr(self.env.cfg.rewards, "shooting_setup_position_gain", 6.0)
        return torch.exp(-position_gain * torch.square(setup_error))

    def _fr_hip_state_world(self):
        fr_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(
            self.env.envs[0],
            self.env.robot_actor_handles[0],
            "FR_thigh_shoulder",
        )
        rb_states = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)
        return rb_states[:, fr_shoulder_idx, 0:3], rb_states[:, fr_shoulder_idx, 7:10]

    def _reward_shooting_ball_vel(self):
        target_vel, _, _ = self._command_xy()
        ball_vel = self._ball_xy_vel()
        vel_error = torch.sum(torch.square((target_vel - ball_vel) / self._shooting_command_scale()), dim=1)
        return torch.exp(-vel_error / (self.env.cfg.rewards.tracking_sigma * 2)) \
            * self._post_kick_gate() * self._separation_gate()

    def _reward_shooting_ball_forward(self):
        """Dense credit for a weak but correctly directed pre-launch strike."""

        min_speed = getattr(self.env.cfg.rewards, "shooting_min_command_speed", 0.2)
        return shooting_forward_velocity_score(
            self._ball_xy_vel(),
            self.env.commands[:, :2],
            min_command_speed=min_speed,
        )

    def _reward_shooting_ball_vel_norm(self):
        _, target_speed, _ = self._command_xy()
        ball_speed = torch.norm(self._ball_xy_vel(), dim=-1)
        speed_scale = torch.norm(self._shooting_command_scale()).clamp(min=1e-6)
        speed_error = torch.square((target_speed - ball_speed) / speed_scale)
        return torch.exp(-2.0 * speed_error) * self._post_kick_gate() * self._separation_gate()

    def _reward_shooting_ball_vel_angle(self):
        _, _, cmd_dir = self._command_xy()
        ball_vel = self._ball_xy_vel()
        ball_speed = torch.norm(ball_vel, dim=-1).clamp(min=1e-6)
        ball_dir = ball_vel / ball_speed.unsqueeze(-1)
        alignment = torch.sum(cmd_dir * ball_dir, dim=-1).clamp(min=-1.0, max=1.0)
        moving_gate = (ball_speed > 0.10).float()
        return 0.5 * (alignment + 1.0) * moving_gate * self._post_kick_gate() * self._separation_gate()

    def _reward_shooting_ball_out(self):
        _, _, cmd_dir = self._command_xy()
        robot_to_ball = self.env.object_pos_world_frame[:, :2] - self.env.base_pos[:, :2]
        ball_distance = torch.norm(robot_to_ball, dim=-1).clamp(min=1e-6)
        ball_dir_from_robot = robot_to_ball / ball_distance.unsqueeze(-1)
        position_alignment = torch.sum(ball_dir_from_robot * cmd_dir, dim=-1).clamp(min=0.0, max=1.0)
        speed_along_cmd = torch.sum(self._ball_xy_vel() * cmd_dir, dim=-1).clamp(min=0.0)
        speed_scale = torch.norm(self._shooting_command_scale()).clamp(min=1e-6)
        setup_distance = getattr(self.env.cfg.rewards, "shooting_setup_distance", 0.45)
        separation = (ball_distance - setup_distance).clamp(min=0.0)
        return torch.tanh(2.0 * separation) * position_alignment * torch.tanh(speed_along_cmd / speed_scale) * self._post_kick_gate()

    def _reward_shooting_robot_ball_pos(self):
        fr_hip_pos_world, _ = self._fr_hip_state_world()
        fr_hip_pos_body = quat_rotate_inverse(self.env.base_quat, fr_hip_pos_world - self.env.base_pos)
        pos_error = torch.norm(self.env.object_local_pos - fr_hip_pos_body, dim=-1)
        return torch.exp(-4.0 * torch.square(pos_error)) * self._pre_kick_gate()

    def _reward_shooting_robot_ball_behind(self):
        # This is a position reward, not an orientation reward: the robot must
        # translate to ball - command_direction * setup_distance.
        return self._shooting_setup_score() * self._pre_kick_gate()

    def _reward_shooting_robot_forward_cmd(self):
        _, _, cmd_dir = self._command_xy()
        body_forward = self._body_forward_xy()
        heading_alignment = torch.sum(body_forward * cmd_dir, dim=-1).clamp(min=-1.0, max=1.0)
        # Gate heading by setup quality. Rotating at an arbitrary position can
        # no longer collect the full reward.
        return 0.5 * (heading_alignment + 1.0) \
            * self._shooting_setup_score() * self._pre_kick_gate()

    def _reward_shooting_ball_in_front(self):
        body_forward = self._body_forward_xy()
        robot_to_ball = self.env.object_pos_world_frame[:, :2] - self.env.base_pos[:, :2]
        distance = torch.norm(robot_to_ball, dim=-1).clamp(min=1e-6)
        robot_to_ball_dir = robot_to_ball / distance.unsqueeze(-1)
        front_alignment = torch.sum(body_forward * robot_to_ball_dir, dim=-1).clamp(min=0.0, max=1.0)
        return front_alignment * self._shooting_setup_score() * self._pre_kick_gate()

    def _reward_shooting_robot_approach_ball(self):
        setup_distance = getattr(self.env.cfg.rewards, "shooting_setup_distance", 0.45)
        speed_scale = getattr(self.env.cfg.rewards, "shooting_setup_progress_speed", 1.0)
        progress = shooting_setup_progress(
            self.env.prev_base_pos[:, :2],
            self.env.base_pos[:, :2],
            self.env.object_pos_world_frame[:, :2],
            self.env.commands[:, :2],
            setup_distance,
            self.env.dt,
            speed_scale,
        )
        # Moving away from the setup pose is negative. The previous clamped
        # approach reward made retreating and orbiting free.
        return progress * self._pre_kick_gate()

    def _reward_shooting_excess_yaw(self):
        free_yaw_rate = getattr(self.env.cfg.rewards, "shooting_free_yaw_rate", 0.75)
        yaw_rate_scale = getattr(self.env.cfg.rewards, "shooting_yaw_rate_scale", 2.0)
        excess_yaw_rate = (torch.abs(self.env.base_ang_vel[:, 2]) - free_yaw_rate).clamp(min=0.0)
        return torch.square(excess_yaw_rate / max(float(yaw_rate_scale), 1e-6)) \
            * self._pre_kick_gate()

    def _reward_shooting_success(self):
        return getattr(
            self.env,
            "shooting_success_buf",
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        ).float()

    def _reward_shooting_launch(self):
        return getattr(
            self.env,
            "shooting_launch_buf",
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        ).float()

    def _reward_shooting_failure(self):
        return getattr(
            self.env,
            "shooting_failure_buf",
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        ).float()
