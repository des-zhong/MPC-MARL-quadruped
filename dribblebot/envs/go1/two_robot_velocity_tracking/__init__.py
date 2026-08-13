from isaacgym import gymapi, gymutil
import torch

from dribblebot.envs.base.legged_robot_config import Cfg
from dribblebot.envs.base.legged_robot_two import TwoRobotLeggedRobot


class TwoRobotVelocityTrackingEasyEnv(TwoRobotLeggedRobot):
    def __init__(
        self,
        sim_device,
        headless,
        num_envs=None,
        prone=False,
        deploy=False,
        cfg: Cfg = None,
        eval_cfg: Cfg = None,
        initial_dynamics_dict=None,
        physics_engine="SIM_PHYSX",
    ):
        if num_envs is not None:
            cfg.env.num_envs = num_envs

        cfg.env.num_robots = int(getattr(cfg.env, "num_robots", 2))
        if cfg.env.num_robots < 1:
            raise ValueError("cfg.env.num_robots must be at least 1")
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(
            cfg,
            sim_params,
            physics_engine,
            sim_device,
            headless,
            initial_dynamics_dict=initial_dynamics_dict,
        )

    def step(self, actions):
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[
            :,
            self.feet_indices,
            0:3,
        ]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos[:, :self.num_actuated_dof].cpu().numpy(),
            "joint_vel": self.dof_vel[:, :self.num_actuated_dof].cpu().numpy(),
            "joint_pos_target": self.joint_pos_target[:, :self.num_actuated_dof].cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": self.foot_positions.detach().cpu().numpy().copy(),
            "body_pos": self.root_states[self.robot_actor_idxs, 0:3].detach().cpu().numpy(),
            "robot_root_states": self.robot_root_states_all.detach().cpu().numpy(),
            "other_body_pos": self.root_states[self.other_robot_actor_idxs, 0:3].detach().cpu().numpy(),
            "other_contact_states": (
                self.other_contact_forces[:, self.other_feet_indices, 2] > 1.
            ).detach().cpu().numpy().copy(),
            "torques": self.torques[:, :self.num_actuated_dof].detach().cpu().numpy(),
            "all_torques": self.torques.detach().cpu().numpy(),
        })

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs
