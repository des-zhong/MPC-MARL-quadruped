import torch


class Rewards:
    def __init__(self, env):
        self.env = env

    def get_reward(self, env_ids = None):
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]()
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
            rew = rew * self.reward_scales[name]
            rew = torch.nan_to_num(rew, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.rew_buf[:] = torch.nan_to_num(self.rew_buf, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination()
            rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
            rew = rew * self.reward_scales["termination"]
            rew = torch.nan_to_num(rew, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew
        self.rew_buf[:] = torch.nan_to_num(self.rew_buf, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

        base_lin_x = torch.nan_to_num(self.base_lin_vel[:, 0], nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        base_ang_z = torch.nan_to_num(self.base_ang_vel[:, 2], nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        cmd_x = torch.nan_to_num(self.commands[:, 0], nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        cmd_yaw = torch.nan_to_num(self.commands[:, 2], nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        self.command_sums["lin_vel_raw"] += base_lin_x
        self.command_sums["ang_vel_raw"] += base_ang_z
        self.command_sums["lin_vel_residual"] += torch.nan_to_num((base_lin_x - cmd_x) ** 2, nan=0.0, posinf=1.0e6)
        self.command_sums["ang_vel_residual"] += torch.nan_to_num((base_ang_z - cmd_yaw) ** 2, nan=0.0, posinf=1.0e6)
        self.command_sums["ep_timesteps"] += 1


