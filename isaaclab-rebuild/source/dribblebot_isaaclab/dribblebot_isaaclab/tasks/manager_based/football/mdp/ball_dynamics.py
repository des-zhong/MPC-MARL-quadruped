"""Legacy match ball drag without adding policy action dimensions."""

import math

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class MatchBallDrag(ActionTerm):
    """Apply Gym's world-XY quadratic drag, held over each 20 ms control step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.coefficient_range[0] <= cfg.coefficient_range[1]:
            raise ValueError("Ball drag coefficients must be non-negative and ordered")
        if cfg.resampling_interval_s <= 0.0:
            raise ValueError("Ball drag resampling interval must be positive")
        self._empty_actions = torch.empty(env.num_envs, 0, device=env.device)
        self._forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
        self._torques = torch.zeros_like(self._forces)
        self.coefficients = torch.empty(env.num_envs, 1, device=env.device)
        self._interval = math.ceil(cfg.resampling_interval_s / env.step_dt)
        self._resample()

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._empty_actions

    @property
    def processed_actions(self):
        return self._empty_actions

    def _resample(self):
        self.coefficients.uniform_(*self.cfg.coefficient_range)

    def process_actions(self, actions):
        del actions
        # Gym resamples globally every 15 s, not at individual episode resets.
        if self._env.common_step_counter > 0 and self._env.common_step_counter % self._interval == 0:
            self._resample()
        velocity = self._asset.data.root_lin_vel_w[:, :2]
        self._forces[:, 0, :2] = -self.coefficients * velocity.square() * velocity.sign()

    def apply_actions(self):
        self._asset.set_external_force_and_torque(self._forces, self._torques, is_global=True)

    def reset(self, env_ids=None):
        self._forces[slice(None) if env_ids is None else env_ids] = 0.0


@configclass
class MatchBallDragCfg(ActionTermCfg):
    class_type: type = MatchBallDrag
    asset_name: str = "ball"
    coefficient_range: tuple[float, float] = (0.1, 0.8)
    resampling_interval_s: float = 15.0
