"""Privileged MPC guidance for decentralized high-level self-play training."""

from __future__ import annotations

import math

import gym
import torch

from quadruped.mpc.opponent_forecast import (
    SlowWalkToBallOpponentForecaster,
    ZeroOpponentForecaster,
)


class MPCTeacherGuidanceWrapper(gym.Wrapper):
    """Add a dense MPC action-agreement reward to the learning team.

    The student's observations remain agent-local.  The teacher alone receives
    the world model's global state and plans the learning-team actions while the
    simple opponent's current action is held over the planning horizon.
    """

    def __init__(
        self,
        env,
        planner,
        state_adapter,
        reward_coefficient=1.0,
        opponent_config=None,
    ):
        super().__init__(env)
        self.env = env
        self.planner = planner
        self.state_adapter = state_adapter
        self.reward_coefficient = float(reward_coefficient)
        self.planner_state = None
        if self.reward_coefficient < 0.0:
            raise ValueError("reward_coefficient must be non-negative")
        expected = 2 * int(env.team_size)
        if planner.num_robots != expected:
            raise ValueError(
                f"Teacher world model has {planner.num_robots} robots; expected {expected}"
            )
        opponent = dict(opponent_config or {})
        mode = str(opponent.get("mode", "slow_walk_to_ball"))
        if mode == "stationary":
            self.opponent_forecaster = ZeroOpponentForecaster(
                env.env, env.team_size, planner.action_adapter
            )
        elif mode == "slow_walk_to_ball":
            self.opponent_forecaster = SlowWalkToBallOpponentForecaster(
                env.env,
                env.team_size,
                planner.action_adapter,
                speed_mps=float(opponent.get("walk_speed_mps", 0.35)),
                stop_distance_m=float(opponent.get("stop_distance_m", 0.25)),
                slow_distance_m=float(opponent.get("slow_distance_m", 1.0)),
                yaw_gain=float(opponent.get("yaw_gain", 1.0)),
            )
        else:
            raise ValueError(f"Unknown opponent mode {mode!r}")
        env.set_opponent_action_provider(self.opponent_forecaster.wrapper_actions)

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def actions(self):
        return self.env.actions

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    def randomize_episode_lengths(self):
        return self.env.randomize_episode_lengths()

    def update_opponent_policy(self, actor_critic, iteration=0):
        return self.env.update_opponent_policy(actor_critic, iteration=iteration)

    def load_opponent_policy_state_dict(self, state_dict, actor_critic, iteration=-1):
        return self.env.load_opponent_policy_state_dict(
            state_dict, actor_critic, iteration=iteration
        )

    def opponent_policy_state_dict(self):
        return self.env.opponent_policy_state_dict()

    def reset(self):
        self.planner_state = None
        observations = self.env.reset()
        self.opponent_forecaster.reset()
        return observations

    def _world_opponent_forecast(self):
        """Return the simple action requested from the simulator opponent."""

        fixed, _ = self.opponent_forecaster.fixed_action_sequence(
            self.planner.config.horizon
        )
        return fixed

    def _per_agent_guidance(self, executed, teacher):
        adapter = self.planner.action_adapter
        executed_skill, executed_params = adapter.unpack(executed)
        teacher_skill, teacher_params = adapter.unpack(teacher)
        executed_norm = adapter.normalize_parameters(executed_skill, executed_params)
        teacher_norm = adapter.normalize_parameters(teacher_skill, teacher_params)
        count = self.env.team_size
        same_skill = executed_skill[:, :count] == teacher_skill[:, :count]
        mask = adapter._selected(
            teacher_skill[:, :count], "mask", executed.dtype
        )
        parameter_error = (
            (executed_norm[:, :count] - teacher_norm[:, :count]).square() * mask
        ).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        disagreement = (~same_skill).to(executed.dtype) + same_skill.to(
            executed.dtype
        ) * parameter_error
        # Center at zero for a skill mismatch; exact agreement approaches
        # (1 - exp(-1)) * coefficient.
        reward = self.reward_coefficient * (
            torch.exp(-disagreement) - math.exp(-1.0)
        )
        return reward, disagreement

    @torch.no_grad()
    def step(self, actions):
        encoded = self.state_adapter.extract_state(self.env.env)["tensor"]
        fixed = self._world_opponent_forecast()
        fixed_mask = torch.zeros(
            self.planner.num_robots, dtype=torch.bool, device=encoded.device
        )
        fixed_mask[self.env.team_size :] = True
        plan = self.planner.plan(
            encoded,
            planner_state=self.planner_state,
            fixed_action_sequence=fixed,
            fixed_robot_mask=fixed_mask,
        )
        self.planner_state = plan.planner_state

        observations, rewards, dones, info = self.env.step(actions)
        executed_skills = torch.as_tensor(
            info["high_level_skill_ids"], dtype=torch.long, device=encoded.device
        )
        executed_commands = torch.as_tensor(
            info["high_level_commands"], dtype=encoded.dtype, device=encoded.device
        )
        executed = self.planner.action_adapter.pack(
            executed_skills, executed_commands
        )
        guidance, disagreement = self._per_agent_guidance(
            executed, plan.first_joint_action
        )
        rewards = rewards + guidance.reshape(-1)

        match_done = dones.view(self.env.match_count, self.env.team_size).any(dim=1)
        if bool(match_done.any().item()):
            self.planner_state.reset(match_done.nonzero(as_tuple=False).squeeze(-1))
        info = dict(info)
        info["mpc_teacher_reward"] = guidance.reshape(-1)
        info["mpc_teacher_disagreement"] = disagreement.reshape(-1)
        info["mpc_teacher_objective"] = plan.best_objective.repeat_interleave(
            self.env.team_size
        )
        return observations, rewards, dones, info
