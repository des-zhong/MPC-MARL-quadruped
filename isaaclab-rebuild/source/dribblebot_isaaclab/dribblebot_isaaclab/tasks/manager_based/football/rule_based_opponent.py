"""Checkpoint-free opponent for IsaacLab match evaluation and curriculum."""

from __future__ import annotations

import torch

from .command_frames import world_xy_to_body_xy


class RuleBasedOpponent:
    """Use one stable attacker while the other opponent occupies a blocking lane.

    Returned actions already use executable world semantics. The self-play
    wrapper therefore installs this object through the action-provider seam,
    not the canonical learned-policy seam.
    """

    preserve_high_level_actions = True

    def __init__(
        self,
        self_play_env,
        *,
        shoot_distance: float = 0.75,
        dribble_distance: float = 1.0,
        walk_speed: float = 0.9,
        dribble_speed: float = 1.0,
        shoot_speed: float = 1.5,
        block_distance: float = 0.75,
        collision_distance: float = 0.65,
        collision_strength: float = 1.5,
        attacker_switch_margin: float = 0.15,
    ) -> None:
        self.wrapper = self_play_env
        self.raw_env = self_play_env.env.env
        self.team_size = int(self_play_env.team_size)
        self.device = self_play_env.device
        if self.team_size < 1:
            raise ValueError("RuleBasedOpponent requires at least one opponent robot")
        if not 0.0 < float(shoot_distance) <= float(dribble_distance):
            raise ValueError("Require 0 < shoot_distance <= dribble_distance")
        if min(float(walk_speed), float(dribble_speed), float(shoot_speed)) <= 0.0:
            raise ValueError("Rule-based opponent speeds must be positive")
        if float(block_distance) < 0.0 or float(collision_distance) <= 0.0:
            raise ValueError("Invalid rule-based opponent clearance")
        self.shoot_distance = float(shoot_distance)
        self.dribble_distance = float(dribble_distance)
        self.walk_speed = float(walk_speed)
        self.dribble_speed = float(dribble_speed)
        self.shoot_speed = float(shoot_speed)
        self.block_distance = float(block_distance)
        self.collision_distance = float(collision_distance)
        self.collision_strength = max(float(collision_strength), 0.0)
        self.attacker_switch_margin = max(float(attacker_switch_margin), 0.0)
        self.attacker_slot = torch.full(
            (self.wrapper.match_count,), -1, dtype=torch.long, device=self.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.attacker_slot.fill_(-1)
        else:
            self.attacker_slot[env_ids] = -1

    @staticmethod
    def _normalize(vector: torch.Tensor, fallback: tuple[float, float]) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        fallback_tensor = vector.new_tensor(fallback).reshape((1,) * (vector.ndim - 1) + (2,))
        return torch.where(norm > 1.0e-6, vector / norm.clamp_min(1.0e-6), fallback_tensor)

    def _collision_steer(
        self, direction: torch.Tensor, own_position: torch.Tensor, learner_positions: torch.Tensor
    ) -> torch.Tensor:
        if direction.ndim == 2:
            direction = direction.unsqueeze(1)
        if self.collision_strength == 0.0:
            return self._normalize(direction, (-1.0, 0.0))
        delta = own_position[:, :, None, :] - learner_positions[:, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        away = self._normalize(delta, (0.0, 1.0))
        influence_distance = 1.5 * self.collision_distance
        influence = (
            (influence_distance - distance.squeeze(-1)).clamp_min(0.0) / influence_distance
        ).square().unsqueeze(-1)
        repulsion = (away * influence).sum(dim=2)
        return self._normalize(direction + self.collision_strength * repulsion, (-1.0, 0.0))

    def _stable_attacker(self, distances: torch.Tensor) -> torch.Tensor:
        nearest = distances.argmin(dim=1)
        rows = torch.arange(len(distances), device=self.device)
        previous = self.attacker_slot
        valid = (previous >= 0) & (previous < self.team_size)
        safe_previous = previous.clamp(0, self.team_size - 1)
        switch = (~valid) | (
            distances[rows, nearest] + self.attacker_switch_margin
            < distances[rows, safe_previous]
        )
        chosen = torch.where(switch, nearest, safe_previous)
        self.attacker_slot[:] = chosen
        return chosen

    def _encode(self, skills: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        """Encode executable commands as hybrid ``[index, raw parameters]``."""

        term = self.raw_env.action_manager.get_term(f"skill_policy_{self.team_size}")
        scales = term._command_scales[skills]
        normalized = torch.where(
            scales > 1.0e-6,
            commands / scales.clamp_min(1.0e-6),
            torch.zeros_like(commands),
        ).clamp(-0.98, 0.98)
        # The action term applies tanh to continuous parameters. Return its
        # inverse so the requested world command is preserved after decoding.
        return torch.cat((skills.to(commands.dtype).unsqueeze(-1), torch.atanh(normalized)), dim=-1)

    @torch.inference_mode()
    def __call__(self) -> torch.Tensor:
        learner_names = tuple(f"robot_{slot}" for slot in range(self.team_size))
        opponent_names = tuple(
            f"robot_{self.team_size + slot}" for slot in range(self.team_size)
        )
        learner_xy = torch.stack(
            [self.raw_env.scene[name].data.root_pos_w[:, :2] for name in learner_names], dim=1
        )
        opponent_xy = torch.stack(
            [self.raw_env.scene[name].data.root_pos_w[:, :2] for name in opponent_names], dim=1
        )
        opponent_quat = torch.stack(
            [self.raw_env.scene[name].data.root_quat_w for name in opponent_names], dim=1
        )
        ball_xy = self.raw_env.scene["ball"].data.root_pos_w[:, :2]
        opponent_ball_delta = ball_xy[:, None, :] - opponent_xy
        opponent_distance = torch.linalg.vector_norm(opponent_ball_delta, dim=-1)
        attacker = self._stable_attacker(opponent_distance)
        learner_distance = torch.linalg.vector_norm(ball_xy[:, None, :] - learner_xy, dim=-1)
        nearest_learner = learner_distance.argmin(dim=1)
        rows = torch.arange(self.wrapper.match_count, device=self.device)

        goal = self.raw_env.scene.env_origins[:, :2].clone()
        goal[:, 0] -= 4.0
        goal_direction = self._normalize(goal - ball_xy, (-1.0, 0.0))
        skills = torch.zeros(
            self.wrapper.match_count, self.team_size, dtype=torch.long, device=self.device
        )
        commands = torch.zeros(
            self.wrapper.match_count, self.team_size, 3, device=self.device
        )

        for slot in range(self.team_size):
            is_attacker = attacker == slot
            own_xy = opponent_xy[:, slot : slot + 1]
            ball_direction = self._collision_steer(
                self._normalize(ball_xy - own_xy[:, 0], (-1.0, 0.0)), own_xy, learner_xy
            )[:, 0]
            dribble_direction = self._collision_steer(
                goal_direction[:, None, :], own_xy, learner_xy
            )[:, 0]
            can_dribble = opponent_distance[:, slot] <= self.dribble_distance
            can_shoot = opponent_distance[:, slot] <= self.shoot_distance
            use_shoot = is_attacker & can_shoot
            use_dribble = is_attacker & ~use_shoot & can_dribble
            use_walk = is_attacker & ~use_shoot & ~use_dribble
            skills[use_dribble, slot] = 1
            skills[use_shoot, slot] = 2

            walk_world = ball_direction * self.walk_speed
            walk_body = world_xy_to_body_xy(walk_world, opponent_quat[:, slot])
            commands[use_walk, slot, :2] = walk_body[use_walk]
            commands[use_walk, slot, 2] = torch.atan2(
                walk_body[use_walk, 1], walk_body[use_walk, 0]
            ).clamp(-1.0, 1.0)
            commands[use_dribble, slot, :2] = dribble_direction[use_dribble] * self.dribble_speed
            commands[use_shoot, slot, :2] = goal_direction[use_shoot] * self.shoot_speed

            if self.team_size > 1:
                learner_position = learner_xy[rows, nearest_learner]
                lane = self._normalize(ball_xy - learner_position, (1.0, 0.0))
                lane_distance = learner_distance[rows, nearest_learner]
                offset = torch.minimum(
                    torch.full_like(lane_distance, self.block_distance), 0.5 * lane_distance
                )
                target = ball_xy - lane * offset.unsqueeze(-1)
                target_delta = target - own_xy[:, 0]
                target_distance = torch.linalg.vector_norm(target_delta, dim=-1)
                block_world = self._collision_steer(
                    target_delta[:, None, :], own_xy, learner_xy
                )[:, 0] * torch.minimum(
                    torch.full_like(target_distance, self.walk_speed), 1.5 * target_distance
                ).unsqueeze(-1)
                block_body = world_xy_to_body_xy(block_world, opponent_quat[:, slot])
                support = ~is_attacker
                commands[support, slot, :2] = block_body[support]
                ball_body = world_xy_to_body_xy(ball_xy - own_xy[:, 0], opponent_quat[:, slot])
                commands[support, slot, 2] = torch.atan2(
                    ball_body[support, 1], ball_body[support, 0]
                ).clamp(-1.0, 1.0)

            learner_delta = own_xy - learner_xy
            clearances = torch.linalg.vector_norm(learner_delta, dim=-1)
            nearest_clearance, nearest_slot = clearances.min(dim=1)
            escape_world = self._normalize(
                learner_delta[rows, nearest_slot], (0.0, 1.0)
            ) * self.walk_speed
            escape_body = world_xy_to_body_xy(escape_world, opponent_quat[:, slot])
            emergency = nearest_clearance < self.collision_distance
            skills[emergency, slot] = 0
            commands[emergency, slot, :2] = escape_body[emergency]
            commands[emergency, slot, 2] = torch.atan2(
                escape_body[emergency, 1], escape_body[emergency, 0]
            ).clamp(-1.0, 1.0)
        return self._encode(skills, commands)


__all__ = ["RuleBasedOpponent"]
