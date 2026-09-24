"""Deterministic, checkpoint-free opponent for high-level evaluation.

The provider returns executable high-level actions in world semantics.  This
is distinct from a learned opponent policy, whose canonical +x actions are
mirrored by ``SharedPolicySelfPlayWrapper`` before execution.
"""

from __future__ import annotations

import math

import torch


class RuleBasedOpponent:
    """Approach, dribble, and shoot with one robot while the other blocks.

    The closest opponent to the ball is the attacker.  It walks to the ball,
    dribbles toward the opponent goal, and requests a shot once it is close
    enough.  Every other opponent targets the line segment between the closest
    learning-team robot and the ball.  A short-range repulsive term steers both
    robots away from learning-team robots without requiring a learned policy.
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
    ):
        self.wrapper = self_play_env
        self.match_env = self_play_env.env
        self.raw_env = self.match_env.env
        self.team_size = int(self_play_env.team_size)
        if self.team_size < 1:
            raise ValueError("RuleBasedOpponent requires at least one opponent robot")
        values = {
            "shoot_distance": shoot_distance,
            "dribble_distance": dribble_distance,
            "walk_speed": walk_speed,
            "dribble_speed": dribble_speed,
            "shoot_speed": shoot_speed,
            "block_distance": block_distance,
            "collision_distance": collision_distance,
            "collision_strength": collision_strength,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Rule-based opponent parameters must be finite")
        if shoot_distance <= 0.0 or dribble_distance <= 0.0:
            raise ValueError("Rule-based shoot/dribble distances must be positive")
        if shoot_distance > dribble_distance:
            raise ValueError("shoot_distance must not exceed dribble_distance")
        if walk_speed <= 0.0 or dribble_speed <= 0.0 or shoot_speed <= 0.0:
            raise ValueError("Rule-based opponent speeds must be positive")
        if block_distance < 0.0 or collision_distance <= 0.0 or collision_strength < 0.0:
            raise ValueError("Invalid rule-based opponent clearance parameters")
        self.shoot_distance = float(shoot_distance)
        self.dribble_distance = float(dribble_distance)
        self.walk_speed = float(walk_speed)
        self.dribble_speed = float(dribble_speed)
        self.shoot_speed = float(shoot_speed)
        self.block_distance = float(block_distance)
        self.collision_distance = float(collision_distance)
        self.collision_strength = float(collision_strength)

    def reset(self, env_ids=None):
        """Keep the provider stateless; retained for wrapper compatibility."""

        del env_ids

    def _command_scales(self):
        cfg = self.raw_env.cfg.env
        return torch.as_tensor(
            [
                getattr(cfg, "high_level_walk_command_scale", [1.2, 0.6, 0.0]),
                getattr(cfg, "high_level_dribble_command_scale", [1.5, 1.5, 1.0]),
                getattr(cfg, "high_level_shoot_command_scale", [1.5, 1.5, 0.0]),
            ],
            dtype=torch.float,
            device=self.wrapper.device,
        ).abs()

    @staticmethod
    def _normalize(vector, fallback):
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        fallback = torch.as_tensor(fallback, dtype=vector.dtype, device=vector.device)
        fallback = fallback.reshape((1,) * (vector.ndim - 1) + (2,))
        return torch.where(norm > 1.0e-6, vector / norm.clamp(min=1.0e-6), fallback)

    def _collision_steer(self, direction, own_position, learner_positions):
        """Add a smooth repulsion from the learning team to a 2-D direction."""

        if learner_positions.shape[1] == 0 or self.collision_strength == 0.0:
            return self._normalize(direction, [-1.0, 0.0])
        delta = own_position[:, :, None, :] - learner_positions[:, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        away = self._normalize(delta, [0.0, 1.0])
        influence_distance = 1.5 * self.collision_distance
        influence = (
            (influence_distance - distance.squeeze(-1)).clamp(min=0.0)
            / influence_distance
        ).square().unsqueeze(-1)
        repulsion = (away * influence).sum(dim=2)
        return self._normalize(
            direction + self.collision_strength * repulsion, [-1.0, 0.0]
        )

    def _world_to_body(self, roots, world_xy):
        world_3d = torch.cat((world_xy, torch.zeros_like(world_xy[..., :1])), dim=-1)
        quaternion = roots[:, :, 3:7]
        vector_part = quaternion[..., :3]
        scalar_part = quaternion[..., 3:4]
        body_3d = (
            world_3d * (2.0 * scalar_part.square() - 1.0)
            - 2.0 * scalar_part * torch.linalg.cross(vector_part, world_3d)
            + 2.0
            * vector_part
            * torch.sum(vector_part * world_3d, dim=-1, keepdim=True)
        )
        return body_3d[..., :2]

    def _encode(self, skills, commands):
        """Encode decisions in the wrapped hybrid or discrete action layout."""

        if self.wrapper.num_actions == 12:
            skill_one_hot = torch.nn.functional.one_hot(
                skills, num_classes=4
            ).to(commands.dtype)
            planar = commands[..., :2]
            direction_vectors = commands.new_tensor(
                [
                    [1.0, 0.0],
                    [2.0**-0.5, -(2.0**-0.5)],
                    [0.0, -1.0],
                    [-(2.0**-0.5), -(2.0**-0.5)],
                    [-1.0, 0.0],
                    [-(2.0**-0.5), 2.0**-0.5],
                    [0.0, 1.0],
                    [2.0**-0.5, 2.0**-0.5],
                ]
            )
            normalized = self._normalize(planar, [1.0, 0.0])
            direction_ids = torch.argmax(
                (normalized.unsqueeze(-2) * direction_vectors).sum(dim=-1),
                dim=-1,
            )
            direction_one_hot = torch.nn.functional.one_hot(
                direction_ids, num_classes=8
            ).to(commands.dtype)
            return torch.cat((skill_one_hot, direction_one_hot), dim=-1)

        scales = self._command_scales()[skills]
        normalized = torch.where(
            scales > 1.0e-6,
            commands / scales.clamp(min=1.0e-6),
            torch.zeros_like(commands),
        ).clamp(-0.98, 0.98)
        raw_commands = torch.atanh(normalized)
        logits = torch.full(
            (*skills.shape, 3), -10.0, dtype=commands.dtype, device=commands.device
        )
        logits.scatter_(-1, skills.unsqueeze(-1), 10.0)
        return torch.cat((logits, raw_commands), dim=-1)

    @torch.no_grad()
    def wrapper_actions(self):
        """Return executable opponent actions in the configured layout."""

        roots = self.wrapper._roots()
        positions = roots[:, :, :2]
        ball = self.raw_env.object_pos_world_frame[:, :2]
        learners = positions[:, : self.team_size]
        opponents = positions[:, self.team_size :]
        batch = opponents.shape[0]
        device = opponents.device
        dtype = opponents.dtype

        opponent_ball_delta = ball[:, None, :] - opponents
        opponent_distances = torch.linalg.vector_norm(opponent_ball_delta, dim=-1)
        attacker_index = opponent_distances.argmin(dim=1)
        can_dribble = opponent_distances <= self.dribble_distance
        can_shoot = opponent_distances <= self.shoot_distance
        # Match the high-level wrapper's nearest-robot hysteresis so its
        # role-aware geometric fallback does not undo a rule decision during
        # a near-tie attacker swap.
        if hasattr(self.match_env, "_skill_affordances") and hasattr(
            self.match_env, "_attacker_mask"
        ):
            affordances = self.match_env._skill_affordances(roots)
            role_mask = self.match_env._attacker_mask(affordances)[
                :, self.team_size :
            ]
            valid_role = role_mask.sum(dim=1) == 1
            assigned = role_mask.long().argmax(dim=1)
            attacker_index = torch.where(valid_role, assigned, attacker_index)
        learner_ball_delta = ball[:, None, :] - learners
        learner_distances = torch.linalg.vector_norm(learner_ball_delta, dim=-1)
        learner_index = learner_distances.argmin(dim=1)
        rows = torch.arange(batch, device=device)

        goal_x = float(getattr(self.raw_env.cfg.env, "team_goal_x", 0.5 * 8.0))
        goal = self.raw_env.env_origins[:, :2].clone()
        goal[:, 0] -= goal_x

        skills = torch.zeros(batch, self.team_size, dtype=torch.long, device=device)
        world_commands = torch.zeros(batch, self.team_size, 3, dtype=dtype, device=device)

        for opponent_index in range(self.team_size):
            is_attacker = attacker_index == opponent_index
            own_position = opponents[:, opponent_index : opponent_index + 1]
            own_roots = roots[:, self.team_size + opponent_index : self.team_size + opponent_index + 1]

            # Attacker: walk to the ball, then use field-frame ball commands.
            goal_direction = self._normalize(goal - ball, [-1.0, 0.0])
            ball_direction = self._normalize(ball - own_position[:, 0], [-1.0, 0.0])
            ball_direction = self._collision_steer(
                ball_direction, own_position, learners
            )[:, 0]
            dribble_direction = self._collision_steer(
                (goal - ball)[:, None, :], own_position, learners
            )[:, 0]
            shoot_direction = goal_direction

            is_shoot = is_attacker & can_shoot[:, opponent_index]
            is_dribble = is_attacker & ~is_shoot & can_dribble[:, opponent_index]
            is_walk = is_attacker & ~is_shoot & ~is_dribble
            skills[is_dribble, opponent_index] = 1
            skills[is_shoot, opponent_index] = 2

            walk_world = ball_direction * self.walk_speed
            walk_body = self._world_to_body(own_roots, walk_world[:, None, :])[:, 0]
            world_commands[is_walk, opponent_index, :2] = walk_body[is_walk]
            world_commands[is_walk, opponent_index, 2] = torch.atan2(
                walk_body[is_walk, 1], walk_body[is_walk, 0]
            ).clamp(-1.0, 1.0)

            dribble_world = dribble_direction * self.dribble_speed
            shoot_world = shoot_direction * self.shoot_speed
            # Direct action providers return executable actions, so ball-skill
            # commands stay in the fixed world frame and point at the -x goal.
            world_field = torch.where(
                is_shoot[:, None], shoot_world, dribble_world
            )
            world_commands[is_dribble, opponent_index, :2] = world_field[is_dribble]
            world_commands[is_shoot, opponent_index, :2] = world_field[is_shoot]

            if self.team_size > 1:
                # Block the nearest learning robot on the learner-to-ball line.
                learner_position = learners[rows, learner_index]
                line = self._normalize(ball - learner_position, [1.0, 0.0])
                learner_ball_distance = learner_distances[rows, learner_index]
                ball_offset = torch.minimum(
                    torch.full_like(learner_ball_distance, self.block_distance),
                    0.5 * learner_ball_distance,
                )
                target = ball - line * ball_offset.unsqueeze(-1)
                target_delta = target - own_position[:, 0]
                target_distance = torch.linalg.vector_norm(target_delta, dim=-1)
                block_world = self._collision_steer(
                    target_delta[:, None, :], own_position, learners
                )[:, 0] * torch.minimum(
                    torch.full_like(target_distance, self.walk_speed),
                    1.5 * target_distance,
                ).unsqueeze(-1)
                block_body = self._world_to_body(own_roots, block_world[:, None, :])[:, 0]
                block_mask = ~is_attacker
                skills[block_mask, opponent_index] = 0
                world_commands[block_mask, opponent_index, :2] = block_body[block_mask]
                ball_body = self._world_to_body(
                    own_roots, (ball - own_position[:, 0])[:, None, :]
                )[:, 0]
                world_commands[block_mask, opponent_index, 2] = torch.atan2(
                    ball_body[block_mask, 1],
                    ball_body[block_mask, 0],
                ).clamp(-1.0, 1.0)

            # Inside the requested clearance, safety takes precedence over
            # ball possession: walk directly away from the closest learner.
            learner_delta = own_position - learners
            learner_clearance = torch.linalg.vector_norm(learner_delta, dim=-1)
            nearest_clearance, nearest_learner = learner_clearance.min(dim=1)
            nearest_delta = learner_delta[rows, nearest_learner]
            escape_world = self._normalize(nearest_delta, [0.0, 1.0]) * self.walk_speed
            escape_body = self._world_to_body(
                own_roots, escape_world[:, None, :]
            )[:, 0]
            emergency = nearest_clearance < self.collision_distance
            skills[emergency, opponent_index] = 0
            world_commands[emergency, opponent_index, :2] = escape_body[emergency]
            world_commands[emergency, opponent_index, 2] = torch.atan2(
                escape_body[emergency, 1], escape_body[emergency, 0]
            ).clamp(-1.0, 1.0)

        return self._encode(skills, world_commands)

    __call__ = wrapper_actions
