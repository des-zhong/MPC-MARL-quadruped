"""Reward and skill semantics reconstructed from compact imagined states."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch


DEFAULT_REWARD_SCALES = {
    "goal": 500.0,
    "accidental_termination": -200.0,
    "ball_goal_progress": 2.0,
    "robot_collision": -20.0,
    "pass": 2.0,
    "invalid_skill": -3.0,
    "approach_ball": 1.0,
    "walk_command_alignment": 0.5,
    "face_ball_while_approaching": 0.5,
    "face_goal_while_moving": 0.75,
    "dribble_ball_control": 2.0,
    "shoot_launch": 10.0,
}


class ImaginedSkillResolver:
    """Apply the wrapper's geometric and collision fallbacks in imagination."""

    def __init__(self, schema, action_adapter, config, controlled_robot_count: int):
        self.schema = schema
        self.action_adapter = action_adapter
        self.config = config
        self.controlled_robot_count = int(controlled_robot_count)

    def affordances(self, states: torch.Tensor, actions: torch.Tensor):
        skills, _ = self.action_adapter.unpack(actions)
        field = states[..., self.schema.slice("field.geometry")]
        scale = field[..., :2].abs().clamp(min=1e-6)
        ball = states[..., self.schema.slice("ball.position")][..., :2]
        invalid = torch.zeros_like(skills, dtype=torch.bool)
        can_dribble = torch.zeros_like(skills, dtype=torch.bool)
        can_shoot = torch.zeros_like(skills, dtype=torch.bool)
        local_xy = torch.zeros(*skills.shape, 2, dtype=states.dtype, device=states.device)
        for robot in range(self.controlled_robot_count):
            position = states[..., self.schema.slice(f"robot_{robot}.position")][..., :2]
            yaw = states[..., self.schema.slice(f"robot_{robot}.yaw_sin_cos")]
            delta = (ball - position) * scale
            sin_yaw, cos_yaw = yaw[..., 0], yaw[..., 1]
            local_x = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
            local_y = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]
            local_xy[..., robot, 0] = local_x
            local_xy[..., robot, 1] = local_y
            distance = torch.linalg.vector_norm(delta, dim=-1)
            can_dribble[..., robot] = distance <= self.config.dribble_affordance_distance_m
            can_shoot[..., robot] = (
                (distance <= self.config.shoot_affordance_distance_m)
                & (local_x >= self.config.shoot_min_forward_m)
                & (local_y.abs() <= self.config.shoot_lateral_reach_m)
            )
            invalid[..., robot] = (
                ((skills[..., robot] == 1) & ~can_dribble[..., robot])
                | ((skills[..., robot] == 2) & ~can_shoot[..., robot])
            )
        return invalid, can_dribble, can_shoot, local_xy

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self.config.apply_skill_fallback_in_rollout:
            return actions
        skills, parameters = self.action_adapter.unpack(actions)
        invalid, can_dribble, can_shoot, local_xy = self.affordances(states, actions)
        resolved_skills = skills.clone()
        resolved_parameters = parameters.clone()
        for robot in range(self.controlled_robot_count):
            requested = skills[..., robot]
            shoot_invalid = (requested == 2) & ~can_shoot[..., robot]
            to_dribble = shoot_invalid & can_dribble[..., robot]
            to_walk = ((requested == 1) & ~can_dribble[..., robot]) | (
                shoot_invalid & ~can_dribble[..., robot]
            )
            resolved_skills[..., robot] = torch.where(
                to_dribble, torch.ones_like(requested), resolved_skills[..., robot]
            )
            resolved_skills[..., robot] = torch.where(
                to_walk, torch.zeros_like(requested), resolved_skills[..., robot]
            )
            direction = local_xy[..., robot, :] / torch.linalg.vector_norm(
                local_xy[..., robot, :], dim=-1, keepdim=True
            ).clamp(min=1e-6)
            walk = direction * self.config.fallback_walk_speed_mps
            walk = torch.cat((walk, torch.zeros_like(walk[..., :1])), dim=-1)
            resolved_parameters[..., robot, :] = torch.where(
                to_walk[..., None], walk, resolved_parameters[..., robot, :]
            )
        if self.config.apply_collision_avoidance_in_rollout:
            resolved_skills, resolved_parameters = self._avoid_collisions(
                states, resolved_skills, resolved_parameters
            )
        resolved = self.action_adapter.pack(resolved_skills, resolved_parameters)
        return self.action_adapter.clip(resolved)

    def _avoid_collisions(
        self,
        states: torch.Tensor,
        skills: torch.Tensor,
        parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match ``HighLevelSkillWrapper._apply_collision_avoidance``."""

        robot_count = self.action_adapter.num_robots
        if robot_count < 2:
            return skills, parameters
        field_scale = states[..., self.schema.slice("field.geometry")][..., :2]
        field_scale = field_scale.abs().clamp(min=1e-6)
        positions = torch.stack(
            [
                states[..., self.schema.slice(f"robot_{robot}.position")][..., :2]
                * field_scale
                for robot in range(robot_count)
            ],
            dim=-2,
        )
        velocities = torch.stack(
            [
                states[
                    ..., self.schema.slice(f"robot_{robot}.linear_velocity")
                ][..., :2]
                for robot in range(robot_count)
            ],
            dim=-2,
        )
        relative_position = positions[..., :, None, :] - positions[..., None, :, :]
        relative_velocity = velocities[..., :, None, :] - velocities[..., None, :, :]
        speed_sq = relative_velocity.square().sum(-1)
        closest_time = -(
            relative_position * relative_velocity
        ).sum(-1) / speed_sq.clamp(min=1e-6)
        closest_time = closest_time.clamp(
            min=0.0, max=self.config.collision_avoidance_lookahead_s
        )
        closest_delta = (
            relative_position + closest_time[..., None] * relative_velocity
        )
        closest_distance = torch.linalg.vector_norm(closest_delta, dim=-1)
        diagonal = torch.eye(
            robot_count, dtype=torch.bool, device=states.device
        )
        closest_distance = closest_distance.masked_fill(diagonal, float("inf"))
        nearest_distance, nearest_robot = closest_distance.min(dim=-1)
        avoidance = nearest_distance < self.config.collision_avoidance_distance_m
        gather_index = nearest_robot[..., None, None].expand(
            *nearest_robot.shape, 1, 2
        )
        current_delta = torch.gather(
            relative_position, -2, gather_index
        ).squeeze(-2)
        predicted_delta = torch.gather(
            closest_delta, -2, gather_index
        ).squeeze(-2)
        current_norm = torch.linalg.vector_norm(
            current_delta, dim=-1, keepdim=True
        )
        escape_delta = torch.where(
            current_norm > 1e-5, current_delta, predicted_delta
        )
        escape_direction = escape_delta / torch.linalg.vector_norm(
            escape_delta, dim=-1, keepdim=True
        ).clamp(min=1e-6)
        danger = (
            self.config.collision_avoidance_distance_m - nearest_distance
        ).clamp(min=0.0) / self.config.collision_avoidance_distance_m
        escape_speed = self.config.collision_avoidance_speed_mps * (
            0.5 + 0.5 * danger
        )
        escape_world = escape_direction * escape_speed[..., None]
        yaw = torch.stack(
            [
                states[..., self.schema.slice(f"robot_{robot}.yaw_sin_cos")]
                for robot in range(robot_count)
            ],
            dim=-2,
        )
        sin_yaw, cos_yaw = yaw[..., 0], yaw[..., 1]
        escape_body = torch.stack(
            (
                cos_yaw * escape_world[..., 0] + sin_yaw * escape_world[..., 1],
                -sin_yaw * escape_world[..., 0] + cos_yaw * escape_world[..., 1],
                torch.zeros_like(escape_world[..., 0]),
            ),
            dim=-1,
        )
        result_skills = torch.where(avoidance, torch.zeros_like(skills), skills)
        result_parameters = torch.where(
            avoidance[..., None], escape_body, parameters
        )
        return result_skills, result_parameters


class AnalyticalRewardReconstructor:
    """Approximate the configured environment reward from state transitions."""

    def __init__(
        self, schema, action_adapter, event_names: Sequence[str], config,
        controlled_robot_count: int,
    ):
        self.schema = schema
        self.action_adapter = action_adapter
        self.event_names = tuple(event_names)
        self.config = config
        self.controlled_robot_count = int(controlled_robot_count)

    def _event(self, events: torch.Tensor, name: str) -> torch.Tensor:
        if name not in self.event_names:
            return torch.zeros(events.shape[:-1], dtype=events.dtype, device=events.device)
        return events[..., self.event_names.index(name)]

    def __call__(
        self,
        predicted_states: torch.Tensor,
        executed_actions: torch.Tensor,
        event_probabilities: torch.Tensor,
        invalid_requested: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        before, after = predicted_states[..., :-1, :], predicted_states[..., 1:, :]
        field = before[..., self.schema.slice("field.geometry")]
        scale = field[..., :2].abs().clamp(min=1e-6)
        goal_x = field[..., 2]
        ball0 = before[..., self.schema.slice("ball.position")][..., :2] * scale
        ball1 = after[..., self.schema.slice("ball.position")][..., :2] * scale
        goal = torch.stack((goal_x, torch.zeros_like(goal_x)), dim=-1)
        progress = (
            torch.linalg.vector_norm(goal - ball0, dim=-1)
            - torch.linalg.vector_norm(goal - ball1, dim=-1)
        ).clamp(-self.config.analytical_macro_dt, self.config.analytical_macro_dt)
        components: Dict[str, torch.Tensor] = {
            "goal": DEFAULT_REWARD_SCALES["goal"] * self.config.analytical_event_dt * self._event(event_probabilities, "goal"),
            "accidental_termination": DEFAULT_REWARD_SCALES["accidental_termination"] * self.config.analytical_event_dt * torch.maximum(
                torch.maximum(
                    self._event(event_probabilities, "own_goal"),
                    self._event(event_probabilities, "out_of_bounds"),
                ),
                torch.stack(
                    [after[..., self.schema.slice(f"robot_{r}.fallen")].squeeze(-1) for r in range(self.controlled_robot_count)],
                    dim=-1,
                ).amax(-1),
            ),
            "ball_goal_progress": DEFAULT_REWARD_SCALES["ball_goal_progress"] * progress,
            "pass": DEFAULT_REWARD_SCALES["pass"] * self.config.analytical_event_dt * self._event(event_probabilities, "pass"),
            "shoot_launch": DEFAULT_REWARD_SCALES["shoot_launch"] * self.config.analytical_event_dt * self._event(event_probabilities, "successful_shot"),
            "invalid_skill": DEFAULT_REWARD_SCALES["invalid_skill"] * self.config.analytical_macro_dt * invalid_requested.to(before.dtype).sum(-1),
        }
        positions = []
        for robot in range(self.action_adapter.num_robots):
            positions.append(after[..., self.schema.slice(f"robot_{robot}.position")][..., :2] * scale)
        collision = torch.zeros_like(progress)
        for left in range(self.controlled_robot_count):
            for right in range(left + 1, self.action_adapter.num_robots):
                overlap = ((1.0 - torch.linalg.vector_norm(positions[left] - positions[right], dim=-1)) / 1.0).clamp(0.0, 1.0).square()
                collision = torch.maximum(collision, overlap)
        components["robot_collision"] = DEFAULT_REWARD_SCALES["robot_collision"] * self.config.analytical_macro_dt * collision

        skills, commands = self.action_adapter.unpack(executed_actions)
        previous_distances = torch.stack(
            [torch.linalg.vector_norm(ball0 - before[..., self.schema.slice(f"robot_{r}.position")][..., :2] * scale, dim=-1) for r in range(self.controlled_robot_count)],
            dim=-1,
        )
        current_distances = torch.stack(
            [torch.linalg.vector_norm(ball1 - positions[r], dim=-1) for r in range(self.controlled_robot_count)],
            dim=-1,
        )
        attacker = previous_distances.argmin(dim=-1, keepdim=True)
        approach = torch.gather((previous_distances - current_distances) / max(self.config.analytical_macro_dt, 1e-6), -1, attacker).squeeze(-1).clamp(-1.0, 1.0)
        team_count = previous_distances.shape[-1]
        team_skills = skills[..., :team_count]
        team_invalid = invalid_requested[..., :team_count]
        attacker_skill = torch.gather(team_skills, -1, attacker).squeeze(-1)
        attacker_invalid = torch.gather(team_invalid, -1, attacker).squeeze(-1)
        attacker_far = torch.gather(previous_distances > self.config.dribble_affordance_distance_m, -1, attacker).squeeze(-1)
        valid_walk = (attacker_skill == 0) & ~attacker_invalid
        components["approach_ball"] = DEFAULT_REWARD_SCALES["approach_ball"] * self.config.analytical_macro_dt * approach * valid_walk * attacker_far

        # Command alignment and body orientation use the same attacker chosen
        # from the pre-transition distances as the simulator reward.
        robot_yaws = torch.stack(
            [before[..., self.schema.slice(f"robot_{r}.yaw_sin_cos")] for r in range(team_count)],
            dim=-2,
        )
        sin_yaw, cos_yaw = robot_yaws[..., 0], robot_yaws[..., 1]
        team_commands = commands[..., :team_count, :2]
        command_world = torch.stack(
            (
                cos_yaw * team_commands[..., 0] - sin_yaw * team_commands[..., 1],
                sin_yaw * team_commands[..., 0] + cos_yaw * team_commands[..., 1],
            ),
            dim=-1,
        )
        command_speed = torch.linalg.vector_norm(command_world, dim=-1)
        ball_direction = torch.stack(
            [(ball0 - before[..., self.schema.slice(f"robot_{r}.position")][..., :2] * scale) for r in range(team_count)],
            dim=-2,
        )
        ball_direction = ball_direction / torch.linalg.vector_norm(ball_direction, dim=-1, keepdim=True).clamp(min=1e-6)
        alignment = (command_world * ball_direction).sum(-1) / command_speed.clamp(min=1e-6)
        speed_fraction = (command_speed / 0.9).clamp(0.0, 1.0)
        per_robot_alignment = alignment.clamp(-1.0, 1.0) * speed_fraction
        selected_alignment = torch.gather(per_robot_alignment, -1, attacker).squeeze(-1)
        components["walk_command_alignment"] = DEFAULT_REWARD_SCALES["walk_command_alignment"] * self.config.analytical_macro_dt * selected_alignment * valid_walk * attacker_far
        forward = torch.stack((cos_yaw, sin_yaw), dim=-1)
        face_ball = (forward * ball_direction).sum(-1).clamp(-1.0, 1.0) * speed_fraction
        selected_face_ball = torch.gather(face_ball, -1, attacker).squeeze(-1)
        components["face_ball_while_approaching"] = DEFAULT_REWARD_SCALES["face_ball_while_approaching"] * self.config.analytical_macro_dt * selected_face_ball * valid_walk * attacker_far

        robot_velocities = torch.stack(
            [after[..., self.schema.slice(f"robot_{r}.linear_velocity")][..., :2] for r in range(team_count)],
            dim=-2,
        )
        goal_direction = goal[..., None, :] - torch.stack(positions[:team_count], dim=-2)
        goal_direction = goal_direction / torch.linalg.vector_norm(goal_direction, dim=-1, keepdim=True).clamp(min=1e-6)
        goalward = ((robot_velocities * goal_direction).sum(-1) / 0.5).clamp(-1.0, 1.0)
        forward_motion = ((robot_velocities * forward).sum(-1) / 0.5).clamp(-1.0, 1.0)
        face_goal = torch.minimum(goalward, forward_motion)
        selected_face_goal = torch.gather(face_goal, -1, attacker).squeeze(-1)
        attacker_valid = ~attacker_invalid
        components["face_goal_while_moving"] = DEFAULT_REWARD_SCALES["face_goal_while_moving"] * self.config.analytical_macro_dt * selected_face_goal * attacker_valid

        ball_velocity = after[..., self.schema.slice("ball.linear_velocity")][..., :2]
        ball_goal_direction = (goal - ball1) / torch.linalg.vector_norm(goal - ball1, dim=-1, keepdim=True).clamp(min=1e-6)
        goalward_ball = ((ball_velocity * ball_goal_direction).sum(-1) / 1.0).clamp(0.0, 1.0)
        command_direction = team_commands / torch.linalg.vector_norm(team_commands, dim=-1, keepdim=True).clamp(min=1e-6)
        command_tracking = ((ball_velocity[..., None, :] * command_direction).sum(-1) / command_speed.clamp(min=1e-6)).clamp(0.0, 1.0)
        controlled = torch.maximum(previous_distances, current_distances) <= 0.8
        moving = (command_speed >= 0.2) & (torch.linalg.vector_norm(ball_velocity, dim=-1)[..., None] >= 0.1)
        valid_dribble = (team_skills == 1) & ~team_invalid
        dribble = (0.5 * command_tracking + 0.5 * goalward_ball[..., None]) * controlled * moving * valid_dribble
        components["dribble_ball_control"] = DEFAULT_REWARD_SCALES["dribble_ball_control"] * self.config.analytical_macro_dt * dribble.amax(-1)
        return sum(components.values()), components
