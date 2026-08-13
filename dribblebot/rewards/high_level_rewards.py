import torch


class HighLevelRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def _field_ball_xy(self):
        return self.env.object_pos_world_frame[:, :2] - self.env.env_origins[:, :2]

    def _team_size(self):
        return int(getattr(self.env.cfg.env, "num_team_robots", self.env.num_robots))

    def _robot_actor_indices(self):
        actor_indices = getattr(self.env, "robot_actor_idxs_all", None)
        if actor_indices is not None:
            return actor_indices[:, : self._team_size()]
        # Compatibility with older two-robot test fixtures/checkpoints.
        return torch.stack(
            (self.env.robot_actor_idxs, self.env.other_robot_actor_idxs), dim=1
        )

    def _robot_xy(self):
        return self.env.root_states[
            self._robot_actor_indices().reshape(-1), :2
        ].view(self.env.num_envs, self._team_size(), 2)

    def _robot_states(self):
        return self.env.root_states[
            self._robot_actor_indices().reshape(-1)
        ].view(self.env.num_envs, self._team_size(), 13)

    def _all_robot_states(self):
        actor_indices = getattr(self.env, "robot_actor_idxs_all", None)
        if actor_indices is None:
            actor_indices = torch.stack(
                (self.env.robot_actor_idxs, self.env.other_robot_actor_idxs), dim=1
            )
        count = int(actor_indices.shape[1])
        return self.env.root_states[actor_indices.reshape(-1)].view(
            self.env.num_envs, count, 13
        )

    def _robot_ball_distances(self):
        robot_xy = self._robot_xy()
        ball_xy = self.env.object_pos_world_frame[:, None, :2]
        return torch.norm(robot_xy - ball_xy, dim=-1)

    def _skill_ids(self, attr_name, default=0):
        values = getattr(
            self.env,
            attr_name,
            torch.full(
                (self.env.num_envs, int(getattr(self.env, "num_robots", 2))),
                default,
                dtype=torch.long,
                device=self.env.device,
            ),
        )
        return values[:, : self._team_size()]

    def _commands(self):
        values = getattr(
            self.env,
            "high_level_commands",
            torch.zeros(
                self.env.num_envs,
                self._team_size(),
                3,
                dtype=torch.float,
                device=self.env.device,
            ),
        )
        return values[:, : self._team_size()]

    def _valid_executed_skill(self, skill_id):
        executed = self._skill_ids("high_level_skill_ids")
        requested = self._skill_ids("high_level_requested_skill_ids")
        invalid = self._skill_ids("high_level_invalid_skill_mask")
        return (executed == skill_id) & (requested == skill_id) & (invalid == 0)

    def _goal_xy(self):
        goal_x = float(getattr(self.env.cfg.env, "team_goal_x", 4.0))
        return torch.stack(
            (
                self.env.env_origins[:, 0] + goal_x,
                self.env.env_origins[:, 1],
            ),
            dim=-1,
        )

    def _ball_goal_direction(self):
        delta = self._goal_xy() - self.env.object_pos_world_frame[:, :2]
        return delta / torch.norm(delta, dim=-1, keepdim=True).clamp(min=1e-6)

    def _ball_goal_progress_speed(self):
        previous = getattr(self.env, "prev_object_pos_world_frame", self.env.object_pos_world_frame)
        goal_xy = self._goal_xy()
        previous_distance = torch.norm(goal_xy - previous[:, :2], dim=-1)
        current_distance = torch.norm(goal_xy - self.env.object_pos_world_frame[:, :2], dim=-1)
        return (previous_distance - current_distance) / max(self.env.dt, 1e-6)

    def _obstacle_distances(self):
        if getattr(self.env, "num_static_opponents", 0) <= 0:
            return None

        obstacle_xy = self.env.root_states[self.env.static_opponent_actor_idxs.reshape(-1), :2].view(
            self.env.num_envs,
            self.env.num_static_opponents,
            2,
        )
        ball_xy = self.env.object_pos_world_frame[:, None, :2]
        return torch.norm(ball_xy - obstacle_xy, dim=-1)

    def _reward_high_level_goal(self):
        return getattr(
            self.env,
            "high_level_goal_buf",
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        ).float()

    def _reward_high_level_accidental_termination(self):
        return getattr(
            self.env,
            "high_level_accidental_termination_buf",
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        ).float()

    def _reward_high_level_ball_goal_progress(self):
        return torch.clamp(self._ball_goal_progress_speed(), min=-1.0, max=1.0)

    def _reward_high_level_possession(self):
        min_distance = torch.min(self._robot_ball_distances(), dim=1).values
        return torch.exp(-2.0 * torch.square(min_distance))

    # def _reward_high_level_robot_spacing(self):
    #     robot_xy = self._robot_xy()
    #     distance = torch.norm(robot_xy[:, 0, :] - robot_xy[:, 1, :], dim=-1)
    #     min_spacing = float(getattr(self.env.cfg.rewards, "high_level_min_robot_spacing", 0.65))
    #     target_spacing = float(getattr(self.env.cfg.rewards, "high_level_target_robot_spacing", 1.5))
    #     too_close = torch.clamp(min_spacing - distance, min=0.0)
    #     useful_spacing = torch.exp(-torch.square(distance - target_spacing))
    #     return useful_spacing - 4.0 * torch.square(too_close)

    def _reward_high_level_robot_collision(self):
        """Penalize the worst overlapping robot pair in each match.

        Isaac Gym's net contact-force tensor does not identify the other body
        in a contact and therefore mixes robot contact with required foot-ground
        and robot-ball contact. Horizontal base clearance provides a stable,
        targeted collision-avoidance signal for teammate and opponent pairs.
        Opponent-opponent pairs are excluded because the learning team cannot
        control a collision caused solely by the frozen opponent policy.
        """

        robot_xy = self._all_robot_states()[:, :, :2]
        robot_count = int(robot_xy.shape[1])
        if robot_count < 2:
            return torch.zeros(
                self.env.num_envs, dtype=torch.float, device=self.env.device
            )

        collision_distance = float(
            getattr(
                self.env.cfg.rewards,
                "high_level_robot_collision_distance",
                0.75,
            )
        )
        if collision_distance <= 0.0:
            raise ValueError(
                "high_level_robot_collision_distance must be positive, "
                f"got {collision_distance}"
            )
        pair_indices = torch.triu_indices(
            robot_count, robot_count, offset=1, device=robot_xy.device
        )
        pair_indices = pair_indices[
            :, pair_indices[0] < min(self._team_size(), robot_count)
        ]
        pair_distances = torch.norm(
            robot_xy[:, pair_indices[0]] - robot_xy[:, pair_indices[1]], dim=-1
        )
        overlap_fraction = torch.clamp(
            (collision_distance - pair_distances) / collision_distance,
            min=0.0,
            max=1.0,
        )
        return overlap_fraction.square().amax(dim=1)

    def _reward_high_level_obstacle_clearance(self):
        distances = self._obstacle_distances()
        if distances is None:
            return torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)

        min_distance = torch.min(distances, dim=1).values
        safe_distance = float(getattr(self.env.cfg.rewards, "high_level_obstacle_safe_distance", 0.55))
        return -torch.square(torch.clamp(safe_distance - min_distance, min=0.0))

    def _reward_high_level_pass(self):
        skill_ids = getattr(self.env, "high_level_skill_ids", None)
        if skill_ids is None:
            return torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)

        robot_xy = self._robot_xy()
        ball_xy = self.env.object_pos_world_frame[:, :2]
        ball_vel = self.env.object_lin_vel[:, :2]
        ball_speed = torch.norm(ball_vel, dim=-1).clamp(min=1e-6)
        ball_dir = ball_vel / ball_speed.unsqueeze(-1)

        rewards = []
        skill_ids = skill_ids[:, : self._team_size()]
        for passer_idx in range(self._team_size()):
            for receiver_idx in range(self._team_size()):
                if receiver_idx == passer_idx:
                    continue
                selected_shoot = (skill_ids[:, passer_idx] == 2).float()
                passer_distance = torch.norm(robot_xy[:, passer_idx, :] - ball_xy, dim=-1)
                receiver_vec = robot_xy[:, receiver_idx, :] - ball_xy
                receiver_distance = torch.norm(receiver_vec, dim=-1).clamp(min=1e-6)
                receiver_dir = receiver_vec / receiver_distance.unsqueeze(-1)
                alignment = torch.sum(ball_dir * receiver_dir, dim=-1).clamp(min=0.0, max=1.0)
                passer_near_ball = torch.exp(-4.0 * torch.square(passer_distance))
                receiver_available = torch.exp(-0.5 * torch.square(receiver_distance - 1.2))
                moving_gate = (ball_speed > 0.35).float()
                rewards.append(selected_shoot * passer_near_ball * receiver_available * alignment * moving_gate)

        if not rewards:
            return torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)
        return torch.stack(rewards, dim=1).amax(dim=1)

    def _reward_high_level_invalid_skill(self):
        invalid_skill_mask = getattr(self.env, "high_level_invalid_skill_mask", None)
        if invalid_skill_mask is None:
            return torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)

        return torch.sum(invalid_skill_mask[:, : self._team_size()].float(), dim=1)

    def _reward_high_level_dribble_ball_control(self):
        """Reward measured, commanded ball motion under a valid dribble skill."""

        valid_dribble = self._valid_executed_skill(1)
        current_distances = self._robot_ball_distances()
        previous_distances = getattr(
            self.env,
            "prev_high_level_robot_ball_distances",
            current_distances,
        )
        previous_distances = previous_distances[:, : self._team_size()]
        control_distance = float(
            getattr(self.env.cfg.rewards, "high_level_dribble_control_distance", 0.8)
        )
        controlled = torch.maximum(current_distances, previous_distances) <= control_distance

        commands = self._commands()[:, :, :2]
        command_speed = torch.norm(commands, dim=-1)
        command_direction = commands / command_speed.unsqueeze(-1).clamp(min=1e-6)
        ball_velocity = self.env.object_lin_vel[:, None, :2]
        command_aligned_speed = torch.sum(ball_velocity * command_direction, dim=-1)

        target_speed = max(
            float(getattr(self.env.cfg.rewards, "high_level_dribble_target_ball_speed", 1.0)),
            1e-6,
        )
        command_tracking = torch.clamp(
            command_aligned_speed / command_speed.clamp(min=1e-6),
            min=0.0,
            max=1.0,
        )
        goalward_speed = torch.sum(
            self.env.object_lin_vel[:, :2] * self._ball_goal_direction(),
            dim=-1,
        )
        goalward_progress = torch.clamp(goalward_speed / target_speed, min=0.0, max=1.0)

        min_command_speed = float(
            getattr(self.env.cfg.rewards, "high_level_skill_command_min_speed", 0.2)
        )
        min_ball_speed = float(
            getattr(self.env.cfg.rewards, "high_level_dribble_min_ball_speed", 0.1)
        )
        moving = (
            (command_speed >= min_command_speed)
            & (torch.norm(self.env.object_lin_vel[:, :2], dim=-1)[:, None] >= min_ball_speed)
        )
        per_robot = (
            0.5 * command_tracking
            + 0.5 * goalward_progress[:, None]
        ) * (valid_dribble & controlled & moving).float()
        return torch.max(per_robot, dim=1).values

    def _reward_high_level_shoot_launch(self):
        """Emit a launch reward only for valid shots that accelerate the ball."""

        valid_shoot = self._valid_executed_skill(2)
        previous_distances = getattr(
            self.env,
            "prev_high_level_robot_ball_distances",
            self._robot_ball_distances(),
        )
        previous_distances = previous_distances[:, : self._team_size()]
        shoot_distance = float(
            getattr(self.env.cfg.rewards, "high_level_shoot_skill_distance", 0.75)
        )

        commands = self._commands()[:, :, :2]
        command_speed = torch.norm(commands, dim=-1)
        command_direction = commands / command_speed.unsqueeze(-1).clamp(min=1e-6)
        current_velocity = self.env.object_lin_vel[:, None, :2]
        previous_velocity = getattr(
            self.env,
            "prev_object_lin_vel",
            self.env.object_lin_vel,
        )[:, None, :2]
        current_projected_speed = torch.sum(current_velocity * command_direction, dim=-1)
        previous_projected_speed = torch.sum(previous_velocity * command_direction, dim=-1)
        delta_projected_speed = current_projected_speed - previous_projected_speed
        ball_speed = torch.norm(current_velocity, dim=-1)
        alignment = current_projected_speed / ball_speed.clamp(min=1e-6)

        min_command_speed = float(
            getattr(self.env.cfg.rewards, "high_level_skill_command_min_speed", 0.2)
        )
        min_ball_speed = float(
            getattr(self.env.cfg.rewards, "high_level_shoot_min_ball_speed", 0.8)
        )
        min_delta_speed = float(
            getattr(self.env.cfg.rewards, "high_level_shoot_min_delta_speed", 0.25)
        )
        target_delta_speed = max(
            float(getattr(self.env.cfg.rewards, "high_level_shoot_target_delta_speed", 1.5)),
            min_delta_speed + 1e-6,
        )
        min_alignment = float(
            getattr(self.env.cfg.rewards, "high_level_shoot_min_command_alignment", 0.6)
        )
        alignment_score = torch.clamp(
            (alignment - min_alignment) / max(1.0 - min_alignment, 1e-6),
            min=0.0,
            max=1.0,
        )
        launch_score = torch.clamp(
            (delta_projected_speed - min_delta_speed)
            / (target_delta_speed - min_delta_speed),
            min=0.0,
            max=1.0,
        )
        launched = (
            valid_shoot
            & (previous_distances <= shoot_distance)
            & (command_speed >= min_command_speed)
            & (current_projected_speed >= min_ball_speed)
            & (delta_projected_speed >= min_delta_speed)
            & (alignment >= min_alignment)
        )
        return torch.max(launch_score * alignment_score * launched.float(), dim=1).values

    def _reward_high_level_approach_ball(self):
        prev_distances = getattr(self.env, "prev_high_level_robot_ball_distances", None)
        if prev_distances is None:
            return torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)

        prev_distances = prev_distances[:, : self._team_size()]
        current_distances = self._robot_ball_distances()
        progress = (prev_distances - current_distances) / max(self.env.dt, 1e-6)
        dribble_distance = float(getattr(self.env.cfg.rewards, "high_level_dribble_skill_distance", 1.0))
        executed_skill_ids = self._skill_ids("high_level_skill_ids")
        requested_skill_ids = self._skill_ids("high_level_requested_skill_ids")
        invalid_skill_mask = self._skill_ids("high_level_invalid_skill_mask")
        far_from_ball = prev_distances > dribble_distance
        valid_requested_walk = (
            (executed_skill_ids == 0)
            & (requested_skill_ids == 0)
            & (invalid_skill_mask == 0)
        )

        # Assign the approach objective to the robot that was closest at the
        # beginning of the physics step. Selecting before measuring progress
        # prevents the policy from switching the rewarded robot after seeing
        # which one happened to move closer.
        attacker = torch.argmin(prev_distances, dim=1, keepdim=True)
        attacker_progress = torch.gather(progress, 1, attacker).squeeze(1)
        attacker_far = torch.gather(far_from_ball, 1, attacker).squeeze(1)
        attacker_walking = torch.gather(valid_requested_walk, 1, attacker).squeeze(1)

        # Keep the sign: moving away must cancel previously earned approach
        # reward instead of being hidden by a zero clamp or by the teammate.
        signed_progress = torch.clamp(attacker_progress, min=-1.0, max=1.0)
        return signed_progress * attacker_far.float() * attacker_walking.float()

    def _reward_high_level_walk_command_alignment(self):
        """Reward walking commands toward the ball and penalize commands away from it."""

        prev_distances = getattr(self.env, "prev_high_level_robot_ball_distances", None)
        if prev_distances is None:
            return torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)

        prev_distances = prev_distances[:, : self._team_size()]
        executed_skill_ids = self._skill_ids("high_level_skill_ids")
        requested_skill_ids = self._skill_ids("high_level_requested_skill_ids")
        invalid_skill_mask = self._skill_ids("high_level_invalid_skill_mask")
        valid_requested_walk = (
            (executed_skill_ids == 0)
            & (requested_skill_ids == 0)
            & (invalid_skill_mask == 0)
        )

        robot_states = self._robot_states()
        quaternion = robot_states[:, :, 3:7]
        qx, qy, qz, qw = quaternion.unbind(dim=-1)
        command_body = self._commands()[:, :, :2]
        # Horizontal part of R(q) @ [cmd_x, cmd_y, 0] for xyzw quaternions.
        command_world = torch.stack(
            (
                (1.0 - 2.0 * (qy.square() + qz.square())) * command_body[:, :, 0]
                + 2.0 * (qx * qy - qz * qw) * command_body[:, :, 1],
                2.0 * (qx * qy + qz * qw) * command_body[:, :, 0]
                + (1.0 - 2.0 * (qx.square() + qz.square())) * command_body[:, :, 1],
            ),
            dim=-1,
        )
        command_speed = torch.norm(command_world, dim=-1)
        command_direction = command_world / command_speed.unsqueeze(-1).clamp(min=1e-6)

        ball_delta = self.env.object_pos_world_frame[:, None, :2] - robot_states[:, :, :2]
        ball_direction = ball_delta / torch.norm(ball_delta, dim=-1, keepdim=True).clamp(min=1e-6)
        alignment = torch.sum(command_direction * ball_direction, dim=-1).clamp(min=-1.0, max=1.0)
        target_speed = max(
            float(getattr(self.env.cfg.rewards, "high_level_approach_walk_speed", 0.9)),
            1e-6,
        )
        speed_fraction = torch.clamp(command_speed / target_speed, min=0.0, max=1.0)

        dribble_distance = float(
            getattr(self.env.cfg.rewards, "high_level_dribble_skill_distance", 1.0)
        )
        far_from_ball = prev_distances > dribble_distance
        attacker = torch.argmin(prev_distances, dim=1, keepdim=True)
        per_robot = (
            alignment
            * speed_fraction
            * far_from_ball.float()
            * valid_requested_walk.float()
        )
        return torch.gather(per_robot, 1, attacker).squeeze(1)

    def _reward_high_level_face_ball_while_approaching(self):
        """Keep the active approaching robot's body pointed at the ball.

        The command-speed gate prevents a stationary robot from accumulating
        this dense orientation reward merely by looking at the ball.  As with
        the approach reward, only the robot that was closest at the beginning
        of the step receives credit, which is also the sole robot in the
        single-robot task.
        """

        prev_distances = getattr(self.env, "prev_high_level_robot_ball_distances", None)
        if prev_distances is None:
            return torch.zeros(self.env.num_envs, dtype=torch.float, device=self.env.device)

        prev_distances = prev_distances[:, : self._team_size()]
        executed_skill_ids = self._skill_ids("high_level_skill_ids")
        requested_skill_ids = self._skill_ids("high_level_requested_skill_ids")
        invalid_skill_mask = self._skill_ids("high_level_invalid_skill_mask")
        valid_requested_walk = (
            (executed_skill_ids == 0)
            & (requested_skill_ids == 0)
            & (invalid_skill_mask == 0)
        )

        robot_states = self._robot_states()
        quaternion = robot_states[:, :, 3:7]
        qx, qy, qz, qw = quaternion.unbind(dim=-1)
        robot_forward_xy = torch.stack(
            (
                1.0 - 2.0 * (qy.square() + qz.square()),
                2.0 * (qx * qy + qz * qw),
            ),
            dim=-1,
        )
        robot_forward_xy = robot_forward_xy / torch.norm(
            robot_forward_xy, dim=-1, keepdim=True
        ).clamp(min=1e-6)

        ball_delta = self.env.object_pos_world_frame[:, None, :2] - robot_states[:, :, :2]
        ball_direction = ball_delta / torch.norm(ball_delta, dim=-1, keepdim=True).clamp(min=1e-6)
        facing_alignment = torch.sum(robot_forward_xy * ball_direction, dim=-1).clamp(
            min=-1.0, max=1.0
        )

        command_speed = torch.norm(self._commands()[:, :, :2], dim=-1)
        target_speed = max(
            float(getattr(self.env.cfg.rewards, "high_level_approach_walk_speed", 0.9)),
            1e-6,
        )
        approach_activity = torch.clamp(command_speed / target_speed, min=0.0, max=1.0)
        dribble_distance = float(
            getattr(self.env.cfg.rewards, "high_level_dribble_skill_distance", 1.0)
        )
        far_from_ball = prev_distances > dribble_distance

        attacker = torch.argmin(prev_distances, dim=1, keepdim=True)
        per_robot = (
            facing_alignment
            * approach_activity
            * far_from_ball.float()
            * valid_requested_walk.float()
        )
        return torch.gather(per_robot, 1, attacker).squeeze(1)

    def _reward_high_level_face_goal_while_moving(self):
        """Require measured motion to be both goalward and body-forward."""

        robot_states = self._robot_states()
        quaternion = robot_states[:, :, 3:7]
        qx, qy, qz, qw = quaternion.unbind(dim=-1)
        robot_forward_xy = torch.stack(
            (
                1.0 - 2.0 * (qy.square() + qz.square()),
                2.0 * (qx * qy + qz * qw),
            ),
            dim=-1,
        )
        robot_forward_xy = robot_forward_xy / torch.norm(
            robot_forward_xy, dim=-1, keepdim=True
        ).clamp(min=1e-6)

        goal_delta = self._goal_xy()[:, None, :] - robot_states[:, :, :2]
        goal_direction = goal_delta / torch.norm(
            goal_delta, dim=-1, keepdim=True
        ).clamp(min=1e-6)
        target_speed = max(
            float(getattr(self.env.cfg.rewards, "high_level_goal_facing_target_speed", 0.5)),
            1e-6,
        )
        velocity = robot_states[:, :, 7:9]
        goalward_activity = torch.clamp(
            torch.sum(velocity * goal_direction, dim=-1) / target_speed,
            min=-1.0, max=1.0,
        )
        forward_activity = torch.clamp(
            torch.sum(velocity * robot_forward_xy, dim=-1) / target_speed,
            min=-1.0, max=1.0,
        )
        executed_skill_ids = self._skill_ids("high_level_skill_ids")
        requested_skill_ids = self._skill_ids("high_level_requested_skill_ids")
        invalid_skill_mask = self._skill_ids("high_level_invalid_skill_mask")
        valid_skill = (executed_skill_ids == requested_skill_ids) & ~invalid_skill_mask

        previous_distances = getattr(
            self.env,
            "prev_high_level_robot_ball_distances",
            self._robot_ball_distances(),
        )
        previous_distances = previous_distances[:, : self._team_size()]
        attacker = torch.argmin(previous_distances, dim=1, keepdim=True)
        # Both conditions must hold. Taking the minimum makes either walking
        # backward or moving away from goal a signed penalty; the previous
        # unsigned speed gate rewarded backward motion while facing the goal.
        per_robot = torch.minimum(
            goalward_activity, forward_activity
        ) * valid_skill.float()
        return torch.gather(per_robot, 1, attacker).squeeze(1)
