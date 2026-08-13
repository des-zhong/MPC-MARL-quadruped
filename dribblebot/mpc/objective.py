"""Modular MPC objective with a fully auditable decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

import torch

from .config import MPCConfig


@dataclass
class MPCObjectiveResult:
    """Per-candidate planning utility and signed additive components."""

    total: torch.Tensor
    components: Dict[str, torch.Tensor]
    valid: torch.Tensor
    return_uncertainty: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class MPCObjective:
    """Score model rollouts while keeping learned reward as the base utility."""

    COMPONENT_NAMES = (
        "predicted_reward_return",
        "uncertainty_penalty",
        "return_uncertainty_penalty",
        "collision_penalty",
        "boundary_penalty",
        "robot_fall_penalty",
        "invalid_skill_penalty",
        "ball_setup_penalty",
        "backward_dribble_penalty",
        "reposition_approach",
        "reposition_command_alignment",
        "skill_switch_penalty",
        "command_smoothness_penalty",
        "ball_progress",
        "possession",
        "terminal_value",
        "predicted_event_adjustment",
    )

    def __init__(
        self,
        schema,
        action_adapter,
        event_names,
        config: MPCConfig,
        terminal_value: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        controlled_robot_count: Optional[int] = None,
    ):
        self.schema = schema
        self.action_adapter = action_adapter
        self.event_names = tuple(event_names)
        self.event_indices = {name: index for index, name in enumerate(self.event_names)}
        self.config = config
        self.terminal_value = terminal_value
        self.controlled_robot_count = int(
            action_adapter.num_robots
            if controlled_robot_count is None
            else controlled_robot_count
        )
        if not 1 <= self.controlled_robot_count <= action_adapter.num_robots:
            raise ValueError(
                "controlled_robot_count must lie between 1 and the joint robot count"
            )
        needs_value = config.objective_mode in {
            "terminal_value_only", "reward_plus_terminal_value"
        }
        if needs_value and terminal_value is None:
            raise ValueError(f"mpc.objective_mode={config.objective_mode!r} requires a terminal-value callable")

    @staticmethod
    def _active_mask(done: torch.Tensor, threshold: float) -> torch.Tensor:
        not_done = (done < threshold).to(done.dtype)
        return torch.cat(
            (
                torch.ones_like(not_done[..., :1]),
                torch.cumprod(not_done[..., :-1], dim=-1),
            ),
            dim=-1,
        )

    def _event(self, rollout: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
        events = rollout["event_probabilities"]
        index = self.event_indices.get(name)
        if index is None:
            return torch.zeros_like(events[..., 0])
        return events[..., index]

    def _geometric_collision(self, predicted_states: torch.Tensor) -> torch.Tensor:
        """Return collision indicators [B,C,H] using masked static box geometry."""

        future = predicted_states[..., 1:, :]
        field = future[..., self.schema.slice("field.geometry")]
        half_extents = field[..., :2].abs().clamp(min=1e-6)
        robot_xy = torch.stack(
            [
                future[..., self.schema.slice(f"robot_{robot}.position")][..., :2]
                for robot in range(self.action_adapter.num_robots)
            ],
            dim=-2,
        )
        ball_xy = future[..., self.schema.slice("ball.position")][..., :2]
        ball_radius_m = field[..., 5].abs()
        collision = torch.zeros_like(future[..., 0])
        for feature in self.schema.features:
            if feature.group != "obstacle":
                continue
            geometry = future[..., feature.start : feature.stop]
            valid = geometry[..., 5] > 0.5
            # The current state schema omits the static box yaw even though the
            # simulator randomizes it.  A circumscribed circle is conservative
            # for every possible orientation and avoids pretending the boxes
            # are axis-aligned.
            obstacle_radius_m = torch.linalg.vector_norm(
                geometry[..., 2:4].abs(), dim=-1
            )
            robot_delta_m = (
                robot_xy - geometry[..., None, :2]
            ) * half_extents[..., None, :]
            ball_delta_m = (ball_xy - geometry[..., :2]) * half_extents
            robot_overlap = (
                torch.linalg.vector_norm(robot_delta_m, dim=-1)
                <= obstacle_radius_m[..., None] + 0.25
            ).any(dim=-1)
            ball_overlap = (
                torch.linalg.vector_norm(ball_delta_m, dim=-1)
                <= obstacle_radius_m + ball_radius_m
            )
            collision = torch.maximum(
                collision,
                ((robot_overlap | ball_overlap) & valid).to(collision.dtype),
            )
        if self.action_adapter.num_robots > 1:
            pairwise = (
                robot_xy[..., :, None, :] - robot_xy[..., None, :, :]
            ) * half_extents[..., None, None, :]
            pairwise_distance = torch.linalg.vector_norm(pairwise, dim=-1)
            diagonal = torch.eye(
                self.action_adapter.num_robots,
                dtype=torch.bool,
                device=robot_xy.device,
            )
            pairwise_distance = pairwise_distance.masked_fill(diagonal, float("inf"))
            teammate_collision = (pairwise_distance < 0.5).any(dim=(-1, -2))
        else:
            teammate_collision = torch.zeros_like(collision, dtype=torch.bool)
        return torch.maximum(collision, teammate_collision.to(collision.dtype))

    def _affordance_violation(
        self,
        predicted_states: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Detect requested skills that the live wrapper would downgrade."""

        before = predicted_states[..., :-1, :]
        field = before[..., self.schema.slice("field.geometry")]
        scale = field[..., :2].abs().clamp(min=1e-6)
        ball_xy = before[..., self.schema.slice("ball.position")][..., :2]
        skills, _ = self.action_adapter.unpack(action_sequences)
        invalid = torch.zeros_like(skills, dtype=torch.bool)
        for robot in range(self.controlled_robot_count):
            position = before[..., self.schema.slice(f"robot_{robot}.position")][..., :2]
            yaw_pair = before[..., self.schema.slice(f"robot_{robot}.yaw_sin_cos")]
            delta_world = (ball_xy - position) * scale
            sin_yaw, cos_yaw = yaw_pair[..., 0], yaw_pair[..., 1]
            local_x = cos_yaw * delta_world[..., 0] + sin_yaw * delta_world[..., 1]
            local_y = -sin_yaw * delta_world[..., 0] + cos_yaw * delta_world[..., 1]
            distance = torch.linalg.vector_norm(delta_world, dim=-1)
            can_dribble = distance <= self.config.dribble_affordance_distance_m
            can_shoot = (
                (distance <= self.config.shoot_affordance_distance_m)
                & (local_x >= self.config.shoot_min_forward_m)
                & (local_y.abs() <= self.config.shoot_lateral_reach_m)
            )
            invalid[..., robot] = (
                ((skills[..., robot] == 1) & ~can_dribble)
                | ((skills[..., robot] == 2) & ~can_shoot)
            )
        return invalid.to(action_sequences.dtype).sum(dim=-1)

    def _ball_setup_error(
        self,
        predicted_states: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Return dimensionless setup error for requested ball skills.

        The low-level dribble controller is most effective with the ball a
        short distance in front of the body.  The shooting controller was
        trained to place the base ``shoot_target_distance_m`` behind the ball
        along the requested (field-frame) kick direction.  Distance-only
        affordances cannot distinguish either setup from a ball underneath the
        robot, so make that distinction explicitly in the planning objective.
        """

        future = predicted_states[..., 1:, :]
        field_scale = future[..., self.schema.slice("field.geometry")][..., :2]
        field_scale = field_scale.abs().clamp(min=1e-6)
        ball_xy = future[..., self.schema.slice("ball.position")][..., :2]
        skills, parameters = self.action_adapter.unpack(action_sequences)
        skills = skills[..., : self.controlled_robot_count]
        parameters = parameters[..., : self.controlled_robot_count, :]
        total_error = torch.zeros_like(skills[..., 0], dtype=action_sequences.dtype)
        dribble_errors = []
        ball_distances = []

        dribble_target = max(float(self.config.dribble_target_forward_m), 1e-6)
        shoot_target = max(float(self.config.shoot_target_distance_m), 1e-6)
        for robot in range(self.controlled_robot_count):
            position = future[
                ..., self.schema.slice(f"robot_{robot}.position")
            ][..., :2]
            yaw_pair = future[
                ..., self.schema.slice(f"robot_{robot}.yaw_sin_cos")
            ]
            delta_world = (ball_xy - position) * field_scale
            sin_yaw, cos_yaw = yaw_pair[..., 0], yaw_pair[..., 1]
            delta_local = torch.stack(
                (
                    cos_yaw * delta_world[..., 0] + sin_yaw * delta_world[..., 1],
                    -sin_yaw * delta_world[..., 0] + cos_yaw * delta_world[..., 1],
                ),
                dim=-1,
            )

            dribble_error = (
                (delta_local[..., 0] - dribble_target).square()
                + delta_local[..., 1].square()
            ) / (dribble_target * dribble_target)
            dribble_errors.append(dribble_error)
            ball_distances.append(torch.linalg.vector_norm(delta_world, dim=-1))

            command_xy = parameters[..., robot, :2]
            command_speed = torch.linalg.vector_norm(command_xy, dim=-1, keepdim=True)
            heading = torch.stack((cos_yaw, sin_yaw), dim=-1)
            command_direction = torch.where(
                command_speed > 1e-6,
                command_xy / command_speed.clamp(min=1e-6),
                heading,
            )
            shoot_target_delta = shoot_target * command_direction
            shoot_error = (
                (delta_world - shoot_target_delta).square().sum(dim=-1)
                / (shoot_target * shoot_target)
            )

            robot_skill = skills[..., robot]
            total_error = total_error + torch.where(
                robot_skill == 1,
                dribble_error,
                torch.where(robot_skill == 2, shoot_error, torch.zeros_like(shoot_error)),
            )

        # Walking is the approach skill. Once the closest walking robot enters
        # control range, continuing toward the ball center is exactly what
        # produces the under-chassis failure. If no robot has started a ball
        # skill yet, shape that final part of the approach toward the same
        # useful front-of-body dribble setup.
        ball_skill_active = (skills != 0).any(dim=-1)
        distances = torch.stack(ball_distances, dim=-1)
        approach_errors = torch.stack(dribble_errors, dim=-1)
        walking_distances = distances.masked_fill(skills != 0, float("inf"))
        approach_robot = walking_distances.argmin(dim=-1, keepdim=True)
        approach_distance = torch.gather(
            walking_distances, -1, approach_robot
        ).squeeze(-1)
        approach_error = torch.gather(
            approach_errors, -1, approach_robot
        ).squeeze(-1)
        shape_approach = (
            ~ball_skill_active
            & (approach_distance <= self.config.dribble_affordance_distance_m)
        )
        total_error = total_error + torch.where(
            shape_approach, approach_error, torch.zeros_like(approach_error)
        )
        # Keep this shaping term subordinate to task reward and hard safety
        # penalties when a model predicts a badly misplaced robot or ball.
        return total_error.clamp(max=4.0)

    def _backward_dribble(
        self,
        predicted_states: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized predicted backward speed under dribble actions."""

        future = predicted_states[..., 1:, :]
        skills, _ = self.action_adapter.unpack(action_sequences)
        backward = torch.zeros_like(skills[..., 0], dtype=action_sequences.dtype)
        speed_scale = float(self.config.dribble_forward_speed_scale_mps)
        for robot in range(self.controlled_robot_count):
            velocity = future[
                ..., self.schema.slice(f"robot_{robot}.linear_velocity")
            ][..., :2]
            yaw_pair = future[
                ..., self.schema.slice(f"robot_{robot}.yaw_sin_cos")
            ]
            forward_speed = (
                yaw_pair[..., 1] * velocity[..., 0]
                + yaw_pair[..., 0] * velocity[..., 1]
            )
            backward = backward + (
                (-forward_speed / speed_scale).clamp(min=0.0, max=1.0)
                * (skills[..., robot] == 1).to(action_sequences.dtype)
            )
        return backward

    def _reposition_approach(
        self,
        predicted_states: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure approach progress and body-command alignment for the attacker.

        Only the closest robot is treated as the attacker, matching the live
        high-level reward and avoiding pulling every teammate toward the ball.
        Command alignment is computed from the state *before* each action. In
        particular, the emphasized first step uses the measured state and
        cannot be fabricated by an optimistic multi-step world-model rollout.
        Both terms switch off inside dribble range, where ball-setup shaping
        takes over and prevents walking over the ball.
        """

        before = predicted_states[..., :-1, :]
        after = predicted_states[..., 1:, :]
        skills, parameters = self.action_adapter.unpack(action_sequences)
        before_scale = before[..., self.schema.slice("field.geometry")][..., :2]
        after_scale = after[..., self.schema.slice("field.geometry")][..., :2]
        before_scale = before_scale.abs().clamp(min=1e-6)
        after_scale = after_scale.abs().clamp(min=1e-6)
        ball_before = before[..., self.schema.slice("ball.position")][..., :2]
        ball_after = after[..., self.schema.slice("ball.position")][..., :2]

        distances_before = []
        distances_after = []
        alignments = []
        for robot in range(self.controlled_robot_count):
            position_before = before[
                ..., self.schema.slice(f"robot_{robot}.position")
            ][..., :2]
            position_after = after[
                ..., self.schema.slice(f"robot_{robot}.position")
            ][..., :2]
            delta_before = (ball_before - position_before) * before_scale
            delta_after = (ball_after - position_after) * after_scale
            distance_before = torch.linalg.vector_norm(delta_before, dim=-1)
            distances_before.append(distance_before)
            distances_after.append(torch.linalg.vector_norm(delta_after, dim=-1))

            yaw_pair = before[
                ..., self.schema.slice(f"robot_{robot}.yaw_sin_cos")
            ]
            sin_yaw, cos_yaw = yaw_pair[..., 0], yaw_pair[..., 1]
            command_body = parameters[..., robot, :2]
            command_world = torch.stack(
                (
                    cos_yaw * command_body[..., 0]
                    - sin_yaw * command_body[..., 1],
                    sin_yaw * command_body[..., 0]
                    + cos_yaw * command_body[..., 1],
                ),
                dim=-1,
            )
            direction_to_ball = delta_before / distance_before.unsqueeze(-1).clamp(
                min=1e-6
            )
            # Metres per second along the direct bearing to the ball. Positive
            # means the requested body-frame walk command points toward it.
            alignments.append((command_world * direction_to_ball).sum(dim=-1))

        distance_before = torch.stack(distances_before, dim=-1)
        distance_after = torch.stack(distances_after, dim=-1)
        command_alignment = torch.stack(alignments, dim=-1)
        attacker = distance_before.argmin(dim=-1, keepdim=True)
        attacker_distance = torch.gather(distance_before, -1, attacker).squeeze(-1)
        attacker_next_distance = torch.gather(
            distance_after, -1, attacker
        ).squeeze(-1)
        attacker_skill = torch.gather(skills, -1, attacker).squeeze(-1)
        attacker_alignment = torch.gather(
            command_alignment, -1, attacker
        ).squeeze(-1)
        active = (
            (attacker_skill == 0)
            & (attacker_distance > self.config.dribble_affordance_distance_m)
        ).to(action_sequences.dtype)
        progress = (attacker_distance - attacker_next_distance).clamp(
            min=-1.0, max=1.0
        )
        alignment_scale = max(
            abs(float(self.action_adapter.bounds[0].low[0])),
            abs(float(self.action_adapter.bounds[0].low[1])),
            abs(float(self.action_adapter.bounds[0].high[0])),
            abs(float(self.action_adapter.bounds[0].high[1])),
            1e-6,
        )
        normalized_alignment = (attacker_alignment / alignment_scale).clamp(
            min=-1.0, max=1.0
        )
        return progress * active, normalized_alignment * active

    def _hybrid_regularization(
        self,
        initial_states: torch.Tensor,
        action_sequences: torch.Tensor,
        discount: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skills, parameters = self.action_adapter.unpack(action_sequences)
        skills = skills[..., : self.controlled_robot_count]
        parameters = parameters[..., : self.controlled_robot_count, :]
        normalized = self.action_adapter.normalize_parameters(skills, parameters)
        masks = self.action_adapter._selected(skills, "mask", parameters.dtype)
        current_skills = torch.stack(
            [
                initial_states[:, self.schema.slice(f"robot_{robot}.skill_one_hot")].argmax(-1)
                for robot in range(self.controlled_robot_count)
            ],
            dim=-1,
        )
        current_commands = torch.stack(
            [
                initial_states[:, self.schema.slice(f"robot_{robot}.previous_command")]
                for robot in range(self.controlled_robot_count)
            ],
            dim=-2,
        )
        current_masks = torch.stack(
            [
                initial_states[:, self.schema.slice(f"robot_{robot}.parameter_mask")]
                for robot in range(self.controlled_robot_count)
            ],
            dim=-2,
        )
        previous_skills = torch.cat(
            (current_skills[:, None, None, :].expand(-1, skills.shape[1], 1, -1), skills[..., :-1, :]),
            dim=-2,
        )
        switch = (skills != previous_skills).to(parameters.dtype).sum(dim=-1)

        previous_commands = torch.cat(
            (
                current_commands[:, None, None, :, :].expand(-1, skills.shape[1], 1, -1, -1),
                normalized[..., :-1, :, :],
            ),
            dim=-3,
        )
        previous_masks = torch.cat(
            (
                current_masks[:, None, None, :, :].expand(-1, skills.shape[1], 1, -1, -1),
                masks[..., :-1, :, :],
            ),
            dim=-3,
        )
        compatible = (skills == previous_skills).unsqueeze(-1).to(parameters.dtype)
        common_mask = masks * previous_masks * compatible
        command_change = ((normalized - previous_commands).square() * common_mask).sum(dim=(-1, -2))
        weights = discount * active
        return (switch * weights).sum(-1), (command_change * weights).sum(-1)

    def evaluate(
        self,
        initial_states: torch.Tensor,
        action_sequences: torch.Tensor,
        rollout: Mapping[str, torch.Tensor],
    ) -> MPCObjectiveResult:
        rewards = rollout["predicted_rewards"]
        if action_sequences.ndim == 3:
            # Backward-compatible single-sequence input used by objective unit
            # tests and diagnostic callers; evaluate it for every rollout row.
            action_sequences = action_sequences[:, None].expand(
                -1, rewards.shape[1], -1, -1
            )
        done = rollout["predicted_done_probabilities"]
        horizon = rewards.shape[-1]
        discount = torch.pow(
            torch.as_tensor(self.config.gamma, dtype=rewards.dtype, device=rewards.device),
            torch.arange(horizon, dtype=rewards.dtype, device=rewards.device),
        )
        if self.config.terminal_handling == "probability_weighted":
            survival = torch.cat(
                (torch.ones_like(done[..., :1]), torch.cumprod(1.0 - done[..., :-1], dim=-1)),
                dim=-1,
            )
            active = survival
            terminal_survival = torch.prod(1.0 - done, dim=-1)
        else:
            active = self._active_mask(done, self.config.termination_threshold)
            terminal_survival = torch.prod(
                (done < self.config.termination_threshold).to(done.dtype), dim=-1
            )
        weights = discount * active
        components = {
            name: torch.zeros_like(rewards[..., 0])
            for name in self.COMPONENT_NAMES
        }

        predicted_return = (rewards * weights).sum(-1)
        if self.config.objective_mode != "terminal_value_only":
            components["predicted_reward_return"] = predicted_return
        state_uncertainty = rollout.get("state_uncertainty", torch.zeros_like(rewards))
        reward_uncertainty = rollout.get("reward_uncertainty", torch.zeros_like(rewards))
        components["uncertainty_penalty"] = -self.config.uncertainty_penalty * (
            state_uncertainty * weights
        ).sum(-1)
        member_rewards = rollout.get("member_predicted_rewards")
        member_returns = None
        if member_rewards is not None:
            member_done = rollout["member_predicted_done_probabilities"]
            member_active = self._active_mask(
                member_done, self.config.termination_threshold
            )
            member_returns = (
                member_rewards
                * member_active
                * discount.reshape((1,) * (member_rewards.ndim - 1) + (horizon,))
            ).sum(-1)
            return_uncertainty = member_returns.std(dim=0, unbiased=False)
        else:
            return_uncertainty = torch.sqrt(
                (
                    reward_uncertainty.clamp(min=0.0)
                    * weights.square()
                ).sum(-1).clamp(min=0.0)
            )
        if self.config.ensemble_objective == "mean_minus_std":
            if member_returns is not None and self.config.objective_mode != "terminal_value_only":
                components["predicted_reward_return"] = member_returns.mean(dim=0)
            components["return_uncertainty_penalty"] = (
                -self.config.return_std_penalty * return_uncertainty
            )
        elif self.config.ensemble_objective == "minimum":
            if member_returns is None:
                raise ValueError(
                    "ensemble_objective='minimum' requires rollout['member_predicted_rewards']; "
                    "the loaded world model does not expose per-member recurrent rollouts"
                )
            if self.config.objective_mode != "terminal_value_only":
                components["predicted_reward_return"] = member_returns.min(dim=0).values

        collision_event = torch.maximum(
            torch.maximum(
                self._event(rollout, "ball_obstacle_collision"),
                self._event(rollout, "robot_obstacle_collision"),
            ),
            self._event(rollout, "teammate_collision"),
        )
        geometric_collision = self._geometric_collision(rollout["predicted_states"])
        collision = torch.maximum(collision_event, geometric_collision)
        components["collision_penalty"] = -self.config.collision_penalty * (
            collision * weights
        ).sum(-1)

        future = rollout["predicted_states"][..., 1:, :]
        ball_xy = future[..., self.schema.slice("ball.position")][..., :2]
        geometric_boundary = (ball_xy.abs() > 1.0).any(dim=-1).to(rewards.dtype)
        boundary = torch.maximum(self._event(rollout, "out_of_bounds"), geometric_boundary)
        components["boundary_penalty"] = -self.config.out_of_bounds_penalty * (
            boundary * weights
        ).sum(-1)

        fallen = torch.stack(
            [
                future[..., self.schema.slice(f"robot_{robot}.fallen")].squeeze(-1)
                for robot in range(self.action_adapter.num_robots)
            ],
            dim=-1,
        ).amax(dim=-1)
        components["robot_fall_penalty"] = -self.config.robot_fall_penalty * (
            fallen * weights
        ).sum(-1)
        affordance = self._affordance_violation(
            rollout["predicted_states"], action_sequences
        )
        components["invalid_skill_penalty"] = -self.config.invalid_skill_penalty * (
            affordance * weights
        ).sum(-1)
        if self.config.ball_setup_penalty:
            setup_error = self._ball_setup_error(
                rollout["predicted_states"], action_sequences
            )
            components["ball_setup_penalty"] = -self.config.ball_setup_penalty * (
                setup_error * weights
            ).sum(-1)
        if self.config.backward_dribble_penalty:
            backward_dribble = self._backward_dribble(
                rollout["predicted_states"], action_sequences
            )
            components["backward_dribble_penalty"] = (
                -self.config.backward_dribble_penalty
                * (backward_dribble * weights).sum(-1)
            )
        if (
            self.config.reposition_approach_coefficient
            or self.config.reposition_command_alignment_coefficient
        ):
            approach, command_alignment = self._reposition_approach(
                rollout["predicted_states"], action_sequences
            )
            first_multiplier = float(self.config.reposition_first_step_multiplier)
            approach_weights = torch.cat(
                (weights[..., :1] * first_multiplier, weights[..., 1:]), dim=-1
            )
            components["reposition_approach"] = (
                self.config.reposition_approach_coefficient
                * (approach * approach_weights).sum(-1)
            )
            components["reposition_command_alignment"] = (
                self.config.reposition_command_alignment_coefficient
                * (command_alignment * approach_weights).sum(-1)
            )

        switches, command_changes = self._hybrid_regularization(
            initial_states, action_sequences, discount, active
        )
        components["skill_switch_penalty"] = -self.config.skill_switch_penalty * switches
        components["command_smoothness_penalty"] = (
            -self.config.command_change_penalty * command_changes
        )

        if self.config.ball_progress_coefficient:
            all_ball_x = rollout["predicted_states"][
                ..., self.schema.slice("ball.position")
            ][..., 0]
            field = rollout["predicted_states"][
                ..., :-1, self.schema.slice("field.geometry")
            ][..., 0]
            progress_m = (all_ball_x[..., 1:] - all_ball_x[..., :-1]) * field
            components["ball_progress"] = self.config.ball_progress_coefficient * (
                progress_m * weights
            ).sum(-1)
        if self.config.possession_coefficient:
            possession = future[
                ..., self.schema.slice("ball.possessed")
            ].squeeze(-1)
            components["possession"] = self.config.possession_coefficient * (
                possession * weights
            ).sum(-1)

        event_adjustment = torch.zeros_like(predicted_return)
        configured_events = dict(self.config.event_coefficients)
        if self.config.goal_probability_bonus:
            configured_events.setdefault("goal", self.config.goal_probability_bonus)
        for name, coefficient in configured_events.items():
            if name not in self.event_indices:
                continue
            event_adjustment = event_adjustment + float(coefficient) * (
                self._event(rollout, name) * weights
            ).sum(-1)
        components["predicted_event_adjustment"] = event_adjustment

        diagnostics = {
            "predicted_discounted_reward_return": predicted_return,
            "terminal_state_value": torch.zeros_like(predicted_return),
            "discounted_terminal_value": torch.zeros_like(predicted_return),
            "terminal_value_contribution": torch.zeros_like(predicted_return),
            "terminal_survival_probability": terminal_survival,
            "terminal_state_uncertainty": state_uncertainty[..., -1],
            "terminal_value_clipped": torch.zeros_like(predicted_return),
        }
        if self.config.objective_mode in {"terminal_value_only", "reward_plus_terminal_value"}:
            terminal = self.terminal_value(rollout["predicted_states"][..., -1, :])
            if terminal.shape != predicted_return.shape:
                raise ValueError(
                    f"Terminal value must return {tuple(predicted_return.shape)}, got {tuple(terminal.shape)}"
                )
            diagnostics["terminal_state_value"] = terminal
            if self.config.terminal_value_clip:
                clipped = terminal
                if self.config.terminal_value_clip_min is not None:
                    clipped = clipped.clamp(min=self.config.terminal_value_clip_min)
                if self.config.terminal_value_clip_max is not None:
                    clipped = clipped.clamp(max=self.config.terminal_value_clip_max)
                diagnostics["terminal_value_clipped"] = (clipped != terminal).to(terminal.dtype)
                terminal = clipped
            discounted = (self.config.gamma ** horizon) * terminal * terminal_survival
            if self.config.terminal_value_uncertainty_gating:
                discounted = discounted * torch.exp(
                    -self.config.terminal_value_uncertainty_beta * state_uncertainty[..., -1]
                )
            contribution = self.config.terminal_value_coefficient * discounted
            diagnostics["discounted_terminal_value"] = discounted
            diagnostics["terminal_value_contribution"] = contribution
            components["terminal_value"] = contribution

        finite_rollout = torch.isfinite(rewards).all(-1)
        for value in rollout.values():
            if (
                torch.is_tensor(value)
                and value.ndim >= 3
                and value.shape[:2] == rewards.shape[:2]
            ):
                finite_rollout &= torch.isfinite(value).flatten(2).all(-1)
        finite_actions = torch.isfinite(action_sequences).flatten(2).all(-1)
        valid = finite_rollout & finite_actions
        total = sum(components.values())
        total = torch.where(
            valid & torch.isfinite(total),
            total,
            torch.full_like(total, self.config.invalid_objective),
        )
        return MPCObjectiveResult(total, components, valid, return_uncertainty, diagnostics)
