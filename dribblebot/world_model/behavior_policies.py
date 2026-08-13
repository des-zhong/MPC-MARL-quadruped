"""Useful high-level behavior mixture for simulator data collection."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch

from .action_adapter import JointActionAdapter, Skill
from .schema import StateSchema


class BehaviorMixture:
    """Mix random-valid, scripted, optional-policy, and targeted actions.

    State positions are stored normalized by the field half-extents.  All
    geometric decisions in this class are deliberately made after converting
    them back to metres; this keeps behavior thresholds and support offsets
    invariant when the configured field dimensions change.
    """

    SOURCE_NAMES = (
        "random_valid",
        "scripted",
        "existing_policy",
        "targeted_rare_event",
    )
    SAMPLING_SOURCE_NAMES = SOURCE_NAMES
    REPEAT_SOURCE_NAME = "repeat_previous"
    OUTPUT_SOURCE_NAMES = SOURCE_NAMES + (REPEAT_SOURCE_NAME,)

    RANDOM_SAMPLING_NAMES = (
        "uniform",
        "gaussian_nominal",
        "boundary_focused",
        "goal_directed",
    )
    DEFAULT_RANDOM_SAMPLING = {
        "uniform": 0.30,
        "gaussian_nominal": 0.30,
        "boundary_focused": 0.30,
        "goal_directed": 0.10,
    }

    TARGETED_SCENARIOS = (
        "goal",
        "own_goal",
        "out_of_bounds",
        "ball_obstacle_collision",
        "robot_obstacle_collision",
        "teammate_collision",
        "possession_acquired",
        "possession_lost",
        "successful_shot",
        "failed_shot",
        "pass",
    )

    def __init__(
        self,
        action_adapter: JointActionAdapter,
        schema: StateSchema,
        mixture: Mapping[str, float],
        repeat_previous_probability: float = 0.35,
        existing_policy: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        seed: int = 42,
        random_sampling: Optional[Mapping[str, float]] = None,
    ):
        self.action_adapter = action_adapter
        self.schema = schema
        self.existing_policy = existing_policy

        weights = self._normalized_weights(mixture, self.SAMPLING_SOURCE_NAMES, "behavior mixture")
        if existing_policy is None and weights["existing_policy"]:
            unavailable = weights["existing_policy"]
            weights["existing_policy"] = 0.0
            denominator = weights["random_valid"] + weights["scripted"]
            if denominator:
                weights["random_valid"] += unavailable * weights["random_valid"] / denominator
                weights["scripted"] += unavailable * weights["scripted"] / denominator
            else:
                weights["random_valid"] = unavailable
        total = sum(weights.values())
        self.weights = torch.tensor(
            [weights[name] / total for name in self.SAMPLING_SOURCE_NAMES],
            dtype=torch.float32,
        )

        random_weights = self._normalized_weights(
            self.DEFAULT_RANDOM_SAMPLING if random_sampling is None else random_sampling,
            self.RANDOM_SAMPLING_NAMES,
            "random sampling mixture",
        )
        random_total = sum(random_weights.values())
        self.random_sampling_weights = torch.tensor(
            [random_weights[name] / random_total for name in self.RANDOM_SAMPLING_NAMES],
            dtype=torch.float32,
        )

        probability = float(repeat_previous_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("repeat_previous_probability must lie in [0, 1]")
        self.repeat_previous_probability = probability
        self.generator = torch.Generator().manual_seed(seed)
        self._obstacle_names = tuple(
            feature.name
            for feature in self.schema.features
            if feature.name.startswith("obstacle_") and feature.name.endswith(".geometry")
        )

    @staticmethod
    def _normalized_weights(
        configured: Mapping[str, float],
        names: Sequence[str],
        description: str,
    ) -> Dict[str, float]:
        unknown = sorted(set(configured) - set(names))
        if unknown:
            raise ValueError(f"Unknown {description} entries: {unknown}")
        weights = {name: float(configured.get(name, 0.0)) for name in names}
        negative = [name for name, value in weights.items() if value < 0.0]
        if negative:
            raise ValueError(f"{description.capitalize()} weights must be non-negative: {negative}")
        if sum(weights.values()) <= 0.0:
            raise ValueError(f"{description.capitalize()} must have positive total weight")
        return weights

    def _generator_for(self, device: Union[torch.device, str]):
        return self.generator if torch.device(device).type == "cpu" else None

    def _field_coordinates(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        field = states[:, self.schema.slice("field.geometry")]
        half_extents = field[:, :2].abs().clamp(min=1e-6)
        robot_xy_normalized = torch.stack(
            [
                states[:, self.schema.slice(f"robot_{i}.position")][:, :2]
                for i in range(self.action_adapter.num_robots)
            ],
            dim=1,
        )
        ball_xy_normalized = states[:, self.schema.slice("ball.position")][:, :2]
        robot_xy = robot_xy_normalized * half_extents[:, None, :]
        ball_xy = ball_xy_normalized * half_extents
        yaw_pairs = torch.stack(
            [
                states[:, self.schema.slice(f"robot_{i}.yaw_sin_cos")]
                for i in range(self.action_adapter.num_robots)
            ],
            dim=1,
        )
        yaw = torch.atan2(yaw_pairs[..., 0], yaw_pairs[..., 1])
        return field, half_extents, robot_xy, ball_xy, yaw

    @staticmethod
    def _unit(vectors: torch.Tensor, fallback: Optional[torch.Tensor] = None) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        result = vectors / norm.clamp(min=1e-6)
        if fallback is None:
            fallback = torch.zeros_like(result)
            fallback[..., 0] = 1.0
        return torch.where(norm > 1e-6, result, fallback)

    @staticmethod
    def _world_to_local(world: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cos_yaw, sin_yaw = yaw.cos(), yaw.sin()
        return torch.stack(
            (
                cos_yaw * world[..., 0] + sin_yaw * world[..., 1],
                -sin_yaw * world[..., 0] + cos_yaw * world[..., 1],
            ),
            dim=-1,
        )

    def _random(self, states: torch.Tensor, scripted_actions: torch.Tensor) -> torch.Tensor:
        batch = states.shape[0]
        device = states.device
        generator = self._generator_for(device)
        actions = self.action_adapter.random_valid((batch,), device, generator)
        skills, parameters = self.action_adapter.unpack(actions)
        normalized = self.action_adapter.normalize_parameters(skills, parameters)

        sampling_mode = torch.multinomial(
            self.random_sampling_weights.to(device),
            batch,
            replacement=True,
            generator=generator,
        )
        robot_shape = (batch, self.action_adapter.num_robots, 3)
        gaussian = torch.randn(*robot_shape, device=device, generator=generator).clamp(-2.0, 2.0) * 0.25
        signs = torch.where(
            torch.rand(*robot_shape, device=device, generator=generator) < 0.5,
            -torch.ones((), device=device),
            torch.ones((), device=device),
        )
        boundary = signs * (
            0.8 + 0.2 * torch.rand(*robot_shape, device=device, generator=generator)
        )
        normalized = torch.where((sampling_mode == 1)[:, None, None], gaussian, normalized)
        normalized = torch.where((sampling_mode == 2)[:, None, None], boundary, normalized)
        sampled = self.action_adapter.pack(
            skills,
            self.action_adapter.denormalize_parameters(skills, normalized),
        )
        return torch.where((sampling_mode == 3)[:, None], scripted_actions, sampled)

    def _scripted(self, states: torch.Tensor) -> torch.Tensor:
        batch = states.shape[0]
        device = states.device
        field, _, robot_xy, ball_xy, yaw = self._field_coordinates(states)
        to_ball = ball_xy[:, None, :] - robot_xy
        distances = torch.linalg.vector_norm(to_ball, dim=-1)
        closest = distances.argmin(dim=1)
        skills = torch.full((batch, self.action_adapter.num_robots), int(Skill.REPOSITION), device=device, dtype=torch.long)
        normalized = torch.zeros(batch, self.action_adapter.num_robots, 3, device=device, dtype=states.dtype)

        goal_xy = torch.stack((field[:, 3], torch.zeros_like(field[:, 3])), dim=-1)
        to_goal = goal_xy - ball_xy
        goal_direction = self._unit(to_goal)
        for robot in range(self.action_adapter.num_robots):
            actor = closest == robot
            ball_direction = self._unit(to_ball[:, robot])
            aligned = torch.sum(ball_direction * goal_direction, dim=-1) > 0.5
            # Match the wrapper's physical affordance radii: shooting requires
            # a tighter setup than dribbling.  Both thresholds are metres.
            shoot_near = distances[:, robot] < 0.75
            dribble_near = distances[:, robot] < 1.00
            shoot = actor & shoot_near & aligned
            dribble = actor & dribble_near & ~shoot
            skills[dribble, robot] = int(Skill.DRIBBLE)
            skills[shoot, robot] = int(Skill.SHOOT)

            behind_ball = ball_xy - 0.45 * goal_direction
            target = torch.where(actor[:, None], to_ball[:, robot], behind_ball - robot_xy[:, robot])
            engaged = dribble_near[:, None] & actor[:, None]
            command_world = torch.where(engaged, goal_direction, self._unit(target))
            normalized[:, robot, :2] = self._world_to_local(command_world, yaw[:, robot])
            desired_direction = torch.where(engaged, goal_direction, self._unit(target))
            desired_yaw = torch.atan2(desired_direction[:, 1], desired_direction[:, 0])
            yaw_error = torch.atan2(
                torch.sin(desired_yaw - yaw[:, robot]),
                torch.cos(desired_yaw - yaw[:, robot]),
            )
            normalized[:, robot, 2] = (yaw_error / torch.pi).clamp(-1.0, 1.0)

        return self.action_adapter.pack(
            skills,
            self.action_adapter.denormalize_parameters(skills, normalized),
        )

    def _nearest_obstacle_direction(
        self,
        states: torch.Tensor,
        half_extents: torch.Tensor,
        origin_xy: torch.Tensor,
    ) -> torch.Tensor:
        if not self._obstacle_names:
            fallback = torch.zeros_like(origin_xy)
            fallback[:, 0] = 1.0
            return fallback
        centers = []
        valid = []
        for name in self._obstacle_names:
            geometry = states[:, self.schema.slice(name)]
            centers.append(geometry[:, :2] * half_extents)
            valid.append(geometry[:, 5] > 0.5)
        obstacle_xy = torch.stack(centers, dim=1)
        valid_mask = torch.stack(valid, dim=1)
        delta = obstacle_xy - origin_xy[:, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        distance = torch.where(valid_mask, distance, torch.full_like(distance, float("inf")))
        nearest = distance.argmin(dim=1)
        rows = torch.arange(states.shape[0], device=states.device)
        direction = delta[rows, nearest]
        any_valid = valid_mask.any(dim=1)
        fallback = torch.zeros_like(direction)
        fallback[:, 0] = 1.0
        return torch.where(any_valid[:, None], self._unit(direction), fallback)

    def _targeted(self, states: torch.Tensor, scenarios: Sequence[str]) -> torch.Tensor:
        """Build deliberate canonical actions for staged rare-event states."""

        batch = states.shape[0]
        device = states.device
        field, half_extents, robot_xy, ball_xy, yaw = self._field_coordinates(states)
        to_ball = ball_xy[:, None, :] - robot_xy
        closest = torch.linalg.vector_norm(to_ball, dim=-1).argmin(dim=1)
        possessor = states[:, self.schema.slice("ball.possessor_one_hot")]
        robot_possessor = possessor[:, 1:].argmax(dim=1)
        valid_possessor = (possessor[:, 0] < 0.5) & (possessor[:, 1:].sum(dim=1) == 1)
        actor = torch.where(valid_possessor, robot_possessor, closest)
        teammate = (actor + 1) % self.action_adapter.num_robots

        goal_xy = torch.stack((field[:, 3], torch.zeros_like(field[:, 3])), dim=-1)
        own_goal_xy = torch.stack((field[:, 2], torch.zeros_like(field[:, 2])), dim=-1)
        goal_direction = self._unit(goal_xy - ball_xy)
        own_goal_direction = self._unit(own_goal_xy - ball_xy)
        teammate_xy = robot_xy.gather(1, teammate[:, None, None].expand(-1, 1, 2)).squeeze(1)
        pass_direction = self._unit(teammate_xy - ball_xy)
        ball_obstacle_direction = self._nearest_obstacle_direction(states, half_extents, ball_xy)
        actor_xy = robot_xy.gather(1, actor[:, None, None].expand(-1, 1, 2)).squeeze(1)
        robot_obstacle_direction = self._nearest_obstacle_direction(states, half_extents, actor_xy)
        sideline_direction = torch.zeros_like(ball_xy)
        sideline_direction[:, 1] = torch.where(ball_xy[:, 1] >= 0.0, 1.0, -1.0)

        skills = torch.full((batch, self.action_adapter.num_robots), int(Skill.REPOSITION), device=device, dtype=torch.long)
        normalized = torch.zeros(batch, self.action_adapter.num_robots, 3, device=device, dtype=states.dtype)
        rows = torch.arange(batch, device=device)

        def command(
            selected: torch.Tensor,
            robots: torch.Tensor,
            skill: Skill,
            world_direction: torch.Tensor,
            magnitude: float,
            yaw_magnitude: float = 0.0,
        ) -> None:
            selected_rows = rows[selected]
            selected_robots = robots[selected]
            if selected_rows.numel() == 0:
                return
            skills[selected_rows, selected_robots] = int(skill)
            local = self._world_to_local(
                self._unit(world_direction[selected]),
                yaw[selected_rows, selected_robots],
            )
            normalized[selected_rows, selected_robots, :2] = float(magnitude) * local
            normalized[selected_rows, selected_robots, 2] = float(yaw_magnitude)

        scenario_masks = {
            name: torch.tensor([scenario == name for scenario in scenarios], device=device, dtype=torch.bool)
            for name in self.TARGETED_SCENARIOS
        }
        command(scenario_masks["goal"], actor, Skill.SHOOT, goal_direction, 0.82)
        command(scenario_masks["own_goal"], actor, Skill.SHOOT, own_goal_direction, 0.78)
        command(scenario_masks["out_of_bounds"], actor, Skill.SHOOT, sideline_direction, 0.72)
        command(
            scenario_masks["ball_obstacle_collision"],
            actor,
            Skill.DRIBBLE,
            ball_obstacle_direction,
            0.88,
            0.20,
        )
        command(
            scenario_masks["robot_obstacle_collision"],
            actor,
            Skill.REPOSITION,
            robot_obstacle_direction,
            0.92,
        )

        if self.action_adapter.num_robots > 1:
            teammate_collision = scenario_masks["teammate_collision"]
            toward_teammate = self._unit(robot_xy[:, 1] - robot_xy[:, 0])
            robot_zero = torch.zeros_like(actor)
            robot_one = torch.ones_like(actor)
            command(teammate_collision, robot_zero, Skill.REPOSITION, toward_teammate, 0.68)
            command(teammate_collision, robot_one, Skill.REPOSITION, -toward_teammate, 0.68)

        command(scenario_masks["possession_acquired"], actor, Skill.DRIBBLE, self._unit(to_ball[rows, actor]), 0.58, -0.15)
        command(scenario_masks["possession_lost"], actor, Skill.SHOOT, goal_direction, 0.63)
        command(scenario_masks["successful_shot"], actor, Skill.SHOOT, goal_direction, 1.00)
        command(scenario_masks["failed_shot"], actor, Skill.SHOOT, -goal_direction, 0.18)
        pass_mask = scenario_masks["pass"] & (self.action_adapter.num_robots > 1)
        command(pass_mask, actor, Skill.SHOOT, pass_direction, 0.54)

        # Receivers/supporting robots actively move into the play.  The
        # scenario-specific magnitudes also make the canonical actions
        # deliberately distinguishable for auditing collected datasets.
        receiver_to_ball = self._unit(ball_xy - teammate_xy)
        command(pass_mask, teammate, Skill.REPOSITION, receiver_to_ball, 0.37)
        success_mask = scenario_masks["successful_shot"]
        command(success_mask, teammate, Skill.REPOSITION, goal_direction, 0.31)
        failed_mask = scenario_masks["failed_shot"]
        command(failed_mask, teammate, Skill.REPOSITION, -sideline_direction, 0.23)

        return self.action_adapter.pack(
            skills,
            self.action_adapter.denormalize_parameters(skills, normalized),
        )

    @staticmethod
    def _validate_scenarios(
        targeted_scenarios: Optional[Sequence[Optional[str]]],
        batch: int,
    ) -> Sequence[str]:
        if targeted_scenarios is None:
            return [""] * batch
        if len(targeted_scenarios) != batch:
            raise ValueError(
                f"targeted_scenarios must contain one entry per environment ({batch}), "
                f"got {len(targeted_scenarios)}"
            )
        scenarios = ["" if value is None else str(value) for value in targeted_scenarios]
        unknown = sorted({name for name in scenarios if name and name not in BehaviorMixture.TARGETED_SCENARIOS})
        if unknown:
            raise ValueError(f"Unknown targeted scenarios: {unknown}")
        return scenarios

    def sample(
        self,
        states: torch.Tensor,
        previous_actions: Optional[torch.Tensor] = None,
        targeted_scenarios: Optional[Sequence[Optional[str]]] = None,
        previous_action_valid: Optional[torch.Tensor] = None,
    ):
        batch = states.shape[0]
        device = states.device
        generator = self._generator_for(device)
        scenarios = list(self._validate_scenarios(targeted_scenarios, batch))
        forced_target = torch.tensor([bool(name) for name in scenarios], device=device, dtype=torch.bool)

        source = torch.multinomial(
            self.weights.to(device),
            batch,
            replacement=True,
            generator=generator,
        )
        source = torch.where(forced_target, torch.full_like(source, 3), source)

        # A non-zero targeted mixture weight is useful even without an external
        # reset manager: choose a deliberate target rather than mislabeling an
        # ordinary scripted action as targeted.
        sampled_target = (source == 3) & ~forced_target
        sampled_rows = sampled_target.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        if sampled_rows:
            choices = torch.randint(
                0,
                len(self.TARGETED_SCENARIOS),
                (len(sampled_rows),),
                generator=self.generator,
            ).tolist()
            for row, choice in zip(sampled_rows, choices):
                scenarios[row] = self.TARGETED_SCENARIOS[choice]

        scripted_actions = self._scripted(states)
        random_actions = self._random(states, scripted_actions)
        actions = torch.where((source == 1)[:, None], scripted_actions, random_actions)

        if bool((source == 3).any().item()):
            targeted_actions = self._targeted(states, scenarios)
            actions = torch.where((source == 3)[:, None], targeted_actions, actions)
        if self.existing_policy is not None and bool((source == 2).any().item()):
            policy_actions = self.existing_policy(states)
            actions = torch.where((source == 2)[:, None], policy_actions, actions)

        repeat = torch.zeros(batch, device=device, dtype=torch.bool)
        if previous_actions is not None:
            if previous_actions.shape != actions.shape:
                raise ValueError(
                    f"previous_actions must have shape {tuple(actions.shape)}, got {tuple(previous_actions.shape)}"
                )
            if previous_action_valid is None:
                valid = torch.ones(batch, device=device, dtype=torch.bool)
            else:
                valid = torch.as_tensor(previous_action_valid, device=device, dtype=torch.bool)
                if valid.shape != (batch,):
                    raise ValueError(
                        f"previous_action_valid must have shape ({batch},), got {tuple(valid.shape)}"
                    )
            repeat = (
                torch.rand(batch, device=device, generator=generator) < self.repeat_previous_probability
            ) & valid & (source != 3)
            actions = torch.where(repeat[:, None], previous_actions, actions)
        elif previous_action_valid is not None:
            raise ValueError("previous_action_valid requires previous_actions")

        names = [self.SAMPLING_SOURCE_NAMES[index] for index in source.detach().cpu().tolist()]
        for row in repeat.nonzero(as_tuple=False).flatten().detach().cpu().tolist():
            names[row] = self.REPEAT_SOURCE_NAME
        return actions, names
