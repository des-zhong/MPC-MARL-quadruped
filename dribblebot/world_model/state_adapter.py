"""Isaac Gym football state extraction isolated from model code."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from .action_adapter import JointActionAdapter
from .schema import EVENT_NAMES, StateSchema, default_state_schema, validate_event_names


def quaternion_to_roll_pitch_yaw(quaternion_xyzw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert xyzw quaternions to robust roll, pitch, yaw angles."""

    x, y, z, w = quaternion_xyzw.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    sin_pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    return roll, pitch, yaw


class FootballWorldModelStateAdapter:
    """Extract a compact global state from the multi-robot environment.

    The repository uses a fixed learning-team frame: learning slots attack +x
    and opponent slots attack -x. Every robot is part of the predicted dynamic
    state. Optional legacy static boxes remain a masked geometry set.
    """

    def __init__(
        self,
        env: Optional[Any] = None,
        max_obstacles: int = 2,
        schema: Optional[StateSchema] = None,
        event_names: Optional[Sequence[str]] = None,
        num_robots: Optional[int] = None,
    ):
        self.env = env
        self.max_obstacles = int(max_obstacles)
        raw = getattr(env, "env", env) if env is not None else None
        schema_robots = None if schema is None else len(
            [
                feature for feature in schema.features
                if feature.name.endswith(".position") and feature.name.startswith("robot_")
            ]
        )
        configured_robots = int(
            num_robots
            if num_robots is not None
            else getattr(
                raw,
                "num_robots",
                getattr(
                    getattr(getattr(raw, "cfg", object()), "env", object()),
                    "num_robots",
                    schema_robots if schema_robots is not None else 2,
                ),
            )
        )
        if configured_robots < 1:
            raise ValueError("num_robots must be at least 1")
        self.num_robots = configured_robots
        self.schema = schema or default_state_schema(max_obstacles, configured_robots)
        schema_robots = len(
            [feature for feature in self.schema.features if feature.name.endswith(".position") and feature.name.startswith("robot_")]
        )
        if schema_robots != self.num_robots:
            raise ValueError(
                f"State schema has {schema_robots} robots but environment/config requests {self.num_robots}"
            )
        schema_obstacles = sum(
            feature.name.startswith("obstacle_") for feature in self.schema.features
        )
        if schema_obstacles != self.max_obstacles:
            raise ValueError(
                f"State schema has {schema_obstacles} obstacles but max_obstacles={self.max_obstacles}"
            )
        self.event_names = validate_event_names(event_names if event_names is not None else EVENT_NAMES)
        self._event_indices = {name: index for index, name in enumerate(self.event_names)}
        self.action_adapter = JointActionAdapter.from_env(env) if env is not None else None

    @property
    def state_dim(self) -> int:
        return self.schema.state_dim

    def _raw_env(self, env: Optional[Any] = None) -> Any:
        candidate = env or self.env
        if candidate is None:
            raise ValueError("An environment is required for state extraction")
        return getattr(candidate, "env", candidate)

    def extract_state(self, env: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        raw = self._raw_env(env)
        wrapper = env or self.env
        device = raw.device
        roots = raw.root_states[raw.robot_actor_idxs_all.reshape(-1)].view(
            raw.num_envs, self.num_robots, 13
        )
        origins = raw.env_origins[:, None, :]
        half_length = max(0.5 * float(getattr(raw.cfg.env, "field_length", 8.0)), 1e-6)
        half_width = max(0.5 * float(getattr(raw.cfg.env, "field_width", 5.0)), 1e-6)
        roll, pitch, yaw = quaternion_to_roll_pitch_yaw(roots[..., 3:7])

        ball_pos = raw.object_pos_world_frame
        ball_vel = raw.object_lin_vel
        ball_ang_vel = raw.object_ang_vel
        ball_relative = ball_pos[:, None, :2] - roots[:, :, :2]
        distances = torch.linalg.vector_norm(ball_relative, dim=-1)
        contact_distance = float(getattr(raw.cfg.ball, "radius", 0.0889)) + 0.32
        ball_contact = distances <= contact_distance
        possession_distance = float(getattr(raw.cfg.rewards, "high_level_dribble_skill_distance", 1.0)) * 0.55
        nearest_distance, nearest_robot = distances.min(dim=1)
        possessed = nearest_distance <= possession_distance
        possessor = torch.zeros(raw.num_envs, self.num_robots + 1, device=device)
        possessor[:, 0] = (~possessed).float()
        possessor[torch.arange(raw.num_envs, device=device), nearest_robot + 1] = possessed.float()

        field_ball = ball_pos[:, :2] - raw.env_origins[:, :2]
        goal_x = float(getattr(raw.cfg.env, "team_goal_x", half_length))
        goal_half_width = float(getattr(raw.cfg.env, "team_goal_half_width", 1.0))
        in_opponent_goal = (field_ball[:, 0] >= goal_x) & (field_ball[:, 1].abs() <= goal_half_width)
        in_own_goal = (field_ball[:, 0] <= -half_length) & (field_ball[:, 1].abs() <= goal_half_width)
        out_of_bounds = (
            (field_ball[:, 0].abs() > half_length) | (field_ball[:, 1].abs() > half_width)
        ) & ~in_opponent_goal & ~in_own_goal

        state = torch.zeros(raw.num_envs, self.state_dim, device=device)
        skill_ids = getattr(wrapper, "skill_ids", getattr(raw, "high_level_skill_ids"))
        commands = getattr(wrapper, "skill_commands", getattr(raw, "high_level_commands"))
        if self.action_adapter is None:
            self.action_adapter = JointActionAdapter.from_env(wrapper)
        masks = self.action_adapter._selected(skill_ids, "mask", state.dtype)
        normalized_commands = self.action_adapter.normalize_parameters(skill_ids, commands)
        gait_phase = getattr(raw, "gait_indices", torch.zeros(raw.num_envs, device=device))

        structured: Dict[str, torch.Tensor] = {}
        for robot in range(self.num_robots):
            prefix = f"robot_{robot}"
            position = roots[:, robot, :3] - origins[:, 0, :]
            position = position.clone()
            position[:, 0] /= half_length
            position[:, 1] /= half_width
            structured[f"{prefix}.position"] = position
            structured[f"{prefix}.yaw_sin_cos"] = torch.stack((yaw[:, robot].sin(), yaw[:, robot].cos()), -1)
            structured[f"{prefix}.roll_pitch"] = torch.stack((roll[:, robot], pitch[:, robot]), -1)
            structured[f"{prefix}.linear_velocity"] = roots[:, robot, 7:10]
            structured[f"{prefix}.angular_velocity"] = roots[:, robot, 10:13]
            structured[f"{prefix}.fallen"] = (roots[:, robot, 2:3] < float(raw.cfg.rewards.terminal_body_height)).float()
            structured[f"{prefix}.ball_contact"] = ball_contact[:, robot : robot + 1].float()
            structured[f"{prefix}.skill_one_hot"] = torch.nn.functional.one_hot(skill_ids[:, robot], 3).float()
            structured[f"{prefix}.previous_command"] = normalized_commands[:, robot]
            structured[f"{prefix}.parameter_mask"] = masks[:, robot]
            phase_angle = 2.0 * torch.pi * gait_phase
            structured[f"{prefix}.gait_phase_sin_cos"] = torch.stack((phase_angle.sin(), phase_angle.cos()), -1)

        normalized_ball_pos = ball_pos - raw.env_origins
        normalized_ball_pos = normalized_ball_pos.clone()
        normalized_ball_pos[:, 0] /= half_length
        normalized_ball_pos[:, 1] /= half_width
        structured.update({
            "ball.position": normalized_ball_pos,
            "ball.linear_velocity": ball_vel,
            "ball.angular_velocity": ball_ang_vel,
            "ball.possessed": possessed[:, None].float(),
            "ball.possessor_one_hot": possessor,
            "ball.in_opponent_goal": in_opponent_goal[:, None].float(),
            "ball.in_own_goal": in_own_goal[:, None].float(),
            "ball.out_of_bounds": out_of_bounds[:, None].float(),
        })

        count = int(getattr(raw, "num_static_opponents", 0))
        obstacle_size = tuple(float(v) for v in getattr(raw, "static_opponent_size", (0.0, 0.0, 0.0)))
        if count:
            obstacle_roots = raw.root_states[raw.static_opponent_actor_idxs.reshape(-1)].view(raw.num_envs, count, 13)
        for obstacle in range(self.max_obstacles):
            geometry = torch.zeros(raw.num_envs, 6, device=device)
            if obstacle < count:
                relative = obstacle_roots[:, obstacle, :2] - raw.env_origins[:, :2]
                geometry[:, 0] = relative[:, 0] / half_length
                geometry[:, 1] = relative[:, 1] / half_width
                geometry[:, 2] = 0.5 * obstacle_size[0]
                geometry[:, 3] = 0.5 * obstacle_size[1]
                geometry[:, 4] = obstacle_size[2]
                geometry[:, 5] = 1.0
            structured[f"obstacle_{obstacle}.geometry"] = geometry
        structured["field.geometry"] = torch.tensor(
            [half_length, half_width, -half_length, goal_x, goal_half_width, float(getattr(raw.cfg.ball, "radius", 0.0889))],
            device=device,
        ).expand(raw.num_envs, -1)
        structured["tensor"] = self.encode_state(structured)
        return structured

    def encode_state(self, state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        reference = next(value for key, value in state_dict.items() if key != "tensor")
        result = reference.new_zeros(reference.shape[0], self.state_dim)
        for feature in self.schema.features:
            if feature.name not in state_dict:
                raise KeyError(f"Missing required state field {feature.name}")
            value = state_dict[feature.name]
            if value.shape != (reference.shape[0], feature.size):
                raise ValueError(f"Field {feature.name} expected shape [batch,{feature.size}], got {value.shape}")
            result[:, feature.start : feature.stop] = value
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError("Extracted state contains NaN or infinite values")
        return result

    def decode_dynamic_state(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if tensor.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {tensor.shape[-1]}")
        return {f.name: tensor[..., f.start : f.stop] for f in self.schema.features if f.dynamic}

    def apply_predicted_delta(
        self,
        current_state: torch.Tensor,
        predicted_delta: torch.Tensor,
        binary_probabilities: Optional[torch.Tensor] = None,
        joint_action: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        dynamic = self.schema.continuous_dynamic_indices
        if predicted_delta.shape[-1] != len(dynamic):
            raise ValueError(f"Expected {len(dynamic)} delta features, got {predicted_delta.shape[-1]}")
        result = current_state.clone()
        result[..., dynamic] = result[..., dynamic] + predicted_delta
        if binary_probabilities is not None:
            binary = self.schema.binary_dynamic_indices
            values = (binary_probabilities >= 0.5).to(result.dtype) if deterministic else torch.bernoulli(binary_probabilities)
            result[..., binary] = values
        for sin_index, cos_index in self.schema.yaw_pairs:
            pair = result[..., [sin_index, cos_index]]
            pair = pair / torch.linalg.vector_norm(pair, dim=-1, keepdim=True).clamp(min=1e-6)
            result[..., sin_index] = pair[..., 0]
            result[..., cos_index] = pair[..., 1]
        if joint_action is not None:
            if self.action_adapter is None:
                raise RuntimeError("Action adapter is required to apply controlled skill state")
            skills, params = self.action_adapter.unpack(joint_action)
            normalized = self.action_adapter.normalize_parameters(skills, params)
            masks = self.action_adapter._selected(skills, "mask", result.dtype)
            for robot in range(self.num_robots):
                result[..., self.schema.slice(f"robot_{robot}.skill_one_hot")] = torch.nn.functional.one_hot(skills[..., robot], 3).to(result.dtype)
                result[..., self.schema.slice(f"robot_{robot}.previous_command")] = normalized[..., robot, :]
                result[..., self.schema.slice(f"robot_{robot}.parameter_mask")] = masks[..., robot, :]
        return result

    def extract_event_labels(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        info: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        labels = state.new_zeros(state.shape[0], len(self.event_names))
        info = info or {}

        def assign(name: str, value: Any) -> None:
            index = self._event_indices.get(name)
            if index is not None:
                labels[:, index] = torch.as_tensor(value, device=state.device, dtype=state.dtype).reshape(-1)

        goal = torch.as_tensor(
            info.get("high_level_goal", next_state[:, self.schema.slice("ball.in_opponent_goal")]),
            device=state.device,
        ).reshape(-1).bool()
        own_goal = next_state[:, self.schema.slice("ball.in_own_goal")].reshape(-1).bool()
        out_of_bounds = torch.as_tensor(
            info.get("high_level_ball_off_border", next_state[:, self.schema.slice("ball.out_of_bounds")]),
            device=state.device,
        ).reshape(-1).bool()
        # The simulator reports crossing the negative goal line as a border
        # event.  Keep the mutually exclusive dataset semantics encoded in the
        # state schema instead of double-labeling an own goal as out-of-bounds.
        out_of_bounds &= ~goal & ~own_goal
        assign("goal", goal)
        assign("own_goal", own_goal)
        assign("out_of_bounds", out_of_bounds)
        assign("ball_obstacle_collision", info.get("high_level_obstacle_contact", torch.zeros(state.shape[0], device=state.device)))
        old_possession = state[:, self.schema.slice("ball.possessed")].squeeze(-1) > 0.5
        new_possession = next_state[:, self.schema.slice("ball.possessed")].squeeze(-1) > 0.5
        assign("possession_acquired", ~old_possession & new_possession)
        assign("possession_lost", old_possession & ~new_possession)
        old_possessor = state[:, self.schema.slice("ball.possessor_one_hot")] > 0.5
        new_possessor = next_state[:, self.schema.slice("ball.possessor_one_hot")] > 0.5
        old_robot = old_possessor[:, 1:]
        new_robot = new_possessor[:, 1:]
        old_valid = ~old_possessor[:, 0] & (old_robot.sum(dim=-1) == 1)
        new_valid = ~new_possessor[:, 0] & (new_robot.sum(dim=-1) == 1)
        changed_robot = (old_robot != new_robot).any(dim=-1)
        pass_event = old_valid & new_valid & changed_robot
        if "high_level_skill_ids" in info:
            executed_skills = torch.as_tensor(info["high_level_skill_ids"], device=state.device).long()
            old_robot_index = old_robot.long().argmax(dim=-1)
            passer_shot = executed_skills.gather(1, old_robot_index[:, None]).squeeze(1) == 2
            pass_event &= passer_shot
        assign("pass", pass_event)
        field = next_state[:, self.schema.slice("field.geometry")]
        robot_positions = torch.stack(
            [
                next_state[:, self.schema.slice(f"robot_{i}.position")][:, :2]
                for i in range(self.num_robots)
            ],
            1,
        )
        robot_obstacle = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
        for obstacle in range(self.max_obstacles):
            geometry = next_state[:, self.schema.slice(f"obstacle_{obstacle}.geometry")]
            half_extent_normalized = torch.stack((geometry[:, 2] / field[:, 0], geometry[:, 3] / field[:, 1]), -1)
            robot_radius_normalized = torch.stack((0.25 / field[:, 0], 0.25 / field[:, 1]), -1)
            overlap = (torch.abs(robot_positions - geometry[:, None, :2]) <= (half_extent_normalized + robot_radius_normalized)[:, None]).all(-1)
            robot_obstacle |= overlap.any(-1) & (geometry[:, 5] > 0.5)
        assign("robot_obstacle_collision", robot_obstacle)
        if self.num_robots > 1:
            pairwise_delta = (
                robot_positions[:, :, None, :] - robot_positions[:, None, :, :]
            ) * field[:, None, None, :2]
            pairwise_distance = torch.linalg.vector_norm(pairwise_delta, dim=-1)
            diagonal = torch.eye(self.num_robots, device=state.device, dtype=torch.bool)
            pairwise_distance = pairwise_distance.masked_fill(diagonal[None], float("inf"))
            teammate_collision = (pairwise_distance < 0.5).any(dim=(-1, -2))
        else:
            teammate_collision = torch.zeros(state.shape[0], dtype=torch.bool, device=state.device)
        assign("teammate_collision", teammate_collision)
        if "shooting_success" in info:
            assign("successful_shot", info["shooting_success"])
        elif "high_level_skill_ids" in info:
            selected_shoot = torch.as_tensor(info["high_level_skill_ids"], device=state.device).eq(2).any(-1)
            ball_velocity = next_state[:, self.schema.slice("ball.linear_velocity")]
            # Use the pre-kick ball position.  After a successful goal the
            # terminal ball is already beyond the goal line, where a vector
            # back to the goal plane would incorrectly reverse alignment.
            ball_xy = state[:, self.schema.slice("ball.position")][:, :2] * field[:, :2]
            goal_xy = torch.stack((field[:, 3], torch.zeros_like(field[:, 3])), dim=-1)
            goal_direction = goal_xy - ball_xy
            goal_direction /= torch.linalg.vector_norm(goal_direction, dim=-1, keepdim=True).clamp(min=1e-6)
            planar_velocity = ball_velocity[:, :2]
            speed = torch.linalg.vector_norm(planar_velocity, dim=-1)
            goal_alignment = torch.sum(planar_velocity * goal_direction, dim=-1) / speed.clamp(min=1e-6)
            successful = selected_shoot & ~pass_event & (speed > 1.0) & (goal_alignment > 0.7)
            assign("successful_shot", successful)
            assign("failed_shot", selected_shoot & ~pass_event & ~successful)
        if "shooting_failure" in info:
            assign("failed_shot", info["shooting_failure"])
        return labels
