"""World-model adapter consuming the manager-based canonical match snapshot."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

try:
    from quadruped.world_model.schema import EVENT_NAMES, StateSchema, default_state_schema, validate_event_names
except ModuleNotFoundError:
    # The migration package is installed as a separate Isaac Lab extension,
    # while the existing model code remains at the repository root. Resolve
    # that sibling package locally without importing any simulator module.
    repository_root = Path(__file__).resolve().parents[4]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from quadruped.world_model.schema import EVENT_NAMES, StateSchema, default_state_schema, validate_event_names


def _wxyz_to_rpy(quaternion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qw, qx, qy, qz = quaternion.unbind(-1)
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx.square() + qy.square()))
    pitch = torch.asin((2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy.square() + qz.square()))
    return roll, pitch, yaw


class IsaacLabFootballWorldModelStateAdapter:
    """Encode `MatchSnapshotRecorder` output into the existing world-model schema."""

    def __init__(
        self,
        *,
        num_robots: int = 4,
        max_obstacles: int = 0,
        field_length: float = 8.0,
        field_width: float = 5.0,
        goal_x: float = 4.0,
        goal_half_width: float = 1.0,
        ball_radius: float = 0.0889,
        event_names: Sequence[str] | None = None,
        command_scales: tuple[tuple[float, float, float], ...] = (
            (1.5, 1.5, 1.0),
            (1.5, 1.5, 1.0),
            (3.0, 3.0, 0.0),
        ),
        schema: StateSchema | None = None,
    ):
        if num_robots < 1 or max_obstacles < 0:
            raise ValueError("num_robots must be positive and max_obstacles non-negative")
        self.num_robots = int(num_robots)
        self.max_obstacles = int(max_obstacles)
        self.field_length = float(field_length)
        self.field_width = float(field_width)
        self.goal_x = float(goal_x)
        self.goal_half_width = float(goal_half_width)
        self.ball_radius = float(ball_radius)
        self.command_scales = torch.tensor(command_scales, dtype=torch.float)
        if self.command_scales.shape != (3, 3):
            raise ValueError("command_scales must have shape (3, 3)")
        self.schema = schema or default_state_schema(self.max_obstacles, self.num_robots)
        self.event_names = validate_event_names(event_names if event_names is not None else EVENT_NAMES)
        self._event_indices = {name: index for index, name in enumerate(self.event_names)}
        if len([f for f in self.schema.features if f.name.startswith("robot_") and f.name.endswith(".position")]) != self.num_robots:
            raise ValueError("schema robot count does not match adapter robot count")

    @property
    def state_dim(self) -> int:
        return self.schema.state_dim

    def extract_state(self, snapshot: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {
            "env_origin_w",
            "robot_root_state_w",
            "robot_joint_pos",
            "robot_joint_vel",
            "ball_root_state_w",
            "skill_id",
            "skill_command",
            "gait_phase",
        }
        missing = sorted(required.difference(snapshot))
        if missing:
            raise KeyError(f"canonical snapshot is missing keys: {missing}")
        roots = snapshot["robot_root_state_w"]
        if roots.ndim != 3 or roots.shape[1:] != (self.num_robots, 13):
            raise ValueError(f"robot_root_state_w must have shape (N,{self.num_robots},13), got {tuple(roots.shape)}")
        device = roots.device
        batch = roots.shape[0]
        origins = snapshot["env_origin_w"]
        ball = snapshot["ball_root_state_w"]
        skill_ids = snapshot["skill_id"].long()
        commands = snapshot["skill_command"]
        gait_phase = snapshot["gait_phase"]
        if skill_ids.shape != (batch, self.num_robots) or commands.shape != (batch, self.num_robots, 3):
            raise ValueError("skill_id/skill_command shapes do not match robot count")
        field_half = roots.new_tensor((max(self.field_length * 0.5, 1.0e-6), max(self.field_width * 0.5, 1.0e-6)))
        field_diag = torch.linalg.vector_norm(field_half).clamp_min(1.0e-6)
        roll, pitch, yaw = _wxyz_to_rpy(roots[..., 3:7])
        robot_xy = roots[..., :2]
        ball_xy = ball[:, :2]
        relative_ball = ball_xy[:, None, :] - robot_xy
        distance = torch.linalg.vector_norm(relative_ball, dim=-1)
        possessed = distance.amin(dim=1) <= 0.55
        possessor = roots.new_zeros((batch, self.num_robots + 1))
        nearest = torch.argmin(distance, dim=1)
        possessor[:, 0] = (~possessed).float()
        possessor[torch.arange(batch, device=device), nearest + 1] = possessed.float()
        field_ball = ball_xy - origins[:, :2]
        in_opponent_goal = (field_ball[:, 0] >= self.goal_x) & (torch.abs(field_ball[:, 1]) <= self.goal_half_width)
        in_own_goal = (field_ball[:, 0] <= -self.goal_x) & (torch.abs(field_ball[:, 1]) <= self.goal_half_width)
        out_of_bounds = (
            (torch.abs(field_ball[:, 0]) > self.field_length * 0.5)
            | (torch.abs(field_ball[:, 1]) > self.field_width * 0.5)
        ) & ~in_opponent_goal & ~in_own_goal
        scales = self.command_scales.to(device=device, dtype=commands.dtype)[skill_ids]
        normalized_commands = commands / scales.abs().clamp_min(1.0e-6)
        masks = (scales.abs() > 1.0e-8).to(commands.dtype)
        structured: dict[str, torch.Tensor] = {}
        for robot_index in range(self.num_robots):
            prefix = f"robot_{robot_index}"
            position = roots[:, robot_index, :3] - origins
            position = position.clone()
            position[:, 0] /= field_half[0]
            position[:, 1] /= field_half[1]
            phase = 2.0 * torch.pi * gait_phase[:, robot_index]
            structured[f"{prefix}.position"] = position
            structured[f"{prefix}.yaw_sin_cos"] = torch.stack((yaw[:, robot_index].sin(), yaw[:, robot_index].cos()), dim=-1)
            structured[f"{prefix}.roll_pitch"] = torch.stack((roll[:, robot_index], pitch[:, robot_index]), dim=-1)
            structured[f"{prefix}.linear_velocity"] = roots[:, robot_index, 7:10]
            structured[f"{prefix}.angular_velocity"] = roots[:, robot_index, 10:13]
            structured[f"{prefix}.fallen"] = (roots[:, robot_index, 2:3] < 0.20).to(commands.dtype)
            structured[f"{prefix}.ball_contact"] = (distance[:, robot_index : robot_index + 1] <= self.ball_radius + 0.32).to(commands.dtype)
            structured[f"{prefix}.skill_one_hot"] = torch.nn.functional.one_hot(skill_ids[:, robot_index], 3).to(commands.dtype)
            structured[f"{prefix}.previous_command"] = normalized_commands[:, robot_index]
            structured[f"{prefix}.parameter_mask"] = masks[:, robot_index]
            structured[f"{prefix}.gait_phase_sin_cos"] = torch.stack((phase.sin(), phase.cos()), dim=-1)
        ball_position = torch.cat((ball[:, :3] - origins, ball[:, 3:]), dim=-1)
        ball_position[:, 0] /= field_half[0]
        ball_position[:, 1] /= field_half[1]
        structured.update(
            {
                "ball.position": ball_position[:, :3],
                "ball.linear_velocity": ball[:, 7:10],
                "ball.angular_velocity": ball[:, 10:13],
                "ball.possessed": possessed[:, None].to(commands.dtype),
                "ball.possessor_one_hot": possessor,
                "ball.in_opponent_goal": in_opponent_goal[:, None].to(commands.dtype),
                "ball.in_own_goal": in_own_goal[:, None].to(commands.dtype),
                "ball.out_of_bounds": out_of_bounds[:, None].to(commands.dtype),
                "field.geometry": roots.new_tensor(
                    (self.field_length * 0.5, self.field_width * 0.5, -self.goal_x, self.goal_x, self.goal_half_width, self.ball_radius)
                ).expand(batch, -1),
            }
        )
        for obstacle in range(self.max_obstacles):
            structured[f"obstacle_{obstacle}.geometry"] = roots.new_zeros((batch, 6))
        structured["tensor"] = self.encode_state(structured)
        return structured

    def encode_state(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        reference = next(value for name, value in fields.items() if name != "tensor")
        result = reference.new_zeros((reference.shape[0], self.state_dim))
        for feature in self.schema.features:
            if feature.name not in fields:
                raise KeyError(f"Missing canonical field {feature.name}")
            value = fields[feature.name]
            if value.shape != (reference.shape[0], feature.size):
                raise ValueError(f"Field {feature.name} expected shape (N,{feature.size}), got {tuple(value.shape)}")
            result[:, feature.start : feature.stop] = value
        if not torch.isfinite(result).all():
            raise ValueError("Encoded world-model state contains non-finite values")
        return result

    @staticmethod
    def stack_recorder_snapshot(
        episode_data: Mapping[str, Any] | Any,
        path: str = "football/snapshot",
    ) -> dict[str, torch.Tensor]:
        """Stack RecorderManager ``EpisodeData`` snapshots into time-major tensors.

        Isaac Lab stores recorder values as a list of tensors under
        ``EpisodeData.data``.  This adapter accepts either the EpisodeData
        object itself or its nested mapping, and intentionally leaves the
        recorder's canonical field names unchanged.
        """

        data = getattr(episode_data, "data", episode_data)
        for component in path.split("/"):
            if not isinstance(data, Mapping) or component not in data:
                raise KeyError(f"Recorder episode is missing snapshot path {path!r}")
            data = data[component]
        if not isinstance(data, Mapping) or not data:
            raise ValueError("Recorder snapshot payload must be a non-empty mapping")
        stacked: dict[str, torch.Tensor] = {}
        for name, values in data.items():
            if isinstance(values, (list, tuple)):
                if not values:
                    raise ValueError(f"Recorder field {name!r} has no samples")
                tensor = torch.stack([torch.as_tensor(value) for value in values], dim=0)
            else:
                tensor = torch.as_tensor(values)
                if tensor.ndim == 0:
                    raise ValueError(f"Recorder field {name!r} must have a time dimension")
            # ``get_episode(env_id)`` normally strips the vector-env axis, but
            # older Isaac Lab releases may leave a singleton axis in place.
            # Use field-specific ranks so a legitimate one-robot axis is never
            # mistaken for the vector-env axis.
            env_axis_rank = {
                "env_origin_w": 3,
                "robot_root_state_w": 4,
                "robot_joint_pos": 4,
                "robot_joint_vel": 4,
                "robot_joint_target": 4,
                "ball_root_state_w": 3,
                "skill_id": 3,
                "requested_skill_id": 3,
                "invalid_skill": 3,
                "skill_command": 4,
                "gait_phase": 3,
                "episode_length": 2,
            }.get(name)
            if env_axis_rank == tensor.ndim and tensor.shape[1] == 1:
                tensor = tensor.squeeze(1)
            stacked[name] = tensor
        lengths = {int(value.shape[0]) for value in stacked.values()}
        if len(lengths) != 1:
            raise ValueError(f"Recorder snapshot fields have inconsistent lengths: {sorted(lengths)}")
        if next(iter(lengths)) < 2:
            raise ValueError("At least two snapshots are required to form a transition")
        return stacked

    def extract_episode_transitions(
        self,
        episode_data: Mapping[str, Any] | Any,
        *,
        info: Mapping[str, Any] | None = None,
        path: str = "football/snapshot",
    ) -> dict[str, torch.Tensor]:
        """Convert a recorder episode into world-model transition tensors."""

        snapshots = self.stack_recorder_snapshot(episode_data, path=path)
        state_snapshot = {name: value[:-1] for name, value in snapshots.items()}
        next_snapshot = {name: value[1:] for name, value in snapshots.items()}
        state = self.extract_state(state_snapshot)["tensor"]
        next_state = self.extract_state(next_snapshot)["tensor"]
        events = self.extract_event_labels(state, next_state, info=info)
        return {
            "state": state,
            "next_state": next_state,
            "event_labels": events,
            "snapshot": snapshots,
        }

    def extract_event_labels(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        info: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Derive the legacy event-label vector from encoded state transitions."""

        if state.shape != next_state.shape or state.shape[-1] != self.state_dim:
            raise ValueError("state and next_state must have the same encoded shape")
        labels = state.new_zeros((state.shape[0], len(self.event_names)))
        info = info or {}

        def assign(name: str, value: Any) -> None:
            index = self._event_indices.get(name)
            if index is None:
                return
            labels[:, index] = torch.as_tensor(value, device=state.device, dtype=state.dtype).reshape(-1)

        def state_bool(tensor: torch.Tensor, feature: str) -> torch.Tensor:
            return tensor[:, self.schema.slice(feature)].reshape(-1) > 0.5

        goal = torch.as_tensor(
            info.get("goal", info.get("match_goal", state_bool(next_state, "ball.in_opponent_goal"))),
            device=state.device,
        ).reshape(-1).bool()
        own_goal = state_bool(next_state, "ball.in_own_goal")
        out_of_bounds = torch.as_tensor(
            info.get("out_of_bounds", info.get("match_ball_out_of_bounds", state_bool(next_state, "ball.out_of_bounds"))),
            device=state.device,
        ).reshape(-1).bool()
        out_of_bounds &= ~goal & ~own_goal
        assign("goal", goal)
        assign("own_goal", own_goal)
        assign("out_of_bounds", out_of_bounds)
        assign("ball_obstacle_collision", info.get("ball_obstacle_collision", torch.zeros_like(goal)))
        assign("robot_obstacle_collision", info.get("robot_obstacle_collision", torch.zeros_like(goal)))

        old_possession = state_bool(state, "ball.possessed")
        new_possession = state_bool(next_state, "ball.possessed")
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
        old_skills = torch.stack(
            [torch.argmax(state[:, self.schema.slice(f"robot_{i}.skill_one_hot")], dim=-1) for i in range(self.num_robots)],
            dim=1,
        )
        old_possessor_index = old_robot.long().argmax(dim=-1)
        pass_event &= old_skills.gather(1, old_possessor_index[:, None]).squeeze(1).eq(2)
        assign("pass", pass_event)

        selected_shoot = old_skills.eq(2).any(dim=-1)
        field = next_state[:, self.schema.slice("field.geometry")]
        ball_xy = state[:, self.schema.slice("ball.position")][:, :2] * field[:, :2]
        goal_xy = torch.stack((field[:, 3], torch.zeros_like(field[:, 3])), dim=-1)
        goal_direction = goal_xy - ball_xy
        goal_direction = goal_direction / torch.linalg.vector_norm(goal_direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
        ball_velocity = next_state[:, self.schema.slice("ball.linear_velocity")][:, :2]
        speed = torch.linalg.vector_norm(ball_velocity, dim=-1)
        alignment = torch.sum(ball_velocity * goal_direction, dim=-1) / speed.clamp_min(1.0e-6)
        successful_shot = selected_shoot & ~pass_event & (speed > 1.0) & (alignment > 0.7)
        assign("successful_shot", info.get("successful_shot", successful_shot))
        assign("failed_shot", selected_shoot & ~successful_shot)

        positions = torch.stack(
            [next_state[:, self.schema.slice(f"robot_{i}.position")][:, :2] for i in range(self.num_robots)],
            dim=1,
        )
        field_xy = field[:, :2].clamp_min(1.0e-6)
        pairwise = (positions[:, :, None] - positions[:, None, :]) * field_xy[:, None, None]
        distances = torch.linalg.vector_norm(pairwise, dim=-1)
        distances = distances.masked_fill(torch.eye(self.num_robots, device=state.device, dtype=torch.bool)[None], float("inf"))
        assign("teammate_collision", (distances < 0.5).any(dim=(-1, -2)))
        return labels


__all__ = ["IsaacLabFootballWorldModelStateAdapter"]
