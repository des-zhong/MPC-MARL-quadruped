"""Schema-driven decentralized observations for future student actors."""

from __future__ import annotations

from typing import Dict, List

import torch


class LocalObservationAdapter:
    """Derive one robot-centric high-level observation per controlled robot.

    The simulator wrapper exposes the same centralized vector as
    both ``obs`` and ``privileged_obs``.  Teacher data therefore derives local
    views from the auditable world-model state schema.  Each view contains the
    robot's own state, teammate-relative state, robot-frame ball state,
    robot-frame static-obstacle geometry, ball/event flags, and field geometry.
    No anonymous state indices are used.
    """

    def __init__(self, schema):
        self.schema = schema
        self.num_robots = len(
            [
                feature
                for feature in schema.features
                if feature.name.startswith("robot_")
                and feature.name.endswith(".position")
            ]
        )
        if self.num_robots < 1:
            raise ValueError("State schema must contain at least one robot")
        self.feature_names = self._feature_names()

    def _feature_names(self) -> List[str]:
        names = [
            "own.position",
            "own.yaw_sin_cos",
            "own.roll_pitch",
            "own.linear_velocity",
            "own.angular_velocity",
            "own.fallen",
            "own.ball_contact",
            "own.skill_one_hot",
            "own.previous_command",
            "own.parameter_mask",
            "own.gait_phase_sin_cos",
        ]
        for teammate_slot in range(self.num_robots - 1):
            names.extend(
                (
                    f"teammate_{teammate_slot}.relative_position",
                    f"teammate_{teammate_slot}.relative_yaw_sin_cos",
                    f"teammate_{teammate_slot}.relative_linear_velocity",
                    f"teammate_{teammate_slot}.fallen",
                    f"teammate_{teammate_slot}.ball_contact",
                )
            )
        names.extend([
            "ball.relative_position",
            "ball.relative_linear_velocity",
            "ball.angular_velocity",
            "ball.flags",
        ])
        names.extend(
            f"{feature.name}.robot_relative"
            for feature in self.schema.features
            if feature.group == "obstacle"
        )
        names.append("field.geometry")
        return names

    @staticmethod
    def _world_to_local(xy: torch.Tensor, yaw_sin_cos: torch.Tensor) -> torch.Tensor:
        sin_yaw = yaw_sin_cos[..., 0]
        cos_yaw = yaw_sin_cos[..., 1]
        return torch.stack(
            (
                cos_yaw * xy[..., 0] + sin_yaw * xy[..., 1],
                -sin_yaw * xy[..., 0] + cos_yaw * xy[..., 1],
            ),
            dim=-1,
        )

    def _slice(self, states: torch.Tensor, name: str) -> torch.Tensor:
        return states[..., self.schema.slice(name)]

    def for_robot(self, states: torch.Tensor, robot: int) -> torch.Tensor:
        if states.shape[-1] != self.schema.state_dim:
            raise ValueError(f"Expected state dimension {self.schema.state_dim}, got {states.shape[-1]}")
        if robot < 0 or robot >= self.num_robots:
            raise ValueError(f"robot must be in [0,{self.num_robots - 1}]")
        own = f"robot_{robot}"
        field = self._slice(states, "field.geometry")
        scale = field[..., :2].abs().clamp(min=1e-6)
        own_position = self._slice(states, f"{own}.position")
        own_yaw = self._slice(states, f"{own}.yaw_sin_cos")

        ball_position = self._slice(states, "ball.position")
        ball_relative_world = (ball_position[..., :2] - own_position[..., :2]) * scale
        ball_relative = torch.cat(
            (
                self._world_to_local(ball_relative_world, own_yaw),
                ball_position[..., 2:3] - own_position[..., 2:3],
            ),
            dim=-1,
        )
        ball_velocity = self._slice(states, "ball.linear_velocity") - self._slice(
            states, f"{own}.linear_velocity"
        )
        ball_velocity = torch.cat(
            (self._world_to_local(ball_velocity[..., :2], own_yaw), ball_velocity[..., 2:3]),
            dim=-1,
        )

        pieces = [
            own_position,
            own_yaw,
            self._slice(states, f"{own}.roll_pitch"),
            self._slice(states, f"{own}.linear_velocity"),
            self._slice(states, f"{own}.angular_velocity"),
            self._slice(states, f"{own}.fallen"),
            self._slice(states, f"{own}.ball_contact"),
            self._slice(states, f"{own}.skill_one_hot"),
            self._slice(states, f"{own}.previous_command"),
            self._slice(states, f"{own}.parameter_mask"),
            self._slice(states, f"{own}.gait_phase_sin_cos"),
        ]
        sin_a, cos_a = own_yaw[..., 0], own_yaw[..., 1]
        for teammate in range(self.num_robots):
            if teammate == robot:
                continue
            other = f"robot_{teammate}"
            other_position = self._slice(states, f"{other}.position")
            other_yaw = self._slice(states, f"{other}.yaw_sin_cos")
            relative_position_world = (
                other_position[..., :2] - own_position[..., :2]
            ) * scale
            relative_position = torch.cat(
                (
                    self._world_to_local(relative_position_world, own_yaw),
                    other_position[..., 2:3] - own_position[..., 2:3],
                ),
                dim=-1,
            )
            sin_b, cos_b = other_yaw[..., 0], other_yaw[..., 1]
            relative_yaw = torch.stack(
                (
                    sin_b * cos_a - cos_b * sin_a,
                    cos_b * cos_a + sin_b * sin_a,
                ),
                -1,
            )
            relative_velocity = self._slice(
                states, f"{other}.linear_velocity"
            ) - self._slice(states, f"{own}.linear_velocity")
            relative_velocity = torch.cat(
                (
                    self._world_to_local(relative_velocity[..., :2], own_yaw),
                    relative_velocity[..., 2:3],
                ),
                dim=-1,
            )
            pieces.extend(
                (
                    relative_position,
                    relative_yaw,
                    relative_velocity,
                    self._slice(states, f"{other}.fallen"),
                    self._slice(states, f"{other}.ball_contact"),
                )
            )
        pieces.extend([
            ball_relative,
            ball_velocity,
            self._slice(states, "ball.angular_velocity"),
            torch.cat(
                (
                    self._slice(states, "ball.possessed"),
                    self._slice(states, "ball.possessor_one_hot"),
                    self._slice(states, "ball.in_opponent_goal"),
                    self._slice(states, "ball.in_own_goal"),
                    self._slice(states, "ball.out_of_bounds"),
                ),
                dim=-1,
            ),
        ])
        for feature in self.schema.features:
            if feature.group != "obstacle":
                continue
            geometry = self._slice(states, feature.name)
            relative_world = (geometry[..., :2] - own_position[..., :2]) * scale
            pieces.append(
                torch.cat(
                    (
                        self._world_to_local(relative_world, own_yaw),
                        geometry[..., 2:],
                    ),
                    dim=-1,
                )
            )
        pieces.append(field)
        result = torch.cat(pieces, dim=-1)
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError("Local observation contains NaN or infinite values")
        return result

    def extract(self, states: torch.Tensor) -> torch.Tensor:
        """Return local observations with shape ``[..., N, local_obs_dim]``."""

        return torch.stack(
            [self.for_robot(states, robot) for robot in range(self.num_robots)],
            dim=-2,
        )

    def metadata(self) -> Dict[str, object]:
        dummy = torch.zeros(1, self.schema.state_dim)
        return {
            "format": "dribblebot_schema_derived_local_observation_v1",
            "num_robots": self.num_robots,
            "local_observation_dim": int(self.extract(dummy).shape[-1]),
            "semantic_groups": list(self.feature_names),
            "source_state_schema_version": self.schema.version,
            "note": (
                "Derived from centralized world-model state because the current "
                "high-level wrapper exposes no per-robot local observation API."
            ),
        }
