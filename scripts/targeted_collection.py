"""Reset-time scenario staging for rare football event collection.

The regular simulator reset remains the source of robot joint state and domain
randomization.  This module only moves selected actors after that reset so a
rare outcome can occur within the next high-level action interval.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Optional, Sequence

import numpy as np
import torch


SUPPORTED_SCENARIOS = (
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


def configured_minimum_counts(config: Mapping[str, object]) -> dict[str, int]:
    """Return validated non-negative event coverage requirements."""

    result = {}
    for name, value in dict(config.get("minimum_event_counts", {})).items():
        count = int(value)
        if count < 0:
            raise ValueError(f"minimum_event_counts[{name!r}] must be non-negative")
        result[str(name)] = count
    return result


def coverage_deficits(
    event_counts: Mapping[str, int],
    minimum_counts: Mapping[str, int],
) -> dict[str, int]:
    return {
        name: max(int(required) - int(event_counts.get(name, 0)), 0)
        for name, required in minimum_counts.items()
        if int(required) > int(event_counts.get(name, 0))
    }


class TargetedScenarioManager:
    """Assign and stage persistent, per-environment rare-event scenarios."""

    def __init__(self, wrapper, config: Optional[Mapping[str, object]] = None, seed: int = 42):
        self.wrapper = wrapper
        self.raw = wrapper.env
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.episode_probability = float(self.config.get("episode_probability", 0.5))
        if not 0.0 <= self.episode_probability <= 1.0:
            raise ValueError("targeted_rare_events.episode_probability must be in [0, 1]")
        self.max_macro_steps = max(int(self.config.get("max_macro_steps", 3)), 1)
        self.ball_speed = max(float(self.config.get("ball_speed", 3.0)), 0.5)
        configured = tuple(self.config.get("scenarios", SUPPORTED_SCENARIOS))
        unknown = sorted(set(configured) - set(SUPPORTED_SCENARIOS))
        if unknown:
            raise ValueError(f"Unsupported targeted scenarios: {unknown}")
        if int(getattr(self.raw, "num_static_opponents", 0)) <= 0:
            configured = tuple(name for name in configured if "obstacle_collision" not in name)
        if int(getattr(self.raw, "num_robots", 2)) < 2:
            configured = tuple(name for name in configured if name not in ("teammate_collision", "pass"))
        self.scenario_names = configured
        self.rng = np.random.default_rng(seed)
        self.active = np.full(self.raw.num_envs, "", dtype="U32")
        self.age = np.zeros(self.raw.num_envs, dtype=np.int16)
        self.staged_counts = Counter()

    def policy_scenarios(self) -> list[str]:
        return self.active.tolist()

    def _choose_scenario(
        self,
        event_counts: Mapping[str, int],
        minimum_counts: Mapping[str, int],
    ) -> str:
        deficits = coverage_deficits(event_counts, minimum_counts)
        candidates = [name for name in self.scenario_names if name in deficits]
        if candidates:
            weights = np.asarray([deficits[name] for name in candidates], dtype=np.float64)
            weights /= weights.sum()
            return str(self.rng.choice(candidates, p=weights))
        return str(self.rng.choice(self.scenario_names))

    def stage(
        self,
        env_ids: torch.Tensor,
        event_counts: Optional[Mapping[str, int]] = None,
        minimum_counts: Optional[Mapping[str, int]] = None,
        force: bool = False,
    ) -> Counter:
        """Stage a subset of newly reset environments and return scenario counts."""

        event_counts = event_counts or {}
        minimum_counts = minimum_counts or {}
        ids = torch.as_tensor(env_ids, device=self.raw.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return Counter()
        ids_cpu = ids.detach().cpu().tolist()
        self.active[ids_cpu] = ""
        self.age[ids_cpu] = 0
        if not self.enabled or not self.scenario_names:
            return Counter()

        selected = []
        scenarios = []
        for env_id in ids_cpu:
            if force or self.rng.random() < self.episode_probability:
                selected.append(env_id)
                scenarios.append(self._choose_scenario(event_counts, minimum_counts))
        if not selected:
            return Counter()

        changed_actor_ids = []
        for env_id, scenario in zip(selected, scenarios):
            changed_actor_ids.extend(self._stage_one(env_id, scenario))
            self.active[env_id] = scenario
            self.age[env_id] = 0
            self.staged_counts[scenario] += 1

        self._commit(selected, changed_actor_ids)
        return Counter(scenarios)

    def observe(self, event_labels: torch.Tensor, event_names: Sequence[str]) -> None:
        """Clear a scenario once its event occurs or its short attempt expires."""

        if not self.enabled:
            return
        labels = event_labels.detach().cpu().numpy() > 0.5
        event_index = {name: index for index, name in enumerate(event_names)}
        active_ids = np.flatnonzero(self.active != "")
        self.age[active_ids] += 1
        for env_id in active_ids:
            scenario = self.active[env_id]
            occurred = scenario in event_index and bool(labels[env_id, event_index[scenario]])
            if occurred or self.age[env_id] >= self.max_macro_steps:
                self.active[env_id] = ""
                self.age[env_id] = 0

    def clear(self, env_ids: torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids).detach().cpu().long().flatten().tolist()
        self.active[ids] = ""
        self.age[ids] = 0

    def _set_robot(self, env_id: int, slot: int, xy, yaw: float = 0.0, velocity=(0.0, 0.0)) -> int:
        if slot < 0 or slot >= int(self.raw.num_robots):
            return -1
        actor_id = int(self.raw.robot_actor_idxs_all[env_id, slot].item())
        origin = self.raw.env_origins[env_id]
        root = self.raw.root_states[actor_id]
        root[0] = origin[0] + float(xy[0])
        root[1] = origin[1] + float(xy[1])
        root[2] = origin[2] + float(self.raw.base_init_state[2].item())
        root[3:7] = torch.tensor(
            [0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)],
            device=root.device,
            dtype=root.dtype,
        )
        root[7:13] = 0.0
        root[7] = float(velocity[0])
        root[8] = float(velocity[1])
        return actor_id

    def _set_ball(self, env_id: int, xy, velocity=(0.0, 0.0)) -> int:
        actor_id = int(self.raw.object_actor_idxs[env_id].item())
        origin = self.raw.env_origins[env_id]
        root = self.raw.root_states[actor_id]
        root[:] = self.raw.object_init_state
        root[:3] += origin
        root[0] = origin[0] + float(xy[0])
        root[1] = origin[1] + float(xy[1])
        radius = float(getattr(self.raw.cfg.ball, "radius", 0.0889))
        root[2] = origin[2] + radius + 0.002
        root[7:13] = 0.0
        root[7] = float(velocity[0])
        root[8] = float(velocity[1])
        return actor_id

    def _set_obstacle(self, env_id: int, xy=(0.0, 0.0), yaw: float = 0.0, slot: int = 0) -> int:
        actor_id = int(self.raw.static_opponent_actor_idxs[env_id, slot].item())
        origin = self.raw.env_origins[env_id]
        root = self.raw.root_states[actor_id]
        root[0] = origin[0] + float(xy[0])
        root[1] = origin[1] + float(xy[1])
        root[3:7] = torch.tensor(
            [0.0, 0.0, np.sin(0.5 * yaw), np.cos(0.5 * yaw)],
            device=root.device,
            dtype=root.dtype,
        )
        root[7:13] = 0.0
        return actor_id

    def _stage_one(self, env_id: int, scenario: str) -> list[int]:
        half_length = 0.5 * float(getattr(self.raw.cfg.env, "field_length", 8.0))
        half_width = 0.5 * float(getattr(self.raw.cfg.env, "field_width", 5.0))
        goal_x = float(getattr(self.raw.cfg.env, "team_goal_x", half_length))
        radius = float(getattr(self.raw.cfg.ball, "radius", 0.0889))
        speed = self.ball_speed
        changed = []

        # Safe two-team formation; scenario branches overwrite relevant actors.
        # Learning slots attack +x and opponent slots attack -x.
        total_robots = int(self.raw.num_robots)
        team_size = int(
            getattr(self.raw.cfg.env, "num_team_robots", total_robots // 2)
        )
        for robot in range(total_robots):
            local_slot = robot % team_size
            lateral = (local_slot - 0.5 * (team_size - 1)) * 0.8
            opponent = robot >= team_size
            x = (1.0 + 0.2 * local_slot) if opponent else (-1.0 - 0.2 * local_slot)
            yaw = np.pi if opponent else 0.0
            changed.append(self._set_robot(env_id, robot, (x, lateral), yaw))
        changed.append(self._set_ball(env_id, (0.0, 0.0)))
        obstacle_count = int(getattr(self.raw, "num_static_opponents", 0))
        for slot in range(obstacle_count):
            angle = 2.0 * np.pi * slot / max(obstacle_count, 1)
            xy = (0.55 * half_length * np.cos(angle), 0.55 * half_width * np.sin(angle))
            changed.append(self._set_obstacle(env_id, xy, 0.0, slot=slot))

        if scenario == "goal":
            changed.append(self._set_ball(env_id, (goal_x - 0.06, 0.0), (speed, 0.0)))
            changed.append(self._set_robot(env_id, 0, (goal_x - 0.42, 0.0), 0.0))
            changed.append(self._set_robot(env_id, 1, (goal_x - 1.3, 0.9), 0.0))
        elif scenario == "own_goal":
            changed.append(self._set_ball(env_id, (-half_length + 0.06, 0.0), (-speed, 0.0)))
            changed.append(self._set_robot(env_id, 0, (-half_length + 0.45, 0.0), np.pi))
            changed.append(self._set_robot(env_id, 1, (-half_length + 1.3, 0.9), np.pi))
        elif scenario == "out_of_bounds":
            changed.append(self._set_ball(env_id, (0.0, half_width - 0.06), (0.0, speed)))
            changed.append(self._set_robot(env_id, 0, (0.0, half_width - 0.42), 0.5 * np.pi))
            changed.append(self._set_robot(env_id, 1, (-1.0, half_width - 1.0), 0.5 * np.pi))
        elif scenario == "ball_obstacle_collision":
            changed.append(self._set_obstacle(env_id, (0.0, 0.0), 0.0))
            obstacle_half_x = 0.5 * float(self.raw.static_opponent_size[0])
            ball_x = -obstacle_half_x - radius - 0.035
            changed.append(self._set_ball(env_id, (ball_x, 0.0), (speed, 0.0)))
            changed.append(self._set_robot(env_id, 0, (ball_x - 0.35, 0.0), 0.0))
            changed.append(self._set_robot(env_id, 1, (-1.2, 0.9), 0.0))
        elif scenario == "robot_obstacle_collision":
            changed.append(self._set_obstacle(env_id, (0.0, 0.0), 0.0))
            changed.append(self._set_robot(env_id, 0, (-0.44, 0.0), 0.0, (0.5, 0.0)))
            changed.append(self._set_robot(env_id, 1, (-1.2, 0.9), 0.0))
            changed.append(self._set_ball(env_id, (1.2, -0.9)))
        elif scenario == "teammate_collision":
            changed.append(self._set_robot(env_id, 0, (-0.22, 0.0), 0.0, (0.3, 0.0)))
            changed.append(self._set_robot(env_id, 1, (0.22, 0.0), np.pi, (-0.3, 0.0)))
            changed.append(self._set_ball(env_id, (1.3, 1.0)))
        elif scenario == "possession_acquired":
            changed.append(self._set_robot(env_id, 0, (0.0, 0.0), 0.0))
            changed.append(self._set_robot(env_id, 1, (-1.3, 0.9), 0.0))
            changed.append(self._set_ball(env_id, (0.65, 0.0), (-2.0, 0.0)))
        elif scenario == "possession_lost":
            changed.append(self._set_robot(env_id, 0, (-0.30, 0.0), 0.0))
            changed.append(self._set_robot(env_id, 1, (-1.4, 1.0), 0.0))
            changed.append(self._set_ball(env_id, (0.0, 0.0), (speed, 0.0)))
        elif scenario == "pass":
            changed.append(self._set_robot(env_id, 0, (-0.30, 0.0), 0.0))
            changed.append(self._set_robot(env_id, 1, (0.68, 0.0), 0.0))
            changed.append(self._set_ball(env_id, (0.0, 0.0), (2.2, 0.0)))
        elif scenario == "successful_shot":
            changed.append(self._set_robot(env_id, 0, (-0.35, 0.0), 0.0))
            changed.append(self._set_robot(env_id, 1, (-1.2, 0.9), 0.0))
            changed.append(self._set_ball(env_id, (0.0, 0.0), (1.5, 0.0)))
        elif scenario == "failed_shot":
            changed.append(self._set_robot(env_id, 0, (-0.35, 0.0), 0.0))
            changed.append(self._set_robot(env_id, 1, (-1.2, 0.9), 0.0))
            changed.append(self._set_ball(env_id, (0.0, 0.0)))
        else:  # Guard against future edits bypassing constructor validation.
            raise ValueError(f"Unsupported targeted scenario {scenario!r}")
        return changed

    def _commit(self, env_ids: Sequence[int], actor_ids: Sequence[int]) -> None:
        # Isaac Gym must already have been imported by the simulator entry point.
        from isaacgym import gymtorch

        actor_tensor = torch.as_tensor(
            [actor_id for actor_id in actor_ids if actor_id >= 0],
            dtype=torch.int32,
            device=self.raw.device,
        ).unique()
        self.raw.gym.set_actor_root_state_tensor_indexed(
            self.raw.sim,
            gymtorch.unwrap_tensor(self.raw.root_states),
            gymtorch.unwrap_tensor(actor_tensor),
            int(actor_tensor.numel()),
        )
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.raw.device)
        object_ids = self.raw.object_actor_idxs[ids]
        self.raw.object_pos_world_frame[ids] = self.raw.root_states[object_ids, :3]
        self.raw.object_lin_vel[ids] = self.raw.root_states[object_ids, 7:10]
        self.raw.object_ang_vel[ids] = self.raw.root_states[object_ids, 10:13]
        if hasattr(self.raw, "prev_object_pos_world_frame"):
            self.raw.prev_object_pos_world_frame[ids] = self.raw.object_pos_world_frame[ids]
        if hasattr(self.raw, "_refresh_two_robot_views"):
            self.raw._refresh_two_robot_views()
        if hasattr(self.raw, "prev_high_level_robot_ball_distances"):
            self.raw.prev_high_level_robot_ball_distances[ids] = self.raw._high_level_robot_ball_distances()[ids]
        if hasattr(self.wrapper, "_clear_high_level_state"):
            self.wrapper._clear_high_level_state(ids)
        for name in (
            "high_level_goal_buf",
            "high_level_ball_off_border_buf",
            "high_level_obstacle_contact_buf",
            "high_level_accidental_termination_buf",
            "last_high_level_goal_buf",
            "last_high_level_ball_off_border_buf",
            "last_high_level_obstacle_contact_buf",
            "last_high_level_accidental_termination_buf",
        ):
            if hasattr(self.raw, name):
                getattr(self.raw, name)[ids] = False
        for name in (
            "low_level_obs_history_full",
            "low_level_actions",
            "last_low_level_actions",
            "high_level_obs_history",
        ):
            if hasattr(self.wrapper, name):
                getattr(self.wrapper, name)[ids] = 0.0
