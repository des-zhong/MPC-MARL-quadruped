"""Simulator evaluation, baseline comparison, and MPC ablations."""

from __future__ import annotations

import math
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, Mapping, Optional

import numpy as np
import torch

from dribblebot.world_model.behavior_policies import BehaviorMixture


def _num_robots(schema) -> int:
    return sum(feature.name.startswith("robot_") and feature.name.endswith(".position") for feature in schema.features)


def aggregate_statistics(values, confidence: float = 0.95) -> Dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "confidence_interval": [None, None],
        }
    # Normal approximation is transparent and stable for the requested
    # hundreds of episodes.  1.96 is used for the configured 95% default.
    z = 1.96 if abs(confidence - 0.95) < 1.0e-8 else 1.96
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = z * std / math.sqrt(max(len(values), 1))
    return {
        "count": int(len(values)),
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "confidence_interval": [mean - half, mean + half],
    }


def _physical_position_error(schema, prediction, target, prefix):
    field = target[:, schema.slice("field.geometry")]
    scale = torch.stack(
        (field[:, 0], field[:, 1], torch.ones_like(field[:, 0])), -1
    )
    names = (
        [f"robot_{robot}.position" for robot in range(_num_robots(schema))]
        if prefix == "robot"
        else ["ball.position"]
    )
    errors = [
        (prediction[:, schema.slice(name)] - target[:, schema.slice(name)]) * scale
        for name in names
    ]
    return torch.cat(errors, -1).square().mean(-1).sqrt()


def _velocity_error(schema, prediction, target, prefix):
    names = (
        [f"robot_{robot}.linear_velocity" for robot in range(_num_robots(schema))]
        if prefix == "robot"
        else ["ball.linear_velocity"]
    )
    errors = [
        prediction[:, schema.slice(name)] - target[:, schema.slice(name)]
        for name in names
    ]
    return torch.cat(errors, -1).square().mean(-1).sqrt()


@torch.no_grad()
def _baseline_step(runtime, requested_action):
    controller = runtime.controller
    state = runtime.state_adapter.extract_state(runtime.env)["tensor"]
    local = runtime.local_adapter.extract(state)
    controller.capture.clear()
    _, reward, done, raw_info = runtime.env.step(
        runtime.model.action_adapter.to_wrapper_action(requested_action)
    )
    info = dict(raw_info)
    executed = controller._executed_action(info, state.device)
    live_next = runtime.state_adapter.extract_state(runtime.env)["tensor"]
    next_state = torch.where(
        controller.capture.valid[:, None], controller.capture.states, live_next
    )
    events = runtime.state_adapter.extract_event_labels(state, next_state, info)
    timeouts = torch.as_tensor(
        info.get("time_outs", np.zeros(runtime.env.num_envs)),
        device=done.device,
    ).bool()
    elapsed = torch.as_tensor(
        info["elapsed_low_level_steps"], device=done.device, dtype=torch.long
    )
    return SimpleNamespace(
        state=state,
        local_observations=local,
        requested_action=requested_action,
        executed_action=executed,
        reward=reward,
        next_state=next_state,
        live_post_step_state=live_next,
        done=done.bool(),
        terminated=done.bool() & ~timeouts,
        truncated=done.bool() & timeouts,
        elapsed_low_level_steps=elapsed,
        event_labels=events,
        info=info,
        plan=None,
    )


class _EpisodeMetrics:
    def __init__(self, count):
        self.rows = [defaultdict(float) for _ in range(count)]
        self.started = [False] * count
        self.completed = []

    def update(
        self,
        transition,
        prediction,
        predicted_reward,
        predicted_done,
        predicted_events,
        uncertainty,
        schema,
        event_names,
        planning_time_per_env,
        max_episodes,
    ):
        field = transition.state[:, schema.slice("field.geometry")]
        start_ball = (
            transition.state[:, schema.slice("ball.position")][:, 0] * field[:, 0]
        )
        final_ball = (
            transition.next_state[:, schema.slice("ball.position")][:, 0]
            * field[:, 0]
        )
        robot_position = _physical_position_error(
            schema, prediction, transition.next_state, "robot"
        )
        ball_position = _physical_position_error(
            schema, prediction, transition.next_state, "ball"
        )
        robot_velocity = _velocity_error(
            schema, prediction, transition.next_state, "robot"
        )
        ball_velocity = _velocity_error(
            schema, prediction, transition.next_state, "ball"
        )
        event_accuracy = (
            (predicted_events >= 0.5)
            == transition.event_labels.bool()
        ).float().mean(-1)
        done_target = transition.done.float()
        num_robots = _num_robots(schema)
        skills = transition.executed_action.reshape(-1, num_robots, 4)[..., 0].long()
        possessed = transition.next_state[
            :, schema.slice("ball.possessed")
        ].squeeze(-1)
        event_index = {name: index for index, name in enumerate(event_names)}
        for env_index in range(transition.state.shape[0]):
            if len(self.completed) >= max_episodes:
                break
            row = self.rows[env_index]
            if not self.started[env_index]:
                row["initial_ball_x"] = float(start_ball[env_index])
                self.started[env_index] = True
            row["episode_return"] += float(transition.reward[env_index])
            row["episode_length"] += 1
            row["simulator_steps"] += int(
                transition.elapsed_low_level_steps[env_index]
            )
            row["shot_count"] += int((skills[env_index] == 2).sum())
            row["possession_duration_macro_steps"] += float(possessed[env_index])
            row["planning_time_seconds"] += float(planning_time_per_env)
            row["model_uncertainty"] += float(uncertainty[env_index])
            row["one_step_robot_position_error_m"] += float(
                robot_position[env_index]
            )
            row["one_step_ball_position_error_m"] += float(
                ball_position[env_index]
            )
            row["one_step_robot_velocity_error"] += float(
                robot_velocity[env_index]
            )
            row["one_step_ball_velocity_error"] += float(
                ball_velocity[env_index]
            )
            row["reward_absolute_error"] += float(
                (predicted_reward[env_index] - transition.reward[env_index]).abs()
            )
            row["termination_brier"] += float(
                (predicted_done[env_index] - done_target[env_index]).square()
            )
            row["event_accuracy"] += float(event_accuracy[env_index])
            row["final_ball_x"] = float(final_ball[env_index])
            for metric, names in (
                ("goal", ("goal",)),
                ("successful_shot", ("successful_shot",)),
                (
                    "cylinder_collision",
                    ("ball_obstacle_collision", "robot_obstacle_collision"),
                ),
                ("teammate_collision", ("teammate_collision",)),
                ("out_of_bounds", ("out_of_bounds",)),
            ):
                value = any(
                    name in event_index
                    and bool(
                        transition.event_labels[
                            env_index, event_index[name]
                        ]
                        > 0.5
                    )
                    for name in names
                )
                row[metric] += int(value)
            fallen = any(
                bool(
                    transition.next_state[
                        env_index, schema.slice(f"robot_{robot}.fallen")
                    ].item()
                    > 0.5
                )
                for robot in range(num_robots)
            )
            row["robot_fall"] += int(fallen)
            if bool(transition.done[env_index]):
                steps = max(row["episode_length"], 1.0)
                for name in (
                    "planning_time_seconds",
                    "model_uncertainty",
                    "one_step_robot_position_error_m",
                    "one_step_ball_position_error_m",
                    "one_step_robot_velocity_error",
                    "one_step_ball_velocity_error",
                    "reward_absolute_error",
                    "termination_brier",
                    "event_accuracy",
                ):
                    row[name] /= steps
                row["ball_progress_m"] = (
                    row["final_ball_x"] - row["initial_ball_x"]
                )
                row["goal_rate"] = float(row["goal"] > 0)
                row["cylinder_collision_rate"] = float(
                    row["cylinder_collision"] > 0
                )
                row["teammate_collision_rate"] = float(
                    row["teammate_collision"] > 0
                )
                row["out_of_bounds_rate"] = float(row["out_of_bounds"] > 0)
                row["robot_fall_rate"] = float(row["robot_fall"] > 0)
                row["successful_shot_rate"] = (
                    row["successful_shot"] / max(row["shot_count"], 1.0)
                )
                self.completed.append(dict(row))
                self.rows[env_index] = defaultdict(float)
                self.started[env_index] = False


@torch.no_grad()
def evaluate_method(
    runtime,
    method: str,
    num_episodes: int,
    confidence: float = 0.95,
) -> Dict[str, object]:
    runtime.controller.reset()
    accumulator = _EpisodeMetrics(runtime.env.num_envs)
    scripted = None
    if method == "scripted":
        scripted = BehaviorMixture(
            runtime.model.action_adapter,
            runtime.model.schema,
            {"scripted": 1.0},
            repeat_previous_probability=0.0,
            seed=int(runtime.config.get("seed", 42)),
        )
    while len(accumulator.completed) < num_episodes:
        state = runtime.state_adapter.extract_state(runtime.env)["tensor"]
        if method == "random_valid":
            action = runtime.model.action_adapter.random_valid(
                (runtime.env.num_envs,), state.device
            )
            transition = _baseline_step(runtime, action)
            planning_time = 0.0
        elif method == "scripted":
            action, _ = scripted.sample(state)
            transition = _baseline_step(runtime, action)
            planning_time = 0.0
        else:
            transition = runtime.controller.step()
            planning_time = (
                transition.plan.planning_time_seconds / runtime.env.num_envs
            )
        (
            predicted_next,
            predicted_reward,
            predicted_done,
            predicted_events,
            prediction_uncertainty,
        ) = runtime.model.predict_next(
            transition.state,
            transition.executed_action,
            deterministic=True,
        )
        accumulator.update(
            transition,
            predicted_next,
            predicted_reward,
            predicted_done,
            predicted_events,
            prediction_uncertainty["mean_state_uncertainty"],
            runtime.model.schema,
            runtime.model.event_names,
            planning_time,
            num_episodes,
        )
    rows = accumulator.completed[:num_episodes]
    keys = sorted({key for row in rows for key in row})
    aggregate = {
        key: aggregate_statistics(
            [row.get(key, float("nan")) for row in rows], confidence
        )
        for key in keys
    }
    goals = sum(row["goal_rate"] for row in rows)
    aggregate["simulator_steps_per_successful_goal"] = {
        "count": int(goals),
        "mean": (
            float(sum(row["simulator_steps"] for row in rows) / goals)
            if goals
            else None
        ),
    }
    return {
        "method": method,
        "episodes": len(rows),
        "metrics": aggregate,
        "raw_episode_metrics": rows,
        "open_loop_plan_validation": {
            "performed": False,
            "reason": (
                "The repository has no exact full simulator snapshot/restore "
                "API; matched open-loop rollouts require fresh seeded envs and "
                "are kept separate from receding-horizon teacher execution."
            ),
        },
    }


def method_overrides(method: str) -> Optional[Dict[str, object]]:
    if method in ("random_valid", "scripted", "mpc"):
        return None
    if method == "greedy_h1":
        return {"horizon": 1}
    if method == "mpc_no_uncertainty":
        return {
            "uncertainty_penalty": 0.0,
            "return_std_penalty": 0.0,
            "ensemble_objective": "mean",
        }
    raise ValueError(f"Unknown MPC evaluation method {method!r}")
