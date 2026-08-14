"""Receding-horizon execution against the real vectorized simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch

from dribblebot.world_model.action_adapter import JointActionAdapter

from .hybrid_cem import HybridCEMMPC, MPCPlanResult
from .local_observation import LocalObservationAdapter
from .planner_state import MPCPlannerState


class TerminalStateCapture:
    """Capture compact states immediately before the raw env auto-resets."""

    def __init__(self, wrapper, adapter):
        self.wrapper = wrapper
        self.raw = wrapper.env
        self.adapter = adapter
        self.original_reset_idx = self.raw.reset_idx
        self.states = torch.zeros(
            self.raw.num_envs,
            adapter.state_dim,
            device=self.raw.device,
        )
        self.valid = torch.zeros(
            self.raw.num_envs,
            dtype=torch.bool,
            device=self.raw.device,
        )

        def reset_with_capture(env_ids):
            if env_ids.numel():
                uncaptured = env_ids[~self.valid[env_ids]]
                if uncaptured.numel():
                    snapshot = self.adapter.extract_state(self.wrapper)["tensor"]
                    self.states[uncaptured] = snapshot[uncaptured]
                    self.valid[uncaptured] = True
            return self.original_reset_idx(env_ids)

        self.raw.reset_idx = reset_with_capture

    def clear(self) -> None:
        self.valid.zero_()

    def restore(self) -> None:
        self.raw.reset_idx = self.original_reset_idx


@dataclass
class MPCTransition:
    """One real macro transition plus its teacher label and diagnostics."""

    state: torch.Tensor
    local_observations: torch.Tensor
    requested_action: torch.Tensor
    executed_action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    next_local_observations: torch.Tensor
    live_post_step_state: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    done: torch.Tensor
    elapsed_low_level_steps: torch.Tensor
    event_labels: torch.Tensor
    info: Dict[str, Any]
    plan: MPCPlanResult
    requested_action_modified: torch.Tensor
    teacher_action_executed: torch.Tensor


def _same_action_schema(left: JointActionAdapter, right: JointActionAdapter) -> bool:
    if left.num_robots != right.num_robots:
        return False
    for skill in range(3):
        a, b = left.bounds[skill], right.bounds[skill]
        if not (
            np.allclose(a.low, b.low)
            and np.allclose(a.high, b.high)
            and np.allclose(a.mask, b.mask)
        ):
            return False
    return True


def validate_environment_compatibility(
    env,
    model,
    checkpoint: Mapping[str, Any],
    configured_macro_action_steps: Optional[int] = None,
) -> None:
    """Fail early when live execution differs from the trained model contract."""

    training = checkpoint.get("training_config", {})
    checkpoint_robot = training.get("environment", {}).get("robot")
    live_robot = str(getattr(getattr(env.env.cfg, "robot", object()), "name", ""))
    if checkpoint_robot and live_robot and checkpoint_robot != live_robot:
        raise ValueError(
            f"Checkpoint robot {checkpoint_robot!r} does not match live robot {live_robot!r}"
        )
    trained_interval = training.get("world_model", {}).get("macro_action_steps")
    live_interval = int(getattr(env, "control_interval", -1))
    requested_interval = (
        live_interval
        if configured_macro_action_steps is None
        else int(configured_macro_action_steps)
    )
    if trained_interval is not None and int(trained_interval) != requested_interval:
        raise ValueError(
            f"Checkpoint macro interval {trained_interval} does not match live interval {requested_interval}"
        )
    if live_interval != requested_interval:
        raise ValueError(
            f"Wrapper control_interval={live_interval} does not match configured value {requested_interval}"
        )
    live_actions = JointActionAdapter.from_env(env)
    if not _same_action_schema(live_actions, model.action_adapter):
        raise ValueError(
            "Live robot count or skill-dependent command bounds/masks differ from "
            "the world-model checkpoint "
            f"(live robots={live_actions.num_robots}, checkpoint robots="
            f"{model.action_adapter.num_robots})"
        )


class MPCSimulatorController:
    """Execute exactly one planned macro action, observe reality, then replan."""

    def __init__(
        self,
        env,
        planner: HybridCEMMPC,
        state_adapter,
        local_observation_adapter: Optional[LocalObservationAdapter] = None,
        capture_terminal_state: bool = True,
        opponent_forecaster=None,
    ):
        self.env = env
        self.planner = planner
        self.state_adapter = state_adapter
        self.action_adapter = planner.action_adapter
        self.local_adapter = local_observation_adapter or LocalObservationAdapter(
            planner.world_model.schema
        )
        self.planner_state: Optional[MPCPlannerState] = None
        self.capture = (
            TerminalStateCapture(env, state_adapter) if capture_terminal_state else None
        )
        self.opponent_forecaster = opponent_forecaster

    def close(self) -> None:
        if self.capture is not None:
            self.capture.restore()
            self.capture = None

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            observations = self.env.reset()
            self.planner_state = None
            if self.capture is not None:
                self.capture.clear()
            if self.opponent_forecaster is not None:
                self.opponent_forecaster.reset()
            return observations
        if self.planner_state is not None:
            self.planner_state.reset(env_ids)
        if self.opponent_forecaster is not None:
            self.opponent_forecaster.reset(env_ids)
        return None

    def act(self, global_states: torch.Tensor) -> MPCPlanResult:
        fixed_action_sequence = fixed_robot_mask = None
        if self.opponent_forecaster is not None:
            fixed_action_sequence, fixed_robot_mask = (
                self.opponent_forecaster.fixed_action_sequence(
                    self.planner.config.horizon
                )
            )
        plan = self.planner.plan(
            global_states,
            self.planner_state,
            fixed_action_sequence=fixed_action_sequence,
            fixed_robot_mask=fixed_robot_mask,
        )
        self.planner_state = plan.planner_state
        return plan

    def _executed_action(self, info: Mapping[str, Any], device) -> torch.Tensor:
        skills = torch.as_tensor(
            info["high_level_skill_ids"], device=device, dtype=torch.long
        )
        commands = torch.as_tensor(
            info["high_level_commands"], device=device, dtype=torch.float
        )
        result = self.action_adapter.pack(skills, commands)
        self.action_adapter.assert_within_bounds(result, atol=1e-5)
        return result

    def observe_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        event_labels: Optional[torch.Tensor] = None,
        requested_action_modified: Optional[torch.Tensor] = None,
    ) -> None:
        if self.planner_state is None:
            return
        reset = dones.bool().clone()
        if requested_action_modified is not None:
            reset |= requested_action_modified.bool()
        if event_labels is not None:
            event_indices = {
                name: index
                for index, name in enumerate(
                    getattr(self.planner.world_model, "event_names", ())
                )
            }
            for name in ("goal", "own_goal", "possession_acquired", "possession_lost"):
                index = event_indices.get(name)
                if index is not None:
                    reset |= event_labels[:, index] > 0.5
        for robot in range(self.action_adapter.num_robots):
            reset |= (
                next_states[
                    :, self.planner.world_model.schema.slice(f"robot_{robot}.fallen")
                ].squeeze(-1)
                > 0.5
            )
        self.planner_state.previous_action = actions.detach()
        if bool(reset.any().item()):
            self.planner_state.reset(reset.nonzero(as_tuple=False).flatten())

    @torch.no_grad()
    def step(
        self,
        execution_policy: Optional[
            Callable[[torch.Tensor, MPCPlanResult], torch.Tensor]
        ] = None,
    ) -> MPCTransition:
        """Plan and execute one wrapper step (already one macro interval)."""

        state = self.state_adapter.extract_state(self.env)["tensor"]
        local = self.local_adapter.extract(state)
        plan = self.act(state)
        teacher_action = plan.first_joint_action
        requested = (
            teacher_action
            if execution_policy is None
            else execution_policy(local, plan).to(
                device=state.device, dtype=state.dtype
            )
        )
        if requested.shape != teacher_action.shape:
            raise ValueError(
                f"Execution policy must return {tuple(teacher_action.shape)}, got {tuple(requested.shape)}"
            )
        self.action_adapter.assert_within_bounds(requested)
        teacher_action_executed = torch.isclose(
            requested, teacher_action, atol=1e-5
        ).all(dim=-1)
        if self.capture is not None:
            self.capture.clear()
        _, reward, done, raw_info = self.env.step(
            self.action_adapter.to_wrapper_action(requested)
        )
        info = dict(raw_info)
        executed = self._executed_action(info, state.device)
        live_next_state = self.state_adapter.extract_state(self.env)["tensor"]
        if self.capture is not None:
            next_state = torch.where(
                self.capture.valid[:, None], self.capture.states, live_next_state
            )
        else:
            next_state = live_next_state
        next_local = self.local_adapter.extract(next_state)
        event_labels = self.state_adapter.extract_event_labels(state, next_state, info)
        timeouts = torch.as_tensor(
            info.get("time_outs", np.zeros(self.env.num_envs)),
            device=done.device,
        ).bool()
        done = done.bool()
        terminated = done & ~timeouts
        truncated = done & timeouts
        elapsed = torch.as_tensor(
            info.get(
                "elapsed_low_level_steps",
                np.full(self.env.num_envs, self.env.control_interval),
            ),
            device=done.device,
            dtype=torch.long,
        )
        modified = ~torch.isclose(requested, executed, atol=1e-5).all(dim=-1)
        self.observe_transition(
            state,
            executed,
            reward,
            next_state,
            done,
            event_labels,
            modified | ~teacher_action_executed,
        )
        if self.opponent_forecaster is not None:
            self.opponent_forecaster.observe(done)
        return MPCTransition(
            state=state,
            local_observations=local,
            requested_action=requested,
            executed_action=executed,
            reward=reward,
            next_state=next_state,
            next_local_observations=next_local,
            live_post_step_state=live_next_state,
            terminated=terminated,
            truncated=truncated,
            done=done,
            elapsed_low_level_steps=elapsed,
            event_labels=event_labels,
            info=info,
            plan=plan,
            requested_action_modified=modified,
            teacher_action_executed=teacher_action_executed,
        )
