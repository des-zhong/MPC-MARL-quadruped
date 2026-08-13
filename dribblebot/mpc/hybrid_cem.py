"""Batched Cross-Entropy Method planning for joint hybrid skill actions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from .config import MPCConfig
from .fallback_policy import PreviousActionFallback, SafeRepositionFallback
from .objective import MPCObjective
from .planner_state import MPCPlannerState


@dataclass
class MPCPlanResult:
    """Structured result from one receding-horizon planning call.

    Shapes use ``B`` environments, ``H`` horizon, ``D`` state dimension,
    ``E`` elites, ``N`` robots, three skills, and three parameters:

    - ``first_joint_action``: ``[B,8]``
    - ``best_action_sequence``: ``[B,H,8]``
    - ``best_objective``: ``[B]``
    - ``predicted_states``: ``[B,H+1,D]``
    - rewards/done probabilities: ``[B,H]``
    - final skill probabilities: ``[B,H,2,3]``
    - final parameter means/stds (physical units): ``[B,H,2,3,3]``
    - ``elite_objectives``: ``[B,E]``
    """

    first_joint_action: torch.Tensor
    best_action_sequence: torch.Tensor
    best_objective: torch.Tensor
    predicted_states: torch.Tensor
    predicted_rewards: torch.Tensor
    predicted_done_probabilities: torch.Tensor
    objective_components: Dict[str, torch.Tensor]
    uncertainty: Dict[str, torch.Tensor]
    final_skill_probabilities: torch.Tensor
    final_parameter_means: torch.Tensor
    final_parameter_stds: torch.Tensor
    elite_objectives: torch.Tensor
    planning_time_seconds: float
    planner_state: MPCPlannerState
    fallback_used: torch.Tensor
    predicted_discounted_reward_return: torch.Tensor
    terminal_state_value: torch.Tensor
    discounted_terminal_value: torch.Tensor
    terminal_value_contribution: torch.Tensor
    terminal_state_uncertainty: torch.Tensor
    terminal_value_clipped: torch.Tensor
    predicted_state_values: torch.Tensor
    terminal_value_coefficient: float
    convergence: Dict[str, torch.Tensor] = field(default_factory=dict)
    candidate_diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)


class HybridCEMMPC:
    """GPU-vectorized CEM over skills and skill-conditioned parameters."""

    num_skills = 3
    parameter_dim = 3

    def __init__(
        self,
        world_model,
        state_adapter=None,
        action_adapter=None,
        objective: Optional[MPCObjective] = None,
        config: Optional[MPCConfig] = None,
        fallback_policy=None,
        terminal_value=None,
    ):
        self.world_model = world_model
        self.state_adapter = state_adapter or world_model.state_adapter
        self.action_adapter = action_adapter or world_model.action_adapter
        self.num_robots = self.action_adapter.num_robots
        self.config = (config or MPCConfig()).validate()
        self.objective = objective or MPCObjective(
            world_model.schema,
            self.action_adapter,
            getattr(world_model, "event_names", ()),
            self.config,
            terminal_value=terminal_value,
        )
        if fallback_policy is None:
            if self.config.fallback_policy == "previous_action":
                fallback_policy = PreviousActionFallback(self.action_adapter)
            elif self.config.fallback_policy == "safe_reposition":
                fallback_policy = SafeRepositionFallback(self.action_adapter)
            else:
                raise ValueError(f"Unknown fallback policy {self.config.fallback_policy!r}")
        self.fallback_policy = fallback_policy
        self._generator = None
        self._generator_device = None

    def _rng(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if self._generator is None or self._generator_device != key:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(self.config.seed))
            self._generator_device = key
        return self._generator

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _defaults(
        self, batch: int, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (batch, self.config.horizon, self.num_robots, self.num_skills)
        probabilities = torch.full(shape, 1.0 / self.num_skills, dtype=dtype, device=device)
        parameter_shape = shape + (self.parameter_dim,)
        means = torch.zeros(parameter_shape, dtype=dtype, device=device)
        stds = torch.full(
            parameter_shape,
            self.config.initial_parameter_std_fraction,
            dtype=dtype,
            device=device,
        )
        return probabilities, means, stds

    def _warm_start_validity(
        self, states: torch.Tensor, planner_state: MPCPlannerState
    ) -> torch.Tensor:
        valid = planner_state.valid.to(states.device).clone()
        if planner_state.previous_state is not None:
            previous = planner_state.previous_state.to(states.device)
            if previous.shape != states.shape:
                return torch.zeros_like(valid)
            old_possessor = previous[
                :, self.world_model.schema.slice("ball.possessor_one_hot")
            ].argmax(-1)
            new_possessor = states[
                :, self.world_model.schema.slice("ball.possessor_one_hot")
            ].argmax(-1)
            valid &= old_possessor == new_possessor
        reset_signal = torch.zeros_like(valid)
        for robot in range(self.num_robots):
            reset_signal |= (
                states[:, self.world_model.schema.slice(f"robot_{robot}.fallen")].squeeze(-1)
                > 0.5
            )
        for name in ("ball.in_opponent_goal", "ball.in_own_goal", "ball.out_of_bounds"):
            reset_signal |= states[:, self.world_model.schema.slice(name)].squeeze(-1) > 0.5
        valid &= ~reset_signal
        threshold = self.config.warm_start_uncertainty_threshold
        if threshold is not None and planner_state.last_plan_uncertainty is not None:
            valid &= planner_state.last_plan_uncertainty.to(states.device) <= float(threshold)
        return valid

    def _initialize(
        self,
        states: torch.Tensor,
        planner_state: Optional[MPCPlannerState],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        defaults = self._defaults(states.shape[0], states.dtype, states.device)
        if not self.config.warm_start or planner_state is None:
            return defaults
        expected_probs = (
            states.shape[0],
            self.config.horizon,
            self.num_robots,
            self.num_skills,
        )
        expected_params = expected_probs + (self.parameter_dim,)
        if (
            tuple(planner_state.skill_probabilities.shape) != expected_probs
            or tuple(planner_state.parameter_means.shape) != expected_params
            or tuple(planner_state.parameter_stds.shape) != expected_params
            or tuple(planner_state.valid.shape) != (states.shape[0],)
        ):
            return defaults
        previous = planner_state.to(states.device)
        valid = self._warm_start_validity(states, previous)
        default_probs, default_means, default_stds = defaults
        shifted_probs = torch.cat(
            (previous.skill_probabilities[:, 1:], default_probs[:, -1:]), dim=1
        )
        shifted_means = torch.cat(
            (previous.parameter_means[:, 1:], default_means[:, -1:]), dim=1
        )
        shifted_stds = torch.cat(
            (previous.parameter_stds[:, 1:], default_stds[:, -1:]), dim=1
        )
        return (
            torch.where(valid[:, None, None, None], shifted_probs, default_probs),
            torch.where(valid[:, None, None, None, None], shifted_means, default_means),
            torch.where(valid[:, None, None, None, None], shifted_stds, default_stds),
        )

    def _sample(
        self,
        probabilities: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, horizon = probabilities.shape[:2]
        generator = self._rng(probabilities.device)
        candidates = self.config.num_candidates
        eps = torch.finfo(probabilities.dtype).eps
        logits = probabilities.clamp(min=eps).log() / self.config.skill_temperature
        uniform = torch.rand(
            batch,
            candidates,
            horizon,
            self.num_robots,
            self.num_skills,
            dtype=probabilities.dtype,
            device=probabilities.device,
            generator=generator,
        ).clamp(min=eps, max=1.0 - eps)
        gumbel = -torch.log(-torch.log(uniform))
        skills = (logits[:, None] + gumbel).argmax(dim=-1)
        if self.config.minimum_skill_duration > 1:
            run_length = torch.ones(
                batch,
                candidates,
                self.num_robots,
                dtype=torch.long,
                device=skills.device,
            )
            for step in range(1, horizon):
                proposed = skills[:, :, step]
                previous = skills[:, :, step - 1]
                forced = run_length < self.config.minimum_skill_duration
                selected = torch.where(forced, previous, proposed)
                changed = selected != previous
                skills[:, :, step] = selected
                run_length = torch.where(changed, torch.ones_like(run_length), run_length + 1)

        noise = torch.randn(
            batch,
            candidates,
            horizon,
            self.num_robots,
            self.num_skills,
            self.parameter_dim,
            dtype=means.dtype,
            device=means.device,
            generator=generator,
        )
        all_normalized = (means[:, None] + stds[:, None] * noise).clamp(-1.0, 1.0)
        gather_index = skills[..., None, None].expand(
            -1, -1, -1, -1, 1, self.parameter_dim
        )
        normalized = torch.gather(all_normalized, -2, gather_index).squeeze(-2)
        parameters = self.action_adapter.denormalize_parameters(skills, normalized)
        actions = self.action_adapter.pack(skills, parameters)
        return actions, skills, normalized

    def _apply_fixed_robot_actions(
        self,
        actions: torch.Tensor,
        fixed_action_sequence: Optional[torch.Tensor],
        fixed_robot_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if fixed_action_sequence is None:
            return actions
        shaped = actions.reshape(*actions.shape[:-1], self.num_robots, 4).clone()
        fixed = fixed_action_sequence.reshape(
            fixed_action_sequence.shape[0],
            fixed_action_sequence.shape[1],
            self.num_robots,
            4,
        )
        mask = fixed_robot_mask.to(device=actions.device, dtype=torch.bool)
        if actions.ndim == 4:
            fixed = fixed[:, None]
        shaped[..., mask, :] = fixed[..., mask, :]
        return shaped.flatten(-2)

    def _evaluate_candidates(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        count = actions.shape[1]
        chunk = self.config.candidate_batch_size or count
        objectives = []
        validity = []
        for start in range(0, count, chunk):
            selected = actions[:, start : start + chunk]
            rollout_method = (
                self.world_model.rollout_members
                if self.config.ensemble_objective in ("mean_minus_std", "minimum")
                and hasattr(self.world_model, "rollout_members")
                else self.world_model.rollout
            )
            rollout = rollout_method(
                states, selected,
                deterministic=self.config.deterministic_world_model,
                stop_on_done=False,
            )
            result = self.objective.evaluate(states, selected, rollout)
            objectives.append(result.total)
            validity.append(result.valid)
        return torch.cat(objectives, dim=1), torch.cat(validity, dim=1)

    @staticmethod
    def _gather_candidates(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        shape = (indices.shape[0], indices.shape[1]) + value.shape[2:]
        expanded = indices.reshape(indices.shape + (1,) * (value.ndim - 2)).expand(shape)
        return torch.gather(value, 1, expanded)

    def _elite_update(
        self,
        probabilities: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        skills: torch.Tensor,
        normalized: torch.Tensor,
        elite_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        elite_skills = self._gather_candidates(skills, elite_indices)
        elite_parameters = self._gather_candidates(normalized, elite_indices)
        one_hot = torch.nn.functional.one_hot(
            elite_skills, num_classes=self.num_skills
        ).to(normalized.dtype)
        empirical_probs = one_hot.mean(dim=1)
        floor = self.config.min_skill_probability
        empirical_probs = floor + (1.0 - self.num_skills * floor) * empirical_probs
        alpha_cat = self.config.categorical_smoothing
        new_probabilities = alpha_cat * probabilities + (1.0 - alpha_cat) * empirical_probs
        new_probabilities = new_probabilities.clamp(min=floor)
        new_probabilities = new_probabilities / new_probabilities.sum(-1, keepdim=True)

        weights = one_hot.unsqueeze(-1)
        counts = weights.sum(dim=1)
        conditional_mean = (weights * elite_parameters.unsqueeze(-2)).sum(dim=1) / counts.clamp(min=1.0)
        centered = elite_parameters.unsqueeze(-2) - conditional_mean.unsqueeze(1)
        conditional_var = (weights * centered.square()).sum(dim=1) / counts.clamp(min=1.0)
        conditional_std = conditional_var.clamp(min=0.0).sqrt()
        observed = counts > 0
        conditional_mean = torch.where(observed, conditional_mean, means)
        conditional_std = torch.where(observed, conditional_std, stds)
        alpha_cont = self.config.continuous_smoothing
        new_means = alpha_cont * means + (1.0 - alpha_cont) * conditional_mean
        new_stds = alpha_cont * stds + (1.0 - alpha_cont) * conditional_std
        new_stds = new_stds.clamp(
            min=self.config.min_parameter_std_fraction,
            max=self.config.max_parameter_std_fraction,
        )
        return new_probabilities, new_means.clamp(-1.0, 1.0), new_stds

    def _mode_sequence(
        self, probabilities: torch.Tensor, means: torch.Tensor
    ) -> torch.Tensor:
        skills = probabilities.argmax(-1)
        gather_index = skills[..., None, None].expand(
            -1, -1, -1, 1, self.parameter_dim
        )
        normalized = torch.gather(means, -2, gather_index).squeeze(-2)
        parameters = self.action_adapter.denormalize_parameters(skills, normalized)
        return self.action_adapter.pack(skills, parameters)

    def _physical_distribution(
        self, means: torch.Tensor, stds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lows = torch.tensor(
            [self.action_adapter.bounds[index].low for index in range(self.num_skills)],
            dtype=means.dtype,
            device=means.device,
        )
        highs = torch.tensor(
            [self.action_adapter.bounds[index].high for index in range(self.num_skills)],
            dtype=means.dtype,
            device=means.device,
        )
        masks = torch.tensor(
            [self.action_adapter.bounds[index].mask for index in range(self.num_skills)],
            dtype=means.dtype,
            device=means.device,
        )
        physical_mean = (lows + 0.5 * (means + 1.0) * (highs - lows)) * masks
        physical_std = (0.5 * stds * (highs - lows)) * masks
        return physical_mean, physical_std

    def _smooth_first_action(
        self,
        sequence: torch.Tensor,
        planner_state: Optional[MPCPlannerState],
    ) -> torch.Tensor:
        amount = self.config.execution_action_smoothing
        if amount <= 0.0 or planner_state is None or planner_state.previous_action is None:
            return sequence
        previous = planner_state.previous_action.to(sequence.device)
        if previous.shape != sequence[:, 0].shape:
            return sequence
        current_skills, current_parameters = self.action_adapter.unpack(sequence[:, 0])
        old_skills, old_parameters = self.action_adapter.unpack(previous)
        current_normalized = self.action_adapter.normalize_parameters(
            current_skills, current_parameters
        )
        old_normalized = self.action_adapter.normalize_parameters(old_skills, old_parameters)
        compatible = (current_skills == old_skills).unsqueeze(-1)
        blended = torch.where(
            compatible,
            (1.0 - amount) * current_normalized + amount * old_normalized,
            current_normalized,
        )
        first = self.action_adapter.pack(
            current_skills,
            self.action_adapter.denormalize_parameters(current_skills, blended),
        )
        result = sequence.clone()
        result[:, 0] = first
        return result

    @torch.no_grad()
    def plan(
        self,
        states: torch.Tensor,
        planner_state: Optional[MPCPlannerState] = None,
        fixed_action_sequence: Optional[torch.Tensor] = None,
        fixed_robot_mask: Optional[torch.Tensor] = None,
    ) -> MPCPlanResult:
        """Plan from encoded global states ``[B,D]`` and return one action per row."""

        if states.ndim != 2 or states.shape[1] != self.world_model.schema.state_dim:
            raise ValueError(
                f"states must have shape [B,{self.world_model.schema.state_dim}], got {tuple(states.shape)}"
            )
        if not bool(torch.isfinite(states).all().item()):
            raise ValueError("MPC initial states contain NaN or infinite values")
        if (fixed_action_sequence is None) != (fixed_robot_mask is None):
            raise ValueError(
                "fixed_action_sequence and fixed_robot_mask must be provided together"
            )
        if fixed_action_sequence is not None:
            expected = (states.shape[0], self.config.horizon, self.action_adapter.action_dim)
            if tuple(fixed_action_sequence.shape) != expected:
                raise ValueError(
                    f"fixed_action_sequence must have shape {expected}, got "
                    f"{tuple(fixed_action_sequence.shape)}"
                )
            if tuple(fixed_robot_mask.shape) != (self.num_robots,):
                raise ValueError(
                    f"fixed_robot_mask must have shape ({self.num_robots},)"
                )
            self.action_adapter.assert_within_bounds(fixed_action_sequence)
        started = time.perf_counter()
        probabilities, means, stds = self._initialize(states, planner_state)
        batch = states.shape[0]
        best_objective = torch.full(
            (batch,),
            self.config.invalid_objective,
            dtype=states.dtype,
            device=states.device,
        )
        best_sequence = None
        last_elites = None
        last_actions = last_objectives = last_valid = None
        convergence_lists = {
            "best_objective": [],
            "mean_elite_objective": [],
            "elite_objective_std": [],
            "skill_entropy": [],
            "mean_parameter_std": [],
            "first_step_skill_probabilities": [],
            "iteration_time_seconds": [],
        }
        batch_rows = torch.arange(batch, device=states.device)
        sampling_seconds = rollout_seconds = update_seconds = 0.0

        for _ in range(self.config.num_iterations):
            self._synchronize(states.device)
            iteration_started = time.perf_counter()
            sample_started = iteration_started
            actions, skills, normalized = self._sample(probabilities, means, stds)
            actions = self._apply_fixed_robot_actions(
                actions, fixed_action_sequence, fixed_robot_mask
            )
            skills, parameters = self.action_adapter.unpack(actions)
            normalized = self.action_adapter.normalize_parameters(skills, parameters)
            self._synchronize(states.device)
            evaluation_started = time.perf_counter()
            sampling_seconds += evaluation_started - sample_started
            objectives, valid = self._evaluate_candidates(states, actions)
            self._synchronize(states.device)
            update_started = time.perf_counter()
            rollout_seconds += update_started - evaluation_started
            elite_objectives, elite_indices = torch.topk(
                objectives, self.config.num_elites, dim=1
            )
            probabilities, means, stds = self._elite_update(
                probabilities, means, stds, skills, normalized, elite_indices
            )
            iteration_best, best_indices = objectives.max(dim=1)
            selected = actions[batch_rows, best_indices]
            improved = iteration_best > best_objective
            if best_sequence is None:
                best_sequence = selected
            else:
                best_sequence = torch.where(improved[:, None, None], selected, best_sequence)
            best_objective = torch.maximum(best_objective, iteration_best)
            entropy = -(
                probabilities.clamp(min=torch.finfo(states.dtype).eps)
                * probabilities.clamp(min=torch.finfo(states.dtype).eps).log()
            ).sum(-1).mean(dim=(1, 2))
            convergence_lists["best_objective"].append(iteration_best)
            convergence_lists["mean_elite_objective"].append(elite_objectives.mean(1))
            convergence_lists["elite_objective_std"].append(
                elite_objectives.std(1, unbiased=False)
            )
            convergence_lists["skill_entropy"].append(entropy)
            convergence_lists["mean_parameter_std"].append(stds.mean(dim=(1, 2, 3, 4)))
            convergence_lists["first_step_skill_probabilities"].append(
                probabilities[:, 0]
            )
            self._synchronize(states.device)
            iteration_finished = time.perf_counter()
            convergence_lists["iteration_time_seconds"].append(
                torch.full(
                    (batch,),
                    iteration_finished - iteration_started,
                    dtype=states.dtype,
                    device=states.device,
                )
            )
            update_seconds += iteration_finished - update_started
            last_elites = elite_objectives
            last_actions, last_objectives, last_valid = actions, objectives, valid

        if self.config.execute_mean_action:
            selected_sequence = self._mode_sequence(probabilities, means)
        else:
            selected_sequence = best_sequence
        selected_sequence = self._smooth_first_action(selected_sequence, planner_state)
        selected_sequence = self._apply_fixed_robot_actions(
            selected_sequence, fixed_action_sequence, fixed_robot_mask
        )
        all_invalid = ~last_valid.any(dim=1)
        previous_action = (
            None
            if planner_state is None or planner_state.previous_action is None
            else planner_state.previous_action.to(states.device)
        )
        fallback_action = self.fallback_policy(states, previous_action)
        fallback_sequence = fallback_action[:, None, :].expand(
            -1, self.config.horizon, -1
        )
        fallback_sequence = self._apply_fixed_robot_actions(
            fallback_sequence, fixed_action_sequence, fixed_robot_mask
        )
        selected_sequence = torch.where(
            all_invalid[:, None, None], fallback_sequence, selected_sequence
        )

        rollout_method = (
            self.world_model.rollout_members
            if self.config.ensemble_objective in ("mean_minus_std", "minimum")
            and hasattr(self.world_model, "rollout_members")
            else self.world_model.rollout
        )
        selected_rollout = rollout_method(
            states,
            selected_sequence[:, None],
            deterministic=self.config.deterministic_world_model,
            stop_on_done=False,
        )
        selected_score = self.objective.evaluate(
            states, selected_sequence[:, None], selected_rollout
        )
        state_uncertainty = selected_rollout.get(
            "state_uncertainty",
            torch.zeros(
                batch,
                1,
                self.config.horizon,
                dtype=states.dtype,
                device=states.device,
            ),
        )[:, 0]
        excessive_uncertainty = torch.zeros_like(all_invalid)
        if self.config.uncertainty_fallback_threshold is not None:
            excessive_uncertainty = (
                state_uncertainty.max(dim=-1).values
                > self.config.uncertainty_fallback_threshold
            )
        fallback_used = all_invalid | excessive_uncertainty | ~selected_score.valid[:, 0]
        if bool((fallback_used & ~all_invalid).any().item()):
            selected_sequence = torch.where(
                fallback_used[:, None, None], fallback_sequence, selected_sequence
            )
            selected_rollout = rollout_method(
                states,
                selected_sequence[:, None],
                deterministic=True,
                stop_on_done=False,
            )
            selected_score = self.objective.evaluate(
                states, selected_sequence[:, None], selected_rollout
            )
            state_uncertainty = selected_rollout.get(
                "state_uncertainty", torch.zeros_like(state_uncertainty[:, None])
            )[:, 0]

        physical_means, physical_stds = self._physical_distribution(means, stds)

        diagnostic_count = min(
            int(self.config.max_candidate_diagnostics), self.config.num_candidates
        )
        candidate_diagnostics = {}
        if diagnostic_count > 0:
            indices = torch.linspace(
                0,
                self.config.num_candidates - 1,
                diagnostic_count,
                device=states.device,
            ).long()
            diagnostic_actions = last_actions.index_select(1, indices)
            diagnostic_rollout = rollout_method(
                states,
                diagnostic_actions,
                deterministic=True,
                stop_on_done=False,
            )
            diagnostic_score = self.objective.evaluate(
                states, diagnostic_actions, diagnostic_rollout
            )
            candidate_diagnostics = {
                "action_sequences": diagnostic_actions,
                "objectives": last_objectives.index_select(1, indices),
                "valid": last_valid.index_select(1, indices),
                "final_states": diagnostic_rollout["predicted_states"][..., -1, :],
                "sample_indices": indices,
                "predicted_discounted_reward_return": diagnostic_score.diagnostics[
                    "predicted_discounted_reward_return"
                ],
                "terminal_state_value": diagnostic_score.diagnostics[
                    "terminal_state_value"
                ],
                "terminal_value_contribution": diagnostic_score.diagnostics[
                    "terminal_value_contribution"
                ],
                "terminal_state_uncertainty": diagnostic_score.diagnostics[
                    "terminal_state_uncertainty"
                ],
            }

        self._synchronize(states.device)
        elapsed = time.perf_counter() - started
        convergence = {
            key: torch.stack(values, dim=1)
            for key, values in convergence_lists.items()
        }
        convergence["sampling_time_seconds"] = torch.as_tensor(
            sampling_seconds, dtype=states.dtype, device=states.device
        )
        convergence["rollout_time_seconds"] = torch.as_tensor(
            rollout_seconds, dtype=states.dtype, device=states.device
        )
        convergence["update_time_seconds"] = torch.as_tensor(
            update_seconds, dtype=states.dtype, device=states.device
        )
        rollout_states = selected_rollout["predicted_states"][:, 0]
        rollout_rewards = selected_rollout["predicted_rewards"][:, 0]
        rollout_dones = selected_rollout["predicted_done_probabilities"][:, 0]
        return_uncertainty = selected_score.return_uncertainty[:, 0]
        safe_prediction = (
            selected_score.valid[:, 0]
            & torch.isfinite(rollout_states).flatten(1).all(-1)
            & torch.isfinite(rollout_rewards).all(-1)
            & torch.isfinite(rollout_dones).all(-1)
        )
        repeated_initial = states[:, None, :].expand(
            -1, self.config.horizon + 1, -1
        )
        rollout_states = torch.where(
            safe_prediction[:, None, None], rollout_states, repeated_initial
        )
        rollout_rewards = torch.where(
            safe_prediction[:, None], rollout_rewards, torch.zeros_like(rollout_rewards)
        )
        rollout_dones = torch.where(
            safe_prediction[:, None], rollout_dones, torch.ones_like(rollout_dones)
        )
        state_uncertainty = torch.where(
            safe_prediction[:, None],
            torch.nan_to_num(state_uncertainty),
            torch.zeros_like(state_uncertainty),
        )
        reward_uncertainty = selected_rollout.get(
            "reward_uncertainty", torch.zeros_like(rollout_rewards[:, None])
        )[:, 0]
        reward_uncertainty = torch.where(
            safe_prediction[:, None],
            torch.nan_to_num(reward_uncertainty),
            torch.zeros_like(reward_uncertainty),
        )
        return_uncertainty = torch.where(
            safe_prediction,
            torch.nan_to_num(return_uncertainty),
            torch.zeros_like(return_uncertainty),
        )
        plan_uncertainty = state_uncertainty.max(dim=-1).values
        fallback_used = fallback_used | ~safe_prediction
        next_state = MPCPlannerState(
            probabilities.detach(),
            means.detach(),
            stds.detach(),
            (~fallback_used).detach(),
            selected_sequence[:, 0].detach(),
            states.detach(),
            plan_uncertainty.detach(),
        )
        component_values = {
            key: torch.nan_to_num(value[:, 0])
            for key, value in selected_score.components.items()
        }
        if bool((~safe_prediction).any().item()):
            for key in component_values:
                component_values[key] = torch.where(
                    safe_prediction,
                    component_values[key],
                    torch.zeros_like(component_values[key]),
                )
            component_values["predicted_reward_return"] = torch.where(
                safe_prediction,
                component_values["predicted_reward_return"],
                torch.full_like(
                    component_values["predicted_reward_return"],
                    self.config.invalid_objective,
                ),
            )
        final_objective = torch.where(
            safe_prediction,
            selected_score.total[:, 0],
            torch.full_like(selected_score.total[:, 0], self.config.invalid_objective),
        )
        diagnostic_values = {
            key: torch.nan_to_num(value[:, 0])
            for key, value in selected_score.diagnostics.items()
        }
        if self.objective.terminal_value is None:
            predicted_state_values = torch.zeros(
                rollout_states.shape[:2], dtype=rollout_states.dtype,
                device=rollout_states.device,
            )
        else:
            predicted_state_values = self.objective.terminal_value(rollout_states)
        return MPCPlanResult(
            first_joint_action=selected_sequence[:, 0],
            best_action_sequence=selected_sequence,
            best_objective=final_objective,
            predicted_states=rollout_states,
            predicted_rewards=rollout_rewards,
            predicted_done_probabilities=rollout_dones,
            objective_components=component_values,
            uncertainty={
                "state": state_uncertainty,
                "reward": reward_uncertainty,
                "return": return_uncertainty,
                "max_state": plan_uncertainty,
            },
            final_skill_probabilities=probabilities,
            final_parameter_means=physical_means,
            final_parameter_stds=physical_stds,
            elite_objectives=last_elites,
            planning_time_seconds=elapsed,
            planner_state=next_state,
            fallback_used=fallback_used,
            predicted_discounted_reward_return=diagnostic_values[
                "predicted_discounted_reward_return"
            ],
            terminal_state_value=diagnostic_values["terminal_state_value"],
            discounted_terminal_value=diagnostic_values["discounted_terminal_value"],
            terminal_value_contribution=diagnostic_values[
                "terminal_value_contribution"
            ],
            terminal_state_uncertainty=diagnostic_values[
                "terminal_state_uncertainty"
            ],
            terminal_value_clipped=diagnostic_values["terminal_value_clipped"],
            predicted_state_values=predicted_state_values,
            terminal_value_coefficient=float(self.config.terminal_value_coefficient),
            convergence=convergence,
            candidate_diagnostics=candidate_diagnostics,
        )
