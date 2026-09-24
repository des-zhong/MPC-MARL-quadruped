"""Batched risk-aware CEM planning for joint hybrid skill actions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from .config import MPCConfig
from .objective import MPCObjective, MPCObjectiveResult
from .planner_state import MPCPlannerState


@dataclass
class MPCPlanResult:
    """Result of choosing the highest-return sampled action sequence."""

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
    predicted_discounted_reward_return: torch.Tensor
    convergence: Dict[str, torch.Tensor] = field(default_factory=dict)
    candidate_diagnostics: Dict[str, torch.Tensor] = field(default_factory=dict)


class HybridCEMMPC:
    """Optimize bounded action sequences using the configured MPC objective."""

    num_skills = 3
    parameter_dim = 3

    def __init__(
        self,
        world_model,
        state_adapter=None,
        action_adapter=None,
        objective: Optional[MPCObjective] = None,
        config: Optional[MPCConfig] = None,
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
        )
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
        probabilities = torch.full(
            shape, 1.0 / self.num_skills, dtype=dtype, device=device
        )
        parameter_shape = shape + (self.parameter_dim,)
        means = torch.zeros(parameter_shape, dtype=dtype, device=device)
        stds = torch.full(
            parameter_shape,
            self.config.initial_parameter_std_fraction,
            dtype=dtype,
            device=device,
        )
        return probabilities, means, stds

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
        valid = previous.valid
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
        skills = (logits[:, None] - torch.log(-torch.log(uniform))).argmax(dim=-1)

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
        """Insert externally forecast robot actions into every candidate."""

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
        if mask.ndim == 1:
            shaped[..., mask, :] = fixed[..., mask, :]
        else:
            expanded = mask[:, None, None, :, None] if actions.ndim == 4 else mask[:, None, :, None]
            shaped = torch.where(expanded, fixed, shaped)
        return shaped.flatten(-2)

    def _evaluate_candidates(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        return_best: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Evaluate candidates, optionally retaining the best candidate data.

        The planner used to evaluate all candidates in chunks and then run the
        selected sequence again with a single candidate.  Apart from doing
        unnecessary model work, that made deterministic world-model outputs
        depend on the candidate batch shape (especially near an OOD threshold).
        When ``return_best`` is true, this method keeps the selected rollout and
        objective from the original chunk evaluation so final validation and
        the returned plan are based on exactly the values used for selection.
        """
        count = actions.shape[1]
        chunk = self.config.candidate_batch_size or count
        returns = []
        validity = []
        finite_returns = []
        batch = states.shape[0]
        batch_rows = torch.arange(batch, device=states.device)
        iteration_best = torch.full(
            (batch,), float("-inf"), dtype=states.dtype, device=states.device
        )
        has_strict_candidate = torch.zeros(
            batch, dtype=torch.bool, device=states.device
        )
        best_rollout = None
        best_score = None
        best_action = None
        for start in range(0, count, chunk):
            selected = actions[:, start : start + chunk]
            rollout = self.world_model.rollout(
                states,
                selected,
                deterministic=self.config.deterministic_world_model,
                stop_on_done=False,
                action_transform=self.objective.resolve_imagined_action,
            )
            score = self.objective.evaluate(states, selected, rollout)
            returns.append(score.total)
            validity.append(score.valid)
            finite = torch.where(
                score.diagnostics["numerically_valid"],
                score.diagnostics["raw_total"],
                torch.full_like(score.total, float("-inf")),
            )
            finite_returns.append(finite)

            if return_best:
                chunk_strict_available = torch.isfinite(score.total).any(dim=1)
                chunk_strict_best, chunk_strict_indices = score.total.max(dim=1)
                chunk_finite_best, chunk_finite_indices = finite.max(dim=1)
                chunk_best = torch.where(
                    chunk_strict_available,
                    chunk_strict_best,
                    chunk_finite_best,
                )
                chunk_indices = torch.where(
                    chunk_strict_available,
                    chunk_strict_indices,
                    chunk_finite_indices,
                )
                improved = (
                    chunk_strict_available
                    & (~has_strict_candidate | (chunk_best > iteration_best))
                ) | (
                    ~chunk_strict_available
                    & ~has_strict_candidate
                    & (chunk_best > iteration_best)
                )

                # Keep only one candidate's data per batch row.  This avoids
                # retaining all rollout tensors while still eliminating the
                # shape-changing final rerun in ``plan``.
                def select_candidate(value):
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.ndim < 2
                        or value.shape[:2] != selected.shape[:2]
                    ):
                        raise ValueError(
                            "MPC candidate result does not have leading [B,C] "
                            f"dimensions: {getattr(value, 'shape', None)}"
                        )
                    return value[batch_rows, chunk_indices].unsqueeze(1)

                selected_rollout = {
                    key: select_candidate(value)
                    for key, value in rollout.items()
                    # Per-ensemble-member tensors use [M,B,C,...] and are
                    # already summarized by ``score``.  The returned plan only
                    # needs standard batch-first rollout fields.
                    if isinstance(value, torch.Tensor)
                    and value.ndim >= 2
                    and value.shape[:2] == selected.shape[:2]
                }
                selected_score = MPCObjectiveResult(
                    total=select_candidate(score.total),
                    components={
                        key: select_candidate(value)
                        for key, value in score.components.items()
                    },
                    valid=select_candidate(score.valid),
                    return_uncertainty=select_candidate(score.return_uncertainty),
                    diagnostics={
                        key: select_candidate(value)
                        for key, value in score.diagnostics.items()
                    },
                )
                selected_action = selected[batch_rows, chunk_indices]
                if best_rollout is None:
                    best_rollout = selected_rollout
                    best_score = selected_score
                    best_action = selected_action
                else:
                    row_mask = improved

                    def choose_rows(old, new):
                        shape = (batch,) + (1,) * (old.ndim - 1)
                        return torch.where(row_mask.reshape(shape), new, old)

                    best_rollout = {
                        key: choose_rows(best_rollout[key], value)
                        for key, value in selected_rollout.items()
                    }
                    best_score = MPCObjectiveResult(
                        total=choose_rows(best_score.total, selected_score.total),
                        components={
                            key: choose_rows(best_score.components[key], value)
                            for key, value in selected_score.components.items()
                        },
                        valid=choose_rows(best_score.valid, selected_score.valid),
                        return_uncertainty=choose_rows(
                            best_score.return_uncertainty,
                            selected_score.return_uncertainty,
                        ),
                        diagnostics={
                            key: choose_rows(best_score.diagnostics[key], value)
                            for key, value in selected_score.diagnostics.items()
                        },
                    )
                    best_action = choose_rows(best_action, selected_action)
                iteration_best = torch.where(improved, chunk_best, iteration_best)
                has_strict_candidate |= chunk_strict_available
        result = (
            torch.cat(returns, dim=1),
            torch.cat(validity, dim=1),
            torch.cat(finite_returns, dim=1),
        )
        if not return_best:
            return result
        return result + (best_action, best_rollout, best_score)

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

        floor = self.config.min_skill_probability
        empirical_probs = one_hot.mean(dim=1)
        empirical_probs = floor + (1.0 - self.num_skills * floor) * empirical_probs
        alpha = self.config.categorical_smoothing
        new_probabilities = alpha * probabilities + (1.0 - alpha) * empirical_probs
        new_probabilities = new_probabilities.clamp(min=floor)
        new_probabilities /= new_probabilities.sum(dim=-1, keepdim=True)

        weights = one_hot.unsqueeze(-1)
        counts = weights.sum(dim=1)
        conditional_mean = (
            weights * elite_parameters.unsqueeze(-2)
        ).sum(dim=1) / counts.clamp(min=1.0)
        centered = elite_parameters.unsqueeze(-2) - conditional_mean.unsqueeze(1)
        conditional_std = (
            (weights * centered.square()).sum(dim=1) / counts.clamp(min=1.0)
        ).clamp(min=0.0).sqrt()
        observed = counts > 0
        conditional_mean = torch.where(observed, conditional_mean, means)
        conditional_std = torch.where(observed, conditional_std, stds)

        alpha = self.config.continuous_smoothing
        new_means = alpha * means + (1.0 - alpha) * conditional_mean
        new_stds = alpha * stds + (1.0 - alpha) * conditional_std
        new_stds = new_stds.clamp(
            min=self.config.min_parameter_std_fraction,
            max=self.config.max_parameter_std_fraction,
        )
        return new_probabilities, new_means.clamp(-1.0, 1.0), new_stds

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
        physical_std = 0.5 * stds * (highs - lows) * masks
        return physical_mean, physical_std

    def _validate_fixed_actions(
        self,
        states: torch.Tensor,
        fixed_action_sequence: Optional[torch.Tensor],
        fixed_robot_mask: Optional[torch.Tensor],
    ) -> None:
        if (fixed_action_sequence is None) != (fixed_robot_mask is None):
            raise ValueError(
                "fixed_action_sequence and fixed_robot_mask must be provided together"
            )
        if fixed_action_sequence is None:
            return
        expected = (
            states.shape[0],
            self.config.horizon,
            self.action_adapter.action_dim,
        )
        if tuple(fixed_action_sequence.shape) != expected:
            raise ValueError(
                f"fixed_action_sequence must have shape {expected}, got "
                f"{tuple(fixed_action_sequence.shape)}"
            )
        if tuple(fixed_robot_mask.shape) not in ((self.num_robots,), (states.shape[0], self.num_robots)):
            raise ValueError("fixed_robot_mask must have shape [robots] or [batch, robots]")
        self.action_adapter.assert_within_bounds(fixed_action_sequence)

    @torch.no_grad()
    def plan(
        self,
        states: torch.Tensor,
        planner_state: Optional[MPCPlannerState] = None,
        fixed_action_sequence: Optional[torch.Tensor] = None,
        fixed_robot_mask: Optional[torch.Tensor] = None,
        reference_action_sequence: Optional[torch.Tensor] = None,
    ) -> MPCPlanResult:
        """Return the first action of the highest predicted-return sequence."""

        if states.ndim != 2 or states.shape[1] != self.world_model.schema.state_dim:
            raise ValueError(
                f"states must have shape [B,{self.world_model.schema.state_dim}], got "
                f"{tuple(states.shape)}"
            )
        if not bool(torch.isfinite(states).all().item()):
            raise ValueError("MPC initial states contain NaN or infinite values")
        self._validate_fixed_actions(states, fixed_action_sequence, fixed_robot_mask)

        started = time.perf_counter()
        probabilities, means, stds = self._initialize(states, planner_state)
        batch = states.shape[0]
        if reference_action_sequence is not None:
            if reference_action_sequence.shape != (batch, self.config.horizon, self.action_adapter.action_dim):
                raise ValueError("Invalid reference action sequence shape")
            reference_action_sequence = self._apply_fixed_robot_actions(reference_action_sequence, fixed_action_sequence, fixed_robot_mask)
            self.action_adapter.assert_within_bounds(reference_action_sequence)
            reference_skills, reference_commands = self.action_adapter.unpack(reference_action_sequence)
            reference_normalized = self.action_adapter.normalize_parameters(reference_skills, reference_commands)
            prior = torch.nn.functional.one_hot(reference_skills.long(), self.num_skills).to(probabilities.dtype)
            # Preserve exploration while centering CEM on an executable student proposal.
            probabilities = .5*probabilities + .5*prior
            means = torch.where(prior[..., None].bool(), reference_normalized[..., None, :], means)
        best_return = torch.full(
            (batch,), float("-inf"), dtype=states.dtype, device=states.device
        )
        has_strict_candidate = torch.zeros(
            batch, dtype=torch.bool, device=states.device
        )
        best_sequence = None
        best_rollout = None
        best_score = None
        last_elites = None
        last_actions = last_returns = last_valid = None
        convergence_lists = {
            "best_objective": [],
            "mean_elite_objective": [],
            "elite_objective_std": [],
            "skill_entropy": [],
            "mean_parameter_std": [],
            "first_step_skill_probabilities": [],
            "iteration_time_seconds": [],
        }

        for _ in range(self.config.num_iterations):
            self._synchronize(states.device)
            iteration_started = time.perf_counter()
            actions, skills, normalized = self._sample(probabilities, means, stds)
            if reference_action_sequence is not None:
                actions[:, 0] = reference_action_sequence
                skills[:, 0] = reference_skills
                normalized[:, 0] = reference_normalized
            actions = self._apply_fixed_robot_actions(
                actions, fixed_action_sequence, fixed_robot_mask
            )
            skills, parameters = self.action_adapter.unpack(actions)
            normalized = self.action_adapter.normalize_parameters(skills, parameters)
            (
                strict_returns,
                valid,
                finite_returns,
                selected,
                selected_rollout,
                selected_score,
            ) = self._evaluate_candidates(
                states, actions, return_best=True
            )
            strict_available = torch.isfinite(strict_returns).any(dim=1)
            recoverable = (~strict_available) & torch.isfinite(finite_returns).any(dim=1)
            if not self.config.relax_ood_when_all_candidates_rejected:
                recoverable.zero_()
            returns = torch.where(
                recoverable[:, None], finite_returns, strict_returns
            )
            elite_returns, elite_indices = torch.topk(
                returns, self.config.num_elites, dim=1
            )
            probabilities, means, stds = self._elite_update(
                probabilities, means, stds, skills, normalized, elite_indices
            )

            strict_best = strict_returns.max(dim=1).values
            relaxed_best = finite_returns.max(dim=1).values
            iteration_best = torch.where(strict_available, strict_best, relaxed_best)
            improved = (
                strict_available
                & (~has_strict_candidate | (iteration_best > best_return))
            ) | (
                ~strict_available
                & ~has_strict_candidate
                & (iteration_best > best_return)
            )
            if best_sequence is None:
                best_sequence = selected
                best_rollout = selected_rollout
                best_score = selected_score
            else:
                best_sequence = torch.where(
                    improved[:, None, None], selected, best_sequence
                )

                def choose_rows(old, new):
                    shape = (batch,) + (1,) * (old.ndim - 1)
                    return torch.where(improved.reshape(shape), new, old)

                best_rollout = {
                    key: choose_rows(best_rollout[key], value)
                    for key, value in selected_rollout.items()
                }
                best_score = MPCObjectiveResult(
                    total=choose_rows(best_score.total, selected_score.total),
                    components={
                        key: choose_rows(best_score.components[key], value)
                        for key, value in selected_score.components.items()
                    },
                    valid=choose_rows(best_score.valid, selected_score.valid),
                    return_uncertainty=choose_rows(
                        best_score.return_uncertainty,
                        selected_score.return_uncertainty,
                    ),
                    diagnostics={
                        key: choose_rows(best_score.diagnostics[key], value)
                        for key, value in selected_score.diagnostics.items()
                    },
                )
            best_return = torch.where(improved, iteration_best, best_return)
            has_strict_candidate |= strict_available

            safe_probabilities = probabilities.clamp(
                min=torch.finfo(states.dtype).eps
            )
            entropy = -(safe_probabilities * safe_probabilities.log()).sum(-1)
            convergence_lists["best_objective"].append(iteration_best)
            convergence_lists["mean_elite_objective"].append(elite_returns.mean(1))
            convergence_lists["elite_objective_std"].append(
                elite_returns.std(1, unbiased=False)
            )
            convergence_lists["skill_entropy"].append(entropy.mean(dim=(1, 2)))
            convergence_lists["mean_parameter_std"].append(
                stds.mean(dim=(1, 2, 3, 4))
            )
            convergence_lists["first_step_skill_probabilities"].append(
                probabilities[:, 0]
            )
            self._synchronize(states.device)
            convergence_lists["iteration_time_seconds"].append(
                torch.full(
                    (batch,),
                    time.perf_counter() - iteration_started,
                    dtype=states.dtype,
                    device=states.device,
                )
            )
            last_elites = elite_returns
            last_actions, last_returns, last_valid = actions, returns, valid

        failed = ~torch.isfinite(best_return)
        if bool(failed.any().item()):
            rows = failed.nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "MPC could not find a finite predicted-reward return for batch rows "
                f"{rows}. This indicates genuinely non-finite model/objective output, "
                "or all candidates were rejected while OOD relaxation was disabled."
            )

        ood_fallback_used = ~has_strict_candidate

        selected_sequence = best_sequence
        selected_rollout = best_rollout
        selected_score = best_score
        selected_valid = selected_score.valid[:, 0] | (
            ood_fallback_used
            & selected_score.diagnostics["numerically_valid"][:, 0]
        )
        if not bool(selected_valid.all().item()):
            rows = (~selected_valid).nonzero(as_tuple=False).flatten().tolist()
            numerically_invalid = (
                ~selected_score.diagnostics["numerically_valid"][:, 0]
            ).nonzero(as_tuple=False).flatten().tolist()
            state_ood = selected_score.diagnostics[
                "state_ood_rejected"
            ][:, 0].nonzero(as_tuple=False).flatten().tolist()
            return_ood = selected_score.diagnostics[
                "return_ood_rejected"
            ][:, 0].nonzero(as_tuple=False).flatten().tolist()
            value_ood = selected_score.diagnostics[
                "terminal_value_ood_rejected"
            ][:, 0].nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "Selected MPC candidate failed its original evaluation for batch "
                f"rows {rows}; numerically_invalid={numerically_invalid}, "
                f"state_ood_rejected={state_ood}, "
                f"return_ood_rejected={return_ood}, "
                f"terminal_value_ood_rejected={value_ood}"
            )

        diagnostic_count = min(
            int(self.config.max_candidate_diagnostics), self.config.num_candidates
        )
        candidate_diagnostics = {}
        if diagnostic_count:
            indices = torch.linspace(
                0,
                self.config.num_candidates - 1,
                diagnostic_count,
                device=states.device,
            ).long()
            diagnostic_actions = last_actions.index_select(1, indices)
            diagnostic_rollout = self.world_model.rollout(
                states,
                diagnostic_actions,
                deterministic=True,
                stop_on_done=False,
                action_transform=self.objective.resolve_imagined_action,
            )
            diagnostic_score = self.objective.evaluate(
                states, diagnostic_actions, diagnostic_rollout
            )
            candidate_diagnostics = {
                "action_sequences": diagnostic_actions,
                "executed_action_sequences": diagnostic_rollout.get(
                    "executed_actions", diagnostic_actions
                ),
                "objectives": last_returns.index_select(1, indices),
                "valid": last_valid.index_select(1, indices),
                "final_states": diagnostic_rollout["predicted_states"][..., -1, :],
                "learned_reward_returns": diagnostic_score.diagnostics[
                    "learned_discounted_reward_return"
                ],
                "analytical_reward_returns": diagnostic_score.diagnostics[
                    "analytical_discounted_reward_return"
                ],
                "max_state_uncertainty": diagnostic_score.diagnostics[
                    "max_state_uncertainty"
                ],
                "sample_indices": indices,
            }

        self._synchronize(states.device)
        elapsed = time.perf_counter() - started
        convergence = {
            key: torch.stack(values, dim=1)
            for key, values in convergence_lists.items()
        }
        selected_sequence = selected_rollout.get(
            "executed_actions", selected_sequence[:, None]
        )[:, 0]
        rollout_states = selected_rollout["predicted_states"][:, 0]
        rollout_rewards = selected_rollout["predicted_rewards"][:, 0]
        rollout_dones = selected_rollout["predicted_done_probabilities"][:, 0]
        state_uncertainty = selected_rollout.get(
            "state_uncertainty", torch.zeros_like(rollout_rewards[:, None])
        )[:, 0]
        reward_uncertainty = selected_rollout.get(
            "reward_uncertainty", torch.zeros_like(rollout_rewards[:, None])
        )[:, 0]
        return_uncertainty = selected_score.return_uncertainty[:, 0]
        plan_uncertainty = state_uncertainty.max(dim=-1).values

        next_state = MPCPlannerState(
            probabilities.detach(),
            means.detach(),
            stds.detach(),
            torch.ones(batch, dtype=torch.bool, device=states.device),
        )
        physical_means, physical_stds = self._physical_distribution(means, stds)
        return MPCPlanResult(
            first_joint_action=selected_sequence[:, 0],
            best_action_sequence=selected_sequence,
            best_objective=best_return,
            predicted_states=rollout_states,
            predicted_rewards=rollout_rewards,
            predicted_done_probabilities=rollout_dones,
            objective_components={
                name: value[:, 0]
                for name, value in selected_score.components.items()
            },
            uncertainty={
                "state": state_uncertainty,
                "reward": reward_uncertainty,
                "return": return_uncertainty,
                "max_state": plan_uncertainty,
                "ood_fallback_used": ood_fallback_used,
            },
            final_skill_probabilities=probabilities,
            final_parameter_means=physical_means,
            final_parameter_stds=physical_stds,
            elite_objectives=last_elites,
            planning_time_seconds=elapsed,
            planner_state=next_state,
            predicted_discounted_reward_return=selected_score.diagnostics[
                "predicted_discounted_reward_return"
            ][:, 0],
            convergence=convergence,
            candidate_diagnostics=candidate_diagnostics,
        )
