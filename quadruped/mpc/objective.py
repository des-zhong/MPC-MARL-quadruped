"""Risk-aware objective for model-predictive control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional

import torch

from .config import MPCConfig
from .analytical_reward import AnalyticalRewardReconstructor, ImaginedSkillResolver


@dataclass
class MPCObjectiveResult:
    """Per-candidate predicted return and basic validity diagnostics."""

    total: torch.Tensor
    components: Dict[str, torch.Tensor]
    valid: torch.Tensor
    return_uncertainty: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


class MPCObjective:
    """Combine predicted return, continuation value, risk, and control costs."""

    COMPONENT_NAMES = (
        "predicted_reward_return",
        "analytical_reward_return",
        "terminal_value",
        "state_uncertainty_penalty",
        "return_uncertainty_penalty",
        "invalid_skill_penalty",
        "skill_switch_penalty",
        "command_change_penalty",
    )

    def __init__(
        self,
        schema,
        action_adapter,
        event_names,
        config: MPCConfig,
        terminal_value: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        controlled_robot_count: Optional[int] = None,
        reward_scales: Optional[Mapping[str, float]] = None,
    ):
        self.schema = schema
        self.action_adapter = action_adapter
        self.event_names = tuple(event_names)
        self.config = config
        self.terminal_value = terminal_value
        self.controlled_robot_count = int(
            action_adapter.num_robots
            if controlled_robot_count is None
            else controlled_robot_count
        )
        self.skill_resolver = ImaginedSkillResolver(
            schema, action_adapter, config, self.controlled_robot_count
        )
        self.analytical_reward = AnalyticalRewardReconstructor(
            schema, action_adapter, self.event_names, config,
            self.controlled_robot_count,
            reward_scales=reward_scales,
        )
        needs_value = config.objective_mode in {
            "terminal_value_only",
            "reward_plus_terminal_value",
        }
        if needs_value and terminal_value is None:
            raise ValueError(
                f"mpc.objective_mode={config.objective_mode!r} requires a terminal value"
            )

    def _survival(self, done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.terminal_handling == "probability_weighted":
            not_done = (1.0 - done.clamp(0.0, 1.0))
        else:
            not_done = (done < self.config.termination_threshold).to(done.dtype)
        active = torch.cat(
            (
                torch.ones_like(not_done[..., :1]),
                torch.cumprod(not_done[..., :-1], dim=-1),
            ),
            dim=-1,
        )
        return active, torch.prod(not_done, dim=-1)

    def _invalid_skills(
        self, predicted_states: torch.Tensor, action_sequences: torch.Tensor
    ) -> torch.Tensor:
        before = predicted_states[..., :-1, :]
        field_scale = before[..., self.schema.slice("field.geometry")][..., :2]
        field_scale = field_scale.abs().clamp(min=1e-6)
        ball_xy = before[..., self.schema.slice("ball.position")][..., :2]
        skills, _ = self.action_adapter.unpack(action_sequences)
        invalid = torch.zeros_like(skills[..., 0], dtype=action_sequences.dtype)
        for robot in range(self.controlled_robot_count):
            position = before[
                ..., self.schema.slice(f"robot_{robot}.position")
            ][..., :2]
            yaw = before[..., self.schema.slice(f"robot_{robot}.yaw_sin_cos")]
            delta = (ball_xy - position) * field_scale
            sin_yaw, cos_yaw = yaw[..., 0], yaw[..., 1]
            local_x = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
            local_y = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            can_dribble = distance <= self.config.dribble_affordance_distance_m
            can_shoot = (
                (distance <= self.config.shoot_affordance_distance_m)
                & (local_x >= self.config.shoot_min_forward_m)
                & (local_y.abs() <= self.config.shoot_lateral_reach_m)
            )
            requested = skills[..., robot]
            invalid += (
                ((requested == 1) & ~can_dribble)
                | ((requested == 2) & ~can_shoot)
            ).to(invalid.dtype)
        return invalid

    def resolve_imagined_action(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Resolve requested actions exactly where each imagined step begins."""

        return self.skill_resolver(states, actions)

    def _action_costs(
        self, initial_states: torch.Tensor, action_sequences: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        skills, parameters = self.action_adapter.unpack(action_sequences)
        skills = skills[..., : self.controlled_robot_count]
        parameters = parameters[..., : self.controlled_robot_count, :]
        normalized = self.action_adapter.normalize_parameters(skills, parameters)
        current_skills = torch.stack(
            [
                initial_states[
                    :, self.schema.slice(f"robot_{robot}.skill_one_hot")
                ].argmax(-1)
                for robot in range(self.controlled_robot_count)
            ],
            dim=-1,
        )
        previous_skills = torch.cat(
            (
                current_skills[:, None, None].expand(
                    -1, skills.shape[1], 1, -1
                ),
                skills[..., :-1, :],
            ),
            dim=2,
        )
        switches = (skills != previous_skills).to(parameters.dtype).sum(-1)
        current_parameters = torch.stack(
            [
                initial_states[
                    :, self.schema.slice(f"robot_{robot}.previous_command")
                ]
                for robot in range(self.controlled_robot_count)
            ],
            dim=-2,
        )
        previous_parameters = torch.cat(
            (
                current_parameters[:, None, None].expand(
                    -1, skills.shape[1], 1, -1, -1
                ),
                normalized[..., :-1, :, :],
            ),
            dim=2,
        )
        compatible = (skills == previous_skills).unsqueeze(-1).to(parameters.dtype)
        changes = (
            (normalized - previous_parameters).square() * compatible
        ).sum(dim=(-1, -2))
        return switches, changes

    def evaluate(
        self,
        initial_states: torch.Tensor,
        action_sequences: torch.Tensor,
        rollout: Mapping[str, torch.Tensor],
    ) -> MPCObjectiveResult:
        """Return a signed component decomposition for every candidate."""

        rewards = rollout["predicted_rewards"]
        if rewards.ndim != 3:
            raise ValueError(
                "rollout['predicted_rewards'] must have shape [B,C,H], got "
                f"{tuple(rewards.shape)}"
            )
        if action_sequences.ndim == 3:
            action_sequences = action_sequences[:, None].expand(
                -1, rewards.shape[1], -1, -1
            )
        if action_sequences.ndim != 4 or action_sequences.shape[:3] != rewards.shape:
            raise ValueError(
                "action_sequences must have shape [B,C,H,A] matching predicted rewards"
            )

        horizon = rewards.shape[-1]
        discount = torch.pow(
            torch.as_tensor(
                self.config.gamma, dtype=rewards.dtype, device=rewards.device
            ),
            torch.arange(horizon, dtype=rewards.dtype, device=rewards.device),
        )
        done = rollout.get("predicted_done_probabilities", torch.zeros_like(rewards))
        active, terminal_survival = self._survival(done)
        weights = active * discount
        components = {
            name: torch.zeros_like(rewards[..., 0]) for name in self.COMPONENT_NAMES
        }
        learned_return = (rewards * weights).sum(dim=-1)
        predicted_return = learned_return
        needs_invalid = (
            self.config.invalid_skill_penalty != 0.0
            or self.analytical_reward.reward_scales["invalid_skill"] != 0.0
            or any(
                self.analytical_reward.reward_scales[name] != 0.0
                for name in (
                    "walk_command_alignment",
                    "face_ball_while_approaching",
                    "face_goal_while_moving",
                    "dribble_ball_control",
                )
            )
        )
        if needs_invalid:
            invalid_mask = self.skill_resolver.affordances(
                rollout["predicted_states"][..., :-1, :], action_sequences
            )[0]
        else:
            skills, _ = self.action_adapter.unpack(action_sequences)
            invalid_mask = torch.zeros_like(skills, dtype=torch.bool)
        invalid = invalid_mask[..., : self.controlled_robot_count].to(rewards.dtype).sum(-1)
        executed_actions = rollout.get("executed_actions", action_sequences)
        if self.config.shooting_options:
            requested_ids, _ = self.action_adapter.unpack(action_sequences)
            timers = torch.cat([rollout["predicted_states"][..., :-1, self.schema.slice(f"robot_{r}.shoot_option_remaining")]
                                for r in range(self.action_adapter.num_robots)], -1)
            invalid_mask = invalid_mask & ~((requested_ids == 2) | (timers > 0))
        analytical_rewards, analytical_components = self.analytical_reward(
            rollout["predicted_states"],
            executed_actions,
            rollout.get(
                "event_probabilities",
                torch.zeros(*rewards.shape, 0, dtype=rewards.dtype, device=rewards.device),
            ),
            invalid_mask,
        )
        analytical_return = (analytical_rewards * weights).sum(dim=-1)
        if self.config.objective_mode != "terminal_value_only":
            if self.config.reward_source in {"learned", "blend"}:
                components["predicted_reward_return"] = (
                    self.config.learned_reward_coefficient * learned_return
                )
            if self.config.reward_source in {"analytical", "blend"}:
                components["analytical_reward_return"] = (
                    self.config.analytical_reward_coefficient * analytical_return
                )
            predicted_return = (
                components["predicted_reward_return"]
                + components["analytical_reward_return"]
            )

        numerically_valid = torch.isfinite(rewards).all(dim=-1)
        numerically_valid &= torch.isfinite(action_sequences).flatten(2).all(dim=-1)
        predicted_states = rollout.get("predicted_states")
        if predicted_states is not None:
            numerically_valid &= torch.isfinite(predicted_states).flatten(2).all(dim=-1)
        numerically_valid &= torch.isfinite(done).all(dim=-1)

        state_uncertainty = rollout.get("state_uncertainty", torch.zeros_like(rewards))
        reward_uncertainty = rollout.get("reward_uncertainty")
        member_rewards = rollout.get("member_predicted_rewards")
        if member_rewards is not None:
            member_done = rollout.get("member_predicted_done_probabilities")
            if member_done is None:
                member_done = torch.zeros_like(member_rewards)
            member_active, _ = self._survival(member_done)
            member_returns = (member_rewards * member_active * discount).sum(-1)
            return_uncertainty = member_returns.std(dim=0, unbiased=False)
        elif reward_uncertainty is None:
            return_uncertainty = torch.zeros_like(predicted_return)
        else:
            return_uncertainty = torch.sqrt(
                (reward_uncertainty.clamp(min=0.0) * weights.square())
                .sum(dim=-1)
                .clamp(min=0.0)
            )
            return_uncertainty = torch.nan_to_num(return_uncertainty)

        components["state_uncertainty_penalty"] = (
            -self.config.uncertainty_penalty * (state_uncertainty * weights).sum(-1)
        )
        components["return_uncertainty_penalty"] = (
            -self.config.return_uncertainty_penalty * return_uncertainty
        )
        components["invalid_skill_penalty"] = (
            -self.config.invalid_skill_penalty * (invalid * weights).sum(-1)
        )
        if (
            self.config.skill_switch_penalty != 0.0
            or self.config.command_change_penalty != 0.0
        ):
            switches, changes = self._action_costs(
                initial_states, action_sequences
            )
            components["skill_switch_penalty"] = (
                -self.config.skill_switch_penalty * (switches * weights).sum(-1)
            )
            components["command_change_penalty"] = (
                -self.config.command_change_penalty * (changes * weights).sum(-1)
            )

        terminal_state_value = torch.zeros_like(predicted_return)
        terminal_value_uncertainty = torch.zeros_like(predicted_return)
        discounted_terminal_value = torch.zeros_like(predicted_return)
        if self.config.objective_mode in {
            "terminal_value_only",
            "reward_plus_terminal_value",
        }:
            terminal_estimate = self.terminal_value(
                rollout["predicted_states"][..., -1, :]
            )
            if isinstance(terminal_estimate, tuple):
                terminal_state_value, terminal_value_uncertainty = terminal_estimate
            else:
                terminal_state_value = terminal_estimate
            if terminal_state_value.shape != predicted_return.shape:
                raise ValueError(
                    "Terminal value returned an incompatible shape: "
                    f"{tuple(terminal_state_value.shape)}"
                )
            discounted_terminal_value = (
                (self.config.gamma ** horizon)
                * terminal_survival
                * terminal_state_value
            )
            if self.config.terminal_value_uncertainty_gating:
                discounted_terminal_value *= torch.exp(
                    -self.config.terminal_value_uncertainty_beta
                    * (
                        state_uncertainty[..., -1]
                        + terminal_value_uncertainty.clamp(min=0.0)
                    )
                )
            components["terminal_value"] = (
                self.config.terminal_value_coefficient * discounted_terminal_value
            )

        raw_total = sum(components.values())
        numerically_valid &= torch.isfinite(raw_total)
        valid = numerically_valid.clone()
        max_state_uncertainty = state_uncertainty.max(dim=-1).values
        state_ood = torch.zeros_like(valid)
        return_ood = torch.zeros_like(valid)
        if self.config.max_state_uncertainty is not None:
            state_ood = max_state_uncertainty > self.config.max_state_uncertainty
            valid &= ~state_ood
        if self.config.max_return_uncertainty is not None:
            return_ood = return_uncertainty > self.config.max_return_uncertainty
            valid &= ~return_ood
        value_ood = torch.zeros_like(valid)
        if self.config.terminal_value_max_uncertainty is not None:
            value_ood = (
                terminal_value_uncertainty
                > self.config.terminal_value_max_uncertainty
            )
            valid &= ~value_ood
        total = torch.where(
            valid, raw_total, torch.full_like(raw_total, float("-inf"))
        )
        diagnostics = {
            "predicted_discounted_reward_return": predicted_return,
            "learned_discounted_reward_return": learned_return,
            "analytical_discounted_reward_return": analytical_return,
            "analytical_rewards": analytical_rewards,
            "max_state_uncertainty": max_state_uncertainty,
            "state_ood_rejected": state_ood,
            "return_ood_rejected": return_ood,
            "terminal_value_ood_rejected": value_ood,
            "numerically_valid": numerically_valid,
            "raw_total": raw_total,
            "terminal_state_value": terminal_state_value,
            "terminal_value_uncertainty": terminal_value_uncertainty,
            "discounted_terminal_value": discounted_terminal_value,
            "terminal_survival_probability": terminal_survival,
        }
        diagnostics.update(
            {f"analytical/{name}": value for name, value in analytical_components.items()}
        )
        return MPCObjectiveResult(
            total=total,
            components=components,
            valid=valid,
            return_uncertainty=return_uncertainty,
            diagnostics=diagnostics,
        )
