"""Configuration for risk-aware hybrid CEM MPC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from dribblebot.world_model.config import deep_update, load_config


@dataclass
class MPCConfig:
    """Planner settings expressed at the world model's macro-action timescale."""

    algorithm: str = "cem"
    horizon: int = 8
    num_candidates: int = 2048
    num_elites: int = 128
    num_iterations: int = 5
    candidate_batch_size: int = 0
    skill_temperature: float = 1.0
    min_skill_probability: float = 0.02
    initial_parameter_std_fraction: float = 0.50
    min_parameter_std_fraction: float = 0.03
    max_parameter_std_fraction: float = 0.75
    categorical_smoothing: float = 0.25
    continuous_smoothing: float = 0.25
    warm_start: bool = True
    deterministic_world_model: bool = True
    gamma: float = 1.0
    uncertainty_penalty: float = 0.0
    return_uncertainty_penalty: float = 0.0
    skill_switch_penalty: float = 0.0
    command_change_penalty: float = 0.0
    invalid_skill_penalty: float = 0.0
    apply_skill_fallback_in_rollout: bool = True
    apply_collision_avoidance_in_rollout: bool = True
    dribble_affordance_distance_m: float = 1.0
    shoot_affordance_distance_m: float = 0.75
    shoot_min_forward_m: float = -0.10
    shoot_lateral_reach_m: float = 0.45
    fallback_walk_speed_mps: float = 0.9
    collision_avoidance_lookahead_s: float = 0.6
    collision_avoidance_distance_m: float = 0.9
    collision_avoidance_speed_mps: float = 0.8
    reward_source: str = "learned"
    learned_reward_coefficient: float = 1.0
    analytical_reward_coefficient: float = 1.0
    analytical_macro_dt: float = 0.2
    analytical_event_dt: float = 0.02
    max_state_uncertainty: Optional[float] = None
    max_return_uncertainty: Optional[float] = None
    relax_ood_when_all_candidates_rejected: bool = True
    objective_mode: str = "reward_only"
    terminal_handling: str = "probability_weighted"
    termination_threshold: float = 0.5
    use_terminal_value: bool = False
    terminal_value_checkpoint: Optional[str] = None
    terminal_value_required: bool = False
    terminal_value_coefficient: float = 1.0
    terminal_value_uncertainty_gating: bool = True
    terminal_value_uncertainty_beta: float = 1.0
    terminal_value_max_uncertainty: Optional[float] = None
    max_candidate_diagnostics: int = 0
    seed: int = 42

    def validate(self) -> "MPCConfig":
        if self.use_terminal_value and self.objective_mode == "reward_only":
            self.objective_mode = "reward_plus_terminal_value"
        if self.algorithm.lower() != "cem":
            raise ValueError(
                f"Unsupported MPC algorithm {self.algorithm!r}; expected 'cem'"
            )
        if self.horizon < 1:
            raise ValueError("mpc.horizon must be at least 1")
        if self.num_candidates < 2:
            raise ValueError("mpc.num_candidates must be at least 2")
        if not 0 < self.num_elites < self.num_candidates:
            raise ValueError(
                "mpc.num_elites must satisfy 0 < num_elites < num_candidates"
            )
        if self.num_iterations < 1:
            raise ValueError("mpc.num_iterations must be at least 1")
        if self.candidate_batch_size < 0:
            raise ValueError("mpc.candidate_batch_size cannot be negative")
        if self.skill_temperature <= 0:
            raise ValueError("mpc.skill_temperature must be positive")
        if not 0 <= self.min_skill_probability < (1.0 / 3.0):
            raise ValueError("mpc.min_skill_probability must lie in [0, 1/3)")
        if not 0 < self.min_parameter_std_fraction <= self.max_parameter_std_fraction:
            raise ValueError(
                "MPC parameter standard-deviation fractions are inconsistent"
            )
        if not (
            self.min_parameter_std_fraction
            <= self.initial_parameter_std_fraction
            <= self.max_parameter_std_fraction
        ):
            raise ValueError(
                "mpc.initial_parameter_std_fraction must lie between min and max"
            )
        for name in ("categorical_smoothing", "continuous_smoothing"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"mpc.{name} must lie in [0, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("mpc.gamma must lie in (0, 1]")
        if self.objective_mode not in {
            "reward_only",
            "terminal_value_only",
            "reward_plus_terminal_value",
        }:
            raise ValueError("Unsupported mpc.objective_mode")
        if self.terminal_handling not in {"hard_threshold", "probability_weighted"}:
            raise ValueError(
                "mpc.terminal_handling must be hard_threshold or probability_weighted"
            )
        if self.reward_source not in {"learned", "analytical", "blend"}:
            raise ValueError("mpc.reward_source must be learned, analytical, or blend")
        if not 0.0 <= self.termination_threshold <= 1.0:
            raise ValueError("mpc.termination_threshold must lie in [0, 1]")
        for name in (
            "uncertainty_penalty",
            "return_uncertainty_penalty",
            "skill_switch_penalty",
            "command_change_penalty",
            "invalid_skill_penalty",
            "dribble_affordance_distance_m",
            "shoot_affordance_distance_m",
            "shoot_lateral_reach_m",
            "fallback_walk_speed_mps",
            "collision_avoidance_lookahead_s",
            "collision_avoidance_distance_m",
            "collision_avoidance_speed_mps",
            "learned_reward_coefficient",
            "analytical_reward_coefficient",
            "analytical_macro_dt",
            "analytical_event_dt",
            "terminal_value_coefficient",
            "terminal_value_uncertainty_beta",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"mpc.{name} cannot be negative")
        for name in (
            "max_state_uncertainty", "max_return_uncertainty",
            "terminal_value_max_uncertainty",
        ):
            value = getattr(self, name)
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"mpc.{name} must be positive when set")
        for name in (
            "collision_avoidance_distance_m",
            "collision_avoidance_speed_mps",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"mpc.{name} must be positive")
        if self.max_candidate_diagnostics < 0:
            raise ValueError("mpc.max_candidate_diagnostics cannot be negative")
        return self

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "MPCConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"Unknown MPC configuration fields: {unknown}")
        return cls(**dict(mapping)).validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_mpc_config(
    path_or_mapping: Union[str, Path, Mapping[str, Any]],
    profile: Optional[str] = None,
) -> tuple[MPCConfig, Dict[str, Any]]:
    """Load a YAML file and resolve an optional named search profile."""

    payload = (
        load_config(path_or_mapping)
        if isinstance(path_or_mapping, (str, Path))
        else dict(path_or_mapping)
    )
    planner = dict(payload.get("mpc", payload))
    profiles = payload.get("mpc_profiles", {})
    selected = profile or payload.get("mpc_profile")
    if selected:
        if selected not in profiles:
            raise ValueError(
                f"Unknown MPC profile {selected!r}; available profiles: "
                f"{sorted(profiles)}"
            )
        planner = deep_update(planner, profiles[selected])
    terminal = planner.pop("terminal_value", None)
    if terminal is not None:
        terminal = dict(terminal)
        aliases = {
            "enabled": "use_terminal_value",
            "checkpoint": "terminal_value_checkpoint",
            "required": "terminal_value_required",
            "coefficient": "terminal_value_coefficient",
            "uncertainty_gating": "terminal_value_uncertainty_gating",
            "uncertainty_beta": "terminal_value_uncertainty_beta",
            "max_uncertainty": "terminal_value_max_uncertainty",
        }
        unknown = sorted(set(terminal) - set(aliases))
        if unknown:
            raise ValueError(f"Unknown mpc.terminal_value fields: {unknown}")
        planner.update({aliases[key]: value for key, value in terminal.items()})
    return MPCConfig.from_mapping(planner), payload
