"""Configuration and validation for hybrid CEM MPC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    gamma: float = 0.99
    uncertainty_penalty: float = 0.10
    return_std_penalty: float = 0.50
    ensemble_objective: str = "mean_minus_std"
    termination_threshold: float = 0.50
    skill_switch_penalty: float = 0.01
    command_change_penalty: float = 0.01
    collision_penalty: float = 1.0
    out_of_bounds_penalty: float = 2.0
    robot_fall_penalty: float = 1.0
    invalid_skill_penalty: float = 2.0
    dribble_affordance_distance_m: float = 1.0
    shoot_affordance_distance_m: float = 0.75
    shoot_min_forward_m: float = -0.10
    shoot_lateral_reach_m: float = 0.45
    ball_setup_penalty: float = 0.0
    dribble_target_forward_m: float = 0.35
    shoot_target_distance_m: float = 0.45
    backward_dribble_penalty: float = 0.0
    dribble_forward_speed_scale_mps: float = 1.0
    reposition_approach_coefficient: float = 0.0
    reposition_command_alignment_coefficient: float = 0.0
    reposition_first_step_multiplier: float = 1.0
    ball_progress_coefficient: float = 0.0
    possession_coefficient: float = 0.0
    goal_probability_bonus: float = 0.0
    event_coefficients: Dict[str, float] = field(default_factory=dict)
    objective_mode: str = "reward_only"
    use_terminal_value: bool = False
    terminal_value_checkpoint: Optional[str] = None
    terminal_value_required: bool = False
    terminal_value_coefficient: float = 1.0
    terminal_handling: str = "hard_threshold"
    terminal_value_clip: bool = False
    terminal_value_clip_min: Optional[float] = None
    terminal_value_clip_max: Optional[float] = None
    terminal_value_uncertainty_gating: bool = False
    terminal_value_uncertainty_beta: float = 1.0
    execute_mean_action: bool = False
    execute_best_sample: bool = True
    execution_action_smoothing: float = 0.0
    minimum_skill_duration: int = 1
    fallback_policy: str = "safe_reposition"
    uncertainty_fallback_threshold: Optional[float] = None
    warm_start_uncertainty_threshold: Optional[float] = None
    invalid_objective: float = -1.0e9
    max_candidate_diagnostics: int = 0
    seed: int = 42

    def validate(self) -> "MPCConfig":
        if self.use_terminal_value and self.objective_mode == "reward_only":
            self.objective_mode = "reward_plus_terminal_value"
        if self.algorithm.lower() != "cem":
            raise ValueError(f"Unsupported MPC algorithm {self.algorithm!r}; expected 'cem'")
        if self.horizon < 1:
            raise ValueError("mpc.horizon must be at least 1")
        if self.num_candidates < 2:
            raise ValueError("mpc.num_candidates must be at least 2")
        if not 0 < self.num_elites < self.num_candidates:
            raise ValueError("mpc.num_elites must satisfy 0 < num_elites < num_candidates")
        if self.num_iterations < 1:
            raise ValueError("mpc.num_iterations must be at least 1")
        if self.candidate_batch_size < 0:
            raise ValueError("mpc.candidate_batch_size cannot be negative")
        if self.skill_temperature <= 0:
            raise ValueError("mpc.skill_temperature must be positive")
        if not 0 <= self.min_skill_probability < (1.0 / 3.0):
            raise ValueError("mpc.min_skill_probability must lie in [0, 1/3)")
        if not 0 < self.min_parameter_std_fraction <= self.max_parameter_std_fraction:
            raise ValueError("MPC parameter standard-deviation fractions are inconsistent")
        if not self.min_parameter_std_fraction <= self.initial_parameter_std_fraction <= self.max_parameter_std_fraction:
            raise ValueError("mpc.initial_parameter_std_fraction must lie between min and max")
        for name in ("categorical_smoothing", "continuous_smoothing", "execution_action_smoothing"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"mpc.{name} must lie in [0, 1]")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("mpc.gamma must lie in (0, 1]")
        if not 0.0 <= self.termination_threshold <= 1.0:
            raise ValueError("mpc.termination_threshold must lie in [0, 1]")
        if self.minimum_skill_duration < 1:
            raise ValueError("mpc.minimum_skill_duration must be at least 1 (1 disables the constraint)")
        if self.execute_mean_action == self.execute_best_sample:
            raise ValueError("Select exactly one of execute_mean_action and execute_best_sample")
        allowed_ensemble = {"mean", "mean_minus_std", "minimum"}
        if self.ensemble_objective not in allowed_ensemble:
            raise ValueError(f"mpc.ensemble_objective must be one of {sorted(allowed_ensemble)}")
        allowed_modes = {"reward_only", "terminal_value_only", "reward_plus_terminal_value"}
        if self.objective_mode not in allowed_modes:
            raise ValueError(f"mpc.objective_mode must be one of {sorted(allowed_modes)}")
        if self.terminal_handling not in {"hard_threshold", "probability_weighted"}:
            raise ValueError("mpc.terminal_handling must be hard_threshold or probability_weighted")
        if self.terminal_value_clip_min is not None and self.terminal_value_clip_max is not None:
            if self.terminal_value_clip_min > self.terminal_value_clip_max:
                raise ValueError("terminal value clipping bounds are reversed")
        for name in (
            "dribble_target_forward_m",
            "shoot_target_distance_m",
            "dribble_forward_speed_scale_mps",
            "reposition_first_step_multiplier",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"mpc.{name} must be positive")
        nonnegative = (
            "uncertainty_penalty",
            "return_std_penalty",
            "skill_switch_penalty",
            "command_change_penalty",
            "collision_penalty",
            "out_of_bounds_penalty",
            "robot_fall_penalty",
            "invalid_skill_penalty",
            "dribble_affordance_distance_m",
            "shoot_affordance_distance_m",
            "shoot_lateral_reach_m",
            "ball_setup_penalty",
            "backward_dribble_penalty",
            "reposition_approach_coefficient",
            "reposition_command_alignment_coefficient",
            "terminal_value_coefficient",
            "terminal_value_uncertainty_beta",
        )
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"mpc.{name} cannot be negative")
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
    """Load a complete YAML file and resolve an optional named MPC profile.

    The returned mapping retains environment and collection sections needed by
    simulator scripts, while the first return value contains validated planner
    fields only.
    """

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
            raise ValueError(f"Unknown MPC profile {selected!r}; available profiles: {sorted(profiles)}")
        planner = deep_update(planner, profiles[selected])
    terminal = planner.pop("terminal_value", None)
    if terminal is not None:
        terminal = dict(terminal)
        aliases = {
            "enabled": "use_terminal_value",
            "checkpoint": "terminal_value_checkpoint",
            "required": "terminal_value_required",
            "coefficient": "terminal_value_coefficient",
            "clip_value": "terminal_value_clip",
            "clip_min": "terminal_value_clip_min",
            "clip_max": "terminal_value_clip_max",
            "uncertainty_gating": "terminal_value_uncertainty_gating",
            "uncertainty_beta": "terminal_value_uncertainty_beta",
        }
        unknown = sorted(set(terminal) - set(aliases))
        if unknown:
            raise ValueError(f"Unknown mpc.terminal_value fields: {unknown}")
        planner.update({aliases[key]: value for key, value in terminal.items()})
    if planner.get("use_terminal_value") and planner.get("objective_mode", "reward_only") == "reward_only":
        # Backward compatibility for the original boolean switch.
        planner["objective_mode"] = "reward_plus_terminal_value"
    return MPCConfig.from_mapping(planner), payload
