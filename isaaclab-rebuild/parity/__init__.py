"""Simulator-independent rollout parity utilities."""

from .behavior_comparison import DEFAULT_BEHAVIOR_THRESHOLDS, compare_behavior_files, compare_behavior_traces
from .behavior_excitation import SKILL_NAMES, generate_coordinator_sequence
from .behavior_schema import (
    BEHAVIOR_SCHEMA_VERSION,
    load_behavior_trace,
    save_behavior_trace,
    validate_behavior_trace,
)
from .comparison import DEFAULT_THRESHOLDS, compare_rollouts
from .excitation import ExcitationConfig, generate_action, generate_action_sequence
from .physics_profile import PARITY_PHYSICS_PROFILE, make_physics_profile
from .policy_contract import make_legacy_policy_contract
from .schema import SCHEMA_VERSION, load_rollout, save_rollout, validate_rollout

__all__ = [
    "DEFAULT_THRESHOLDS",
    "DEFAULT_BEHAVIOR_THRESHOLDS",
    "BEHAVIOR_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ExcitationConfig",
    "PARITY_PHYSICS_PROFILE",
    "SKILL_NAMES",
    "compare_rollouts",
    "compare_behavior_files",
    "compare_behavior_traces",
    "generate_action",
    "generate_action_sequence",
    "generate_coordinator_sequence",
    "load_behavior_trace",
    "load_rollout",
    "make_physics_profile",
    "make_legacy_policy_contract",
    "save_behavior_trace",
    "save_rollout",
    "validate_behavior_trace",
    "validate_rollout",
]
