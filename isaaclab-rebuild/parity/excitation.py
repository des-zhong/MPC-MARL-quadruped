"""Deterministic open-loop action sequences for cross-simulator rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ExcitationConfig:
    """Parameters for zero or sinusoidal normalized joint actions."""

    kind: str = "zero"
    amplitude: float = 0.35
    frequency_hz: float = 0.75
    phase_stride_rad: float = 0.37

    def validate(self) -> None:
        if self.kind not in {"zero", "sine"}:
            raise ValueError("excitation kind must be 'zero' or 'sine'")
        if not np.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("amplitude must be finite and non-negative")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz < 0.0:
            raise ValueError("frequency_hz must be finite and non-negative")
        if not np.isfinite(self.phase_stride_rad):
            raise ValueError("phase_stride_rad must be finite")


def _validate_names(action_names: Sequence[str]) -> list[str]:
    names = list(action_names)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("action_names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ValueError("action_names must be unique")
    return names


def generate_action(
    step_index: int,
    step_dt: float,
    num_envs: int,
    action_names: Sequence[str],
    config: ExcitationConfig,
) -> np.ndarray:
    """Generate one action frame with name-stable joint phases.

    Phase indices come from sorted action names, so a simulator-specific joint
    ordering does not change the signal assigned to a named joint.
    """

    config.validate()
    names = _validate_names(action_names)
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if step_dt <= 0.0 or not np.isfinite(step_dt):
        raise ValueError("step_dt must be finite and positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    if config.kind == "zero":
        return np.zeros((num_envs, len(names)), dtype=np.float32)

    rank_by_name = {name: index for index, name in enumerate(sorted(names))}
    phase = np.asarray([rank_by_name[name] for name in names], dtype=np.float64) * config.phase_stride_rad
    time_s = float(step_index) * step_dt
    joint_action = config.amplitude * np.sin(2.0 * np.pi * config.frequency_hz * time_s + phase)
    return np.broadcast_to(joint_action.astype(np.float32), (num_envs, len(names))).copy()


def generate_action_sequence(
    steps: int,
    step_dt: float,
    num_envs: int,
    action_names: Sequence[str],
    config: ExcitationConfig,
) -> np.ndarray:
    """Generate a complete action sequence with shape ``(T, N, A)``."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    return np.stack(
        [generate_action(index, step_dt, num_envs, action_names, config) for index in range(steps)],
        axis=0,
    )
