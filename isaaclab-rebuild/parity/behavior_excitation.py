"""Deterministic coordinator inputs for frozen-skill behavior traces."""

from __future__ import annotations

import numpy as np


SKILL_NAMES = ("walk", "dribble", "shoot")


def generate_coordinator_sequence(
    steps: int,
    num_envs: int,
    mode: str,
    command_input: tuple[float, float, float] = (0.25, -0.15, 0.10),
    switch_interval: int = 8,
) -> np.ndarray:
    """Return `(T,N,6)` logits plus raw tanh-command inputs."""

    if steps <= 0 or num_envs <= 0:
        raise ValueError("steps and num_envs must be positive")
    if mode not in (*SKILL_NAMES, "switch"):
        raise ValueError(f"mode must be one of {(*SKILL_NAMES, 'switch')}")
    if switch_interval <= 0:
        raise ValueError("switch_interval must be positive")
    command = np.asarray(command_input, dtype=np.float32)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("command_input must contain three finite values")

    sequence = np.zeros((steps, num_envs, 6), dtype=np.float32)
    sequence[..., 3:6] = command
    fixed_skill = SKILL_NAMES.index(mode) if mode in SKILL_NAMES else None
    for step in range(steps):
        skill_id = fixed_skill if fixed_skill is not None else (step // switch_interval) % len(SKILL_NAMES)
        sequence[step, :, skill_id] = 1.0
    return sequence
