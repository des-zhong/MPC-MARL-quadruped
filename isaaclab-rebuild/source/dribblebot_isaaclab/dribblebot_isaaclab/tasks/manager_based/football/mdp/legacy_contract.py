"""Pure tensor helpers for the legacy low-level policy contract.

This module deliberately has no Isaac Lab imports.  It is shared by manager
terms at runtime and by simulator-free regression tests under the old Python
environment.
"""

from __future__ import annotations

import math

import torch


LEGACY_HISTORY_LENGTH = 15
LEGACY_WALK_OBSERVATION_DIM = 72
LEGACY_BALL_OBSERVATION_DIM = 75

LEGACY_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

LEGACY_COMMAND_SCALE = (
    2.0,
    2.0,
    0.25,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.15,
    0.3,
    0.3,
    1.0,
    1.0,
    1.0,
)


def compose_legacy_command(base_command: torch.Tensor, gait_parameters: torch.Tensor) -> torch.Tensor:
    """Compose and scale the original 15-value ``RCSensor`` observation."""

    _require_matrix("base_command", base_command, 3)
    _require_matrix("gait_parameters", gait_parameters, 12)
    _require_same_batch(base_command, gait_parameters)
    command = torch.cat((base_command, gait_parameters), dim=-1)
    return command * command.new_tensor(LEGACY_COMMAND_SCALE)


def advance_gait_phase(phase: torch.Tensor, frequency: torch.Tensor, dt: float) -> torch.Tensor:
    """Advance the scalar gait phase using the old policy-step convention."""

    if phase.ndim != 1 or frequency.ndim != 1:
        raise ValueError("phase and frequency must be one-dimensional tensors.")
    if phase.shape != frequency.shape:
        raise ValueError("phase and frequency must have the same shape.")
    return torch.remainder(phase + float(dt) * frequency, 1.0)


def legacy_gait_clock(
    gait_phase: torch.Tensor,
    gait_parameters: torch.Tensor,
    pacing_offset: bool = False,
) -> torch.Tensor:
    """Return the four legacy sine clocks in FL, FR, RL, RR order."""

    if gait_phase.ndim != 1:
        raise ValueError("gait_phase must be one-dimensional.")
    _require_matrix("gait_parameters", gait_parameters, 12)
    if gait_phase.shape[0] != gait_parameters.shape[0]:
        raise ValueError("gait_phase and gait_parameters must have the same batch size.")

    phase = gait_parameters[:, 2]
    offset = gait_parameters[:, 3]
    bound = gait_parameters[:, 4]
    duration = gait_parameters[:, 5]

    if pacing_offset:
        foot_phase = torch.stack(
            (
                gait_phase + phase + offset + bound,
                gait_phase + bound,
                gait_phase + offset,
                gait_phase + phase,
            ),
            dim=-1,
        )
    else:
        foot_phase = torch.stack(
            (
                gait_phase + phase + offset + bound,
                gait_phase + offset,
                gait_phase + bound,
                gait_phase + phase,
            ),
            dim=-1,
        )

    foot_phase = torch.remainder(foot_phase, 1.0)
    duration = duration.unsqueeze(-1)
    remapped = foot_phase.clone()
    stance = foot_phase < duration
    swing = foot_phase > duration
    remapped[stance] = (foot_phase * (0.5 / duration))[stance]
    remapped[swing] = (0.5 + (foot_phase - duration) * (0.5 / (1.0 - duration)))[swing]
    return torch.sin(2.0 * math.pi * remapped)


def assemble_legacy_observation(
    projected_gravity: torch.Tensor,
    scaled_command: torch.Tensor,
    joint_position: torch.Tensor,
    scaled_joint_velocity: torch.Tensor,
    current_action: torch.Tensor,
    previous_action: torch.Tensor,
    gait_clock: torch.Tensor,
    heading: torch.Tensor,
    gait_phase: torch.Tensor,
    ball_position: torch.Tensor | None = None,
) -> torch.Tensor:
    """Concatenate sensors in the exact order used by legacy checkpoints."""

    expected = (
        ("projected_gravity", projected_gravity, 3),
        ("scaled_command", scaled_command, 15),
        ("joint_position", joint_position, 12),
        ("scaled_joint_velocity", scaled_joint_velocity, 12),
        ("current_action", current_action, 12),
        ("previous_action", previous_action, 12),
        ("gait_clock", gait_clock, 4),
        ("heading", heading, 1),
        ("gait_phase", gait_phase, 1),
    )
    for name, value, width in expected:
        _require_matrix(name, value, width)
    tensors = [value for _, value, _ in expected]

    if ball_position is not None:
        _require_matrix("ball_position", ball_position, 3)
        tensors.insert(0, ball_position)

    _require_same_batch(*tensors)
    observation = torch.cat(tensors, dim=-1)
    expected_width = LEGACY_BALL_OBSERVATION_DIM if ball_position is not None else LEGACY_WALK_OBSERVATION_DIM
    if observation.shape[1] != expected_width:
        raise RuntimeError(f"Legacy observation has width {observation.shape[1]}, expected {expected_width}.")
    return observation


def add_legacy_observation_noise(observation: torch.Tensor, include_ball: bool) -> torch.Tensor:
    """Apply the original uniform sensor noise with one name-stable scale vector."""

    expected_width = LEGACY_BALL_OBSERVATION_DIM if include_ball else LEGACY_WALK_OBSERVATION_DIM
    _require_matrix("observation", observation, expected_width)
    noise_scale = observation.new_zeros(expected_width)
    offset = 0
    if include_ball:
        noise_scale[:3] = 0.05
        offset = 3
    noise_scale[offset : offset + 3] = 0.05
    noise_scale[offset + 18 : offset + 30] = 0.01
    noise_scale[offset + 30 : offset + 42] = 0.075
    return observation + (2.0 * torch.rand_like(observation) - 1.0) * noise_scale


def update_zero_padded_history(
    history: torch.Tensor,
    observation: torch.Tensor,
    append_mask: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reset selected histories, then append one frame for selected environments."""

    if history.ndim != 3:
        raise ValueError("history must have shape (num_envs, history_length, observation_dim).")
    if observation.ndim != 2:
        raise ValueError("observation must have shape (num_envs, observation_dim).")
    if history.shape[0] != observation.shape[0] or history.shape[2] != observation.shape[1]:
        raise ValueError("history and observation dimensions are incompatible.")

    num_envs = history.shape[0]
    reset_mask = _normalize_mask(reset_mask, num_envs, history.device, default=False)
    append_mask = _normalize_mask(append_mask, num_envs, history.device, default=True)
    updated = history.clone()
    updated[reset_mask] = 0.0
    if torch.any(append_mask):
        updated[append_mask, :-1] = updated[append_mask, 1:].clone()
        updated[append_mask, -1] = observation[append_mask]
    return updated


def flatten_history(history: torch.Tensor) -> torch.Tensor:
    """Flatten oldest-to-newest history frames like ``HistoryWrapper``."""

    if history.ndim != 3:
        raise ValueError("history must have three dimensions.")
    return history.reshape(history.shape[0], -1)


def _normalize_mask(
    mask: torch.Tensor | None,
    num_envs: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    if mask is None:
        return torch.full((num_envs,), default, dtype=torch.bool, device=device)
    if mask.ndim != 1 or mask.shape[0] != num_envs:
        raise ValueError("mask must have shape (num_envs,).")
    return mask.to(device=device, dtype=torch.bool)


def _require_matrix(name: str, value: torch.Tensor, width: int) -> None:
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape (num_envs, {width}).")


def _require_same_batch(*values: torch.Tensor) -> None:
    batch_sizes = {value.shape[0] for value in values}
    if len(batch_sizes) != 1:
        raise ValueError("All legacy observation tensors must have the same batch size.")
