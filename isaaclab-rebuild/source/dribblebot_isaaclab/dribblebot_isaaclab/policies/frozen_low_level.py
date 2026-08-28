"""Simulator-independent inference adapters for legacy low-level policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


WALK_SKILL_ID = 0
DRIBBLE_SKILL_ID = 1
SHOOT_SKILL_ID = 2
SKILL_NAMES = ("walk", "dribble", "shoot")
ACTION_HISTORY_SEMANTICS = (
    "duplicate_previous_policy_output",
    "successive_policy_outputs",
)
LEGACY_WRAPPER_COMMAND_SCALES = (
    (1.2, 0.6, 0.0),
    (1.5, 1.5, 1.0),
    (1.5, 1.5, 0.0),
)
REPRODUCTION_COMMAND_SCALES = (
    (1.5, 1.5, 1.0),
    (1.5, 1.5, 1.0),
    (3.0, 3.0, 0.0),
)


@dataclass(frozen=True)
class FrozenPolicySpec:
    """Trusted TorchScript artifacts and their runtime safety limits."""

    name: str
    body_path: Path
    adaptation_path: Path
    action_clip: float = 1.0
    expected_action_dim: int = 12


class FrozenLowLevelPolicy:
    """Compose a TorchScript adaptation module and deterministic actor body."""

    def __init__(
        self,
        spec: FrozenPolicySpec,
        body: torch.nn.Module,
        adaptation: torch.nn.Module,
        device: str,
    ) -> None:
        self.spec = spec
        self.device = torch.device(device)
        self.body = body.eval()
        self.adaptation = adaptation.eval()
        self.history_dim, self.latent_dim = _module_input_output_dims(self.adaptation, spec.name)
        body_input_dim, self.action_dim = _module_input_output_dims(self.body, spec.name)
        if body_input_dim != self.history_dim + self.latent_dim:
            raise ValueError(
                f"{spec.name} actor expects {body_input_dim} values, but history+latent provides "
                f"{self.history_dim + self.latent_dim}."
            )
        if self.action_dim != int(spec.expected_action_dim):
            raise ValueError(
                f"{spec.name} actor produces {self.action_dim} actions, expected {spec.expected_action_dim}."
            )
        if not math.isfinite(spec.action_clip) or spec.action_clip <= 0.0:
            raise ValueError(f"{spec.name} action clip must be finite and positive.")

    @classmethod
    def load(cls, spec: FrozenPolicySpec, device: str = "cpu") -> "FrozenLowLevelPolicy":
        """Load trusted local TorchScript files on the requested device."""

        body_path = Path(spec.body_path).expanduser().resolve()
        adaptation_path = Path(spec.adaptation_path).expanduser().resolve()
        if not body_path.is_file():
            raise FileNotFoundError(f"Policy actor body does not exist: {body_path}")
        if not adaptation_path.is_file():
            raise FileNotFoundError(f"Policy adaptation module does not exist: {adaptation_path}")
        body = torch.jit.load(str(body_path), map_location=device)
        adaptation = torch.jit.load(str(adaptation_path), map_location=device)
        return cls(spec, body=body, adaptation=adaptation, device=device)

    def __call__(self, history: torch.Tensor) -> torch.Tensor:
        """Infer clipped actions, zeroing rows with invalid inputs or outputs."""

        if history.ndim != 2 or history.shape[1] != self.history_dim:
            raise ValueError(
                f"{self.spec.name} policy expects history shape (N, {self.history_dim}), "
                f"received {tuple(history.shape)}."
            )
        history = history.to(self.device)
        valid_input = torch.isfinite(history).all(dim=-1)
        safe_history = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            latent = self.adaptation.forward(safe_history)
            action = self.body.forward(torch.cat((safe_history, latent), dim=-1))
        if action.ndim != 2 or action.shape != (history.shape[0], self.action_dim):
            raise RuntimeError(
                f"{self.spec.name} actor returned shape {tuple(action.shape)}, "
                f"expected ({history.shape[0]}, {self.action_dim})."
            )
        valid_output = torch.isfinite(action).all(dim=-1)
        action = torch.nan_to_num(
            action,
            nan=0.0,
            posinf=self.spec.action_clip,
            neginf=-self.spec.action_clip,
        ).clamp(-self.spec.action_clip, self.spec.action_clip)
        action[~(valid_input & valid_output)] = 0.0
        return action


class FrozenSkillPolicySet:
    """Route a batch of histories through walking, dribbling, and shooting policies."""

    def __init__(
        self,
        policies: Mapping[int, FrozenLowLevelPolicy],
        full_observation_dim: int = 75,
        history_length: int = 15,
        object_sensor_width: int = 3,
    ) -> None:
        expected_ids = {WALK_SKILL_ID, DRIBBLE_SKILL_ID, SHOOT_SKILL_ID}
        if set(policies) != expected_ids:
            raise ValueError(f"Policy set must contain skill ids {sorted(expected_ids)}.")
        self.policies = dict(policies)
        self.full_observation_dim = int(full_observation_dim)
        self.history_length = int(history_length)
        self.object_sensor_width = int(object_sensor_width)
        action_dims = {policy.action_dim for policy in self.policies.values()}
        devices = {policy.device for policy in self.policies.values()}
        if len(action_dims) != 1:
            raise ValueError("All skill policies must produce the same action dimension.")
        if len(devices) != 1:
            raise ValueError("All skill policies must use the same device.")
        self.action_dim = action_dims.pop()
        self.device = devices.pop()
        if self.full_observation_dim <= self.object_sensor_width:
            raise ValueError("Object sensor width must be smaller than the full observation width.")
        if self.history_length <= 0:
            raise ValueError("History length must be positive.")

    @classmethod
    def load(
        cls,
        specs: Mapping[int, FrozenPolicySpec],
        device: str,
        **kwargs,
    ) -> "FrozenSkillPolicySet":
        policies = {
            skill_id: FrozenLowLevelPolicy.load(spec, device=device)
            for skill_id, spec in specs.items()
        }
        return cls(policies, **kwargs)

    def route(self, full_history: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        """Infer one action per row, adapting 75D histories for the walking policy."""

        full_width = self.full_observation_dim * self.history_length
        if full_history.ndim != 2 or full_history.shape[1] != full_width:
            raise ValueError(f"Full history must have shape (N, {full_width}).")
        if skill_ids.ndim != 1 or skill_ids.shape[0] != full_history.shape[0]:
            raise ValueError("skill_ids must have shape (N,).")
        skill_ids = skill_ids.to(device=self.device, dtype=torch.long)
        invalid = (skill_ids < WALK_SKILL_ID) | (skill_ids > SHOOT_SKILL_ID)
        if torch.any(invalid):
            bad_ids = torch.unique(skill_ids[invalid]).detach().cpu().tolist()
            raise ValueError(f"Unknown skill ids: {bad_ids}")

        full_history = full_history.to(self.device)
        actions = torch.zeros(full_history.shape[0], self.action_dim, device=self.device)
        for skill_id, policy in self.policies.items():
            mask = skill_ids == skill_id
            policy_history = adapt_history_for_policy(
                full_history[mask],
                expected_history_dim=policy.history_dim,
                full_observation_dim=self.full_observation_dim,
                history_length=self.history_length,
                object_sensor_width=self.object_sensor_width,
            )
            actions[mask] = policy(policy_history)
        return actions


def decode_high_level_action(
    action: torch.Tensor,
    command_scales: torch.Tensor,
    input_clip: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode three skill logits and three normalized command values."""

    if action.ndim != 2 or action.shape[1] != 6:
        raise ValueError("High-level action must have shape (N, 6).")
    if command_scales.shape != (3, 3):
        raise ValueError("command_scales must have shape (3, 3).")
    if not math.isfinite(input_clip) or input_clip <= 0.0:
        raise ValueError("input_clip must be finite and positive.")
    valid_input = torch.isfinite(action).all(dim=-1)
    safe_action = torch.nan_to_num(
        action,
        nan=0.0,
        posinf=input_clip,
        neginf=-input_clip,
    ).clamp(-input_clip, input_clip)
    skill_ids = torch.argmax(safe_action[:, :3], dim=-1)
    scales = command_scales.to(device=action.device, dtype=action.dtype)[skill_ids]
    commands = torch.tanh(safe_action[:, 3:6]) * scales
    commands[:, 2] = torch.where(
        skill_ids == SHOOT_SKILL_ID,
        torch.zeros_like(commands[:, 2]),
        commands[:, 2],
    )
    commands[~valid_input] = 0.0
    return skill_ids, commands, ~valid_input


def decode_fixed_skill_action(
    action: torch.Tensor,
    skill_id: int,
    command_scales: torch.Tensor,
    input_clip: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a three-value normalized command for one fixed frozen skill."""

    if action.ndim != 2 or action.shape[1] != 3:
        raise ValueError("Fixed-skill action must have shape (N, 3).")
    if skill_id < WALK_SKILL_ID or skill_id > SHOOT_SKILL_ID:
        raise ValueError(f"Unknown fixed skill id: {skill_id}")
    coordinator_action = torch.zeros(action.shape[0], 6, dtype=action.dtype, device=action.device)
    coordinator_action[:, skill_id] = 1.0
    coordinator_action[:, 3:6] = action
    return decode_high_level_action(coordinator_action, command_scales, input_clip=input_clip)


def observation_action_pair(
    current_action: torch.Tensor,
    previous_action: torch.Tensor,
    semantics: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the current/previous action slices seen by a frozen policy."""

    if current_action.shape != previous_action.shape:
        raise ValueError("current_action and previous_action must have the same shape")
    if semantics not in ACTION_HISTORY_SEMANTICS:
        raise ValueError(f"Unknown action-history semantics: {semantics!r}")
    if semantics == "duplicate_previous_policy_output":
        return current_action, current_action
    return current_action, previous_action


def adapt_history_for_policy(
    full_history: torch.Tensor,
    expected_history_dim: int,
    full_observation_dim: int = 75,
    history_length: int = 15,
    object_sensor_width: int = 3,
) -> torch.Tensor:
    """Remove the leading object sensor when routing a ball history to walking."""

    if full_history.ndim != 2:
        raise ValueError("History must have shape (N, history_dim).")
    full_history_dim = full_observation_dim * history_length
    if full_history.shape[1] != full_history_dim:
        raise ValueError(f"History width must be {full_history_dim}.")
    if expected_history_dim == full_history_dim:
        return full_history

    no_object_dim = (full_observation_dim - object_sensor_width) * history_length
    if expected_history_dim == no_object_dim:
        frames = full_history.view(full_history.shape[0], history_length, full_observation_dim)
        return frames[:, :, object_sensor_width:].reshape(full_history.shape[0], no_object_dim)
    raise ValueError(
        f"Policy expects history width {expected_history_dim}; adapter can provide "
        f"{full_history_dim} or {no_object_dim}."
    )


def _module_input_output_dims(module: torch.nn.Module, label: str) -> tuple[int, int]:
    linear_weights = [parameter for parameter in module.parameters() if parameter.ndim == 2]
    if not linear_weights:
        raise ValueError(f"{label} TorchScript module contains no linear weight matrices.")
    return int(linear_weights[0].shape[1]), int(linear_weights[-1].shape[0])
