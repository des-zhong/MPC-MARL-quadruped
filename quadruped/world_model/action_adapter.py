"""Hybrid multi-robot action validation, normalization, and sampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Mapping, Sequence, Union

import torch
import math


class Skill(IntEnum):
    REPOSITION = 0  # Named "walk" by the existing low-level wrapper.
    DRIBBLE = 1
    SHOOT = 2


@dataclass(frozen=True)
class SkillBounds:
    low: tuple[float, float, float]
    high: tuple[float, float, float]
    mask: tuple[float, float, float]


class JointActionAdapter:
    """Canonical action layout: N repetitions of [skill_id, p0, p1, p2]."""

    params_per_robot = 3

    def __init__(self, bounds: Mapping[int, SkillBounds], num_robots: int = 2):
        self.num_robots = int(num_robots)
        if self.num_robots < 1:
            raise ValueError("num_robots must be at least 1")
        self.action_dim = 4 * self.num_robots
        keys = {int(key) for key in bounds}
        missing = set(range(3)) - keys
        extra = keys - set(range(3))
        if missing or extra:
            raise ValueError(
                f"Skill bounds require exactly IDs 0, 1, 2; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        validated = {}
        for key, value in bounds.items():
            skill = int(key)
            if any(len(tuple(getattr(value, name))) != 3 for name in ("low", "high", "mask")):
                raise ValueError(f"Skill {skill} bounds and mask must each contain three values")
            low = tuple(float(item) for item in value.low)
            high = tuple(float(item) for item in value.high)
            mask = tuple(float(item) for item in value.mask)
            if not all(math.isfinite(item) for item in low + high + mask):
                raise ValueError(f"Skill {skill} bounds contain NaN or infinite values")
            if any(lo > hi for lo, hi in zip(low, high)):
                raise ValueError(f"Skill {skill} has a lower bound above its upper bound")
            if any(item not in (0.0, 1.0) for item in mask):
                raise ValueError(f"Skill {skill} parameter masks must contain only 0 or 1")
            if any(not active and (lo != 0.0 or hi != 0.0) for lo, hi, active in zip(low, high, mask)):
                raise ValueError(f"Skill {skill} masked parameters must have zero bounds")
            validated[skill] = SkillBounds(low, high, mask)
        self.bounds = validated

    @classmethod
    def from_env(cls, env: Any) -> "JointActionAdapter":
        raw = getattr(env, "env", env)
        cfg = raw.cfg.env
        scales = (
            getattr(cfg, "high_level_walk_command_scale"),
            getattr(cfg, "high_level_dribble_command_scale"),
            getattr(cfg, "high_level_shoot_command_scale"),
        )
        bounds = {}
        for skill_id, values in enumerate(scales):
            high = tuple(abs(float(v)) for v in values)
            mask = tuple(float(v > 0.0) for v in high)
            bounds[skill_id] = SkillBounds(tuple(-v for v in high), high, mask)
        return cls(bounds, int(getattr(raw, "num_robots", getattr(cfg, "num_robots", 2))))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JointActionAdapter":
        return cls(
            {int(k): SkillBounds(**v) for k, v in payload["bounds"].items()},
            int(payload.get("num_robots", 2)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_robots": self.num_robots,
            "layout": [
                item
                for robot in range(self.num_robots)
                for item in (f"robot_{robot}.skill_id", f"robot_{robot}.parameters[3]")
            ],
            "skill_names": {"0": "reposition/walk", "1": "dribble", "2": "shoot"},
            "bounds": {str(k): {"low": list(v.low), "high": list(v.high), "mask": list(v.mask)} for k, v in self.bounds.items()},
        }

    def _unpack_unchecked(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action last dimension {self.action_dim}, got {actions.shape[-1]}")
        shaped = actions.reshape(*actions.shape[:-1], self.num_robots, 4)
        skills_float = shaped[..., 0]
        skills = skills_float.long()
        return skills, shaped[..., 1:]

    def unpack(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skills, parameters = self._unpack_unchecked(actions)
        skills_float = actions.reshape(*actions.shape[:-1], self.num_robots, 4)[..., 0]
        if not torch.allclose(skills_float, skills.to(skills_float.dtype)):
            raise ValueError("Skill IDs must be integer-valued")
        self.validate_skill_ids(skills)
        return skills, parameters

    def pack(self, skills: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        self.validate_skill_ids(skills)
        if skills.shape[-1] != self.num_robots or parameters.shape != skills.shape + (3,):
            raise ValueError(
                f"Expected skills [...,{self.num_robots}] and parameters "
                f"[... ,{self.num_robots},3], got {skills.shape}, {parameters.shape}"
            )
        return torch.cat((skills.to(parameters.dtype).unsqueeze(-1), parameters), dim=-1).flatten(-2)

    @staticmethod
    def validate_skill_ids(skill_ids: torch.Tensor) -> None:
        invalid = (skill_ids < 0) | (skill_ids > 2)
        if bool(torch.any(invalid).item()):
            values = torch.unique(skill_ids[invalid]).detach().cpu().tolist()
            raise ValueError(f"Invalid skill IDs {values}; valid IDs are 0, 1, 2")

    def _selected(self, skills: torch.Tensor, field: str, dtype: torch.dtype) -> torch.Tensor:
        table = torch.tensor([getattr(self.bounds[i], field) for i in range(3)], device=skills.device, dtype=dtype)
        return table[skills]

    def masks(self, actions: torch.Tensor) -> torch.Tensor:
        skills, params = self.unpack(actions)
        return self._selected(skills, "mask", params.dtype)

    def clip(self, actions: torch.Tensor) -> torch.Tensor:
        skills, params = self.unpack(actions)
        low = self._selected(skills, "low", params.dtype)
        high = self._selected(skills, "high", params.dtype)
        mask = self._selected(skills, "mask", params.dtype)
        return self.pack(skills, torch.maximum(torch.minimum(params, high), low) * mask)

    def assert_within_bounds(self, actions: torch.Tensor, atol: float = 1e-6) -> None:
        if not bool(torch.isfinite(actions).all().item()):
            raise ValueError("Actions contain NaN or infinite values")
        skills, params = self.unpack(actions)
        low = self._selected(skills, "low", params.dtype)
        high = self._selected(skills, "high", params.dtype)
        mask = self._selected(skills, "mask", params.dtype)
        if bool(torch.any((params < low - atol) | (params > high + atol)).item()):
            raise ValueError("Action parameters lie outside their skill-dependent bounds")
        if bool(torch.any(torch.abs(params * (1.0 - mask)) > atol).item()):
            raise ValueError("Unused action parameters must be zero")

    def normalize_parameters(self, skills: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        low = self._selected(skills, "low", parameters.dtype)
        high = self._selected(skills, "high", parameters.dtype)
        mask = self._selected(skills, "mask", parameters.dtype)
        span = (high - low).clamp(min=1e-6)
        return (2.0 * (parameters - low) / span - 1.0) * mask

    def denormalize_parameters(self, skills: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
        low = self._selected(skills, "low", normalized.dtype)
        high = self._selected(skills, "high", normalized.dtype)
        mask = self._selected(skills, "mask", normalized.dtype)
        return (low + 0.5 * (normalized.clamp(-1.0, 1.0) + 1.0) * (high - low)) * mask

    def normalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        skills, params = self.unpack(actions)
        return self.pack(skills, self.normalize_parameters(skills, params))

    def denormalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        skills, params = self.unpack(actions)
        return self.pack(skills, self.denormalize_parameters(skills, params))

    def random_valid(self, shape: Sequence[int], device: Union[torch.device, str] = "cpu", generator=None) -> torch.Tensor:
        skills = torch.randint(
            0,
            3,
            tuple(shape) + (self.num_robots,),
            device=device,
            generator=generator,
        )
        unit = 2.0 * torch.rand(
            tuple(shape) + (self.num_robots, 3),
            device=device,
            generator=generator,
        ) - 1.0
        params = self.denormalize_parameters(skills, unit)
        return self.pack(skills, params)

    def to_wrapper_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert canonical physical commands to wrapper logits/raw tanh inputs."""

        self.assert_within_bounds(actions)
        skills, params = self.unpack(actions)
        scales = self._selected(skills, "high", params.dtype).clamp(min=1e-6)
        raw = torch.atanh((params / scales).clamp(-0.999999, 0.999999))
        logits = torch.nn.functional.one_hot(skills, num_classes=3).to(params.dtype) * 10.0
        return torch.cat((logits, raw), dim=-1).flatten(-2)
