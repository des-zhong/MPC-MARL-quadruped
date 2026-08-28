"""TorchScript adapter for the archived 34D/4-frame coordinator policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from .paths import checkpoint_root
except ImportError:  # simulator-free standalone module loading
    def checkpoint_root() -> Path:
        return Path(__file__).resolve().parents[5] / "checkpoints" / "reproduction"


@dataclass(frozen=True)
class CoordinatorPolicySpec:
    body_path: Path
    adaptation_path: Path
    history_dim: int = 136
    action_dim: int = 6
    action_clip: float = 10.0


class FrozenCoordinatorPolicy:
    """Inference-only high-level coordinator body plus adaptation module."""

    def __init__(self, spec: CoordinatorPolicySpec, body: torch.nn.Module, adaptation: torch.nn.Module, device: str):
        self.spec = spec
        self.device = torch.device(device)
        self.body = body.eval()
        self.adaptation = adaptation.eval()

    @classmethod
    def load(cls, root: str | Path | None = None, device: str = "cpu") -> "FrozenCoordinatorPolicy":
        directory = Path(root).expanduser().resolve() if root is not None else checkpoint_root() / "high_level"
        spec = CoordinatorPolicySpec(
            body_path=directory / "body_latest.jit",
            adaptation_path=directory / "adaptation_module_latest.jit",
        )
        if not spec.body_path.is_file() or not spec.adaptation_path.is_file():
            raise FileNotFoundError(
                f"Coordinator checkpoint requires {spec.body_path} and {spec.adaptation_path}"
            )
        body = torch.jit.load(str(spec.body_path), map_location=device)
        adaptation = torch.jit.load(str(spec.adaptation_path), map_location=device)
        return cls(spec, body, adaptation, device)

    def __call__(self, observation: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        history = observation["obs_history"] if isinstance(observation, dict) else observation
        if history.ndim != 2 or history.shape[-1] != self.spec.history_dim:
            raise ValueError(
                f"Coordinator expects history shape (N,{self.spec.history_dim}), got {tuple(history.shape)}"
            )
        history = history.to(self.device)
        valid = torch.isfinite(history).all(dim=-1)
        safe = torch.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.inference_mode():
            latent = self.adaptation.forward(safe)
            action = self.body.forward(torch.cat((safe, latent), dim=-1))
        if action.shape != (history.shape[0], self.spec.action_dim):
            raise RuntimeError(f"Coordinator returned shape {tuple(action.shape)}")
        action = torch.nan_to_num(action, nan=0.0, posinf=self.spec.action_clip, neginf=-self.spec.action_clip)
        action = action.clamp(-self.spec.action_clip, self.spec.action_clip)
        action[~valid] = 0.0
        return action


__all__ = ["CoordinatorPolicySpec", "FrozenCoordinatorPolicy"]
