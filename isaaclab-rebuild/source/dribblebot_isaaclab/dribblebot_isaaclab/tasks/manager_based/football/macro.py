"""Rate adapter between manager-based low-level ticks and coordinator actions.

The underlying :class:`ManagerBasedRLEnv` remains the authoritative 20 ms
manager step.  This wrapper owns only the slower coordinator rate: one external
``step`` executes a fixed number of low-level manager steps, sums rewards, and
ORs termination signals.  Keeping this seam outside ``ActionTerm`` preserves
the manager lifecycle and makes the timing contract explicit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym


class MacroActionWrapper(gym.Wrapper):
    """Execute one coordinator action for a fixed number of manager steps."""

    def __init__(self, env: gym.Env, control_interval: int = 10):
        super().__init__(env)
        interval = int(control_interval)
        if interval <= 0:
            raise ValueError("control_interval must be a positive integer")
        self.control_interval = interval
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.device = getattr(env, "device", None)
        self.last_low_level_steps = None
        self.last_terminated = None
        self.last_truncated = None

    def reset(self, **kwargs: Any):
        result = self.env.reset(**kwargs)
        self.last_low_level_steps = None
        self.last_terminated = None
        self.last_truncated = None
        return result

    def step(self, action: Any):
        """Run the low-level loop and return a Gymnasium five-tuple.

        Completed rows continue to receive zero actions so vectorized
        manager environments can perform their normal internal reset. Their
        rewards are masked after the first terminal tick, matching the legacy
        high-level wrapper's macro-step semantics.
        """

        total_reward = None
        terminated_total = None
        truncated_total = None
        done_total = None
        elapsed = self._zeros_long_like(action)
        info: dict[str, Any] = {}
        observation = None

        for _ in range(self.control_interval):
            active = self._active_mask(done_total)
            elapsed = elapsed + active.to(dtype=elapsed.dtype)
            low_level_action = self._mask_action(action, active)
            result = self.env.step(low_level_action)
            if len(result) != 5:
                raise RuntimeError(
                    "MacroActionWrapper requires a Gymnasium five-tuple from "
                    "ManagerBasedRLEnv.step()."
                )
            observation, reward, terminated, truncated, low_level_info = result
            terminated = self._as_bool_tensor(terminated)
            truncated = self._as_bool_tensor(truncated)
            done = terminated | truncated

            if total_reward is None:
                total_reward = self._zeros_like(reward)
                terminated_total = self._zeros_like(terminated)
                truncated_total = self._zeros_like(truncated)
                done_total = self._zeros_like(done)
            total_reward = total_reward + self._mask_value(reward, active)
            terminated_total = terminated_total | (terminated & active)
            truncated_total = truncated_total | (truncated & active)
            done_total = done_total | (done & active)
            if isinstance(low_level_info, Mapping):
                info = dict(low_level_info)

            if self._all(done_total):
                break

        if total_reward is None:
            raise RuntimeError("MacroActionWrapper did not execute a low-level step")
        self.last_low_level_steps = elapsed
        self.last_terminated = terminated_total
        self.last_truncated = truncated_total
        info["macro_control_interval"] = self.control_interval
        info["elapsed_low_level_steps"] = elapsed
        info["macro_terminated"] = terminated_total
        info["macro_truncated"] = truncated_total
        return observation, total_reward, terminated_total, truncated_total, info

    def _active_mask(self, done: Any):
        if done is None:
            return self._ones_bool_like(self._template_value())
        return ~done

    def _template_value(self):
        if self.device is not None:
            import torch

            return torch.zeros(self.num_envs, device=self.device)
        return [False] * self.num_envs

    def _zeros_long_like(self, value: Any):
        if self.device is not None:
            import torch

            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        try:
            import numpy as np

            return np.zeros(self.num_envs, dtype=np.int64)
        except ImportError:  # pragma: no cover - numpy is a runtime dependency
            return [0] * self.num_envs

    def _zeros_like(self, value: Any):
        if hasattr(value, "clone"):
            return value.clone().zero_()
        try:
            import numpy as np

            return np.zeros_like(value)
        except (ImportError, TypeError):
            return 0.0

    def _ones_bool_like(self, value: Any):
        if hasattr(value, "clone"):
            return value.clone().bool().fill_(True)
        try:
            import numpy as np

            return np.ones_like(value, dtype=bool)
        except (ImportError, TypeError):
            return True

    def _as_bool_tensor(self, value: Any):
        if hasattr(value, "bool"):
            return value.bool()
        try:
            import numpy as np

            return np.asarray(value, dtype=bool)
        except (ImportError, TypeError):
            return bool(value)

    def _mask_action(self, action: Any, active: Any):
        if hasattr(action, "clone"):
            result = action.clone()
            result[~active] = 0.0
            return result
        try:
            import numpy as np

            result = np.array(action, copy=True)
            result[~active] = 0.0
            return result
        except (ImportError, TypeError):
            return action if bool(active) else 0.0

    def _mask_value(self, value: Any, active: Any):
        if hasattr(value, "clone"):
            return value * active.to(dtype=value.dtype)
        try:
            import numpy as np

            return np.asarray(value) * np.asarray(active, dtype=np.asarray(value).dtype)
        except (ImportError, TypeError):
            return value if bool(active) else 0.0

    def _all(self, value: Any) -> bool:
        if hasattr(value, "all"):
            return bool(value.all().detach().cpu().item())
        try:
            import numpy as np

            return bool(np.asarray(value).all())
        except (ImportError, TypeError):
            return bool(value)


def make_manager_based_macro_env(cfg=None, render_mode=None, **kwargs):
    """Gym entry point that wraps a manager-based environment at coordinator rate."""

    if cfg is None:
        raise ValueError("make_manager_based_macro_env requires an Isaac Lab cfg")
    from isaaclab.envs import ManagerBasedRLEnv

    raw_env = ManagerBasedRLEnv(cfg=cfg, render_mode=render_mode, **kwargs)
    interval = int(getattr(cfg, "macro_control_interval", 10))
    return MacroActionWrapper(raw_env, control_interval=interval)


__all__ = ["MacroActionWrapper", "make_manager_based_macro_env"]
