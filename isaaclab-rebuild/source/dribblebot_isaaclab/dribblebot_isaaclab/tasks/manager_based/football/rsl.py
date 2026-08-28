"""RSL-RL adapter for the agent-expanded match self-play wrapper."""

from __future__ import annotations

import copy
from typing import Any

import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv


class FrozenRslRlOpponentPolicy:
    """Detached deterministic actor snapshot with the self-play callable API."""

    def __init__(self, actor, device: str | torch.device = "cpu"):
        # RSL-RL caches a ``Normal`` distribution whose mean/std tensors are
        # produced by the latest forward pass.  Python deepcopy rejects those
        # non-leaf tensors after the first PPO rollout.  The distribution is
        # transient inference state, so exclude it while cloning the module.
        cached_distribution = getattr(actor, "distribution", None)
        if hasattr(actor, "distribution"):
            actor.distribution = None
        try:
            self.actor = copy.deepcopy(actor).to(device).eval()
        finally:
            if hasattr(actor, "distribution"):
                actor.distribution = cached_distribution
        self.device = torch.device(device)
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    def __call__(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        history = observation["obs_history"].to(self.device)
        privileged = observation.get("privileged_obs", history[..., :34]).to(self.device)
        tensor_dict = TensorDict(
            {"policy": history, "critic": privileged},
            batch_size=[history.shape[0]],
            device=self.device,
        )
        with torch.inference_mode():
            if hasattr(self.actor, "act_inference"):
                actions = self.actor.act_inference(tensor_dict)
            else:
                actions = self.actor(tensor_dict, stochastic_output=False)
        if actions.ndim != 2 or actions.shape[-1] != 6:
            raise RuntimeError(f"RSL-RL opponent actor returned shape {tuple(actions.shape)}, expected (N,6)")
        return torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)


class MatchSelfPlayRslRlVecEnvWrapper(VecEnv):
    """Expose ``MatchSelfPlayWrapper`` with RSL-RL's vector-env contract.

    Isaac Lab's stock ``RslRlVecEnvWrapper`` intentionally validates the
    underlying object as a single ``ManagerBasedRLEnv`` and derives
    ``num_envs`` from the physical match count.  Self-play expands each match
    into one learning sample per team robot, so this adapter owns that shape
    conversion explicitly and leaves physics/manager lifecycle untouched.
    """

    def __init__(self, env, clip_actions: float | None = None, *, auto_reset: bool = True):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.num_actions)
        self.num_train_envs = int(getattr(env, "num_train_envs", self.num_envs))
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.cfg = env.cfg
        self._observations: TensorDict | None = None
        if auto_reset:
            self.env.reset()
        else:
            # ``train_self_play.py`` can hand us an already-reset environment
            # to avoid a second synchronous PhysX reset during app startup.
            observation = self.env.get_observations()
            if observation is None:
                raise RuntimeError("auto_reset=False requires an already-reset environment")
            self._observations = self._tensor_dict(observation)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    def _tensor_dict(self, observation: dict[str, torch.Tensor]) -> TensorDict:
        history = observation["obs_history"]
        privileged = observation["privileged_obs"]
        payload = {
            # The coordinator actor consumes the archived 136D history.
            "policy": history,
            # Keep critic input explicit; a future privileged critic can add
            # state features without changing the actor contract.
            "critic": privileged,
            "obs": observation["obs"],
            "obs_history": history,
        }
        return TensorDict(payload, batch_size=[self.num_envs], device=self.device)

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        observation, info = self.env.reset()
        self._observations = self._tensor_dict(observation)
        return self._observations, info

    def get_observations(self) -> TensorDict:
        if self._observations is None:
            self._observations = self._tensor_dict(self.env.get_observations())
        return self._observations

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -float(self.clip_actions), float(self.clip_actions))
        observation, reward, terminated, truncated, info = self.env.step(actions.to(self.device))
        self._observations = self._tensor_dict(observation)
        extras = dict(info)
        extras["time_outs"] = truncated
        return self._observations, reward, (terminated | truncated).to(dtype=torch.long), extras

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int = -1) -> int:
        if hasattr(self.env, "seed"):
            result = self.env.seed(seed)
            return int(seed if result is None else result)
        if hasattr(self.env, "reset"):
            self.env.reset(seed=seed)
        return int(seed)

    def set_opponent_callable(self, policy) -> None:
        self.env.set_opponent_callable(policy)

    def update_opponent_snapshot(self, policy, iteration: int, force: bool = False) -> bool:
        return self.env.update_opponent_snapshot(policy, iteration=iteration, force=force)

    def load_opponent_checkpoint(self, root=None, device: str = "cpu", iteration: int = -1) -> None:
        self.env.load_opponent_checkpoint(root=root, device=device, iteration=iteration)


def install_opponent_snapshot_schedule(
    runner,
    env: MatchSelfPlayRslRlVecEnvWrapper,
    *,
    interval: int = 500,
    policy_device: str = "cpu",
    initial_snapshot: bool = True,
) -> None:
    """Attach a detached actor snapshot callback to an RSL-RL runner.

    ``OnPolicyRunner`` has no self-play callback. Wrapping only its optimizer
    update method keeps rollout/storage semantics unchanged while making the
    snapshot cadence explicit and unit-testable.
    """

    interval = int(interval)
    if interval <= 0:
        raise ValueError("opponent snapshot interval must be positive")

    def current_policy():
        policy = getattr(runner.alg, "policy", None)
        if policy is not None:
            return policy
        # Kept for lightweight test doubles and older RSL-RL adapters.
        return runner.alg.get_policy()

    raw_self_play = env.env
    next_iteration = int(getattr(runner, "current_learning_iteration", 0))
    if initial_snapshot and getattr(raw_self_play, "opponent_policy_callable", None) is None:
        raw_self_play.update_opponent_snapshot(
            FrozenRslRlOpponentPolicy(current_policy(), device=policy_device),
            iteration=next_iteration,
            force=True,
        )

    original_update = runner.alg.update

    def update_with_opponent_snapshot(*args, **kwargs):
        nonlocal next_iteration
        result = original_update(*args, **kwargs)
        next_iteration += 1
        if next_iteration % interval == 0:
            raw_self_play.update_opponent_snapshot(
                FrozenRslRlOpponentPolicy(current_policy(), device=policy_device),
                iteration=next_iteration,
                force=True,
            )
        return result

    runner.alg.update = update_with_opponent_snapshot


__all__ = [
    "FrozenRslRlOpponentPolicy",
    "MatchSelfPlayRslRlVecEnvWrapper",
    "install_opponent_snapshot_schedule",
]
