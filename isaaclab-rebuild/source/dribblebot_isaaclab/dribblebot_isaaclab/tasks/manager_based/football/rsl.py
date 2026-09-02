"""RSL-RL adapter for the agent-expanded match self-play wrapper."""

from __future__ import annotations

import copy
from typing import Any

import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from .hybrid_actor_critic import HYBRID_ACTION_DIM, HybridActorCritic, actor_output_to_hybrid_action


class HybridOnPolicyRunner(OnPolicyRunner):
    """Resolve the project hybrid policy without modifying RSL-RL on disk."""

    def _construct_algorithm(self, obs):
        # RSL-RL 3.0.1 resolves ``policy.class_name`` with eval() in this
        # module's parent implementation. Keep that external-version quirk in
        # one adapter seam rather than leaking registration into launchers.
        import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

        rsl_on_policy_runner.HybridActorCritic = HybridActorCritic
        return super()._construct_algorithm(obs)


class FrozenRslRlOpponentPolicy:
    """Detached deterministic actor snapshot with the self-play callable API."""

    def __init__(self, actor, device: str | torch.device = "cpu"):
        # Self-play inference does not need the critic or stochastic action
        # distribution. Keeping only these two modules mirrors the Isaac Gym
        # FrozenOpponentPolicy and avoids copying cached non-leaf tensors.
        self.actor_body = copy.deepcopy(actor.actor).to(device).eval()
        self.actor_obs_normalizer = copy.deepcopy(actor.actor_obs_normalizer).to(device).eval()
        self.device = torch.device(device)
        for parameter in self.actor_body.parameters():
            parameter.requires_grad_(False)
        for parameter in self.actor_obs_normalizer.parameters():
            parameter.requires_grad_(False)

    def __call__(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        history = observation["obs_history"].to(self.device)
        with torch.inference_mode():
            actions = self.actor_body(self.actor_obs_normalizer(history))
        if actions.ndim != 2:
            raise RuntimeError(f"RSL-RL opponent actor returned shape {tuple(actions.shape)}")
        if actions.shape[-1] != 6:
            raise RuntimeError(
                f"RSL-RL opponent actor returned shape {tuple(actions.shape)}, expected actor output (N,6)"
            )
        # Both archived Gaussian-logit actors and the hybrid actor use the same
        # deterministic six-output head: three skill logits plus three means.
        actions = actor_output_to_hybrid_action(actions)
        if actions.shape[-1] != HYBRID_ACTION_DIM:
            raise RuntimeError(f"Opponent action adapter returned shape {tuple(actions.shape)}, expected (N,4)")
        return torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)

    def state_dict(self) -> dict[str, torch.Tensor]:
        payload = {
            f"actor.{key}": value.detach().cpu()
            for key, value in self.actor_body.state_dict().items()
        }
        payload.update(
            {
                f"actor_obs_normalizer.{key}": value.detach().cpu()
                for key, value in self.actor_obs_normalizer.state_dict().items()
            }
        )
        return payload

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in state_dict.items()
            if key.startswith("actor.")
        }
        normalizer_state = {
            key.removeprefix("actor_obs_normalizer."): value
            for key, value in state_dict.items()
            if key.startswith("actor_obs_normalizer.")
        }
        if not actor_state:
            raise ValueError("Opponent snapshot contains no actor.* weights")
        self.actor_body.load_state_dict(actor_state, strict=True)
        self.actor_obs_normalizer.load_state_dict(normalizer_state, strict=True)


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
    checkpoint_infos: dict[str, Any] | None = None,
    snapshot_template: FrozenRslRlOpponentPolicy | None = None,
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
    pool_payload = None
    if isinstance(checkpoint_infos, dict):
        pool_payload = checkpoint_infos.get("dribblebot_opponent_pool")
    # An explicitly configured rule/checkpoint opponent has higher priority
    # than a learned pool embedded in the resumed learner checkpoint.
    if str(getattr(raw_self_play.cfg, "opponent_mode", "zero")) != "zero":
        pool_payload = None
        initial_snapshot = False
    if pool_payload is not None:
        policies_state = pool_payload.get("policies", [])
        iterations = pool_payload.get("iterations", [])
        if len(policies_state) != len(iterations) or not policies_state:
            raise ValueError("Checkpoint opponent pool is empty or malformed")
        print(
            f"[SELFPLAY] restoring opponent pool entries={len(policies_state)} "
            f"iterations={list(iterations)}",
            flush=True,
        )
        reconstructed = []
        for index, state_dict in enumerate(policies_state):
            print(f"[SELFPLAY] cloning opponent template {index + 1}/{len(policies_state)}", flush=True)
            if snapshot_template is None:
                snapshot = FrozenRslRlOpponentPolicy(current_policy(), device=policy_device)
            elif index == 0:
                snapshot = snapshot_template
            else:
                snapshot = copy.deepcopy(snapshot_template)
            print(f"[SELFPLAY] loading opponent weights {index + 1}/{len(policies_state)}", flush=True)
            snapshot.load_state_dict(state_dict)
            reconstructed.append(snapshot)
        raw_self_play.restore_opponent_pool(reconstructed, [int(value) for value in iterations])
        print("[SELFPLAY] opponent pool restore complete", flush=True)
    elif initial_snapshot and getattr(raw_self_play, "opponent_policy_callable", None) is None:
        snapshot = snapshot_template or FrozenRslRlOpponentPolicy(current_policy(), device=policy_device)
        if snapshot_template is not None:
            snapshot.load_state_dict(current_policy().state_dict())
        raw_self_play.update_opponent_snapshot(
            snapshot,
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

    original_save = runner.save

    def save_with_opponent_pool(path: str, infos=None):
        payload = dict(infos) if isinstance(infos, dict) else {}
        pool_state = raw_self_play.opponent_pool_state_dict()
        if pool_state is not None:
            payload["dribblebot_opponent_pool"] = pool_state
        return original_save(path, infos=payload or None)

    runner.save = save_with_opponent_pool


__all__ = [
    "FrozenRslRlOpponentPolicy",
    "HybridOnPolicyRunner",
    "MatchSelfPlayRslRlVecEnvWrapper",
    "install_opponent_snapshot_schedule",
]
