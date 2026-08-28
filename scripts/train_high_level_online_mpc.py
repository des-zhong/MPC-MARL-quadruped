"""Online high-level self-play with a replay-trained world model and MPC.

This entry point deliberately lives beside ``train_high_level.py``.  The
existing PPO/self-play pipeline remains the default; this script opts into a
small runner extension which records global macro transitions, updates the
world model and centralized terminal-value model from bounded replay, and
distills the first-step MPC distribution back into the decentralized hybrid
policy.

The low-level Walk/Dribble/Shoot policies are loaded by the existing
high-level setup and are never modified.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Mapping, Optional

if "torch" in sys.modules:
    isaacgym = sys.modules.get("isaacgym")
else:
    try:
        import isaacgym
    except ImportError:
        isaacgym = None

import torch
import torch.nn.functional as F

from dribblebot.mpc.config import load_mpc_config
from dribblebot.mpc.hybrid_cem import HybridCEMMPC
from dribblebot.mpc.objective import MPCObjective
from dribblebot.mpc.terminal_value import (
    ReturnNormalizer,
    TerminalValueModel,
    ValueModelConfig,
    load_value_checkpoint,
    save_value_checkpoint,
)
from dribblebot.world_model.ensemble import WorldModelEnsemble
from dribblebot.world_model.losses import feature_group_weights, one_step_member_loss
from dribblebot.world_model.normalizer import WorldModelNormalizer
from dribblebot.world_model.schema import EVENT_NAMES
from dribblebot.world_model.state_adapter import FootballWorldModelStateAdapter
from dribblebot.world_model.trainer import load_checkpoint, save_checkpoint


class WorldModelReplayBuffer:
    """Bounded transition replay with an explicit historical sampling mix.

    A ring buffer alone can over-sample the newest opponent when its capacity
    is small.  ``recent_fraction`` and ``recent_window`` reserve part of every
    minibatch for recent data while sampling the remainder from older entries,
    retaining opponent diversity until entries actually age out.
    """

    MODEL_KEYS = (
        "state", "joint_action", "reward", "next_state", "terminated",
        "truncated", "event_labels",
    )

    def __init__(self, capacity: int = 100_000, recent_fraction: float = 0.5,
                 recent_window: int = 10_000, seed: int = 42):
        if int(capacity) < 1:
            raise ValueError("replay capacity must be positive")
        if not 0.0 <= float(recent_fraction) <= 1.0:
            raise ValueError("recent_fraction must lie in [0, 1]")
        self.capacity = int(capacity)
        self.recent_fraction = float(recent_fraction)
        self.recent_window = max(1, min(int(recent_window), self.capacity))
        self._items = deque(maxlen=self.capacity)
        self._generator = torch.Generator().manual_seed(int(seed))

    def __len__(self):
        return len(self._items)

    def add_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        if not batch:
            return
        lengths = [int(value.shape[0]) for value in batch.values() if torch.is_tensor(value)]
        if not lengths or len(set(lengths)) != 1:
            raise ValueError("replay batch tensors must share a non-empty first dimension")
        count = lengths[0]
        for row in range(count):
            item = {}
            for key, value in batch.items():
                if torch.is_tensor(value):
                    item[key] = value[row].detach().cpu().clone()
                else:
                    item[key] = value[row] if isinstance(value, (list, tuple)) else value
            self._items.append(item)

    def _sample_indices(self, count: int) -> torch.Tensor:
        size = len(self._items)
        if size < 1:
            raise ValueError("cannot sample an empty replay buffer")
        count = int(count)
        recent_start = max(0, size - self.recent_window)
        recent_count = min(count, int(round(count * self.recent_fraction)))
        historical_count = count - recent_count
        recent = torch.randint(
            recent_start, size, (recent_count,), generator=self._generator
        ) if recent_count else torch.empty(0, dtype=torch.long)
        if historical_count and recent_start:
            historical = torch.randint(
                0, recent_start, (historical_count,), generator=self._generator
            )
        elif historical_count:
            historical = torch.randint(
                0, size, (historical_count,), generator=self._generator
            )
        else:
            historical = torch.empty(0, dtype=torch.long)
        return torch.cat((recent, historical), dim=0)

    def sample(self, count: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        indices = self._sample_indices(count)
        # Deque indexing is linear. Materialize references once so a large
        # historical buffer still has O(size + batch) rather than O(size*batch)
        # sampling cost.
        items = list(self._items)
        rows = [items[int(index)] for index in indices]
        keys = rows[0].keys()
        result = {}
        for key in keys:
            values = [row[key] for row in rows]
            if torch.is_tensor(values[0]):
                result[key] = torch.stack(values).to(device) if device is not None else torch.stack(values)
        return result

    def model_sample(self, count: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        result = self.sample(count, device)
        missing = [key for key in self.MODEL_KEYS if key not in result]
        if missing:
            raise ValueError(f"replay transition is missing world-model fields {missing}")
        return result


class OnlineWorldModelTrainer:
    """One-step ensemble updates on a changing transition replay."""

    def __init__(self, model, replay: WorldModelReplayBuffer, device="cuda",
                 learning_rate=3e-4, weight_decay=1e-5, batch_size=1024,
                 updates_per_interval=8, gradient_clip_norm=10.0,
                 optimizer_states=None, loss_config=None):
        self.model = model
        self.replay = replay
        requested = str(device)
        self.device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = max(1, int(batch_size))
        self.updates_per_interval = max(1, int(updates_per_interval))
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.optimizers = [
            torch.optim.AdamW(member.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
            for member in self.model.members
        ]
        if optimizer_states is not None:
            if len(optimizer_states) != len(self.optimizers):
                raise ValueError("world-model checkpoint optimizer count is incompatible")
            for optimizer, state in zip(self.optimizers, optimizer_states):
                optimizer.load_state_dict(state)
        self.loss_config = dict(loss_config or {})
        self.feature_weights = feature_group_weights(
            self.model.schema, self.loss_config, self.device
        )

    def update(self) -> Dict[str, float]:
        if len(self.replay) < max(2, self.batch_size):
            return {"world_model/update_skipped": 1.0, "world_model/replay_size": float(len(self.replay))}
        self.model.train()
        totals: Dict[str, float] = {}
        updates = 0
        for _ in range(self.updates_per_interval):
            batch = self.replay.model_sample(self.batch_size, self.device)
            for member_index, optimizer in enumerate(self.optimizers):
                size = batch["state"].shape[0]
                indices = torch.randint(size, (size,), device=self.device)
                bootstrap = {
                    key: value[indices] if torch.is_tensor(value) and value.shape[0] == size else value
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                losses = one_step_member_loss(
                    self.model, member_index, bootstrap, self.feature_weights,
                    reward_weight=float(self.loss_config.get("reward_weight", 1.0)),
                    termination_weight=float(
                        self.loss_config.get("termination_weight", 1.0)
                    ),
                    event_weight=float(self.loss_config.get("event_weight", 1.0)),
                    reward_variance_weight=float(
                        self.loss_config.get("reward_variance_weight", 0.1)
                    ),
                )
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.members[member_index].parameters(), self.gradient_clip_norm
                )
                optimizer.step()
                for key, value in losses.items():
                    totals[f"world_model/{key}"] = (
                        totals.get(f"world_model/{key}", 0.0)
                        + float(value.detach())
                    )
                updates += 1
        self.model.eval()
        totals["world_model/replay_size"] = float(len(self.replay))
        totals["world_model/updates"] = float(updates)
        return {
            key: (
                value / max(updates, 1)
                if key not in {"world_model/replay_size", "world_model/updates"}
                else value
            )
            for key, value in totals.items()
        }


class OnlineTerminalValueTrainer:
    """Monte-Carlo centralized value trained from completed real rollouts."""

    def __init__(self, model, replay: WorldModelReplayBuffer, device="cuda",
                 learning_rate=3e-4, batch_size=1024, updates_per_interval=4,
                 gamma=0.99, gradient_clip_norm=10.0, pretrained=False,
                 optimizer_state=None, enabled=True):
        self.model = model
        self.replay = replay
        requested = str(device)
        self.device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(learning_rate))
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.batch_size = max(1, int(batch_size))
        self.updates_per_interval = max(1, int(updates_per_interval))
        self.gamma = float(gamma)
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.enabled = bool(enabled)
        self.ready = bool(pretrained) if self.enabled else True

    def update(self) -> Dict[str, float]:
        if not self.enabled:
            return {"terminal_value/enabled": 0.0}
        if len(self.replay) < max(2, self.batch_size):
            return {"terminal_value/update_skipped": 1.0}
        self.model.train()
        total = 0.0
        for _ in range(self.updates_per_interval):
            if len(self.replay) < self.batch_size:
                break
            batch = self.replay.sample(self.batch_size, self.device)
            target = self.model.return_normalizer.normalize(
                batch["return_to_go"].float().reshape(-1)
            )
            predictions = self.model.forward_members(batch["state"].float())
            loss = predictions.new_zeros(())
            for member in predictions:
                mask = torch.rand_like(target) < 0.8
                if not bool(mask.any()):
                    mask[0] = True
                loss = loss + F.huber_loss(member[mask], target[mask])
            loss = loss / predictions.shape[0]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            total += float(loss.detach())
        self.ready = True
        self.model.eval()
        return {"terminal_value/loss": total / self.updates_per_interval}


def _wrapper_actions_to_canonical(wrapper_actions, match_env, action_adapter):
    """Decode high-level wrapper logits/raw commands without mutating the env."""

    if wrapper_actions.shape[-1] != 6:
        raise ValueError("expected [..., 6] high-level wrapper actions")
    skills = wrapper_actions[..., :3].argmax(dim=-1)
    scales = match_env._command_scales(skills)
    commands = torch.tanh(wrapper_actions[..., 3:6]) * scales
    commands = commands.clone()
    commands[..., 2] = torch.where(skills == 2, torch.zeros_like(commands[..., 2]), commands[..., 2])
    # These may be team-local actions while the model adapter covers both
    # teams. The caller composes this canonical [skill, command] layout into a
    # full joint action before validating it with ``action_adapter``.
    return torch.cat(
        (skills.to(commands.dtype).unsqueeze(-1), commands), dim=-1
    ).flatten(-2)


def _mpc_distribution_to_policy_targets(plan, action_adapter, team_size):
    """Convert CEM's bounded skill mixture to one raw hybrid-policy target."""

    probabilities = plan.final_skill_probabilities[:, 0, :team_size].detach()
    # CEM is parameterized in normalized command coordinates. Convert that
    # distribution to the policy's pre-tanh coordinate system. The fallback
    # keeps this helper usable with older plan records that only persisted the
    # physical final distribution.
    if hasattr(plan, "planner_state") and plan.planner_state is not None:
        normalized_means = plan.planner_state.parameter_means[
            :, 0, :team_size
        ].detach().clamp(-0.999, 0.999)
        normalized_stds = plan.planner_state.parameter_stds[
            :, 0, :team_size
        ].detach()
    else:
        physical_means = plan.final_parameter_means[:, 0, :team_size].detach()
        physical_stds = plan.final_parameter_stds[:, 0, :team_size].detach()
        low = torch.tensor(
            [action_adapter.bounds[index].low for index in range(3)],
            device=probabilities.device, dtype=physical_means.dtype,
        ).view(1, 1, 3, 3)
        high = torch.tensor(
            [action_adapter.bounds[index].high for index in range(3)],
            device=probabilities.device, dtype=physical_means.dtype,
        ).view(1, 1, 3, 3)
        normalized_means = (
            2.0 * (physical_means - low) / (high - low).clamp(min=1e-6) - 1.0
        ).clamp(-0.999, 0.999)
        normalized_stds = physical_stds / (0.5 * (high - low).clamp(min=1e-6))
    batch, robots = probabilities.shape[:2]
    raw_means_by_skill = torch.atanh(normalized_means)
    raw_stds_by_skill = normalized_stds / (1.0 - normalized_means.square()).clamp(min=0.05)
    masks = torch.tensor(
        [action_adapter.bounds[index].mask for index in range(3)],
        device=probabilities.device, dtype=normalized_means.dtype,
    )
    return (
        raw_means_by_skill.reshape(batch * robots, 3, 3),
        raw_stds_by_skill.clamp(min=1e-4).reshape(batch * robots, 3, 3),
        probabilities.reshape(batch * robots, 3),
        masks,
    )


def mpc_action_agreement_reward(
    executed,
    teacher,
    action_adapter,
    controlled_robot_count,
    coefficient,
):
    """Dense per-agent signal shared by the MPC ablation conditions."""

    coefficient = float(coefficient)
    if coefficient < 0.0:
        raise ValueError("MPC guidance reward coefficient cannot be negative")
    executed_skill, executed_parameters = action_adapter.unpack(executed)
    teacher_skill, teacher_parameters = action_adapter.unpack(teacher)
    executed_normalized = action_adapter.normalize_parameters(
        executed_skill, executed_parameters
    )
    teacher_normalized = action_adapter.normalize_parameters(
        teacher_skill, teacher_parameters
    )
    count = int(controlled_robot_count)
    same_skill = (
        executed_skill[:, :count] == teacher_skill[:, :count]
    )
    mask = action_adapter._selected(
        teacher_skill[:, :count], "mask", executed.dtype
    )
    parameter_error = (
        (
            executed_normalized[:, :count]
            - teacher_normalized[:, :count]
        ).square()
        * mask
    ).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    disagreement = (~same_skill).to(executed.dtype) + same_skill.to(
        executed.dtype
    ) * parameter_error
    reward = coefficient * (
        torch.exp(-disagreement) - executed.new_tensor(0.36787944117144233)
    )
    return reward, disagreement


class OnlineMPCSelfPlayExtension:
    """Runner hook that couples online model learning and policy distillation."""

    def __init__(self, env, match_env, world_model, planner, state_adapter,
                 replay, value_replay, terminal_trainer, world_model_trainer, args):
        self.env = env
        self.match_env = match_env
        self.world_model = world_model
        self.planner = planner
        self.state_adapter = state_adapter
        self.action_adapter = world_model.action_adapter
        self.replay = replay
        self.value_replay = value_replay
        self.terminal_trainer = terminal_trainer
        self.world_model_trainer = world_model_trainer
        self.args = args
        self.runner = None
        self._pending = None
        self._planner_state = None
        self._metrics: Dict[str, float] = {}
        self._distill_optimizer = None
        self._high_level_steps = 0
        self._episode_states = [[] for _ in range(self.train_matches)]
        self._episode_rewards = [[] for _ in range(self.train_matches)]
        # TerminalStateCapture is installed by build_online_training before
        # this extension is constructed.
        self.capture = getattr(args, "terminal_state_capture", None)

    def bind_runner(self, runner):
        self.runner = runner
        # Reuse PPO's optimizer so Adam state for the actor and exploration
        # scale has a single owner. The critic receives no distillation
        # gradient and continues to train only through PPO.
        self._distill_optimizer = runner.alg.optimizer

    @property
    def train_matches(self):
        return int(self.env.num_train_envs // self.env.team_size)

    def before_env_step(self, iteration, obs_before, actions):
        batch = self.train_matches
        state = self.state_adapter.extract_state(self.match_env)["tensor"][:batch]
        if self.capture is not None:
            self.capture.clear()
        plan = None
        if (
            self._high_level_steps >= int(self.args.mpc_warmup_steps)
            and (
                not self.planner.config.use_terminal_value
                or self.terminal_trainer.ready
            )
        ):
            opponent_wrapper = self.env.preview_opponent_actions()[:batch]
            opponent_action = _wrapper_actions_to_canonical(
                opponent_wrapper, self.match_env, self.action_adapter
            )
            horizon = int(self.planner.config.horizon)
            fixed = torch.zeros(
                batch, horizon, self.action_adapter.action_dim,
                dtype=state.dtype, device=state.device,
            )
            fixed[:, :, 4 * self.env.team_size :] = opponent_action[:, None].expand(
                -1, horizon, -1
            )
            fixed_mask = torch.zeros(self.action_adapter.num_robots, dtype=torch.bool, device=state.device)
            fixed_mask[self.env.team_size:] = True
            plan = self.planner.plan(
                state, planner_state=self._planner_state,
                fixed_action_sequence=fixed, fixed_robot_mask=fixed_mask,
            )
            self._planner_state = plan.planner_state
        self._pending = {
            "iteration": int(iteration), "state": state.detach(),
            "obs_before": {key: value.detach().clone() for key, value in obs_before.items()},
            "opponent_observation": self.env._team_observations(1)[:batch].detach().clone(),
            "opponent_obs_history": self.env._history[:batch, 1].detach().clone(),
            "actions": actions.detach().clone(), "plan": plan,
        }

    def process_env_step(self, iteration, obs, actions, rewards, dones, infos):
        if self._pending is None:
            return
        pending = self._pending
        batch = pending["state"].shape[0]
        team = self.env.team_size
        live_next = self.state_adapter.extract_state(self.match_env)["tensor"][:batch]
        if self.capture is None:
            next_state = live_next
        else:
            next_state = torch.where(
                self.capture.valid[:batch, None], self.capture.states[:batch], live_next
            )
        info = dict(infos)
        skills = torch.as_tensor(
            info["high_level_skill_ids"],
            device=pending["state"].device,
            dtype=torch.long,
        )[:batch]
        commands = torch.as_tensor(
            info["high_level_commands"],
            device=pending["state"].device,
            dtype=torch.float,
        )[:batch]
        executed = self.action_adapter.pack(skills, commands)
        self.action_adapter.assert_within_bounds(executed, atol=1e-4)
        plan = pending["plan"]
        guidance_coefficient = float(
            getattr(self.args, "mpc_guidance_reward_coefficient", 0.0)
        )
        if plan is not None and guidance_coefficient > 0.0:
            guidance, disagreement = mpc_action_agreement_reward(
                executed,
                plan.first_joint_action,
                self.action_adapter,
                team,
                guidance_coefficient,
            )
            rewards[: batch * team].add_(
                guidance.reshape(-1).to(rewards.device)
            )
            infos["mpc_guidance_reward"] = guidance.reshape(-1).detach()
            infos["mpc_guidance_disagreement"] = disagreement.reshape(
                -1
            ).detach()
        match_rewards = torch.as_tensor(
            info.get("high_level_match_rewards", rewards[:batch * team].reshape(batch, team)[:, 0]),
            device=pending["state"].device, dtype=torch.float,
        ).reshape(-1)[:batch]
        match_done = dones[:batch * team].reshape(batch, team).any(dim=1)
        timeout_values = torch.as_tensor(
            info.get("time_outs", torch.zeros(batch * team, dtype=torch.bool)),
            device=pending["state"].device,
        ).reshape(-1)[:batch * team].reshape(batch, team).any(dim=1).bool()
        terminated = match_done & ~timeout_values
        elapsed_steps = torch.as_tensor(
            info.get(
                "elapsed_low_level_steps",
                torch.full((batch,), int(self.match_env.control_interval)),
            ),
            dtype=torch.long,
        ).reshape(-1)[:batch]
        event_info = {}
        for key in (
            "high_level_goal", "high_level_ball_off_border",
            "high_level_obstacle_contact", "high_level_skill_ids",
            "shooting_success", "shooting_failure",
        ):
            if key in info:
                event_info[key] = torch.as_tensor(info[key])[:batch]
        event_labels = self.state_adapter.extract_event_labels(pending["state"], next_state, event_info)
        event_labels = event_labels[:, : self.world_model.num_events]
        requested_own = _wrapper_actions_to_canonical(
            pending["actions"].reshape(batch, team, 6), self.match_env, self.action_adapter
        )
        requested = executed.clone()
        requested[:, : 4 * team] = requested_own
        next_obs = {key: value.detach().clone() for key, value in obs.items()}
        obs_before = pending["obs_before"]
        if "opponent_pool_selected_iterations" in info:
            selected_opponent_iterations = torch.as_tensor(
                info["opponent_pool_selected_iterations"], dtype=torch.long
            ).reshape(-1)[:batch]
        else:
            selected_opponent_iterations = torch.full(
                (batch,), int(info.get("opponent_snapshot_iteration", -1)),
                dtype=torch.long,
            )
        # Store a centralized transition for dynamics/value learning.  The
        # explicit opponent action and snapshot ID make self-play provenance
        # auditable and keep older opponents in the replay distribution.
        self.replay.add_batch({
            "state": pending["state"], "joint_action": executed,
            "requested_joint_action": requested,
            "our_action": executed[:, : 4 * team],
            "opponent_action": executed[:, 4 * team :], "reward": match_rewards,
            "next_state": next_state, "terminated": terminated,
            "truncated": timeout_values, "done": match_done,
            "elapsed_low_level_steps": elapsed_steps,
            "event_labels": event_labels,
            "current_observation": obs_before["obs"].reshape(batch, team, -1),
            "next_observation": next_obs["obs"].reshape(batch, team, -1),
            "current_obs_history": obs_before["obs_history"].reshape(batch, team, -1),
            "next_obs_history": next_obs["obs_history"].reshape(batch, team, -1),
            "opponent_observation": pending["opponent_observation"],
            "opponent_obs_history": pending["opponent_obs_history"],
            "opponent_snapshot_iteration": selected_opponent_iterations,
        })
        if self.terminal_trainer.enabled:
            state_cpu = pending["state"].detach().cpu()
            rewards_cpu = match_rewards.detach().cpu().tolist()
            done_cpu = match_done.detach().cpu().tolist()
            for row in range(batch):
                self._episode_states[row].append(state_cpu[row])
                self._episode_rewards[row].append(float(rewards_cpu[row]))
                if done_cpu[row]:
                    continuation = 0.0
                    returns = []
                    for reward in reversed(self._episode_rewards[row]):
                        continuation = (
                            reward
                            + self.terminal_trainer.gamma * continuation
                        )
                        returns.append(continuation)
                    returns.reverse()
                    self.value_replay.add_batch({
                        "state": torch.stack(self._episode_states[row]),
                        "return_to_go": torch.tensor(
                            returns, dtype=torch.float
                        ),
                        "opponent_snapshot_iteration": torch.full(
                            (len(returns),),
                            int(selected_opponent_iterations[row]),
                            dtype=torch.long,
                        ),
                    })
                    self._episode_states[row].clear()
                    self._episode_rewards[row].clear()
        # A zero KL coefficient is a real ablation, not an unbounded target
        # cache.  Keep MPC planning active for pipeline parity, but do not
        # retain teacher labels when no distillation update can consume them.
        if plan is not None and float(self.args.mpc_kl_coefficient) > 0.0:
            targets = _mpc_distribution_to_policy_targets(plan, self.action_adapter, team)
            self._distill_records = getattr(self, "_distill_records", [])
            self._distill_records.append({
                "obs_history": obs_before["obs_history"].reshape(batch * team, -1).detach().cpu(),
                "target_command_mean": targets[0].detach().cpu(),
                "target_std": targets[1].detach().cpu(),
                "target_probs": targets[2].detach().cpu(),
                "target_masks": targets[3].detach().cpu()[None].expand(batch * team, -1, -1),
            })
        if self._planner_state is not None and bool(match_done.any()):
            self._planner_state.reset(match_done.nonzero(as_tuple=False).flatten())
        self._high_level_steps += 1
        self._pending = None

    def _distill_update(self, iteration):
        coefficient = float(self.args.mpc_kl_coefficient)
        if coefficient <= 0.0:
            # Defensive cleanup also handles resuming an extension that was
            # previously configured with distillation enabled.
            self._distill_records = []
            return {
                "mpc_distillation/enabled": 0.0,
                "mpc_distillation/coefficient": 0.0,
            }
        if self._high_level_steps < int(self.args.mpc_warmup_steps):
            return {}
        records = getattr(self, "_distill_records", [])
        if not records:
            return {}
        # Keep the same recent/historical policy-state mixture as dynamics
        # replay without duplicating another large ring buffer.
        merged = {key: torch.cat([record[key] for record in records], dim=0) for key in records[0]}
        self._distill_records = []
        count = min(int(self.args.mpc_distillation_batch_size), merged["obs_history"].shape[0])
        indices = torch.randperm(merged["obs_history"].shape[0])[:count]
        device = self.runner.device
        histories = merged["obs_history"][indices].to(device)
        target_command_mean = merged["target_command_mean"][indices].to(device)
        target_std = merged["target_std"][indices].to(device)
        target_probs = merged["target_probs"][indices].to(device)
        target_masks = merged["target_masks"][indices].to(device)
        if not bool(torch.isfinite(histories).all()):
            return {
                "mpc_distillation/update_skipped_nonfinite_observation": 1.0,
            }
        targets = {
            "command_mean": target_command_mean,
            "std": target_std,
            "probabilities": target_probs,
            "masks": target_masks,
        }
        if not all(bool(torch.isfinite(value).all()) for value in targets.values()):
            return {
                "mpc_distillation/update_skipped_nonfinite_target": 1.0,
            }
        # CEM maintains a probability simplex, but normalize defensively in
        # case a checkpoint or a custom planner supplies a slightly malformed
        # final distribution.
        target_probs = target_probs.clamp_min(1.0e-6)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        target_std = target_std.abs().clamp_min(1.0e-4)
        policy = self.runner.alg.actor_critic
        policy.update_distribution(histories)
        policy_mean = policy.action_mean
        policy_std = policy.action_std
        if not (
            bool(torch.isfinite(policy_mean).all())
            and bool(torch.isfinite(policy_std).all())
        ):
            raise FloatingPointError(
                "High-level policy became non-finite before MPC distillation. "
                "Do not resume from the resulting checkpoint."
            )
        target_log_probs = target_probs.clamp(min=1e-6).log()
        policy_log_probs = torch.log_softmax(policy_mean[:, :3], dim=-1)
        categorical = (target_probs * (target_log_probs - policy_log_probs)).sum(dim=-1).mean()
        policy_command_mean = policy_mean[:, 3:]
        policy_command_std = policy_std[:, 3:].clamp(min=1e-6)
        command_kl = (
            torch.log(policy_command_std[:, None])
            - torch.log(target_std.clamp(min=1e-6))
            + (
                target_std.square()
                + (target_command_mean - policy_command_mean[:, None]).square()
            ) / (2.0 * policy_command_std[:, None].square())
            - 0.5
        )
        conditional_kl = (command_kl * target_masks).sum(dim=-1) / target_masks.sum(
            dim=-1
        ).clamp(min=1.0)
        continuous = (target_probs * conditional_kl).sum(dim=-1).mean()
        loss = coefficient * (categorical + continuous)
        if not bool(torch.isfinite(loss)):
            return {
                "mpc_distillation/update_skipped_nonfinite_loss": 1.0,
            }
        self._distill_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        if not bool(torch.isfinite(grad_norm)):
            self._distill_optimizer.zero_grad(set_to_none=True)
            return {
                "mpc_distillation/update_skipped_nonfinite_gradient": 1.0,
            }
        parameter_backup = [parameter.detach().clone() for parameter in policy.parameters()]
        self._distill_optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
            with torch.no_grad():
                for parameter, backup in zip(policy.parameters(), parameter_backup):
                    parameter.copy_(backup)
                    state = self._distill_optimizer.state.get(parameter)
                    if state is not None:
                        state.clear()
            self._distill_optimizer.zero_grad(set_to_none=True)
            return {
                "mpc_distillation/update_skipped_nonfinite_parameters": 1.0,
            }
        from dribblebot_learn.ppo_cse.actor_critic import AC_Args
        with torch.no_grad():
            policy.std.clamp_(min=AC_Args.min_action_std, max=AC_Args.max_action_std)
        return {
            "mpc_distillation/kl_loss": float((categorical + continuous).detach()),
            "mpc_distillation/categorical_kl": float(categorical.detach()),
            "mpc_distillation/continuous_kl": float(continuous.detach()),
            "mpc_distillation/coefficient": coefficient,
            "mpc_distillation/enabled": 1.0,
        }

    def after_rollout(self, iteration):
        if (int(iteration) + 1) % int(self.args.world_model_update_interval) == 0:
            self._metrics = {}
            self._metrics.update(self.world_model_trainer.update())
            self._metrics.update(self.terminal_trainer.update())
            self._metrics["terminal_value/replay_size"] = float(len(self.value_replay))
            self._metrics["online/high_level_steps"] = float(self._high_level_steps)

    def after_policy_update(self, iteration):
        metrics = dict(self._metrics)
        self._metrics = {}
        metrics.update(self._distill_update(iteration))
        return metrics

    def save_checkpoint(self, path, iteration):
        output = Path(path)
        world_path = output / f"world_model_online_{iteration}.pt"
        save_checkpoint(
            world_path, self.world_model,
            self.world_model_trainer.optimizers,
            [
                torch.optim.lr_scheduler.ConstantLR(
                    opt, factor=1.0, total_iters=1
                )
                for opt in self.world_model_trainer.optimizers
            ],
            iteration,
            {},
            {
                "online": {
                    "enabled": True,
                    "replay_capacity": self.replay.capacity,
                    "update_interval": int(self.args.world_model_update_interval),
                },
                "world_model": {
                    "macro_action_steps": int(self.match_env.control_interval),
                },
                "environment": {
                    "team_size": int(self.env.team_size),
                    "num_robots": int(self.action_adapter.num_robots),
                },
            },
            0,
        )
        world_latest = output / "world_model_online_latest.pt"
        shutil.copy2(world_path, world_latest)
        saved = [str(world_path), str(world_latest)]
        if not self.terminal_trainer.enabled:
            return saved
        value_path = output / f"terminal_value_online_{iteration}.pt"
        value_config = ValueModelConfig(
            device=str(self.terminal_trainer.device),
            gamma=self.terminal_trainer.gamma,
        )
        save_value_checkpoint(
            value_path, self.terminal_trainer.model, self.terminal_trainer.optimizer,
            iteration, value_config, {}, "online_replay",
        )
        value_latest = output / "terminal_value_online_latest.pt"
        shutil.copy2(value_path, value_latest)
        saved.extend((str(value_path), str(value_latest)))
        return saved


def _build_world_model(checkpoint_path, state_adapter, config, device):
    if checkpoint_path:
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        if model.action_adapter.num_robots != state_adapter.num_robots:
            raise ValueError("world-model checkpoint robot count does not match self-play environment")
        if model.schema.to_dict() != state_adapter.schema.to_dict():
            raise ValueError("world-model checkpoint state schema does not match the live environment")
        if model.action_adapter.to_dict()["bounds"] != state_adapter.action_adapter.to_dict()["bounds"]:
            raise ValueError("world-model checkpoint action bounds do not match high-level command scales")
        model.eval()
        return model, checkpoint
    schema = state_adapter.schema
    dynamic = schema.continuous_dynamic_indices
    normalizer = WorldModelNormalizer(
        torch.zeros(schema.state_dim), torch.ones(schema.state_dim),
        torch.zeros(len(dynamic)), torch.ones(len(dynamic)),
    )
    model_cfg = dict(config.get("model", {}))
    model_cfg["num_events"] = len(EVENT_NAMES)
    model = WorldModelEnsemble(schema, state_adapter.action_adapter, normalizer, **model_cfg)
    model.event_names = EVENT_NAMES
    model.to(device).eval()
    return model, {"training_config": config}


def _build_terminal_value(checkpoint_path, world_model, device):
    if checkpoint_path:
        model, payload = load_value_checkpoint(checkpoint_path, device)
        if model.schema.to_dict() != world_model.schema.to_dict():
            raise ValueError("terminal-value checkpoint schema does not match world model")
        model.eval()
        return model, payload
    model = TerminalValueModel(
        world_model.schema, world_model.normalizer,
        hidden_dims=(256, 256, 128), ensemble_size=3,
        return_normalizer=ReturnNormalizer(),
    ).to(device)
    model.eval()
    return model, {}


def configure_online_mpc_objective(mpc_config, args):
    """Apply the controlled reward-only/value ablation to one MPC config."""

    terminal_enabled = (
        getattr(args, "mpc_terminal_value", "enabled") == "enabled"
    )
    mpc_config.reward_source = "learned"
    mpc_config.learned_reward_coefficient = 1.0
    mpc_config.analytical_reward_coefficient = 0.0
    mpc_config.seed = int(getattr(args, "seed", 42))
    mpc_config.terminal_value_checkpoint = None
    if terminal_enabled:
        mpc_config.objective_mode = "reward_plus_terminal_value"
        mpc_config.use_terminal_value = True
        # The online trainer can initialize and fit V from real replay, so an
        # offline checkpoint is useful but not required.
        mpc_config.terminal_value_required = False
    else:
        mpc_config.objective_mode = "reward_only"
        mpc_config.use_terminal_value = False
        mpc_config.terminal_value_required = False
    mpc_config.validate()
    return terminal_enabled


def build_arg_parser():
    from scripts.train_high_level import build_arg_parser as base_parser
    parser = base_parser()
    parser.description = "Online high-level self-play with replay-trained world-model MPC."
    parser.add_argument("--world-model-checkpoint", default=None)
    parser.add_argument("--world-model-config", default="configs/world_model_as2.yaml")
    parser.add_argument("--mpc-config", default="configs/mpc_joint_teams.yaml")
    parser.add_argument("--mpc-profile", default="teacher_training")
    parser.add_argument("--terminal-value-checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-subdir",
        default="high_level_online_mpc",
        help="Subdirectory under the W&B run used for online policy/model checkpoints.",
    )
    parser.add_argument(
        "--world-model-update-interval", type=int, default=25,
        help="PPO iterations between online world-model/value update bursts.",
    )
    parser.add_argument(
        "--world-model-replay-buffer-size", type=int, default=100000,
        help="Maximum high-level match transitions retained across opponent snapshots.",
    )
    parser.add_argument("--world-model-replay-recent-fraction", type=float, default=0.5)
    parser.add_argument("--world-model-replay-recent-window", type=int, default=10000)
    parser.add_argument("--world-model-update-batch-size", type=int, default=1024)
    parser.add_argument("--world-model-updates-per-interval", type=int, default=8)
    parser.add_argument(
        "--mpc-horizon", type=int, default=None,
        help="Number of high-level skill intervals imagined by MPC.",
    )
    parser.add_argument(
        "--mpc-num-samples", "--mpc-num-candidates",
        dest="mpc_num_samples", type=int, default=None,
        help="Hybrid action sequences sampled per CEM iteration.",
    )
    parser.add_argument(
        "--mpc-num-iterations", type=int, default=None,
        help="CEM refinement iterations per MPC call.",
    )
    parser.add_argument(
        "--mpc-kl-coefficient", "--mpc-kl-coef", type=float, default=0.05,
        help="Coefficient on forward KL(MPC || high-level policy); zero disables distillation.",
    )
    parser.add_argument(
        "--mpc-guidance-reward-coefficient",
        type=float,
        default=0.0,
        help=(
            "Dense PPO reward for agreement with MPC's first action; the "
            "controlled ablation launchers set this to one."
        ),
    )
    parser.add_argument(
        "--mpc-terminal-value",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Enable or remove the terminal continuation value from MPC ranking.",
    )
    parser.add_argument(
        "--mpc-warmup-steps", "--mpc-warmup", type=int, default=500,
        help="Vectorized high-level environment steps collected before MPC/distillation can start.",
    )
    parser.add_argument("--mpc-distillation-batch-size", type=int, default=1024)
    # Online CEM is substantially more expensive than policy-only PPO.
    parser.set_defaults(num_envs=32, project="as2_high_level_online_mpc")
    return parser


def validate_online_args(args):
    if args.world_model_update_interval < 1:
        raise ValueError("--world-model-update-interval must be positive")
    if args.world_model_replay_buffer_size < 1:
        raise ValueError("--world-model-replay-buffer-size must be positive")
    if args.mpc_kl_coefficient < 0.0:
        raise ValueError("--mpc-kl-coefficient cannot be negative")
    if args.mpc_guidance_reward_coefficient < 0.0:
        raise ValueError(
            "--mpc-guidance-reward-coefficient cannot be negative"
        )
    if args.mpc_warmup_steps < 0:
        raise ValueError("--mpc-warmup-steps cannot be negative")
    for name in ("mpc_horizon", "mpc_num_samples", "mpc_num_iterations"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def train_robot(args):
    if isaacgym is None:
        raise ImportError(
            "The online self-play entry point requires Isaac Gym to be "
            "available and imported before torch."
        )
    from scripts.train_high_level import (
        set_training_seed,
        validate_high_level_training_args,
    )
    validate_high_level_training_args(args)
    validate_online_args(args)
    set_training_seed(getattr(args, "seed", 42))
    from scripts.train_high_level import configure_high_level_cfg, load_skill_policies
    from dribblebot.envs.base.legged_robot_config import Cfg
    from dribblebot.envs.as2.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    from dribblebot.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from dribblebot.envs.wrappers.shared_self_play_wrapper import SharedPolicySelfPlayWrapper
    try:
        from scripts.collect_world_model_data import TerminalStateCapture
    except ModuleNotFoundError as error:
        if error.name != "scripts.collect_world_model_data":
            raise
        # The collector was moved to discard/ in lightweight deployments; the
        # simulator controller contains the same capture primitive.
        from dribblebot.mpc.simulator_controller import TerminalStateCapture
    from dribblebot_learn.ppo_cse import Runner, RunnerArgs
    from dribblebot_learn.ppo_cse.actor_critic import AC_Args
    from dribblebot_learn.ppo_cse.ppo import PPO_Args
    import wandb

    configure_high_level_cfg(Cfg, args)
    skills = load_skill_policies(args)
    if args.validate_skill_policies_only:
        print("Validated all AS2 low-level skill policies; training was not started.")
        return
    RunnerArgs.resume = bool(args.resume)
    RunnerArgs.resume_policy_only = args.resume_mode == "policy-only"
    RunnerArgs.resume_path = args.resume_run
    RunnerArgs.resume_checkpoint = args.resume_checkpoint
    RunnerArgs.checkpoint_dir = args.checkpoint_dir
    RunnerArgs.self_play_update_interval = args.self_play_update_interval
    RunnerArgs.skill_entropy_initial_coef = args.skill_entropy_coef
    RunnerArgs.skill_entropy_final_coef = args.skill_entropy_final_coef
    RunnerArgs.skill_entropy_anneal_iterations = args.skill_entropy_anneal_iterations
    PPO_Args.learning_rate = args.learning_rate
    PPO_Args.adaptation_module_learning_rate = args.learning_rate
    PPO_Args.max_learning_rate = args.learning_rate
    PPO_Args.schedule = args.schedule
    PPO_Args.desired_kl = args.desired_kl
    PPO_Args.entropy_coef = args.entropy_coef
    PPO_Args.num_learning_epochs = args.ppo_epochs
    PPO_Args.skill_entropy_coef = args.skill_entropy_coef
    PPO_Args.skill_action_stride = 6
    PPO_Args.num_skill_logits = 3
    PPO_Args.stop_on_excessive_kl = True
    PPO_Args.max_kl_factor = args.max_kl_factor
    AC_Args.init_noise_std = args.init_noise_std
    AC_Args.max_action_std = args.max_noise_std
    AC_Args.action_mean_bound = args.action_mean_bound
    AC_Args.hybrid_skill_policy = True
    AC_Args.skill_action_stride = 6
    AC_Args.num_skill_logits = 3
    AC_Args.adaptation_labels = []
    AC_Args.adaptation_dims = []

    run = wandb.init(project=args.project or "as2_high_level_online_mpc", config={
        "AC_Args": vars(AC_Args), "PPO_Args": vars(PPO_Args),
        "RunnerArgs": vars(RunnerArgs), "Cfg": vars(Cfg),
        "online_mpc": vars(args),
        "critic_reuse": {
            "reused_for_terminal_value": False,
            "reason": (
                "PPO critic consumes decentralized local observation history; "
                "MPC imagines a centralized Markov state without that history."
            ),
        },
        "self_play": {
            "team_size": args.num_robots,
            "opponent_snapshot_interval": args.self_play_update_interval,
            "opponent_pool_size": args.opponent_pool_size,
            "opponent_latest_probability": args.opponent_latest_probability,
        },
        "skill_policy_metadata": {
            skill: record.get("policy_metadata", {})
            for skill, record in skills.items()
        },
    })
    raw_env = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=args.headless, cfg=Cfg)
    match_env = HighLevelSkillWrapper(raw_env, skills)
    env = SharedPolicySelfPlayWrapper(
        match_env, team_size=args.num_robots, opponent_device=args.policy_device,
        opponent_pool_size=args.opponent_pool_size,
        opponent_latest_probability=args.opponent_latest_probability,
    )
    state_adapter = FootballWorldModelStateAdapter(
        match_env, max_obstacles=0, num_robots=2 * args.num_robots,
    )
    capture = TerminalStateCapture(match_env, state_adapter)
    args.terminal_state_capture = capture
    config = __import__("dribblebot.world_model.config", fromlist=["load_config"]).load_config(args.world_model_config)
    world_model, world_checkpoint = _build_world_model(args.world_model_checkpoint, state_adapter, config, args.device)
    mpc_config, mpc_payload = load_mpc_config(args.mpc_config, args.mpc_profile)
    if args.mpc_horizon is not None:
        mpc_config.horizon = args.mpc_horizon
    if args.mpc_num_samples is not None:
        mpc_config.num_candidates = args.mpc_num_samples
        mpc_config.num_elites = max(
            1, min(mpc_config.num_elites, mpc_config.num_candidates - 1)
        )
    if args.mpc_num_iterations is not None:
        mpc_config.num_iterations = args.mpc_num_iterations
    configured_terminal_path = mpc_config.terminal_value_checkpoint
    terminal_enabled = configure_online_mpc_objective(mpc_config, args)
    terminal_path = None
    if terminal_enabled:
        terminal_path = args.terminal_value_checkpoint or (
            configured_terminal_path
            if configured_terminal_path
            and Path(configured_terminal_path).is_file()
            else None
        )
    terminal_model, terminal_checkpoint = _build_terminal_value(
        terminal_path, world_model, args.device
    )
    objective = MPCObjective(
        world_model.schema, world_model.action_adapter, world_model.event_names,
        mpc_config,
        terminal_value=(
            terminal_model.predict_with_uncertainty
            if terminal_enabled
            else None
        ),
        controlled_robot_count=args.num_robots,
    )
    planner = HybridCEMMPC(world_model, state_adapter, world_model.action_adapter, objective, mpc_config)
    replay = WorldModelReplayBuffer(
        args.world_model_replay_buffer_size,
        args.world_model_replay_recent_fraction,
        args.world_model_replay_recent_window,
        seed=getattr(args, "seed", 42),
    )
    value_replay = WorldModelReplayBuffer(
        args.world_model_replay_buffer_size,
        args.world_model_replay_recent_fraction,
        args.world_model_replay_recent_window,
        seed=getattr(args, "seed", 42) + 1,
    )
    wm_trainer = OnlineWorldModelTrainer(
        world_model, replay, args.device,
        learning_rate=float(config.get("training", {}).get("learning_rate", 3e-4)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-5)),
        batch_size=args.world_model_update_batch_size,
        updates_per_interval=args.world_model_updates_per_interval,
        optimizer_states=world_checkpoint.get("optimizer_states"),
        loss_config=config.get("loss", {}),
    )
    value_trainer = OnlineTerminalValueTrainer(
        terminal_model, value_replay, args.device,
        batch_size=args.world_model_update_batch_size,
        gamma=mpc_config.gamma, pretrained=bool(terminal_path),
        optimizer_state=terminal_checkpoint.get("optimizer_state"),
        enabled=terminal_enabled,
    )
    extension = OnlineMPCSelfPlayExtension(
        env, match_env, world_model, planner, state_adapter, replay, value_replay,
        value_trainer, wm_trainer, args,
    )
    RunnerArgs.checkpoint_dir = args.checkpoint_dir or str(
        Path(run.dir) / "tmp" / "legged_data" / args.checkpoint_subdir
    )
    run.config.update(
        {"resolved_checkpoint_dir": RunnerArgs.checkpoint_dir},
        allow_val_change=True,
    )
    # Match the base trainer's high-level actor initialization even though
    # constructing the world and terminal-value networks consumed RNG state.
    set_training_seed(getattr(args, "seed", 42))
    runner = Runner(env, device=args.device, training_extension=extension)
    try:
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True, eval_freq=100)
    finally:
        capture.restore()
        wandb.finish()


def parse_args():
    return build_arg_parser().parse_args()


if __name__ == "__main__":
    train_robot(parse_args())
