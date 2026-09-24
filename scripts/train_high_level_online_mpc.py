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
import os
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Mapping, Optional


# This module imports Torch before the shared high-level trainer, so establish
# the same Python-owned CPU threading defaults here as well.
_CPU_THREADS = os.environ.get("DRIBBLEBOT_CPU_THREADS", "4")
for _THREAD_ENV in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_THREAD_ENV, _CPU_THREADS)

if "torch" in sys.modules:
    isaacgym = sys.modules.get("isaacgym")
else:
    try:
        import isaacgym
    except ImportError:
        isaacgym = None

import torch
import torch.nn.functional as F

from quadruped.mpc.config import load_mpc_config
from quadruped.mpc.hybrid_cem import HybridCEMMPC
from quadruped.mpc.objective import MPCObjective
from quadruped.mpc.terminal_value import (
    ReturnNormalizer,
    TerminalValueModel,
    ValueModelConfig,
    compute_discounted_returns,
    load_value_checkpoint,
    save_value_checkpoint,
)
from quadruped.world_model.ensemble import WorldModelEnsemble
from quadruped.world_model.losses import feature_group_weights, one_step_member_loss
from quadruped.world_model.normalizer import WorldModelNormalizer
from quadruped.world_model.schema import EVENT_NAMES
from quadruped.world_model.state_adapter import FootballWorldModelStateAdapter
from quadruped.world_model.trainer import load_checkpoint, save_checkpoint


def skill_fingerprint(skills):
    """Content identity of the executable skill bundle, independent of paths."""
    import hashlib
    import json
    hashes = {
        name: {key: artifact.get("sha256") for key, artifact in
               record.get("policy_metadata", {}).get("artifacts", {}).items()}
        for name, record in skills.items()
    }
    if not hashes or any(not entries.get("body") for entries in hashes.values()):
        raise ValueError("MPC requires hashed skill artifacts to identify its dynamics")
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


class WorldModelReplayBuffer:
    """Bounded opponent-stratified replay with a recent/historical mix.

    Capacity is balanced across opponent snapshot IDs so a long interval
    against the latest opponent cannot erase every older opponent.  Each
    minibatch reserves ``recent_fraction`` for the latest snapshot and samples
    the remainder evenly across historical snapshots.
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
        self._buckets = {}
        self._size = 0
        self._sequence = 0
        self._generator = torch.Generator().manual_seed(int(seed))

    def __len__(self):
        return self._size

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
            self._append_item(item)

    @staticmethod
    def _opponent_iteration(item):
        """Return a stable opponent ID for stratified retention/sampling."""

        value = item.get("opponent_snapshot_iteration", -1)
        if torch.is_tensor(value):
            value = value.reshape(-1)[0].item() if value.numel() else -1
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _append_item(self, item):
        """Append while reserving capacity for every known opponent.

        A plain deque evicts the oldest snapshot wholesale.  That made a
        100k-transition buffer forget all but the current opponent when
        snapshots were rotated every few hundred PPO iterations.  Evict from
        the most over-represented opponent bucket instead; with the small
        opponent pool used by self-play this keeps every snapshot represented
        until the buffer is genuinely full of distinct opponents.
        """

        incoming_id = self._opponent_iteration(item)
        if incoming_id not in self._buckets:
            self._buckets[incoming_id] = deque()
        if self._size >= self.capacity:
            ids_after_insert = len(self._buckets)
            fair_share = max(float(self.capacity) / max(ids_after_insert, 1), 1.0)
            oversized = [
                key for key, bucket in self._buckets.items()
                if len(bucket) > fair_share
            ]
            if oversized:
                evict_id = max(oversized, key=lambda key: len(self._buckets[key]))
            else:
                candidates = list(self._buckets)
                if incoming_id in candidates and len(candidates) > 1:
                    candidates.remove(incoming_id)
                evict_id = max(candidates, key=lambda key: len(self._buckets[key]))
            self._buckets[evict_id].popleft()
            self._size -= 1
            if not self._buckets[evict_id]:
                del self._buckets[evict_id]
        self._buckets.setdefault(incoming_id, deque()).append((self._sequence, item))
        self._sequence += 1
        self._size += 1

    def items(self):
        """Return replay rows in insertion order for checkpointing/auditing."""

        sequenced = [entry for bucket in self._buckets.values() for entry in bucket]
        sequenced.sort(key=lambda entry: entry[0])
        return [item for _, item in sequenced]

    def state_dict(self):
        return {
            "format": "dribblebot_world_model_replay_v2",
            "capacity": self.capacity,
            "recent_fraction": self.recent_fraction,
            "recent_window": self.recent_window,
            "items": self.items(),
        }

    def load_state_dict(self, payload):
        if not isinstance(payload, Mapping):
            raise ValueError("replay state must be a mapping")
        items = payload.get("items", [])
        if not isinstance(items, (list, tuple)):
            raise ValueError("replay state items must be a list")
        self._buckets.clear()
        self._size = 0
        self._sequence = 0
        for item in items[-self.capacity:]:
            if not isinstance(item, Mapping):
                raise ValueError("replay state contains a non-mapping item")
            self._append_item(dict(item))

    def opponent_counts(self):
        return {key: len(bucket) for key, bucket in self._buckets.items()}

    def _sample_rows(self, count: int):
        if self._size < 1:
            raise ValueError("cannot sample an empty replay buffer")
        count = int(count)
        recent_count = min(count, int(round(count * self.recent_fraction)))
        historical_count = count - recent_count
        ordered = self.items()
        recent_pool = ordered[-self.recent_window :]
        rows = []
        for _ in range(recent_count):
            index = int(torch.randint(
                len(recent_pool), (1,), generator=self._generator
            ))
            rows.append(recent_pool[index])

        opponent_ids = sorted(self._buckets)
        newest_id = opponent_ids[-1]
        historical_pools = {
            key: [entry[1] for entry in self._buckets[key]]
            for key in opponent_ids
            if key != newest_id
        }
        historical_ids = sorted(historical_pools)
        if not historical_pools:
            # A short buffer or a large recent window has no disjoint
            # historical slice; sample the complete buffer with replacement.
            historical_pools = {-1: ordered}
            historical_ids = [-1]
        order = torch.randperm(
            len(historical_ids), generator=self._generator
        ).tolist()
        for offset in range(historical_count):
            opponent_id = historical_ids[order[offset % len(order)]]
            pool = historical_pools[opponent_id]
            index = int(torch.randint(len(pool), (1,), generator=self._generator))
            rows.append(pool[index])
        return rows

    def prune_teacher_targets(self, current_step, max_age):
        """Remove expired/invalid targets before deciding whether to update."""
        removed = 0
        for opponent, bucket in list(self._buckets.items()):
            retained = deque()
            for sequence, item in bucket:
                step, weight = item.get("teacher_step"), item.get("teacher_weight")
                valid = (step is not None and weight is not None
                         and bool(torch.isfinite(torch.as_tensor(step)).all())
                         and bool(torch.isfinite(torch.as_tensor(weight)).all())
                         and float(weight) > 0
                         and 0 <= current_step - int(step) <= max_age)
                if valid:
                    retained.append((sequence, item))
                else:
                    removed += 1
            if retained:
                self._buckets[opponent] = retained
            else:
                del self._buckets[opponent]
        self._size -= removed
        return removed

    def sample(self, count: int, device: Optional[torch.device] = None,
               replacement: bool = True) -> Dict[str, torch.Tensor]:
        if replacement:
            rows = self._sample_rows(count)
        else:
            # Teacher updates use distinct eligible examples. Dynamics replay
            # retains its existing opponent-stratified sampling by default.
            if count > len(self):
                raise ValueError("Not enough distinct replay rows for this batch")
            ordered = self.items()
            indices = torch.randperm(len(ordered), generator=self._generator)[:count]
            rows = [ordered[int(index)] for index in indices]
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
                 optimizer_states=None, loss_config=None, pretrained=False):
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
        self.update_bursts = 0
        self.gradient_updates = 0
        self.pretrained = bool(pretrained)
        self.ready = self.pretrained
        self.feature_weights = feature_group_weights(
            self.model.schema, self.loss_config, self.device
        )

    def update(self) -> Dict[str, float]:
        if getattr(self, "frozen", False):
            self.model.eval()
            return {"world_model/frozen": 1.0, "world_model/gradient_updates_total": 0.0}
        if len(self.replay) < max(2, self.batch_size):
            return {
                "world_model/update_skipped": 1.0,
                "world_model/replay_size": float(len(self.replay)),
                "world_model/ready": float(self.ready),
                "world_model/update_bursts": float(self.update_bursts),
                "world_model/gradient_updates_total": float(self.gradient_updates),
            }
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
                self.gradient_updates += 1
                for key, value in losses.items():
                    totals[f"world_model/{key}"] = (
                        totals.get(f"world_model/{key}", 0.0)
                        + float(value.detach())
                    )
                updates += 1
        if updates:
            self.update_bursts += 1
            self.ready = True
        self.model.eval()
        averaged_keys = set(totals)
        totals["world_model/replay_size"] = float(len(self.replay))
        totals["world_model/updates"] = float(updates)
        totals["world_model/ready"] = float(self.ready)
        totals["world_model/update_bursts"] = float(self.update_bursts)
        totals["world_model/gradient_updates_total"] = float(self.gradient_updates)
        return {
            key: (
                value / max(updates, 1)
                if key in averaged_keys
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
        self.pretrained = bool(pretrained)
        self.ready = self.pretrained if self.enabled else True
        self.update_bursts = 0
        self.gradient_updates = 0
        # Online Monte Carlo returns are non-stationary as the curriculum and
        # opponent pool change. Keep target scaling fitted to the data actually
        # seen by this trainer; the previous fixed (0, 1) scale made sparse
        # goal returns unnecessarily difficult to regress.
        initial_mean = float(self.model.return_normalizer.mean)
        initial_std = float(self.model.return_normalizer.std)
        self._return_count = 1 if (self.pretrained and initial_std > 1.0) else 0
        self._return_mean = initial_mean
        self._return_m2 = initial_std * initial_std if self._return_count else 0.0

    def _update_return_normalizer(self, values):
        values = values.detach().float().reshape(-1)
        if not values.numel():
            return
        count = int(values.numel())
        mean = float(values.mean())
        m2 = float(((values - mean) ** 2).sum())
        if self._return_count == 0:
            total, combined_mean, combined_m2 = count, mean, m2
        else:
            total = self._return_count + count
            delta = mean - self._return_mean
            combined_mean = self._return_mean + delta * count / total
            combined_m2 = (self._return_m2 + m2
                           + delta * delta * self._return_count * count / total)
        self._return_count = total
        self._return_mean = combined_mean
        self._return_m2 = combined_m2
        std = max((combined_m2 / max(total, 1)) ** .5, 1.0)
        self.model.rescale_returns(combined_mean, std)

    def update(self) -> Dict[str, float]:
        if not self.enabled:
            return {"terminal_value/enabled": 0.0}
        if len(self.replay) < max(2, self.batch_size):
            return {
                "terminal_value/update_skipped": 1.0,
                "terminal_value/ready": float(self.ready),
                "terminal_value/update_bursts": float(self.update_bursts),
                "terminal_value/gradient_updates_total": float(self.gradient_updates),
            }
        self.model.train()
        total = 0.0
        performed = 0
        for _ in range(self.updates_per_interval):
            if len(self.replay) < self.batch_size:
                break
            batch = self.replay.sample(self.batch_size, self.device)
            raw_target = batch["return_to_go"].float().reshape(-1)
            self._update_return_normalizer(raw_target)
            target = self.model.return_normalizer.normalize(
                raw_target
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
            self.gradient_updates += 1
            performed += 1
            total += float(loss.detach())
        if performed:
            self.update_bursts += 1
        self.ready = self.ready or bool(performed)
        self.model.eval()
        return {
            "terminal_value/loss": total / max(performed, 1),
            "terminal_value/ready": float(self.ready),
            "terminal_value/update_bursts": float(self.update_bursts),
            "terminal_value/gradient_updates_total": float(self.gradient_updates),
            "terminal_value/return_mean": float(self._return_mean),
            "terminal_value/return_std": float(max(
                (self._return_m2 / max(self._return_count, 1)) ** .5, 1.0)),
            "terminal_value/return_samples": float(self._return_count),
        }


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
        self._last_plan_high_level_step = -1
        self._plans_computed = 0
        self._planning_seconds = 0.0
        self._metrics: Dict[str, float] = {}
        self._distill_optimizer = None
        self._high_level_steps = 0
        self._episode_states = [[] for _ in range(self.train_matches)]
        self._episode_rewards = [[] for _ in range(self.train_matches)]
        self._episode_terminated = [[] for _ in range(self.train_matches)]
        self._episode_truncated = [[] for _ in range(self.train_matches)]
        # TerminalStateCapture is installed by build_online_training before
        # this extension is constructed.
        self.capture = getattr(args, "terminal_state_capture", None)
        self._load_online_replay()
        self._model_refresh_start_updates = self.world_model_trainer.gradient_updates

    def _load_online_replay(self):
        checkpoint = getattr(self.args, "online_replay_checkpoint", None)
        if (
            checkpoint is None
            and bool(getattr(self.args, "resume", False))
            and bool(getattr(self.args, "auto_resume_online_replay", True))
        ):
            resume = getattr(self.args, "resume_checkpoint", None)
            if resume:
                candidate = Path(resume).expanduser().resolve().parent / (
                    "online_replay_latest.pt"
                )
                if candidate.is_file():
                    checkpoint = str(candidate)
        if not checkpoint:
            return
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"online replay checkpoint does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "dribblebot_online_replay_v1":
            raise ValueError(f"Unsupported online replay checkpoint format: {path}")
        if not self._replay_skills_match(payload):
            print(f"Discarding dynamics/value replay with changed or unknown skills: {path}")
            return
        self.replay.load_state_dict(payload.get("dynamics", {}))
        self.value_replay.load_state_dict(payload.get("value", {}))
        counters = payload.get("trainer_counters", {})
        self.world_model_trainer.update_bursts = int(
            counters.get("world_model_update_bursts", 0)
        )
        self.world_model_trainer.gradient_updates = int(
            counters.get("world_model_gradient_updates", 0)
        )
        self.world_model_trainer.ready = (
            bool(getattr(self.world_model_trainer, "pretrained", False))
            or self.world_model_trainer.gradient_updates > 0
        )
        self.terminal_trainer.update_bursts = int(
            counters.get("terminal_value_update_bursts", 0)
        )
        self.terminal_trainer.gradient_updates = int(
            counters.get("terminal_value_gradient_updates", 0)
        )
        self.terminal_trainer.ready = (
            self.terminal_trainer.ready
            or self.terminal_trainer.gradient_updates > 0
        )
        self._high_level_steps = int(payload.get("high_level_steps", 0))
        self._plans_computed = int(counters.get("mpc_plans_computed", 0))
        self._planning_seconds = float(
            counters.get("mpc_planning_seconds", 0.0)
        )
        print(
            f"Loaded {len(self.replay)} dynamics and {len(self.value_replay)} "
            f"value transitions from {path}"
        )

    def _replay_skills_match(self, payload):
        expected = getattr(self.args, "skill_fingerprint", None)
        if getattr(self.args, "shooting_options", False) and payload.get("execution_contract") != "shooting_option_v1":
            return False
        return expected is None or payload.get("skill_fingerprint") == expected

    def _model_refresh_complete(self):
        return (not getattr(self.args, "mpc_require_model_refresh", False)
                or self.world_model_trainer.gradient_updates
                - getattr(self, "_model_refresh_start_updates", 0)
                >= int(getattr(self.args, "mpc_min_world_model_updates", 1)))

    def _planner_ready(self):
        """Require useful replay/model updates before trusting MPC labels."""

        if self._high_level_steps < int(self.args.mpc_warmup_steps):
            return False
        if len(self.replay) < int(getattr(self.args, "mpc_min_replay_size", 1)):
            return False
        if (
            (not bool(getattr(self.world_model_trainer, "pretrained", False))
             or bool(getattr(self.args, "mpc_require_model_refresh", False)))
            and (self.world_model_trainer.gradient_updates
                 - (getattr(self, "_model_refresh_start_updates", 0)
                    if getattr(self.args, "mpc_require_model_refresh", False) else 0))
            < int(getattr(self.args, "mpc_min_world_model_updates", 1))
        ):
            return False
        if self.planner.config.use_terminal_value:
            if getattr(self.args, "mpc_value_fallback", False):
                value_ready = (getattr(self, "_value_quality_ready", False)
                               and self.terminal_trainer.ready
                               and self.terminal_trainer.gradient_updates >= int(
                                   getattr(self.args, "mpc_min_terminal_value_updates", 1)))
                # Keep training V, but do not let an unreliable V block useful
                # short-horizon reward planning or contaminate its objective.
                self.planner.config.objective_mode = (
                    "reward_plus_terminal_value" if value_ready else "reward_only")
                self._metrics["mpc/terminal_value_active"] = float(value_ready)
                return True
            if getattr(self.args, "shooting_options", False) and not getattr(self, "_value_quality_ready", False):
                return False
            if not self.terminal_trainer.ready:
                return False
            if (
                not self.terminal_trainer.pretrained
                and self.terminal_trainer.gradient_updates
                < int(getattr(self.args, "mpc_min_terminal_value_updates", 1))
            ):
                return False
        return True

    def _readiness_metrics(self):
        opponent_counts = self.replay.opponent_counts()
        metrics = {
            "mpc/readiness": float(self._planner_ready()),
            "mpc/warmup_steps": float(self.args.mpc_warmup_steps),
            "mpc/world_model_update_bursts": float(
                self.world_model_trainer.update_bursts
            ),
            "mpc/world_model_gradient_updates": float(
                self.world_model_trainer.gradient_updates
            ),
            "mpc/world_model_pretrained": float(
                bool(getattr(self.world_model_trainer, "pretrained", False))
            ),
            "mpc/model_refresh_required": float(getattr(self.args, "mpc_require_model_refresh", False)),
            "mpc/model_updates_since_start": float(
                self.world_model_trainer.gradient_updates - self._model_refresh_start_updates),
            "mpc/terminal_value_update_bursts": float(
                self.terminal_trainer.update_bursts
            ),
            "mpc/terminal_value_gradient_updates": float(
                self.terminal_trainer.gradient_updates
            ),
            "mpc/replay_size": float(len(self.replay)),
            "mpc/min_replay_size": float(
                getattr(self.args, "mpc_min_replay_size", 1)
            ),
            "mpc/replan_interval": float(
                getattr(self.args, "mpc_replan_interval", 1)
            ),
            "mpc/plans_computed_total": float(self._plans_computed),
            "mpc/planning_seconds_total": float(self._planning_seconds),
            "mpc/planning_seconds_per_call": float(
                self._planning_seconds / max(self._plans_computed, 1)
            ),
            "world_model/replay_opponent_count": float(len(opponent_counts)),
        }
        if opponent_counts:
            metrics["world_model/replay_oldest_opponent_iteration"] = float(
                min(opponent_counts)
            )
            metrics["world_model/replay_latest_opponent_iteration"] = float(
                max(opponent_counts)
            )
        return metrics

    def bind_runner(self, runner):
        self.runner = runner
        # Reuse PPO's optimizer so Adam state for the actor and exploration
        # scale has a single owner. The critic receives no distillation
        # gradient and continues to train only through PPO.
        self._distill_optimizer = runner.alg.optimizer

    @property
    def train_matches(self):
        return int(self.env.num_train_envs // self.env.team_size)

    def _timeout_return_target(self, row, next_state, match_done, timeout):
        """Build a completed episode target with correct timeout bootstrapping."""

        rewards = self._episode_rewards[row]
        terminated = self._episode_terminated[row]
        truncated = self._episode_truncated[row]
        bootstrap = 0.0
        bootstrap_enabled = bool(
            getattr(self.args, "terminal_bootstrap_on_timeout", True)
        )
        if (
            bool(timeout)
            and bootstrap_enabled
            and self.terminal_trainer.ready
        ):
            with torch.no_grad():
                bootstrap = float(
                    self.terminal_trainer.model.predict(
                        next_state[row : row + 1].to(self.terminal_trainer.device)
                    )[0].detach().cpu()
                )
        returns = compute_discounted_returns(
            rewards,
            terminated,
            truncated,
            self.terminal_trainer.gamma,
            bootstrap_value=bootstrap,
            bootstrap_on_truncation=bool(timeout and bootstrap_enabled),
        )
        tail_exclusion = int(
            getattr(self.args, "terminal_timeout_tail_exclusion", 0)
        )
        if bool(timeout) and bootstrap_enabled and not self.terminal_trainer.ready:
            # Do not train on the least reliable suffix while the bootstrap
            # model is still uninitialized.  This is configurable and defaults
            # to retaining all data once the warm-up model is ready.
            tail_exclusion = max(0, tail_exclusion)
            if tail_exclusion:
                keep = max(0, len(returns) - tail_exclusion)
                returns = returns[:keep]
        return returns

    def _observe_value_quality(self, states, returns):
        """Score newly completed trajectories before adding them to training."""
        with torch.no_grad():
            target = torch.as_tensor(returns, device=self.terminal_trainer.device, dtype=torch.float)
            prediction = self.terminal_trainer.model.predict(torch.stack(states).to(target.device)).reshape(-1)
            measurements = {"mse": (prediction-target).square().mean().item(),
                            "mean": target.mean().item(), "second": target.square().mean().item()}
            quality = getattr(self, "_value_quality", measurements.copy())
            for key, value in measurements.items():
                quality[key] = .95 * quality[key] + .05 * value
            self._value_quality = quality
            self._value_quality_samples = getattr(self, "_value_quality_samples", 0) + len(returns)
            variance = max(quality["second"] - quality["mean"]**2, 1e-6)
            self._value_quality_ready = self._value_quality_samples >= 2048 and quality["mse"] < .9 * variance
            self._metrics.update({"terminal_value/prequential_mse": quality["mse"],
                                  "terminal_value/prequential_variance": variance,
                                  "terminal_value/validation_ready": float(self._value_quality_ready)})

    def before_env_step(self, iteration, obs_before, actions):
        batch = self.train_matches
        state = self.state_adapter.extract_state(self.match_env)["tensor"][:batch]
        if self.capture is not None:
            self.capture.clear()
        plan = None
        replan_interval = max(
            1, int(getattr(self.args, "mpc_replan_interval", 1))
        )
        quality_gate = (not bool(getattr(self.args, "mpc_quality_filter", False))
                        or (getattr(self, "_quality_samples", 0) >= 512
                            and getattr(self, "_ball_prediction_error", float('inf'))
                            <= float(getattr(self.args, "mpc_max_ball_error", .25))))
        should_plan = (
            self._planner_ready() and quality_gate
            and self._high_level_steps % replan_interval == 0
        )
        if bool(getattr(self.args, "mpc_quality_filter", False)):
            self._metrics["mpc_quality/model_gate_open"] = float(quality_gate)
        plan_indices = torch.arange(batch, device=state.device)
        query_budget = int(getattr(self.args, "mpc_query_budget", 0))
        if should_plan and 0 < query_budget < batch:
            # Round-robin coverage prevents permanently ignoring difficult states.
            start = (self._plans_computed * query_budget) % batch
            plan_indices = (torch.arange(query_budget, device=state.device) + start) % batch
            self._planner_state = None
        teacher_agent_mask = torch.ones(batch, self.env.team_size, dtype=torch.bool, device=state.device)
        if should_plan:
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
            planner_kwargs = {}
            role_fixed = (
                bool(getattr(self.match_env.cfg.env, "high_level_role_aware_fallback", True))
                and bool(getattr(self.match_env.cfg.env, "high_level_use_geometric_skill_fallback", True))
            )
            # Keep support behavior deterministic for every MPC plan. If the
            # planner is allowed to optimize both teammates, it can assign a
            # farther robot to approach while the nearer robot remains in a
            # support role. The coordinator's nearest-attacker arbitration is
            # the execution contract, so only the selected attacker is free.
            if role_fixed:
                    attackers = self.match_env._attacker_mask(self.match_env._skill_affordances())[:batch]
                    teacher_agent_mask = attackers[:, :self.env.team_size]
                    supports = ~teacher_agent_mask
                    support_commands = self.match_env._walk_support_commands(attacker_mask=self.match_env._attacker_mask(self.match_env._skill_affordances()))[:batch, :self.env.team_size]
                    own_fixed = torch.cat((torch.zeros_like(support_commands[..., :1]), support_commands), dim=-1)
                    fixed_view = fixed.view(batch, horizon, self.action_adapter.num_robots, 4)
                    fixed_view[:, :, :self.env.team_size] = own_fixed[:, None]
                    fixed_mask = fixed_mask[None].expand(batch, -1).clone()
                    fixed_mask[:, :self.env.team_size] = supports
            if bool(getattr(self.args, "mpc_quality_filter", False)):
                own_reference = _wrapper_actions_to_canonical(actions.reshape(batch, self.env.team_size, 6), self.match_env, self.action_adapter)
                reference = fixed.clone()
                reference[:, :, :4*self.env.team_size] = own_reference[:, None]
                planner_kwargs["reference_action_sequence"] = reference[plan_indices]
            plan = self.planner.plan(
                state[plan_indices], planner_state=self._planner_state,
                fixed_action_sequence=fixed[plan_indices], fixed_robot_mask=(fixed_mask[plan_indices] if fixed_mask.ndim == 2 else fixed_mask),
                **planner_kwargs,
            )
            self._planner_state = plan.planner_state if len(plan_indices) == batch else None
            self._last_plan_high_level_step = self._high_level_steps
            self._plans_computed += 1
            self._planning_seconds += float(plan.planning_time_seconds)
        elif self._planner_ready() and replan_interval > 1:
            # The warm-start distribution advances one horizon slot per plan.
            # It is stale after a deliberately skipped decision, so do not
            # feed it back into the next CEM call.
            self._planner_state = None
        self._pending = {
            "iteration": int(iteration), "state": state.detach(),
            "obs_before": {key: value.detach().clone() for key, value in obs_before.items()},
            "opponent_observation": self.env._team_observations(1)[:batch].detach().clone(),
            "opponent_obs_history": self.env._history[:batch, 1].detach().clone(),
            "actions": actions.detach().clone(), "plan": plan,
            "plan_indices": plan_indices,
            "teacher_agent_mask": teacher_agent_mask,
        }
        if getattr(self.args, "shooting_options", False):
            self._pending["teacher_agent_mask"] = teacher_agent_mask & (self.match_env.shoot_option_remaining[:batch, :self.env.team_size] <= 0)

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
        plan_indices = pending.get("plan_indices", torch.arange(batch, device=executed.device))
        # Evaluate one-step dynamics before this transition enters replay.
        if bool(getattr(self.args, "mpc_quality_filter", False)) and self._high_level_steps % 8 == 0:
            with torch.no_grad():
                predicted = self.world_model.predict_next(pending["state"], executed, deterministic=True)[0]
                schema = self.world_model.schema
                scale = pending["state"][:, schema.slice("field.geometry")][:, :2].abs()
                ball_slice = schema.slice("ball.position")
                error = ((predicted[:, ball_slice][:, :2]-next_state[:, ball_slice][:, :2])*scale).norm(dim=-1)
                # Compare only complete macro-transitions, not early terminal resets.
                live = ~dones[:batch*team].reshape(batch, team).any(-1).bool()
                from quadruped.mpc.prediction_metrics import robot_position_errors
                robot_ema = getattr(self, '_robot_position_error_ema', {})
                for key, value in robot_position_errors(schema, predicted, next_state, live).items():
                    # Match the ball-error EMA and evaluate before replay insertion.
                    robot_ema[key] = .95 * robot_ema.get(key, value) + .05 * value
                    self._metrics[key] = robot_ema[key]
                self._robot_position_error_ema = robot_ema
                shot_rows = live & (skills[:, :team] == 2).any(-1)
                if shot_rows.any():
                    velocity_slice = schema.slice("ball.linear_velocity")
                    velocity_error = (predicted[:, velocity_slice][:, :2] - next_state[:, velocity_slice][:, :2]).norm(dim=-1)
                    for name, value in (("_shot_ball_error", error[shot_rows].mean()),
                                        ("_shot_velocity_error", velocity_error[shot_rows].mean())):
                        numeric = float(value)
                        setattr(self, name, .95 * getattr(self, name, numeric) + .05 * numeric)
                    self._shot_quality_samples = getattr(self, "_shot_quality_samples", 0) + int(shot_rows.sum())
                    self._metrics.update({"mpc_quality/shot_ball_error_m": self._shot_ball_error,
                                          "mpc_quality/shot_velocity_error_mps": self._shot_velocity_error,
                                          "mpc_quality/shot_samples": self._shot_quality_samples})
                if live.any():
                    value = float(error[live].mean())
                    previous = getattr(self, "_ball_prediction_error", value)
                    self._ball_prediction_error = .95*previous + .05*value
                    self._quality_samples = getattr(self, "_quality_samples", 0) + int(live.sum())
                    self._metrics["mpc_quality/ball_error_m"] = self._ball_prediction_error
                    self._metrics["mpc_quality/prequential_samples"] = self._quality_samples
        guidance_coefficient = float(
            getattr(self.args, "mpc_guidance_reward_coefficient", 0.0)
        )
        if plan is not None and guidance_coefficient > 0.0:
            guidance, disagreement = mpc_action_agreement_reward(
                executed[plan_indices],
                plan.first_joint_action,
                self.action_adapter,
                team,
                guidance_coefficient,
            )
            reward_rows = (plan_indices[:, None]*team + torch.arange(team, device=plan_indices.device)).flatten()
            rewards[reward_rows] += guidance.reshape(-1).to(rewards.device)
            infos["mpc_guidance_reward"] = guidance.reshape(-1).detach()
            infos["mpc_guidance_disagreement"] = disagreement.reshape(
                -1
            ).detach()
        if "high_level_match_rewards" not in info:
            raise KeyError(
                "Online MPC training requires SharedPolicySelfPlayWrapper to "
                "provide 'high_level_match_rewards'; refusing to use a single "
                "robot's shaped reward as a silent fallback."
            )
        match_rewards = torch.as_tensor(
            info["high_level_match_rewards"],
            device=pending["state"].device,
            dtype=torch.float,
        )
        if match_rewards.ndim > 1:
            match_rewards = match_rewards.reshape(match_rewards.shape[0], -1)[:, 0]
        match_rewards = match_rewards.reshape(-1)[:batch]
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
            "high_level_goal", "high_level_opponent_goal", "high_level_ball_off_border",
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
            timeout_cpu = timeout_values.detach().cpu().tolist()
            terminated_cpu = terminated.detach().cpu().tolist()
            for row in range(batch):
                self._episode_states[row].append(state_cpu[row])
                self._episode_rewards[row].append(float(rewards_cpu[row]))
                self._episode_terminated[row].append(bool(terminated_cpu[row]))
                self._episode_truncated[row].append(bool(timeout_cpu[row]))
                if done_cpu[row]:
                    returns = self._timeout_return_target(
                        row, next_state, bool(done_cpu[row]), bool(timeout_cpu[row])
                    )
                    states = self._episode_states[row]
                    if len(returns) < len(states):
                        states = states[: len(returns)]
                    if len(returns):
                        if getattr(self.args, "shooting_options", False):
                            self._observe_value_quality(states, returns)
                        self.value_replay.add_batch({
                            "state": torch.stack(states),
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
                    self._episode_terminated[row].clear()
                    self._episode_truncated[row].clear()
        # A zero KL coefficient is a real ablation, not an unbounded target
        # cache.  Keep MPC planning active for pipeline parity, but do not
        # retain teacher labels when no distillation update can consume them.
        if plan is not None and float(self.args.mpc_kl_coefficient) > 0.0:
            targets = _mpc_distribution_to_policy_targets(plan, self.action_adapter, team)
            teacher_weight = torch.ones(len(plan_indices), device=executed.device)
            quality_valid = torch.ones(len(plan_indices), dtype=torch.bool, device=executed.device)
            if bool(getattr(self.args, "mpc_quality_filter", False)):
                from quadruped.mpc.teacher_quality import improvement_mask, selected_action_targets
                # Compare the chosen teacher action against the sampled student
                # action with identical future actions and opponent forecasts.
                reference = plan.best_action_sequence.clone()
                own_view = reference[:, 0, :4*team].reshape(-1, team, 4)
                own_view[:] = torch.where(pending["teacher_agent_mask"][plan_indices, :, None],
                                          requested[plan_indices, :4*team].reshape(-1, team, 4), own_view)
                pair = torch.stack((plan.best_action_sequence, reference), dim=1)
                with torch.no_grad():
                    # Distillation executes only the first macro-action; the
                    # student's feedback policy supplies later actions.  Use
                    # a one-step score for label quality so an open-loop
                    # continuation cannot make an otherwise bad first action
                    # look attractive.
                    quality_horizon = 1 if bool(getattr(
                        self.args, "mpc_quality_first_action", True)) else pair.shape[2]
                    quality_pair = pair[:, :, :quality_horizon]
                    imagined = self.world_model.rollout(
                        pending["state"][plan_indices], quality_pair, deterministic=True,
                        stop_on_done=False, action_transform=self.planner.objective.resolve_imagined_action)
                    score = self.planner.objective.evaluate(
                        pending["state"][plan_indices], quality_pair, imagined)
                quality_valid, teacher_weight, advantage = improvement_mask(
                    score.total[:, 0], score.total[:, 1], score.valid.all(-1),
                    float(getattr(self.args, "mpc_advantage_margin", .1)))
                model_ready = (getattr(self, "_quality_samples", 0) >= 512
                               and getattr(self, "_ball_prediction_error", float('inf'))
                               <= float(getattr(self.args, "mpc_max_ball_error", .25)))
                quality_valid &= model_ready
                if getattr(self.args, "shooting_options", False):
                    chosen_ids, _ = self.action_adapter.unpack(plan.first_joint_action)
                    shooting_candidate = (chosen_ids[:, :team] == 2).any(-1)
                    shot_model_ready = (getattr(self, "_shot_quality_samples", 0) >= 256
                                        and getattr(self, "_shot_ball_error", float('inf')) < .1
                                        and getattr(self, "_shot_velocity_error", float('inf')) < .5)
                    quality_valid &= ~shooting_candidate | shot_model_ready
                    self._metrics["mpc_quality/shot_model_ready"] = float(shot_model_ready)
                targets = selected_action_targets(plan, self.action_adapter, team)
                if getattr(self.args, "shooting_options", False):
                    # The shooting option chooses its own goal-directed command.
                    targets[3][2] = 0
                finite = torch.isfinite(advantage)
                self._metrics["mpc_quality/predicted_advantage"] = float(advantage[finite].mean()) if finite.any() else 0.
                self._metrics["mpc_quality/score_horizon"] = float(quality_horizon)
                self._metrics["mpc_quality/accepted_fraction"] = float(quality_valid.float().mean())
                self._metrics["mpc_quality/model_ready"] = float(model_ready)
                self._metrics["mpc_quality/query_matches"] = len(plan_indices)
            self._distill_records = getattr(self, "_distill_records", [])
            # Rows for which CEM had to relax the world-model OOD filter are
            # finite, but are not trustworthy teacher labels.  Keep the
            # policy on its on-policy PPO signal for those rows instead of
            # replaying an arbitrary extrapolation from the learned model.
            fallback = plan.uncertainty.get("ood_fallback_used")
            if fallback is None:
                teacher_valid = torch.ones(
                    len(plan_indices), dtype=torch.bool, device=obs_before["obs_history"].device
                )
            else:
                teacher_valid = ~fallback.to(
                    device=obs_before["obs_history"].device, dtype=torch.bool
                )
            teacher_valid &= quality_valid.to(teacher_valid.device)
            self._distill_records.append({
                "teacher_weight": teacher_weight.repeat_interleave(team).detach().cpu(),
                "teacher_step": torch.full((len(plan_indices)*team,), self._high_level_steps, dtype=torch.long),
                "obs_history": obs_before["obs_history"].reshape(batch, team, -1)[plan_indices].reshape(-1, self.env.num_obs_history).detach().cpu(),
                "target_command_mean": targets[0].detach().cpu(),
                "target_std": targets[1].detach().cpu(),
                "target_probs": targets[2].detach().cpu(),
                "target_masks": targets[3].detach().cpu()[None].expand(len(plan_indices) * team, -1, -1),
                "opponent_snapshot_iteration": selected_opponent_iterations[plan_indices.cpu()].repeat_interleave(
                    team
                ).detach().cpu(),
                "teacher_valid": (teacher_valid[:, None] & pending["teacher_agent_mask"][plan_indices]).flatten().detach().cpu(),
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
        if not self._planner_ready():
            return {
                "mpc_distillation/enabled": 0.0,
                "mpc_distillation/update_skipped_not_ready": 1.0,
            }
        records = getattr(self, "_distill_records", [])
        if not records:
            return {}
        # Keep the same recent/historical policy-state mixture as dynamics
        # replay without duplicating another large ring buffer.
        merged = {key: torch.cat([record[key] for record in records], dim=0) for key in records[0]}
        self._distill_records = []
        valid = merged.pop("teacher_valid", None)
        if valid is not None:
            valid = valid.to(dtype=torch.bool).reshape(-1)
            if valid.numel() != merged["obs_history"].shape[0]:
                return {
                    "mpc_distillation/update_skipped_invalid_teacher_mask": 1.0,
                }
            if not bool(valid.all()):
                merged = {key: value[valid] for key, value in merged.items()}
        if not merged["obs_history"].shape[0]:
            return {
                "mpc_distillation/update_skipped_no_valid_teacher_targets": 1.0,
            }
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
        # The policy owns one shared command Gaussian, while CEM stores one
        # conditional command distribution per skill. Fitting all three at
        # once pulls that shared head toward mutually incompatible commands.
        # Distill only the most likely teacher skill; retain the full soft
        # distribution for the categorical KL above.
        teacher_skill = target_probs.argmax(dim=-1)
        rows = torch.arange(target_probs.shape[0], device=target_probs.device)
        selected_command_kl = command_kl[rows, teacher_skill]
        selected_mask = target_masks[rows, teacher_skill]
        continuous = (
            (selected_command_kl * selected_mask).sum(dim=-1)
            / selected_mask.sum(dim=-1).clamp(min=1.0)
        ).mean()
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
        from quadruped_learn.ppo_cse.actor_critic import AC_Args
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
            self._metrics.update(self.world_model_trainer.update())
            self._metrics.update(self.terminal_trainer.update())
            self._metrics["terminal_value/replay_size"] = float(len(self.value_replay))
            self._metrics["online/high_level_steps"] = float(self._high_level_steps)

    def after_policy_update(self, iteration):
        metrics = dict(self._metrics)
        self._metrics = {}
        metrics.update(self._distill_update(iteration))
        metrics.update(self._readiness_metrics())
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
                "skill_fingerprint": (getattr(self.args, "skill_fingerprint", None)
                                      if self._model_refresh_complete() else None),
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
        if getattr(self.args, "freeze_world_model", False):
            shutil.copy2(self.args.world_model_checkpoint, world_path)
        world_latest = output / "world_model_online_latest.pt"
        shutil.copy2(world_path, world_latest)
        saved = [str(world_path), str(world_latest)]
        if self.terminal_trainer.enabled:
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
        if bool(getattr(self.args, "save_online_replay", True)):
            replay_payload = {
                "format": "dribblebot_online_replay_v1",
                "execution_contract": "shooting_option_v1" if getattr(self.args, "shooting_options", False) else "legacy",
                "skill_fingerprint": getattr(self.args, "skill_fingerprint", None),
                "high_level_steps": self._high_level_steps,
                "dynamics": self.replay.state_dict(),
                "value": self.value_replay.state_dict(),
                "trainer_counters": {
                    "world_model_update_bursts": self.world_model_trainer.update_bursts,
                    "world_model_gradient_updates": self.world_model_trainer.gradient_updates,
                    "terminal_value_update_bursts": self.terminal_trainer.update_bursts,
                    "terminal_value_gradient_updates": self.terminal_trainer.gradient_updates,
                    "mpc_plans_computed": self._plans_computed,
                    "mpc_planning_seconds": self._planning_seconds,
                },
            }
            replay_path = output / f"online_replay_{iteration}.pt"
            replay_latest = output / "online_replay_latest.pt"
            temporary = replay_path.with_suffix(replay_path.suffix + ".tmp")
            torch.save(replay_payload, temporary)
            temporary.replace(replay_path)
            shutil.copy2(replay_path, replay_latest)
            saved.extend((str(replay_path), str(replay_latest)))
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
    """Make imagined execution and rewards match the real training contract."""

    terminal_enabled = (
        getattr(args, "mpc_terminal_value", "enabled") == "enabled"
    )
    if getattr(args, "mpc_uncertainty", "enabled") == "disabled":
        # Remove every uncertainty-dependent planning decision while keeping
        # the same ensemble and its training/diagnostics for matched ablations.
        mpc_config.uncertainty_penalty = 0.0
        mpc_config.return_uncertainty_penalty = 0.0
        mpc_config.max_state_uncertainty = None
        mpc_config.max_return_uncertainty = None
        mpc_config.terminal_value_uncertainty_gating = False
        mpc_config.terminal_value_max_uncertainty = None
    mpc_config.reward_source = "analytical"
    mpc_config.shooting_options = bool(getattr(args, "shooting_options", False))
    if mpc_config.shooting_options:
        mpc_config.terminal_value_coefficient = 1.0
    mpc_config.learned_reward_coefficient = 0.0
    mpc_config.analytical_reward_coefficient = 1.0
    mpc_config.apply_skill_fallback_in_rollout = bool(
        getattr(args, "use_geometric_skill_fallback", True)
    )
    mpc_config.apply_collision_avoidance_in_rollout = bool(
        getattr(args, "collision_avoidance", False)
    )
    # The aligned analytical reward already contains the configured invalid
    # skill penalty. Do not add the objective-level copy, which would charge
    # the same false selection twice. The real and imagined execution paths
    # both apply the configured geometric fallback; the environment's request
    # diagnostic still teaches PPO the affordance boundary.
    mpc_config.invalid_skill_penalty = 0.0
    mpc_config.skill_switch_penalty = 0.0
    mpc_config.command_change_penalty = 0.0
    mpc_config.analytical_robot_collision_distance_m = float(
        getattr(args, "robot_collision_distance", 0.70)
    )
    mpc_config.analytical_robot_collision_lookahead_s = float(
        getattr(args, "robot_collision_lookahead", 0.25)
    )
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
    parser.add_argument(
        "--world-model-checkpoint",
        default="checkpoints/reproduction/world_model/best.pt",
        help=(
            "Pretrained dynamics initialization. Set to an empty value only "
            "for a deliberate from-scratch world-model ablation."
        ),
    )
    parser.add_argument("--freeze-world-model", action="store_true",
                        help="Require a compatible pretrained model and disable all dynamics updates.")
    parser.add_argument("--world-model-config", default="configs/world_model_as2.yaml")
    parser.add_argument("--mpc-query-budget", type=int, default=0,
                        help="Maximum matches queried per plan; zero queries all matches.")
    parser.add_argument("--mpc-quality-filter", action="store_true")
    parser.add_argument("--mpc-value-fallback", action="store_true",
                        help="Plan on rewards alone until terminal value passes validation.")
    parser.add_argument("--mpc-advantage-margin", type=float, default=.1)
    parser.add_argument("--mpc-max-ball-error", type=float, default=.25)
    parser.add_argument("--mpc-config", default="configs/mpc_joint_teams.yaml")
    parser.add_argument("--mpc-profile", default="teacher_training")
    parser.add_argument("--terminal-value-checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-subdir",
        default="high_level_online_mpc",
        help="Subdirectory under the W&B run used for online policy/model checkpoints.",
    )
    parser.add_argument(
        "--world-model-update-interval", type=int, default=50,
        help="PPO iterations between online world-model/value update bursts.",
    )
    parser.add_argument(
        "--world-model-replay-buffer-size", type=int, default=100000,
        help="Maximum high-level match transitions retained across opponent snapshots.",
    )
    parser.add_argument("--world-model-replay-recent-fraction", type=float, default=0.5)
    parser.add_argument("--world-model-replay-recent-window", type=int, default=10000)
    parser.add_argument("--world-model-update-batch-size", type=int, default=1024)
    parser.add_argument("--world-model-updates-per-interval", type=int, default=4)
    parser.add_argument(
        "--mpc-horizon", type=int, default=2,
        help="Number of high-level skill intervals imagined by MPC.",
    )
    parser.add_argument(
        "--mpc-num-samples", "--mpc-num-candidates",
        dest="mpc_num_samples", type=int, default=128,
        help="Hybrid action sequences sampled per CEM iteration.",
    )
    parser.add_argument(
        "--mpc-num-iterations", type=int, default=3,
        help="CEM refinement iterations per MPC call.",
    )
    parser.add_argument(
        "--mpc-replan-interval",
        type=int,
        default=2,
        help=(
            "Run CEM every N high-level decisions. The default N=2 halves "
            "planner cost while PPO acts on skipped decisions."
        ),
    )
    parser.add_argument(
        "--mpc-kl-coefficient", "--mpc-kl-coef", type=float, default=0.05,
        help="Coefficient on forward KL(MPC || high-level policy); zero disables distillation.",
    )
    quality_score_group = parser.add_mutually_exclusive_group()
    quality_score_group.add_argument(
        "--mpc-quality-first-action",
        dest="mpc_quality_first_action",
        action="store_true",
        help=(
            "Score teacher labels using the first macro-action only. This "
            "matches execution, where the student supplies continuation actions."
        ),
    )
    quality_score_group.add_argument(
        "--no-mpc-quality-first-action",
        dest="mpc_quality_first_action",
        action="store_false",
        help="Score teacher labels over the full imagined MPC horizon.",
    )
    parser.set_defaults(mpc_quality_first_action=True)
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
        "--mpc-uncertainty", choices=("enabled", "disabled"), default="enabled",
        help=("Use configured uncertainty penalties and gates, or disable all "
              "uncertainty-dependent planning decisions for an ablation."),
    )
    parser.add_argument(
        "--mpc-terminal-value",
        choices=("enabled", "disabled"),
        default="disabled",
        help=(
            "Enable the learned terminal continuation value. Disabled by "
            "default because online training has no pretrained value model."
        ),
    )
    parser.add_argument(
        "--mpc-warmup-steps", "--mpc-warmup", type=int, default=2400,
        help="Vectorized high-level environment steps collected before MPC/distillation can start.",
    )
    parser.add_argument(
        "--mpc-min-replay-size", type=int, default=20_000,
        help=(
            "Minimum real high-level transitions required in dynamics replay "
            "before MPC can run. This readiness gate is applied in addition "
            "to --mpc-warmup-steps."
        ),
    )
    parser.add_argument(
        "--mpc-min-world-model-updates", type=int, default=48,
        help="Minimum online world-model gradient updates before MPC can run.",
    )
    parser.add_argument(
        "--mpc-min-terminal-value-updates", type=int, default=4,
        help=(
            "Minimum online terminal-value gradient updates before MPC can "
            "run when no pretrained terminal-value checkpoint was loaded."
        ),
    )
    timeout_group = parser.add_mutually_exclusive_group()
    timeout_group.add_argument(
        "--terminal-bootstrap-on-timeout",
        dest="terminal_bootstrap_on_timeout",
        action="store_true",
        help="Bootstrap timeout return targets from V(next_state).",
    )
    timeout_group.add_argument(
        "--no-terminal-bootstrap-on-timeout",
        dest="terminal_bootstrap_on_timeout",
        action="store_false",
        help="Treat timeout targets as finite-horizon returns.",
    )
    parser.set_defaults(terminal_bootstrap_on_timeout=True)
    parser.add_argument(
        "--terminal-timeout-tail-exclusion", type=int, default=8,
        help=(
            "Number of final timeout states omitted until a terminal-value "
            "model is ready to bootstrap them."
        ),
    )
    parser.add_argument(
        "--online-replay-checkpoint", default=None,
        help="Optional saved dynamics/value replay checkpoint to restore.",
    )
    replay_resume_group = parser.add_mutually_exclusive_group()
    replay_resume_group.add_argument(
        "--auto-resume-online-replay",
        dest="auto_resume_online_replay",
        action="store_true",
        help="When resuming a policy, also restore adjacent online_replay_latest.pt if present.",
    )
    replay_resume_group.add_argument(
        "--no-auto-resume-online-replay",
        dest="auto_resume_online_replay",
        action="store_false",
        help="Start dynamics/value replay and readiness counters from scratch.",
    )
    parser.set_defaults(auto_resume_online_replay=False)
    replay_save_group = parser.add_mutually_exclusive_group()
    replay_save_group.add_argument(
        "--save-online-replay", dest="save_online_replay", action="store_true"
    )
    replay_save_group.add_argument(
        "--no-save-online-replay", dest="save_online_replay", action="store_false"
    )
    parser.set_defaults(save_online_replay=True)
    parser.add_argument("--mpc-distillation-batch-size", type=int, default=1024)
    # Online CEM is substantially more expensive than policy-only PPO.
    parser.set_defaults(
        # Match the standard MAPPO/discrete trainers so comparisons use the
        # same number of parallel simulator environments by default.
        num_envs=256,
        project="as2_high_level_online_mpc",
        self_play_update_interval=400,
    )
    return parser


def validate_online_args(args):
    if getattr(args, "mpc_query_budget", 0) < 0:
        raise ValueError("--mpc-query-budget cannot be negative")
    if getattr(args, "mpc_max_ball_error", .25) <= 0 or getattr(args, "mpc_advantage_margin", .1) < 0:
        raise ValueError("MPC error threshold must be positive and advantage margin non-negative")
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
    for name in (
        "mpc_min_replay_size",
        "mpc_min_world_model_updates",
        "mpc_min_terminal_value_updates",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.mpc_min_replay_size) > int(args.world_model_replay_buffer_size):
        raise ValueError(
            "--mpc-min-replay-size cannot exceed "
            "--world-model-replay-buffer-size"
        )
    if int(args.terminal_timeout_tail_exclusion) < 0:
        raise ValueError("--terminal-timeout-tail-exclusion cannot be negative")
    for name in (
        "mpc_horizon",
        "mpc_num_samples",
        "mpc_num_iterations",
        "mpc_replan_interval",
    ):
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
    from scripts.train_high_level import (
        configure_high_level_cfg,
        load_skill_policies,
        resolved_high_level_reward_scales,
    )
    from quadruped.envs.base.legged_robot_config import Cfg
    from quadruped.envs.as2.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    from quadruped.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from quadruped.envs.wrappers.shared_self_play_wrapper import SharedPolicySelfPlayWrapper
    try:
        from scripts.collect_world_model_data import TerminalStateCapture
    except ModuleNotFoundError as error:
        if error.name != "scripts.collect_world_model_data":
            raise
        # The collector was moved to discard/ in lightweight deployments; the
        # simulator controller contains the same capture primitive.
        from quadruped.mpc.simulator_controller import TerminalStateCapture
    from quadruped_learn.ppo_cse import Runner, RunnerArgs
    from quadruped_learn.ppo_cse.actor_critic import AC_Args
    from quadruped_learn.ppo_cse.ppo import PPO_Args
    import wandb

    configure_high_level_cfg(Cfg, args)
    skills = load_skill_policies(args)
    args.skill_fingerprint = skill_fingerprint(skills)
    if args.validate_skill_policies_only:
        print("Validated all AS2 low-level skill policies; training was not started.")
        return
    RunnerArgs.resume = bool(args.resume)
    RunnerArgs.resume_policy_only = args.resume_mode == "policy-only"
    RunnerArgs.resume_path = args.resume_run
    RunnerArgs.resume_checkpoint = args.resume_checkpoint
    RunnerArgs.save_video_interval = args.save_video_interval
    RunnerArgs.checkpoint_dir = args.checkpoint_dir
    RunnerArgs.num_steps_per_env = getattr(args, "rollout_steps", 24)
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
            "opponent_action_selection": "sample",
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
    config = __import__("quadruped.world_model.config", fromlist=["load_config"]).load_config(args.world_model_config)
    if not getattr(args, "freeze_world_model", False) and getattr(args, "shooting_options", False) and args.world_model_checkpoint == "checkpoints/reproduction/world_model/best.pt":
        print("Shooting option state changed: fitting a fresh dynamics model from this run.")
        args.world_model_checkpoint = None
    if getattr(args, "freeze_world_model", False) and not args.world_model_checkpoint:
        raise ValueError("--freeze-world-model requires --world-model-checkpoint")
    world_model, world_checkpoint = _build_world_model(args.world_model_checkpoint, state_adapter, config, args.device)
    args.mpc_require_model_refresh = (
        world_checkpoint.get("training_config", {}).get("skill_fingerprint")
        != args.skill_fingerprint
    )
    if getattr(args, "freeze_world_model", False):
        if args.mpc_require_model_refresh:
            raise ValueError("Frozen world model must have matching skill provenance; recollect and retrain with the current skills")
        trained_steps = world_checkpoint.get("training_config", {}).get("world_model", {}).get("macro_action_steps")
        if trained_steps != int(match_env.control_interval):
            raise ValueError("Frozen world model macro-action interval does not match the environment")
        world_model.requires_grad_(False)
        world_model.eval()
    if args.mpc_require_model_refresh:
        print("MPC model has changed/unknown skill provenance; requiring fresh online updates before teaching.")
    mpc_config, mpc_payload = load_mpc_config(args.mpc_config, args.mpc_profile)
    if args.mpc_horizon is not None:
        mpc_config.horizon = args.mpc_horizon
    if args.mpc_num_samples is not None:
        mpc_config.num_candidates = args.mpc_num_samples
        target_elites = max(1, mpc_config.num_candidates // 8)
        mpc_config.num_elites = max(
            1,
            min(
                mpc_config.num_elites,
                target_elites,
                mpc_config.num_candidates - 1,
            ),
        )
    if args.mpc_num_iterations is not None:
        mpc_config.num_iterations = args.mpc_num_iterations
    if bool(getattr(args, "mpc_quality_filter", False)):
        mpc_config.apply_skill_fallback_in_rollout = bool(args.use_geometric_skill_fallback)
        mpc_config.apply_collision_avoidance_in_rollout = bool(args.collision_avoidance)
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
        reward_scales=resolved_high_level_reward_scales(args),
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
        optimizer_states=(None if args.mpc_require_model_refresh or getattr(args, "freeze_world_model", False) else world_checkpoint.get("optimizer_states")),
        loss_config=config.get("loss", {}),
        pretrained=bool(args.world_model_checkpoint),
    )
    wm_trainer.frozen = bool(getattr(args, "freeze_world_model", False))
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
    from scripts.train_high_level import save_checkpoint_config
    save_checkpoint_config(run, RunnerArgs.checkpoint_dir)
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
    from scripts.train_high_level import successful_training_exit
    successful_training_exit()
