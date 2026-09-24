"""Train the high-level policy by replaying MPC-labelled transitions.

This is a replay-first companion to ``train_high_level_online_mpc.py``.  It
keeps the existing Isaac Gym/self-play/world-model/MPC setup, but stores every
MPC distillation example in a bounded replay buffer and samples that buffer
repeatedly after each rollout.  The default retains two PPO policy epochs so
real simulator outcomes continue to correct imperfect MPC labels.

This is deliberately a policy-distillation experiment, not a drop-in SAC
implementation: the coordinator has a categorical skill choice plus
skill-conditional continuous commands, while the existing RLPD reference code
only supports a continuous tanh-Gaussian actor. A small on-policy PPO update is
kept enabled by default so imperfect world-model/MPC labels cannot erase the
actual game-outcome learning signal.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Allow both ``python -m scripts.train_high_level_mpc_replay`` and the usual
# ``python scripts/train_high_level_mpc_replay.py`` launch form.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the online-MPC module before importing torch here.  Its import guard
# loads Isaac Gym before PyTorch, which is required by Isaac Gym's bindings.
import scripts.train_high_level_online_mpc as online_mpc

import torch


class ReplayMPCSelfPlayExtension(online_mpc.OnlineMPCSelfPlayExtension):
    """Online-MPC extension with persistent, replayed teacher targets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        replay_capacity = int(self.args.mpc_replay_capacity)
        replay_recent_fraction = float(self.args.mpc_replay_recent_fraction)
        replay_recent_window = int(self.args.mpc_replay_recent_window)
        if replay_capacity < 1:
            raise ValueError("--mpc-replay-capacity must be positive")
        if not 0.0 <= replay_recent_fraction <= 1.0:
            raise ValueError("--mpc-replay-recent-fraction must lie in [0, 1]")
        if replay_recent_window < 1:
            raise ValueError("--mpc-replay-recent-window must be positive")
        self.teacher_replay = online_mpc.WorldModelReplayBuffer(
            capacity=replay_capacity,
            recent_fraction=replay_recent_fraction,
            recent_window=replay_recent_window,
            seed=int(getattr(self.args, "seed", 42)) + 17,
        )
        self._replay_updates = 0
        self._replay_samples = 0
        self._teacher_targets_seen = 0
        self._teacher_targets_rejected = 0
        self._kl_ramp_start_step = None
        self._load_teacher_replay()

    def bind_runner(self, runner):
        super().bind_runner(runner)
        # ``train_high_level_online_mpc`` configures PPO before constructing
        # this extension.  Set it again defensively so this script remains
        # replay-first even if the base trainer changes its defaults later.
        from quadruped_learn.ppo_cse.ppo import PPO_Args

        PPO_Args.num_learning_epochs = int(self.args.on_policy_ppo_epochs)

    def _load_teacher_replay(self):
        checkpoint = getattr(self.args, "mpc_replay_checkpoint", None)
        if (
            checkpoint is None
            and bool(getattr(self.args, "resume", False))
            and bool(getattr(self.args, "auto_resume_mpc_replay", False))
        ):
            resume = getattr(self.args, "resume_checkpoint", None)
            if resume:
                candidate = Path(resume).expanduser().resolve().parent / (
                    "mpc_teacher_replay_latest.pt"
                )
                if candidate.is_file():
                    checkpoint = str(candidate)
        if not checkpoint:
            return
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MPC replay checkpoint does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "dribblebot_mpc_teacher_replay_v1":
            raise ValueError(f"Unsupported MPC replay checkpoint format: {path}")
        if not self._replay_skills_match(payload):
            print(f"Discarding MPC teacher replay with changed or unknown skills: {path}")
            return
        replay_state = dict(payload.get("replay", {"items": payload.get("items", [])}))
        if bool(getattr(self.args, "mpc_quality_filter", False)):
            replay_state["items"] = [
                item for item in replay_state.get("items", [])
                if "teacher_step" in item and "teacher_weight" in item
            ]
        self.teacher_replay.load_state_dict(replay_state)
        counters = payload.get("counters", {})
        self._replay_updates = int(counters.get("updates", 0))
        self._replay_samples = int(counters.get("samples", 0))
        self._teacher_targets_seen = int(counters.get("teacher_targets_seen", 0))
        self._teacher_targets_rejected = int(
            counters.get("teacher_targets_rejected", 0)
        )
        ramp_start = counters.get("kl_ramp_start_step")
        self._kl_ramp_start_step = (
            None if ramp_start is None else int(ramp_start)
        )
        print(
            f"Loaded {len(self.teacher_replay)} MPC distillation transitions "
            f"from {path}"
        )

    def _store_current_rollout_targets(self):
        records = getattr(self, "_distill_records", [])
        # The base extension builds this short-lived list only when the MPC KL
        # coefficient is positive.  Move its tensors into the bounded replay
        # immediately, then clear it so no transition is consumed twice by the
        # base one-shot distillation update.
        self._distill_records = []
        if not records:
            return 0
        keys = records[0].keys()
        merged = {
            key: torch.cat([record[key] for record in records], dim=0)
            for key in keys
            if torch.is_tensor(records[0][key])
        }
        valid = merged.pop("teacher_valid", None)
        if valid is not None:
            valid = valid.to(dtype=torch.bool).reshape(-1)
            if valid.numel() != merged["obs_history"].shape[0]:
                raise ValueError("MPC teacher-valid mask does not match replay rows")
            # OOD-relaxed plans are safe acting fallbacks, not reliable
            # teacher labels. Do not let model extrapolations enter replay.
            if not bool(valid.all()):
                self._teacher_targets_rejected += int((~valid).sum().item())
                merged = {key: value[valid] for key, value in merged.items()}
        self._teacher_targets_seen += int(
            valid.numel() if valid is not None else merged["obs_history"].shape[0]
        )
        if not merged["obs_history"].shape[0]:
            return 0
        self.teacher_replay.add_batch(merged)
        return int(merged["obs_history"].shape[0])

    def process_env_step(self, *args, **kwargs):
        super().process_env_step(*args, **kwargs)
        self._store_current_rollout_targets()

    @staticmethod
    def _finite_parameters(module):
        return all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters())

    def _replay_update(self):
        target_coefficient = float(self.args.mpc_kl_coefficient)
        updates_requested = int(self.args.mpc_replay_updates_per_rollout)
        batch_size = int(self.args.mpc_replay_batch_size)
        minimum_size = int(self.args.mpc_replay_min_size)
        guarded = bool(getattr(self.args, "mpc_quality_filter", False))
        pruned = 0
        if guarded:
            pruned = self.teacher_replay.prune_teacher_targets(
                self._high_level_steps, self.args.mpc_teacher_max_age)
        freshness_metrics = {
            "mpc_replay/eligible_size": float(len(self.teacher_replay)),
            "mpc_replay/expired_targets_removed": float(pruned),
            "mpc_replay/samples_per_update": 0.0,
        }
        if target_coefficient <= 0.0:
            return {
                "mpc_replay/enabled": 0.0,
                "mpc_replay/size": float(len(self.teacher_replay)),
                "mpc_replay/effective_kl_coefficient": 0.0,
            }
        if updates_requested < 1:
            raise ValueError("--mpc-replay-updates-per-rollout must be positive")
        if batch_size < 1:
            raise ValueError("--mpc-replay-batch-size must be positive")
        if minimum_size < 1:
            raise ValueError("--mpc-replay-min-size must be positive")
        quality_ready = (
            not bool(getattr(self.args, "mpc_quality_filter", False))
            or (getattr(self, "_quality_samples", 0) >= 512
                and getattr(self, "_ball_prediction_error", float("inf"))
                <= self.args.mpc_max_ball_error)
        )
        if not self._planner_ready() or not quality_ready:
            return {
                **freshness_metrics,
                "mpc_replay/enabled": 1.0,
                "mpc_replay/update_skipped_not_ready": 1.0,
                "mpc_replay/size": float(len(self.teacher_replay)),
            }
        adaptive_batch = bool(getattr(self.args, "mpc_adaptive_replay_batch", False))
        required_size = minimum_size if adaptive_batch else max(minimum_size, batch_size)
        if len(self.teacher_replay) < required_size:
            return {
                **freshness_metrics,
                "mpc_replay/enabled": 1.0,
                "mpc_replay/update_skipped_small_buffer": 1.0,
                "mpc_replay/size": float(len(self.teacher_replay)),
            }

        if adaptive_batch:
            batch_size = min(batch_size, len(self.teacher_replay))

        if self._kl_ramp_start_step is None:
            self._kl_ramp_start_step = int(self._high_level_steps)
        ramp_steps = int(self.args.mpc_kl_ramp_steps)
        ramp_elapsed = max(
            0, int(self._high_level_steps) - self._kl_ramp_start_step
        )
        ramp_progress = (
            1.0
            if ramp_steps == 0
            else min(1.0, float(ramp_elapsed) / float(ramp_steps))
        )
        coefficient = target_coefficient * ramp_progress
        if coefficient <= 0.0:
            return {
                **freshness_metrics,
                "mpc_replay/enabled": 1.0,
                "mpc_replay/size": float(len(self.teacher_replay)),
                "mpc_replay/updates": 0.0,
                "mpc_replay/effective_kl_coefficient": 0.0,
                "mpc_replay/target_kl_coefficient": target_coefficient,
                "mpc_replay/kl_ramp_progress": ramp_progress,
            }

        policy = self.runner.alg.actor_critic
        device = self.runner.device
        real_guard = None
        if getattr(self.args, "mpc_real_rollout_guard", False):
            anchor = getattr(self, "_real_rollout_anchor", None)
            if anchor is None:
                return {**freshness_metrics, "mpc_replay/update_skipped_missing_real_rollout": 1.0}
            from quadruped.mpc.rollout_guard import RealRolloutGuard
            from quadruped_learn.ppo_cse.ppo import PPO_Args
            real_guard = RealRolloutGuard(policy, anchor, PPO_Args.clip_param)
        total_loss = 0.0
        total_categorical = 0.0
        total_continuous = 0.0
        actual_kl = 0.0
        used_samples = 0
        rejected_steps = 0
        performed = 0
        for _ in range(updates_requested):
            batch = self.teacher_replay.sample(batch_size, device=device, replacement=not guarded)
            histories = batch["obs_history"].float()
            target_command_mean = batch["target_command_mean"].float()
            target_std = batch["target_std"].float().abs().clamp_min(1.0e-4)
            target_probs = batch["target_probs"].float().clamp_min(1.0e-6)
            target_masks = batch["target_masks"].float()
            target_probs = target_probs / target_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-6)
            if not all(
                bool(torch.isfinite(value).all())
                for value in (
                    histories,
                    target_command_mean,
                    target_std,
                    target_probs,
                    target_masks,
                )
            ):
                continue

            policy.update_distribution(histories)
            policy_mean = policy.action_mean
            policy_std = policy.action_std.clamp(min=1.0e-6)
            if not (
                bool(torch.isfinite(policy_mean).all())
                and bool(torch.isfinite(policy_std).all())
            ):
                continue

            categorical = (
                target_probs
                * (target_probs.log() - torch.log_softmax(policy_mean[:, :3], dim=-1))
            ).sum(dim=-1).mean()
            policy_command_mean = policy_mean[:, 3:]
            policy_command_std = policy_std[:, 3:].clamp(min=1.0e-6)
            command_kl = (
                torch.log(policy_command_std[:, None])
                - torch.log(target_std)
                + (
                    target_std.square()
                    + (target_command_mean - policy_command_mean[:, None]).square()
                )
                / (2.0 * policy_command_std[:, None].square())
                - 0.5
            )
            teacher_skill = target_probs.argmax(dim=-1)
            rows = torch.arange(
                target_probs.shape[0], device=target_probs.device
            )
            selected_command_kl = command_kl[rows, teacher_skill]
            selected_mask = target_masks[rows, teacher_skill]
            continuous = (
                (selected_command_kl * selected_mask).sum(dim=-1)
                / selected_mask.sum(dim=-1).clamp(min=1.0)
            ).mean()
            guarded = bool(getattr(self.args, "mpc_quality_filter", False))
            if guarded:
                from quadruped.mpc.teacher_quality import bounded_mean_loss, policy_kl
                old_mean, old_std = policy_mean.detach().clone(), policy_std.detach().clone()
                weights = batch.get("teacher_weight", torch.ones(len(histories), device=device)).clamp(0., 1.)
                categorical_rows = (target_probs * (target_probs.log()-torch.log_softmax(policy_mean[:, :3], -1))).sum(-1)
                mean_rows = bounded_mean_loss(policy_mean, target_command_mean, target_probs, target_masks)
                # Never imitate CEM's search variance through the policy std.
                loss = ((categorical_rows+mean_rows)*weights).mean()
                continuous = mean_rows.mean()
            else:
                loss = coefficient * (categorical + continuous)
            if not bool(torch.isfinite(loss)):
                continue

            self._distill_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            if not bool(torch.isfinite(grad_norm)):
                self._distill_optimizer.zero_grad(set_to_none=True)
                continue
            backup = [parameter.detach().clone() for parameter in policy.parameters()]
            if guarded:
                import copy
                optimizer_backup = copy.deepcopy(self._distill_optimizer.state_dict())
            self._distill_optimizer.step()
            if guarded:
                # Adam is approximately invariant to a scalar loss multiplier.
                # Apply the ramp to the parameter step instead, and backtrack
                # until the actual hybrid-policy KL satisfies the trust limit.
                proposed = [parameter.detach().clone() for parameter in policy.parameters()]
                accepted = False
                fraction = min(1., coefficient / .01)  # 0.01 is the nominal full teacher step
                with torch.no_grad():
                    for _backtrack in range(8):
                        for parameter, previous, proposal in zip(policy.parameters(), backup, proposed):
                            parameter.copy_(previous + fraction*(proposal-previous))
                        policy.update_distribution(histories)
                        delta = policy_kl(old_mean, old_std, policy.action_mean, policy.action_std)
                        anchor_ok = real_guard is None or real_guard.accepts(policy, self.args.mpc_policy_kl_limit)
                        if torch.isfinite(delta) and float(delta) <= self.args.mpc_policy_kl_limit and anchor_ok:
                            actual_kl += float(delta)
                            accepted = True
                            break
                        fraction *= .5
                    if not accepted:
                        for parameter, previous in zip(policy.parameters(), backup):
                            parameter.copy_(previous)
                        self._distill_optimizer.load_state_dict(optimizer_backup)
                        rejected_steps += 1
                        continue
            if not self._finite_parameters(policy):
                with torch.no_grad():
                    for parameter, previous in zip(policy.parameters(), backup):
                        parameter.copy_(previous)
                        state = self._distill_optimizer.state.get(parameter)
                        if state is not None:
                            state.clear()
                self._distill_optimizer.zero_grad(set_to_none=True)
                continue
            from quadruped_learn.ppo_cse.actor_critic import AC_Args

            with torch.no_grad():
                policy.std.clamp_(
                    min=AC_Args.min_action_std,
                    max=AC_Args.max_action_std,
                )
            total_loss += float(loss.detach())
            total_categorical += float(categorical.detach())
            total_continuous += float(continuous.detach())
            performed += 1
            used_samples += len(histories)

        self._replay_updates += performed
        self._replay_samples += used_samples
        return {
            **freshness_metrics,
            "mpc_replay/samples_per_update": used_samples / max(performed, 1),
            "mpc_replay/enabled": 1.0,
            "mpc_replay/size": float(len(self.teacher_replay)),
            "mpc_replay/updates": float(performed),
            "mpc_replay/actual_policy_kl": actual_kl / max(performed, 1),
            "mpc_replay/real_rollout_guard_enabled": float(real_guard is not None),
            "mpc_replay/rejected_policy_steps": rejected_steps,
            "mpc_replay/updates_total": float(self._replay_updates),
            "mpc_replay/samples_total": float(self._replay_samples),
            "mpc_replay/kl_loss": total_loss / max(performed, 1),
            "mpc_replay/categorical_kl": total_categorical / max(performed, 1),
            "mpc_replay/continuous_kl": total_continuous / max(performed, 1),
            "mpc_replay/effective_kl_coefficient": coefficient,
            "mpc_replay/target_kl_coefficient": target_coefficient,
            "mpc_replay/kl_ramp_progress": ramp_progress,
            "mpc_replay/reuse_ratio": float(
                self._replay_samples / max(len(self.teacher_replay), 1)
            ),
            "mpc_replay/teacher_targets_seen": float(self._teacher_targets_seen),
            "mpc_replay/teacher_targets_rejected": float(
                self._teacher_targets_rejected
            ),
        }

    def after_rollout(self, iteration):
        if getattr(self.args, "mpc_real_rollout_guard", False):
            # Returns/advantages are computed before this hook; PPO clears
            # storage after its update, so preserve a representative real batch.
            storage = self.runner.alg.storage
            count = storage.observation_histories.flatten(0, 1).shape[0]
            indices = torch.randperm(count, device=storage.actions.device)[:2048]
            self._real_rollout_anchor = {
                key: value.flatten(0, 1)[indices].detach().clone()
                for key, value in {'histories': storage.observation_histories,
                                   'actions': storage.actions,
                                   'log_prob': storage.actions_log_prob,
                                   'advantages': storage.advantages}.items()
            }
        super().after_rollout(iteration)

    def after_policy_update(self, iteration):
        # Do not call the base implementation: it would consume a one-shot
        # rollout cache.  World-model/value metrics are produced in
        # ``after_rollout`` and are forwarded here unchanged.
        metrics = dict(self._metrics)
        self._metrics = {}
        metrics.update(self._replay_update())
        metrics.update(self._readiness_metrics())
        self._check_teacher_activity(metrics, iteration)
        return metrics

    def _check_teacher_activity(self, metrics, iteration):
        if self._high_level_steps < int(self.args.mpc_warmup_steps):
            self._teacher_idle_rollouts = 0
            return
        updated = metrics.get('mpc_replay/updates', 0) > 0
        idle = 0 if updated else getattr(self, '_teacher_idle_rollouts', 0) + 1
        self._teacher_idle_rollouts = idle
        metrics['mpc_replay/consecutive_idle_rollouts'] = idle
        reason = next((key for key, value in metrics.items()
                       if key.startswith('mpc_replay/update_skipped_') and value),
                      'rejected proposals or zero teacher ramp')
        if idle and idle % 50 == 0:
            print(f'MPC has made no teacher policy updates for {idle} rollouts: {reason}; '
                  f'eligible targets={metrics.get("mpc_replay/eligible_size", 0)}', flush=True)
        limit = int(getattr(self.args, 'mpc_max_idle_rollouts', 0))
        if limit and idle >= limit:
            from quadruped_learn.ppo_cse import RunnerArgs
            directory = Path(RunnerArgs.checkpoint_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'actor_critic_mpc_idle_{iteration}.pt'
            temporary = path.with_suffix('.pt.tmp')
            torch.save(self.runner.alg.actor_critic.state_dict(), temporary)
            temporary.replace(path)
            raise RuntimeError(f'MPC teacher inactive for {idle} rollouts: {reason}. '
                               f'Stopping the experiment; recovery policy weights saved to {path}. '
                               'This recovery file is not a full optimizer/replay checkpoint.')

    def save_checkpoint(self, path, iteration):
        saved = list(super().save_checkpoint(path, iteration))
        if not bool(getattr(self.args, "save_mpc_replay", False)):
            return saved
        output = Path(path)
        payload = {
            "format": "dribblebot_mpc_teacher_replay_v1",
            "skill_fingerprint": getattr(self.args, "skill_fingerprint", None),
            "execution_contract": "shooting_option_v1" if getattr(self.args, "shooting_options", False) else "legacy",
            "capacity": self.teacher_replay.capacity,
            "recent_fraction": self.teacher_replay.recent_fraction,
            "recent_window": self.teacher_replay.recent_window,
            "replay": self.teacher_replay.state_dict(),
            "counters": {
                "updates": self._replay_updates,
                "samples": self._replay_samples,
                "kl_ramp_start_step": self._kl_ramp_start_step,
                "teacher_targets_seen": self._teacher_targets_seen,
                "teacher_targets_rejected": self._teacher_targets_rejected,
            },
        }
        replay_path = output / f"mpc_teacher_replay_{iteration}.pt"
        latest_path = output / "mpc_teacher_replay_latest.pt"
        temporary = replay_path.with_suffix(replay_path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(replay_path)
        shutil.copy2(replay_path, latest_path)
        saved.extend((str(replay_path), str(latest_path)))
        return saved


def build_arg_parser():
    parser = online_mpc.build_arg_parser()
    parser.add_argument('--mpc-max-idle-rollouts', type=int, default=0,
                        help='Stop after this many rollouts without a teacher update after warmup; 0 warns only.')
    parser.add_argument('--mpc-adaptive-replay-batch', action='store_true',
                        help='Use available distinct teacher targets once minimum replay size is met.')
    parser.add_argument("--mpc-real-rollout-guard", action="store_true",
                        help="Reject teacher steps that degrade the real-rollout PPO surrogate.")
    parser.description = (
        "Train the high-level policy by replaying MPC distillation transitions."
    )
    parser.add_argument(
        "--on-policy-ppo-epochs",
        type=int,
        default=2,
        help=(
            "PPO policy epochs per rollout. Two epochs preserve real game "
            "outcome learning while replayed MPC updates provide guidance. "
            "Set zero only for an intentional replay-only ablation."
        ),
    )
    # Teacher labels are model-generated and become stale as the policy and
    # world model evolve. Keep replay deliberately recent and light-weight;
    # PPO remains the source of truth for real game outcomes.
    parser.add_argument("--mpc-policy-kl-limit", type=float, default=.003)
    parser.add_argument("--mpc-teacher-max-age", type=int, default=1024)
    parser.add_argument("--mpc-replay-capacity", type=int, default=50_000)
    parser.add_argument("--mpc-replay-recent-fraction", type=float, default=1.0)
    parser.add_argument("--mpc-replay-recent-window", type=int, default=20_000)
    parser.add_argument("--mpc-replay-batch-size", type=int, default=1024)
    parser.add_argument(
        "--mpc-replay-updates-per-rollout",
        type=int,
        default=1,
        help="Replay gradient updates after each environment rollout.",
    )
    parser.add_argument(
        "--mpc-replay-min-size",
        type=int,
        default=20_000,
        help="Minimum representative MPC target replay size before updates begin.",
    )
    parser.add_argument(
        "--mpc-kl-ramp-steps",
        type=int,
        default=40_000,
        help=(
            "High-level decision steps over which the replay KL coefficient "
            "ramps from zero to --mpc-kl-coefficient."
        ),
    )
    parser.add_argument(
        "--mpc-replay-checkpoint",
        default=None,
        help="Optional replay checkpoint to load before training.",
    )
    replay_resume_group = parser.add_mutually_exclusive_group()
    replay_resume_group.add_argument(
        "--auto-resume-mpc-replay",
        dest="auto_resume_mpc_replay",
        action="store_true",
        help="Restore adjacent mpc_teacher_replay_latest.pt when resuming.",
    )
    replay_resume_group.add_argument(
        "--no-auto-resume-mpc-replay",
        dest="auto_resume_mpc_replay",
        action="store_false",
        help="Start teacher replay empty when resuming a policy.",
    )
    parser.set_defaults(auto_resume_mpc_replay=False)
    parser.add_argument(
        "--save-mpc-replay",
        action="store_true",
        help="Save the bounded MPC replay beside policy/model checkpoints.",
    )
    parser.set_defaults(
        project="as2_high_level_mpc_replay",
        checkpoint_subdir="high_level_mpc_replay",
        # Look beyond immediate contact while limiting model-error growth and
        # planner cost. Learn continuation value before distilling its targets.
        mpc_horizon=4,
        mpc_terminal_value="enabled",
        mpc_min_terminal_value_updates=48,
        mpc_kl_coefficient=0.01,
        mpc_guidance_reward_coefficient=0.0,
        save_mpc_replay=True,
    )
    return parser


def validate_args(args):
    if args.mpc_max_idle_rollouts < 0:
        raise ValueError('--mpc-max-idle-rollouts must be nonnegative')
    if args.mpc_real_rollout_guard and not args.mpc_quality_filter:
        raise ValueError("--mpc-real-rollout-guard requires --mpc-quality-filter")
    if int(args.on_policy_ppo_epochs) < 0:
        raise ValueError("--on-policy-ppo-epochs cannot be negative")
    if int(args.mpc_replay_capacity) < 1:
        raise ValueError("--mpc-replay-capacity must be positive")
    if not 0.0 <= float(args.mpc_replay_recent_fraction) <= 1.0:
        raise ValueError("--mpc-replay-recent-fraction must lie in [0, 1]")
    if int(args.mpc_replay_recent_window) < 1:
        raise ValueError("--mpc-replay-recent-window must be positive")
    if int(args.mpc_replay_batch_size) < 1:
        raise ValueError("--mpc-replay-batch-size must be positive")
    if int(args.mpc_replay_updates_per_rollout) < 1:
        raise ValueError("--mpc-replay-updates-per-rollout must be positive")
    if int(args.mpc_replay_min_size) < 1:
        raise ValueError("--mpc-replay-min-size must be positive")
    if getattr(args, "mpc_policy_kl_limit", .003) <= 0 or getattr(args, "mpc_teacher_max_age", 1024) < 1:
        raise ValueError("MPC policy KL limit and teacher max age must be positive")
    if int(args.mpc_kl_ramp_steps) < 0:
        raise ValueError("--mpc-kl-ramp-steps cannot be negative")
    if float(args.mpc_kl_coefficient) <= 0.0:
        raise ValueError(
            "--mpc-kl-coefficient must be positive so MPC transitions are stored"
        )


def train_robot(args):
    validate_args(args)
    # The base online trainer reads ``ppo_epochs`` when configuring PPO and
    # validates that it is at least one.  Keep one as the validation/config
    # sentinel for the replay-first zero setting; ``bind_runner`` replaces the
    # actual PPO epoch count with the requested value before learning starts.
    args.ppo_epochs = max(1, int(args.on_policy_ppo_epochs))
    original_extension = online_mpc.OnlineMPCSelfPlayExtension
    online_mpc.OnlineMPCSelfPlayExtension = ReplayMPCSelfPlayExtension
    try:
        online_mpc.train_robot(args)
    finally:
        online_mpc.OnlineMPCSelfPlayExtension = original_extension


def parse_args():
    return build_arg_parser().parse_args()


if __name__ == "__main__":
    train_robot(parse_args())
    from scripts.train_high_level import successful_training_exit
    successful_training_exit()
