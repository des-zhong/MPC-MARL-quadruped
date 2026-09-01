"""Train the high-level policy by replaying MPC-labelled transitions.

This is a replay-first companion to ``train_high_level_online_mpc.py``.  It
keeps the existing Isaac Gym/self-play/world-model/MPC setup, but stores every
MPC distillation example in a bounded replay buffer and samples that buffer
repeatedly after each rollout.  The default sets PPO policy epochs to zero, so
the high-level actor is trained primarily from replay rather than from the
short-lived on-policy rollout buffer.

This is deliberately a policy-distillation experiment, not a drop-in SAC
implementation: the coordinator has a categorical skill choice plus
skill-conditional continuous commands, while the existing RLPD reference code
only supports a continuous tanh-Gaussian actor.
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
        if checkpoint is None and bool(getattr(self.args, "resume", False)):
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
        self.teacher_replay.load_state_dict(
            payload.get("replay", {"items": payload.get("items", [])})
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
        self.teacher_replay.add_batch(merged)
        return int(merged["obs_history"].shape[0])

    def process_env_step(self, *args, **kwargs):
        super().process_env_step(*args, **kwargs)
        self._store_current_rollout_targets()

    @staticmethod
    def _finite_parameters(module):
        return all(bool(torch.isfinite(parameter).all()) for parameter in module.parameters())

    def _replay_update(self):
        coefficient = float(self.args.mpc_kl_coefficient)
        updates_requested = int(self.args.mpc_replay_updates_per_rollout)
        batch_size = int(self.args.mpc_replay_batch_size)
        minimum_size = int(self.args.mpc_replay_min_size)
        if coefficient <= 0.0:
            return {
                "mpc_replay/enabled": 0.0,
                "mpc_replay/size": float(len(self.teacher_replay)),
            }
        if updates_requested < 1:
            raise ValueError("--mpc-replay-updates-per-rollout must be positive")
        if batch_size < 1:
            raise ValueError("--mpc-replay-batch-size must be positive")
        if minimum_size < 1:
            raise ValueError("--mpc-replay-min-size must be positive")
        if not self._planner_ready():
            return {
                "mpc_replay/enabled": 1.0,
                "mpc_replay/update_skipped_not_ready": 1.0,
                "mpc_replay/size": float(len(self.teacher_replay)),
            }
        if len(self.teacher_replay) < max(minimum_size, batch_size):
            return {
                "mpc_replay/enabled": 1.0,
                "mpc_replay/update_skipped_small_buffer": 1.0,
                "mpc_replay/size": float(len(self.teacher_replay)),
            }

        policy = self.runner.alg.actor_critic
        device = self.runner.device
        total_loss = 0.0
        total_categorical = 0.0
        total_continuous = 0.0
        performed = 0
        for _ in range(updates_requested):
            batch = self.teacher_replay.sample(batch_size, device=device)
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
            conditional_kl = (command_kl * target_masks).sum(dim=-1) / target_masks.sum(
                dim=-1
            ).clamp(min=1.0)
            continuous = (target_probs * conditional_kl).sum(dim=-1).mean()
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
            self._distill_optimizer.step()
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

        self._replay_updates += performed
        self._replay_samples += performed * batch_size
        return {
            "mpc_replay/enabled": 1.0,
            "mpc_replay/size": float(len(self.teacher_replay)),
            "mpc_replay/updates": float(performed),
            "mpc_replay/updates_total": float(self._replay_updates),
            "mpc_replay/samples_total": float(self._replay_samples),
            "mpc_replay/kl_loss": total_loss / max(performed, 1),
            "mpc_replay/categorical_kl": total_categorical / max(performed, 1),
            "mpc_replay/continuous_kl": total_continuous / max(performed, 1),
            "mpc_replay/reuse_ratio": float(
                self._replay_samples / max(len(self.teacher_replay), 1)
            ),
        }

    def after_policy_update(self, iteration):
        # Do not call the base implementation: it would consume a one-shot
        # rollout cache.  World-model/value metrics are produced in
        # ``after_rollout`` and are forwarded here unchanged.
        metrics = dict(self._metrics)
        self._metrics = {}
        metrics.update(self._replay_update())
        return metrics

    def save_checkpoint(self, path, iteration):
        saved = list(super().save_checkpoint(path, iteration))
        if not bool(getattr(self.args, "save_mpc_replay", False)):
            return saved
        output = Path(path)
        payload = {
            "format": "dribblebot_mpc_teacher_replay_v1",
            "capacity": self.teacher_replay.capacity,
            "recent_fraction": self.teacher_replay.recent_fraction,
            "recent_window": self.teacher_replay.recent_window,
            "replay": self.teacher_replay.state_dict(),
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
    parser.description = (
        "Train the high-level policy by replaying MPC distillation transitions."
    )
    parser.add_argument(
        "--on-policy-ppo-epochs",
        type=int,
        default=0,
        help=(
            "PPO policy epochs per rollout. The default zero makes replayed "
            "MPC updates the primary policy-learning signal."
        ),
    )
    parser.add_argument("--mpc-replay-capacity", type=int, default=100_000)
    parser.add_argument("--mpc-replay-recent-fraction", type=float, default=0.5)
    parser.add_argument("--mpc-replay-recent-window", type=int, default=10_000)
    parser.add_argument("--mpc-replay-batch-size", type=int, default=1024)
    parser.add_argument(
        "--mpc-replay-updates-per-rollout",
        type=int,
        default=20,
        help="Replay gradient updates after each environment rollout.",
    )
    parser.add_argument(
        "--mpc-replay-min-size",
        type=int,
        default=1024,
        help="Minimum replay size before updates begin.",
    )
    parser.add_argument(
        "--mpc-replay-checkpoint",
        default=None,
        help="Optional replay checkpoint to load before training.",
    )
    parser.add_argument(
        "--save-mpc-replay",
        action="store_true",
        help="Save the bounded MPC replay beside policy/model checkpoints.",
    )
    parser.set_defaults(
        project="as2_high_level_mpc_replay",
        checkpoint_subdir="high_level_mpc_replay",
        mpc_kl_coefficient=0.05,
        mpc_guidance_reward_coefficient=0.0,
    )
    return parser


def validate_args(args):
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
