"""Bootstrapped ensemble trainer and self-contained checkpoints."""

from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from .action_adapter import JointActionAdapter
from .dataset import EventAwareSampler, WorldModelDataset
from .ensemble import WorldModelEnsemble
from .losses import feature_group_weights, multi_step_member_loss, one_step_member_loss
from .normalizer import WorldModelNormalizer
from .schema import EVENT_NAMES, StateSchema, validate_event_names


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def repository_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def periodic_checkpoint_due(epoch: int, interval: int) -> bool:
    """Return whether this zero-based epoch completes a rolling-save interval."""

    interval = int(interval)
    return interval > 0 and (int(epoch) + 1) % interval == 0


# Backward-compatible import for callers/tests written before rolling.pt was
# separated from the true validation-best checkpoint.
periodic_best_checkpoint_due = periodic_checkpoint_due


def _to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def fit_normalizer(dataset: WorldModelDataset, schema: StateSchema, normalize_reward: bool = False) -> WorldModelNormalizer:
    states = dataset.all("state").float()
    next_states = dataset.all("next_state").float()
    dynamic = schema.continuous_dynamic_indices
    deltas = next_states[:, dynamic] - states[:, dynamic]
    continuous_state = schema.indices(kinds=("continuous", "angle_pair"))
    return WorldModelNormalizer.fit(states, deltas, dataset.all("reward").float(), continuous_state, normalize_reward)


def save_checkpoint(
    path: Union[str, Path],
    model: WorldModelEnsemble,
    optimizers,
    schedulers,
    epoch: int,
    validation_metrics: Mapping[str, float],
    training_config: Mapping[str, object],
    seed: int,
) -> None:
    payload = {
        "format": "dribblebot_world_model_v1",
        "model_state": model.state_dict(),
        "model_config": model.config,
        "optimizer_states": [optimizer.state_dict() for optimizer in optimizers],
        "scheduler_states": [scheduler.state_dict() for scheduler in schedulers],
        "normalizer": model.normalizer.state_dict(),
        "state_schema": model.schema.to_dict(),
        "action_schema": model.action_adapter.to_dict(),
        "skill_parameter_bounds": model.action_adapter.to_dict()["bounds"],
        "cylinder_schema": [f.__dict__ for f in model.schema.features if f.group == "obstacle"],
        "event_names": list(getattr(model, "event_names", EVENT_NAMES[: model.num_events])),
        "training_config": dict(training_config),
        "epoch": int(epoch),
        "validation_metrics": dict(validation_metrics),
        "random_seed": int(seed),
        "repository_commit": repository_commit(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Union[str, Path], device: Union[str, torch.device] = "cpu"):
    payload = torch.load(path, map_location=device)
    if payload.get("format") != "dribblebot_world_model_v1":
        raise ValueError(f"Unsupported checkpoint format in {path}")
    schema = StateSchema.from_dict(payload["state_schema"])
    action_adapter = JointActionAdapter.from_dict(payload["action_schema"])
    normalizer = WorldModelNormalizer.from_state_dict(payload["normalizer"])
    model = WorldModelEnsemble(schema, action_adapter, normalizer, **payload["model_config"])
    model.load_state_dict(payload["model_state"])
    stored_event_names = payload.get("event_names", EVENT_NAMES[: model.num_events])
    model.event_names = validate_event_names(stored_event_names)
    if len(model.event_names) != model.num_events:
        raise ValueError(
            f"Checkpoint has {model.num_events} event outputs but {len(model.event_names)} event names"
        )
    model.to(device)
    return model, payload


class WorldModelTrainer:
    def __init__(
        self,
        model: WorldModelEnsemble,
        train: WorldModelDataset,
        validation: WorldModelDataset,
        config: Mapping[str, object],
        train_sampler=None,
    ):
        self.model = model
        self.train_dataset = train
        self.validation_dataset = validation
        self.config = config
        training = config["training"]
        requested = str(training.get("device", "cuda"))
        self.device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizers = [
            torch.optim.AdamW(member.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
            for member in self.model.members
        ]
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5) for opt in self.optimizers]
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(training.get("mixed_precision", True)) and self.device.type == "cuda")
        self.feature_weights = feature_group_weights(model.schema, config.get("loss", {}), self.device)
        sampler = train_sampler or EventAwareSampler(
            train,
            float(config.get("dataset", {}).get("rare_event_fraction", 0.25)),
            int(config.get("seed", 42)),
        )
        self.train_loader = DataLoader(train, batch_size=int(training["batch_size"]), sampler=sampler, num_workers=int(training.get("num_workers", 0)), drop_last=False)
        self.validation_loader = DataLoader(validation, batch_size=int(training["batch_size"]), shuffle=False, num_workers=int(training.get("num_workers", 0)))

    def _loss(self, member_index: int, batch) -> Dict[str, torch.Tensor]:
        loss_cfg = self.config.get("loss", {})
        return one_step_member_loss(
            self.model, member_index, batch, self.feature_weights,
            float(loss_cfg.get("reward_weight", 1.0)),
            float(loss_cfg.get("termination_weight", 1.0)),
            float(loss_cfg.get("event_weight", 1.0)),
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        count = 0
        clip = float(self.config["training"].get("gradient_clip_norm", 10.0))
        for batch in self.train_loader:
            batch = _to_device(batch, self.device)
            batch_size = batch["state"].shape[0]
            for member_index, optimizer in enumerate(self.optimizers):
                # A bootstrap resample within every shuffled minibatch gives each member independent data.
                indices = torch.randint(0, batch_size, (batch_size,), device=self.device)
                boot = {key: value[indices] if torch.is_tensor(value) and value.shape[0] == batch_size else value for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    losses = self._loss(member_index, boot)
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.members[member_index].parameters(), clip
                )
                self.scaler.step(optimizer)
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach())
                totals["gradient_norm"] = totals.get("gradient_norm", 0.0) + float(gradient_norm)
                count += 1
            self.scaler.update()
        metrics = {key: value / max(count, 1) for key, value in totals.items()}
        training = self.config["training"]
        multi_weight = float(training.get("multi_step_loss_weight", 0.0))
        if multi_weight > 0 and epoch >= int(training.get("multi_step_start_epoch", 10)):
            sequence_length = int(training.get("sequence_length", 5))
            locations = self.train_dataset.sequences(sequence_length)
            random.shuffle(locations)
            batch_size = int(training.get("sequence_batch_size", 256))
            maximum = min(len(locations), max(len(self.train_loader), 1) * batch_size)
            progress = epoch / max(int(training.get("max_epochs", 1)) - 1, 1)
            teacher_probability = float(training.get("teacher_forcing_probability_start", 1.0)) + progress * (
                float(training.get("teacher_forcing_probability_end", 0.0)) - float(training.get("teacher_forcing_probability_start", 1.0))
            )
            multi_total = 0.0
            multi_count = 0
            for start in range(0, maximum, batch_size):
                selected = locations[start : start + batch_size]
                if not selected:
                    continue
                samples = [self.train_dataset.get_sequence(ep, offset, sequence_length) for ep, offset in selected]
                keys = samples[0].keys()
                sequence = {key: torch.stack([sample[key] for sample in samples]).to(self.device) for key in keys}
                for member_index, optimizer in enumerate(self.optimizers):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                        loss = multi_weight * multi_step_member_loss(
                            self.model, member_index, sequence, teacher_probability,
                            float(training.get("multi_step_discount", 0.8)),
                        )
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.members[member_index].parameters(), clip)
                    self.scaler.step(optimizer)
                    multi_total += float(loss.detach())
                    multi_count += 1
                self.scaler.update()
            metrics["multi_step_loss"] = multi_total / max(multi_count, 1)
            metrics["teacher_forcing_probability"] = teacher_probability
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        count = 0
        member_losses = [0.0] * len(self.model.members)
        for batch in self.validation_loader:
            batch = _to_device(batch, self.device)
            for member_index in range(len(self.model.members)):
                losses = self._loss(member_index, batch)
                member_losses[member_index] += float(losses["loss"])
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                count += 1
        batches = max(count // max(len(self.model.members), 1), 1)
        member_losses = [value / batches for value in member_losses]
        metrics = {key: value / max(count, 1) for key, value in totals.items()}
        metrics.update({f"member_{i}_loss": value for i, value in enumerate(member_losses)})
        return metrics

    def fit(
        self,
        output_dir: Union[str, Path],
        resume: Optional[Union[str, Path]] = None,
        epoch_callback: Optional[
            Callable[[int, Mapping[str, float], Mapping[str, float], Mapping[str, object]], None]
        ] = None,
    ) -> Dict[str, list]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        training = self.config["training"]
        start_epoch = 0
        best = float("inf")
        if resume:
            payload = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(payload["model_state"])
            for optimizer, state in zip(self.optimizers, payload["optimizer_states"]): optimizer.load_state_dict(state)
            for scheduler, state in zip(self.schedulers, payload["scheduler_states"]): scheduler.load_state_dict(state)
            start_epoch = int(payload["epoch"]) + 1
            best = float(
                payload.get("validation_metrics", {}).get("loss", float("inf"))
            )
            # A resumed latest/rolling checkpoint may be worse than the best
            # validation checkpoint from the same run. Preserve that lower
            # threshold instead of treating the first resumed epoch as best.
            candidate_best_paths = {
                output / "best.pt",
                Path(resume).expanduser().resolve().parent / "best.pt",
            }
            for candidate in candidate_best_paths:
                if not candidate.exists():
                    continue
                candidate_payload = torch.load(candidate, map_location="cpu")
                candidate_loss = candidate_payload.get(
                    "validation_metrics", {}
                ).get("loss")
                if candidate_loss is not None:
                    best = min(best, float(candidate_loss))
        patience = 0
        rolling_checkpoint_interval = int(
            training.get(
                "rolling_checkpoint_interval",
                training.get("best_checkpoint_interval", 10),
            )
        )
        if rolling_checkpoint_interval < 0:
            raise ValueError(
                "training.rolling_checkpoint_interval must be non-negative"
            )
        history = {"train": [], "validation": []}
        for epoch in range(start_epoch, int(training["max_epochs"])):
            epoch_started_at = time.perf_counter()
            train_metrics = self.train_epoch(epoch)
            validation_metrics = self.validate()
            for scheduler in self.schedulers: scheduler.step(validation_metrics["loss"])
            history["train"].append(train_metrics)
            history["validation"].append(validation_metrics)
            save_checkpoint(output / "latest.pt", self.model, self.optimizers, self.schedulers, epoch, validation_metrics, self.config, int(self.config.get("seed", 42)))
            validation_improved = validation_metrics["loss"] < best
            if validation_improved:
                best = validation_metrics["loss"]
                patience = 0
            else:
                patience += 1
            periodic_checkpoint_save = periodic_checkpoint_due(
                epoch, rolling_checkpoint_interval
            )
            if validation_improved:
                save_checkpoint(output / "best.pt", self.model, self.optimizers, self.schedulers, epoch, validation_metrics, self.config, int(self.config.get("seed", 42)))
            if periodic_checkpoint_save:
                save_checkpoint(output / "rolling.pt", self.model, self.optimizers, self.schedulers, epoch, validation_metrics, self.config, int(self.config.get("seed", 42)))
            (output / "history.json").write_text(json.dumps(history, indent=2))
            if epoch_callback is not None:
                learning_rates = [float(optimizer.param_groups[0]["lr"]) for optimizer in self.optimizers]
                epoch_callback(
                    epoch,
                    train_metrics,
                    validation_metrics,
                    {
                        "best_validation_loss": best,
                        "early_stopping_patience": patience,
                        "epoch_seconds": time.perf_counter() - epoch_started_at,
                        "is_best": validation_improved,
                        "periodic_checkpoint_save": periodic_checkpoint_save,
                        "learning_rate": float(np.mean(learning_rates)),
                    },
                )
            print(f"epoch={epoch} train={train_metrics['loss']:.6f} validation={validation_metrics['loss']:.6f}")
            if patience >= int(training.get("early_stopping_patience", 20)):
                break
        final_epoch = start_epoch + len(history["train"]) - 1
        save_checkpoint(output / "final.pt", self.model, self.optimizers, self.schedulers, final_epoch, history["validation"][-1], self.config, int(self.config.get("seed", 42)))
        return history
