"""Terminal value learning from complete, real simulator trajectories.

The value model deliberately remains independent of the learned dynamics model.
It reuses the dynamics checkpoint's state schema and state normalization, but its
targets are Monte-Carlo returns computed only from recorded simulator rewards.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dribblebot.world_model.normalizer import WorldModelNormalizer
from dribblebot.world_model.schema import StateSchema


def compute_discounted_returns(
    rewards,
    terminated,
    truncated,
    gamma: float,
    *,
    bootstrap_value: float = 0.0,
    bootstrap_on_truncation: bool = False,
):
    """Compute one episode's return-to-go without crossing its boundary.

    A true terminal always has zero continuation value. A truncated final row
    can use an explicitly supplied bootstrap; when bootstrapping is disabled,
    its suffix remains a valid finite-horizon Monte-Carlo target.
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    if rewards.ndim != 1 or terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("rewards, terminated, and truncated must be equally sized 1-D arrays")
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError("gamma must lie in (0, 1]")
    if np.any((terminated | truncated)[:-1]):
        raise ValueError("Terminal flags may only occur on the final episode row")
    continuation = (
        float(bootstrap_value)
        if len(rewards) and truncated[-1] and bootstrap_on_truncation
        else 0.0
    )
    if len(rewards) and terminated[-1]:
        continuation = 0.0
    result = np.empty_like(rewards)
    for index in range(len(rewards) - 1, -1, -1):
        continuation = rewards[index] + float(gamma) * continuation
        result[index] = continuation
    return result.astype(np.float32)


@dataclass
class ValueModelConfig:
    enabled: bool = True
    gamma: float = 0.99
    target_type: str = "monte_carlo"
    bootstrap_on_truncation: bool = True
    exclude_truncated_tail_steps: int = 8
    hidden_dims: Sequence[int] = (512, 512, 256)
    activation: str = "silu"
    layer_norm: bool = True
    dropout: float = 0.0
    loss: str = "huber"
    huber_delta: float = 1.0
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    batch_size: int = 1024
    max_epochs: int = 200
    early_stopping_patience: int = 20
    gradient_clip_norm: float = 10.0
    num_workers: int = 0
    mixed_precision: bool = True
    normalize_targets: bool = True
    ensemble_size: int = 1
    seed: int = 42
    device: str = "cuda"
    split: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ValueModelConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown value_model configuration fields: {unknown}")
        result = cls(**dict(value))
        if result.target_type != "monte_carlo":
            raise ValueError("Only target_type='monte_carlo' is implemented; td_lambda is reserved")
        if result.ensemble_size < 1:
            raise ValueError("value_model.ensemble_size must be positive")
        if not 0.0 < result.gamma <= 1.0:
            raise ValueError("value_model.gamma must lie in (0, 1]")
        if not math.isclose(sum(result.split.values()), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("value_model.split fractions must sum to one")
        return result


class ReturnNormalizer:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = torch.as_tensor(mean).float()
        self.std = torch.as_tensor(std).float().clamp(min=1e-6)

    @classmethod
    def fit(cls, targets: torch.Tensor) -> "ReturnNormalizer":
        return cls(targets.mean(), targets.std(unbiased=False))

    def normalize(self, value):
        return (value - self.mean.to(value.device)) / self.std.to(value.device)

    def denormalize(self, value):
        return value * self.std.to(value.device) + self.mean.to(value.device)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_state_dict(cls, value):
        return cls(value["mean"], value["std"])


def _activation(name: str):
    choices = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}
    if name.lower() not in choices:
        raise ValueError(f"Unsupported value activation {name!r}")
    return choices[name.lower()]


class TerminalValueModel(nn.Module):
    """Centralized scalar V(s), with predictions returned on reward scale."""

    def __init__(
        self,
        schema: StateSchema,
        state_normalizer: WorldModelNormalizer,
        hidden_dims: Sequence[int] = (512, 512, 256),
        activation: str = "silu",
        layer_norm: bool = True,
        dropout: float = 0.0,
        return_normalizer: Optional[ReturnNormalizer] = None,
        return_statistics: Optional[Mapping[str, object]] = None,
    ):
        super().__init__()
        self.schema = schema
        self.state_normalizer = state_normalizer
        self.return_normalizer = return_normalizer or ReturnNormalizer()
        self.return_statistics = dict(return_statistics or {})
        self.config = {
            "hidden_dims": list(hidden_dims), "activation": activation,
            "layer_norm": bool(layer_norm), "dropout": float(dropout),
        }
        layers = []
        width = schema.state_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(width, int(hidden)))
            if layer_norm:
                layers.append(nn.LayerNorm(int(hidden)))
            layers.append(_activation(activation)())
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            width = int(hidden)
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.state_normalizer.to(device)
        self.return_normalizer.mean = self.return_normalizer.mean.to(device)
        self.return_normalizer.std = self.return_normalizer.std.to(device)
        return result

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.schema.state_dim:
            raise ValueError(f"Expected state dimension {self.schema.state_dim}, got {states.shape[-1]}")
        normalized = self.state_normalizer.normalize_state(states)
        return self.network(normalized).squeeze(-1)

    @torch.no_grad()
    def predict(self, states: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        prediction = self.return_normalizer.denormalize(self(states))
        self.train(was_training)
        return prediction


def _load_manifest(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    return list(manifest["episodes"]), json.loads((root / "metadata.json").read_text())


def build_value_dataset(
    source: Union[str, Path],
    output: Union[str, Path],
    config: ValueModelConfig,
) -> Dict[str, object]:
    """Postprocess teacher/world-model episode shards into real value targets."""

    source, output = Path(source), Path(output)
    entries, metadata = _load_manifest(source)
    output.mkdir(parents=True, exist_ok=True)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(config.seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    count = len(shuffled)
    train_end = int(count * config.split["train"])
    validation_end = train_end + int(count * config.split["validation"])
    groups = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    output_entries = []
    return_samples = []
    for entry in entries:
        with np.load(source / entry["path"], allow_pickle=False) as shard:
            arrays = {key: shard[key] for key in shard.files}
        state_key = "global_state" if "global_state" in arrays else "state"
        reward_key = "real_reward" if "real_reward" in arrays else "reward"
        behavior = arrays.get("behavior_source", np.full(len(arrays[reward_key]), "unknown"))
        returns = compute_discounted_returns(
            arrays[reward_key], arrays["terminated"], arrays["truncated"], config.gamma
        )
        valid = np.ones(len(returns), dtype=bool)
        # Without a prior critic there is no reliable truncation bootstrap.
        # Exclude a configurable suffix while retaining earlier finite-horizon targets.
        if config.bootstrap_on_truncation and bool(arrays["truncated"][-1]):
            tail = min(config.exclude_truncated_tail_steps, len(valid))
            valid[len(valid) - tail :] = False
        episode_return = float(returns[0])
        value_arrays = {
            "episode_id": np.asarray(arrays["episode_id"], dtype=np.int64),
            "step_id": np.asarray(arrays["step_id"], dtype=np.int64),
            "global_state": np.asarray(arrays[state_key], dtype=np.float32),
            "return_to_go": returns,
            "terminated": np.asarray(arrays["terminated"], dtype=bool),
            "truncated": np.asarray(arrays["truncated"], dtype=bool),
            "behavior_source": np.asarray(behavior),
            "episode_return": np.full(len(returns), episode_return, dtype=np.float32),
            "remaining_episode_length": np.arange(len(returns), 0, -1, dtype=np.int32),
            "valid_value_target": valid,
        }
        filename = f"episode_{int(entry['episode_id']):08d}.npz"
        with (episodes_dir / filename).open("wb") as handle:
            np.savez_compressed(handle, **value_arrays)
        new_entry = dict(entry)
        new_entry["path"] = f"episodes/{filename}"
        new_entry["valid_value_targets"] = int(valid.sum())
        output_entries.append(new_entry)
        return_samples.extend(returns[valid].tolist())
    by_id = {int(entry["episode_id"]): entry for entry in output_entries}
    for split, selected in groups.items():
        payload = {
            "format": "dribblebot_terminal_value_v1", "split": split,
            "episodes": [by_id[int(entry["episode_id"])] for entry in selected],
        }
        (output / f"{split}_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    manifest = {"format": "dribblebot_terminal_value_v1", "episodes": output_entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    values = np.asarray(return_samples, dtype=np.float64)
    stats = {
        "count": int(len(values)), "mean": float(values.mean()) if len(values) else 0.0,
        "std": float(values.std()) if len(values) else 1.0,
        "percentiles": {
            str(p): float(np.percentile(values, p)) if len(values) else 0.0
            for p in (1, 5, 50, 95, 99)
        },
    }
    value_metadata = {
        "format": "dribblebot_terminal_value_v1", "source_dataset": str(source.resolve()),
        "source_metadata": metadata, "gamma": config.gamma,
        "target_type": config.target_type,
        "truncation_strategy": (
            f"exclude final {config.exclude_truncated_tail_steps} rows of truncated episodes"
            if config.bootstrap_on_truncation else "finite-horizon Monte Carlo with zero cutoff bootstrap"
        ),
        "return_statistics": stats, "config": asdict(config),
        "target_provenance": "real_simulator_rewards_only",
    }
    (output / "metadata.json").write_text(json.dumps(value_metadata, indent=2, sort_keys=True))
    return value_metadata


class ValueDataset(Dataset):
    def __init__(self, root: Union[str, Path], split: str):
        self.root = Path(root)
        manifest = json.loads((self.root / f"{split}_manifest.json").read_text())
        self.samples = []
        self.episode_ids = []
        for entry in manifest["episodes"]:
            with np.load(self.root / entry["path"], allow_pickle=False) as shard:
                arrays = {key: shard[key] for key in shard.files}
            valid = arrays["valid_value_target"].astype(bool)
            for step in np.flatnonzero(valid):
                self.samples.append((arrays["global_state"][step], arrays["return_to_go"][step]))
                self.episode_ids.append(int(entry["episode_id"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        state, target = self.samples[index]
        return {"global_state": torch.from_numpy(np.asarray(state)).float(), "return_to_go": torch.tensor(target).float()}

    def targets(self):
        return torch.tensor([sample[1] for sample in self.samples]).float()


class CombinedValueDataset(Dataset):
    """Replay view over old and new episode-preserving value datasets."""

    def __init__(self, datasets: Sequence[ValueDataset]):
        if not datasets:
            raise ValueError("At least one value dataset is required")
        self.datasets = list(datasets)
        self.offsets = np.cumsum([0] + [len(dataset) for dataset in datasets])
        self.root = Path("combined_replay")

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        source = int(np.searchsorted(self.offsets[1:], index, side="right"))
        return self.datasets[source][index - int(self.offsets[source])]

    def targets(self):
        return torch.cat([dataset.targets() for dataset in self.datasets])


def save_value_checkpoint(path, model, optimizer, epoch, config, metrics, dataset_manifest):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    payload = {
        "format": "dribblebot_terminal_value_v1", "model_state": model.state_dict(),
        "model_config": model.config, "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch), "config": asdict(config), "validation_metrics": dict(metrics),
        "state_schema": model.schema.to_dict(),
        "state_normalizer": model.state_normalizer.state_dict(),
        "return_normalizer": model.return_normalizer.state_dict(),
        "gamma": float(config.gamma), "dataset_manifest": str(dataset_manifest),
        "return_statistics": dict(getattr(model, "return_statistics", {})),
        "repository_commit": commit,
    }
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def load_value_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    if payload.get("format") != "dribblebot_terminal_value_v1":
        raise ValueError(f"Unsupported terminal value checkpoint: {path}")
    model = TerminalValueModel(
        StateSchema.from_dict(payload["state_schema"]),
        WorldModelNormalizer.from_state_dict(payload["state_normalizer"]),
        return_normalizer=ReturnNormalizer.from_state_dict(payload["return_normalizer"]),
        **payload["model_config"],
    )
    model.load_state_dict(payload["model_state"]); model.to(device); model.eval()
    model.return_statistics = dict(payload.get("return_statistics", {}))
    return model, payload


def value_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    error = prediction - target
    mse = float(error.square().mean()) if error.numel() else float("nan")
    mae = float(error.abs().mean()) if error.numel() else float("nan")
    variance = float(target.var(unbiased=False)) if target.numel() else 0.0
    explained = 1.0 - float(error.var(unbiased=False)) / max(variance, 1e-12)
    p = prediction.detach().cpu().numpy(); t = target.detach().cpu().numpy()
    pearson = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 and np.std(p) and np.std(t) else float("nan")
    pr = np.argsort(np.argsort(p)); tr = np.argsort(np.argsort(t))
    spearman = float(np.corrcoef(pr, tr)[0, 1]) if len(p) > 1 else float("nan")
    huber = float(torch.nn.functional.huber_loss(prediction, target)) if error.numel() else float("nan")
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae, "huber": huber,
            "explained_variance": explained, "pearson": pearson, "spearman": spearman}


class TerminalValueTrainer:
    def __init__(self, model, train, validation, config: ValueModelConfig):
        self.model, self.train, self.validation, self.config = model, train, validation, config
        requested = config.device
        self.device = torch.device(requested if requested != "cuda" or torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and self.device.type == "cuda")

    def _loss(self, prediction, target):
        if self.config.loss == "huber":
            return torch.nn.functional.huber_loss(prediction, target, delta=self.config.huber_delta)
        if self.config.loss == "mse":
            return torch.nn.functional.mse_loss(prediction, target)
        raise ValueError(f"Unknown value loss {self.config.loss!r}")

    def _run(self, dataset, training):
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=training, num_workers=self.config.num_workers)
        predictions, targets, total = [], [], 0.0
        self.model.train(training)
        for batch in loader:
            states = batch["global_state"].to(self.device); raw = batch["return_to_go"].to(self.device)
            target = self.model.return_normalizer.normalize(raw) if self.config.normalize_targets else raw
            if training: self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                prediction = self.model(states); loss = self._loss(prediction, target)
            if training:
                self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer); self.scaler.update()
            total += float(loss) * len(raw)
            predictions.append(self.model.return_normalizer.denormalize(prediction.detach()) if self.config.normalize_targets else prediction.detach())
            targets.append(raw.detach())
        metrics = value_metrics(torch.cat(predictions), torch.cat(targets))
        metrics["loss"] = total / max(len(dataset), 1)
        return metrics

    def fit(self, output, resume=None):
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        start = 0
        if resume:
            payload = torch.load(resume, map_location=self.device)
            self.model.load_state_dict(payload["model_state"]); self.optimizer.load_state_dict(payload["optimizer_state"])
            start = int(payload["epoch"]) + 1
        best, patience, history = float("inf"), 0, {"train": [], "validation": []}
        for epoch in range(start, self.config.max_epochs):
            train_metrics = self._run(self.train, True)
            with torch.no_grad(): validation_metrics = self._run(self.validation, False)
            history["train"].append(train_metrics); history["validation"].append(validation_metrics)
            save_value_checkpoint(output / "latest.pt", self.model, self.optimizer, epoch, self.config, validation_metrics, self.train.root / "train_manifest.json")
            if validation_metrics["loss"] < best:
                best, patience = validation_metrics["loss"], 0
                save_value_checkpoint(output / "best.pt", self.model, self.optimizer, epoch, self.config, validation_metrics, self.train.root / "train_manifest.json")
            else: patience += 1
            (output / "history.json").write_text(json.dumps(history, indent=2))
            if patience >= self.config.early_stopping_patience: break
        return history
