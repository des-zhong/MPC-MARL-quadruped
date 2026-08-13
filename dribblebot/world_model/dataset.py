"""Episode-preserving compressed NumPy world-model datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


REQUIRED_KEYS = (
    "episode_id", "step_id", "state", "joint_action", "reward", "next_state",
    "terminated", "truncated", "elapsed_low_level_steps", "event_labels", "behavior_source",
)


@dataclass
class Episode:
    episode_id: int
    arrays: Dict[str, np.ndarray]

    def validate(self) -> None:
        missing = [key for key in REQUIRED_KEYS if key not in self.arrays]
        if missing:
            raise ValueError(f"Episode {self.episode_id} is missing fields {missing}")
        length = len(self.arrays["step_id"])
        if length == 0:
            raise ValueError(f"Episode {self.episode_id} is empty")
        for key, value in self.arrays.items():
            if len(value) != length:
                raise ValueError(f"Episode {self.episode_id} field {key} has length {len(value)}, expected {length}")
        if not np.array_equal(self.arrays["step_id"], np.arange(length)):
            raise ValueError(f"Episode {self.episode_id} step IDs are not contiguous from zero")
        for key in ("state", "joint_action", "reward", "next_state"):
            if not np.isfinite(self.arrays[key]).all():
                raise ValueError(f"Episode {self.episode_id} field {key} contains NaN/Inf")


class EpisodeShardWriter:
    """Write each complete episode atomically as one compressed shard."""

    def __init__(self, output: Union[str, Path], metadata: Mapping[str, object]):
        self.output = Path(output)
        self.episodes_dir = self.output / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        (self.output / "metadata.json").write_text(json.dumps(self.metadata, indent=2, sort_keys=True))
        self.entries: List[Dict[str, object]] = []

    def write_episode(self, episode: Episode) -> Path:
        episode.validate()
        filename = f"episode_{episode.episode_id:08d}.npz"
        path = self.episodes_dir / filename
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **episode.arrays)
        temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.entries.append({"episode_id": episode.episode_id, "path": f"episodes/{filename}", "length": len(episode.arrays["step_id"]), "sha256": digest})
        self._write_manifest()
        return path

    def _write_manifest(self) -> None:
        payload = {"format": "dribblebot_world_model_v1", "episodes": sorted(self.entries, key=lambda item: item["episode_id"])}
        (self.output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_manifest(root: Union[str, Path], split: Optional[str] = None) -> Dict[str, object]:
    root = Path(root)
    path = root / (f"{split}_manifest.json" if split else "manifest.json")
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    return json.loads(path.read_text())


def split_episodes(
    root: Union[str, Path],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Create deterministic, leakage-free split manifests by whole episode."""

    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("Dataset split fractions must sum to one")
    root = Path(root)
    manifest = load_manifest(root)
    entries = list(manifest["episodes"])
    rng = np.random.default_rng(seed)
    rng.shuffle(entries)
    count = len(entries)
    train_end = int(count * train_fraction)
    validation_end = train_end + int(count * validation_fraction)
    groups = {"train": entries[:train_end], "validation": entries[train_end:validation_end], "test": entries[validation_end:]}
    result = {}
    for name, selected in groups.items():
        payload = {"format": manifest["format"], "split": name, "seed": seed, "episodes": selected}
        (root / f"{name}_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        result[name] = [int(item["episode_id"]) for item in selected]
    assert_no_episode_leakage(result)
    return result


def assert_no_episode_leakage(splits: Mapping[str, Sequence[int]]) -> None:
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise ValueError(f"Episode leakage between {left} and {right}: {sorted(overlap)}")


class WorldModelDataset(Dataset):
    """In-memory transition dataset loaded from episode shards."""

    def __init__(self, root: Union[str, Path], split: str = "train"):
        self.root = Path(root)
        self.split = split
        manifest = load_manifest(self.root, split)
        self.episodes: List[Dict[str, np.ndarray]] = []
        self.index: List[tuple[int, int]] = []
        for episode_index, entry in enumerate(manifest["episodes"]):
            with np.load(self.root / entry["path"], allow_pickle=False) as shard:
                arrays = {key: shard[key] for key in shard.files}
            Episode(int(entry["episode_id"]), arrays).validate()
            self.episodes.append(arrays)
            self.index.extend((episode_index, step) for step in range(len(arrays["step_id"])))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> Dict[str, Union[torch.Tensor, str]]:
        episode_index, step = self.index[item]
        episode = self.episodes[episode_index]
        result: Dict[str, Union[torch.Tensor, str]] = {}
        for key, value in episode.items():
            selected = value[step]
            if value.dtype.kind in "USO":
                result[key] = str(selected)
            else:
                result[key] = torch.from_numpy(np.asarray(selected))
        return result

    def all(self, key: str) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([episode[key] for episode in self.episodes], axis=0))

    def sequences(self, length: int) -> List[tuple[int, int]]:
        result = []
        for episode_index, episode in enumerate(self.episodes):
            terminal = np.asarray(episode["terminated"]) | np.asarray(episode["truncated"])
            for start in range(max(0, len(terminal) - length + 1)):
                if terminal[start : start + length - 1].any():
                    continue
                result.append((episode_index, start))
        return result

    def get_sequence(self, episode_index: int, start: int, length: int) -> Dict[str, torch.Tensor]:
        episode = self.episodes[episode_index]
        return {
            key: torch.from_numpy(value[start : start + length])
            for key, value in episode.items()
            if value.dtype.kind not in "USO"
        }


class EventAwareSampler(Sampler[int]):
    """Mix uniform transitions with event-type-balanced rare transitions."""

    def __init__(self, dataset: WorldModelDataset, rare_fraction: float = 0.25, seed: int = 42):
        self.dataset = dataset
        self.rare_fraction = float(rare_fraction)
        self.generator = torch.Generator().manual_seed(seed)
        events = dataset.all("event_labels").bool()
        self.rare_by_event = [
            torch.nonzero(events[:, event_index], as_tuple=False).flatten()
            for event_index in range(events.shape[1])
            if bool(events[:, event_index].any().item())
        ]
        self.rare = torch.nonzero(events.any(dim=-1), as_tuple=False).flatten()

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        count = len(self.dataset)
        indices = torch.randint(0, count, (count,), generator=self.generator)
        rare_count = min(count, int(round(count * self.rare_fraction))) if self.rare_by_event else 0
        if rare_count:
            # Choose the event type first, then a transition carrying that
            # event.  This prevents frequent labels such as failed shots from
            # drowning out goals, passes, and boundary/collision outcomes.
            event_choices = torch.randint(
                0,
                len(self.rare_by_event),
                (rare_count,),
                generator=self.generator,
            )
            choices = torch.empty(rare_count, dtype=torch.long)
            for event_index, candidates in enumerate(self.rare_by_event):
                selected = torch.nonzero(event_choices == event_index, as_tuple=False).flatten()
                if selected.numel():
                    choices[selected] = candidates[
                        torch.randint(0, len(candidates), (len(selected),), generator=self.generator)
                    ]
            indices[:rare_count] = choices
            indices = indices[torch.randperm(count, generator=self.generator)]
        return iter(indices.tolist())
