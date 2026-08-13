"""Source-aware replay mixing for iterative world-model fine-tuning."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class MixedWorldModelDataset(Dataset):
    """In-memory episode dataset composed from immutable source datasets."""

    def __init__(self, sources: Sequence[Tuple[str, object]]):
        if not sources:
            raise ValueError("MixedWorldModelDataset requires at least one source")
        self.episodes = []
        self.index = []
        self.transition_sources = []
        self.episode_sources = []
        for source_name, dataset in sources:
            for episode in dataset.episodes:
                episode_index = len(self.episodes)
                self.episodes.append(episode)
                self.episode_sources.append(str(source_name))
                for step in range(len(episode["step_id"])):
                    self.index.append((episode_index, step))
                    self.transition_sources.append(str(source_name))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        episode_index, step = self.index[item]
        episode = self.episodes[episode_index]
        result = {}
        for key, value in episode.items():
            selected = value[step]
            if value.dtype.kind in "USO":
                result[key] = str(selected)
            else:
                result[key] = torch.from_numpy(np.asarray(selected))
        return result

    def all(self, key):
        return torch.from_numpy(
            np.concatenate([episode[key] for episode in self.episodes], axis=0)
        )

    def sequences(self, length):
        result = []
        for episode_index, episode in enumerate(self.episodes):
            terminal = np.asarray(episode["terminated"]) | np.asarray(
                episode["truncated"]
            )
            for start in range(max(0, len(terminal) - length + 1)):
                if terminal[start : start + length - 1].any():
                    continue
                result.append((episode_index, start))
        return result

    def get_sequence(self, episode_index, start, length):
        episode = self.episodes[episode_index]
        return {
            key: torch.from_numpy(value[start : start + length])
            for key, value in episode.items()
            if value.dtype.kind not in "USO"
        }


class ReplayMixSampler(Sampler[int]):
    """Sample configurable initial/previous/new/rare replay proportions."""

    def __init__(
        self,
        dataset: MixedWorldModelDataset,
        weights: Mapping[str, float],
        seed: int = 42,
    ):
        self.dataset = dataset
        self.generator = torch.Generator().manual_seed(int(seed))
        groups: Dict[str, list] = {
            "initial_random": [],
            "initial_scripted": [],
            "previous_mpc": [],
            "newest_mpc": [],
            "rare_events": [],
        }
        for index, ((episode_index, step), source) in enumerate(
            zip(dataset.index, dataset.transition_sources)
        ):
            episode = dataset.episodes[episode_index]
            if source == "initial":
                behavior = str(episode["behavior_source"][step])
                group = (
                    "initial_random"
                    if behavior in ("random_valid", "repeat_previous")
                    else "initial_scripted"
                )
                groups[group].append(index)
            elif source == "newest_mpc":
                groups["newest_mpc"].append(index)
            else:
                groups["previous_mpc"].append(index)
            if bool(np.asarray(episode["event_labels"][step]).any()):
                groups["rare_events"].append(index)
        unknown = sorted(set(weights) - set(groups))
        if unknown:
            raise ValueError(f"Unknown replay sampling groups: {unknown}")
        available = {
            name: torch.as_tensor(indices, dtype=torch.long)
            for name, indices in groups.items()
            if indices and float(weights.get(name, 0.0)) > 0.0
        }
        if not available:
            raise ValueError("No configured replay sampling group has data")
        raw_weights = torch.tensor(
            [float(weights[name]) for name in available], dtype=torch.float
        )
        if bool((raw_weights < 0).any()):
            raise ValueError("Replay sampling weights cannot be negative")
        self.names = tuple(available)
        self.groups = available
        self.probabilities = raw_weights / raw_weights.sum()

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        group_choices = torch.multinomial(
            self.probabilities,
            len(self.dataset),
            replacement=True,
            generator=self.generator,
        )
        result = torch.empty(len(self.dataset), dtype=torch.long)
        for group_index, name in enumerate(self.names):
            locations = torch.nonzero(
                group_choices == group_index, as_tuple=False
            ).flatten()
            if not locations.numel():
                continue
            candidates = self.groups[name]
            sampled = torch.randint(
                0,
                len(candidates),
                (len(locations),),
                generator=self.generator,
            )
            result[locations] = candidates[sampled]
        return iter(result.tolist())
