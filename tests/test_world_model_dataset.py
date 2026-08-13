import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dribblebot.world_model.dataset import (
    Episode,
    EpisodeShardWriter,
    EventAwareSampler,
    WorldModelDataset,
    assert_no_episode_leakage,
    split_episodes,
)
from dribblebot.world_model.schema import EVENT_NAMES


def make_episode(identifier, length=3, state_dim=5):
    arrays = {
        "episode_id": np.full(length, identifier, np.int64), "step_id": np.arange(length, dtype=np.int64),
        "state": np.zeros((length, state_dim), np.float32), "joint_action": np.zeros((length, 8), np.float32),
        "reward": np.zeros(length, np.float32), "next_state": np.ones((length, state_dim), np.float32),
        "terminated": np.array([False] * (length - 1) + [True]), "truncated": np.zeros(length, bool),
        "elapsed_low_level_steps": np.full(length, 10, np.int16),
        "event_labels": np.zeros((length, len(EVENT_NAMES)), np.float32),
        "behavior_source": np.asarray(["random_valid"] * length),
    }
    return Episode(identifier, arrays)


def test_episode_splitting_has_no_leakage(tmp_path):
    writer = EpisodeShardWriter(tmp_path, {"test": True})
    for index in range(20): writer.write_episode(make_episode(index))
    splits = split_episodes(tmp_path, 0.8, 0.1, 0.1, seed=7)
    assert_no_episode_leakage(splits)
    assert sum(map(len, splits.values())) == 20


def test_dataset_preserves_sequences(tmp_path):
    writer = EpisodeShardWriter(tmp_path, {})
    for index in range(10): writer.write_episode(make_episode(index, 5))
    split_episodes(tmp_path)
    dataset = WorldModelDataset(tmp_path, "train")
    assert len(dataset) == 40
    assert dataset.sequences(3)


def test_invalid_episode_rejected():
    episode = make_episode(1)
    episode.arrays["step_id"] = np.array([0, 2, 3])
    with pytest.raises(ValueError, match="not contiguous"):
        episode.validate()


def test_explicit_leakage_rejected():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_episode_leakage({"train": [1, 2], "test": [2, 3]})


def test_event_sampler_balances_event_types_not_raw_event_frequency():
    class FakeDataset:
        def __init__(self):
            self.events = torch.zeros(100, 2)
            self.events[:99, 0] = 1.0
            self.events[99, 1] = 1.0

        def __len__(self):
            return len(self.events)

        def all(self, key):
            assert key == "event_labels"
            return self.events

    sampled = list(EventAwareSampler(FakeDataset(), rare_fraction=1.0, seed=7))
    assert sampled.count(99) > 25
    assert any(index < 99 for index in sampled)
