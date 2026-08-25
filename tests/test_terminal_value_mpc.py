import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc.terminal_value import (
    ReturnNormalizer, TerminalValueModel, ValueModelConfig,
    ValueDataset, build_value_dataset, compute_discounted_returns,
    load_value_checkpoint,
    save_value_checkpoint,
)
from dribblebot.world_model.normalizer import WorldModelNormalizer
from dribblebot.world_model.schema import default_state_schema


def test_discounted_returns_and_boundaries():
    assert np.allclose(
        compute_discounted_returns([1, 2, 3], [0, 0, 1], [0, 0, 0], .5),
        [2.75, 3.5, 3],
    )
    assert compute_discounted_returns([4], [1], [0], .9, bootstrap_value=100)[0] == 4
    with pytest.raises(ValueError):
        compute_discounted_returns([1, 2], [1, 0], [0, 0], .9)


def test_value_shape_normalization_and_checkpoint(tmp_path):
    schema = default_state_schema(1); dynamic = len(schema.continuous_dynamic_indices)
    normalizer = WorldModelNormalizer(torch.zeros(schema.state_dim), torch.ones(schema.state_dim), torch.zeros(dynamic), torch.ones(dynamic))
    model = TerminalValueModel(schema, normalizer, hidden_dims=(8,), return_normalizer=ReturnNormalizer(4, 2))
    assert model.predict(torch.zeros(3, schema.state_dim)).shape == (3,)
    value = torch.tensor([2., 4., 6.]); assert torch.allclose(model.return_normalizer.denormalize(model.return_normalizer.normalize(value)), value)
    optimizer = torch.optim.Adam(model.parameters())
    config = ValueModelConfig(hidden_dims=(8,), device="cpu")
    path = tmp_path / "value.pt"
    save_value_checkpoint(path, model, optimizer, 0, config, {"loss": 1}, "manifest")
    loaded, _ = load_value_checkpoint(path)
    assert torch.allclose(model.predict(torch.zeros(3, schema.state_dim)), loaded.predict(torch.zeros(3, schema.state_dim)))


def test_value_ensemble_reports_epistemic_uncertainty():
    schema = default_state_schema(0)
    dynamic = len(schema.continuous_dynamic_indices)
    normalizer = WorldModelNormalizer(
        torch.zeros(schema.state_dim), torch.ones(schema.state_dim),
        torch.zeros(dynamic), torch.ones(dynamic),
    )
    model = TerminalValueModel(schema, normalizer, hidden_dims=(8,), ensemble_size=3)
    mean, uncertainty = model.predict_with_uncertainty(
        torch.zeros(4, schema.state_dim)
    )
    assert mean.shape == uncertainty.shape == (4,)
    assert torch.all(uncertainty >= 0)


def test_teacher_value_dataset_adds_imagined_terminal_states(tmp_path):
    source = tmp_path / "teacher"
    episodes = source / "episodes"
    episodes.mkdir(parents=True)
    arrays = {
        "episode_id": np.zeros(3, dtype=np.int64),
        "step_id": np.arange(3, dtype=np.int64),
        "global_state": np.zeros((3, 4), dtype=np.float32),
        "real_reward": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "terminated": np.asarray([False, False, True]),
        "truncated": np.zeros(3, dtype=bool),
        "behavior_source": np.asarray(["mpc"] * 3),
        "predicted_plan_states": np.zeros((3, 3, 4), dtype=np.float32),
    }
    arrays["predicted_plan_states"][0, -1] = 9.0
    np.savez_compressed(episodes / "episode_00000000.npz", **arrays)
    entry = {
        "episode_id": 0, "path": "episodes/episode_00000000.npz", "length": 3,
    }
    (source / "manifest.json").write_text(json.dumps({"episodes": [entry]}))
    (source / "metadata.json").write_text(json.dumps({"format": "teacher"}))
    output = tmp_path / "value"
    config = ValueModelConfig(
        device="cpu", bootstrap_on_truncation=False,
        split={"train": 0.0, "validation": 0.0, "test": 1.0},
    )
    build_value_dataset(source, output, config)
    dataset = ValueDataset(output, "test")
    assert len(dataset) == 4
    imagined = dataset[3]
    assert imagined["sample_kind"] == "imagined_terminal"
    assert torch.all(imagined["global_state"] == 9.0)
    assert imagined["return_to_go"].item() == pytest.approx(3.0)
