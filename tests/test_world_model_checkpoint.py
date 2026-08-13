from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds
from dribblebot.world_model.ensemble import WorldModelEnsemble
from dribblebot.world_model.normalizer import WorldModelNormalizer
from dribblebot.world_model.schema import default_state_schema
from dribblebot.world_model.trainer import (
    load_checkpoint,
    periodic_best_checkpoint_due,
    save_checkpoint,
)


def build():
    schema = default_state_schema(1)
    adapter = JointActionAdapter({i: SkillBounds((-1, -1, -1 if i == 1 else 0), (1, 1, 1 if i == 1 else 0), (1, 1, 1 if i == 1 else 0)) for i in range(3)})
    normalizer = WorldModelNormalizer(torch.zeros(schema.state_dim), torch.ones(schema.state_dim), torch.zeros(len(schema.continuous_dynamic_indices)), torch.ones(len(schema.continuous_dynamic_indices)))
    model = WorldModelEnsemble(schema, adapter, normalizer, ensemble_size=2, hidden_dims=(16,), skill_embedding_dim=4, cylinder_embedding_dim=4)
    optimizers = [torch.optim.Adam(member.parameters()) for member in model.members]
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, 1) for optimizer in optimizers]
    return model, adapter, optimizers, schedulers


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    model, adapter, optimizers, schedulers = build()
    state = torch.zeros(3, model.schema.state_dim); action = adapter.random_valid((3,))
    expected = model.predict_next(state, action)[0]
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizers, schedulers, 0, {"loss": 1.0}, {"training": {}}, 42)
    loaded, payload = load_checkpoint(path)
    assert torch.allclose(expected, loaded.predict_next(state, action)[0])
    assert payload["state_schema"]["state_dim"] == model.schema.state_dim
    assert tuple(payload["event_names"]) == model.event_names
    assert loaded.event_names == model.event_names


def test_periodic_best_checkpoint_uses_completed_epoch_count():
    assert not periodic_best_checkpoint_due(0, 10)
    assert not periodic_best_checkpoint_due(8, 10)
    assert periodic_best_checkpoint_due(9, 10)
    assert periodic_best_checkpoint_due(19, 10)
    assert not periodic_best_checkpoint_due(9, 0)


def test_one_optimizer_step_does_not_nan():
    model, adapter, optimizers, _ = build()
    state = torch.zeros(8, model.schema.state_dim); action = adapter.random_valid((8,))
    before = model.members[0](model.normalizer.normalize_state(state), action)["delta_mean"].square().mean()
    optimizers[0].zero_grad(); before.backward(); optimizers[0].step()
    after = model.members[0](model.normalizer.normalize_state(state), action)["delta_mean"].square().mean()
    assert torch.isfinite(after)


def test_tiny_deterministic_target_can_be_overfit():
    torch.manual_seed(0)
    model, adapter, _, _ = build()
    member = model.members[0]
    optimizer = torch.optim.Adam(member.parameters(), lr=3e-3)
    state = torch.randn(16, model.schema.state_dim) * 0.1
    action = adapter.random_valid((16,))
    with torch.no_grad():
        initial = member(model.normalizer.normalize_state(state), action)["delta_mean"].square().mean()
    for _ in range(40):
        loss = member(model.normalizer.normalize_state(state), action)["delta_mean"].square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    final = member(model.normalizer.normalize_state(state), action)["delta_mean"].square().mean()
    assert final < initial
