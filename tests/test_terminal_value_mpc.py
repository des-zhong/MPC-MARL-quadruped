import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc.config import MPCConfig
from dribblebot.mpc.objective import MPCObjective
from dribblebot.mpc.terminal_value import (
    ReturnNormalizer, TerminalValueModel, ValueModelConfig,
    build_value_dataset, compute_discounted_returns, load_value_checkpoint,
    save_value_checkpoint,
)
from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds
from dribblebot.world_model.normalizer import WorldModelNormalizer
from dribblebot.world_model.schema import EVENT_NAMES, default_state_schema


def test_discounted_returns_and_boundaries():
    assert np.allclose(
        compute_discounted_returns([1, 2, 3], [0, 0, 1], [0, 0, 0], .5),
        [2.75, 3.5, 3],
    )
    assert compute_discounted_returns([4], [1], [0], .9, bootstrap_value=100)[0] == 4
    with pytest.raises(ValueError):
        compute_discounted_returns([1, 2], [1, 0], [0, 0], .9)


def _objective_case(done=None):
    schema = default_state_schema(1)
    adapter = JointActionAdapter({
        index: SkillBounds((-1, -1, -1), (1, 1, 1), (1, 1, 1))
        for index in range(3)
    })
    state = torch.zeros(1, schema.state_dim)
    state[:, schema.slice("field.geometry")] = torch.tensor([4., 2.5, -4., 4., 1., .09])
    for _, cos in schema.yaw_pairs: state[:, cos] = 1
    states = state[:, None, None].expand(1, 2, 3, -1).clone()
    states[0, 1, -1, schema.slice("ball.position").start] = 1
    actions = adapter.pack(torch.zeros(1, 2, 2, dtype=torch.long), torch.zeros(1, 2, 2, 3))
    rollout = {
        "predicted_states": states,
        "predicted_rewards": torch.tensor([[[2., 0.], [1., 0.]]]),
        "predicted_done_probabilities": torch.zeros(1, 2, 2) if done is None else done,
        "state_uncertainty": torch.zeros(1, 2, 2),
        "reward_uncertainty": torch.zeros(1, 2, 2),
        "event_probabilities": torch.zeros(1, 2, 2, len(EVENT_NAMES)),
    }
    base = dict(horizon=2, num_candidates=4, num_elites=1, num_iterations=1,
                ensemble_objective="mean", uncertainty_penalty=0, return_std_penalty=0,
                collision_penalty=0, out_of_bounds_penalty=0, robot_fall_penalty=0,
                invalid_skill_penalty=0, skill_switch_penalty=0, command_change_penalty=0,
                gamma=.5)
    value = lambda x: 10 * x[..., schema.slice("ball.position").start]
    return schema, adapter, state, actions, rollout, base, value


def test_terminal_discount_mask_ranking_and_modes():
    schema, adapter, state, actions, rollout, base, value = _objective_case()
    reward = MPCObjective(schema, adapter, EVENT_NAMES, MPCConfig(**base), None).evaluate(state, actions, rollout)
    augmented = MPCObjective(
        schema, adapter, EVENT_NAMES,
        MPCConfig(**base, objective_mode="reward_plus_terminal_value"), value,
    ).evaluate(state, actions, rollout)
    assert reward.total.argmax(-1).item() == 0
    assert augmented.total.argmax(-1).item() == 1
    assert torch.allclose(augmented.diagnostics["discounted_terminal_value"], torch.tensor([[0., 2.5]]))
    zero = MPCObjective(
        schema, adapter, EVENT_NAMES,
        MPCConfig(**base, objective_mode="reward_plus_terminal_value", terminal_value_coefficient=0), value,
    ).evaluate(state, actions, rollout)
    assert torch.equal(zero.total, reward.total)
    value_only = MPCObjective(
        schema, adapter, EVENT_NAMES,
        MPCConfig(**base, objective_mode="terminal_value_only"), value,
    ).evaluate(state, actions, rollout)
    assert torch.equal(value_only.components["predicted_reward_return"], torch.zeros_like(reward.total))


def test_predicted_termination_removes_value():
    done = torch.tensor([[[0., 0.], [0., 1.]]])
    schema, adapter, state, actions, rollout, base, value = _objective_case(done)
    result = MPCObjective(
        schema, adapter, EVENT_NAMES,
        MPCConfig(**base, objective_mode="reward_plus_terminal_value"), value,
    ).evaluate(state, actions, rollout)
    assert result.diagnostics["terminal_value_contribution"][0, 1] == 0


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
