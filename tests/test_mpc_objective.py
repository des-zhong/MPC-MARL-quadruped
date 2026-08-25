import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc.config import MPCConfig
from dribblebot.mpc.objective import MPCObjective
from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds
from dribblebot.world_model.schema import EVENT_NAMES, default_state_schema


def _case(gamma=1.0):
    schema = default_state_schema(1)
    adapter = JointActionAdapter(
        {
            index: SkillBounds((-1, -1, -1), (1, 1, 1), (1, 1, 1))
            for index in range(3)
        },
        num_robots=1,
    )
    state = torch.zeros(1, schema.state_dim)
    actions = adapter.pack(
        torch.zeros(1, 2, 3, 1, dtype=torch.long),
        torch.zeros(1, 2, 3, 1, 3),
    )
    states = state[:, None, None].expand(-1, 2, 4, -1).clone()
    rollout = {
        "predicted_states": states,
        "predicted_rewards": torch.tensor([[[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]]),
        "predicted_done_probabilities": torch.zeros(1, 2, 3),
        "state_uncertainty": torch.zeros(1, 2, 3),
        "reward_uncertainty": torch.zeros(1, 2, 3),
        "event_probabilities": torch.zeros(1, 2, 3, len(EVENT_NAMES)),
    }
    config = MPCConfig(
        horizon=3,
        num_candidates=4,
        num_elites=1,
        num_iterations=1,
        gamma=gamma,
    ).validate()
    return MPCObjective(schema, adapter, EVENT_NAMES, config), state, actions, rollout


def test_objective_is_exact_undiscounted_predicted_return():
    objective, state, actions, rollout = _case()
    result = objective.evaluate(state, actions, rollout)

    torch.testing.assert_close(result.total, torch.tensor([[6.0, 6.0]]))
    torch.testing.assert_close(
        result.components["predicted_reward_return"], result.total
    )
    torch.testing.assert_close(result.total, sum(result.components.values()))


def test_gamma_is_the_only_optional_return_weighting():
    objective, state, actions, rollout = _case(gamma=0.5)
    result = objective.evaluate(state, actions, rollout)

    torch.testing.assert_close(result.total, torch.tensor([[2.75, 3.5]]))


def test_non_reward_predictions_do_not_change_candidate_ranking():
    objective, state, actions, rollout = _case()
    baseline = objective.evaluate(state, actions, rollout).total

    rollout["state_uncertainty"][0, 0] = 1.0e6
    rollout["reward_uncertainty"][0, 0] = 1.0e6
    rollout["event_probabilities"][0, 0] = 1.0
    rollout["predicted_done_probabilities"][0, 0] = 1.0
    changed = objective.evaluate(state, actions, rollout).total

    torch.testing.assert_close(changed, baseline)


def test_uncertainty_penalties_change_candidate_ranking():
    objective, state, actions, rollout = _case()
    objective.config.uncertainty_penalty = 1.0
    objective.config.return_uncertainty_penalty = 1.0
    rollout["state_uncertainty"][0, 0] = 2.0
    rollout["reward_uncertainty"][0, 0] = 4.0

    result = objective.evaluate(state, actions, rollout)

    assert result.total[0, 0] < result.total[0, 1]
    assert result.components["state_uncertainty_penalty"][0, 0] < 0
    assert result.components["return_uncertainty_penalty"][0, 0] < 0


def test_terminal_value_can_reverse_short_horizon_reward_ranking():
    objective, state, actions, rollout = _case(gamma=0.5)
    terminal_index = objective.schema.slice("ball.position").start
    rollout["predicted_rewards"] = torch.tensor(
        [[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    rollout["predicted_states"][0, 1, -1, terminal_index] = 2.0
    objective.config.objective_mode = "reward_plus_terminal_value"
    objective.terminal_value = lambda terminal: 10.0 * terminal[..., terminal_index]

    result = objective.evaluate(state, actions, rollout)

    assert result.components["predicted_reward_return"].argmax(-1).item() == 0
    assert result.total.argmax(-1).item() == 1
    torch.testing.assert_close(
        result.components["terminal_value"], torch.tensor([[0.0, 2.5]])
    )


def test_terminal_value_ensemble_uncertainty_gates_value():
    objective, state, actions, rollout = _case()
    objective.config.objective_mode = "reward_plus_terminal_value"
    objective.config.terminal_value_uncertainty_gating = True
    objective.config.terminal_value_uncertainty_beta = 1.0
    objective.terminal_value = lambda terminal: (
        torch.ones(terminal.shape[:-1]),
        torch.tensor([[10.0, 0.0]]),
    )
    result = objective.evaluate(state, actions, rollout)
    assert result.components["terminal_value"][0, 0] < 1.0e-3
    assert result.components["terminal_value"][0, 1] > 0.9
    torch.testing.assert_close(
        result.diagnostics["terminal_value_uncertainty"],
        torch.tensor([[10.0, 0.0]]),
    )


def test_non_finite_model_predictions_are_invalid():
    objective, state, actions, rollout = _case()
    rollout["predicted_rewards"][0, 0, 1] = float("nan")
    result = objective.evaluate(state, actions, rollout)

    assert not result.valid[0, 0]
    assert result.valid[0, 1]
    assert result.total[0, 0] == float("-inf")
