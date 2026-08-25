import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc.config import MPCConfig
from dribblebot.mpc.objective import MPCObjective
from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds
from dribblebot.world_model.metrics import reward_ranking_metrics
from dribblebot.world_model.schema import EVENT_NAMES, default_state_schema


def _setup(reward_source="analytical"):
    schema = default_state_schema(0, 1)
    adapter = JointActionAdapter({
        0: SkillBounds((-1.2, -0.6, 0.0), (1.2, 0.6, 0.0), (1, 1, 0)),
        1: SkillBounds((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0), (1, 1, 1)),
        2: SkillBounds((-3.0, -3.0, 0.0), (3.0, 3.0, 0.0), (1, 1, 0)),
    }, 1)
    config = MPCConfig(
        horizon=1, num_candidates=4, num_elites=1, num_iterations=1,
        reward_source=reward_source, objective_mode="reward_only",
    ).validate()
    objective = MPCObjective(schema, adapter, EVENT_NAMES, config)
    state = torch.zeros(1, schema.state_dim)
    state[:, schema.slice("field.geometry")] = torch.tensor([4., 2.5, 4., -4., 1., 0.])
    state[:, schema.slice("robot_0.yaw_sin_cos")] = torch.tensor([0., 1.])
    return schema, adapter, objective, state


def test_invalid_shoot_is_walked_toward_ball_inside_imagination():
    schema, adapter, objective, state = _setup()
    state[:, schema.slice("robot_0.position")] = torch.tensor([-0.5, 0., 0.3])
    state[:, schema.slice("ball.position")] = torch.tensor([0.5, 0., 0.1])
    action = adapter.pack(torch.tensor([[2]]), torch.tensor([[[2., 0., 0.]]]))
    resolved = objective.resolve_imagined_action(state, action)
    skills, commands = adapter.unpack(resolved)
    assert skills.item() == 0
    assert commands[0, 0, 0] > 0


def test_analytical_goal_probability_changes_ranking_without_reward_head():
    schema, adapter, objective, state = _setup()
    actions = adapter.pack(torch.zeros(1, 2, 1, 1, dtype=torch.long), torch.zeros(1, 2, 1, 1, 3))
    states = state[:, None, None].expand(-1, 2, 2, -1).clone()
    events = torch.zeros(1, 2, 1, len(EVENT_NAMES))
    events[0, 1, 0, EVENT_NAMES.index("goal")] = 1.0
    rollout = {
        "predicted_states": states,
        "predicted_rewards": torch.zeros(1, 2, 1),
        "predicted_done_probabilities": torch.zeros(1, 2, 1),
        "state_uncertainty": torch.zeros(1, 2, 1),
        "reward_uncertainty": torch.zeros(1, 2, 1),
        "event_probabilities": events,
        "executed_actions": actions,
    }
    result = objective.evaluate(state, actions, rollout)
    assert result.total.argmax(-1).item() == 1
    assert result.components["analytical_reward_return"][0, 1] == 10.0


def test_hard_uncertainty_threshold_rejects_ood_candidate():
    schema, adapter, objective, state = _setup("learned")
    objective.config.max_state_uncertainty = 0.5
    actions = adapter.pack(torch.zeros(1, 2, 1, 1, dtype=torch.long), torch.zeros(1, 2, 1, 1, 3))
    rollout = {
        "predicted_states": state[:, None, None].expand(-1, 2, 2, -1).clone(),
        "predicted_rewards": torch.ones(1, 2, 1),
        "predicted_done_probabilities": torch.zeros(1, 2, 1),
        "state_uncertainty": torch.tensor([[[0.1], [0.6]]]),
        "reward_uncertainty": torch.zeros(1, 2, 1),
        "event_probabilities": torch.zeros(1, 2, 1, len(EVENT_NAMES)),
    }
    result = objective.evaluate(state, actions, rollout)
    assert result.valid.tolist() == [[True, False]]
    assert result.total[0, 1] == float("-inf")


def test_reward_ranking_metrics_detect_order_and_regret():
    exact = reward_ranking_metrics(torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 3.]), 1)
    reverse = reward_ranking_metrics(torch.tensor([3., 2., 1.]), torch.tensor([1., 2., 3.]), 1)
    assert exact["spearman"] == pytest.approx(1.0)
    assert exact["selection_regret"] == 0.0
    assert reverse["spearman"] == pytest.approx(-1.0)
    assert reverse["selection_regret"] == 2.0


def test_imagined_collision_avoidance_matches_escape_walk_semantics():
    schema = default_state_schema(0, 2)
    adapter = JointActionAdapter({
        0: SkillBounds((-1.2, -0.6, 0.0), (1.2, 0.6, 0.0), (1, 1, 0)),
        1: SkillBounds((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0), (1, 1, 1)),
        2: SkillBounds((-3.0, -3.0, 0.0), (3.0, 3.0, 0.0), (1, 1, 0)),
    }, 2)
    config = MPCConfig(
        horizon=1, num_candidates=4, num_elites=1, num_iterations=1,
        objective_mode="reward_only", reward_source="analytical",
    ).validate()
    objective = MPCObjective(
        schema, adapter, EVENT_NAMES, config, controlled_robot_count=1
    )
    state = torch.zeros(1, schema.state_dim)
    state[:, schema.slice("field.geometry")] = torch.tensor(
        [4., 2.5, 4., -4., 1., 0.]
    )
    state[:, schema.slice("robot_0.position")] = torch.tensor([0., 0., .3])
    state[:, schema.slice("robot_1.position")] = torch.tensor([.1, 0., .3])
    state[:, schema.slice("robot_0.yaw_sin_cos")] = torch.tensor([0., 1.])
    state[:, schema.slice("robot_1.yaw_sin_cos")] = torch.tensor([0., 1.])
    action = adapter.pack(
        torch.tensor([[1, 2]]), torch.tensor([[[.5, 0., 0.], [.5, 0., 0.]]])
    )
    resolved = objective.resolve_imagined_action(state, action)
    skills, commands = adapter.unpack(resolved)
    assert skills.tolist() == [[0, 0]]
    assert commands[0, 0, 0] < 0
    assert commands[0, 1, 0] > 0
