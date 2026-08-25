import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc import HybridCEMMPC, MPCConfig
from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds
from dribblebot.world_model.ensemble import WorldModelEnsemble
from dribblebot.world_model.normalizer import WorldModelNormalizer
from dribblebot.world_model.schema import default_state_schema


def _model_and_state(batch=2):
    schema = default_state_schema(2)
    adapter = JointActionAdapter(
        {
            0: SkillBounds((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0), (1, 1, 1)),
            1: SkillBounds((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0), (1, 1, 1)),
            2: SkillBounds((-3.0, -3.0, 0.0), (3.0, 3.0, 0.0), (1, 1, 0)),
        }
    )
    dynamic = len(schema.continuous_dynamic_indices)
    normalizer = WorldModelNormalizer(
        torch.zeros(schema.state_dim),
        torch.ones(schema.state_dim),
        torch.zeros(dynamic),
        torch.ones(dynamic),
    )
    model = WorldModelEnsemble(
        schema,
        adapter,
        normalizer,
        ensemble_size=2,
        hidden_dims=(16,),
        skill_embedding_dim=4,
        cylinder_embedding_dim=4,
    )
    state = torch.zeros(batch, schema.state_dim)
    state[:, schema.slice("field.geometry")] = torch.tensor(
        [4.0, 2.5, -4.0, 4.0, 1.0, 0.09]
    )
    state[:, schema.slice("ball.possessor_one_hot")][:, 0] = 1.0
    for _, cos_index in schema.yaw_pairs:
        state[:, cos_index] = 1.0
    return model, state


def _config(**overrides):
    values = dict(
        horizon=3,
        num_candidates=12,
        num_elites=3,
        num_iterations=2,
        max_candidate_diagnostics=0,
        seed=7,
    )
    values.update(overrides)
    return MPCConfig(**values).validate()


def test_cem_shapes_bounds_masks_probabilities_and_stds():
    model, state = _model_and_state(2)
    result = HybridCEMMPC(model, config=_config()).plan(state)
    assert result.first_joint_action.shape == (2, 8)
    assert result.best_action_sequence.shape == (2, 3, 8)
    assert result.predicted_states.shape == (2, 4, model.schema.state_dim)
    assert result.final_skill_probabilities.shape == (2, 3, 2, 3)
    assert result.final_parameter_means.shape == (2, 3, 2, 3, 3)
    assert result.elite_objectives.shape == (2, 3)
    model.action_adapter.assert_within_bounds(result.best_action_sequence)
    assert torch.allclose(
        result.final_skill_probabilities.sum(-1),
        torch.ones_like(result.final_skill_probabilities[..., 0]),
    )
    assert torch.all(
        result.final_skill_probabilities >= result.planner_state.skill_probabilities.new_tensor(0.02)
    )
    assert torch.all(result.planner_state.parameter_stds > 0)
    # Shoot yaw is masked and therefore always exactly zero.
    assert torch.all(result.final_parameter_means[..., 2, 2] == 0)
    assert torch.all(result.final_parameter_stds[..., 2, 2] == 0)


def test_single_and_vectorized_planning():
    model, single = _model_and_state(1)
    planner = HybridCEMMPC(model, config=_config(num_iterations=1))
    assert planner.plan(single).first_joint_action.shape == (1, 8)
    vector = single.expand(4, -1).clone()
    assert planner.plan(vector).first_joint_action.shape == (4, 8)


def test_fixed_robot_forecast_is_preserved_in_every_planned_step():
    model, state = _model_and_state(2)
    planner = HybridCEMMPC(model, config=_config(num_iterations=1))
    skills = torch.ones(2, 3, 2, dtype=torch.long)
    parameters = torch.zeros(2, 3, 2, 3)
    parameters[..., 1, 0] = 0.4
    fixed = model.action_adapter.pack(skills, parameters)
    result = planner.plan(
        state,
        fixed_action_sequence=fixed,
        fixed_robot_mask=torch.tensor([False, True]),
    )
    planned = result.best_action_sequence.reshape(2, 3, 2, 4)
    expected = fixed.reshape(2, 3, 2, 4)
    assert torch.allclose(planned[..., 1, :], expected[..., 1, :])


def test_warm_start_shifts_final_distribution():
    model, state = _model_and_state(2)
    planner = HybridCEMMPC(model, config=_config())
    result = planner.plan(state)
    probabilities, means, stds = planner._initialize(state, result.planner_state)
    assert torch.allclose(
        probabilities[:, 0], result.final_skill_probabilities[:, 1]
    )
    assert torch.allclose(means[:, 0], result.planner_state.parameter_means[:, 1])
    assert torch.allclose(stds[:, 0], result.planner_state.parameter_stds[:, 1])


def test_fixed_seed_planning_is_repeatable():
    model, state = _model_and_state(1)
    first = HybridCEMMPC(model, config=_config()).plan(state)
    second = HybridCEMMPC(model, config=_config()).plan(state)
    assert torch.equal(first.first_joint_action, second.first_joint_action)
    assert torch.equal(first.final_skill_probabilities, second.final_skill_probabilities)


def test_planner_returns_action_from_highest_reward_sequence():
    schema = default_state_schema(max_obstacles=0, num_robots=1)
    adapter = JointActionAdapter(
        {
            index: SkillBounds((-1, -1, -1), (1, 1, 1), (1, 1, 1))
            for index in range(3)
        },
        num_robots=1,
    )

    class RewardModel:
        state_adapter = object()
        event_names = ()

        def __init__(self):
            self.schema = schema
            self.action_adapter = adapter

        def rollout(self, states, actions, **kwargs):
            batch, candidates, horizon = actions.shape[:3]
            rewards = actions[..., 1]
            return {
                "predicted_states": states[:, None, None].expand(
                    batch, candidates, horizon + 1, -1
                ).clone(),
                "predicted_rewards": rewards,
                "predicted_done_probabilities": torch.zeros_like(rewards),
                "state_uncertainty": torch.zeros_like(rewards),
                "reward_uncertainty": torch.zeros_like(rewards),
            }

    planner = HybridCEMMPC(
        RewardModel(),
        config=MPCConfig(
            horizon=2,
            num_candidates=4,
            num_elites=1,
            num_iterations=1,
        ),
    )

    def fixed_samples(probabilities, means, stds):
        del means, stds
        values = torch.tensor(
            [-1.0, 0.0, 0.5, 1.0],
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        skills = torch.zeros(1, 4, 2, 1, dtype=torch.long)
        normalized = torch.zeros(1, 4, 2, 1, 3)
        normalized[..., 0] = values[None, :, None, None]
        actions = adapter.pack(
            skills, adapter.denormalize_parameters(skills, normalized)
        )
        return actions, skills, normalized

    planner._sample = fixed_samples
    result = planner.plan(torch.zeros(1, schema.state_dim))

    assert result.first_joint_action[0, 1] == 1.0
    assert result.best_objective[0] == 2.0


def test_invalid_predictions_fail_explicitly(monkeypatch):
    model, state = _model_and_state(1)

    def invalid_rollout(initial, actions, **kwargs):
        batch, candidates, horizon = actions.shape[:3]
        nan = torch.full(
            (batch, candidates, horizon), float("nan"), dtype=initial.dtype
        )
        return {
            "predicted_states": torch.full(
                (batch, candidates, horizon + 1, model.schema.state_dim),
                float("nan"),
            ),
            "predicted_rewards": nan,
            "predicted_done_probabilities": nan,
            "state_uncertainty": nan,
            "reward_uncertainty": nan,
            "event_probabilities": torch.full(
                (batch, candidates, horizon, model.num_events), float("nan")
            ),
            "member_predicted_rewards": nan.unsqueeze(0).expand(2, -1, -1, -1),
            "member_predicted_done_probabilities": nan.unsqueeze(0).expand(
                2, -1, -1, -1
            ),
        }

    monkeypatch.setattr(model, "rollout", invalid_rollout)
    with pytest.raises(RuntimeError, match="finite predicted-reward return"):
        HybridCEMMPC(model, config=_config(num_iterations=1)).plan(state)


def test_all_ood_candidates_use_finite_uncertainty_penalized_fallback(monkeypatch):
    model, state = _model_and_state(1)
    original_rollout = model.rollout

    def all_ood_rollout(initial, actions, **kwargs):
        rollout = original_rollout(initial, actions, **kwargs)
        rollout["state_uncertainty"] = torch.full_like(
            rollout["state_uncertainty"], 2.0
        )
        return rollout

    monkeypatch.setattr(model, "rollout", all_ood_rollout)
    result = HybridCEMMPC(
        model,
        config=_config(
            num_iterations=1,
            max_state_uncertainty=0.5,
            uncertainty_penalty=0.1,
        ),
    ).plan(state)

    assert torch.isfinite(result.best_objective).all()
    assert result.uncertainty["ood_fallback_used"].tolist() == [True]
