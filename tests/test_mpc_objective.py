import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc.config import MPCConfig
from dribblebot.mpc.objective import MPCObjective
from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds
from dribblebot.world_model.schema import EVENT_NAMES, default_state_schema


def _adapter():
    return JointActionAdapter(
        {
            0: SkillBounds((-1, -1, -1), (1, 1, 1), (1, 1, 1)),
            1: SkillBounds((-1, -1, -1), (1, 1, 1), (1, 1, 1)),
            2: SkillBounds((-2, -2, 0), (2, 2, 0), (1, 1, 0)),
        }
    )


def _case():
    schema = default_state_schema(1)
    adapter = _adapter()
    state = torch.zeros(1, schema.state_dim)
    state[:, schema.slice("field.geometry")] = torch.tensor(
        [4.0, 2.5, -4.0, 4.0, 1.0, 0.09]
    )
    for _, cos_index in schema.yaw_pairs:
        state[:, cos_index] = 1.0
    skills = torch.zeros(1, 2, 2, dtype=torch.long)
    parameters = torch.zeros(1, 2, 2, 3)
    actions = adapter.pack(skills, parameters)
    states = state[:, None, None].expand(-1, 2, 3, -1).clone()
    rollout = {
        "predicted_states": states,
        "predicted_rewards": torch.ones(1, 2, 2),
        "predicted_done_probabilities": torch.zeros(1, 2, 2),
        "state_uncertainty": torch.zeros(1, 2, 2),
        "reward_uncertainty": torch.zeros(1, 2, 2),
        "event_probabilities": torch.zeros(1, 2, 2, len(EVENT_NAMES)),
    }
    return schema, adapter, state, actions, rollout


def _base_config(**overrides):
    values = dict(
        horizon=2,
        num_candidates=4,
        num_elites=1,
        num_iterations=1,
        uncertainty_penalty=0,
        return_std_penalty=0,
        ensemble_objective="mean",
        collision_penalty=0,
        out_of_bounds_penalty=0,
        robot_fall_penalty=0,
        invalid_skill_penalty=0,
        skill_switch_penalty=0,
        command_change_penalty=0,
    )
    values.update(overrides)
    return MPCConfig(**values).validate()


def test_objective_decomposition_sums_exactly():
    schema, adapter, state, actions, rollout = _case()
    result = MPCObjective(
        schema, adapter, EVENT_NAMES, _base_config()
    ).evaluate(state, actions, rollout)
    assert torch.allclose(result.total, sum(result.components.values()))


def test_uncertainty_penalty_decreases_objective():
    schema, adapter, state, actions, rollout = _case()
    rollout["state_uncertainty"].fill_(2.0)
    plain = MPCObjective(
        schema, adapter, EVENT_NAMES, _base_config()
    ).evaluate(state, actions, rollout)
    penalized = MPCObjective(
        schema,
        adapter,
        EVENT_NAMES,
        _base_config(uncertainty_penalty=0.5),
    ).evaluate(state, actions, rollout)
    assert torch.all(penalized.total < plain.total)


def test_collision_penalty_decreases_objective():
    schema, adapter, state, actions, rollout = _case()
    rollout["event_probabilities"][
        ..., EVENT_NAMES.index("robot_obstacle_collision")
    ] = 1.0
    plain = MPCObjective(
        schema, adapter, EVENT_NAMES, _base_config()
    ).evaluate(state, actions, rollout)
    penalized = MPCObjective(
        schema,
        adapter,
        EVENT_NAMES,
        _base_config(collision_penalty=1.0),
    ).evaluate(state, actions, rollout)
    assert torch.all(penalized.total < plain.total)


def test_skill_switch_penalty_counts_temporal_changes():
    schema, adapter, state, actions, rollout = _case()
    switched_skills = torch.tensor([[[0, 0], [1, 0]]])
    switched = adapter.pack(switched_skills, torch.zeros(1, 2, 2, 3))
    result = MPCObjective(
        schema,
        adapter,
        EVENT_NAMES,
        _base_config(skill_switch_penalty=1.0),
    ).evaluate(state, switched, rollout)
    assert torch.all(result.components["skill_switch_penalty"] < 0)


def test_ball_setup_penalty_prefers_ball_in_front_for_dribbling():
    schema, adapter, state, _, rollout = _case()
    skills = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    skills[..., 0] = 1
    actions = adapter.pack(skills, torch.zeros(1, 2, 2, 2, 3))

    # Candidate 0 leaves the ball under robot 0. Candidate 1 keeps it 0.35 m
    # ahead. State positions are normalized by the 4 m field half-length.
    rollout["predicted_states"][:, 1, 1:, schema.slice("ball.position")][..., 0] = 0.35 / 4.0
    result = MPCObjective(
        schema,
        adapter,
        EVENT_NAMES,
        _base_config(ball_setup_penalty=0.5),
    ).evaluate(state, actions, rollout)

    assert result.components["ball_setup_penalty"][0, 0] < 0
    torch.testing.assert_close(
        result.components["ball_setup_penalty"][0, 1], torch.tensor(0.0)
    )
    assert result.total[0, 1] > result.total[0, 0]


def test_ball_setup_penalty_stops_close_walk_approach_in_front():
    schema, adapter, state, _, rollout = _case()
    skills = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    actions = adapter.pack(skills, torch.zeros(1, 2, 2, 2, 3))

    rollout["predicted_states"][:, 1, 1:, schema.slice("ball.position")][..., 0] = 0.35 / 4.0
    result = MPCObjective(
        schema,
        adapter,
        EVENT_NAMES,
        _base_config(ball_setup_penalty=0.5),
    ).evaluate(state, actions, rollout)

    assert result.total[0, 1] > result.total[0, 0]


def test_shoot_setup_uses_requested_field_frame_direction():
    schema, adapter, state, _, rollout = _case()
    skills = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    skills[..., 0] = 2
    parameters = torch.zeros(1, 2, 2, 2, 3)
    parameters[..., 0, 1] = 1.0  # Shoot toward field +y.
    actions = adapter.pack(skills, parameters)

    # The preferred base-to-ball offset follows the shot command, rather than
    # always assuming that the field +x direction is in front.
    rollout["predicted_states"][:, 1, 1:, schema.slice("ball.position")][..., 1] = 0.45 / 2.5
    result = MPCObjective(
        schema,
        adapter,
        EVENT_NAMES,
        _base_config(ball_setup_penalty=0.5),
    ).evaluate(state, actions, rollout)

    torch.testing.assert_close(
        result.components["ball_setup_penalty"][0, 1], torch.tensor(0.0)
    )
    assert result.total[0, 1] > result.total[0, 0]


def test_backward_dribble_penalty_prefers_body_forward_motion():
    schema, adapter, state, _, rollout = _case()
    skills = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    skills[..., 0] = 1
    actions = adapter.pack(skills, torch.zeros(1, 2, 2, 2, 3))

    velocity = schema.slice("robot_0.linear_velocity")
    rollout["predicted_states"][:, 0, 1:, velocity][..., 0] = -1.0
    rollout["predicted_states"][:, 1, 1:, velocity][..., 0] = 1.0
    result = MPCObjective(
        schema,
        adapter,
        EVENT_NAMES,
        _base_config(backward_dribble_penalty=0.5),
    ).evaluate(state, actions, rollout)

    assert result.components["backward_dribble_penalty"][0, 0] < 0
    torch.testing.assert_close(
        result.components["backward_dribble_penalty"][0, 1], torch.tensor(0.0)
    )
    assert result.total[0, 1] > result.total[0, 0]


def test_reposition_approach_prefers_immediate_toward_ball_command():
    schema, adapter, state, _, rollout = _case()
    ball = schema.slice("ball.position")
    robot = schema.slice("robot_0.position")
    # Ball is two metres in world +x from robot 0. Candidate 0 commands and
    # predicts motion away; candidate 1 commands and predicts motion toward it.
    rollout["predicted_states"][..., ball][..., 0] = 2.0 / 4.0
    rollout["predicted_states"][:, 0, 1:, robot][..., 0] = -0.1 / 4.0
    rollout["predicted_states"][:, 1, 1:, robot][..., 0] = 0.1 / 4.0
    skills = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    parameters = torch.zeros(1, 2, 2, 2, 3)
    parameters[:, 0, :, 0, 0] = -1.0
    parameters[:, 1, :, 0, 0] = 1.0
    actions = adapter.pack(skills, parameters)

    result = MPCObjective(
        schema,
        adapter,
        EVENT_NAMES,
        _base_config(
            reposition_approach_coefficient=1.0,
            reposition_command_alignment_coefficient=1.0,
            reposition_first_step_multiplier=3.0,
        ),
    ).evaluate(state, actions, rollout)

    assert result.components["reposition_approach"][0, 1] > 0
    assert result.components["reposition_approach"][0, 0] < 0
    assert result.components["reposition_command_alignment"][0, 1] > 0
    assert result.components["reposition_command_alignment"][0, 0] < 0
    assert result.total[0, 1] > result.total[0, 0]
