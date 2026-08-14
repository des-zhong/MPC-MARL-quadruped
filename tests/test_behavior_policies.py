import pytest

torch = pytest.importorskip("torch")

from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds
from dribblebot.world_model.behavior_policies import BehaviorMixture
from dribblebot.world_model.schema import default_state_schema


@pytest.fixture
def action_adapter():
    return JointActionAdapter({
        0: SkillBounds((-1.2, -0.6, 0.0), (1.2, 0.6, 0.0), (1.0, 1.0, 0.0)),
        1: SkillBounds((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0), (1.0, 1.0, 1.0)),
        2: SkillBounds((-3.0, -3.0, 0.0), (3.0, 3.0, 0.0), (1.0, 1.0, 0.0)),
    })


def _states(field_half_length=4.0, field_half_width=2.5, batch=1):
    schema = default_state_schema(1)
    states = torch.zeros(batch, schema.state_dim)
    states[:, schema.slice("field.geometry")] = torch.tensor([
        field_half_length,
        field_half_width,
        -field_half_length,
        field_half_length,
        1.0,
        0.09,
    ])

    robot_positions_metres = ((-0.7, -0.2), (0.9, 0.8))
    for robot, (x, y) in enumerate(robot_positions_metres):
        states[:, schema.slice(f"robot_{robot}.position")][:, :2] = torch.tensor([
            x / field_half_length,
            y / field_half_width,
        ])
        states[:, schema.slice(f"robot_{robot}.yaw_sin_cos")][:, 1] = 1.0
    states[:, schema.slice("ball.position")][:, :2] = torch.tensor([
        -0.1 / field_half_length,
        0.0,
    ])
    states[:, schema.slice("ball.possessor_one_hot")][:, 0] = 1.0
    obstacle = states[:, schema.slice("obstacle_0.geometry")]
    obstacle[:, :] = torch.tensor([
        1.2 / field_half_length,
        -0.4 / field_half_width,
        0.45,
        0.45,
        0.5,
        1.0,
    ])
    return schema, states


def test_scripted_geometry_is_invariant_to_field_normalization(action_adapter):
    schema, small_field = _states(4.0, 2.5)
    _, large_field = _states(8.0, 5.0)
    policy = BehaviorMixture(action_adapter, schema, {"scripted": 1.0}, repeat_previous_probability=0.0)

    assert torch.allclose(policy._scripted(small_field), policy._scripted(large_field), atol=1e-6)


def test_joint_team_scripted_policy_attacks_opposite_goals():
    adapter = JointActionAdapter(
        {
            0: SkillBounds((-1.2, -0.6, 0.0), (1.2, 0.6, 0.0), (1.0, 1.0, 0.0)),
            1: SkillBounds((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0), (1.0, 1.0, 1.0)),
            2: SkillBounds((-3.0, -3.0, 0.0), (3.0, 3.0, 0.0), (1.0, 1.0, 0.0)),
        },
        num_robots=4,
    )
    schema = default_state_schema(0, num_robots=4)
    states = torch.zeros(1, schema.state_dim)
    states[:, schema.slice("field.geometry")] = torch.tensor(
        [4.0, 2.5, -4.0, 4.0, 1.0, 0.09]
    )
    positions = ((-0.2, 0.0), (-1.5, 1.0), (0.2, 0.0), (1.5, -1.0))
    for robot, (x, y) in enumerate(positions):
        states[:, schema.slice(f"robot_{robot}.position")][:, :2] = torch.tensor(
            [x / 4.0, y / 2.5]
        )
        states[:, schema.slice(f"robot_{robot}.yaw_sin_cos")][:, 1] = 1.0
    policy = BehaviorMixture(
        adapter,
        schema,
        {"scripted": 1.0},
        repeat_previous_probability=0.0,
        team_size=2,
    )

    skills, commands = adapter.unpack(policy._scripted(states))

    assert skills[0, 0].item() == 2
    assert skills[0, 2].item() == 2
    assert commands[0, 0, 0] > 0.0
    assert commands[0, 2, 0] < 0.0


def test_random_sampling_goal_directed_weight_is_honored(action_adapter):
    schema, states = _states(batch=32)
    policy = BehaviorMixture(
        action_adapter,
        schema,
        {"random_valid": 1.0},
        repeat_previous_probability=0.0,
        random_sampling={"goal_directed": 1.0},
    )

    sampled, sources = policy.sample(states)
    assert torch.equal(sampled, policy._scripted(states))
    assert sources == ["random_valid"] * len(states)


def test_targeted_scenarios_are_distinct_valid_and_forced(action_adapter):
    scenarios = [
        "goal",
        "pass",
        "ball_obstacle_collision",
        "out_of_bounds",
        "successful_shot",
        "failed_shot",
    ]
    schema, states = _states(batch=len(scenarios))
    policy = BehaviorMixture(
        action_adapter,
        schema,
        {"random_valid": 1.0},
        repeat_previous_probability=1.0,
    )
    previous = action_adapter.random_valid((len(scenarios),))

    actions, sources = policy.sample(
        states,
        previous_actions=previous,
        targeted_scenarios=scenarios,
        previous_action_valid=torch.ones(len(scenarios), dtype=torch.bool),
    )

    action_adapter.assert_within_bounds(actions)
    assert sources == ["targeted_rare_event"] * len(scenarios)
    assert all(not torch.equal(actions[index], previous[index]) for index in range(len(scenarios)))
    assert torch.unique(actions, dim=0).shape[0] == len(scenarios)


def test_repeat_respects_valid_mask_and_reports_truthful_source(action_adapter):
    schema, states = _states(batch=3)
    policy = BehaviorMixture(
        action_adapter,
        schema,
        {"scripted": 1.0},
        repeat_previous_probability=1.0,
    )
    previous = action_adapter.random_valid((3,))

    actions, sources = policy.sample(
        states,
        previous_actions=previous,
        targeted_scenarios=["goal", "", None],
        previous_action_valid=torch.tensor([True, True, False]),
    )

    assert not torch.equal(actions[0], previous[0])
    assert torch.equal(actions[1], previous[1])
    assert not torch.equal(actions[2], previous[2])
    assert sources == ["targeted_rare_event", "repeat_previous", "scripted"]


def test_invalid_sampling_configuration_and_scenario_raise(action_adapter):
    schema, states = _states()
    with pytest.raises(ValueError, match="Unknown random sampling"):
        BehaviorMixture(
            action_adapter,
            schema,
            {"scripted": 1.0},
            random_sampling={"typo": 1.0},
        )
    policy = BehaviorMixture(action_adapter, schema, {"scripted": 1.0})
    with pytest.raises(ValueError, match="Unknown targeted scenarios"):
        policy.sample(states, targeted_scenarios=["not_an_event"])
