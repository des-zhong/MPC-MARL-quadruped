import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds
from dribblebot.world_model.ensemble import WorldModelEnsemble
from dribblebot.world_model.normalizer import WorldModelNormalizer
from dribblebot.world_model.schema import (
    EVENT_NAMES,
    LEGACY_EVENT_NAMES,
    default_state_schema,
    event_names_from_metadata,
)
from dribblebot.world_model.state_adapter import FootballWorldModelStateAdapter, quaternion_to_roll_pitch_yaw
from scripts.collect_world_model_data import TerminalStateCapture, _executed_actions


@pytest.fixture
def action_adapter():
    return JointActionAdapter({
        0: SkillBounds((-1.2, -0.6, 0.0), (1.2, 0.6, 0.0), (1.0, 1.0, 0.0)),
        1: SkillBounds((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0), (1.0, 1.0, 1.0)),
        2: SkillBounds((-3.0, -3.0, 0.0), (3.0, 3.0, 0.0), (1.0, 1.0, 0.0)),
    })


@pytest.fixture
def model(action_adapter):
    schema = default_state_schema(2)
    dynamic = len(schema.continuous_dynamic_indices)
    normalizer = WorldModelNormalizer(torch.zeros(schema.state_dim), torch.ones(schema.state_dim), torch.zeros(dynamic), torch.ones(dynamic))
    return WorldModelEnsemble(schema, action_adapter, normalizer, ensemble_size=3, hidden_dims=(32, 32), skill_embedding_dim=4, cylinder_embedding_dim=8)


def test_action_normalization_round_trip(action_adapter):
    action = action_adapter.random_valid((128,))
    recovered = action_adapter.denormalize_action(action_adapter.normalize_action(action))
    assert torch.allclose(action, recovered, atol=1e-5)


def test_skill_dependent_bounds_and_masks(action_adapter):
    action = action_adapter.random_valid((1024,))
    action_adapter.assert_within_bounds(action)
    skills, params = action_adapter.unpack(action)
    assert torch.all(params[..., 2][skills == 2] == 0)


def test_invalid_skill_id_raises(action_adapter):
    action = torch.zeros(1, 8); action[0, 0] = 9
    with pytest.raises(ValueError, match="Invalid skill"):
        action_adapter.unpack(action)


def test_wrapper_action_round_trip_semantics(action_adapter):
    action = action_adapter.random_valid((8,))
    wrapper_action = action_adapter.to_wrapper_action(action).reshape(8, 2, 6)
    assert torch.equal(wrapper_action[..., :3].argmax(-1), action_adapter.unpack(action)[0])


def test_collector_records_executed_wrapper_action(action_adapter):
    info = {
        "high_level_skill_ids": [[0, 2], [1, 0]],
        "high_level_commands": [
            [[0.4, -0.2, 0.0], [1.5, -2.0, 0.0]],
            [[0.7, 0.3, -0.1], [-0.2, 0.1, 0.0]],
        ],
    }
    actions = _executed_actions(action_adapter, info, "cpu")
    skills, commands = action_adapter.unpack(actions)
    assert skills.tolist() == info["high_level_skill_ids"]
    assert torch.allclose(commands, torch.tensor(info["high_level_commands"]))


def test_angle_conversion_and_wrapping():
    yaw = torch.tensor([math.pi - 0.1])
    quaternion = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)), -1)
    _, _, recovered = quaternion_to_roll_pitch_yaw(quaternion)
    assert torch.allclose(recovered, yaw, atol=1e-6)


def test_state_extraction_is_finite_and_decodable():
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            field_length=8.0, field_width=5.0, team_goal_x=4.0, team_goal_half_width=1.0,
            high_level_walk_command_scale=(1.2, 0.6, 0.0), high_level_dribble_command_scale=(1.5, 1.5, 1.0),
            high_level_shoot_command_scale=(3.0, 3.0, 0.0),
        ),
        ball=SimpleNamespace(radius=0.09),
        rewards=SimpleNamespace(terminal_body_height=0.2, high_level_dribble_skill_distance=1.0),
    )
    roots = torch.zeros(6, 13); roots[:, 6] = 1.0
    roots[[0, 1, 2, 3], 2] = 0.34
    raw = SimpleNamespace(
        device="cpu", num_envs=2, root_states=roots,
        robot_actor_idxs_all=torch.tensor([[0, 1], [2, 3]]), env_origins=torch.zeros(2, 3),
        object_pos_world_frame=torch.tensor([[0.2, 0.0, 0.09], [0.0, 0.3, 0.09]]),
        object_lin_vel=torch.zeros(2, 3), object_ang_vel=torch.zeros(2, 3),
        cfg=cfg, gait_indices=torch.zeros(2), num_static_opponents=1,
        static_opponent_actor_idxs=torch.tensor([[4], [5]]), static_opponent_size=(0.45, 0.45, 0.5),
        high_level_skill_ids=torch.zeros(2, 2, dtype=torch.long), high_level_commands=torch.zeros(2, 2, 3),
    )
    wrapper = SimpleNamespace(env=raw, skill_ids=raw.high_level_skill_ids, skill_commands=raw.high_level_commands)
    adapter = FootballWorldModelStateAdapter(wrapper)
    structured = adapter.extract_state()
    assert torch.isfinite(structured["tensor"]).all()
    assert structured["tensor"].shape == (2, adapter.state_dim)
    decoded = adapter.decode_dynamic_state(structured["tensor"])
    assert torch.equal(decoded["ball.position"], structured["ball.position"])


def _event_test_states(possessors):
    schema = default_state_schema(0)
    states = torch.zeros(len(possessors), schema.state_dim)
    states[:, schema.slice("field.geometry")] = torch.tensor([4.0, 2.5, -4.0, 4.0, 1.0, 0.09])
    for row, possessor in enumerate(possessors):
        states[row, schema.slice("ball.possessor_one_hot")] = torch.tensor(possessor)
        states[row, schema.slice("ball.possessed")] = float(possessor[0] < 0.5)
    return schema, states


def test_pass_event_requires_direct_robot_to_robot_possessor_change():
    # r0->r1 and r1->r0 are passes.  Unchanged possession, loss/acquisition,
    # and malformed multi-hot values must not be labeled as passes.
    schema, state = _event_test_states([
        (0, 1, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 1, 1),
    ])
    _, next_state = _event_test_states([
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
        (0, 0, 1),
    ])
    labels = FootballWorldModelStateAdapter(max_obstacles=0, schema=schema).extract_event_labels(state, next_state)
    assert labels[:, EVENT_NAMES.index("pass")].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    executed = torch.tensor([[2, 0], [0, 2], [0, 0], [0, 0], [0, 0], [2, 0]])
    labels = FootballWorldModelStateAdapter(max_obstacles=0, schema=schema).extract_event_labels(
        state,
        next_state,
        {"high_level_skill_ids": executed},
    )
    assert labels[:, EVENT_NAMES.index("pass")].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    executed[0, 0] = 1
    labels = FootballWorldModelStateAdapter(max_obstacles=0, schema=schema).extract_event_labels(
        state,
        next_state,
        {"high_level_skill_ids": executed},
    )
    assert labels[0, EVENT_NAMES.index("pass")].item() == 0.0


def test_own_goal_is_not_double_labeled_out_of_bounds():
    schema, state = _event_test_states([(1, 0, 0)])
    _, next_state = _event_test_states([(1, 0, 0)])
    next_state[:, schema.slice("ball.in_own_goal")] = 1.0
    labels = FootballWorldModelStateAdapter(max_obstacles=0, schema=schema).extract_event_labels(
        state,
        next_state,
        {"high_level_ball_off_border": torch.tensor([True])},
    )
    assert labels[0, EVENT_NAMES.index("own_goal")].item() == 1.0
    assert labels[0, EVENT_NAMES.index("out_of_bounds")].item() == 0.0


def test_pass_is_not_mislabeled_as_shot_success_or_failure():
    schema, state = _event_test_states([(0, 1, 0)])
    _, next_state = _event_test_states([(0, 0, 1)])
    next_state[:, schema.slice("ball.linear_velocity")][:, 0] = 2.0
    adapter = FootballWorldModelStateAdapter(max_obstacles=0, schema=schema)
    labels = adapter.extract_event_labels(
        state,
        next_state,
        {"high_level_skill_ids": torch.tensor([[2, 0]])},
    )
    assert labels[0, EVENT_NAMES.index("pass")].item() == 1.0
    assert labels[0, EVENT_NAMES.index("successful_shot")].item() == 0.0
    assert labels[0, EVENT_NAMES.index("failed_shot")].item() == 0.0


def test_shot_success_requires_speed_toward_goal():
    schema, state = _event_test_states([(1, 0, 0), (1, 0, 0)])
    next_state = state.clone()
    velocity = next_state[:, schema.slice("ball.linear_velocity")]
    velocity[0, 0] = 2.0
    velocity[1, 1] = 2.0
    labels = FootballWorldModelStateAdapter(max_obstacles=0, schema=schema).extract_event_labels(
        state,
        next_state,
        {"high_level_skill_ids": torch.tensor([[2, 0], [2, 0]])},
    )
    assert labels[:, EVENT_NAMES.index("successful_shot")].tolist() == [1.0, 0.0]
    assert labels[:, EVENT_NAMES.index("failed_shot")].tolist() == [0.0, 1.0]


def test_legacy_metadata_event_schema_preserves_width_and_order():
    metadata_names = event_names_from_metadata({"event_names": list(LEGACY_EVENT_NAMES)})
    assert metadata_names == LEGACY_EVENT_NAMES
    schema, state = _event_test_states([(0, 1, 0)])
    _, next_state = _event_test_states([(0, 0, 1)])
    adapter = FootballWorldModelStateAdapter(max_obstacles=0, schema=schema, event_names=metadata_names)
    labels = adapter.extract_event_labels(state, next_state, {"high_level_goal": torch.tensor([True])})
    assert labels.shape == (1, len(LEGACY_EVENT_NAMES))
    assert labels[0, metadata_names.index("goal")].item() == 1.0
    assert "pass" not in metadata_names


def test_ensemble_output_shapes_and_uncertainty(model, action_adapter):
    state = torch.zeros(7, model.schema.state_dim)
    action = action_adapter.random_valid((7,))
    next_state, reward, done, event, uncertainty = model.predict_next(state, action)
    assert next_state.shape == state.shape
    assert reward.shape == done.shape == (7,)
    assert event.shape == (7, len(EVENT_NAMES))
    assert torch.all(uncertainty["state_total_variance"] >= 0)


def test_single_robot_action_encoding(action_adapter):
    schema = default_state_schema(max_obstacles=1, num_robots=1)
    adapter = JointActionAdapter(action_adapter.bounds, num_robots=1)
    dynamic = len(schema.continuous_dynamic_indices)
    normalizer = WorldModelNormalizer(
        torch.zeros(schema.state_dim),
        torch.ones(schema.state_dim),
        torch.zeros(dynamic),
        torch.ones(dynamic),
    )
    single_robot_model = WorldModelEnsemble(
        schema,
        adapter,
        normalizer,
        ensemble_size=1,
        hidden_dims=(32,),
        skill_embedding_dim=16,
        cylinder_embedding_dim=8,
    )

    outputs = single_robot_model.forward_members(
        torch.zeros(4, schema.state_dim),
        adapter.random_valid((4,)),
    )

    assert outputs["delta_mean"].shape == (1, 4, dynamic)


def test_multistep_rollout_shapes(model, action_adapter):
    state = torch.zeros(2, model.schema.state_dim)
    actions = action_adapter.random_valid((2, 5, 4))
    result = model.rollout(state, actions)
    assert result["predicted_states"].shape == (2, 5, 5, model.schema.state_dim)
    assert result["predicted_rewards"].shape == (2, 5, 4)
    assert result["event_probabilities"].shape == (2, 5, 4, len(EVENT_NAMES))


def test_static_context_unchanged_and_yaw_normalized(model, action_adapter):
    state = torch.randn(3, model.schema.state_dim)
    for sin_index, cos_index in model.schema.yaw_pairs:
        state[:, sin_index] = 0; state[:, cos_index] = 1
    actions = action_adapter.random_valid((3, 2, 3))
    result = model.rollout(state, actions)["predicted_states"]
    assert torch.equal(result[:, :, :, model.schema.static_indices], state[:, None, None, model.schema.static_indices].expand(-1, 2, 4, -1))
    for sin_index, cos_index in model.schema.yaw_pairs:
        norm = result[..., [sin_index, cos_index]].norm(dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_terminated_trajectories_are_masked(model, action_adapter):
    for member in model.members:
        torch.nn.init.zeros_(member.termination_logit.weight); member.termination_logit.bias.data.fill_(100)
        torch.nn.init.zeros_(member.reward_mean.weight); member.reward_mean.bias.data.fill_(1)
    state = torch.zeros(1, model.schema.state_dim)
    actions = action_adapter.random_valid((1, 2, 3))
    reward = model.rollout(state, actions)["predicted_rewards"]
    assert torch.all(reward[:, :, 1:] == 0)


def test_discounted_return_shape(model, action_adapter):
    values = model.evaluate_action_sequences(torch.zeros(2, model.schema.state_dim), action_adapter.random_valid((2, 6, 5)), 0.99, 0.1)
    assert values.shape == (2, 6)


def test_cpu_inference(model, action_adapter):
    model.cpu()
    model.predict_next(torch.zeros(1, model.schema.state_dim), action_adapter.random_valid((1,)))


def test_vectorized_terminal_capture_records_each_environment_once():
    class FakeAdapter:
        state_dim = 2
        def extract_state(self, wrapper):
            return {"tensor": wrapper.env.values.clone()}
    class Raw:
        def __init__(self):
            self.num_envs = 3; self.device = "cpu"; self.values = torch.arange(6).reshape(3, 2).float()
        def reset_idx(self, env_ids):
            self.values[env_ids] = -1
    wrapper = SimpleNamespace(env=Raw())
    capture = TerminalStateCapture(wrapper, FakeAdapter())
    expected = wrapper.env.values.clone()
    wrapper.env.reset_idx(torch.tensor([0, 2]))
    assert torch.equal(capture.states[[0, 2]], expected[[0, 2]])
    assert capture.valid.tolist() == [True, False, True]
    capture.restore()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_inference(model, action_adapter):
    model.cuda()
    model.predict_next(torch.zeros(1, model.schema.state_dim, device="cuda"), action_adapter.random_valid((1,), "cuda"))
