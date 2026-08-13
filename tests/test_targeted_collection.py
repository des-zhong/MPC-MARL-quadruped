from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scripts.targeted_collection import (
    TargetedScenarioManager,
    configured_minimum_counts,
    coverage_deficits,
)
from scripts.collect_world_model_data import _episode_limits


class OfflineScenarioManager(TargetedScenarioManager):
    def _commit(self, env_ids, actor_ids):
        self.committed_env_ids = list(env_ids)
        self.committed_actor_ids = list(actor_ids)


def _wrapper(num_envs=4):
    robot_ids = torch.arange(num_envs * 2).view(num_envs, 2)
    ball_ids = torch.arange(num_envs * 2, num_envs * 3)
    obstacle_ids = torch.arange(num_envs * 3, num_envs * 4).view(num_envs, 1)
    roots = torch.zeros(num_envs * 4, 13)
    roots[:, 6] = 1.0
    raw = SimpleNamespace(
        num_envs=num_envs,
        num_static_opponents=1,
        static_opponent_size=(0.45, 0.45, 0.5),
        device=torch.device("cpu"),
        robot_actor_idxs_all=robot_ids,
        object_actor_idxs=ball_ids,
        static_opponent_actor_idxs=obstacle_ids,
        root_states=roots,
        env_origins=torch.zeros(num_envs, 3),
        base_init_state=torch.tensor([0.0, 0.0, 0.34, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        object_init_state=torch.tensor([0.0, 0.0, 0.09, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        object_pos_world_frame=torch.zeros(num_envs, 3),
        object_lin_vel=torch.zeros(num_envs, 3),
        object_ang_vel=torch.zeros(num_envs, 3),
        cfg=SimpleNamespace(
            env=SimpleNamespace(field_length=8.0, field_width=5.0, team_goal_x=4.0),
            ball=SimpleNamespace(radius=0.09),
        ),
    )
    return SimpleNamespace(env=raw)


@pytest.mark.parametrize(
    "scenario, expected_ball_xy, expected_velocity",
    [
        ("goal", (3.94, 0.0), (3.0, 0.0)),
        ("out_of_bounds", (0.0, 2.44), (0.0, 3.0)),
        ("pass", (0.0, 0.0), (2.2, 0.0)),
    ],
)
def test_targeted_reset_stages_events_one_step_from_completion(
    scenario,
    expected_ball_xy,
    expected_velocity,
):
    wrapper = _wrapper(1)
    manager = OfflineScenarioManager(
        wrapper,
        {"enabled": True, "episode_probability": 1.0, "scenarios": [scenario], "ball_speed": 3.0},
    )
    manager.stage(torch.tensor([0]), force=True)
    ball = wrapper.env.root_states[wrapper.env.object_actor_idxs[0]]
    assert torch.allclose(ball[:2], torch.tensor(expected_ball_xy), atol=1e-6)
    assert ball[2].item() == pytest.approx(wrapper.env.cfg.ball.radius + 0.002)
    assert torch.allclose(ball[7:9], torch.tensor(expected_velocity), atol=1e-6)
    assert manager.policy_scenarios() == [scenario]


def test_obstacle_collision_stage_uses_real_box_and_ball_geometry():
    wrapper = _wrapper(1)
    manager = OfflineScenarioManager(
        wrapper,
        {"enabled": True, "episode_probability": 1.0, "scenarios": ["ball_obstacle_collision"]},
    )
    manager.stage(torch.tensor([0]), force=True)
    ball = wrapper.env.root_states[wrapper.env.object_actor_idxs[0]]
    obstacle = wrapper.env.root_states[wrapper.env.static_opponent_actor_idxs[0, 0]]
    collision_extent = 0.5 * wrapper.env.static_opponent_size[0] + wrapper.env.cfg.ball.radius
    gap = abs(float(ball[0] - obstacle[0])) - collision_extent
    assert 0.0 < gap < float(ball[7]) * 0.02


def test_coverage_deficits_and_scenario_completion_are_explicit():
    minimum = configured_minimum_counts(
        {"minimum_event_counts": {"goal": 3, "pass": 2, "out_of_bounds": 0}}
    )
    assert coverage_deficits({"goal": 1, "pass": 2}, minimum) == {"goal": 2}

    wrapper = _wrapper(2)
    manager = OfflineScenarioManager(
        wrapper,
        {"enabled": True, "episode_probability": 1.0, "scenarios": ["goal"], "max_macro_steps": 2},
    )
    manager.stage(torch.tensor([0, 1]), force=True)
    events = torch.zeros(2, 2)
    events[0, 0] = 1.0
    manager.observe(events, ("goal", "pass"))
    assert manager.policy_scenarios() == ["", "goal"]
    manager.observe(torch.zeros_like(events), ("goal", "pass"))
    assert manager.policy_scenarios() == ["", ""]


def test_explicit_num_episodes_is_a_hard_collection_cap():
    args = SimpleNamespace(num_episodes=7)
    config = {"num_episodes": 100, "max_extra_episodes": 50}
    assert _episode_limits(args, config, coverage_enabled=True) == (7, 7)


def test_configured_target_can_retain_coverage_extension():
    args = SimpleNamespace(num_episodes=None)
    config = {"num_episodes": 100, "max_extra_episodes": 50}
    assert _episode_limits(args, config, coverage_enabled=True) == (100, 150)
