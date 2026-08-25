from types import SimpleNamespace

import gym
import isaacgym
import torch
import torch.nn as nn

assert isaacgym

from dribblebot.envs.wrappers.shared_self_play_wrapper import (
    SharedPolicySelfPlayWrapper,
)


class _FakeMatchEnv(gym.Env):
    def __init__(self, matches=32, team_size=2):
        self.num_robots = 2 * team_size
        self.num_envs = matches
        self.num_train_envs = matches
        self.history_length = 4
        self.max_episode_length = 100
        self.device = torch.device("cpu")
        self.episode_length_buf = torch.zeros(matches, dtype=torch.long)
        self.cfg = SimpleNamespace()


class _FakeActor:
    def __init__(self, history_dim):
        self.adaptation_module = nn.Linear(history_dim, 2)
        self.actor_body = nn.Linear(history_dim + 2, 6)


def test_opponent_pool_preserves_anchor_and_evicts_oldest_non_anchor():
    env = SharedPolicySelfPlayWrapper(
        _FakeMatchEnv(), team_size=2, opponent_pool_size=3
    )
    actor = _FakeActor(env.num_obs_history)

    for iteration in (0, 10, 20, 30):
        env.update_opponent_policy(actor, iteration=iteration)

    assert env.opponent_pool_iterations == [0, 20, 30]
    assert len(env.opponent_pool) == 3


def test_opponent_pool_probability_controls_per_match_assignment():
    env = SharedPolicySelfPlayWrapper(
        _FakeMatchEnv(),
        team_size=2,
        opponent_pool_size=3,
        opponent_latest_probability=1.0,
    )
    actor = _FakeActor(env.num_obs_history)
    env.update_opponent_policy(actor, iteration=0)
    env.update_opponent_policy(actor, iteration=10)

    # Ongoing matches keep their current policy until the next episode.
    assert torch.all(env.opponent_assignment == 0)
    env._sample_opponent_assignments()
    assert torch.all(env.opponent_assignment == 1)
    env.opponent_latest_probability = 0.0
    env._sample_opponent_assignments()
    assert torch.all(env.opponent_assignment == 0)


def test_opponent_pool_checkpoint_round_trip():
    source = SharedPolicySelfPlayWrapper(
        _FakeMatchEnv(), team_size=2, opponent_pool_size=4
    )
    actor = _FakeActor(source.num_obs_history)
    source.update_opponent_policy(actor, iteration=0)
    source.update_opponent_policy(actor, iteration=20)
    payload = source.opponent_pool_state_dict()

    restored = SharedPolicySelfPlayWrapper(
        _FakeMatchEnv(), team_size=2, opponent_pool_size=4
    )
    restored.load_opponent_pool_state_dict(payload, actor)

    assert restored.opponent_pool_iterations == [0, 20]
    for expected, actual in zip(source.opponent_pool, restored.opponent_pool):
        for key, tensor in expected.state_dict().items():
            torch.testing.assert_close(tensor, actual.state_dict()[key])


def test_local_role_reward_penalizes_passive_attacker_and_crowding_support():
    wrapper = SharedPolicySelfPlayWrapper.__new__(SharedPolicySelfPlayWrapper)
    wrapper.match_count = 1
    wrapper.team_size = 2
    wrapper.device = torch.device("cpu")

    root_states = torch.zeros(4, 13)
    root_states[:, 6] = 1.0
    root_states[:, 0] = torch.tensor([-0.5, -0.6, 2.0, 3.0])
    raw = SimpleNamespace(
        root_states=root_states,
        robot_actor_idxs_all=torch.tensor([[0, 1, 2, 3]]),
        object_pos_world_frame=torch.zeros(1, 3),
        object_lin_vel=torch.zeros(1, 3),
        dt=0.02,
    )
    wrapper.env = SimpleNamespace(
        env=raw,
        control_interval=10,
        cfg=SimpleNamespace(
            rewards=SimpleNamespace(
                high_level_dribble_skill_distance=1.0,
                high_level_dribble_target_ball_speed=1.0,
                high_level_skill_command_target_speed=0.8,
                high_level_support_min_ball_distance=1.15,
                high_level_local_attacker_ball_skill_scale=2.0,
                high_level_local_attacker_command_assist_scale=-4.0,
                high_level_local_role_conflict_scale=-3.0,
                high_level_local_support_ball_crowding_scale=-3.0,
            )
        ),
    )
    info = {
        "high_level_attacker_mask": torch.tensor([[True, False, False, False]]),
        "high_level_decision_robot_ball_distances": torch.tensor(
            [[0.5, 0.6, 2.0, 3.0]]
        ),
        "high_level_skill_ids": torch.tensor([[1, 0, 0, 0]]),
        "high_level_requested_skill_ids": torch.tensor([[1, 1, 0, 0]]),
        "high_level_invalid_skill_mask": torch.zeros(1, 4, dtype=torch.bool),
        "high_level_role_conflict_mask": torch.tensor(
            [[False, True, False, False]]
        ),
        "high_level_commands": torch.tensor(
            [[[0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        "elapsed_low_level_steps": torch.tensor([10]),
    }

    reward = wrapper._local_role_rewards(info)

    assert reward.shape == (1, 2)
    assert reward[0, 0] < 0.0  # command without robot/ball consequence
    assert reward[0, 1] < reward[0, 0]  # role conflict plus ball crowding
