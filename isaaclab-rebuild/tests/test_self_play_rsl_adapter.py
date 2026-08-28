"""Simulator-free shape contract for the self-play RSL-RL adapter."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
    / "tasks"
    / "manager_based"
    / "football"
    / "rsl.py"
)
SPEC = importlib.util.spec_from_file_location("dribblebot_self_play_rsl_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _FakeSelfPlay:
    num_envs = 4
    num_actions = 6
    num_train_envs = 4
    match_count = 2
    team_size = 2
    device = torch.device("cpu")
    max_episode_length = 30
    cfg = SimpleNamespace(is_finite_horizon=False)

    def __init__(self):
        self._length = torch.zeros(2, dtype=torch.long)
        self.opponent_policy_callable = None
        self.snapshot_iterations = []

    @property
    def episode_length_buf(self):
        return self._length.repeat_interleave(self.team_size)

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self._length[:] = value.reshape(self.match_count, self.team_size)[:, 0]

    def _observation(self):
        return {
            "obs": torch.zeros(4, 34),
            "privileged_obs": torch.ones(4, 34),
            "obs_history": torch.zeros(4, 136),
        }

    def reset(self, **kwargs):
        return self._observation(), {}

    def get_observations(self):
        return self._observation()

    def step(self, action):
        return (
            self._observation(),
            torch.ones(4),
            torch.zeros(4, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
            {},
        )

    def close(self):
        return None

    def update_opponent_snapshot(self, policy, iteration, force=False):
        self.opponent_policy_callable = policy
        self.snapshot_iterations.append(int(iteration))
        return True

    def set_opponent_callable(self, policy):
        self.opponent_policy_callable = policy

    def load_opponent_checkpoint(self, *args, **kwargs):
        raise NotImplementedError


def test_adapter_expands_match_agents_and_preserves_history_contract() -> None:
    wrapper = MODULE.MatchSelfPlayRslRlVecEnvWrapper(_FakeSelfPlay(), clip_actions=10.0)
    observations, _ = wrapper.reset()
    assert observations.batch_size == torch.Size([4])
    assert observations["policy"].shape == (4, 136)
    _, reward, done, extras = wrapper.step(torch.zeros(4, 6))
    assert reward.shape == (4,)
    assert done.dtype == torch.long
    assert extras["time_outs"].shape == (4,)


def test_adapter_sets_one_episode_length_per_match() -> None:
    wrapper = MODULE.MatchSelfPlayRslRlVecEnvWrapper(_FakeSelfPlay())
    wrapper.episode_length_buf = torch.tensor([3, 3, 5, 5])
    assert wrapper.episode_length_buf.tolist() == [3, 3, 5, 5]


class _FakeActor(torch.nn.Module):
    def forward(self, observations, stochastic_output=False):
        return torch.zeros(observations["policy"].shape[0], 6, device=observations["policy"].device)


def test_frozen_rsl_opponent_preserves_callable_contract() -> None:
    policy = MODULE.FrozenRslRlOpponentPolicy(_FakeActor(), device="cpu")
    actions = policy(
        {
            "obs_history": torch.zeros(4, 136),
            "privileged_obs": torch.zeros(4, 34),
        }
    )
    assert actions.shape == (4, 6)
    assert torch.isfinite(actions).all()


def test_frozen_rsl_opponent_ignores_and_restores_cached_distribution() -> None:
    actor = _FakeActor()
    cached_distribution = torch.distributions.Normal(
        torch.zeros(4, 6, requires_grad=True) * 2.0,
        torch.ones(4, 6),
    )
    actor.distribution = cached_distribution

    policy = MODULE.FrozenRslRlOpponentPolicy(actor, device="cpu")

    assert actor.distribution is cached_distribution
    assert policy.actor.distribution is None


def test_opponent_snapshot_schedule_wraps_only_optimizer_updates() -> None:
    raw = _FakeSelfPlay()
    env = MODULE.MatchSelfPlayRslRlVecEnvWrapper(raw)

    class _Algorithm:
        def __init__(self):
            self.calls = 0

        def update(self):
            self.calls += 1
            return {"loss": 0.0}

        def get_policy(self):
            return _FakeActor()

    runner = SimpleNamespace(current_learning_iteration=0, alg=_Algorithm())
    MODULE.install_opponent_snapshot_schedule(runner, env, interval=2, policy_device="cpu")
    runner.alg.update()
    runner.alg.update()
    runner.alg.update()
    assert runner.alg.calls == 3
    assert raw.snapshot_iterations == [0, 2]


def test_opponent_snapshot_schedule_continues_from_resumed_iteration() -> None:
    raw = _FakeSelfPlay()
    env = MODULE.MatchSelfPlayRslRlVecEnvWrapper(raw)

    class _Algorithm:
        def update(self):
            return {"loss": 0.0}

        def get_policy(self):
            return _FakeActor()

    runner = SimpleNamespace(current_learning_iteration=498, alg=_Algorithm())
    MODULE.install_opponent_snapshot_schedule(
        runner,
        env,
        interval=500,
        policy_device="cpu",
        initial_snapshot=False,
    )
    runner.alg.update()
    runner.alg.update()
    assert raw.snapshot_iterations == [500]


def test_opponent_snapshot_schedule_initializes_resumed_actor_at_current_iteration() -> None:
    raw = _FakeSelfPlay()
    env = MODULE.MatchSelfPlayRslRlVecEnvWrapper(raw)

    class _Algorithm:
        def update(self):
            return {"loss": 0.0}

        def get_policy(self):
            return _FakeActor()

    runner = SimpleNamespace(current_learning_iteration=500, alg=_Algorithm())
    MODULE.install_opponent_snapshot_schedule(runner, env, interval=500, policy_device="cpu")
    assert raw.snapshot_iterations == [500]
