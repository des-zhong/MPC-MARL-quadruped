from types import SimpleNamespace

import torch

from scripts.train_high_level_online_mpc import (
    OnlineMPCSelfPlayExtension,
    WorldModelReplayBuffer,
    build_arg_parser,
)


def _add_opponent_rows(replay, opponent, count):
    replay.add_batch({
        "value": torch.arange(count, dtype=torch.float),
        "opponent_snapshot_iteration": torch.full(
            (count,), opponent, dtype=torch.long
        ),
    })


def test_replay_retains_and_balances_historical_opponents():
    replay = WorldModelReplayBuffer(
        capacity=12, recent_fraction=0.5, recent_window=4, seed=7
    )
    _add_opponent_rows(replay, 0, 12)
    _add_opponent_rows(replay, 10, 12)
    _add_opponent_rows(replay, 20, 12)

    assert len(replay) == 12
    assert set(replay.opponent_counts()) == {0, 10, 20}
    assert all(count > 0 for count in replay.opponent_counts().values())

    sampled = replay.sample(60)["opponent_snapshot_iteration"]
    counts = torch.bincount(sampled // 10, minlength=3)
    # Half of each batch is the latest opponent; the historical half is
    # balanced across the two older opponent snapshots.
    torch.testing.assert_close(counts, torch.tensor([15, 15, 30]))


def test_replay_checkpoint_round_trip_preserves_opponents():
    replay = WorldModelReplayBuffer(capacity=8, seed=11)
    _add_opponent_rows(replay, 0, 8)
    _add_opponent_rows(replay, 40, 8)

    restored = WorldModelReplayBuffer(capacity=8, seed=11)
    restored.load_state_dict(replay.state_dict())

    assert restored.opponent_counts() == replay.opponent_counts()
    assert len(restored) == len(replay)


def test_online_defaults_delay_mpc_and_rotate_opponents():
    args = build_arg_parser().parse_args([])

    assert args.world_model_checkpoint is None
    assert args.auto_resume_online_replay
    assert args.mpc_warmup_steps == 2400
    assert args.mpc_min_replay_size == 20_000
    assert args.mpc_min_world_model_updates == 48
    assert args.self_play_update_interval == 400


def test_online_replay_auto_resume_can_be_disabled_for_scratch_models():
    args = build_arg_parser().parse_args(["--no-auto-resume-online-replay"])

    assert not args.auto_resume_online_replay


class _ConstantValue:
    def predict(self, states):
        return torch.full((states.shape[0],), 10.0, device=states.device)


def _return_extension(ready=True):
    extension = OnlineMPCSelfPlayExtension.__new__(OnlineMPCSelfPlayExtension)
    extension.args = SimpleNamespace(
        terminal_bootstrap_on_timeout=True,
        terminal_timeout_tail_exclusion=1,
    )
    extension.terminal_trainer = SimpleNamespace(
        gamma=0.5,
        ready=ready,
        device=torch.device("cpu"),
        model=_ConstantValue(),
    )
    extension._episode_rewards = [[1.0, 2.0]]
    extension._episode_terminated = [[False, False]]
    extension._episode_truncated = [[False, True]]
    return extension


def test_timeout_return_bootstraps_from_captured_next_state():
    extension = _return_extension(ready=True)

    returns = extension._timeout_return_target(
        0, torch.zeros(1, 3), match_done=True, timeout=True
    )

    torch.testing.assert_close(
        torch.as_tensor(returns), torch.tensor([4.5, 7.0])
    )


def test_unready_timeout_target_excludes_unbootstrapped_tail():
    extension = _return_extension(ready=False)

    returns = extension._timeout_return_target(
        0, torch.zeros(1, 3), match_done=True, timeout=True
    )

    torch.testing.assert_close(torch.as_tensor(returns), torch.tensor([2.0]))
