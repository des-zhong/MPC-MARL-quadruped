import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc.opponent_forecast import SlowWalkToBallOpponentForecaster
from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds


def _adapter():
    return JointActionAdapter(
        {
            0: SkillBounds((-1.0, -1.0, -0.8), (1.0, 1.0, 0.8), (1, 1, 1)),
            1: SkillBounds((-1.5, -1.5, -1.0), (1.5, 1.5, 1.0), (1, 1, 1)),
            2: SkillBounds((-3.0, -3.0, 0.0), (3.0, 3.0, 0.0), (1, 1, 0)),
        },
        num_robots=4,
    )


class _Environment:
    num_envs = 2
    device = torch.device("cpu")

    def _skill_affordances(self):
        # The learning team occupies slots 0 and 1. Opponent slot 2 is far
        # enough to walk at full speed; slot 3 is already inside stop range.
        local_ball = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [0.1, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 2.0], [-2.0, 0.0]],
            ]
        )
        return {"local_ball_xy": local_ball}


def test_simple_opponent_walks_slowly_toward_ball_and_stops_when_close():
    adapter = _adapter()
    opponent = SlowWalkToBallOpponentForecaster(
        _Environment(),
        team_size=2,
        action_adapter=adapter,
        speed_mps=0.4,
        stop_distance_m=0.2,
        slow_distance_m=1.0,
        yaw_gain=1.0,
    )

    fixed, mask = opponent.fixed_action_sequence(horizon=3)
    skills, commands = adapter.unpack(fixed)

    assert fixed.shape == (2, 3, 16)
    assert mask.tolist() == [False, False, True, True]
    assert torch.all(skills[..., 2:] == 0)
    torch.testing.assert_close(commands[0, 0, 2, :2], torch.tensor([0.4, 0.0]))
    torch.testing.assert_close(commands[0, 0, 3], torch.zeros(3))
    torch.testing.assert_close(commands[1, 0, 2, :2], torch.tensor([0.0, 0.4]))
    torch.testing.assert_close(commands[1, 0, 3, :2], torch.tensor([-0.4, 0.0]))
    assert torch.linalg.vector_norm(commands[..., 2:, :2], dim=-1).max() <= 0.4
    torch.testing.assert_close(fixed[:, 0], opponent.current_joint_action())


def test_wrapper_actions_encode_the_same_simple_opponent_command():
    adapter = _adapter()
    opponent = SlowWalkToBallOpponentForecaster(
        _Environment(), team_size=2, action_adapter=adapter
    )

    raw = opponent.wrapper_actions()
    assert raw.shape == (2, 2, 6)
    assert torch.all(raw[..., :3].argmax(dim=-1) == 0)
