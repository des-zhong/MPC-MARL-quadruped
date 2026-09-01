import math
import types

import isaacgym
import pytest

assert isaacgym
torch = pytest.importorskip("torch")

from dribblebot.rewards.soccer_rewards import SoccerRewards


def yaw_quaternion(yaw):
    half = 0.5 * torch.as_tensor(yaw, dtype=torch.float)
    return torch.tensor([[0.0, 0.0, torch.sin(half), torch.cos(half)]])


def test_dribble_velocity_reward_compares_in_body_frame():
    env = types.SimpleNamespace(
        device=torch.device("cpu"),
        commands=torch.tensor([[1.0, 0.0, 0.0]]),
        object_lin_vel=torch.tensor([[0.0, 1.0, 0.0]]),
        base_quat=yaw_quaternion(math.pi / 2.0),
        cfg=types.SimpleNamespace(
            commands=types.SimpleNamespace(ball_xy_frame="body"),
            rewards=types.SimpleNamespace(
                dribbling_command_scale=[1.5, 1.5, 1.0],
                tracking_sigma=0.25,
            ),
        ),
    )

    reward = SoccerRewards(env)._reward_dribbling_ball_vel()

    assert torch.allclose(reward, torch.ones_like(reward), atol=1e-6)


def test_shooting_setup_rotates_body_command_for_world_geometry():
    env = types.SimpleNamespace(
        device=torch.device("cpu"),
        commands=torch.tensor([[1.0, 0.0, 0.0]]),
        base_quat=yaw_quaternion(math.pi / 2.0),
        base_pos=torch.tensor([[0.0, 0.5, 0.0]]),
        object_pos_world_frame=torch.tensor([[0.0, 1.0, 0.0]]),
        cfg=types.SimpleNamespace(
            commands=types.SimpleNamespace(ball_xy_frame="body"),
            rewards=types.SimpleNamespace(shooting_setup_distance=0.5),
        ),
    )

    command_direction, target_xy, setup_error = SoccerRewards(
        env
    )._shooting_setup_geometry()

    assert torch.allclose(command_direction, torch.tensor([[0.0, 1.0]]), atol=1e-6)
    assert torch.allclose(target_xy, torch.tensor([[0.0, 0.5]]), atol=1e-6)
    assert torch.allclose(setup_error, torch.zeros_like(setup_error), atol=1e-6)


def test_dribble_motion_penalty_follows_negative_body_command():
    env = types.SimpleNamespace(
        device=torch.device("cpu"),
        commands=torch.tensor([[-1.0, 0.0, 0.0]]),
        base_lin_vel=torch.tensor([[-0.5, 0.0, 0.0]]),
        base_quat=yaw_quaternion(0.0),
        cfg=types.SimpleNamespace(
            commands=types.SimpleNamespace(ball_xy_frame="body"),
            rewards=types.SimpleNamespace(dribbling_forward_speed_scale=1.0),
        ),
    )

    penalty = SoccerRewards(env)._reward_dribbling_backward_motion()

    assert torch.equal(penalty, torch.zeros_like(penalty))
