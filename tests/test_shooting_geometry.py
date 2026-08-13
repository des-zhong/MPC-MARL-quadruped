import unittest

import torch

from dribblebot.rewards.shooting_geometry import (
    shooting_forward_velocity_score,
    shooting_setup_geometry,
    shooting_setup_progress,
)


class ShootingGeometryTest(unittest.TestCase):
    def test_target_is_behind_ball_in_command_frame(self):
        base_xy = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        ball_xy = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
        command_xy = torch.tensor([[3.0, 0.0], [0.0, 2.0]])

        command_direction, target_xy, setup_error = shooting_setup_geometry(
            base_xy,
            ball_xy,
            command_xy,
            setup_distance=0.45,
        )

        torch.testing.assert_close(
            command_direction,
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )
        torch.testing.assert_close(
            target_xy,
            torch.tensor([[0.55, 0.0], [1.0, 1.55]]),
        )
        torch.testing.assert_close(setup_error, torch.tensor([0.55, 0.55]))

    def test_progress_is_signed(self):
        ball_xy = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        command_xy = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        previous_base_xy = torch.tensor([[0.0, 0.0], [0.4, 0.0]])
        base_xy = torch.tensor([[0.2, 0.0], [0.2, 0.0]])

        progress = shooting_setup_progress(
            previous_base_xy,
            base_xy,
            ball_xy,
            command_xy,
            setup_distance=0.45,
            dt=0.1,
            speed_scale=1.0,
        )

        torch.testing.assert_close(progress, torch.tensor([1.0, -1.0]))

    def test_forward_velocity_score_credits_partial_aligned_strikes(self):
        command = torch.tensor(
            [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [0.1, 0.0]]
        )
        ball_velocity = torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.1, 0.0]]
        )

        score = shooting_forward_velocity_score(
            ball_velocity,
            command,
            min_command_speed=0.2,
        )

        torch.testing.assert_close(score, torch.tensor([0.5, 1.0, 0.0, 0.0, 0.0]))

    def test_forward_velocity_score_softly_penalizes_oblique_strikes(self):
        command = torch.tensor([[2.0, 0.0]])
        ball_velocity = torch.tensor([[1.0, 1.0]])

        score = shooting_forward_velocity_score(ball_velocity, command)

        expected = torch.tensor([0.5 / (2.0 ** 0.5)])
        torch.testing.assert_close(score, expected)


if __name__ == "__main__":
    unittest.main()
