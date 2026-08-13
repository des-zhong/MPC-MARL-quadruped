import unittest

import torch

from dribblebot.rewards.dribbling_geometry import (
    dribbling_backward_motion_penalty,
    dribbling_setup_score,
)


class DribblingGeometryTest(unittest.TestCase):
    def test_setup_prefers_ball_ahead_over_under_chassis(self):
        ball_local = torch.tensor([[0.35, 0.0], [0.0, 0.0], [-0.20, 0.0]])

        score = dribbling_setup_score(ball_local, target_forward=0.35)

        self.assertAlmostEqual(float(score[0]), 1.0, places=6)
        self.assertGreater(float(score[0]), float(score[1]))
        self.assertGreater(float(score[1]), float(score[2]))

    def test_backward_motion_is_negative_and_forward_motion_is_unpenalized(self):
        velocity = torch.tensor([[1.0, 0.0], [-0.5, 0.0], [0.0, 1.0]])
        forward = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        penalty = dribbling_backward_motion_penalty(velocity, forward)

        torch.testing.assert_close(penalty, torch.tensor([0.0, -0.5, 0.0]))


if __name__ == "__main__":
    unittest.main()
