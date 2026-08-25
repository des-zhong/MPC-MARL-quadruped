"""CPU-only tests for high-level evaluation row formatting."""

import unittest

import isaacgym
import numpy as np
import torch

assert isaacgym

from scripts.play_high_level import row_from_step

class HighLevelRowTests(unittest.TestCase):
    def test_learning_team_local_rewards_are_padded_for_opponents(self):
        state = {
            "robot_xy": np.zeros((1, 4, 2), dtype=np.float32),
            "robot_vel": np.zeros((1, 4, 2), dtype=np.float32),
            "ball_xy": np.zeros((1, 2), dtype=np.float32),
            "ball_vel": np.zeros((1, 2), dtype=np.float32),
            "robot_ball_dist": np.zeros((1, 4), dtype=np.float32),
            "obstacle_xy": None,
        }
        info = {
            "high_level_local_role_rewards": np.array([[1.25, -0.5]], dtype=np.float32),
        }

        row = row_from_step(
            step=0,
            high_level_dt=0.1,
            state=state,
            action=torch.zeros((1, 6)),
            reward=torch.zeros(1),
            done=torch.zeros(1, dtype=torch.bool),
            info=info,
        )

        self.assertEqual(row["robot0_local_role_reward"], 1.25)
        self.assertEqual(row["robot1_local_role_reward"], -0.5)
        self.assertEqual(row["robot2_local_role_reward"], 0.0)
        self.assertEqual(row["robot3_local_role_reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
