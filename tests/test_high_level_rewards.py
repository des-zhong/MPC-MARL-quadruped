"""CPU-only regression tests for AS2 high-level soccer rewards."""

import unittest
from types import SimpleNamespace

import torch

from dribblebot.rewards.high_level_rewards import HighLevelRewards
from scripts.train_high_level import HIGH_LEVEL_REWARD_SCALES


def _fake_env(num_envs):
    robot0 = torch.arange(num_envs, dtype=torch.long)
    robot1 = robot0 + num_envs
    root_states = torch.zeros(2 * num_envs, 13)
    root_states[robot0, 0] = -0.5
    root_states[robot1, 0] = -2.0
    root_states[:, 6] = 1.0
    return SimpleNamespace(
        num_envs=num_envs,
        num_robots=2,
        device=torch.device("cpu"),
        dt=0.02,
        robot_actor_idxs=robot0,
        other_robot_actor_idxs=robot1,
        root_states=root_states,
        object_pos_world_frame=torch.zeros(num_envs, 3),
        prev_object_pos_world_frame=torch.zeros(num_envs, 3),
        object_lin_vel=torch.zeros(num_envs, 3),
        prev_object_lin_vel=torch.zeros(num_envs, 3),
        env_origins=torch.zeros(num_envs, 3),
        high_level_skill_ids=torch.zeros(num_envs, 2, dtype=torch.long),
        high_level_requested_skill_ids=torch.zeros(num_envs, 2, dtype=torch.long),
        high_level_invalid_skill_mask=torch.zeros(num_envs, 2, dtype=torch.bool),
        high_level_commands=torch.zeros(num_envs, 2, 3),
        prev_high_level_robot_ball_distances=torch.full((num_envs, 2), 0.5),
        cfg=SimpleNamespace(
            env=SimpleNamespace(team_goal_x=4.0),
            rewards=SimpleNamespace(
                high_level_dribble_skill_distance=1.0,
                high_level_dribble_control_distance=0.8,
                high_level_skill_command_min_speed=0.2,
                high_level_dribble_min_ball_speed=0.1,
                high_level_dribble_target_ball_speed=1.0,
                high_level_approach_walk_speed=0.9,
                high_level_goal_facing_target_speed=0.5,
                high_level_shoot_skill_distance=0.75,
                high_level_shoot_min_ball_speed=0.8,
                high_level_shoot_min_delta_speed=0.25,
                high_level_shoot_target_delta_speed=1.5,
                high_level_shoot_min_command_alignment=0.6,
            ),
        ),
    )


class HighLevelRewardTests(unittest.TestCase):
    def test_nonzero_configured_terms_have_implementations(self):
        expected = {
            "high_level_goal": 500.0,
            "high_level_accidental_termination": -200.0,
            "high_level_ball_goal_progress": 2.0,
            "high_level_robot_collision": -10.0,
            "high_level_pass": 2.0,
            "high_level_invalid_skill": -3.0,
            "high_level_approach_ball": 1.0,
            "high_level_walk_command_alignment": 0.5,
            "high_level_face_ball_while_approaching": 0.5,
            "high_level_face_goal_while_moving": 0.75,
            "high_level_dribble_ball_control": 2.0,
            "high_level_shoot_launch": 10.0,
        }
        self.assertEqual(HIGH_LEVEL_REWARD_SCALES, expected)
        for reward_name in HIGH_LEVEL_REWARD_SCALES:
            self.assertTrue(
                hasattr(HighLevelRewards, "_reward_" + reward_name),
                reward_name,
            )

    def test_robot_collision_penalizes_only_pairs_inside_clearance(self):
        env = _fake_env(3)
        env.cfg.rewards.high_level_robot_collision_distance = 0.75
        env.root_states[env.robot_actor_idxs, :2] = 0.0
        env.root_states[env.other_robot_actor_idxs, 0] = torch.tensor(
            [0.75, 0.375, 0.0]
        )
        env.root_states[env.other_robot_actor_idxs, 1] = 0.0

        penalty = HighLevelRewards(env)._reward_high_level_robot_collision()

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.0, 0.25, 1.0])))

    def test_robot_collision_ignores_frozen_opponent_only_pairs(self):
        env = _fake_env(1)
        env.num_robots = 4
        env.cfg.env.num_team_robots = 2
        env.cfg.rewards.high_level_robot_collision_distance = 0.75
        env.robot_actor_idxs_all = torch.tensor([[0, 1, 2, 3]])
        env.root_states = torch.zeros(4, 13)
        env.root_states[:, 6] = 1.0
        env.root_states[:, 0] = torch.tensor([-3.0, -2.0, 2.0, 2.0])

        reward = HighLevelRewards(env)
        self.assertEqual(float(reward._reward_high_level_robot_collision()), 0.0)

        env.root_states[2, 0] = -2.0
        self.assertEqual(float(reward._reward_high_level_robot_collision()), 1.0)

    def test_dribble_reward_requires_valid_controlled_ball_motion(self):
        env = _fake_env(5)
        env.high_level_skill_ids[:, 0] = torch.tensor([1, 1, 0, 1, 1])
        env.high_level_requested_skill_ids[:, 0] = torch.tensor([1, 1, 0, 2, 1])
        env.high_level_invalid_skill_mask[3, 0] = True
        env.high_level_commands[:, 0, 0] = 1.0
        env.high_level_commands[4, 0, :2] = torch.tensor([0.0, 1.0])
        env.object_lin_vel[:, :2] = torch.tensor(
            [
                [1.0, 0.0],  # valid goal-directed dribble
                [0.0, 0.0],  # selected dribble but no physical consequence
                [1.0, 0.0],  # identical motion under walk
                [1.0, 0.0],  # invalid shoot request fell back to dribble
                [0.0, 1.0],  # controlled lateral dribble
            ]
        )

        reward = HighLevelRewards(env)._reward_high_level_dribble_ball_control()

        self.assertGreater(float(reward[0]), float(reward[4]))
        self.assertGreater(float(reward[4]), 0.0)
        self.assertEqual(float(reward[1]), 0.0)
        self.assertEqual(float(reward[2]), 0.0)
        self.assertEqual(float(reward[3]), 0.0)

    def test_shoot_reward_requires_new_aligned_ball_launch(self):
        env = _fake_env(6)
        env.high_level_skill_ids[:, 0] = torch.tensor([2, 2, 0, 2, 2, 2])
        env.high_level_requested_skill_ids[:, 0] = torch.tensor([2, 2, 0, 2, 2, 2])
        env.high_level_invalid_skill_mask[3, 0] = True
        env.high_level_commands[:, 0, 0] = 2.0
        env.object_lin_vel[:, :2] = torch.tensor(
            [
                [2.0, 0.0],  # valid new launch
                [2.0, 0.0],  # ball was already moving
                [2.0, 0.0],  # identical launch under walk
                [2.0, 0.0],  # invalid request
                [0.0, 2.0],  # launch is not aligned with the command
                [0.0, 0.0],  # no launch
            ]
        )
        env.prev_object_lin_vel[1, 0] = 2.0

        reward = HighLevelRewards(env)._reward_high_level_shoot_launch()

        self.assertGreater(float(reward[0]), 0.0)
        self.assertTrue(torch.equal(reward[1:], torch.zeros(5)))

    def test_approach_uses_signed_progress_of_previously_closest_walking_robot(self):
        env = _fake_env(4)
        env.root_states[env.robot_actor_idxs, 0] = torch.tensor([-1.9, -2.1, -1.9, -0.5])
        env.root_states[env.other_robot_actor_idxs, 0] = torch.tensor([-3.0, -2.5, -3.0, -3.0])
        env.prev_high_level_robot_ball_distances = torch.tensor(
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [0.6, 3.0]]
        )
        env.high_level_requested_skill_ids[:, 0] = torch.tensor([0, 0, 1, 0])
        env.high_level_invalid_skill_mask[2, 0] = True

        reward = HighLevelRewards(env)._reward_high_level_approach_ball()

        self.assertTrue(torch.equal(reward, torch.tensor([1.0, -1.0, 0.0, 0.0])))

    def test_walk_command_alignment_penalizes_commands_away_from_ball(self):
        env = _fake_env(4)
        env.root_states[env.robot_actor_idxs, 0] = -2.0
        env.root_states[env.other_robot_actor_idxs, 0] = -3.0
        env.prev_high_level_robot_ball_distances = torch.tensor(
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [0.6, 3.0]]
        )
        env.high_level_commands[:, 0, 0] = torch.tensor([0.9, -0.9, 0.9, 0.9])
        env.high_level_requested_skill_ids[2, 0] = 2
        env.high_level_invalid_skill_mask[2, 0] = True

        reward = HighLevelRewards(env)._reward_high_level_walk_command_alignment()

        self.assertTrue(torch.allclose(reward, torch.tensor([1.0, -1.0, 0.0, 0.0])))

    def test_face_ball_reward_requires_an_active_far_ball_approach(self):
        env = _fake_env(5)
        env.root_states[env.robot_actor_idxs, 0] = -2.0
        env.root_states[env.other_robot_actor_idxs, 0] = -3.0
        env.prev_high_level_robot_ball_distances = torch.tensor(
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [0.6, 3.0]]
        )
        # Identity faces +x (toward the ball); a pi-yaw quaternion faces -x.
        env.root_states[env.robot_actor_idxs[1], 5:7] = torch.tensor([1.0, 0.0])
        env.high_level_commands[[0, 1, 3], 0, 0] = 0.9
        env.high_level_requested_skill_ids[3, 0] = 1
        env.high_level_skill_ids[3, 0] = 1

        reward = HighLevelRewards(env)._reward_high_level_face_ball_while_approaching()

        self.assertTrue(torch.allclose(reward, torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0])))

    def test_face_goal_reward_requires_goalward_body_forward_motion(self):
        env = _fake_env(5)
        env.root_states[env.robot_actor_idxs, 0] = -2.0
        env.root_states[env.other_robot_actor_idxs, 0] = -3.0
        env.prev_high_level_robot_ball_distances = torch.tensor(
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]
        )
        # Identity faces the +x opponent goal; a pi-yaw faces away from it.
        env.root_states[env.robot_actor_idxs[1], 5:7] = torch.tensor([1.0, 0.0])
        env.root_states[env.robot_actor_idxs[[0, 1, 3]], 7] = 0.5
        env.root_states[env.robot_actor_idxs[4], 7] = -0.5
        env.high_level_requested_skill_ids[3, 0] = 1
        env.high_level_skill_ids[3, 0] = 0
        env.high_level_invalid_skill_mask[3, 0] = True

        reward = HighLevelRewards(env)._reward_high_level_face_goal_while_moving()

        self.assertTrue(
            torch.allclose(reward, torch.tensor([1.0, -1.0, 0.0, 0.0, -1.0]))
        )


if __name__ == "__main__":
    unittest.main()
