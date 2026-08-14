import torch

from dribblebot.envs.wrappers.team_frame import (
    mirror_high_level_commands,
    mirror_high_level_policy_actions,
)


def test_mirror_commands_preserves_walk_and_rotates_ball_skills():
    commands = torch.tensor(
        [
            [0.8, -0.3, 0.4],
            [1.1, -0.7, 0.2],
            [-1.4, 0.5, 0.0],
        ]
    )
    skills = torch.tensor([0, 1, 2])

    mirrored = mirror_high_level_commands(commands, skills)

    torch.testing.assert_close(mirrored[0], commands[0])
    torch.testing.assert_close(mirrored[1], torch.tensor([-1.1, 0.7, 0.2]))
    torch.testing.assert_close(mirrored[2], torch.tensor([1.4, -0.5, 0.0]))
    torch.testing.assert_close(mirror_high_level_commands(mirrored, skills), commands)


def test_mirror_policy_actions_uses_selected_skill_and_preserves_logits_and_yaw():
    actions = torch.tensor(
        [
            [4.0, 1.0, 0.0, 0.8, -0.3, 0.4],
            [0.0, 4.0, 1.0, 1.1, -0.7, 0.2],
            [0.0, 1.0, 4.0, -1.4, 0.5, 0.0],
        ]
    )

    mirrored = mirror_high_level_policy_actions(actions)

    torch.testing.assert_close(mirrored[:, :3], actions[:, :3])
    torch.testing.assert_close(mirrored[0, 3:], actions[0, 3:])
    torch.testing.assert_close(mirrored[1, 3:], torch.tensor([-1.1, 0.7, 0.2]))
    torch.testing.assert_close(mirrored[2, 3:], torch.tensor([1.4, -0.5, 0.0]))
    torch.testing.assert_close(mirror_high_level_policy_actions(mirrored), actions)
