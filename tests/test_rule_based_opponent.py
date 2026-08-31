from types import SimpleNamespace

import isaacgym
import torch

from dribblebot.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
from dribblebot.envs.wrappers.rule_based_opponent import RuleBasedOpponent


def _make_controller(robot_xy, ball_xy=(0.0, 0.0), **kwargs):
    robot_xy = torch.as_tensor(robot_xy, dtype=torch.float)
    roots = torch.zeros(1, robot_xy.shape[0], 13)
    roots[0, :, :2] = robot_xy
    roots[0, :, 6] = 1.0
    cfg = SimpleNamespace(
        env=SimpleNamespace(
            team_goal_x=4.0,
            high_level_walk_command_scale=[1.5, 1.5, 1.0],
            high_level_dribble_command_scale=[1.5, 1.5, 1.0],
            high_level_shoot_command_scale=[3.0, 3.0, 0.0],
        )
    )
    raw = SimpleNamespace(
        cfg=cfg,
        object_pos_world_frame=torch.tensor([[*ball_xy, 0.0]]),
        env_origins=torch.zeros(1, 3),
    )
    match = SimpleNamespace(env=raw)
    wrapper = SimpleNamespace(
        env=match,
        team_size=2,
        device=torch.device("cpu"),
        _roots=lambda: roots,
    )
    return RuleBasedOpponent(wrapper, **kwargs), cfg.env


def _decode(actions, cfg):
    skills = actions[..., :3].argmax(dim=-1)
    scale_table = torch.tensor(
        [
            cfg.high_level_walk_command_scale,
            cfg.high_level_dribble_command_scale,
            cfg.high_level_shoot_command_scale,
        ]
    )
    commands = torch.tanh(actions[..., 3:]) * scale_table[skills]
    return skills, commands


def test_nearest_opponent_shoots_at_negative_x_goal_and_other_blocks():
    controller, cfg = _make_controller(
        [(-2.0, 1.0), (-1.0, 0.0), (0.5, 0.0), (2.0, 1.0)],
        collision_distance=0.2,
    )

    skills, commands = _decode(controller.wrapper_actions(), cfg)

    assert skills.tolist() == [[2, 0]]
    assert commands[0, 0, 0] < 0.0  # opponent goal is at -x
    assert commands[0, 1, 0] < 0.0  # blocker moves onto learner-ball segment
    assert commands[0, 1, 1] < 0.0


def test_attacker_dribbles_when_close_but_outside_shoot_distance():
    controller, cfg = _make_controller(
        [(-2.0, 1.0), (-1.0, 0.0), (0.8, 0.0), (2.0, 1.0)],
        shoot_distance=0.5,
        dribble_distance=1.0,
        collision_distance=0.2,
    )

    skills, commands = _decode(controller(), cfg)

    assert skills[0, 0].item() == 1
    assert commands[0, 0, 0] < 0.0


def test_attacker_walks_away_inside_collision_clearance():
    controller, cfg = _make_controller(
        [(0.25, 0.0), (-2.0, 1.0), (0.4, 0.0), (2.0, 1.0)],
        collision_distance=0.3,
    )

    skills, commands = _decode(controller(), cfg)

    assert skills[0, 0].item() == 0
    # The learner is to the attacker's -x side, so the escape is +x.
    assert commands[0, 0, 0] > 0.0


def test_rule_provider_walk_command_is_not_replaced_by_support_fallback():
    wrapper = HighLevelSkillWrapper.__new__(HighLevelSkillWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 1
    wrapper.num_robots = 2
    wrapper.requested_skill_ids = torch.zeros(1, 2, dtype=torch.long)
    wrapper.skill_ids = torch.zeros(1, 2, dtype=torch.long)
    wrapper.skill_transition_mask = torch.zeros(1, 2, dtype=torch.bool)
    wrapper.invalid_skill_mask = torch.zeros(1, 2, dtype=torch.bool)
    wrapper.collision_avoidance_mask = torch.zeros(1, 2, dtype=torch.bool)
    wrapper.attacker_command_assist_mask = torch.zeros(1, 2, dtype=torch.bool)
    wrapper.role_conflict_mask = torch.zeros(1, 2, dtype=torch.bool)
    wrapper.skill_commands = torch.zeros(1, 2, 3)
    wrapper.decision_robot_ball_distances = torch.zeros(1, 2)
    wrapper.preserve_external_high_level_actions = torch.tensor([[False, True]])
    wrapper.env = SimpleNamespace(
        cfg=SimpleNamespace(
            env=SimpleNamespace(
                high_level_use_geometric_skill_fallback=True,
                high_level_role_aware_fallback=True,
                high_level_collision_avoidance=False,
                high_level_action_input_clip=10.0,
                high_level_walk_command_scale=[1.0, 1.0, 1.0],
                high_level_dribble_command_scale=[1.0, 1.0, 1.0],
                high_level_shoot_command_scale=[1.0, 1.0, 0.0],
                high_level_command_obs_scale=[1.0, 1.0, 1.0],
            ),
            rewards=SimpleNamespace(high_level_skill_command_min_speed=0.2),
        ),
        high_level_requested_skill_ids=torch.zeros(1, 2, dtype=torch.long),
        high_level_skill_ids=torch.zeros(1, 2, dtype=torch.long),
        high_level_invalid_skill_mask=torch.zeros(1, 2, dtype=torch.bool),
        high_level_commands=torch.zeros(1, 2, 3),
    )
    wrapper._skill_affordances = lambda: {
        "distance": torch.ones(1, 2),
        "can_dribble": torch.zeros(1, 2, dtype=torch.bool),
        "can_shoot": torch.zeros(1, 2, dtype=torch.bool),
    }
    wrapper._attacker_mask = lambda affordances: torch.tensor([[True, False]])
    wrapper._walk_to_ball_commands = lambda affordances: torch.zeros(1, 2, 3)
    wrapper._walk_support_commands = lambda attacker_mask: torch.zeros(1, 2, 3)

    action = torch.zeros(1, 2, 6)
    action[0, :, 0] = 10.0
    action[0, 1, 3] = 0.4
    wrapper._decode_action(action)

    torch.testing.assert_close(
        wrapper.skill_commands[0, 1, 0], torch.tanh(torch.tensor(0.4))
    )
