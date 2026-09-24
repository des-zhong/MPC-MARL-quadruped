"""Reproduce a controlled backward-dribble reward discrepancy without a GPU."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from quadruped.world_model.schema import default_state_schema, EVENT_NAMES
    from quadruped.world_model.action_adapter import JointActionAdapter, SkillBounds
    from quadruped.mpc.config import MPCConfig
    from quadruped.mpc.analytical_reward import AnalyticalRewardReconstructor
    from quadruped.rewards.high_level_rewards import HighLevelRewards

    schema = default_state_schema(0, 2)
    adapter = JointActionAdapter({i: SkillBounds((-3., -3., -1.), (3., 3., 1.), (1., 1., 1.))
                                  for i in range(3)}, num_robots=2)
    state = torch.zeros(1, 2, schema.state_dim)
    state[:, :, schema.slice('field.geometry')] = torch.tensor([4., 2.5, -4., 4., 1., .0889])
    state[:, :, schema.slice('robot_0.position')] = torch.tensor([-.1, 0., .3])
    state[:, :, schema.slice('robot_1.position')] = torch.tensor([.7, .6, .3])
    for robot in range(2):
        state[:, :, schema.slice(f'robot_{robot}.yaw_sin_cos')] = torch.tensor([0., 1.])
    state[:, 1, schema.slice('ball.linear_velocity')] = torch.tensor([-1., 0., 0.])
    skills = torch.tensor([[1, 0]])
    commands = torch.tensor([[[-1., 0., 0.], [0., 0., 0.]]])
    actions = adapter.pack(skills[:, None], commands[:, None])
    config = MPCConfig()
    imagined = AnalyticalRewardReconstructor(schema, adapter, EVENT_NAMES, config, 1,
                                            {'high_level_dribble_ball_control': 1.})
    _, components = imagined(state, actions, torch.zeros(1, 1, len(EVENT_NAMES)),
                             torch.zeros(1, 1, 2, dtype=torch.bool))
    roots = torch.zeros(2, 13)
    roots[:, :3] = torch.tensor([[-.4, 0., .3], [2.8, 1.5, .3]])
    env = NS(num_envs=1, num_robots=2, device='cpu', root_states=roots,
             robot_actor_idxs_all=torch.tensor([[0, 1]]), env_origins=torch.zeros(1, 3),
             object_pos_world_frame=torch.zeros(1, 3), object_lin_vel=torch.tensor([[-1., 0., 0.]]),
             high_level_skill_ids=skills, high_level_requested_skill_ids=skills,
             high_level_invalid_skill_mask=torch.zeros_like(skills), high_level_commands=commands,
             cfg=NS(env=NS(num_team_robots=1, team_goal_x=4.), rewards=NS()))
    real_rate = HighLevelRewards(env)._reward_high_level_dribble_ball_control().item()
    result = {'scenario': 'Controlled ball moves backward at 1 m/s, following a backward dribble command',
              'real_reward_rate': real_rate, 'real_reward_for_0.2s_constant_state': real_rate * .2,
              'mpc_reward_for_0.2s': components['dribble_ball_control'].item()}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
