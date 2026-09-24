"""Controlled evaluation interventions; does not modify training or checkpoints.

Pass ordinary play_high_level arguments after --mode. 'hold' preserves a
learner's shoot request and command for 0.8 seconds; 'aim' changes only the
shoot command to a 3 m/s goal-directed command. Geometry fallback still applies.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=['baseline', 'hold', 'aim', 'aim_hold'], default='baseline')
    parser.add_argument('--sample-skills', action='store_true', help='Sample categorical skills as in training; commands remain deterministic.')
    parser.add_argument('--hold-seconds', type=float, default=.8)
    diagnostic, remaining = parser.parse_known_args()
    from scripts import play_high_level as playback
    import torch

    context = {}
    probability_rows = []
    original_make = playback.make_env
    original_load = playback.load_high_level_policy

    def make_env(args, policies):
        env, raw = original_make(args, policies)
        context.update(env=env, raw=raw)
        original_step = env.step

        def step(action):
            result = original_step(action)
            if 'remaining' in context:
                context['remaining'][result[2].bool()] = 0
            context['active'] &= ~result[2].bool()
            return result

        env.step = step
        return env, raw

    def load(args):
        record = original_load(args)
        policy = record['policy']

        def act(obs):
            action = policy(obs).clone()
            if 'active' not in context:
                context['active'] = torch.ones(len(action), dtype=torch.bool, device=action.device)
            probabilities = action[:, :3].softmax(-1)
            raw = context['raw']
            near = raw._high_level_robot_ball_distances()[:, :int(raw.cfg.env.num_team_robots)].reshape(-1) <= .8
            mask = near & context['active']
            probability_rows.append({'active_near_rows': int(mask.sum()),
                'shoot_probability_sum': float(probabilities[mask, 2].sum()),
                'shoot_argmax_count': int(((action[:, :3].argmax(-1) == 2) & mask).sum())})
            if diagnostic.sample_skills:
                skill = torch.distributions.Categorical(probs=probabilities).sample()
                action[:, :3] = torch.nn.functional.one_hot(skill, 3).to(action.dtype)
            if diagnostic.mode == 'baseline':
                return action
            raw = context['raw']
            shoot = action[:, :3].argmax(-1) == 2
            if diagnostic.mode in ('aim', 'aim_hold'):
                goal = raw.env_origins[:, :2].clone()
                goal[:, 0] += float(raw.cfg.env.team_goal_x)
                direction = goal - raw.object_pos_world_frame[:, :2]
                direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                team = int(raw.cfg.env.num_team_robots)
                direction = direction[:, None, :].expand(-1, team, -1).reshape(-1, 2)
                scale = torch.tensor(raw.cfg.env.high_level_shoot_command_scale[:2], device=action.device)
                action[shoot, 3:5] = torch.atanh((3.0 * direction[shoot] / scale).clamp(-.95, .95))
            if diagnostic.mode in ('hold', 'aim_hold'):
                if 'remaining' not in context:
                    context['remaining'] = torch.zeros(len(action), dtype=torch.long, device=action.device)
                    context['cached'] = action.clone()
                left = context['remaining']
                keep = left > 0
                begin = shoot & ~keep
                action[keep] = context['cached'][keep]
                context['cached'][begin] = action[begin]
                left[keep] -= 1
                macro_dt = raw.dt * context['env'].control_interval
                left[begin] = max(0, math.ceil(diagnostic.hold_seconds / macro_dt - 1e-6) - 1)
            return action

        record['policy'] = act
        return record

    playback.make_env = make_env
    playback.load_high_level_policy = load
    args = playback.parse_args(remaining)
    print('Diagnostic intervention:', diagnostic.mode, flush=True)
    playback.run(args)
    Path(args.csv).with_suffix('.probabilities.json').write_text(json.dumps(probability_rows, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == '__main__':
    main()
