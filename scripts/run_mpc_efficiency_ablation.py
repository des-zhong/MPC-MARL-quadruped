"""Launch matched MAPPO/MPC experiments; print commands unless --execute is given."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(condition, seed, iterations, num_envs, device, python=sys.executable,
            shoot_policy_dir='checkpoints/reproduction/shoot', mpc_performance=False):
    module = {'mappo': 'scripts.train_high_level_mappo_fsp',
              'discrete': 'scripts.train_discrete_high_level',
              'frozen': 'scripts.train_high_level_frozen_world_model',
              'mpc': 'scripts.train_high_level_mpc_replay'}[condition]
    args = [python, '-m', module, '--seed', str(seed), '--iterations', str(iterations),
            '--num-envs', str(num_envs), '--num-robots', '2', '--headless',
            '--device', device, '--policy-device', device,
            '--centralized-critic', '--soccer-curriculum', '--attack-diagnostics',
            '--shooting-options', '--ppo-epochs', '5',
            '--attack-position-reward', '1', '--goalward-launch-reward',
            '--rollout-steps', '48', '--self-play-update-interval', '1200',
            '--opponent-latest-probability', '.2', '--shoot-setup-event-reward', '1',
            '--use-geometric-skill-fallback', '--skill-policy-source', 'local',
            '--walk-policy-dir', 'checkpoints/reproduction/walk',
            '--dribble-policy-dir', 'checkpoints/reproduction/dribble',
            '--shoot-policy-dir', shoot_policy_dir,
            '--project', 'as2_efficiency_'+condition]
    if condition in ('mpc', 'frozen'):
        args += ['--mpc-quality-filter', '--mpc-query-budget', '128',
                 '--world-model-update-interval', '10', '--world-model-updates-per-interval', '20',
                 '--mpc-replan-interval', '8', '--mpc-horizon', '4',
                 '--mpc-num-samples', '64', '--mpc-num-iterations', '2',
                 '--mpc-terminal-value', 'enabled', '--no-terminal-bootstrap-on-timeout', '--mpc-replay-batch-size', '64',
                 '--mpc-replay-min-size', '64', '--mpc-policy-kl-limit', '.003',
                 '--on-policy-ppo-epochs', '5', '--mpc-real-rollout-guard',
                 '--mpc-value-fallback', '--mpc-teacher-max-age', '256',
                 '--mpc-replay-updates-per-rollout', '1']
        if mpc_performance:
            args += ['--mpc-query-budget', '256', '--mpc-num-samples', '128',
                     '--mpc-num-iterations', '3', '--mpc-kl-ramp-steps', '2400',
                     '--mpc-max-idle-rollouts', '250',
                     '--mpc-replay-batch-size', '256', '--mpc-adaptive-replay-batch',
                     '--project', 'as2_efficiency_mpc_extra_compute']
    if condition == 'mpc':
        args += ['--world-model-checkpoint', 'checkpoints/offline_teacher/best.pt',
                 '--mpc-warmup-steps', '192']
    if condition == 'frozen':
        args += ['--mpc-terminal-value', 'disabled']
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['mappo', 'discrete', 'mpc', 'both', 'all'], default='both')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--iterations', type=int, default=1200)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--device', default='cuda:7')
    parser.add_argument('--shoot-policy-dir', default='checkpoints/reproduction/shoot', help='Common frozen shooting policy for every condition.')
    parser.add_argument('--manifest', default='/tmp/mpc_efficiency_experiments.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mpc-performance', action='store_true',
                        help='Give MPC additional search compute; report this budget difference in comparisons.')
    args = parser.parse_args()
    if args.shoot_policy_dir is None:
        sys.path.insert(0, str(ROOT))
        from scripts.validate_shooting_option import validated_policy_dir
        args.shoot_policy_dir = validated_policy_dir()
    if args.iterations < 1 or args.num_envs < 1:
        parser.error('iterations and num-envs must be positive')
    conditions = ['mappo', 'mpc'] if args.condition == 'both' else (['mappo', 'discrete', 'mpc'] if args.condition == 'all' else [args.condition])
    jobs = [{'condition': c, 'seed': s, 'argv': command(c, s, args.iterations, args.num_envs, args.device,
                                                     shoot_policy_dir=args.shoot_policy_dir,
                                                     mpc_performance=args.mpc_performance)}
            for s in args.seeds for c in conditions]
    Path(args.manifest).write_text(json.dumps(jobs, indent=2)+'\n')
    env = dict(os.environ)
    env.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/dribblebot_torch_extensions')
    for job in jobs:
        print(shlex.join(job['argv']), flush=True)
        if args.execute:
            subprocess.run(job['argv'], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
