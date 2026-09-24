"""Single goal-learning profile used by all three Bash training launchers."""
import argparse
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_mpc_efficiency_ablation import command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True, choices=['mappo', 'discrete', 'mpc', 'frozen'])
    parser.add_argument('--cuda', type=int, default=int(os.environ.get('DRIBBLEBOT_CUDA_INDEX', 7)))
    parser.add_argument('--seed', type=int, default=42)
    # Long runs are needed to distinguish optimizer noise from convergence;
    # checkpoints remain available every save interval for early stopping.
    parser.add_argument('--iterations', type=int, default=6000)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--shoot-policy-dir', default='checkpoints/reproduction/shoot')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--mpc-performance', action='store_true',
                        help='Enable the explicitly extra-compute MPC preset.')
    args, extra = parser.parse_known_args()
    if args.cuda < 0 or args.iterations < 1 or args.num_envs < 1:
        parser.error('GPU index must be nonnegative; iterations and num-envs must be positive')
    if args.mpc_performance and args.condition != 'mpc':
        parser.error('--mpc-performance applies only to MPC')
    argv = command(args.condition, args.seed, args.iterations, args.num_envs,
                   f'cuda:{args.cuda}', shoot_policy_dir=args.shoot_policy_dir,
                   mpc_performance=args.mpc_performance) + extra
    print(shlex.join(argv), flush=True)
    if not args.dry_run:
        os.chdir(ROOT)
        os.execv(sys.executable, argv)


if __name__ == '__main__':
    main()
