"""Fast, matched learning curves for full MPC and its two component ablations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_high_level_learning_curves import numbered_checkpoints

DEFAULT_MPC = "wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay"
DEFAULT_OPPONENT = "wandb/run-20260920_145323-1s2mw9rb/files/tmp/legged_data/high_level"


def evenly_spaced(values, limit):
    """Keep endpoints and spread the remaining evaluation points across training."""
    if limit == 0 or len(values) <= limit:
        return list(values)
    if limit == 1:
        return [values[-1]]
    return [values[round(i * (len(values) - 1) / (limit - 1))] for i in range(limit)]


def policy_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def build_command(args):
    methods = {
        "MPC": policy_path(args.mpc_dir),
        "MPC w.o. terminal value": policy_path(args.no_terminal_value_dir),
        "MPC w.o. uncertainty": policy_path(args.no_uncertainty_dir),
    }
    common = None
    for label, directory in methods.items():
        if not (directory / "config.yaml").is_file():
            raise FileNotFoundError(f"Missing training config for {label}: {directory / 'config.yaml'}")
        checkpoints = set(map(int, numbered_checkpoints(directory)))
        if not checkpoints:
            raise ValueError(f"No complete numbered checkpoint exports for {label}: {directory}")
        common = checkpoints if common is None else common & checkpoints
    available = sorted(c for c in common if args.max_iteration is None or c <= args.max_iteration)
    if not available:
        raise ValueError("No common complete numbered checkpoints within the requested iteration range")
    candidates = evenly_spaced(available, args.max_points)
    opponent_dir = policy_path(args.opponent_dir)
    opponent_max = args.opponent_max_iteration
    if args.opponent_checkpoints:
        tokens = args.opponent_checkpoints.split(",")
        if any(not token.strip().isdigit() for token in tokens):
            raise ValueError("--opponent-checkpoints must contain comma-separated numbered checkpoints")
        opponent_iterations = sorted(set(int(token) for token in tokens))
        if opponent_max is not None and any(c > opponent_max for c in opponent_iterations):
            raise ValueError("Explicit opponent checkpoints exceed --opponent-max-iteration")
    else:
        # With no explicit cap, retain the common training range default.
        if opponent_max is None:
            opponent_max = available[-1]
        opponent_iterations = sorted({
            int(path.stem[len("ac_weights_"):])
            for path in opponent_dir.glob("ac_weights_*.pt")
            if path.stem[len("ac_weights_"):].isdigit()
            and int(path.stem[len("ac_weights_"):]) <= opponent_max
        })
        opponent_iterations = evenly_spaced(opponent_iterations, args.opponents)
    if not opponent_iterations:
        raise ValueError(f"No numbered opponents available in {opponent_dir}")
    for iteration in opponent_iterations:
        path = opponent_dir / f"ac_weights_{iteration}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing opponent: {path}")
    device = f"cuda:{args.cuda}"
    command = [sys.executable, str(ROOT / "scripts/compare_high_level_learning_curves.py")]
    for label, directory in methods.items():
        command += ["--method", f"{label}={directory}"]
    command += [
        "--candidate-checkpoints", ",".join(map(str, candidates)),
        "--opponent-dir", str(opponent_dir),
        "--opponent-checkpoints", ",".join(map(str, opponent_iterations)),
        "--episodes-per-opponent", str(args.episodes_per_opponent),
        "--steps", str(args.steps), "--seeds", str(args.eval_seed),
        "--device", device, "--policy-device", device,
        "--metric", args.metric, "--x-axis", "environment-steps",
        "--num-robots", "2", "--incremental-plot", "--no-overwrite",
        "--output-dir", str(policy_path(args.output_dir)),
    ]
    for skill in ("walk", "dribble", "shoot"):
        directory = getattr(args, f"{skill}_policy_dir")
        if directory:
            command += [f"--{skill}-policy-dir", str(policy_path(directory))]
    return command, candidates, opponent_iterations


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpc-dir", default=DEFAULT_MPC)
    parser.add_argument("--no-terminal-value-dir", default="outputs/mpc_ablations/no_terminal_value/seed_42")
    parser.add_argument("--no-uncertainty-dir", default="outputs/mpc_ablations/no_uncertainty/seed_42")
    parser.add_argument("--opponent-dir", default=DEFAULT_OPPONENT,
                        help="Shared frozen opponent pool (defaults to the existing MAPPO run).")
    parser.add_argument("--opponent-checkpoints", help="Pin the pool explicitly, e.g. 0,400,1200.")
    parser.add_argument("--opponent-max-iteration", type=int,
                        help="Inclusive opponent checkpoint cap; defaults to the last common candidate.")
    parser.add_argument("--opponents", type=int, default=3,
                        help="Number of evenly spaced opponents; 0 uses all available.")
    parser.add_argument("--max-points", type=int, default=8,
                        help="Maximum common checkpoint points per curve; 0 evaluates all.")
    parser.add_argument("--max-iteration", type=int)
    parser.add_argument("--episodes-per-opponent", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000,
                        help="Safety cap; simulation stops when all initial episodes finish.")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=4)
    parser.add_argument("--metric", choices=("win-rate", "goal-rate", "expected-score"), default="win-rate")
    parser.add_argument("--output-dir", default="outputs/mpc_ablation_learning_curves")
    for skill in ("walk", "dribble", "shoot"):
        parser.add_argument(f"--{skill}-policy-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if min(args.cuda, args.max_points, args.opponents) < 0:
        parser.error("cuda, max-points, and opponents must be nonnegative")
    if args.steps < 1 or args.episodes_per_opponent < 1:
        parser.error("steps and episodes-per-opponent must be positive")
    if args.max_iteration is not None and args.max_iteration < 0:
        parser.error("max-iteration must be nonnegative")
    if args.opponent_max_iteration is not None and args.opponent_max_iteration < 0:
        parser.error("opponent-max-iteration must be nonnegative")
    return args


def main():
    args = parse_args()
    command, candidates, opponents = build_command(args)
    print(f"Common candidate iterations: {candidates}", flush=True)
    print(f"Fixed opponent iterations: {opponents}", flush=True)
    print(f"{3 * len(candidates) * len(opponents)} matchups; "
          f"{len(opponents) * args.episodes_per_opponent} episodes per curve point", flush=True)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        env = dict(os.environ)
        env.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/dribblebot_torch_extensions")
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
