"""Compare two high-level checkpoints without rendering (100 matches x 10 repeats).

A match is one completed environment episode, ending in a goal, timeout, or
another terminal event. Only the first episode in each environment is counted.
Repetitions are disjoint groups simulated in parallel using one seeded random
stream per batch; all repetitions share one simulator startup by default.
Initial conditions are symmetric: centered stationary ball, paired team poses
rotated by 180 degrees, nominal joints and physics, and a flat field. No
attacking curriculum or observation noise is used.
Team A supplies the remaining saved environment configuration and low-level skills;
both teams use those skills. Hybrid and discrete coordinators retain their
native observations/actions. Checkpoints need compatible shared execution
settings and their accompanying JIT exports and training configuration.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_matches(path, matches, *, start_env=0, total_envs=None):
    """Read exactly one completed episode per environment, ignoring later games."""
    total_envs = matches if total_envs is None else total_envs
    completed = {}
    goals = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            env = int(row["env_id"])
            if not 0 <= env < total_envs:
                raise ValueError(f"Unexpected environment {env} in {path}")
            if not start_env <= env < start_env + matches:
                continue
            if env in completed:
                continue
            a, b = goals.get(env, (0, 0))
            a |= int(row["high_level_goal"])
            b |= int(row["high_level_opponent_goal"])
            goals[env] = (a, b)
            if int(row["done"]):
                completed[env] = (a, b)
    if len(completed) != matches:
        raise RuntimeError(
            f"Only {len(completed)}/{matches} matches completed in {path}; "
            "increase --steps and rerun. Incomplete matches are not counted."
        )
    return sum(a for a, _ in completed.values()), sum(b for _, b in completed.values())


def summarize(rows):
    rates = [row["goal_rate"] for row in rows if row["goal_rate"] is not None]
    return {
        "defined_repeats": len(rates),
        "undefined_repeats": len(rows) - len(rates),
        "mean_goal_rate": statistics.mean(rates) if rates else None,
        "variance": statistics.pvariance(rates) if rates else None,
        "sample_variance": statistics.variance(rates) if len(rates) > 1 else None,
    }


def evaluation_command(args, seed, output, repeats=1):
    suffix = args.team_a.stem[len("ac_weights_"):]
    return [
        sys.executable, "-m", "scripts.validate_high_level",
        "--high-level-policy-source", "local",
        "--high-level-policy-dir", str(args.team_a.parent),
        "--high-level-checkpoint", suffix,
        "--opponent-high-level-checkpoint", str(args.team_b),
        "--num-envs", str(args.matches * repeats), "--num-robots", str(args.num_robots),
        "--steps", str(args.steps), "--stop-after-episodes", "1",
        "--first-episode-only", "--export-all-envs", "--outcomes-only", "--no-plot", "--no-video",
        "--headless", "--no-progress", "--training-environment", "--training-skills",
        "--fair-match-init",
        "--device", args.device, "--policy-device", args.policy_device,
        "--seed", str(seed), "--csv", str(output / "outcomes.csv"),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-a", type=Path, required=True, help="Path to ac_weights_<checkpoint>.pt")
    parser.add_argument("--team-b", type=Path, required=True, help="Path to ac_weights_*.pt or opponent_ac_weights_*.pt")
    parser.add_argument("--matches", type=int, default=100, help="Parallel matches per repetition (default: 100)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--repeats-per-batch", type=int, default=None,
                        help="Repetitions simulated together (default: all); reduce for lower GPU memory use")
    parser.add_argument("--steps", type=int, default=600, help="Maximum high-level steps per repetition; errors if any match is unfinished")
    parser.add_argument("--num-robots", type=int, default=2, help="Robots per team")
    parser.add_argument("--seed", type=int, default=42, help="Batch i uses seed + i; repetitions use disjoint environments")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-device", default=None, help="Defaults to --device to avoid CPU/GPU transfers")
    parser.add_argument("--output", type=Path, default=Path("outputs/goal_rate"))
    args = parser.parse_args(argv)
    args.policy_device = args.policy_device or args.device
    if args.repeats_per_batch is None:
        args.repeats_per_batch = args.repeats
    for name in ("repeats_per_batch", "matches", "repeats", "steps", "num_robots"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("team_a", "team_b"):
        path = getattr(args, name).expanduser().resolve()
        prefixes = ("ac_weights_",) if name == "team_a" else ("ac_weights_", "opponent_ac_weights_")
        if not path.is_file() or path.suffix != ".pt" or not path.stem.startswith(prefixes):
            parser.error(f"--{name.replace('_', '-')}: expected an existing policy weights checkpoint, got {path}")
        setattr(args, name, path)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Prevent accidentally mixing a partial/new evaluation with old results.
    if any(args.output.iterdir()):
        parser.error("--output must be an empty or new directory")
    rows = []
    for batch_index, start in enumerate(range(0, args.repeats, args.repeats_per_batch)):
        count = min(args.repeats_per_batch, args.repeats - start)
        seed = args.seed + batch_index
        output = args.output / f"batch_{batch_index + 1:02d}"
        output.mkdir()
        command = evaluation_command(args, seed, output, repeats=count)
        (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(f"Batch {batch_index + 1}: {count} repeats, {args.matches * count} parallel matches, seed={seed}", flush=True)
        with (output / "evaluation.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Evaluation failed; see {output / 'evaluation.log'}")
        for offset in range(count):
            a, b = read_matches(output / "outcomes.csv", args.matches,
                                start_env=offset * args.matches, total_envs=count * args.matches)
            rate = a / (a + b) if a + b else None
            rows.append(dict(repeat=start + offset + 1, seed=seed, batch=batch_index + 1,
                             first_env=offset * args.matches, matches=args.matches,
                             team_a_goals=a, team_b_goals=b, goal_rate=rate))
            label = f"{rate:.6f}" if rate is not None else "undefined (no goals)"
            print(f"  Repeat {start + offset + 1}: A={a}, B={b}, goal rate={label}", flush=True)
        with (args.output / "repeats.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "team_a": str(args.team_a), "team_b": str(args.team_b),
        "matches_per_repeat": args.matches, "repeats": args.repeats,
        "repeats_per_batch": args.repeats_per_batch,
        "initial_conditions": "centered stationary ball; team poses related by a 180-degree rotation; nominal joints; no domain randomization, observation noise, or attacking curriculum",
        "sampling": "disjoint environment groups; one seeded random stream per batch",
        "variance_definition": "population variance across defined repeat goal rates (ddof=0)",
        "zero_goal_handling": "undefined repeats excluded from mean and variance",
        **summarize(rows), "results": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for key in ("mean_goal_rate", "variance", "sample_variance", "defined_repeats", "undefined_repeats"):
        print(f"{key}: {report[key]}")
    print(f"Report: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
