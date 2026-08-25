"""Select a high-level checkpoint with fixed-seed, fixed-opponent evaluation."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_csv_tokens(value):
    return [token.strip() for token in str(value).split(",") if token.strip()]


def numbered_checkpoints(policy_dir):
    checkpoints = []
    for path in policy_dir.glob("ac_weights_*.pt"):
        token = path.stem[len("ac_weights_") :]
        if not token.isdigit():
            continue
        if not (policy_dir / f"body_{token}.jit").is_file():
            continue
        if not (policy_dir / f"adaptation_module_{token}.jit").is_file():
            continue
        checkpoints.append(int(token))
    return [str(value) for value in sorted(checkpoints)]


def resolve_tokens(configured, available, label):
    if configured == "auto":
        if label == "candidate":
            return list(available)
        if len(available) <= 3:
            return list(available)
        indices = (0, len(available) // 2, len(available) - 1)
        return [available[index] for index in indices]
    tokens = parse_csv_tokens(configured)
    if not tokens:
        raise ValueError(f"No {label} checkpoints were configured")
    return tokens


def opponent_path(policy_dir, checkpoint):
    requested = Path(checkpoint).expanduser()
    if requested.is_file():
        return requested.resolve()
    path = policy_dir / f"ac_weights_{checkpoint}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Opponent actor checkpoint does not exist: {path}")
    return path.resolve()


def summarize_rollout(csv_path, candidate, opponent, seed):
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"Evaluation produced no rows: {csv_path}")

    def total(key):
        return sum(float(row.get(key, 0.0) or 0.0) for row in rows)

    goals = total("high_level_goal")
    opponent_goals = total("high_level_opponent_goal")
    return {
        "candidate": str(candidate),
        "opponent": str(opponent),
        "seed": int(seed),
        "steps": len(rows),
        "episodes": int(total("done")),
        "goals": int(goals),
        "opponent_goals": int(opponent_goals),
        "goal_difference": int(goals - opponent_goals),
        "off_border": int(total("high_level_ball_off_border")),
        "accidental_terminations": int(
            total("high_level_accidental_termination")
        ),
        "total_reward": total("reward"),
        "mean_reward_per_step": total("reward") / len(rows),
    }


def aggregate_results(results):
    by_candidate = {}
    for result in results:
        by_candidate.setdefault(result["candidate"], []).append(result)
    aggregates = []
    for candidate, rows in by_candidate.items():
        count = len(rows)
        aggregates.append(
            {
                "candidate": candidate,
                "evaluations": count,
                "goals": sum(row["goals"] for row in rows),
                "opponent_goals": sum(row["opponent_goals"] for row in rows),
                "mean_goal_difference": sum(
                    row["goal_difference"] for row in rows
                )
                / count,
                "mean_reward_per_step": sum(
                    row["mean_reward_per_step"] for row in rows
                )
                / count,
                "mean_off_border": sum(row["off_border"] for row in rows)
                / count,
                "mean_accidental_terminations": sum(
                    row["accidental_terminations"] for row in rows
                )
                / count,
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (
            row["mean_goal_difference"],
            row["goals"],
            -row["opponent_goals"],
            row["mean_reward_per_step"],
            -row["mean_off_border"],
        ),
        reverse=True,
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    policy_dir = Path(args.policy_dir).expanduser().resolve()
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"Policy directory does not exist: {policy_dir}")
    available = numbered_checkpoints(policy_dir)
    if not available:
        raise FileNotFoundError(
            f"No complete numbered high-level checkpoints found in {policy_dir}"
        )
    candidates = resolve_tokens(args.candidate_checkpoints, available, "candidate")
    opponents = resolve_tokens(args.opponent_checkpoints, available, "opponent")
    seeds = [int(token) for token in parse_csv_tokens(args.seeds)]
    if not seeds:
        raise ValueError("At least one fixed seed is required")

    output_dir = Path(args.output_dir).expanduser().resolve()
    rollout_dir = output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total_runs = len(candidates) * len(opponents) * len(seeds)
    run_index = 0
    for candidate in candidates:
        for opponent in opponents:
            resolved_opponent = opponent_path(policy_dir, opponent)
            for seed in seeds:
                run_index += 1
                stem = f"candidate_{candidate}_opponent_{opponent}_seed_{seed}"
                metrics_path = rollout_dir / f"{stem}.csv"
                command = [
                    args.python,
                    str(ROOT / "scripts" / "play_high_level.py"),
                    "--num-robots",
                    str(args.num_robots),
                    "--high-level-policy-source",
                    "local",
                    "--high-level-policy-dir",
                    str(policy_dir),
                    "--high-level-checkpoint",
                    str(candidate),
                    "--opponent-high-level-checkpoint",
                    str(resolved_opponent),
                    "--skill-policy-source",
                    "local",
                    "--walk-policy-dir",
                    str(Path(args.walk_policy_dir).expanduser()),
                    "--dribble-policy-dir",
                    str(Path(args.dribble_policy_dir).expanduser()),
                    "--shoot-policy-dir",
                    str(Path(args.shoot_policy_dir).expanduser()),
                    "--device",
                    args.device,
                    "--policy-device",
                    args.policy_device,
                    "--steps",
                    str(args.steps),
                    "--seed",
                    str(seed),
                    "--csv",
                    str(metrics_path),
                    "--headless",
                    "--no-video",
                    "--no-plot",
                ]
                if args.allow_training_config_mismatch:
                    command.append("--allow-training-config-mismatch")
                print(
                    f"[{run_index}/{total_runs}] candidate={candidate} "
                    f"opponent={opponent} seed={seed}",
                    flush=True,
                )
                completed = subprocess.run(
                    command,
                    cwd=str(ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Evaluation failed for {stem}:\n{completed.stdout}"
                    )
                results.append(
                    summarize_rollout(metrics_path, candidate, opponent, seed)
                )

    aggregates = aggregate_results(results)
    write_csv(output_dir / "evaluations.csv", results)
    write_csv(output_dir / "checkpoint_ranking.csv", aggregates)
    best = aggregates[0]
    selection = {
        "best_checkpoint": best["candidate"],
        "ranking_rule": [
            "mean_goal_difference",
            "goals",
            "fewest_opponent_goals",
            "mean_reward_per_step",
            "fewest_off_border",
        ],
        "candidates": candidates,
        "opponents": opponents,
        "seeds": seeds,
        "best_metrics": best,
    }
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best checkpoint: {best['candidate']}")
    print(f"Ranking: {output_dir / 'checkpoint_ranking.csv'}")
    print(f"Selection: {output_dir / 'best_checkpoint.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument(
        "--candidate-checkpoints",
        default="auto",
        help="Comma-separated candidate suffixes, or auto for all complete numbered checkpoints.",
    )
    parser.add_argument(
        "--opponent-checkpoints",
        default="auto",
        help="Comma-separated fixed actor suffixes, or auto for early/middle/latest snapshots.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument(
        "--python",
        default="/home/zhz/anaconda3/envs/legged_env/bin/python",
    )
    parser.add_argument("--walk-policy-dir", default="checkpoints/reproduction/walk")
    parser.add_argument("--dribble-policy-dir", default="checkpoints/reproduction/dribble")
    parser.add_argument("--shoot-policy-dir", default="checkpoints/reproduction/shoot")
    parser.add_argument("--output-dir", default="outputs/high_level_benchmark")
    parser.add_argument("--allow-training-config-mismatch", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
