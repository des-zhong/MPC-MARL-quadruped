"""Evaluate one high-level policy against every saved opponent pool.

Each ``opponent_pool_*.pt`` is evaluated as one opponent checkpoint.  Ten
parallel matches are run for each checkpoint (one completed episode per
match), and the reported rate is learning-team goals divided by both teams'
goals.  Terminal events are OR-reduced per match so a goal is counted once.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _evaluate_one(args, pool, output):
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "outcomes.csv"
    command = [
        sys.executable, "scripts/validate_high_level.py",
        "--high-level-policy-source", "local",
        "--high-level-policy-dir", str(args.policy.parent),
        "--opponent-pool-checkpoint", str(pool),
        "--num-envs", str(args.games), "--num-robots", str(args.num_robots),
        "--steps", str(args.steps), "--stop-after-episodes", "1",
        "--export-all-envs", "--outcomes-only", "--no-plot", "--no-video",
        "--training-environment", "--training-skills",
        "--device", args.device, "--policy-device", args.policy_device,
        "--csv", str(csv_path), "--seed", str(args.seed), "--no-progress",
    ]
    if args.headless:
        command.append("--headless")
    subprocess.run(command, check=True)

    terminal_keys = ("high_level_goal", "high_level_opponent_goal")
    by_env = {}
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            env_id = int(row["env_id"])
            event = by_env.setdefault(env_id, {key: 0 for key in terminal_keys})
            for key in terminal_keys:
                event[key] |= int(row[key])
    if len(by_env) != args.games:
        raise RuntimeError(f"Expected {args.games} completed matches, got {len(by_env)} in {csv_path}")
    our = sum(item["high_level_goal"] for item in by_env.values())
    opponent = sum(item["high_level_opponent_goal"] for item in by_env.values())
    wins = sum(
        item["high_level_goal"] > item["high_level_opponent_goal"]
        for item in by_env.values()
    )
    return our, opponent, wins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Candidate ac_weights_*.pt")
    parser.add_argument(
        "--opponent-pool-dir", type=Path, nargs="+", required=True,
        help="One or more directories containing opponent_pool_*.pt files.",
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/high_level_pool_eval"))
    parser.add_argument(
        "--metric", choices=("goal-rate", "win-rate"), default="goal-rate",
        help="Reported ratio: our goals/(both goals), or matches won/games.",
    )
    args = parser.parse_args()
    args.policy = args.policy.expanduser().resolve()
    args.opponent_pool_dir = [path.expanduser().resolve() for path in args.opponent_pool_dir]
    if not args.policy.is_file():
        parser.error(f"candidate policy does not exist: {args.policy}")
    pool_groups = []
    for directory in args.opponent_pool_dir:
        pools = sorted(directory.glob("opponent_pool_*.pt"))
        if not pools:
            parser.error(f"no opponent_pool_*.pt files found in {directory}")
        pool_groups.append((directory, pools))
    if args.games < 1 or args.steps < 1:
        parser.error("--games and --steps must be positive")

    rows, groups, total_our, total_opponent, total_wins, total_games = [], [], 0, 0, 0, 0
    for directory, pools in pool_groups:
        group_our = group_opponent = group_wins = 0
        for index, pool in enumerate(pools):
            output_dir = args.output / directory.name / pool.stem
            our, opponent, wins = _evaluate_one(args, pool, output_dir)
            group_our += our
            group_opponent += opponent
            group_wins += wins
            total_games += args.games
            group_rate = group_wins / ((index + 1) * args.games)
            total_our += our
            total_opponent += opponent
            total_wins += wins
            goal_rate = our / (our + opponent) if our + opponent else float("nan")
            win_rate = wins / args.games
            cumulative_goal_rate = total_our / (total_our + total_opponent) if total_our + total_opponent else float("nan")
            cumulative_win_rate = total_wins / total_games
            rate = goal_rate if args.metric == "goal-rate" else win_rate
            cumulative = cumulative_goal_rate if args.metric == "goal-rate" else cumulative_win_rate
            row = {"pool_dir": str(directory), "checkpoint": pool.name, "our_goals": our, "opponent_goals": opponent,
                   "games": args.games, "wins": wins, "goal_rate": goal_rate,
                   "win_rate": win_rate, "reported_rate": rate,
                   "cumulative_goal_rate": cumulative_goal_rate,
                   "cumulative_win_rate": cumulative_win_rate,
                   "cumulative_reported_rate": cumulative}
            rows.append(row)
            print(f"{directory.name}/{pool.name}: {args.metric}={rate:.2%} (wins {wins}/{args.games}); total: {cumulative:.2%}", flush=True)
        group_goal_rate = group_our / (group_our + group_opponent) if group_our + group_opponent else float("nan")
        group_win_rate = group_wins / (len(pools) * args.games)
        groups.append({"pool_dir": str(directory), "games": len(pools) * args.games,
                       "our_goals": group_our, "opponent_goals": group_opponent,
                       "wins": group_wins, "goal_rate": group_goal_rate,
                       "win_rate": group_win_rate})
        selected_group = group_goal_rate if args.metric == "goal-rate" else group_win_rate
        print(f"{directory.name}: {args.metric}={selected_group:.2%} ({group_wins}/{len(pools) * args.games} wins)", flush=True)

    report = {"policy": str(args.policy), "opponent_pool_dir": [str(path) for path in args.opponent_pool_dir],
              "games_per_checkpoint": args.games, "metric": args.metric, "checkpoints": rows,
              "pool_groups": groups,
              "total_our_goals": total_our, "total_opponent_goals": total_opponent,
              "total_wins": total_wins,
              "total_goal_rate": total_our / (total_our + total_opponent) if total_our + total_opponent else None,
              "total_win_rate": total_wins / total_games}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    selected = report["total_goal_rate"] if args.metric == "goal-rate" else report["total_win_rate"]
    print(f"Total {args.metric}: {selected:.2%}" if selected is not None else f"Total {args.metric}: undefined")


if __name__ == "__main__":
    main()
