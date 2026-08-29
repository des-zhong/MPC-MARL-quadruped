"""Evaluate and plot high-level soccer learning curves on a fixed benchmark.

Training rewards are deliberately not compared here: MPC-guided runs contain
method-specific optimization terms.  Every numbered policy checkpoint is
instead played against the same frozen MAPPO-FSP checkpoint(s), initial-state
seeds, and low-level skill policies.  The primary metric is expected match
score, where a win is 1, a draw is 0.5, and a loss is 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_csv_tokens(value):
    return [token.strip() for token in str(value).split(",") if token.strip()]


def parse_labeled_values(values, value_parser=str):
    result = {}
    for configured in values or []:
        if "=" not in configured:
            raise ValueError(
                f"Expected LABEL=VALUE, got {configured!r}"
            )
        label, value = configured.split("=", 1)
        label = label.strip()
        value = value.strip()
        if not label or not value:
            raise ValueError(f"Expected non-empty LABEL=VALUE, got {configured!r}")
        if label in result:
            raise ValueError(f"Duplicate configuration for {label!r}")
        result[label] = value_parser(value)
    return result


def filename_slug(value):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return slug or "method"


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
    return [str(value) for value in sorted(set(checkpoints))]


def resolve_candidate_checkpoints(configured, available, label):
    if configured == "auto":
        return list(available)
    requested = parse_csv_tokens(configured)
    if not requested:
        raise ValueError(f"No checkpoints requested for {label}")
    missing = sorted(set(requested) - set(available), key=int)
    if missing:
        raise FileNotFoundError(
            f"{label} is missing complete checkpoint exports: {missing}"
        )
    return sorted(set(requested), key=int)


def resolve_opponents(configured, opponent_dir):
    available = numbered_checkpoints(opponent_dir)
    if not available:
        # Opponents only require actor weights, not JIT exports.
        available = sorted(
            {
                path.stem[len("ac_weights_") :]
                for path in opponent_dir.glob("ac_weights_*.pt")
                if path.stem[len("ac_weights_") :].isdigit()
            },
            key=int,
        )
    if not available:
        raise FileNotFoundError(
            f"No numbered actor checkpoints found in opponent directory {opponent_dir}"
        )

    if configured == "auto":
        tokens = [available[-1]]
    elif configured == "suite":
        indices = (0, len(available) // 2, len(available) - 1)
        tokens = list(dict.fromkeys(available[index] for index in indices))
    else:
        tokens = parse_csv_tokens(configured)
        if not tokens:
            raise ValueError("At least one opponent checkpoint is required")

    opponents = []
    for token in tokens:
        requested = Path(token).expanduser()
        if requested.is_file():
            path = requested.resolve()
            name = path.stem
        else:
            suffix = "latest" if token in ("latest", "last") else token
            path = (opponent_dir / f"ac_weights_{suffix}.pt").resolve()
            name = suffix
        if not path.is_file():
            raise FileNotFoundError(f"Opponent checkpoint does not exist: {path}")
        opponents.append((name, path))
    return opponents


def _wandb_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def timesteps_per_iteration(policy_dir):
    """Read the runner's actual agent-transition batch size from W&B config."""

    config_path = policy_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No training config found at {config_path}; pass "
            f"--steps-per-iteration 'LABEL=N'"
        )
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required to read checkpoint config.yaml") from error

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    runner = _wandb_value(payload.get("RunnerArgs", {})) or {}
    cfg = _wandb_value(payload.get("Cfg", {})) or {}
    env = _wandb_value(cfg.get("env", {})) if isinstance(cfg, dict) else {}
    if not isinstance(runner, dict) or not isinstance(env, dict):
        raise ValueError(f"Malformed RunnerArgs/Cfg.env in {config_path}")
    try:
        rollout_steps = int(runner["num_steps_per_env"])
        match_envs = int(env["num_envs"])
        team_size = int(env["num_team_robots"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Could not infer rollout batch size from {config_path}; pass "
            f"--steps-per-iteration 'LABEL=N'"
        ) from error
    scale = rollout_steps * match_envs * team_size
    if scale < 1:
        raise ValueError(f"Invalid timesteps-per-iteration value {scale} in {config_path}")
    return scale


def summarize_rollout(
    csv_path,
    expected_rows=None,
    expected_steps=None,
    expected_envs=None,
):
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"Evaluation produced no rows: {csv_path}")
    if expected_rows is not None and len(rows) != int(expected_rows):
        raise RuntimeError(
            f"Evaluation row count is incomplete in {csv_path}: "
            f"expected {expected_rows}, found {len(rows)}"
        )
    if expected_steps is not None:
        observed_steps = {int(row["step"]) for row in rows}
        if observed_steps != set(range(int(expected_steps))):
            raise RuntimeError(
                f"Evaluation step coverage is incomplete in {csv_path}"
            )
    if expected_envs is not None:
        observed_envs = {int(row.get("env_id", 0)) for row in rows}
        if observed_envs != set(range(int(expected_envs))):
            raise RuntimeError(
                f"Evaluation environment coverage is incomplete in {csv_path}"
            )
    if expected_steps is not None and expected_envs is not None:
        observed_pairs = {
            (int(row["step"]), int(row.get("env_id", 0))) for row in rows
        }
        expected_pairs = {
            (step, env_id)
            for step in range(int(expected_steps))
            for env_id in range(int(expected_envs))
        }
        if observed_pairs != expected_pairs:
            raise RuntimeError(
                f"Evaluation step/environment coverage is incomplete in {csv_path}"
            )

    required_metrics = (
        "done",
        "high_level_goal",
        "high_level_opponent_goal",
        "reward",
    )
    missing_metrics = [
        key for key in required_metrics if any(key not in row for row in rows)
    ]
    if missing_metrics:
        raise RuntimeError(
            f"Evaluation metrics are missing columns in {csv_path}: "
            f"{', '.join(missing_metrics)}"
        )

    def total(key):
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Evaluation metric {key!r} contains non-finite values in {csv_path}"
            )
        return sum(values)

    episodes = int(total("done"))
    wins = int(total("high_level_goal"))
    losses = int(total("high_level_opponent_goal"))
    draws = episodes - wins - losses
    if draws < 0:
        raise RuntimeError(
            f"Terminal goal flags exceed episode count in {csv_path}: "
            f"episodes={episodes}, wins={wins}, losses={losses}"
        )
    total_reward = total("reward")
    return {
        "rollout_steps": len(rows),
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "goals_for": wins,
        "goals_against": losses,
        "goal_difference": wins - losses,
        "off_border": int(total("high_level_ball_off_border")),
        "accidental_terminations": int(
            total("high_level_accidental_termination")
        ),
        "total_reward": total_reward,
        "mean_reward_per_step": total_reward / len(rows),
        "expected_score_percent": (
            100.0 * (wins + 0.5 * draws) / episodes
            if episodes
            else float("nan")
        ),
        "goal_difference_per_episode": (
            (wins - losses) / episodes if episodes else float("nan")
        ),
    }


def score_and_confidence(wins, draws, losses):
    """Expected score and a Wilson 95% interval in percent.

    Draws contribute half a success.  Wilson intervals remain informative at
    the 0% and 100% boundaries, unlike a normal interval based on the observed
    variance.
    """

    count = int(wins + draws + losses)
    if count < 1:
        return float("nan"), float("nan"), float("nan")
    mean = (float(wins) + 0.5 * float(draws)) / count
    z = 1.96
    denominator = 1.0 + z * z / count
    center = (mean + z * z / (2.0 * count)) / denominator
    half_width = (
        z
        * math.sqrt(
            mean * (1.0 - mean) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return (
        100.0 * mean,
        100.0 * max(0.0, center - half_width),
        100.0 * min(1.0, center + half_width),
    )


def aggregate_evaluations(rows):
    grouped = {}
    for row in rows:
        key = (row["method"], row["checkpoint"])
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for (method, checkpoint), group in grouped.items():
        evaluations = len(group)
        episodes = sum(int(row["episodes"]) for row in group)
        wins = sum(int(row["wins"]) for row in group)
        losses = sum(int(row["losses"]) for row in group)
        draws = sum(int(row["draws"]) for row in group)
        score, ci_low, ci_high = score_and_confidence(wins, draws, losses)
        rollout_steps = sum(int(row["rollout_steps"]) for row in group)
        total_reward = sum(float(row["total_reward"]) for row in group)
        aggregates.append(
            {
                "method": method,
                "checkpoint": checkpoint,
                "training_iteration": int(group[0]["training_iteration"]),
                "training_environment_steps": int(
                    group[0]["training_environment_steps"]
                ),
                "evaluations": evaluations,
                "episodes": episodes,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "expected_score_percent": score,
                "score_ci95_low": ci_low,
                "score_ci95_high": ci_high,
                "goal_difference_per_episode": (
                    (wins - losses) / episodes if episodes else float("nan")
                ),
                "goals_for_per_episode": (
                    wins / episodes if episodes else float("nan")
                ),
                "goals_against_per_episode": (
                    losses / episodes if episodes else float("nan")
                ),
                "mean_reward_per_step": (
                    total_reward / rollout_steps
                    if rollout_steps
                    else float("nan")
                ),
            }
        )
    return sorted(
        aggregates,
        key=lambda row: (row["method"], int(row["checkpoint"])),
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write an empty CSV to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(path, aggregates, method_order, x_axis):
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
    plotted = 0
    for index, method in enumerate(method_order):
        rows = [
            row
            for row in aggregates
            if row["method"] == method
            and math.isfinite(float(row["expected_score_percent"]))
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: int(row["training_iteration"]))
        if x_axis == "environment-steps":
            x = [float(row["training_environment_steps"]) / 1e6 for row in rows]
        else:
            x = [int(row["training_iteration"]) for row in rows]
        y = [float(row["expected_score_percent"]) for row in rows]
        low = [float(row["score_ci95_low"]) for row in rows]
        high = [float(row["score_ci95_high"]) for row in rows]
        color = colors[index % len(colors)]
        axis.plot(x, y, marker="o", linewidth=2.2, markersize=5, label=method, color=color)
        axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        plotted += 1
    if not plotted:
        raise RuntimeError("No completed episodes were available to plot")

    axis.axhline(50.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Expected match score vs frozen MAPPO-FSP (%)")
    axis.set_xlabel(
        "High-level agent-environment transitions (millions)"
        if x_axis == "environment-steps"
        else "PPO training iteration"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def stream_subprocess(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_out_of_memory = False
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            normalized = line.lower()
            cuda_out_of_memory = cuda_out_of_memory or (
                "cuda" in normalized and "out of memory" in normalized
            )
        return_code = process.wait()
    if return_code:
        hint = ""
        if cuda_out_of_memory:
            hint = (
                " CUDA ran out of memory; select another GPU with --cuda N "
                "(or --device cuda:N), or reduce --eval-num-envs."
            )
        raise RuntimeError(
            f"Evaluation command failed with exit code {return_code}.{hint} "
            f"See {log_path}"
        )


def run_and_summarize_rollout(
    command,
    log_path,
    metrics_path,
    expected_rows,
    expected_steps,
    expected_envs,
):
    """Run an evaluator, accepting complete output from a teardown crash."""

    try:
        stream_subprocess(command, log_path)
    except RuntimeError as process_error:
        # Isaac Gym can segfault while tearing down its native context after
        # the rollout has already been flushed to disk (typically reported as
        # exit code -11). Recover only when the CSV proves that the requested
        # rollout completed in full.
        try:
            summary = summarize_rollout(
                metrics_path,
                expected_rows,
                expected_steps,
                expected_envs,
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            csv.Error,
        ) as summary_error:
            # A process that fails before exporting any metrics (for example,
            # during CUDA allocation) should report the process failure, not a
            # distracting secondary FileNotFoundError for the expected CSV.
            if isinstance(summary_error, FileNotFoundError):
                raise process_error from None
            raise process_error from summary_error
        print(
            "Warning: evaluator exited nonzero after producing a complete, "
            "valid rollout; accepting the CSV.",
            flush=True,
        )
        return summary
    return summarize_rollout(
        metrics_path,
        expected_rows,
        expected_steps,
        expected_envs,
    )


def build_methods(args):
    configured = parse_labeled_values(args.method, Path)
    scale_overrides = parse_labeled_values(args.steps_per_iteration, int)
    offsets = parse_labeled_values(args.step_offset, int)
    unknown = (set(scale_overrides) | set(offsets)) - set(configured)
    if unknown:
        raise ValueError(f"Overrides refer to unknown methods: {sorted(unknown)}")

    methods = []
    for label, configured_path in configured.items():
        policy_dir = configured_path.expanduser().resolve()
        if not policy_dir.is_dir():
            raise FileNotFoundError(f"Policy directory for {label} does not exist: {policy_dir}")
        scale = scale_overrides.get(label)
        if scale is None:
            scale = timesteps_per_iteration(policy_dir)
        if scale < 1:
            raise ValueError(f"steps-per-iteration for {label} must be positive")
        offset = offsets.get(label, 0)
        if offset < 0:
            raise ValueError(f"step offset for {label} cannot be negative")
        available = numbered_checkpoints(policy_dir)
        if not available:
            raise FileNotFoundError(
                f"No complete numbered checkpoints found for {label} in {policy_dir}"
            )
        methods.append(
            {
                "label": label,
                "policy_dir": policy_dir,
                "checkpoints": resolve_candidate_checkpoints(
                    args.candidate_checkpoints, available, label
                ),
                "steps_per_iteration": scale,
                "step_offset": offset,
            }
        )
    return methods


def run(args):
    methods = build_methods(args)
    opponent_dir = Path(args.opponent_dir).expanduser().resolve()
    if not opponent_dir.is_dir():
        raise FileNotFoundError(f"Opponent directory does not exist: {opponent_dir}")
    opponents = resolve_opponents(args.opponent_checkpoints, opponent_dir)
    seeds = [int(token) for token in parse_csv_tokens(args.seeds)]
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if args.eval_num_envs < 1 or args.steps < 1:
        raise ValueError("--eval-num-envs and --steps must be positive")
    print(
        f"Evaluation simulator device: {args.device}; "
        f"policy device: {args.policy_device}",
        flush=True,
    )

    skill_dirs = {
        "walk": Path(args.walk_policy_dir).expanduser().resolve(),
        "dribble": Path(args.dribble_policy_dir).expanduser().resolve(),
        "shoot": Path(args.shoot_policy_dir).expanduser().resolve(),
    }
    for skill, directory in skill_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{skill} policy directory does not exist: {directory}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    benchmark_slug = (
        f"steps_{args.steps}_envs_{args.eval_num_envs}_"
        f"domain_rand_{int(args.domain_rand)}_fixed_init_{int(args.fixed_init)}"
    )
    rollout_root = output_dir / "rollouts" / benchmark_slug
    rollout_root.mkdir(parents=True, exist_ok=True)
    expected_rows = args.steps * args.eval_num_envs
    total_runs = sum(len(method["checkpoints"]) for method in methods)
    total_runs *= len(opponents) * len(seeds)
    run_index = 0
    evaluations = []

    for method in methods:
        label = method["label"]
        method_slug = filename_slug(label)
        for checkpoint in method["checkpoints"]:
            iteration = int(checkpoint)
            training_steps = method["step_offset"] + (
                iteration + 1
            ) * method["steps_per_iteration"]
            for opponent_name, opponent_path in opponents:
                for seed in seeds:
                    run_index += 1
                    stem = f"opponent_{filename_slug(opponent_name)}_seed_{seed}"
                    rollout_dir = (
                        rollout_root / method_slug / f"checkpoint_{checkpoint}"
                    )
                    metrics_path = rollout_dir / f"{stem}.csv"
                    log_path = rollout_dir / f"{stem}.log"
                    reused = False
                    if metrics_path.is_file() and not args.overwrite:
                        try:
                            summary = summarize_rollout(
                                metrics_path,
                                expected_rows,
                                args.steps,
                                args.eval_num_envs,
                            )
                            reused = True
                        except (OSError, RuntimeError, ValueError):
                            reused = False
                    if reused:
                        print(
                            f"[{run_index}/{total_runs}] reuse {label} checkpoint={checkpoint} "
                            f"opponent={opponent_name} seed={seed}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{run_index}/{total_runs}] evaluate {label} checkpoint={checkpoint} "
                            f"opponent={opponent_name} seed={seed}",
                            flush=True,
                        )
                        command = [
                            args.python,
                            str(ROOT / "scripts" / "play_high_level.py"),
                            "--num-robots",
                            str(args.num_robots),
                            "--num-envs",
                            str(args.eval_num_envs),
                            "--export-all-envs",
                            "--high-level-policy-source",
                            "local",
                            "--high-level-policy-dir",
                            str(method["policy_dir"]),
                            "--high-level-checkpoint",
                            checkpoint,
                            "--opponent-high-level-checkpoint",
                            str(opponent_path),
                            "--skill-policy-source",
                            "local",
                            "--walk-policy-dir",
                            str(skill_dirs["walk"]),
                            "--dribble-policy-dir",
                            str(skill_dirs["dribble"]),
                            "--shoot-policy-dir",
                            str(skill_dirs["shoot"]),
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
                        if args.domain_rand:
                            command.append("--domain-rand")
                        if args.fixed_init:
                            command.append("--fixed-init")
                        if args.allow_training_config_mismatch:
                            command.append("--allow-training-config-mismatch")
                        summary = run_and_summarize_rollout(
                            command,
                            log_path,
                            metrics_path,
                            expected_rows,
                            args.steps,
                            args.eval_num_envs,
                        )

                    evaluations.append(
                        {
                            "method": label,
                            "policy_dir": str(method["policy_dir"]),
                            "checkpoint": checkpoint,
                            "training_iteration": iteration,
                            "training_environment_steps": training_steps,
                            "opponent": opponent_name,
                            "opponent_path": str(opponent_path),
                            "seed": seed,
                            **summary,
                        }
                    )

    aggregates = aggregate_evaluations(evaluations)
    write_csv(output_dir / "evaluations.csv", evaluations)
    write_csv(output_dir / "learning_curve.csv", aggregates)
    plot_path = output_dir / "learning_curves.png"
    plot_learning_curves(
        plot_path,
        aggregates,
        [method["label"] for method in methods],
        args.x_axis,
    )
    metadata = {
        "primary_metric": "expected_match_score_percent",
        "metric_definition": "100 * (wins + 0.5 * draws) / completed episodes",
        "metric_rationale": (
            "Direct task outcome on a shared frozen benchmark; unlike shaped "
            "training reward it is not changed by MPC or KL auxiliary terms."
        ),
        "confidence_interval": (
            "Wilson 95% interval using wins + 0.5 * draws as effective successes."
        ),
        "x_axis": args.x_axis,
        "methods": [
            {
                "label": method["label"],
                "policy_dir": str(method["policy_dir"]),
                "checkpoints": method["checkpoints"],
                "steps_per_iteration": method["steps_per_iteration"],
                "step_offset": method["step_offset"],
            }
            for method in methods
        ],
        "opponents": [
            {"name": name, "path": str(path)} for name, path in opponents
        ],
        "seeds": seeds,
        "steps_per_rollout": args.steps,
        "parallel_eval_matches": args.eval_num_envs,
        "simulator_device": args.device,
        "policy_device": args.policy_device,
        "domain_randomization": bool(args.domain_rand),
        "fixed_initialization": bool(args.fixed_init),
    }
    (output_dir / "evaluation_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Learning-curve plot: {plot_path}")
    print(f"Aggregated metrics: {output_dir / 'learning_curve.csv'}")
    print(f"Raw evaluation metrics: {output_dir / 'evaluations.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="LABEL=POLICY_DIR",
        help=(
            "Method label and directory containing numbered ac_weights/body/"
            "adaptation_module exports. Repeat once per method."
        ),
    )
    parser.add_argument(
        "--opponent-dir",
        required=True,
        help="MAPPO-FSP directory providing the common frozen opponent checkpoint(s).",
    )
    parser.add_argument(
        "--candidate-checkpoints",
        default="auto",
        help="Comma-separated numbered checkpoints, or auto for all complete exports.",
    )
    parser.add_argument(
        "--opponent-checkpoints",
        default="auto",
        help=(
            "Comma-separated checkpoint suffixes/paths; auto uses the final numbered "
            "MAPPO-FSP checkpoint, and suite uses early/middle/final checkpoints."
        ),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--eval-num-envs", type=int, default=16)
    parser.add_argument("--num-robots", type=int, default=2)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default="cuda:0",
        help="Simulator device string, for example cuda:6 or cpu.",
    )
    device_group.add_argument(
        "--cuda",
        "--cuda-device",
        dest="cuda_index",
        type=int,
        metavar="N",
        help="Use CUDA device N (equivalent to --device cuda:N).",
    )
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument(
        "--fixed-init",
        action="store_true",
        help="Use the same deterministic match initialization for every evaluation seed.",
    )
    parser.add_argument(
        "--x-axis",
        choices=("environment-steps", "iterations"),
        default="environment-steps",
    )
    parser.add_argument(
        "--steps-per-iteration",
        action="append",
        default=[],
        metavar="LABEL=N",
        help="Override config-derived high-level agent transitions per PPO iteration.",
    )
    parser.add_argument(
        "--step-offset",
        action="append",
        default=[],
        metavar="LABEL=N",
        help="Add prior-training transitions to a method's x-axis values.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with Isaac Gym available (defaults to this interpreter).",
    )
    parser.add_argument("--walk-policy-dir", default="checkpoints/reproduction/walk")
    parser.add_argument("--dribble-policy-dir", default="checkpoints/reproduction/dribble")
    parser.add_argument("--shoot-policy-dir", default="checkpoints/reproduction/shoot")
    parser.add_argument("--output-dir", default="outputs/learning_curve_comparison")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-training-config-mismatch", action="store_true")
    args = parser.parse_args()
    if args.cuda_index is not None:
        if args.cuda_index < 0:
            parser.error("--cuda must be a non-negative device index")
        args.device = f"cuda:{args.cuda_index}"
    return args


if __name__ == "__main__":
    run(parse_args())
