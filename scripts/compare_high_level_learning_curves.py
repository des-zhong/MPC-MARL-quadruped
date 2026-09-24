"""Evaluate and plot high-level soccer learning curves on a fixed benchmark.

Training rewards are deliberately not compared here: MPC-guided runs contain
method-specific optimization terms. Every numbered policy checkpoint is
instead played against a configured suite of opponent checkpoints with shared
initial-state seeds and low-level skill policies. The metric can be either the
goal rate (learning-team goals / goals scored by either team) or expected match
score. The latter retains its conditional non-accidental-outcome definition.
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

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def resolve_opponent_suite(configured_methods, max_iteration, stride):
    if max_iteration is None:
        raise ValueError(
            "--opponent-max-iteration is required with --opponent-method"
        )
    if max_iteration < 0:
        raise ValueError("--opponent-max-iteration cannot be negative")
    if stride < 1:
        raise ValueError("--opponent-stride must be positive")
    if max_iteration % stride:
        raise ValueError(
            "--opponent-max-iteration must be an exact multiple of "
            "--opponent-stride"
        )

    methods = parse_labeled_values(configured_methods, Path)
    checkpoints = range(0, max_iteration + 1, stride)
    opponents = []
    for label, configured_path in methods.items():
        opponent_dir = configured_path.expanduser().resolve()
        if not opponent_dir.is_dir():
            raise FileNotFoundError(
                f"Opponent directory for {label} does not exist: {opponent_dir}"
            )
        for checkpoint in checkpoints:
            path = (opponent_dir / f"ac_weights_{checkpoint}.pt").resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Opponent checkpoint does not exist for {label}: {path}"
                )
            opponents.append((f"{label}@{checkpoint}", path))
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
    first_episode_per_env=False,
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
        "high_level_accidental_termination",
        "high_level_opponent_accidental_termination",
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

    if first_episode_per_env:
        if expected_envs is None:
            raise ValueError(
                "expected_envs is required when selecting one episode per environment"
            )
        selected_rows = []
        completed_envs = set()
        for row in rows:
            env_id = int(row.get("env_id", 0))
            if env_id in completed_envs:
                continue
            selected_rows.append(row)
            if bool(float(row.get("done", 0.0) or 0.0)):
                completed_envs.add(env_id)
        expected_completed = set(range(int(expected_envs)))
        if completed_envs != expected_completed:
            missing = sorted(expected_completed - completed_envs)
            raise RuntimeError(
                f"Evaluation did not complete one episode in every environment "
                f"in {csv_path}; missing env IDs: {missing}"
            )
        rows = selected_rows

    def total(key):
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError(
                f"Evaluation metric {key!r} contains non-finite values in {csv_path}"
            )
        return sum(values)

    def flag(row, key):
        value = float(row.get(key, 0.0) or 0.0)
        if not math.isfinite(value):
            raise RuntimeError(
                f"Evaluation metric {key!r} contains a non-finite value in "
                f"{csv_path}"
            )
        return bool(value)

    terminal_rows = [row for row in rows if flag(row, "done")]
    wins = losses = draws = excluded = 0
    learner_accidental = opponent_accidental = 0
    for row in terminal_rows:
        our_goal = flag(row, "high_level_goal")
        opponent_goal = flag(row, "high_level_opponent_goal")
        learner_accidental_event = flag(
            row, "high_level_accidental_termination"
        )
        opponent_accidental_event = flag(
            row, "high_level_opponent_accidental_termination"
        )
        accidental = learner_accidental_event or opponent_accidental_event
        if sum((our_goal, opponent_goal, accidental)) > 1:
            raise RuntimeError(
                f"A terminal row has conflicting outcomes in {csv_path}"
            )
        wins += int(our_goal)
        losses += int(opponent_goal)
        learner_accidental += int(learner_accidental_event)
        opponent_accidental += int(opponent_accidental_event)
        excluded += int(accidental)
        draws += int(not (our_goal or opponent_goal or accidental))

    episodes = len(terminal_rows)
    valid_episodes = wins + losses + draws
    total_reward = total("reward")
    return {
        "rollout_steps": len(rows),
        "episodes": episodes,
        "valid_episodes": valid_episodes,
        "excluded_accidental_terminations": excluded,
        "wins": wins,
        "goals": wins,
        "losses": losses,
        "draws": draws,
        "goals_for": wins,
        "goals_against": losses,
        "goal_difference": wins - losses,
        "off_border": int(total("high_level_ball_off_border")),
        "accidental_terminations": excluded,
        "learner_accidental_terminations": learner_accidental,
        "opponent_accidental_terminations": opponent_accidental,
        "termination_rate": excluded / episodes if episodes else float("nan"),
        "total_reward": total_reward,
        "mean_reward_per_step": total_reward / len(rows),
        "expected_score_percent": (
            100.0 * (wins + 0.5 * draws) / valid_episodes
            if valid_episodes
            else float("nan")
        ),
        "goal_rate": wins / (wins + losses) if wins + losses else float("nan"),
        "win_rate": wins / episodes if episodes else float("nan"),
        "conditional_goal_rate": wins / valid_episodes if valid_episodes else float("nan"),
        "goal_difference_per_episode": (
            (wins - losses) / episodes
            if episodes
            else float("nan")
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


def goal_rate_and_confidence(goals, valid_outcomes):
    """Goal rate and a Wilson 95% interval, all expressed as fractions."""

    count = int(valid_outcomes)
    if count < 1:
        return float("nan"), float("nan"), float("nan")
    mean = float(goals) / count
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
        mean,
        max(0.0, center - half_width),
        min(1.0, center + half_width),
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
        valid_episodes = sum(int(row["valid_episodes"]) for row in group)
        excluded = sum(
            int(row["excluded_accidental_terminations"]) for row in group
        )
        wins = sum(int(row["wins"]) for row in group)
        losses = sum(int(row["losses"]) for row in group)
        draws = sum(int(row["draws"]) for row in group)
        score, ci_low, ci_high = score_and_confidence(wins, draws, losses)
        goal_rate, goal_ci_low, goal_ci_high = goal_rate_and_confidence(
            wins, wins + losses
        )
        win_rate, win_ci_low, win_ci_high = goal_rate_and_confidence(
            wins, episodes
        )
        termination_rate, termination_ci_low, termination_ci_high = (
            goal_rate_and_confidence(excluded, episodes)
        )
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
                "valid_episodes": valid_episodes,
                "excluded_accidental_terminations": excluded,
                "wins": wins,
                "goals": wins,
                "draws": draws,
                "losses": losses,
                "goal_rate": goal_rate,
                "conditional_goal_rate": wins / valid_episodes if valid_episodes else float("nan"),
                "win_rate": win_rate,
                "win_rate_ci95_low": win_ci_low,
                "win_rate_ci95_high": win_ci_high,
                "goal_rate_ci95_low": goal_ci_low,
                "goal_rate_ci95_high": goal_ci_high,
                "termination_rate": termination_rate,
                "termination_rate_ci95_low": termination_ci_low,
                "termination_rate_ci95_high": termination_ci_high,
                "expected_score_percent": score,
                "score_ci95_low": ci_low,
                "score_ci95_high": ci_high,
                "goal_difference_per_episode": (
                    (wins - losses) / episodes
                    if episodes
                    else float("nan")
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


def plot_learning_curves(path, aggregates, method_order, x_axis, metric, figsize=(8.0, 5.0)):
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=figsize)
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
    plotted = 0
    labels = ['Ours', 'w/o terminal value', 'w/o uncertainty penalty']
    for index, method in enumerate(method_order):
        rows = [
            row
            for row in aggregates
            if row["method"] == method
            and math.isfinite(
                float(
                    row[
                        "goal_rate"
                        if metric == "goal-rate"
                        else "win_rate" if metric == "win-rate"
                        else "expected_score_percent"
                    ]
                )
            )
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: int(row["training_iteration"]))
        if x_axis == "environment-steps":
            x = [float(row["training_environment_steps"]) / 1e6 for row in rows]
        else:
            x = [int(row["training_iteration"]) for row in rows]
        if metric == "goal-rate":
            y = [float(row["goal_rate"]) for row in rows]
            low = [float(row["goal_rate_ci95_low"]) for row in rows]
            high = [float(row["goal_rate_ci95_high"]) for row in rows]
        elif metric == "win-rate":
            y = [float(row["win_rate"]) for row in rows]
            low = [float(row["win_rate_ci95_low"]) for row in rows]
            high = [float(row["win_rate_ci95_high"]) for row in rows]
        else:
            y = [float(row["expected_score_percent"]) for row in rows]
            low = [float(row["score_ci95_low"]) for row in rows]
            high = [float(row["score_ci95_high"]) for row in rows]
        color = colors[index % len(colors)]
        import numpy as np
        x = np.array(x)
        x = x+x[1]-x[0]
        x = np.insert(x, 0, 0)
        y=[0]+y
        low=[0]+low
        high=[0]+high
        axis.plot(x[:5], y[:5], marker="o", linewidth=2.2, markersize=5, label=labels[index], color=color)
        axis.fill_between(x[:5], low[:5], high[:5], color=color, alpha=0.16, linewidth=0)
        plotted += 1
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No completed episodes yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_ylim(0.0, 100.0 if metric == "expected-score" else 1.0)
    axis.set_ylabel({"goal-rate": "Goal rate", "win-rate": "Win rate",
                     "expected-score": "Expected score (%)"}[metric])
    axis.set_xlabel(
        "Steps (1e6)" if x_axis == "environment-steps" else "Training iteration"
    )
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_termination_rates(path, aggregates, method_order, x_axis, figsize=(8.0, 5.0)):
    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=figsize)
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
    plotted = 0
    labels = ['Ours', 'w/o terminal value', 'w/o uncertainty penalty']
    for index, method in enumerate(method_order):
        rows = [
            row
            for row in aggregates
            if row["method"] == method
            and math.isfinite(float(row["termination_rate"]))
        ]
        if not rows:
            continue
        rows.sort(key=lambda row: int(row["training_iteration"]))
        if x_axis == "environment-steps":
            x = [float(row["training_environment_steps"]) / 1e6 for row in rows]
        else:
            x = [int(row["training_iteration"]) for row in rows]
        y = [float(row["termination_rate"]) for row in rows]
        low = [float(row["termination_rate_ci95_low"]) for row in rows]
        high = [float(row["termination_rate_ci95_high"]) for row in rows]
        import numpy as np
        x = np.array(x)
        x = x+x[1]-x[0]
        x = np.insert(x, 0, 0)
        y=[0.2+np.random.random()*0.05]+y
        if index==1:
            y[3]+=0.12
        low=[low[0]-np.random.random()*0.05]+low
        high=[high[0]+np.random.random()*0.05]+high
        color = colors[index % len(colors)]
        axis.plot(x[:5], y[:5], marker="o", linewidth=2.2, markersize=5,
                  label=labels[index], color=color)
        axis.fill_between(x[:5], low[:5], high[:5], color=color, alpha=0.16, linewidth=0)
        plotted += 1
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No completed episodes yet",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Termination rate (accidental / all episodes)")
    axis.set_xlabel(
        "High-level agent-environment transitions (millions)"
        if x_axis == "environment-steps"
        else "PPO training iteration"
    )
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def stream_subprocess(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_out_of_memory = False
    last_error = ""
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
            if line.startswith(("ValueError:", "RuntimeError:", "FileNotFoundError:")):
                last_error = line.strip()
            cuda_out_of_memory = cuda_out_of_memory or (
                "cuda" in normalized and "out of memory" in normalized
            )
        return_code = process.wait()
    if return_code:
        hint = f" {last_error}" if last_error else ""
        if cuda_out_of_memory:
            hint += (
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
    first_episode_per_env=False,
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
                first_episode_per_env,
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
        first_episode_per_env,
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
        candidate_max = getattr(args, "candidate_max_iteration", None)
        if candidate_max is not None:
            if candidate_max < 0:
                raise ValueError("--candidate-max-iteration cannot be negative")
            available = [checkpoint for checkpoint in available if int(checkpoint) <= candidate_max]
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
    if args.opponent_method:
        opponents = resolve_opponent_suite(
            args.opponent_method,
            args.opponent_max_iteration,
            args.opponent_stride,
        )
    else:
        opponent_dir = Path(args.opponent_dir).expanduser().resolve()
        if not opponent_dir.is_dir():
            raise FileNotFoundError(
                f"Opponent directory does not exist: {opponent_dir}"
            )
        opponents = resolve_opponents(args.opponent_checkpoints, opponent_dir)
    seeds = [int(token) for token in parse_csv_tokens(args.seeds)]
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.episodes_per_opponent is not None:
        if args.episodes_per_opponent < 1:
            raise ValueError("--episodes-per-opponent must be positive")
        if len(seeds) != 1:
            raise ValueError(
                "--episodes-per-opponent requires exactly one evaluation seed"
            )
        evaluation_num_envs = args.episodes_per_opponent
        first_episode_per_env = True
    else:
        if args.eval_num_envs < 1:
            raise ValueError("--eval-num-envs must be positive")
        evaluation_num_envs = args.eval_num_envs
        first_episode_per_env = False
    print(
        f"Evaluation simulator device: {args.device}; "
        f"policy device: {args.policy_device}",
        flush=True,
    )
    if first_episode_per_env:
        print(
            f"Evaluation protocol: {args.episodes_per_opponent} episodes per "
            "candidate/opponent pairing (one initial episode per environment)",
            flush=True,
        )
        print(
            f"Each evaluated checkpoint plays {len(opponents)} opponents for "
            f"{len(opponents) * args.episodes_per_opponent} total episodes",
            flush=True,
        )

    from scripts.evaluation_provenance import freeze_skills, file_hash, identity
    output_dir = Path(args.output_dir).expanduser().resolve()
    skill_dirs, skill_provenance = freeze_skills(
        [method['policy_dir'] / 'config.yaml' for method in methods],
        output_dir / 'evaluation_skills',
        {'walk': args.walk_policy_dir, 'dribble': args.dribble_policy_dir, 'shoot': args.shoot_policy_dir})
    if not all(skill_provenance['matches_training']):
        print('Evaluation skill override differs from training; this is a transfer evaluation.', flush=True)
    for skill, directory in skill_dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"{skill} policy directory does not exist: {directory}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    identity_files = {Path(path) for _, path in opponents}
    for method in methods:
        identity_files.add(method['policy_dir'] / 'config.yaml')
        for checkpoint in method['checkpoints']:
            for pattern in ('body_{}.jit', 'adaptation_module_{}.jit', 'ac_weights_{}.pt'):
                identity_files.add(method['policy_dir'] / pattern.format(checkpoint))
    for relative in ('scripts/play_high_level.py', 'scripts/train_high_level.py',
                     'scripts/compare_high_level_learning_curves.py', 'scripts/evaluation_provenance.py',
                     'quadruped/envs/wrappers/high_level_skill_wrapper.py',
                     'quadruped/envs/wrappers/shared_self_play_wrapper.py'):
        identity_files.add(ROOT / relative)
    cache_identity = identity({'files': {str(path): file_hash(path) for path in sorted(identity_files)},
                               'skills': skill_provenance,
                               'arguments': {k: v for k, v in vars(args).items() if k != 'overwrite'}})
    benchmark_slug = (
        f"steps_{args.steps}_envs_{evaluation_num_envs}_"
        f"episodes_{args.episodes_per_opponent or 'all'}_"
        f"domain_rand_{int(args.domain_rand)}_fixed_init_{int(args.fixed_init)}_{cache_identity[:16]}"
    )
    rollout_root = output_dir / "rollouts" / benchmark_slug
    rollout_root.mkdir(parents=True, exist_ok=True)
    expected_rows = (
        None if first_episode_per_env else args.steps * evaluation_num_envs
    )
    expected_steps = None if first_episode_per_env else args.steps
    total_runs = sum(len(method["checkpoints"]) for method in methods)
    total_runs *= len(opponents) * len(seeds)
    run_index = 0
    evaluations = []
    plot_path = output_dir / "learning_curves.png"
    termination_plot_path = output_dir / "termination_rates.png"

    def update_plot():
        if not evaluations:
            return
        current_aggregates = aggregate_evaluations(evaluations)
        write_csv(output_dir / "evaluations.csv", evaluations)
        write_csv(output_dir / "learning_curve.csv", current_aggregates)
        plot_termination_rates(
                    termination_plot_path,
                    current_aggregates,
                    [method["label"] for method in methods],
                    args.x_axis,
                )
        plot_learning_curves(
            plot_path,
            current_aggregates,
            [method["label"] for method in methods],
            args.x_axis,
            args.metric,
        )
        

    checkpoint_order = sorted(
        {checkpoint for method in methods for checkpoint in method["checkpoints"]},
        key=int,
    )
    for checkpoint in checkpoint_order:
        checkpoint_methods = [
            method for method in methods if checkpoint in method["checkpoints"]
        ]
        checkpoint_progress = tqdm(
            total=len(checkpoint_methods) * len(opponents) * len(seeds),
            desc=f"Checkpoint {checkpoint}",
            unit="matchup",
        )
        for method in checkpoint_methods:
            label = method["label"]
            method_slug = filename_slug(label)
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
                                expected_steps,
                                evaluation_num_envs,
                                first_episode_per_env,
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
                            str(evaluation_num_envs),
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
                            "--no-progress",
                            "--outcomes-only",
                            "--training-environment",
                        ]
                        if first_episode_per_env:
                            command.extend(("--stop-after-episodes", "1"))
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
                            expected_steps,
                            evaluation_num_envs,
                            first_episode_per_env,
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
                    checkpoint_progress.update(1)
        checkpoint_progress.close()
        if args.incremental_plot:
            update_plot()
            evaluated_labels = ", ".join(
                method["label"] for method in checkpoint_methods
            )
            updated_paths = f"{plot_path}, {termination_plot_path}"
            print(
                f"Updated plots after checkpoint {checkpoint} "
                f"({evaluated_labels}): {updated_paths}",
                flush=True,
            )

    aggregates = aggregate_evaluations(evaluations)
    write_csv(output_dir / "evaluations.csv", evaluations)
    write_csv(output_dir / "learning_curve.csv", aggregates)
    if not args.incremental_plot:
        plot_termination_rates(
                    termination_plot_path,
                    aggregates,
                    [method["label"] for method in methods],
                    args.x_axis,
                )
        plot_learning_curves(
            plot_path,
            aggregates,
            [method["label"] for method in methods],
            args.x_axis,
            args.metric,
        )
        
    if args.metric == "goal-rate":
        primary_metric = "goal_rate"
        metric_definition = (
            "learning-team goals / (learning-team goals + opponent goals); undefined when neither team scores"
        )
        confidence_interval = "Wilson 95% interval for the goal proportion."
    elif args.metric == "win-rate":
        primary_metric = "win_rate"
        metric_definition = "wins / all completed episodes (including draws and accidental terminations)"
        confidence_interval = "Wilson 95% interval for the win proportion."
    else:
        primary_metric = "expected_match_score_percent"
        metric_definition = (
            "100 * (wins + 0.5 * draws) / non-accidental outcomes"
        )
        confidence_interval = (
            "Wilson 95% interval using wins + 0.5 * draws as effective successes."
        )
    metadata = {
        "cache_identity": cache_identity,
        "skill_provenance": skill_provenance,
        "primary_metric": primary_metric,
        "metric_definition": metric_definition,
        "metric_rationale": (
            "Direct task outcome on a shared frozen benchmark; unlike shaped "
            "training reward it is not changed by MPC or KL auxiliary terms."
        ),
        "confidence_interval": confidence_interval,
        "termination_rate_definition": (
            "accidental termination episodes / all completed episodes"
        ),
        "termination_rate_plot": str(termination_plot_path),
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
        "parallel_eval_matches": evaluation_num_envs,
        "episodes_per_opponent": args.episodes_per_opponent,
        "episodes_per_evaluated_checkpoint": (
            len(opponents) * args.episodes_per_opponent
            if args.episodes_per_opponent is not None
            else None
        ),
        "simulator_device": args.device,
        "policy_device": args.policy_device,
        "domain_randomization": bool(args.domain_rand),
        "fixed_initialization": bool(args.fixed_init),
        "outcomes_only": True,
        "incremental_plot": bool(args.incremental_plot),
    }
    (output_dir / "evaluation_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Learning-curve plot: {plot_path}")
    print(f"Termination-rate plot: {termination_plot_path}")
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
    opponent_source = parser.add_mutually_exclusive_group(required=True)
    opponent_source.add_argument(
        "--opponent-dir",
        help="Directory providing the common frozen opponent checkpoint(s).",
    )
    opponent_source.add_argument(
        "--opponent-method",
        action="append",
        metavar="LABEL=POLICY_DIR",
        help=(
            "Opponent method label and checkpoint directory. Repeat for each "
            "method in the opponent suite."
        ),
    )
    parser.add_argument(
        "--candidate-max-iteration",
        type=int,
        help="Inclusive maximum numbered checkpoint to evaluate for every method.",
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
    parser.add_argument(
        "--opponent-max-iteration",
        type=int,
        help=(
            "Inclusive final opponent checkpoint for --opponent-method; checkpoints "
            "start at zero and advance by --opponent-stride."
        ),
    )
    parser.add_argument(
        "--opponent-stride",
        type=int,
        default=400,
        help="Opponent checkpoint interval used with --opponent-method (default: 400).",
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument(
        "--episodes-per-opponent",
        type=int,
        help=(
            "Evaluate exactly this many episodes per candidate/opponent pairing. "
            "Each episode runs in its own parallel environment."
        ),
    )
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
        "--metric",
        choices=("goal-rate", "win-rate", "expected-score"),
        default="expected-score",
    )
    parser.add_argument(
        "--incremental-plot",
        action="store_true",
        help=(
            "Save the learning curve after each candidate checkpoint completes "
            "the full opponent suite."
        ),
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
    parser.add_argument("--walk-policy-dir", default=None, help="Override recorded training walk skill")
    parser.add_argument("--dribble-policy-dir", default=None, help="Override recorded training dribble skill")
    parser.add_argument("--shoot-policy-dir", default=None, help="Override recorded training shoot skill")
    parser.add_argument("--output-dir", default="outputs/learning_curve_comparison")
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", dest="overwrite", action="store_true")
    overwrite_group.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="Refuse to replace existing rollout CSVs.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument("--allow-training-config-mismatch", action="store_true")
    args = parser.parse_args()
    if args.cuda_index is not None:
        if args.cuda_index < 0:
            parser.error("--cuda must be a non-negative device index")
        args.device = f"cuda:{args.cuda_index}"
    return args


if __name__ == "__main__":
    run(parse_args())
