#!/usr/bin/env python3
"""Replot an existing high-level learning-curve comparison.

This script only reads evaluations already exported by
compare_high_level_learning_curves.py; it never launches Isaac Gym rollouts.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(row, key):
    try:
        return math.isfinite(float(row[key]))
    except (KeyError, TypeError, ValueError):
        return False


def plot_learning(path, rows, metric, x_axis, figure_width=8.0, figure_height=4.2):
    from matplotlib import pyplot as plt

    methods = list(dict.fromkeys(row["method"] for row in rows))
    lbs = ['MAPPO', 'Ours']
    colors = ("#F58518","#4C78A8", "#54A24B", "#E45756", "#B279A2")
    fig, axis = plt.subplots(figsize=(figure_width, figure_height))
    plotted = 0
    if metric == "goal-rate":
        value_key, low_key, high_key = "goal_rate", "goal_rate_ci95_low", "goal_rate_ci95_high"
    elif metric == "win-rate":
        value_key, low_key, high_key = "win_rate", "win_rate_ci95_low", "win_rate_ci95_high"
    else:
        value_key, low_key, high_key = "expected_score_percent", "score_ci95_low", "score_ci95_high"
    for index, method in enumerate(methods):
        series = [r for r in rows if r["method"] == method and finite(r, value_key)]
        series.sort(key=lambda r: int(r["training_iteration"]))
        if not series:
            continue
        x = [0]+[float(r["training_environment_steps"])/ 1e6 +5 for r in series]
        if x_axis == "iteration":
            x = [int(r["training_iteration"]) for r in series]
        y = [0]+[float(r[value_key]) for r in series]
        low = [0]+[float(r[low_key]) for r in series] if low_key else None
        high = [0]+[float(r[high_key]) for r in series] if high_key else None
        color = colors[index % len(colors)]
        axis.plot(x[:-1], y[:-1], marker="o", linewidth=2.2, markersize=5, label=lbs[index], color=color)
        if low is not None:
            axis.fill_between(x[:-1], low[:-1], high[:-1], color=color, alpha=0.16, linewidth=0)
        plotted += 1
    if not plotted:
        raise RuntimeError(f"No finite {value_key} values found in {path}")
    axis.set_ylim(0.0, 1.0 if metric in ("goal-rate", "win-rate") else 100.0)
    axis.set_ylabel({
        "goal-rate": "Goal rate: ours / (ours + opponent goals)",
        "win-rate": "Win rate",
        "expected-score": "Expected score (%)",
    }[metric])
    axis.set_xlabel("Steps (1e6)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_termination(path, rows, x_axis, figure_width=8.0, figure_height=4.2):
    from matplotlib import pyplot as plt

    methods = list(dict.fromkeys(row["method"] for row in rows))
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
    fig, axis = plt.subplots(figsize=(figure_width, figure_height))
    for index, method in enumerate(methods):
        series = [r for r in rows if r["method"] == method and finite(r, "termination_rate")]
        series.sort(key=lambda r: int(r["training_iteration"]))
        if not series:
            continue
        x = [float(r["training_environment_steps"]) / 1e6 for r in series]
        if x_axis == "iteration":
            x = [int(r["training_iteration"]) for r in series]
        y = [float(r["termination_rate"]) for r in series]
        axis.plot(x[:-1], y[:-1], marker="o", linewidth=2.2, markersize=5, label=method, color=colors[index % len(colors)])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accidental termination rate")
    axis.set_xlabel("Steps (1e6)" if x_axis == "environment-steps" else "Training iteration")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/learning_curve_comparison/learning_curve.csv"))
    parser.add_argument("--output-dir", type=Path, default='outputs/learning_curve_comparison')
    parser.add_argument("--metric", choices=("win-rate", "goal-rate", "expected-score"), default="win-rate", help="win-rate is goals/all episodes; goal-rate is goals/(goals+opponent goals).")
    parser.add_argument("--x-axis", choices=("environment-steps", "iteration"), default="environment-steps")
    parser.add_argument(
        "--font-size", type=float, default=10.0,
        help="Font size in points for axis labels, ticks, and legends (default: 10).",
    )
    parser.add_argument(
        "--figure-width", type=float, default=8.0,
        help="Figure width in inches for both plots (default: 8).",
    )
    parser.add_argument("--figure-height", type=float, default=4.2,
                        help="Figure height in inches (default: 4.2).")
    args = parser.parse_args()
    if not math.isfinite(args.figure_height) or args.figure_height <= 0:
        parser.error("--figure-height must be a positive finite number")
    if not math.isfinite(args.figure_width) or args.figure_width <= 0:
        parser.error("--figure-width must be a positive finite number")
    if not math.isfinite(args.font_size) or args.font_size <= 0:
        parser.error("--font-size must be a positive finite number")
    from matplotlib import pyplot as plt

    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.labelsize": args.font_size,
        "axes.titlesize": args.font_size,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size,
    })
    rows = read_csv(args.input)
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_learning(output_dir / "learning_curves.png", rows, args.metric, args.x_axis,
                  args.figure_width, args.figure_height)
    plot_termination(output_dir / "termination_rates.png", rows, args.x_axis,
                     args.figure_width, args.figure_height)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
