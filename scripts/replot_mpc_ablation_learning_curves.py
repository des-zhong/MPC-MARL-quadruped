"""Replot MPC ablations from saved CSVs without running the simulator."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_high_level_learning_curves import (
    aggregate_evaluations, goal_rate_and_confidence, plot_learning_curves,
    plot_termination_rates,
)


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"No evaluation rows in {path}")
    # Raw evaluations have one row per pairing; aggregate counts before rates.
    if "opponent" in fields:
        rows = aggregate_evaluations(rows)
    # Recompute from counts, also supporting older aggregates without win CIs.
    for row in rows:
        rate, low, high = goal_rate_and_confidence(int(row["wins"]), int(row["episodes"]))
        row.update(win_rate=rate, win_rate_ci95_low=low, win_rate_ci95_high=high)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("outputs/mpc_ablation_learning_curves/learning_curve.csv"),
                        help="learning_curve.csv or evaluations.csv from the comparison.")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to a replot subdirectory next to the input CSV.")
    parser.add_argument("--metric", choices=("win-rate", "goal-rate", "expected-score"),
                        default="win-rate")
    parser.add_argument("--x-axis", choices=("environment-steps", "iteration", "iterations"),
                        default="environment-steps")
    parser.add_argument("--font-size", type=float, default=10.0,
                        help="Font size in points (default: 10).")
    parser.add_argument("--figure-width", type=float, default=9.0,
                        help="Figure width in inches (default: 8).")
    parser.add_argument("--figure-height", type=float, default=4.0,
                        help="Figure height in inches (default: 4.2).")
    args = parser.parse_args(argv)
    for name in ("font_size", "figure_width", "figure_height"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a positive finite number")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        key: args.font_size for key in (
            "font.size", "axes.labelsize", "axes.titlesize",
            "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
        )
    })
    rows = load_rows(args.input)
    output = args.output_dir or args.input.parent / "replot"
    plot_path = output / "learning_curves.png"
    plot_learning_curves(plot_path, rows, list(dict.fromkeys(row["method"] for row in rows)),
                         args.x_axis, args.metric, figsize=(args.figure_width, args.figure_height))
    termination_path = output / "termination_rates.png"
    plot_termination_rates(termination_path, rows,
                           list(dict.fromkeys(row["method"] for row in rows)),
                           args.x_axis, figsize=(args.figure_width, args.figure_height))
    print(f"Wrote {plot_path} and {plot_path.with_suffix('.pdf')}")
    print(f"Wrote {termination_path} and {termination_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
