"""Measure how much training each method needs to reach a stable benchmark score.

The input is produced by ``scripts/compare_high_level_learning_curves.py``.
For each method, this script finds the first checkpoint whose *expected match
score* (win = 1, draw = 0.5, loss = 0) has a Wilson 95% lower confidence bound
at or above the requested target for ``N`` consecutive checkpoints.  The
reported budget is the training environment-transition count at that
checkpoint (and the equivalent PPO iteration), so MAPPO and MPC are compared
on the same sample unit rather than on method-specific shaped rewards.

Both ``learning_curve.csv`` (preferred) and raw ``evaluations.csv`` files are
accepted.  Raw files are re-aggregated before the threshold test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def parse_labeled_paths(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=CSV_PATH, got {value!r}")
        label, path = value.split("=", 1)
        label, path = label.strip(), path.strip()
        if not label or not path:
            raise ValueError(f"Expected non-empty LABEL=CSV_PATH, got {value!r}")
        if label in result:
            raise ValueError(f"Duplicate method label: {label!r}")
        result[label] = Path(path).expanduser().resolve()
    if not result:
        raise ValueError("At least one --method LABEL=CSV_PATH is required")
    return result


def wilson_lower(successes: float, trials: float, z: float = 1.96) -> float:
    if trials <= 0:
        return float("nan")
    p = max(0.0, min(1.0, successes / trials))
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - half)


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        return default
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {key} value: {value!r}")
    return parsed


def read_rows(label: str, path: Path) -> list[dict[str, float | int | str]]:
    if path.is_dir():
        curve = path / "learning_curve.csv"
        raw = path / "evaluations.csv"
        path = curve if curve.is_file() else raw
    if not path.is_file():
        raise FileNotFoundError(f"CSV does not exist for {label}: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty for {label}: {path}")

    # compare_high_level_learning_curves.py writes one CSV containing all
    # methods. Reusing that file for multiple --method entries is supported
    # when labels match its ``method`` column.
    if "method" in rows[0]:
        available = sorted({row.get("method", "") for row in rows})
        matching = [row for row in rows if row.get("method") == label]
        if matching:
            rows = matching
        elif len(available) == 1:
            rows = rows
        else:
            raise ValueError(
                f"{path} contains methods {available}; no rows match label {label!r}"
            )

    fields = set(rows[0])
    if {"expected_score_percent", "score_ci95_low"}.issubset(fields):
        # Aggregated output from compare_high_level_learning_curves.py.
        result = []
        for row in rows:
            result.append({
                "checkpoint": int(_number(row, "checkpoint")),
                "training_iteration": int(_number(row, "training_iteration", _number(row, "checkpoint"))),
                "training_environment_steps": int(_number(row, "training_environment_steps")),
                "episodes": int(_number(row, "episodes")),
                "expected_score_percent": _number(row, "expected_score_percent"),
                "score_ci95_low": _number(row, "score_ci95_low"),
                "wins": int(_number(row, "wins")),
                "draws": int(_number(row, "draws")),
                "losses": int(_number(row, "losses")),
            })
        return result

    required = {"checkpoint", "training_iteration", "training_environment_steps", "wins", "draws", "losses"}
    missing = required - fields
    if missing:
        raise ValueError(f"{path} is neither a learning_curve.csv nor evaluations.csv; missing {sorted(missing)}")

    # Raw evaluations.csv may contain multiple opponents/seeds per checkpoint.
    grouped: dict[int, dict[str, float]] = {}
    for row in rows:
        checkpoint = int(_number(row, "checkpoint"))
        item = grouped.setdefault(checkpoint, {
            "training_iteration": _number(row, "training_iteration"),
            "training_environment_steps": _number(row, "training_environment_steps"),
            "episodes": 0.0, "wins": 0.0, "draws": 0.0, "losses": 0.0,
        })
        item["episodes"] += _number(row, "episodes")
        item["wins"] += _number(row, "wins")
        item["draws"] += _number(row, "draws")
        item["losses"] += _number(row, "losses")
    result = []
    for checkpoint, item in grouped.items():
        trials = item["wins"] + item["draws"] + item["losses"]
        score = (item["wins"] + 0.5 * item["draws"]) / trials if trials else float("nan")
        result.append({
            "checkpoint": checkpoint,
            "training_iteration": int(item["training_iteration"]),
            "training_environment_steps": int(item["training_environment_steps"]),
            "episodes": int(item["episodes"]),
            "expected_score_percent": 100.0 * score,
            "score_ci95_low": 100.0 * wilson_lower(item["wins"] + 0.5 * item["draws"], trials),
            "wins": int(item["wins"]), "draws": int(item["draws"]), "losses": int(item["losses"]),
        })
    return result


def find_attainment(rows: list[dict[str, float | int | str]], target_percent: float, consecutive: int) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: (int(row["training_iteration"]), int(row["checkpoint"])))
    qualifying = [float(row["score_ci95_low"]) >= target_percent for row in ordered]
    attainment_index = None
    for index in range(consecutive - 1, len(ordered)):
        if all(qualifying[index - consecutive + 1 : index + 1]):
            attainment_index = index
            break
    best = max(ordered, key=lambda row: float(row["expected_score_percent"]))
    result: dict[str, object] = {
        "target_expected_score_percent": target_percent,
        "required_consecutive_checkpoints": consecutive,
        "attained": attainment_index is not None,
        "best_expected_score_percent": float(best["expected_score_percent"]),
        "best_score_ci95_low": float(best["score_ci95_low"]),
        "best_checkpoint": int(best["checkpoint"]),
    }
    if attainment_index is not None:
        row = ordered[attainment_index]
        result.update({
            "attainment_checkpoint": int(row["checkpoint"]),
            "attainment_training_iteration": int(row["training_iteration"]),
            "attainment_training_environment_steps": int(row["training_environment_steps"]),
            "attainment_training_environment_steps_millions": float(row["training_environment_steps"]) / 1e6,
            "attainment_expected_score_percent": float(row["expected_score_percent"]),
            "attainment_score_ci95_low": float(row["score_ci95_low"]),
            "attainment_evaluation_episodes": int(row["episodes"]),
            "attainment_wins": int(row["wins"]),
            "attainment_draws": int(row["draws"]),
            "attainment_losses": int(row["losses"]),
        })
    else:
        result["attainment_training_environment_steps"] = None
        result["attainment_training_environment_steps_millions"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, metavar="LABEL=CSV_PATH", help="learning_curve.csv or evaluations.csv; repeat per method")
    parser.add_argument("--target-score", type=float, default=55.0, help="Expected match score target in percent (default: 55)")
    parser.add_argument("--consecutive-checkpoints", type=int, default=3, help="Number of consecutive checkpoints whose Wilson lower bound must reach the target (default: 3)")
    parser.add_argument("--output-dir", default="outputs/sample_efficiency")
    args = parser.parse_args()
    if not 0.0 < args.target_score < 100.0:
        parser.error("--target-score must be between 0 and 100")
    if args.consecutive_checkpoints < 1:
        parser.error("--consecutive-checkpoints must be positive")

    methods = parse_labeled_paths(args.method)
    summary = []
    curves = {}
    for label, path in methods.items():
        rows = read_rows(label, path)
        curves[label] = rows
        result = {"method": label, "source": str(path)}
        result.update(find_attainment(rows, args.target_score, args.consecutive_checkpoints))
        summary.append(result)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0])
    with (output_dir / "sample_efficiency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    protocol = {
        "metric": "expected_match_score_percent",
        "metric_definition": "100 * (wins + 0.5 * draws) / (wins + draws + losses)",
        "metric_rationale": "A direct task outcome on the same frozen benchmark; it does not include MAPPO/MPC-specific shaped rewards.",
        "stability_rule": f"Wilson 95% lower bound >= target for {args.consecutive_checkpoints} consecutive checkpoints",
        "target_score_percent": args.target_score,
        "methods": [dict(row) for row in summary],
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"Metric: expected match score; stable target: Wilson 95% lower bound >= {args.target_score:.1f}% for {args.consecutive_checkpoints} consecutive checkpoints")
    for row in summary:
        budget = row["attainment_training_environment_steps"]
        if budget is None:
            print(f"{row['method']}: target not reached (best lower bound {float(row['best_score_ci95_low']):.1f}%)")
        else:
            print(f"{row['method']}: {budget:,} training environment steps ({row['attainment_training_iteration']} iterations), checkpoint {row['attainment_checkpoint']}")
    print(f"Results: {output_dir / 'sample_efficiency.csv'}")


if __name__ == "__main__":
    main()
