"""Build a paper-ready ablation table and test the incremental claims.

The preferred input is ``evaluations.csv`` produced by
``compare_high_level_learning_curves.py``.  All methods must have been run on
the same opponent/initialization-seed blocks.  An aggregate
``learning_curve.csv`` is also accepted, but its claim checks are descriptive
because paired evaluation blocks are no longer available.  The script
deliberately uses match outcomes rather than shaped training reward, whose
definition changes when auxiliary training objectives are added.

The reported metrics are:

* final expected match score at the largest common training budget;
* mean expected score over the common learning-curve interval (normalized
  area under the learning curve, or AULC);
* interpolated environment steps to a pre-declared target score.

Paired bootstrap intervals resample the common opponent/seed benchmark blocks.
They quantify evaluation variability.  Independent training runs are still
needed to make claims about training-seed variability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


DEFAULT_METHODS = (
    "MAPPO-FSP",
    "Ours w/o Terminal Value",
    "Ours",
)
UPWARD_METRICS = ("final_score_percent", "mean_curve_score_percent")


def parse_csv_tokens(value):
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _finite_float(row, name, row_number):
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Row {row_number} has no valid {name!r}: {row.get(name)!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"Row {row_number} has non-finite {name!r}: {value}")
    return value


def read_evaluations(path, training_run=None):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Evaluation CSV is empty: {path}")

    columns = set(rows[0])
    required = {"method", "checkpoint", "training_environment_steps"}
    missing = sorted(required - columns)
    if missing:
        raise ValueError(
            f"Evaluation CSV is missing required columns: {', '.join(missing)}"
        )
    outcome_columns = {"episodes", "wins", "draws", "losses"}
    has_outcomes = outcome_columns <= columns
    score_column = next(
        (
            column
            for column in (
                "expected_score_percent",
                "mean_score_percent",
                "score_percent",
                "win_rate",
                "score",
                "performance",
            )
            if column in columns
        ),
        None,
    )
    if not has_outcomes and score_column is None:
        raise ValueError(
            "Evaluation CSV needs episodes/wins/draws/losses (evaluations.csv) "
            "or a score column such as expected_score_percent (learning_curve.csv)"
        )

    parsed = []
    for row_number, row in enumerate(rows, start=2):
        method = str(row["method"]).strip()
        if not method:
            raise ValueError(f"Row {row_number} has an empty method label")
        steps = _finite_float(row, "training_environment_steps", row_number)
        if has_outcomes:
            episodes = int(_finite_float(row, "episodes", row_number))
            wins = int(_finite_float(row, "wins", row_number))
            draws = int(_finite_float(row, "draws", row_number))
            losses = int(_finite_float(row, "losses", row_number))
            if min(steps, episodes, wins, draws, losses) < 0:
                raise ValueError(f"Row {row_number} contains a negative count or step")
            if wins + draws + losses != episodes:
                raise ValueError(
                    f"Row {row_number} has episodes={episodes}, but W/D/L sum to "
                    f"{wins + draws + losses}"
                )
            outcome = (episodes, wins, draws, losses)
        else:
            score = _finite_float(row, score_column, row_number)
            if score_column in {"win_rate", "score", "performance"} and score <= 1.0:
                score *= 100.0
            if not 0.0 <= score <= 100.0:
                raise ValueError(
                    f"Row {row_number} score must lie in [0, 100], got {score}"
                )
            # A learning_curve.csv has already aggregated outcomes. Keep the
            # supplied score as a one-block pseudo-outcome; no binomial CI is
            # claimed for this format, but paired bootstrap over blocks remains
            # useful when several evaluation files/replicates are supplied.
            outcome = (1, 0.0, 0.0, 0.0, score)
        # training_run is optional for forward compatibility with concatenated
        # evaluations from repeated training seeds. It must be explicitly
        # supplied in that case so duplicate benchmark blocks remain distinct.
        key = (
            str(
                training_run
                if training_run is not None
                else row.get("training_run", "0")
            ).strip()
            or "0",
            str(row.get("opponent", "aggregate")).strip() or "aggregate",
            str(row.get("seed", "0")).strip() or "0",
        )
        parsed.append(
            {
                "method": method,
                "checkpoint": str(row["checkpoint"]).strip(),
                "steps": steps,
                "key": key,
                "outcome": outcome,
            }
        )
    return parsed


def read_evaluation_inputs(configured):
    """Read one PATH or repeated TRAINING_RUN=PATH inputs."""

    if configured is None:
        configured = []
    if isinstance(configured, (str, Path)):
        configured = [str(configured)]
    configured = list(configured) or ["outputs/ablation_study/evaluations.csv"]
    combined = []
    input_records = []
    explicit_runs = set()
    for item in configured:
        item = str(item)
        candidate = Path(item).expanduser()
        if "=" in item and not candidate.is_file():
            training_run, path_token = item.split("=", 1)
            training_run = training_run.strip()
            candidate = Path(path_token).expanduser()
            if not training_run:
                raise ValueError(f"Expected TRAINING_RUN=PATH, got {item!r}")
        else:
            training_run = None
        if len(configured) > 1 and training_run is None:
            raise ValueError(
                "Multiple --evaluations inputs must use TRAINING_RUN=PATH labels"
            )
        if training_run is not None:
            if training_run in explicit_runs:
                raise ValueError(f"Duplicate training-run label {training_run!r}")
            explicit_runs.add(training_run)
        resolved = candidate.resolve()
        combined.extend(read_evaluations(resolved, training_run=training_run))
        input_records.append(
            {"training_run": training_run or "from_csv", "path": str(resolved)}
        )
    return combined, input_records


def build_evaluation_index(rows, methods):
    """Return method -> step -> paired block -> W/D/L counts."""

    requested = set(methods)
    available = {row["method"] for row in rows}
    missing = [method for method in methods if method not in available]
    if missing:
        raise ValueError(
            "The following requested methods are absent from evaluations.csv: "
            + ", ".join(missing)
        )

    index = {method: {} for method in methods}
    checkpoint_steps = {}
    for row in rows:
        method = row["method"]
        if method not in requested:
            continue
        checkpoint_key = (method, row["checkpoint"])
        previous_step = checkpoint_steps.setdefault(checkpoint_key, row["steps"])
        if previous_step != row["steps"]:
            raise ValueError(
                f"{method} checkpoint {row['checkpoint']} has inconsistent "
                "training_environment_steps"
            )
        point = index[method].setdefault(row["steps"], {})
        if row["key"] in point:
            raise ValueError(
                f"Duplicate evaluation block for {method}, checkpoint "
                f"{row['checkpoint']}, key={row['key']}. Add a distinct "
                "training_run column before concatenating repeated runs."
            )
        point[row["key"]] = row["outcome"]

    reference_keys = None
    for method in methods:
        if len(index[method]) < 2:
            raise ValueError(
                f"{method} needs at least two evaluated checkpoints to measure "
                "learning efficiency"
            )
        for steps, point in index[method].items():
            keys = set(point)
            if reference_keys is None:
                reference_keys = keys
            if keys != reference_keys:
                missing_keys = sorted(reference_keys - keys)
                extra_keys = sorted(keys - reference_keys)
                raise ValueError(
                    f"Benchmark blocks differ at {method}, steps={steps:g}; "
                    f"missing={missing_keys}, extra={extra_keys}. Finish the "
                    "same opponent/seed suite for every checkpoint."
                )
    if not reference_keys:
        raise ValueError("No common opponent/seed benchmark blocks were found")
    return index, sorted(reference_keys)


def expected_score(outcomes):
    episodes = sum(item[0] for item in outcomes)
    if episodes < 1:
        raise ValueError("A benchmark aggregate contains no completed episodes")
    if any(len(item) >= 5 for item in outcomes):
        if not all(len(item) >= 5 for item in outcomes):
            raise ValueError("Cannot mix direct scores and outcome-count rows")
        # Direct learning-curve input has no outcome counts. The fifth tuple
        # element preserves its already-computed expected score.
        return sum(
            float(item[4]) * float(item[0]) for item in outcomes
        ) / episodes
    wins = sum(item[1] for item in outcomes)
    draws = sum(item[2] for item in outcomes)
    return 100.0 * (wins + 0.5 * draws) / episodes


def make_curve(method_index, sampled_keys):
    curve = []
    for steps in sorted(method_index):
        point = method_index[steps]
        curve.append(
            (steps, expected_score([point[key] for key in sampled_keys]))
        )
    return curve


def common_support(index, methods):
    start = max(min(index[method]) for method in methods)
    end = min(max(index[method]) for method in methods)
    if end <= start:
        raise ValueError(
            "Methods do not share a non-empty training-step interval; evaluate "
            "checkpoints over an overlapping environment-step budget"
        )
    return start, end


def interpolate(curve, x):
    if x < curve[0][0] or x > curve[-1][0]:
        raise ValueError(f"Cannot interpolate {x:g} outside curve support")
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x == x0:
            return y0
        if x <= x1:
            if x1 == x0:
                return y1
            fraction = (x - x0) / (x1 - x0)
            return y0 + fraction * (y1 - y0)
    return curve[-1][1]


def mean_curve_score(curve, start, end):
    points = [(start, interpolate(curve, start))]
    points.extend((x, y) for x, y in curve if start < x < end)
    points.append((end, interpolate(curve, end)))
    area = sum(
        0.5 * (y0 + y1) * (x1 - x0)
        for (x0, y0), (x1, y1) in zip(points, points[1:])
    )
    return area / (end - start)


def steps_to_target(curve, start, end, target, consecutive=1):
    """First linearly interpolated target crossing sustained at checkpoints."""

    points = [(start, interpolate(curve, start))]
    points.extend((x, y) for x, y in curve if start < x < end)
    points.append((end, interpolate(curve, end)))
    # Remove duplicate boundary points if start/end are exact checkpoints.
    deduplicated = []
    for point in points:
        if deduplicated and point[0] == deduplicated[-1][0]:
            deduplicated[-1] = point
        else:
            deduplicated.append(point)
    points = deduplicated

    for index, (x, y) in enumerate(points):
        if y < target:
            continue
        if index + consecutive > len(points):
            continue
        if any(points[offset][1] < target for offset in range(index, index + consecutive)):
            continue
        if index == 0 or points[index - 1][1] >= target:
            return x
        previous_x, previous_y = points[index - 1]
        if y == previous_y:
            return x
        fraction = (target - previous_y) / (y - previous_y)
        return previous_x + fraction * (x - previous_x)
    return None


def curve_metrics(curve, start, end, target, consecutive=1):
    return {
        "final_score_percent": interpolate(curve, end),
        "mean_curve_score_percent": mean_curve_score(curve, start, end),
        "steps_to_target": steps_to_target(
            curve, start, end, target, consecutive=consecutive
        ),
    }


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def confidence_interval(values, confidence):
    tail = (1.0 - confidence) / 2.0
    return [quantile(values, tail), quantile(values, 1.0 - tail)]


def bootstrap_metrics(
    index,
    methods,
    key_groups,
    start,
    end,
    target,
    consecutive,
    samples,
    seed,
):
    distributions = {
        method: {metric: [] for metric in (*UPWARD_METRICS, "steps_to_target")}
        for method in methods
    }
    # Resampling a single aggregate block would create a zero-width interval
    # and falsely look like statistical evidence. Keep single-curve inputs
    # descriptive instead.
    if len(key_groups) < 2:
        return distributions
    rng = random.Random(seed)
    for _ in range(samples):
        sampled_keys = []
        for _ in key_groups:
            sampled_keys.extend(key_groups[rng.randrange(len(key_groups))])
        for method in methods:
            metrics = curve_metrics(
                make_curve(index[method], sampled_keys),
                start,
                end,
                target,
                consecutive,
            )
            for metric, value in metrics.items():
                if value is not None:
                    distributions[method][metric].append(value)
    return distributions


def bootstrap_groups(keys):
    """Cluster by training run when replicated runs are available."""

    by_training_run = {}
    for key in keys:
        by_training_run.setdefault(key[0], []).append(key)
    if len(by_training_run) >= 2:
        benchmark_suite = None
        for training_run, run_keys in by_training_run.items():
            suite = {(key[1], key[2]) for key in run_keys}
            if benchmark_suite is None:
                benchmark_suite = suite
            elif suite != benchmark_suite:
                raise ValueError(
                    f"Training run {training_run!r} uses a different "
                    "opponent/seed suite; repeated runs must share one benchmark"
                )
        return [
            sorted(by_training_run[training_run])
            for training_run in sorted(by_training_run)
        ], "training_run"
    return [[key] for key in keys], "opponent_seed_evaluation_block"


def summarize_methods(
    index,
    methods,
    keys,
    start,
    end,
    target,
    consecutive,
    bootstrap,
    confidence,
):
    summaries = {}
    for method in methods:
        point = curve_metrics(
            make_curve(index[method], keys), start, end, target, consecutive
        )
        summary = dict(point)
        summary["target_reached"] = point["steps_to_target"] is not None
        for metric in UPWARD_METRICS:
            summary[f"{metric}_ci"] = confidence_interval(
                bootstrap[method][metric], confidence
            )
        finite_steps = bootstrap[method]["steps_to_target"]
        summary["steps_to_target_ci"] = confidence_interval(
            finite_steps, confidence
        )
        summary["target_reach_probability"] = (
            len(finite_steps) / len(bootstrap[method]["final_score_percent"])
            if bootstrap[method]["final_score_percent"]
            else None
        )
        summaries[method] = summary
    return summaries


def metric_delta_samples(bootstrap, baseline, candidate, metric):
    baseline_values = bootstrap[baseline][metric]
    candidate_values = bootstrap[candidate][metric]
    # Upward metrics are always present and bootstrap lists remain aligned.
    if metric in UPWARD_METRICS:
        return [
            candidate_value - baseline_value
            for baseline_value, candidate_value in zip(
                baseline_values, candidate_values
            )
        ]
    return []


def assess_delta(point, interval):
    if interval[0] is None:
        if point > 0:
            return "directionally_consistent"
        if point < 0:
            return "opposite_direction"
        return "no_difference"
    if interval[0] > 0:
        return "supported"
    if interval[1] < 0:
        return "opposite_direction"
    return "inconclusive"


def comparison_result(
    name,
    baseline,
    candidate,
    criteria,
    summaries,
    bootstrap,
    confidence,
):
    metrics = {}
    for metric in UPWARD_METRICS:
        point = summaries[candidate][metric] - summaries[baseline][metric]
        samples = metric_delta_samples(
            bootstrap, baseline, candidate, metric
        )
        interval = confidence_interval(samples, confidence)
        metrics[metric] = {
            "candidate_minus_baseline": point,
            "ci": interval,
            "assessment": assess_delta(point, interval),
        }

    baseline_steps = summaries[baseline]["steps_to_target"]
    candidate_steps = summaries[candidate]["steps_to_target"]
    metrics["steps_to_target"] = {
        "baseline": baseline_steps,
        "candidate": candidate_steps,
        "steps_saved": (
            baseline_steps - candidate_steps
            if baseline_steps is not None and candidate_steps is not None
            else None
        ),
        "baseline_reach_probability": summaries[baseline][
            "target_reach_probability"
        ],
        "candidate_reach_probability": summaries[candidate][
            "target_reach_probability"
        ],
    }
    assessments = [metrics[metric]["assessment"] for metric in criteria]
    if assessments and all(value == "supported" for value in assessments):
        verdict = "supported"
    elif any(value == "opposite_direction" for value in assessments):
        verdict = "opposite_direction"
    elif assessments and all(
        value == "directionally_consistent" for value in assessments
    ):
        verdict = "directionally_consistent"
    else:
        verdict = "inconclusive"
    return {
        "claim": name,
        "baseline": baseline,
        "candidate": candidate,
        "criteria": list(criteria),
        "verdict": verdict,
        "metrics": metrics,
    }


def build_claims(methods, summaries, bootstrap, confidence):
    # The baseline comparison is contextual. Only the no-terminal/full pair is
    # a controlled component ablation.
    return [
        comparison_result(
            "Full method improves over MAPPO-FSP",
            methods[0],
            methods[2],
            UPWARD_METRICS,
            summaries,
            bootstrap,
            confidence,
        ),
        comparison_result(
            "Terminal value improves final performance",
            methods[1],
            methods[2],
            ("final_score_percent",),
            summaries,
            bootstrap,
            confidence,
        ),
    ]


def _format_ci(point, interval, digits=1):
    if interval[0] is None:
        return f"{point:.{digits}f}"
    return (
        f"{point:.{digits}f} "
        f"[{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"
    )


def _format_steps(summary):
    value = summary["steps_to_target"]
    if value is None:
        return "NR"
    interval = summary["steps_to_target_ci"]
    value /= 1.0e6
    if interval[0] is None:
        return f"{value:.2f}"
    return f"{value:.2f} [{interval[0] / 1e6:.2f}, {interval[1] / 1e6:.2f}]"


def _tex_escape(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def write_csv(path, methods, summaries):
    rows = []
    for method in methods:
        summary = summaries[method]
        rows.append(
            {
                "method": method,
                "final_score_percent": summary["final_score_percent"],
                "final_score_ci_low": summary["final_score_percent_ci"][0],
                "final_score_ci_high": summary["final_score_percent_ci"][1],
                "mean_curve_score_percent": summary[
                    "mean_curve_score_percent"
                ],
                "mean_curve_score_ci_low": summary[
                    "mean_curve_score_percent_ci"
                ][0],
                "mean_curve_score_ci_high": summary[
                    "mean_curve_score_percent_ci"
                ][1],
                "steps_to_target": summary["steps_to_target"],
                "steps_to_target_ci_low": summary["steps_to_target_ci"][0],
                "steps_to_target_ci_high": summary["steps_to_target_ci"][1],
                "target_reach_probability": summary[
                    "target_reach_probability"
                ],
            }
        )
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path, methods, summaries, target, bold_method):
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        (
            r"Method & Final Expected Score (\%) $\uparrow$ & Mean Curve Score (\%) "
            rf"$\uparrow$ & Steps to {target:g}\% (M) $\downarrow$ \\"
        ),
        r"\midrule",
    ]
    for method in methods:
        summary = summaries[method]
        cells = [
            _tex_escape(method),
            _format_ci(
                summary["final_score_percent"],
                summary["final_score_percent_ci"],
            ),
            _format_ci(
                summary["mean_curve_score_percent"],
                summary["mean_curve_score_percent_ci"],
            ),
            _format_steps(summary),
        ]
        if method == bold_method:
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claim_report(path, methods, summaries, claims, metadata):
    target = metadata["target_score_percent"]
    if metadata["bootstrap_samples_performed"]:
        unit_label = (
            "training runs"
            if metadata["bootstrap_unit"] == "training_run"
            else "opponent/seed blocks"
        )
        uncertainty_text = (
            "Intervals are paired "
            f"{100.0 * metadata['confidence']:.0f}% bootstrap intervals over "
            f"{metadata['bootstrap_units']} {unit_label}."
        )
    else:
        uncertainty_text = (
            "Only one aggregate benchmark block was available, so intervals "
            "and inferential claim verdicts are intentionally omitted."
        )
    lines = [
        "# Ablation study",
        "",
        (
            f"Metrics use the common training interval "
            f"{metadata['common_step_start']:g}--{metadata['common_step_end']:g} "
            f"agent-environment transitions. {uncertainty_text}"
        ),
        "",
        "| Method | Final score (%) | Mean curve score (%) | "
        f"Steps to {target:g}% (M) |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        summary = summaries[method]
        lines.append(
            f"| {method} | "
            f"{_format_ci(summary['final_score_percent'], summary['final_score_percent_ci'])} | "
            f"{_format_ci(summary['mean_curve_score_percent'], summary['mean_curve_score_percent_ci'])} | "
            f"{_format_steps(summary)} |"
        )
    lines.extend(("", "## Claim checks", ""))
    display_metric = {
        "final_score_percent": "final-score delta",
        "mean_curve_score_percent": "mean-curve-score delta",
    }
    for claim in claims:
        lines.append(
            f"- **{claim['verdict'].replace('_', ' ').title()}** — "
            f"{claim['claim']} ({claim['baseline']} -> {claim['candidate']})."
        )
        for metric in claim["criteria"]:
            result = claim["metrics"][metric]
            lines.append(
                f"  {display_metric[metric]}: "
                f"{_format_ci(result['candidate_minus_baseline'], result['ci'])} "
                "percentage points."
            )
        steps = claim["metrics"]["steps_to_target"]
        if steps["steps_saved"] is not None:
            lines.append(
                f"  target-step difference: {steps['steps_saved'] / 1e6:.2f}M "
                "steps saved (descriptive)."
            )
        else:
            lines.append(
                "  target-step difference: not comparable because at least one "
                "method did not reach the target."
            )
    lines.extend(
        (
            "",
            "## Interpretation limits",
            "",
            "Expected score is 100 * (wins + 0.5 * draws) / completed matches. "
            "Mean curve score is normalized AULC and is the primary efficiency "
            "metric; steps-to-target is secondary because it depends on the "
            "chosen threshold.",
            "",
            "Sample efficiency here means real agent-environment transitions, "
            "not wall-clock or planner compute. Warm-start and method-specific "
            "real-data collection must be included in each method's step offset.",
            "",
            "The bootstrap resamples evaluation opponent/seed blocks. If each "
            "method has only one trained policy run, the intervals do not include "
            "training-seed variance. Repeat every training condition with multiple "
            "seeds and add a `training_run` column before concatenating CSV files "
            "for a paper-level learning claim.",
        )
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    methods = parse_csv_tokens(args.method_order)
    if len(methods) != 3:
        raise ValueError(
            "--method-order must contain exactly three labels in this order: "
            "MAPPO-FSP, full method without terminal value, full method"
        )
    if len(set(methods)) != len(methods):
        raise ValueError("--method-order labels must be unique")
    if not 0.0 <= args.target_score <= 100.0:
        raise ValueError("--target-score must lie in [0, 100]")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie in (0, 1)")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    if args.target_consecutive < 1:
        raise ValueError("--target-consecutive must be positive")

    rows, input_records = read_evaluation_inputs(args.evaluations)
    index, keys = build_evaluation_index(rows, methods)
    start, end = common_support(index, methods)
    key_groups, bootstrap_unit = bootstrap_groups(keys)
    bootstrap = bootstrap_metrics(
        index,
        methods,
        key_groups,
        start,
        end,
        args.target_score,
        args.target_consecutive,
        args.bootstrap_samples,
        args.seed,
    )
    summaries = summarize_methods(
        index,
        methods,
        keys,
        start,
        end,
        args.target_score,
        args.target_consecutive,
        bootstrap,
        args.confidence,
    )
    claims = build_claims(methods, summaries, bootstrap, args.confidence)
    metadata = {
        "inputs": input_records,
        "method_order": methods,
        "method_stage_semantics": {
            methods[0]: "MAPPO fictitious self-play baseline",
            methods[1]: "full method with only terminal value disabled",
            methods[2]: "full method with terminal value enabled",
        },
        "common_step_start": start,
        "common_step_end": end,
        "target_score_percent": args.target_score,
        "target_consecutive_checkpoints": args.target_consecutive,
        "benchmark_blocks": len(keys),
        "training_runs": len({key[0] for key in keys}),
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_units": len(key_groups),
        "bootstrap_samples_requested": args.bootstrap_samples,
        "bootstrap_samples_performed": (
            args.bootstrap_samples if len(key_groups) >= 2 else 0
        ),
        "bootstrap_seed": args.seed,
        "confidence": args.confidence,
        "primary_performance_metric": "final_score_percent",
        "primary_efficiency_metric": "mean_curve_score_percent",
        "secondary_efficiency_metric": "steps_to_target",
        "elo_omitted_because": (
            "Against one fixed reference population, Elo is only a monotonic "
            "log-odds transform of expected score and adds no independent evidence."
        ),
        "uncertainty_scope": (
            "Training-run clusters when at least two labeled training runs are "
            "supplied; otherwise paired opponent/seed evaluation blocks only."
        ),
    }
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "ablation_table.csv", methods, summaries)
    write_tex(
        output / "ablation_table.tex",
        methods,
        summaries,
        args.target_score,
        args.bold_method,
    )
    write_claim_report(
        output / "claim_report.md", methods, summaries, claims, metadata
    )
    payload = {
        "metadata": metadata,
        "methods": summaries,
        "claims": claims,
    }
    (output / "ablation_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for claim in claims:
        print(f"{claim['verdict']}: {claim['claim']}")
    print(f"LaTeX table: {output / 'ablation_table.tex'}")
    print(f"Claim report: {output / 'claim_report.md'}")
    if getattr(args, "fail_on_unsupported", False) and any(
        claim["verdict"] != "supported" for claim in claims
    ):
        raise SystemExit(1)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluations",
        action="append",
        default=[],
        metavar="[TRAINING_RUN=]PATH",
        help=(
            "Raw fixed-benchmark evaluations CSV. Repeat as TRAINING_RUN=PATH "
            "to combine independent training runs."
        ),
    )
    parser.add_argument(
        "--method-order",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Three comma-separated labels ordered as FSP,no-terminal,full."
        ),
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=60.0,
        help="Pre-declared expected match-score target in percent.",
    )
    parser.add_argument(
        "--target-consecutive",
        type=int,
        default=1,
        help="Number of consecutive evaluated points that must remain at target.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bold-method", default="Ours")
    parser.add_argument("--output-dir", default="outputs/ablation_study")
    parser.add_argument(
        "--fail-on-unsupported",
        action="store_true",
        help="Exit nonzero unless both pre-declared claim checks are supported.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
