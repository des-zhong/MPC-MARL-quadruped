import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.analyze_ablation_study import (
    DEFAULT_METHODS,
    bootstrap_groups,
    build_evaluation_index,
    curve_metrics,
    mean_curve_score,
    read_evaluations,
    run,
    steps_to_target,
)


class AblationStudyAnalysisTests(unittest.TestCase):
    def test_curve_metrics_interpolate_common_budget_and_target(self):
        curve = [(0.0, 20.0), (100.0, 40.0), (200.0, 80.0)]

        self.assertAlmostEqual(mean_curve_score(curve, 50.0, 150.0), 42.5)
        self.assertAlmostEqual(
            steps_to_target(curve, 0.0, 200.0, 60.0), 150.0
        )
        metrics = curve_metrics(curve, 50.0, 150.0, 60.0)
        self.assertAlmostEqual(metrics["final_score_percent"], 60.0)
        self.assertAlmostEqual(metrics["mean_curve_score_percent"], 42.5)
        self.assertAlmostEqual(metrics["steps_to_target"], 150.0)

    def test_target_can_require_sustained_checkpoints(self):
        curve = [
            (0.0, 40.0),
            (100.0, 70.0),
            (200.0, 50.0),
            (300.0, 80.0),
            (400.0, 90.0),
        ]

        self.assertAlmostEqual(
            steps_to_target(curve, 0.0, 400.0, 60.0, consecutive=1),
            200.0 / 3.0,
        )
        self.assertAlmostEqual(
            steps_to_target(curve, 0.0, 400.0, 60.0, consecutive=2),
            700.0 / 3.0,
        )

    def test_bootstrap_clusters_repeated_training_runs(self):
        keys = [
            (training_run, opponent, seed)
            for training_run in ("train0", "train1")
            for opponent in ("early", "final")
            for seed in ("0", "1")
        ]

        groups, unit = bootstrap_groups(keys)

        self.assertEqual(unit, "training_run")
        self.assertEqual(len(groups), 2)
        self.assertEqual([len(group) for group in groups], [4, 4])

    def test_complete_analysis_writes_table_and_supports_synthetic_claims(self):
        scores = {
            "MAPPO-FSP": (20, 30, 40),
            "Ours w/o Terminal Value": (30, 50, 70),
            "Ours": (45, 70, 85),
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            evaluations = root / "evaluations.csv"
            rows = []
            for method, curve_scores in scores.items():
                for checkpoint, score in enumerate(curve_scores):
                    for seed in (0, 1):
                        # Twenty decisive matches per block make every listed
                        # expected score exactly representable.
                        wins = int(score / 5)
                        rows.append(
                            {
                                "method": method,
                                "checkpoint": checkpoint,
                                "training_environment_steps": checkpoint
                                * 1_000_000,
                                "opponent": "frozen_final",
                                "seed": seed,
                                "episodes": 20,
                                "wins": wins,
                                "draws": 0,
                                "losses": 20 - wins,
                            }
                        )
            with evaluations.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output = root / "analysis"
            payload = run(
                SimpleNamespace(
                    evaluations=str(evaluations),
                    method_order=",".join(DEFAULT_METHODS),
                    target_score=60.0,
                    target_consecutive=1,
                    bootstrap_samples=100,
                    confidence=0.95,
                    seed=7,
                    bold_method="Ours",
                    output_dir=str(output),
                )
            )

            self.assertTrue((output / "ablation_table.csv").is_file())
            self.assertTrue((output / "ablation_table.tex").is_file())
            self.assertTrue((output / "claim_report.md").is_file())
            parsed = json.loads(
                (output / "ablation_results.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            [claim["verdict"] for claim in payload["claims"]],
            ["supported", "supported"],
        )
        self.assertEqual(parsed["metadata"]["common_step_end"], 2_000_000)
        self.assertAlmostEqual(
            payload["methods"]["Ours w/o Terminal Value"]["steps_to_target"],
            1_500_000,
        )

    def test_mismatched_benchmark_blocks_are_rejected(self):
        rows = []
        for method in DEFAULT_METHODS:
            for checkpoint in (0, 1):
                for seed in (0, 1):
                    if method == "Ours" and checkpoint == 1 and seed == 1:
                        continue
                    rows.append(
                        {
                            "method": method,
                            "checkpoint": str(checkpoint),
                            "steps": float(checkpoint),
                            "key": ("0", "opponent", str(seed)),
                            "outcome": (2, 1, 0, 1),
                        }
                    )

        with self.assertRaisesRegex(ValueError, "Benchmark blocks differ"):
            build_evaluation_index(rows, DEFAULT_METHODS)

    def test_read_rejects_inconsistent_outcome_counts(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "bad.csv"
            row = {
                "method": "MAPPO-FSP",
                "checkpoint": 0,
                "training_environment_steps": 1,
                "opponent": "opponent",
                "seed": 0,
                "episodes": 3,
                "wins": 1,
                "draws": 1,
                "losses": 0,
            }
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "W/D/L sum"):
                read_evaluations(path)

    def test_aggregate_learning_curve_remains_descriptive(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            path = root / "learning_curve.csv"
            rows = []
            for method_index, method in enumerate(DEFAULT_METHODS):
                for checkpoint in (0, 1):
                    rows.append(
                        {
                            "method": method,
                            "checkpoint": checkpoint,
                            "training_environment_steps": checkpoint + 1,
                            "expected_score_percent": 20
                            + 10 * checkpoint
                            + method_index,
                        }
                    )
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            payload = run(
                SimpleNamespace(
                    evaluations=str(path),
                    method_order=",".join(DEFAULT_METHODS),
                    target_score=25.0,
                    target_consecutive=1,
                    bootstrap_samples=20,
                    confidence=0.95,
                    seed=3,
                    bold_method="Ours",
                    output_dir=str(root / "out"),
                )
            )

        self.assertEqual(
            payload["metadata"]["bootstrap_samples_performed"], 0
        )
        self.assertEqual(
            payload["claims"][1]["verdict"], "directionally_consistent"
        )


if __name__ == "__main__":
    unittest.main()
