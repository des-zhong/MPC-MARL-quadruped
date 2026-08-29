import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.compare_high_level_learning_curves import (
    aggregate_evaluations,
    numbered_checkpoints,
    parse_args,
    run_and_summarize_rollout,
    score_and_confidence,
    summarize_rollout,
    timesteps_per_iteration,
)


class LearningCurveComparisonTests(unittest.TestCase):
    def test_cuda_index_option_selects_requested_simulator_device(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "compare_high_level_learning_curves.py",
                "--method",
                "method=checkpoints",
                "--opponent-dir",
                "checkpoints",
                "--cuda",
                "6",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.device, "cuda:6")

    def test_numbered_checkpoints_requires_complete_policy_exports(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for name in (
                "ac_weights_0.pt",
                "body_0.jit",
                "adaptation_module_0.jit",
                "ac_weights_400.pt",
                "body_400.jit",
                "ac_weights_latest.pt",
            ):
                (root / name).touch()

            self.assertEqual(numbered_checkpoints(root), ["0"])

    def test_timesteps_per_iteration_uses_match_count_and_learning_team(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "config.yaml").write_text(
                """
RunnerArgs:
  value:
    num_steps_per_env: 24
Cfg:
  value:
    env:
      num_envs: 32
      num_team_robots: 2
""".strip(),
                encoding="utf-8",
            )

            self.assertEqual(timesteps_per_iteration(root), 1536)

    def test_rollout_summary_and_aggregate_use_expected_match_score(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "rollout.csv"
            rows = [
                {"done": 1, "high_level_goal": 1, "high_level_opponent_goal": 0, "reward": 2},
                {"done": 1, "high_level_goal": 0, "high_level_opponent_goal": 1, "reward": -1},
                {"done": 1, "high_level_goal": 0, "high_level_opponent_goal": 0, "reward": 0},
                {"done": 0, "high_level_goal": 0, "high_level_opponent_goal": 0, "reward": 1},
            ]
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            summary = summarize_rollout(path, expected_rows=4)

        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["draws"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertAlmostEqual(summary["expected_score_percent"], 50.0)
        self.assertAlmostEqual(summary["goal_difference_per_episode"], 0.0)

        evaluation = {
            "method": "full",
            "checkpoint": "400",
            "training_iteration": 400,
            "training_environment_steps": 1234,
            **summary,
        }
        aggregate = aggregate_evaluations([evaluation])[0]
        self.assertEqual(aggregate["episodes"], 3)
        self.assertAlmostEqual(aggregate["expected_score_percent"], 50.0)
        self.assertLessEqual(aggregate["score_ci95_low"], 50.0)
        self.assertGreaterEqual(aggregate["score_ci95_high"], 50.0)

    def test_score_confidence_remains_informative_at_boundary(self):
        score, low, high = score_and_confidence(wins=10, draws=0, losses=0)
        self.assertEqual(score, 100.0)
        self.assertGreater(low, 0.0)
        self.assertLess(low, 100.0)
        self.assertEqual(high, 100.0)

    def test_complete_csv_survives_evaluator_teardown_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            metrics_path = root / "rollout.csv"
            rows = [
                {
                    "step": step,
                    "env_id": env_id,
                    "done": int(step == 1),
                    "high_level_goal": int(step == 1 and env_id == 0),
                    "high_level_opponent_goal": 0,
                    "reward": 1,
                }
                for step in range(2)
                for env_id in range(2)
            ]
            with metrics_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with mock.patch(
                "scripts.compare_high_level_learning_curves.stream_subprocess",
                side_effect=RuntimeError("exit code -11"),
            ):
                summary = run_and_summarize_rollout(
                    ["unused"],
                    root / "rollout.log",
                    metrics_path,
                    expected_rows=4,
                    expected_steps=2,
                    expected_envs=2,
                )

        self.assertEqual(summary["rollout_steps"], 4)
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["wins"], 1)

    def test_missing_csv_does_not_mask_evaluator_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with mock.patch(
                "scripts.compare_high_level_learning_curves.stream_subprocess",
                side_effect=RuntimeError("CUDA ran out of memory"),
            ):
                with self.assertRaisesRegex(RuntimeError, "CUDA ran out of memory") as raised:
                    run_and_summarize_rollout(
                        ["unused"],
                        root / "rollout.log",
                        root / "missing.csv",
                        expected_rows=4,
                        expected_steps=2,
                        expected_envs=2,
                    )

        self.assertIsNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
