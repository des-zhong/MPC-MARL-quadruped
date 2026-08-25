"""Run the resumable world-model and terminal-value training workflow."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dribblebot.mpc.workflow_paths import ModelWorkflowPaths


STAGES = (
    "all",
    "world_model",
    "terminal_bootstrap",
    "iterative",
    "terminal_final",
    "validate",
)
POLICY_ARGUMENTS = [
    "--walk-policy-dir",
    "checkpoints/reproduction/walk",
    "--dribble-policy-dir",
    "checkpoints/reproduction/dribble",
    "--shoot-policy-dir",
    "checkpoints/reproduction/shoot",
]


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.paths = ModelWorkflowPaths.for_run(args.run)

    def command(self, arguments: list[str]) -> None:
        command = [sys.executable, *arguments]
        print(f"[pipeline] {shlex.join(command)}", flush=True)
        if not self.args.dry_run:
            subprocess.run(command, check=True, cwd=ROOT, env=os.environ.copy())

    def completed(self, path: Path) -> bool:
        if path.is_file() and not self.args.force:
            print(f"[pipeline] already complete: {path}")
            return True
        return False

    def world_model(self) -> None:
        if self.completed(self.paths.world_model_checkpoint):
            return
        self.command(
            [
                "scripts/train_world_model.py",
                "--config",
                "configs/world_model_as2.yaml",
                "--dataset",
                str(self.paths.initial_dataset),
                "--output",
                str(self.paths.world_model_output),
                "--device",
                self.args.device,
                "--wandb-mode",
                self.args.wandb_mode,
            ]
        )

    def terminal_bootstrap(self) -> None:
        checkpoint = self.paths.bootstrap_value_output / "best.pt"
        if self.completed(checkpoint):
            return
        self.command(
            [
                "scripts/train_terminal_value.py",
                "--config",
                "configs/terminal_value.yaml",
                "--dataset",
                str(self.paths.initial_dataset),
                "--world-model-checkpoint",
                str(self.paths.world_model_checkpoint),
                "--output",
                str(self.paths.bootstrap_value_output),
                "--wandb-name",
                f"terminal-value-{self.args.run}-bootstrap",
                "--device",
                self.args.device,
                "--wandb-mode",
                self.args.wandb_mode,
            ]
        )

    def iterative(self) -> None:
        if self.completed(self.paths.working_root / "iterative_summary.json"):
            return
        self.command(
            [
                "scripts/run_iterative_mpc_world_model.py",
                "--config",
                "configs/iterative_mpc_world_model.yaml",
                "--mpc-config",
                "configs/mpc_joint_teams.yaml",
                "--initial-dataset",
                str(self.paths.initial_dataset),
                "--initial-world-model-checkpoint",
                str(self.paths.world_model_checkpoint),
                "--terminal-value-checkpoint",
                str(self.paths.bootstrap_value_output / "best.pt"),
                "--working-root",
                str(self.paths.working_root),
                "--device",
                self.args.device,
                "--policy-device",
                "cpu",
                *POLICY_ARGUMENTS,
            ]
        )

    def final_world_model(self) -> Path:
        active = self.paths.working_root / "active_checkpoint.json"
        if active.is_file():
            payload = json.loads(active.read_text())
            return Path(payload["active_checkpoint"])
        candidates = sorted(
            self.paths.working_root.glob("checkpoints/world_model_iteration_*/best.pt")
        )
        return candidates[-1] if candidates else self.paths.world_model_checkpoint

    def teacher_datasets(self) -> list[Path]:
        return sorted(
            path.parent
            for path in self.paths.working_root.glob("teacher_iteration_*/manifest.json")
        )

    def terminal_final(self) -> None:
        checkpoint = self.paths.final_value_output / "best.pt"
        if self.completed(checkpoint):
            return
        teachers = self.teacher_datasets()
        if not teachers and not self.args.dry_run:
            raise FileNotFoundError(
                f"No completed teacher datasets found in {self.paths.working_root}"
            )
        datasets = [str(self.paths.initial_dataset), *(str(path) for path in teachers)]
        self.command(
            [
                "scripts/train_terminal_value.py",
                "--config",
                "configs/terminal_value.yaml",
                "--dataset",
                *datasets,
                "--world-model-checkpoint",
                str(self.final_world_model()),
                "--output",
                str(self.paths.final_value_output),
                "--wandb-name",
                f"terminal-value-{self.args.run}-mpc",
                "--device",
                self.args.device,
                "--wandb-mode",
                self.args.wandb_mode,
            ]
        )

    def validate(self) -> None:
        datasets = sorted(self.paths.final_value_output.glob("value_dataset_*"))
        datasets = [path for path in datasets if (path / "test_manifest.json").is_file()]
        if not datasets and not self.args.dry_run:
            raise FileNotFoundError(
                f"No processed terminal-value test dataset in {self.paths.final_value_output}"
            )
        value_dataset = datasets[-1] if datasets else self.paths.final_value_output / "value_dataset_00"
        self.command(
            [
                "scripts/evaluate_terminal_value.py",
                "--checkpoint",
                str(self.paths.final_value_output / "best.pt"),
                "--dataset",
                str(value_dataset),
                "--split",
                "test",
                "--output",
                str(self.paths.validation_output / "terminal_value"),
                "--device",
                self.args.device,
                "--minimum-spearman",
                "0.65",
                "--minimum-pairwise-accuracy",
                "0.65",
                "--maximum-rmse",
                "2.5",
                "--fail-on-threshold",
            ]
        )
        if self.args.candidate_ranking:
            self.command(
                [
                    "scripts/evaluate_mpc_candidate_ranking.py",
                    "--config",
                    "configs/mpc_joint_teams.yaml",
                    "--world-model-checkpoint",
                    str(self.final_world_model()),
                    "--terminal-value-checkpoint",
                    str(self.paths.final_value_output / "best.pt"),
                    "--profile",
                    "teacher_high_quality",
                    "--num-candidates",
                    "64",
                    "--num-trials",
                    "10",
                    "--skill-policy-source",
                    "local",
                    *POLICY_ARGUMENTS,
                    "--num-robots",
                    "2",
                    "--device",
                    self.args.device,
                    "--policy-device",
                    "cpu",
                    "--output",
                    str(self.paths.validation_output / "candidate_ranking.json"),
                    "--fail-on-threshold",
                ]
            )

    def run(self) -> None:
        stage = self.args.stage
        if stage == "world_model":
            self.world_model()
        elif stage == "terminal_bootstrap":
            self.terminal_bootstrap()
        elif stage == "iterative":
            self.world_model()
            self.terminal_bootstrap()
            self.iterative()
        elif stage == "terminal_final":
            self.terminal_final()
        elif stage == "validate":
            self.validate()
        else:
            self.world_model()
            self.terminal_bootstrap()
            self.iterative()
            self.terminal_final()
            self.validate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES, nargs="?", default="all")
    parser.add_argument(
        "--run",
        default="as2",
        help="Dataset/run version created by collect.bash (default: as2).",
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:6"))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed stages.")
    parser.add_argument(
        "--candidate-ranking",
        action="store_true",
        help="Also run the slower simulator candidate-ranking validation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    Pipeline(parse_args()).run()
