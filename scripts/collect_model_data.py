"""Small, safe entry point for initial or manual MPC data collection."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dribblebot.mpc.workflow_paths import ModelWorkflowPaths


POLICY_ARGUMENTS = [
    "--skill-policy-source",
    "local",
    "--walk-policy-dir",
    "checkpoints/reproduction/walk",
    "--dribble-policy-dir",
    "checkpoints/reproduction/dribble",
    "--shoot-policy-dir",
    "checkpoints/reproduction/shoot",
]


def _require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to write into existing directory: {path}\n"
            "Choose a new run name so data from different environments is not mixed."
        )


def _run(command: list[str], dry_run: bool) -> None:
    print(f"[collect] {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT, env=os.environ.copy())


def collect_initial(args: argparse.Namespace, paths: ModelWorkflowPaths) -> None:
    _require_new_directory(paths.initial_dataset)
    command = [
        sys.executable,
        "scripts/collect_world_model_data.py",
        "--config",
        "configs/world_model_as2.yaml",
        "--output",
        str(paths.initial_dataset),
        "--device",
        args.device,
        *POLICY_ARGUMENTS,
    ]
    if args.episodes is not None:
        command.extend(("--num-episodes", str(args.episodes)))
    _run(command, args.dry_run)


def collect_mpc(args: argparse.Namespace, paths: ModelWorkflowPaths) -> None:
    teacher = paths.teacher_output(args.iteration)
    expansion = paths.expansion_output(args.iteration)
    _require_new_directory(teacher)
    _require_new_directory(expansion)
    for required in (
        paths.world_model_checkpoint,
        paths.bootstrap_value_output / "best.pt",
    ):
        if not required.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Required checkpoint is missing: {required}")
    _run(
        [
            sys.executable,
            "scripts/collect_mpc_teacher_rollouts.py",
            "--config",
            "configs/mpc_joint_teams.yaml",
            "--profile",
            "teacher_training",
            "--world-model-checkpoint",
            str(paths.world_model_checkpoint),
            "--terminal-value-checkpoint",
            str(paths.bootstrap_value_output / "best.pt"),
            "--terminal-value-coefficient",
            "0.05",
            "--output",
            str(teacher),
            "--world-model-expansion-output",
            str(expansion),
            "--num-episodes",
            str(args.episodes or 2000),
            "--num-envs",
            str(args.num_envs),
            "--device",
            args.device,
            *POLICY_ARGUMENTS,
        ],
        args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect a fresh, versioned dataset for the model pipeline."
    )
    parser.add_argument("mode", choices=("initial", "mpc"), nargs="?", default="initial")
    parser.add_argument(
        "run",
        nargs="?",
        default="as2",
        help="Dataset/run version, for example env_v2 (default: as2).",
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:6"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.iteration < 1:
        parser.error("--iteration must be positive")
    return args


def main() -> None:
    args = parse_args()
    paths = ModelWorkflowPaths.for_run(args.run)
    if args.mode == "initial":
        collect_initial(args, paths)
    else:
        collect_mpc(args, paths)


if __name__ == "__main__":
    main()
