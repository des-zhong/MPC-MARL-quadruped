"""Run the migrated-task smoke suite and Gym/Lab parity matrix on the host GPU."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REBUILD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REBUILD_ROOT.parent
PACKAGE_SOURCE = REBUILD_ROOT / "source" / "dribblebot_isaaclab"
DEFAULT_ISAACLAB_PYTHON = Path(
    "/home/xander/Code/02_Manipulation/manifold_manipulation/manifold_manip/.venv/bin/python"
)

SMOKE_TASKS = (
    "Isaac-DribbleBot-AS2-Velocity-Flat-Play-v0",
    "Isaac-DribbleBot-AS2-Dribble-Flat-Play-v0",
    "Isaac-DribbleBot-AS2-Shooting-Flat-Play-v0",
    "Isaac-DribbleBot-AS2-Skill-Flat-Play-v0",
)
PARITY_TASKS = {
    "velocity": "Isaac-DribbleBot-AS2-Velocity-Flat-Play-v0",
    "dribble": "Isaac-DribbleBot-AS2-Dribble-Flat-Play-v0",
}
EXCITATIONS = ("zero", "sine")


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--isaaclab-python",
        type=Path,
        default=Path(os.environ.get("DRIBBLEBOT_ISAACLAB_PYTHON", DEFAULT_ISAACLAB_PYTHON)),
    )
    parser.add_argument(
        "--isaacgym-python",
        type=Path,
        default=Path(os.environ.get("DRIBBLEBOT_ISAACGYM_PYTHON", REPOSITORY_ROOT / ".venv/bin/python")),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--smoke_steps", type=int, default=200)
    parser.add_argument("--parity_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REBUILD_ROOT / "outputs" / "host-validation" / timestamp,
    )
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--skip_parity", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _prepend_pythonpath(env: dict[str, str], *paths: Path) -> None:
    entries = [str(path) for path in paths]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)


def _prepend_interpreter_bin(env: dict[str, str], interpreter: Path) -> None:
    """Expose console tools installed beside an explicitly invoked Python."""

    entries = [str(interpreter.parent)]
    existing = env.get("PATH")
    if existing:
        entries.append(existing)
    env["PATH"] = os.pathsep.join(entries)


def _run(command: list[str], env: dict[str, str], dry_run: bool, check: bool = True) -> int:
    print(f"[HOST-VALIDATION] {shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def _preflight_commands(args: argparse.Namespace) -> list[tuple[list[str], dict[str, str]]]:
    lab_env = os.environ.copy()
    lab_env.setdefault("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")
    _prepend_pythonpath(lab_env, PACKAGE_SOURCE, REBUILD_ROOT)
    _prepend_interpreter_bin(lab_env, args.isaaclab_python)
    gym_env = os.environ.copy()
    _prepend_pythonpath(gym_env, REBUILD_ROOT, REPOSITORY_ROOT)
    _prepend_interpreter_bin(gym_env, args.isaacgym_python)
    device_index = args.device.split(":", 1)[1] if ":" in args.device else "0"
    lab_code = (
        "import torch; "
        "assert torch.cuda.is_available(), 'Isaac Lab interpreter cannot see CUDA'; "
        f"print('Isaac Lab CUDA:', torch.cuda.get_device_name({device_index}))"
    )
    # Isaac Gym must be imported before torch so its bundled extensions and
    # plugin paths are initialized in the supported order.
    gym_code = (
        "from isaacgym import gymapi; import torch; "
        "assert torch.cuda.is_available(), 'Isaac Gym interpreter cannot see CUDA'; "
        f"print('Isaac Gym CUDA:', torch.cuda.get_device_name({device_index}))"
    )
    commands = []
    if not (args.skip_smoke and args.skip_parity):
        commands.append(([str(args.isaaclab_python), "-c", lab_code], lab_env))
    if not args.skip_parity:
        commands.append(([str(args.isaacgym_python), "-c", gym_code], gym_env))
    return commands


def main() -> int:
    args = parse_args()
    if args.num_envs <= 0 or args.smoke_steps < 0 or args.parity_steps <= 0:
        raise ValueError("num_envs and parity_steps must be positive; smoke_steps must be non-negative")
    if not args.dry_run:
        interpreters = []
        if not (args.skip_smoke and args.skip_parity):
            interpreters.append(args.isaaclab_python)
        if not args.skip_parity:
            interpreters.append(args.isaacgym_python)
        for interpreter in interpreters:
            if not interpreter.is_file():
                raise FileNotFoundError(f"Python interpreter was not found: {interpreter}")

    lab_env = os.environ.copy()
    lab_env.setdefault("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")
    _prepend_pythonpath(lab_env, PACKAGE_SOURCE, REBUILD_ROOT)
    _prepend_interpreter_bin(lab_env, args.isaaclab_python)
    gym_env = os.environ.copy()
    _prepend_pythonpath(gym_env, REBUILD_ROOT, REPOSITORY_ROOT)
    _prepend_interpreter_bin(gym_env, args.isaacgym_python)

    for command, command_env in _preflight_commands(args):
        _run(command, command_env, args.dry_run)

    if not args.skip_smoke:
        for task in SMOKE_TASKS:
            _run(
                [
                    str(args.isaaclab_python),
                    str(REBUILD_ROOT / "scripts/random_agent.py"),
                    "--task",
                    task,
                    "--num_envs",
                    str(args.num_envs),
                    "--steps",
                    str(args.smoke_steps),
                    "--device",
                    args.device,
                    "--headless",
                ],
                lab_env,
                args.dry_run,
            )

    comparison_results: dict[str, bool] = {}
    if not args.skip_parity:
        for task_name, lab_task in PARITY_TASKS.items():
            for excitation in EXCITATIONS:
                stem = f"{task_name}-{excitation}"
                gym_archive = args.output_dir / f"gym-{stem}.npz"
                lab_archive = args.output_dir / f"lab-{stem}.npz"
                report_path = args.output_dir / f"report-{stem}.json"
                common = [
                    "--num_envs",
                    str(args.num_envs),
                    "--steps",
                    str(args.parity_steps),
                    "--seed",
                    str(args.seed),
                    "--excitation",
                    excitation,
                ]
                _run(
                    [
                        str(args.isaacgym_python),
                        str(REBUILD_ROOT / "scripts/record_isaacgym_rollout.py"),
                        "--task",
                        task_name,
                        "--output",
                        str(gym_archive),
                        "--device",
                        args.device,
                        *common,
                    ],
                    gym_env,
                    args.dry_run,
                )
                _run(
                    [
                        str(args.isaaclab_python),
                        str(REBUILD_ROOT / "scripts/record_isaaclab_rollout.py"),
                        "--task",
                        lab_task,
                        "--output",
                        str(lab_archive),
                        "--device",
                        args.device,
                        "--headless",
                        *common,
                    ],
                    lab_env,
                    args.dry_run,
                )
                return_code = _run(
                    [
                        str(args.isaaclab_python),
                        str(REBUILD_ROOT / "scripts/compare_rollouts.py"),
                        str(gym_archive),
                        str(lab_archive),
                        "--output",
                        str(report_path),
                    ],
                    lab_env,
                    args.dry_run,
                    check=False,
                )
                comparison_results[stem] = return_code == 0

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "device": args.device,
            "isaaclab_python": str(args.isaaclab_python),
            "isaacgym_python": str(args.isaacgym_python),
            "smoke_tasks": [] if args.skip_smoke else list(SMOKE_TASKS),
            "comparison_passed": comparison_results,
        }
        (args.output_dir / "manifest.json").write_text(
            f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )

    if comparison_results and not all(comparison_results.values()):
        failed = [name for name, passed in comparison_results.items() if not passed]
        print(f"[HOST-VALIDATION] Parity gates failed: {failed}", file=sys.stderr)
        return 1
    print("[HOST-VALIDATION] Requested validation matrix completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
