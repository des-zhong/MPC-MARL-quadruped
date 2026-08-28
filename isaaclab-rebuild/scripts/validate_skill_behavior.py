"""Run the Gym/Lab frozen-skill behavior parity matrix on the host GPU."""

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
PACKAGE_SOURCE = REBUILD_ROOT / "source/dribblebot_isaaclab"
DEFAULT_ISAACLAB_PYTHON = Path(
    "/home/xander/Code/02_Manipulation/manifold_manipulation/manifold_manip/.venv/bin/python"
)
MODES = ("walk", "dribble", "shoot", "switch")


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
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--switch_interval", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=REBUILD_ROOT / "outputs/skill-behavior" / timestamp,
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _prepend(env: dict[str, str], key: str, *entries: Path) -> None:
    values = [str(entry) for entry in entries]
    if env.get(key):
        values.append(env[key])
    env[key] = os.pathsep.join(values)


def _run(command: list[str], env: dict[str, str], dry_run: bool, check: bool = True) -> int:
    print(f"[SKILL-BEHAVIOR] {shlex.join(command)}", flush=True)
    if dry_run:
        return 0
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def main() -> int:
    args = parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.switch_interval <= 0:
        raise ValueError("num_envs, steps, and switch_interval must be positive")
    if not args.dry_run:
        for interpreter in (args.isaaclab_python, args.isaacgym_python):
            if not interpreter.is_file():
                raise FileNotFoundError(f"Python interpreter was not found: {interpreter}")

    gym_env = os.environ.copy()
    _prepend(gym_env, "PYTHONPATH", REBUILD_ROOT, REPOSITORY_ROOT)
    _prepend(gym_env, "PATH", args.isaacgym_python.parent)
    lab_env = os.environ.copy()
    lab_env.setdefault("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")
    _prepend(lab_env, "PYTHONPATH", PACKAGE_SOURCE, REBUILD_ROOT)
    _prepend(lab_env, "PATH", args.isaaclab_python.parent)

    results = {}
    for mode in args.modes:
        gym_archive = args.output_dir / f"gym-{mode}.npz"
        lab_archive = args.output_dir / f"lab-{mode}.npz"
        report = args.output_dir / f"report-{mode}.json"
        common = [
            "--num_envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--mode",
            mode,
            "--switch_interval",
            str(args.switch_interval),
        ]
        _run(
            [
                str(args.isaacgym_python),
                str(REBUILD_ROOT / "scripts/record_isaacgym_skill_behavior.py"),
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
                str(REBUILD_ROOT / "scripts/record_isaaclab_skill_behavior.py"),
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
                str(REBUILD_ROOT / "scripts/compare_skill_behaviors.py"),
                str(gym_archive),
                str(lab_archive),
                "--output",
                str(report),
            ],
            lab_env,
            args.dry_run,
            check=False,
        )
        results[mode] = return_code == 0

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "device": args.device,
            "isaaclab_python": str(args.isaaclab_python),
            "isaacgym_python": str(args.isaacgym_python),
            "steps": args.steps,
            "num_envs": args.num_envs,
            "switch_interval": args.switch_interval,
            "comparison_passed": results,
        }
        (args.output_dir / "manifest.json").write_text(
            f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
    if results and not all(results.values()):
        failed = [mode for mode, passed in results.items() if not passed]
        print(f"[SKILL-BEHAVIOR] Failed modes: {failed}", file=sys.stderr)
        return 1
    print("[SKILL-BEHAVIOR] Requested matrix completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
