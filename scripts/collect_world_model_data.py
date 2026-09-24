"""Collect complete held-out episodes using frozen PPO and low-level policies."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.play_high_level import parse_args, run_cli
from quadruped.mpc.simulator_controller import TerminalStateCapture  # compatibility for training callers


if __name__ == "__main__":
    args = parse_args(defaults={
        "high_level_policy_source": "local", "training_skills": True,
        "headless": True, "no_video": True, "no_plot": True,
        "outcomes_only": True, "stop_after_episodes": 3,
        "num_envs": 1, "steps": 10000, "seed": 20260915,
    })
    if not args.world_model_output or not args.world_model_checkpoint:
        raise ValueError("Provide --world-model-output and --world-model-checkpoint")
    run_cli(args)
