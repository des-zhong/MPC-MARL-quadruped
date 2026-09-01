"""Run MPC collection, mixed replay fine-tuning, evaluation, and acceptance."""

from __future__ import annotations

import argparse
import json

from quadruped.mpc.iterative import run_iterative_pipeline


def main(args):
    overrides = {
        key: value
        for key, value in {
            "mpc_config": args.mpc_config,
            "initial_world_model_checkpoint": args.initial_world_model_checkpoint,
            "initial_dataset": args.initial_dataset,
            "working_root": args.working_root,
        }.items()
        if value is not None
    }
    skill_overrides = {
        key: value
        for key, value in {
            "walk_policy_dir": args.walk_policy_dir,
            "dribble_policy_dir": args.dribble_policy_dir,
            "shoot_policy_dir": args.shoot_policy_dir,
            "terminal_value_checkpoint": args.terminal_value_checkpoint,
            "terminal_value_coefficient": args.terminal_value_coefficient,
            "num_envs": args.num_envs,
            "device": args.device,
            "policy_device": args.policy_device,
        }.items()
        if value is not None
    }
    if skill_overrides:
        overrides["skill_policies"] = skill_overrides
    summary = run_iterative_pipeline(
        args.config,
        dry_run=args.dry_run,
        overrides=overrides,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/iterative_mpc_world_model.yaml"
    )
    parser.add_argument("--mpc-config", default=None)
    parser.add_argument("--initial-world-model-checkpoint", default=None)
    parser.add_argument("--initial-dataset", default=None)
    parser.add_argument("--working-root", default=None)
    parser.add_argument("--walk-policy-dir", default=None)
    parser.add_argument("--dribble-policy-dir", default=None)
    parser.add_argument("--shoot-policy-dir", default=None)
    parser.add_argument("--terminal-value-checkpoint", default=None)
    parser.add_argument("--terminal-value-coefficient", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
