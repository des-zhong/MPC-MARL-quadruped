"""Run MPC collection, mixed replay fine-tuning, evaluation, and acceptance."""

from __future__ import annotations

import argparse
import json

from dribblebot.mpc.iterative import run_iterative_pipeline


def main(args):
    summary = run_iterative_pipeline(args.config, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/iterative_mpc_world_model.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
