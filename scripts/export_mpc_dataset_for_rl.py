"""Export MPC teacher episodes for decentralized actor and critic pretraining."""

from __future__ import annotations

import argparse
import json

from dribblebot.mpc.rl_export import export_teacher_dataset_for_rl


def main(args):
    summary = export_teacher_dataset_for_rl(
        args.input,
        args.output,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_key=args.value_key,
        verify_hashes=args.verify_hashes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/mpc_teacher")
    parser.add_argument("--output", default="data/rl_pretraining")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--value-key", default=None)
    parser.add_argument("--verify-hashes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
