"""Build leakage-free terminal-value targets from real episode shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dribblebot.mpc.terminal_value import ValueModelConfig, build_value_dataset
from dribblebot.world_model.config import load_config


def main(args):
    config = ValueModelConfig.from_mapping(load_config(args.config)["value_model"])
    result = build_value_dataset(args.dataset, args.output, config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/terminal_value.yaml")
    parser.add_argument("--dataset", default="data/mpc_teacher")
    parser.add_argument("--output", default="data/terminal_value")
    main(parser.parse_args())
