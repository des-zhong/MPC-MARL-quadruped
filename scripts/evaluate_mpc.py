"""Evaluate reward-only MPC against valid random and scripted baselines."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import isaacgym

assert isaacgym
import numpy as np
import torch

from dribblebot.mpc.evaluation import (
    evaluate_method,
    method_overrides,
)
from dribblebot.mpc.runtime import add_simulator_arguments, build_runtime
from dribblebot.world_model.config import load_config


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_summary(results):
    rows = []
    for result in results:
        row = {"method": result["method"], "episodes": result["episodes"]}
        for name, stats in result["metrics"].items():
            if isinstance(stats, dict):
                row[f"{name}_mean"] = stats.get("mean")
                row[f"{name}_std"] = stats.get("std")
                row[f"{name}_median"] = stats.get("median")
        rows.append(row)
    return rows


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    methods = args.methods or [
        "random_valid",
        "scripted",
        "greedy_h1",
        "mpc",
    ]
    results = []
    for method in methods:
        _set_seed(args.seed)
        runtime = build_runtime(
            args,
            mpc_overrides=method_overrides(method),
        )
        try:
            result = evaluate_method(
                runtime,
                method,
                args.num_episodes,
                confidence=args.confidence,
            )
            result["mpc_config"] = runtime.mpc_config.to_dict()
            results.append(result)
        finally:
            runtime.controller.close()
            runtime.env.close()
    if args.run_ablations:
        evaluation = load_config(args.config).get("evaluation", {})
        specs = []
        for horizon in evaluation.get("horizons", (1, 3, 5, 8, 10, 15)):
            specs.append((f"horizon_{horizon}", {"horizon": int(horizon)}))
        for candidates in evaluation.get(
            "candidate_counts", (128, 256, 512, 1024, 2048, 4096)
        ):
            candidates = int(candidates)
            specs.append(
                (
                    f"candidates_{candidates}",
                    {
                        "num_candidates": candidates,
                        "num_elites": min(
                            max(8, candidates // 8), candidates - 1
                        ),
                    },
                )
            )
        for iterations in evaluation.get("cem_iterations", (1, 3, 5, 7)):
            specs.append(
                (
                    f"iterations_{iterations}",
                    {"num_iterations": int(iterations)},
                )
            )
        for label, overrides in specs:
            _set_seed(args.seed)
            runtime = build_runtime(args, mpc_overrides=overrides)
            try:
                result = evaluate_method(
                    runtime,
                    "mpc",
                    args.ablation_episodes,
                    confidence=args.confidence,
                )
                result["method"] = label
                result["mpc_config"] = runtime.mpc_config.to_dict()
                results.append(result)
            finally:
                runtime.controller.close()
                runtime.env.close()
    payload = {
        "seed": args.seed,
        "identical_reset_seed_requested": True,
        "results": results,
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    rows = _flatten_summary(results)
    if rows:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mpc.yaml")
    parser.add_argument(
        "--world-model-checkpoint",
        default="checkpoints/world_model_as2/best.pt",
    )
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--profile", default="fast_validation")
    parser.add_argument("--output", default="outputs/mpc_evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--ablation-episodes", type=int, default=100)
    add_simulator_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
