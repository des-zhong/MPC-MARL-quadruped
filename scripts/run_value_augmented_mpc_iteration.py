"""Batch collect, update, validate, and gate world/value models iteratively."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from dribblebot.mpc.iterative import build_collection_command, finetune_and_gate
from dribblebot.mpc.terminal_value import ValueDataset, load_value_checkpoint, value_metrics
from dribblebot.world_model.config import load_config


@torch.no_grad()
def _metrics(checkpoint, dataset, device):
    model, _ = load_value_checkpoint(checkpoint, device)
    prediction, target = [], []
    for batch in DataLoader(dataset, batch_size=2048):
        prediction.append(model.predict(batch["global_state"].to(device)).cpu())
        target.append(batch["return_to_go"])
    return value_metrics(torch.cat(prediction), torch.cat(target))


def main(args):
    config = load_config(args.config)
    root = Path(config["working_root"]); root.mkdir(parents=True, exist_ok=True)
    active_world = Path(config["initial_world_model_checkpoint"])
    active_value = Path(config["initial_terminal_value_checkpoint"]) if config.get("initial_terminal_value_checkpoint") else None
    teachers, expansions, decisions = [], [], []
    count = int(config["iterative_training"]["num_iterations"])
    for iteration in range(count):
        command = build_collection_command(config, iteration, active_world, root)
        teacher = root / f"teacher_iteration_{iteration:03d}"
        if active_value is not None:
            command.extend(("--terminal-value-checkpoint", str(active_value)))
        if args.dry_run:
            print(" ".join(command)); continue
        subprocess.run(command, check=True)
        teachers.append(teacher)
        expansion = root / "world_model" / f"mpc_iteration_{iteration:03d}"
        expansions.append(expansion)

        world_output = root / "checkpoints" / f"world_model_iteration_{iteration:03d}"
        world_decision = finetune_and_gate(
            config, active_world, config["initial_dataset"], expansions, world_output
        )
        active_world = Path(world_decision["active_checkpoint_after"])

        value_output = root / "checkpoints" / f"terminal_value_iteration_{iteration:03d}"
        sources = [str(config.get("initial_value_dataset", config["initial_dataset"]))] + [str(path) for path in teachers]
        train_command = [
            sys.executable, "scripts/train_terminal_value.py", "--config",
            str(config.get("terminal_value_config", "configs/terminal_value.yaml")),
            "--world-model-checkpoint", str(active_world), "--output", str(value_output),
            "--dataset", *sources,
        ]
        if active_value is not None:
            train_command.extend(("--resume", str(active_value)))
        subprocess.run(train_command, check=True)
        candidate = value_output / "best.pt"
        validation = ValueDataset(value_output / f"value_dataset_{len(sources)-1:03d}", "validation")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        candidate_metrics = _metrics(candidate, validation, device)
        old_metrics = _metrics(active_value, validation, device) if active_value else None
        finite = all(
            torch.isfinite(torch.tensor(candidate_metrics[key]))
            for key in ("mse", "rmse", "mae", "huber")
        )
        accepted = bool(finite) and (
            old_metrics is None or candidate_metrics["rmse"] <= 1.02 * old_metrics["rmse"]
        )
        if accepted: active_value = candidate
        decision = {
            "iteration": iteration, "world_model": world_decision,
            "value_model": {"accepted": accepted, "candidate": str(candidate),
                            "active": str(active_value), "old_metrics": old_metrics,
                            "candidate_metrics": candidate_metrics},
            "value_replay_sources": sources,
            "value_policy_interpretation": "V approximates continuation under behavior represented in replay, not guaranteed V*",
        }
        decisions.append(decision)
        (root / "active_checkpoints.json").write_text(json.dumps({
            "world_model": str(active_world), "terminal_value": str(active_value),
            "iteration": iteration,
        }, indent=2))
    summary = {"dry_run": args.dry_run, "decisions": decisions,
               "active_world_model": str(active_world),
               "active_terminal_value": None if active_value is None else str(active_value)}
    (root / "value_augmented_iterative_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/iterative_mpc_world_model.yaml")
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
