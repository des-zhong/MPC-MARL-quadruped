"""Orchestration for iterative MPC collection and world-model fine-tuning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import torch

from dribblebot.world_model.config import deep_update, load_config
from dribblebot.world_model.dataset import WorldModelDataset
from dribblebot.world_model.trainer import (
    WorldModelTrainer,
    load_checkpoint,
    seed_everything,
)

from .acceptance import (
    ModelAcceptanceConfig,
    ModelAcceptanceGate,
    evaluate_model_for_acceptance,
)
from .replay import MixedWorldModelDataset, ReplayMixSampler


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def build_collection_command(
    config: Mapping[str, object],
    iteration: int,
    checkpoint: Union[str, Path],
    working_root: Union[str, Path],
) -> List[str]:
    root = Path(working_root)
    iterative = config["iterative_training"]
    command = [
        sys.executable,
        "scripts/collect_mpc_teacher_rollouts.py",
        "--config",
        str(config["mpc_config"]),
        "--world-model-checkpoint",
        str(checkpoint),
        "--output",
        str(root / f"teacher_iteration_{iteration:03d}"),
        "--world-model-expansion-output",
        str(root / f"world_model" / f"mpc_iteration_{iteration:03d}"),
        "--num-episodes",
        str(int(iterative["episodes_per_iteration"])),
    ]
    profile = config.get("mpc_profile")
    if profile:
        command.extend(("--profile", str(profile)))
    skill = config.get("skill_policies", {})
    for key, value in skill.items():
        cli_key = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(cli_key)
        elif value is not None:
            command.extend((cli_key, str(value)))
    return command


def _evaluation_pair(old_model, new_model, original, recent, device, evaluation):
    return (
        {
            "original": evaluate_model_for_acceptance(
                old_model, original, device, **evaluation
            ),
            "recent": evaluate_model_for_acceptance(
                old_model, recent, device, **evaluation
            ),
        },
        {
            "original": evaluate_model_for_acceptance(
                new_model, original, device, **evaluation
            ),
            "recent": evaluate_model_for_acceptance(
                new_model, recent, device, **evaluation
            ),
        },
    )


def finetune_and_gate(
    config: Mapping[str, object],
    active_checkpoint: Union[str, Path],
    initial_dataset_root: Union[str, Path],
    expansion_roots: Sequence[Union[str, Path]],
    iteration_output: Union[str, Path],
) -> Dict[str, object]:
    if not expansion_roots:
        raise ValueError("At least one MPC expansion dataset is required")
    iterative = config["iterative_training"]
    requested_device = str(iterative.get("device", "cuda"))
    device = (
        requested_device
        if requested_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    initial_train = WorldModelDataset(initial_dataset_root, "train")
    original_validation = WorldModelDataset(initial_dataset_root, "validation")
    source_datasets = [("initial", initial_train)]
    for root in expansion_roots[:-1]:
        source_datasets.append(("previous_mpc", WorldModelDataset(root, "train")))
    newest_train = WorldModelDataset(expansion_roots[-1], "train")
    recent_validation = WorldModelDataset(expansion_roots[-1], "validation")
    source_datasets.append(("newest_mpc", newest_train))
    mixed = MixedWorldModelDataset(source_datasets)
    sampling = config.get("world_model_update", {}).get(
        "sampling",
        {
            "initial_random": 0.25,
            "initial_scripted": 0.20,
            "previous_mpc": 0.25,
            "newest_mpc": 0.20,
            "rare_events": 0.10,
        },
    )
    sampler = ReplayMixSampler(mixed, sampling, seed)
    model, checkpoint_payload = load_checkpoint(active_checkpoint, device)
    training_config = deep_update(
        checkpoint_payload["training_config"],
        config.get("world_model_update", {}).get("training_overrides", {}),
    )
    training = dict(training_config["training"])
    training["device"] = device
    training["max_epochs"] = int(iterative.get("max_finetune_epochs", 50))
    training["early_stopping_patience"] = int(
        iterative.get("early_stopping_patience", 10)
    )
    training_config["training"] = training
    trainer = WorldModelTrainer(
        model,
        mixed,
        original_validation,
        training_config,
        train_sampler=sampler,
    )
    output = Path(iteration_output)
    trainer.fit(output)
    candidate_checkpoint = output / "best.pt"
    old_model, _ = load_checkpoint(active_checkpoint, device)
    new_model, _ = load_checkpoint(candidate_checkpoint, device)
    evaluation = dict(config.get("acceptance_evaluation", {}))
    old_metrics, new_metrics = _evaluation_pair(
        old_model,
        new_model,
        original_validation,
        recent_validation,
        device,
        evaluation,
    )
    gate = ModelAcceptanceGate(
        ModelAcceptanceConfig.from_mapping(config.get("model_acceptance", {}))
    )
    decision = gate.compare(old_metrics, new_metrics)
    decision.update(
        {
            "active_checkpoint_before": str(Path(active_checkpoint).resolve()),
            "candidate_checkpoint": str(candidate_checkpoint.resolve()),
            "active_checkpoint_after": str(
                (
                    candidate_checkpoint
                    if decision["accepted"]
                    else Path(active_checkpoint)
                ).resolve()
            ),
            "normalizer_policy": (
                "Preserved from the accepted checkpoint during model-only "
                "fine-tuning; it is not silently refit on newest MPC data."
            ),
            "replay_sources": [name for name, _ in source_datasets],
            "replay_sampling": dict(sampling),
        }
    )
    _atomic_json(output / "acceptance_decision.json", decision)
    return decision


def run_iterative_pipeline(
    config_path: Union[str, Path],
    dry_run: bool = False,
) -> Dict[str, object]:
    config = load_config(config_path)
    required = (
        "mpc_config",
        "initial_world_model_checkpoint",
        "initial_dataset",
        "working_root",
        "iterative_training",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Iterative MPC config is missing fields: {missing}")
    root = Path(config["working_root"])
    root.mkdir(parents=True, exist_ok=True)
    pointer_path = root / "active_checkpoint.json"
    if pointer_path.exists():
        active_checkpoint = Path(
            json.loads(pointer_path.read_text())["active_checkpoint"]
        )
    else:
        active_checkpoint = Path(config["initial_world_model_checkpoint"])
    iteration_count = int(config["iterative_training"]["num_iterations"])
    expansion_roots: List[Path] = []
    decisions = []
    commands = []
    for iteration in range(iteration_count):
        command = build_collection_command(
            config, iteration, active_checkpoint, root
        )
        commands.append(command)
        expansion_root = (
            root / "world_model" / f"mpc_iteration_{iteration:03d}"
        )
        expansion_roots.append(expansion_root)
        if dry_run:
            continue
        subprocess.run(command, check=True)
        candidate_output = (
            root / "checkpoints" / f"world_model_iteration_{iteration:03d}"
        )
        decision = finetune_and_gate(
            config,
            active_checkpoint,
            config["initial_dataset"],
            expansion_roots,
            candidate_output,
        )
        decisions.append(decision)
        active_checkpoint = Path(decision["active_checkpoint_after"])
        _atomic_json(
            pointer_path,
            {
                "active_checkpoint": str(active_checkpoint.resolve()),
                "iteration": iteration,
                "accepted": bool(decision["accepted"]),
            },
        )
    summary = {
        "dry_run": bool(dry_run),
        "commands": commands,
        "decisions": decisions,
        "active_checkpoint": str(active_checkpoint),
    }
    _atomic_json(root / "iterative_summary.json", summary)
    return summary
