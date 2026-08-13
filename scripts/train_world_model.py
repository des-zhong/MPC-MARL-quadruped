"""Train the probabilistic skill-level world-model ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from dribblebot.world_model.action_adapter import JointActionAdapter
from dribblebot.world_model.config import load_config
from dribblebot.world_model.dataset import WorldModelDataset
from dribblebot.world_model.ensemble import WorldModelEnsemble
from dribblebot.world_model.schema import StateSchema, event_names_from_metadata
from dribblebot.world_model.trainer import WorldModelTrainer, fit_normalizer, seed_everything


def _wandb_config(args, config, metadata, train, validation, model, trainer):
    """Build a compact, queryable run config without uploading dataset metadata."""

    return {
        "world_model": config,
        "paths": {
            "config": str(Path(args.config).resolve()),
            "dataset": str(Path(args.dataset).resolve()),
            "output": str(Path(args.output).resolve()),
            "resume_checkpoint": str(Path(args.resume).resolve()) if args.resume else None,
        },
        "dataset": {
            "robot": metadata.get("robot"),
            "train_transitions": len(train),
            "validation_transitions": len(validation),
            "train_episodes": len(train.episodes),
            "validation_episodes": len(validation.episodes),
            "event_names": list(model.event_names),
            "state_dimension": model.schema.state_dim,
            "joint_action_dimension": model.action_adapter.action_dim,
        },
        "runtime": {
            "device": str(trainer.device),
            "mixed_precision": trainer.scaler.is_enabled(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
    }


def _wandb_epoch_logger(wandb):
    def log_epoch(
        epoch: int,
        train_metrics: Mapping[str, float],
        validation_metrics: Mapping[str, float],
        state: Mapping[str, object],
    ) -> None:
        metrics = {f"train/{key}": value for key, value in train_metrics.items()}
        metrics.update({f"validation/{key}": value for key, value in validation_metrics.items()})
        metrics.update(
            {
                "epoch": epoch,
                "optimization/learning_rate": state["learning_rate"],
                "timing/epoch_seconds": state["epoch_seconds"],
                "early_stopping/patience": state["early_stopping_patience"],
                "validation/best_loss": state["best_validation_loss"],
                "checkpoint/is_best": int(bool(state["is_best"])),
                "checkpoint/periodic_best_save": int(
                    bool(state["periodic_best_save"])
                ),
            }
        )
        wandb.log(metrics, step=epoch)

    return log_epoch


def main(args) -> None:
    config = load_config(args.config)
    if args.num_robots is not None:
        if int(args.num_robots) < 1:
            raise ValueError("--num-robots must be at least 1")
        config["environment"]["team_size"] = int(args.num_robots)
        config["environment"]["num_robots"] = 2 * int(args.num_robots)
        config["world_model"]["max_obstacles"] = 0
    seed_everything(int(config.get("seed", 42)))
    root = Path(args.dataset)
    metadata = json.loads((root / "metadata.json").read_text())
    dataset_robot = metadata.get("robot")
    configured_robot = config.get("environment", {}).get("robot")
    if dataset_robot and configured_robot and dataset_robot != configured_robot:
        raise ValueError(
            f"Dataset contains {dataset_robot!r} transitions, but config selects {configured_robot!r}. "
            "Use the matching world-model config."
        )
    schema = StateSchema.from_dict(metadata["state_schema"])
    action_adapter = JointActionAdapter.from_dict(metadata["action_schema"])
    dataset_num_robots = action_adapter.num_robots
    configured_num_robots = config.get("environment", {}).get("num_robots")
    if configured_num_robots is not None and int(configured_num_robots) != dataset_num_robots:
        raise ValueError(
            f"Dataset contains {dataset_num_robots} robots, but config selects "
            f"{configured_num_robots}. Use a matching config or recollect the dataset."
        )
    metadata_num_robots = metadata.get("num_robots")
    if metadata_num_robots is not None and int(metadata_num_robots) != dataset_num_robots:
        raise ValueError("Dataset num_robots metadata disagrees with its action schema")
    event_names = event_names_from_metadata(metadata)
    train = WorldModelDataset(root, "train")
    validation = WorldModelDataset(root, "validation")
    normalizer = fit_normalizer(train, schema, bool(config["training"].get("normalize_reward", False)))
    model_config = dict(config["model"])
    configured_event_count = model_config.get("num_events")
    if configured_event_count is not None and int(configured_event_count) != len(event_names):
        raise ValueError(
            f"model.num_events={configured_event_count} does not match dataset event schema {event_names}"
        )
    model_config["num_events"] = len(event_names)
    model = WorldModelEnsemble(schema, action_adapter, normalizer, **model_config)
    model.event_names = event_names
    trainer = WorldModelTrainer(model, train, validation, config)

    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        tags=args.wandb_tags,
        id=args.wandb_id,
        resume="allow" if args.wandb_id else None,
        mode=args.wandb_mode,
        config=_wandb_config(args, config, metadata, train, validation, model, trainer),
        job_type="world-model-training",
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("validation/*", step_metric="epoch")
    wandb.define_metric("optimization/*", step_metric="epoch")
    wandb.define_metric("timing/*", step_metric="epoch")
    wandb.define_metric("early_stopping/*", step_metric="epoch")
    wandb.define_metric("validation/loss", summary="min")

    try:
        history = trainer.fit(args.output, args.resume, epoch_callback=_wandb_epoch_logger(wandb))
        completed_epochs = len(history["train"])
        if completed_epochs:
            validation_losses = [metrics["loss"] for metrics in history["validation"]]
            run.summary["best_validation_loss_this_session"] = min(validation_losses)
            run.summary["epochs_completed_this_session"] = completed_epochs
            run.summary["final_train_loss"] = history["train"][-1]["loss"]
            run.summary["final_validation_loss"] = history["validation"][-1]["loss"]
        if args.wandb_save_checkpoints:
            output = Path(args.output).resolve()
            for filename in ("best.pt", "final.pt", "history.json"):
                path = output / filename
                if path.exists():
                    wandb.save(str(path), base_path=str(output), policy="end")
    finally:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/world_model_as2.yaml")
    parser.add_argument("--dataset", default="data/world_model_as2")
    parser.add_argument("--output", default="checkpoints/world_model_as2")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--num-robots", type=int, default=None,
        help="Robots per team; the dataset contains twice this many robot actors.",
    )
    parser.add_argument("--wandb-project", default="as2_world_model")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-id", default=None, help="Stable W&B run ID to resume.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Use 'offline' to sync later with `wandb sync`, or 'disabled' for no logging.",
    )
    parser.add_argument(
        "--no-wandb-save-checkpoints",
        action="store_false",
        dest="wandb_save_checkpoints",
        help="Do not upload best.pt, final.pt, and history.json at the end of training.",
    )
    parser.set_defaults(wandb_save_checkpoints=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
