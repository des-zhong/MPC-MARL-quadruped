"""Train V(s) on real simulator Monte-Carlo returns.

The value model can reuse the schema/normalizer from a completed world-model
checkpoint, or fit the exact same normalizer directly from the world-model
training split. The latter allows world-model and value-model training to run
concurrently on separate devices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dribblebot.mpc.terminal_value import (
    CombinedValueDataset, ReturnNormalizer, TerminalValueModel, TerminalValueTrainer, ValueDataset,
    ValueModelConfig, build_value_dataset,
)
from dribblebot.world_model.dataset import WorldModelDataset
from dribblebot.world_model.config import load_config
from dribblebot.world_model.schema import StateSchema
from dribblebot.world_model.trainer import fit_normalizer, load_checkpoint, seed_everything


def _wandb_config(args, config, processed_roots, train, validation, model, trainer):
    return {
        "terminal_value": config.__dict__,
        "paths": {
            "config": str(Path(args.config).resolve()),
            "datasets": [str(Path(item).resolve()) for item in args.dataset],
            "processed_datasets": [str(path.resolve()) for path in processed_roots],
            "output": str(Path(args.output).resolve()),
            "resume_checkpoint": str(Path(args.resume).resolve()) if args.resume else None,
            "world_model_checkpoint": (
                str(Path(args.world_model_checkpoint).resolve())
                if args.world_model_checkpoint else None
            ),
            "normalizer_dataset": (
                str(Path(args.normalizer_dataset).resolve())
                if args.normalizer_dataset else None
            ),
        },
        "dataset": {
            "train_value_targets": len(train),
            "validation_value_targets": len(validation),
            "state_dimension": model.schema.state_dim,
            "return_statistics": model.return_statistics,
        },
        "runtime": {
            "device": str(trainer.device),
            "mixed_precision": trainer.scaler.is_enabled(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
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
        metrics.update(
            {f"validation/{key}": value for key, value in validation_metrics.items()}
        )
        metrics.update(
            {
                "epoch": epoch,
                "optimization/learning_rate": state["learning_rate"],
                "timing/epoch_seconds": state["epoch_seconds"],
                "early_stopping/patience": state["early_stopping_patience"],
                "validation/best_loss": state["best_validation_loss"],
                "checkpoint/is_best": int(bool(state["is_best"])),
            }
        )
        wandb.log(metrics, step=epoch)

    return log_epoch


def load_state_preprocessing(args):
    """Return the state schema/normalizer without requiring trained dynamics."""

    if args.world_model_checkpoint:
        world_model, _ = load_checkpoint(args.world_model_checkpoint, "cpu")
        return world_model.schema, world_model.normalizer

    # Preserve the original no-argument behavior for existing workflows. An
    # explicit --normalizer-dataset always selects concurrent-safe fitting.
    legacy_checkpoint = Path("checkpoints/world_model_as2/best.pt")
    if args.normalizer_dataset is None and legacy_checkpoint.exists():
        world_model, _ = load_checkpoint(legacy_checkpoint, "cpu")
        return world_model.schema, world_model.normalizer

    dataset_root = Path(args.normalizer_dataset or args.dataset[0])
    metadata_path = dataset_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Dataset-derived preprocessing requires a raw world-model dataset "
            f"with metadata.json, but it was not found in {dataset_root}"
        )
    metadata = json.loads(metadata_path.read_text())
    if "state_schema" not in metadata:
        raise ValueError(f"Dataset metadata in {dataset_root} has no state_schema")
    world_model_config = load_config(args.world_model_config)
    schema = StateSchema.from_dict(metadata["state_schema"])
    training_dataset = WorldModelDataset(dataset_root, "train")
    normalizer = fit_normalizer(
        training_dataset,
        schema,
        bool(world_model_config["training"].get("normalize_reward", False)),
    )
    return schema, normalizer


def main(args):
    payload = load_config(args.config)
    value_mapping = dict(payload["value_model"])
    if args.device is not None:
        value_mapping["device"] = args.device
    config = ValueModelConfig.from_mapping(value_mapping)
    seed_everything(config.seed)
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
        config={
            "terminal_value": config.__dict__,
            "paths": {
                "config": str(Path(args.config).resolve()),
                "datasets": [str(Path(item).resolve()) for item in args.dataset],
            },
        },
        job_type="terminal-value-training",
    )
    output_in_wandb_run = args.output is None
    if output_in_wandb_run:
        args.output = (
            run.dir
            if args.wandb_mode != "disabled"
            else "checkpoints/terminal_value"
        )
    print(f"Terminal-value checkpoint directory: {Path(args.output).resolve()}")
    try:
        processed_dataset_base = (
            Path("data/terminal_value_processed") / run.id
            if output_in_wandb_run
            else Path(args.output)
        )
        processed_roots = []
        for index, item in enumerate(args.dataset):
            dataset = Path(item)
            manifest = json.loads((dataset / "manifest.json").read_text())
            if manifest.get("format") != "dribblebot_terminal_value_v1":
                processed = processed_dataset_base / f"value_dataset_{index:03d}"
                build_value_dataset(dataset, processed, config)
                dataset = processed
            processed_roots.append(dataset)
        train_parts = [ValueDataset(dataset, "train") for dataset in processed_roots]
        validation_parts = [
            ValueDataset(dataset, "validation") for dataset in processed_roots
        ]
        train = (
            train_parts[0]
            if len(train_parts) == 1
            else CombinedValueDataset(train_parts)
        )
        validation = validation_parts[-1]
        if not len(train) or not len(validation):
            raise ValueError(
                "Value training requires non-empty episode-level train and validation splits"
            )
        state_schema, state_normalizer = load_state_preprocessing(args)
        return_normalizer = (
            ReturnNormalizer.fit(train.targets())
            if config.normalize_targets else ReturnNormalizer()
        )
        model = TerminalValueModel(
            state_schema, state_normalizer,
            hidden_dims=config.hidden_dims, activation=config.activation,
            layer_norm=config.layer_norm, dropout=config.dropout,
            ensemble_size=config.ensemble_size,
            return_normalizer=return_normalizer,
            return_statistics=json.loads(
                (processed_roots[0] / "metadata.json").read_text()
            ).get("return_statistics", {}),
        )
        trainer = TerminalValueTrainer(model, train, validation, config)
        wandb.config.update(
            _wandb_config(
                args, config, processed_roots, train, validation, model, trainer
            ),
            allow_val_change=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("validation/*", step_metric="epoch")
        wandb.define_metric("optimization/*", step_metric="epoch")
        wandb.define_metric("timing/*", step_metric="epoch")
        wandb.define_metric("early_stopping/*", step_metric="epoch")
        wandb.define_metric("validation/loss", summary="min")

        history = trainer.fit(
            args.output,
            args.resume,
            epoch_callback=_wandb_epoch_logger(wandb),
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot([row["loss"] for row in history["train"]], label="train loss")
            axis.plot([row["loss"] for row in history["validation"]], label="validation loss")
            axis.plot(
                [row["selection_score"] for row in history["validation"]],
                label="validation selection score",
            )
            axis.set(xlabel="epoch", ylabel="objective", title="Terminal value training")
            axis.grid(True, linestyle=":", alpha=.3); axis.legend(); figure.tight_layout()
            figure.savefig(Path(args.output) / "training_curve.png", dpi=160); plt.close(figure)
        except ImportError:
            pass
        completed_epochs = len(history["train"])
        if completed_epochs:
            validation_losses = [metrics["loss"] for metrics in history["validation"]]
            selection_scores = [
                metrics["selection_score"] for metrics in history["validation"]
            ]
            run.summary["best_validation_loss_this_session"] = min(validation_losses)
            run.summary["best_validation_selection_score_this_session"] = min(
                selection_scores
            )
            run.summary["epochs_completed_this_session"] = completed_epochs
            run.summary["final_train_loss"] = history["train"][-1]["loss"]
            run.summary["final_validation_loss"] = history["validation"][-1]["loss"]
        # Register files explicitly so they are uploaded whether they were
        # written directly in run.dir or staged from an external --output.
        if args.wandb_save_checkpoints:
            output = Path(args.output).resolve()
            for filename in (
                "best.pt", "final.pt", "history.json", "training_curve.png"
            ):
                path = output / filename
                if path.exists():
                    wandb.save(str(path), base_path=str(output), policy="end")
        print(json.dumps({"epochs": completed_epochs, "best": str(Path(args.output) / "best.pt")}, indent=2))
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/terminal_value.yaml")
    parser.add_argument("--dataset", nargs="+", default=["data/mpc_teacher"])
    preprocessing = parser.add_mutually_exclusive_group()
    preprocessing.add_argument(
        "--world-model-checkpoint",
        default=None,
        help="Reuse schema and normalization from an existing world-model checkpoint.",
    )
    preprocessing.add_argument(
        "--normalizer-dataset",
        default=None,
        help=(
            "Fit world-model-compatible state normalization from this raw dataset. "
            "Defaults to the first --dataset when no checkpoint is supplied."
        ),
    )
    parser.add_argument(
        "--world-model-config",
        default="configs/world_model_as2.yaml",
        help="World-model config controlling dataset-derived reward normalization.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Checkpoint directory. When omitted, save directly in the active "
            "W&B run's files directory."
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb-project", default="as2_terminal_value")
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
        help="Do not upload best.pt, final.pt, history.json, and the training curve.",
    )
    parser.set_defaults(wandb_save_checkpoints=True)
    main(parser.parse_args())
