"""Train V(s) on real simulator Monte-Carlo returns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dribblebot.mpc.terminal_value import (
    CombinedValueDataset, ReturnNormalizer, TerminalValueModel, TerminalValueTrainer, ValueDataset,
    ValueModelConfig, build_value_dataset,
)
from dribblebot.world_model.config import load_config
from dribblebot.world_model.trainer import load_checkpoint, seed_everything


def main(args):
    payload = load_config(args.config)
    config = ValueModelConfig.from_mapping(payload["value_model"])
    seed_everything(config.seed)
    processed_roots = []
    for index, item in enumerate(args.dataset):
        dataset = Path(item)
        manifest = json.loads((dataset / "manifest.json").read_text())
        if manifest.get("format") != "dribblebot_terminal_value_v1":
            processed = Path(args.output) / f"value_dataset_{index:03d}"
            build_value_dataset(dataset, processed, config)
            dataset = processed
        processed_roots.append(dataset)
    train_parts = [ValueDataset(dataset, "train") for dataset in processed_roots]
    validation_parts = [ValueDataset(dataset, "validation") for dataset in processed_roots]
    train = train_parts[0] if len(train_parts) == 1 else CombinedValueDataset(train_parts)
    validation = validation_parts[-1]
    if not len(train) or not len(validation):
        raise ValueError("Value training requires non-empty episode-level train and validation splits")
    world_model, _ = load_checkpoint(args.world_model_checkpoint, "cpu")
    return_normalizer = ReturnNormalizer.fit(train.targets()) if config.normalize_targets else ReturnNormalizer()
    model = TerminalValueModel(
        world_model.schema, world_model.normalizer,
        hidden_dims=config.hidden_dims, activation=config.activation,
        layer_norm=config.layer_norm, dropout=config.dropout,
        return_normalizer=return_normalizer,
        return_statistics=json.loads((processed_roots[0] / "metadata.json").read_text()).get(
            "return_statistics", {}
        ),
    )
    trainer = TerminalValueTrainer(model, train, validation, config)
    history = trainer.fit(args.output, args.resume)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot([row["loss"] for row in history["train"]], label="train")
        axis.plot([row["loss"] for row in history["validation"]], label="validation")
        axis.set(xlabel="epoch", ylabel=config.loss, title="Terminal value training")
        axis.grid(True, linestyle=":", alpha=.3); axis.legend(); figure.tight_layout()
        figure.savefig(Path(args.output) / "training_curve.png", dpi=160); plt.close(figure)
    except ImportError:
        pass
    print(json.dumps({"epochs": len(history["train"]), "best": str(Path(args.output) / "best.pt")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/terminal_value.yaml")
    parser.add_argument("--dataset", nargs="+", default=["data/mpc_teacher"])
    parser.add_argument("--world-model-checkpoint", default="checkpoints/world_model_as2/best.pt")
    parser.add_argument("--output", default="checkpoints/terminal_value")
    parser.add_argument("--resume", default=None)
    main(parser.parse_args())
