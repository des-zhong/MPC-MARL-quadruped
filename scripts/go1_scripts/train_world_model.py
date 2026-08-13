"""Train the probabilistic Go1 skill-level world-model ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dribblebot.world_model.action_adapter import JointActionAdapter
from dribblebot.world_model.config import load_config
from dribblebot.world_model.dataset import WorldModelDataset
from dribblebot.world_model.ensemble import WorldModelEnsemble
from dribblebot.world_model.schema import StateSchema, event_names_from_metadata
from dribblebot.world_model.trainer import WorldModelTrainer, fit_normalizer, seed_everything


def main(args) -> None:
    config = load_config(args.config)
    if args.num_robots is not None:
        if int(args.num_robots) < 1:
            raise ValueError("--num-robots must be at least 1")
        config["environment"]["num_robots"] = int(args.num_robots)
        config["world_model"]["max_obstacles"] = int(args.num_robots)
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
    trainer.fit(args.output, args.resume)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/world_model.yaml")
    parser.add_argument("--dataset", default="data/world_model")
    parser.add_argument("--output", default="checkpoints/world_model")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-robots", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
