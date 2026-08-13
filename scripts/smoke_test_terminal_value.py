"""Tiny end-to-end terminal-value MPC smoke test, with optional real execution."""

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
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from dribblebot.mpc import HybridCEMMPC, MPCConfig
from dribblebot.mpc.terminal_value import (
    ReturnNormalizer, TerminalValueModel, TerminalValueTrainer, ValueDataset,
    ValueModelConfig, build_value_dataset, load_value_checkpoint,
)
from dribblebot.mpc.visualization import plot_top_down
from dribblebot.world_model.trainer import load_checkpoint


class _Prefix(Dataset):
    def __init__(self, dataset, count):
        self.dataset = dataset; self.count = min(int(count), len(dataset)); self.root = dataset.root
    def __len__(self): return self.count
    def __getitem__(self, index): return self.dataset[index]
    def targets(self): return torch.stack([self[index]["return_to_go"] for index in range(self.count)])


def main(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = ValueModelConfig(
        hidden_dims=(32, 32), batch_size=64, max_epochs=2,
        early_stopping_patience=2, device=args.device, mixed_precision=False,
        split={"train": .8, "validation": .1, "test": .1},
    )
    source_manifest = json.loads((Path(args.dataset) / "manifest.json").read_text())
    if source_manifest.get("format") == "dribblebot_terminal_value_v1":
        value_data = Path(args.dataset)
    else:
        value_data = output / "value_dataset"
        build_value_dataset(args.dataset, value_data, config)
    train = _Prefix(ValueDataset(value_data, "train"), args.samples)
    validation = _Prefix(ValueDataset(value_data, "validation"), max(32, args.samples // 4))
    world_model, _ = load_checkpoint(args.world_model_checkpoint, args.device)
    model = TerminalValueModel(
        world_model.schema, world_model.normalizer, hidden_dims=config.hidden_dims,
        return_normalizer=ReturnNormalizer.fit(train.targets()),
    )
    checkpoint_dir = output / "checkpoint"
    TerminalValueTrainer(model, train, validation, config).fit(checkpoint_dir)
    value_model, _ = load_value_checkpoint(checkpoint_dir / "best.pt", args.device)
    state = train[0]["global_state"].to(args.device)[None]
    common = dict(horizon=3, num_candidates=32, num_elites=4, num_iterations=2,
                  candidate_batch_size=32, ensemble_objective="mean")
    reward_plan = HybridCEMMPC(world_model, config=MPCConfig(**common)).plan(state)
    value_plan = HybridCEMMPC(
        world_model,
        config=MPCConfig(**common, objective_mode="reward_plus_terminal_value"),
        terminal_value=value_model.predict,
    ).plan(state)
    figure = plot_top_down(
        world_model.schema, state[0], value_plan.predicted_states[0],
        value_plan.best_action_sequence[0], value_plan.uncertainty["state"][0],
        output=output / "value_augmented_plan.png",
        title=(f"R_H={float(value_plan.predicted_discounted_reward_return[0]):.3f}, "
               f"gamma^H V={float(value_plan.discounted_terminal_value[0]):.3f}, "
               f"J={float(value_plan.best_objective[0]):.3f}"),
        terminal_value=float(value_plan.terminal_state_value[0]),
    )
    plt.close(figure)
    simulator_output = None
    if args.run_simulator:
        simulator_output = output / "real_value_mpc"
        subprocess.run([
            sys.executable, "scripts/collect_mpc_teacher_rollouts.py",
            "--config", args.mpc_config, "--world-model-checkpoint", args.world_model_checkpoint,
            "--terminal-value-checkpoint", str(checkpoint_dir / "best.pt"),
            "--output", str(simulator_output / "teacher"),
            "--world-model-expansion-output", str(simulator_output / "world_model"),
            "--num-episodes", str(args.simulator_episodes), "--profile", "fast_validation",
        ], check=True)
    result = {
        "value_checkpoint_reloaded": str(checkpoint_dir / "best.pt"),
        "reward_only_objective": float(reward_plan.best_objective[0]),
        "value_augmented_objective": float(value_plan.best_objective[0]),
        "objective_decomposition": {
            "predicted_discounted_reward_return": float(value_plan.predicted_discounted_reward_return[0]),
            "terminal_state_value": float(value_plan.terminal_state_value[0]),
            "discounted_terminal_value": float(value_plan.discounted_terminal_value[0]),
            "terminal_value_contribution": float(value_plan.terminal_value_contribution[0]),
            **{key: float(value[0]) for key, value in value_plan.objective_components.items()},
        },
        "real_simulator_collection": None if simulator_output is None else str(simulator_output),
        "diagnostic_plot": str(output / "value_augmented_plan.png"),
    }
    (output / "smoke_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/world_model_as2")
    parser.add_argument("--world-model-checkpoint", default="checkpoints/world_model_as2/best.pt")
    parser.add_argument("--mpc-config", default="configs/mpc.yaml")
    parser.add_argument("--output", default="outputs/terminal_value_smoke")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-simulator", action="store_true")
    parser.add_argument("--simulator-episodes", type=int, default=4)
    main(parser.parse_args())
