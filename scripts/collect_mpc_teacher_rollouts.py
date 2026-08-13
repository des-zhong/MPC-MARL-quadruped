"""Collect real receding-horizon MPC teacher episodes and expansion data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import isaacgym

assert isaacgym

from dribblebot.mpc.collection import (
    TeacherRolloutCollector,
    build_collection_metadata,
    finalize_expansion_dataset,
)
from dribblebot.mpc.runtime import add_simulator_arguments, build_runtime
from dribblebot.mpc.teacher_dataset import TeacherEpisodeWriter
from dribblebot.world_model.dataset import EpisodeShardWriter


def main(args):
    expansion = Path(args.world_model_expansion_output)
    if (expansion / "manifest.json").exists():
        raise FileExistsError(
            f"Expansion dataset {expansion} already exists; use a new version directory"
        )
    runtime = build_runtime(args)
    teacher_metadata, expansion_metadata = build_collection_metadata(
        runtime.model,
        runtime.checkpoint,
        runtime.checkpoint_id,
        runtime.mpc_config,
        runtime.config["environment"],
        runtime.local_adapter,
        {
            skill: record.get("policy_metadata", {})
            for skill, record in getattr(runtime.env, "skill_policies", {}).items()
        },
    )
    teacher_metadata["macro_action_steps"] = runtime.env.control_interval
    teacher_metadata["low_level_control_dt_seconds"] = float(runtime.env.env.dt)
    teacher_metadata["physics_dt_seconds"] = float(
        runtime.env.env.sim_params.dt
    )
    expansion_metadata.update(
        {
            "macro_action_steps": runtime.env.control_interval,
            "low_level_control_dt_seconds": float(runtime.env.env.dt),
            "physics_dt_seconds": float(runtime.env.env.sim_params.dt),
        }
    )
    teacher_writer = TeacherEpisodeWriter(
        args.output, teacher_metadata, resume=args.resume
    )
    expansion_writer = EpisodeShardWriter(expansion, expansion_metadata)
    collector = TeacherRolloutCollector(
        runtime.controller,
        teacher_writer,
        expansion_writer,
        runtime.checkpoint_id,
        behavior_mode="mpc",
    )
    try:
        summary = collector.collect(
            args.num_episodes,
            max_macro_steps=args.max_macro_steps,
        )
        split_cfg = runtime.config.get("world_model_expansion", {})
        summary["expansion"] = finalize_expansion_dataset(
            expansion,
            float(split_cfg.get("train_fraction", 0.8)),
            float(split_cfg.get("validation_fraction", 0.2)),
            float(split_cfg.get("test_fraction", 0.0)),
            int(runtime.config.get("seed", 42)),
        )
        summary["teacher_output"] = str(Path(args.output).resolve())
        summary["world_model_expansion_output"] = str(expansion.resolve())
        summary_path = Path(args.output) / "collection_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        runtime.controller.close()
        runtime.env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mpc.yaml")
    parser.add_argument(
        "--world-model-checkpoint",
        default="checkpoints/world_model_as2/best.pt",
    )
    parser.add_argument("--output", default="data/mpc_teacher")
    parser.add_argument(
        "--world-model-expansion-output",
        default="data/world_model_iterations/mpc_iteration_000",
    )
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--max-macro-steps", type=int, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--resume", action="store_true")
    add_simulator_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
