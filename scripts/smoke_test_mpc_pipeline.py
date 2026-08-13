"""Run a short end-to-end MPC collection and verify every saved artifact."""

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
from matplotlib import pyplot as plt

from dribblebot.mpc.collection import (
    TeacherRolloutCollector,
    build_collection_metadata,
    finalize_expansion_dataset,
)
from dribblebot.mpc.runtime import add_simulator_arguments, build_runtime
from dribblebot.mpc.teacher_dataset import (
    TEACHER_REQUIRED_KEYS,
    TeacherDataset,
    TeacherEpisodeWriter,
)
from dribblebot.mpc.visualization import plot_top_down
from dribblebot.world_model.dataset import EpisodeShardWriter, WorldModelDataset


def main(args):
    output = Path(args.output)
    teacher_root = output / "teacher"
    expansion_root = output / "world_model_expansion"
    runtime = build_runtime(args, max_candidate_diagnostics=16)
    teacher_metadata, expansion_metadata = build_collection_metadata(
        runtime.model,
        runtime.checkpoint,
        runtime.checkpoint_id,
        runtime.mpc_config,
        runtime.config["environment"],
        runtime.local_adapter,
    )
    teacher_writer = TeacherEpisodeWriter(teacher_root, teacher_metadata)
    expansion_writer = EpisodeShardWriter(
        expansion_root, expansion_metadata
    )
    collector = TeacherRolloutCollector(
        runtime.controller,
        teacher_writer,
        expansion_writer,
        runtime.checkpoint_id,
    )
    try:
        summary = collector.collect(
            num_episodes=args.num_envs,
            max_macro_steps=args.num_macro_steps,
        )
        finalize_expansion_dataset(
            expansion_root, 1.0, 0.0, 0.0, runtime.config.get("seed", 42)
        )
    finally:
        runtime.controller.close()
        runtime.env.close()
    teacher = TeacherDataset(teacher_root, verify_hashes=True)
    if not len(teacher):
        raise RuntimeError("Smoke test did not save a teacher episode")
    episode = teacher.load_episode(0)
    missing = [key for key in TEACHER_REQUIRED_KEYS if key not in episode.arrays]
    if missing:
        raise RuntimeError(f"Smoke teacher dataset is missing fields {missing}")
    expansion = WorldModelDataset(expansion_root, "train")
    if not len(expansion):
        raise RuntimeError("Smoke test did not save world-model transitions")
    ground_truth_sources = {
        str(value)
        for shard in expansion.episodes
        for value in shard["ground_truth_source"].astype(str)
    }
    if ground_truth_sources != {"real_simulator"}:
        raise RuntimeError("Expansion dataset contains a non-real ground-truth source")
    figure = plot_top_down(
        runtime.model.schema,
        episode.arrays["global_state"][0],
        episode.arrays["predicted_plan_states"][0],
        episode.arrays["predicted_plan"][0],
        episode.arrays["state_uncertainty"][0],
        actual_future=[
            episode.arrays["global_state"][0],
            episode.arrays["real_next_global_state"][0],
        ],
        output=output / "smoke_tactical.png",
        title="MPC pipeline smoke test",
    )
    plt.close(figure)
    result = {
        **summary,
        "teacher_episodes_reloaded": len(teacher),
        "teacher_transitions_reloaded": teacher.transition_count,
        "expansion_transitions_reloaded": len(expansion),
        "diagnostic_figure": str((output / "smoke_tactical.png").resolve()),
        "required_fields_verified": len(TEACHER_REQUIRED_KEYS),
    }
    (output / "smoke_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mpc.yaml")
    parser.add_argument(
        "--world-model-checkpoint",
        default="checkpoints/world_model_as2/best.pt",
    )
    parser.add_argument("--num-macro-steps", type=int, default=20)
    parser.add_argument("--output", default="outputs/mpc_smoke")
    parser.add_argument("--profile", default="fast_validation")
    add_simulator_arguments(parser)
    parser.set_defaults(num_envs=4)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
