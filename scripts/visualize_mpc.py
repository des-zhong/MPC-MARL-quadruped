"""Generate compact MPC videos and episode-level diagnostic figures."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import isaacgym

assert isaacgym
import numpy as np
import torch
from matplotlib import pyplot as plt

from dribblebot.mpc.runtime import add_simulator_arguments, build_runtime
from dribblebot.mpc.visualization import (
    figure_to_rgb,
    multi_step_prediction_errors,
    plot_mpc_execution_diagnostics,
    plot_prediction_vs_reality,
    plot_skill_and_parameters,
    plot_top_down,
    save_video_or_frames,
)


def _resize_nearest(frame, height):
    if frame.shape[0] == height:
        return frame[..., :3]
    indices = np.linspace(0, frame.shape[0] - 1, height).astype(int)
    return frame[indices, ..., :3]


def _remove_legacy_episode_outputs(episode_dir):
    """Remove artifacts produced by older, verbose visualization runs."""

    for directory_name in ("frames", "mpc_episode"):
        directory = episode_dir / directory_name
        if directory.is_dir():
            shutil.rmtree(directory)
    for pattern in ("cem_convergence_*.png", "candidate_endpoints_*.png"):
        for path in episode_dir.glob(pattern):
            path.unlink()


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime(
        args,
        max_candidate_diagnostics=args.candidate_diagnostics,
    )
    runtime.controller.reset()
    completed = 0
    step = 0
    episode_frames = []
    actual_states = []
    predicted_states = []
    actual_rewards = []
    predicted_rewards = []
    uncertainties = []
    actual_events = []
    predicted_events = []
    selected_actions = []
    step_diagnostics = []
    summaries = []
    try:
        while completed < args.episodes:
            transition = runtime.controller.step()
            env_index = 0
            episode_dir = save_dir / f"episode_{completed:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            if step == 0:
                _remove_legacy_episode_outputs(episode_dir)
            plan = transition.plan
            team_size = (
                int(runtime.config["environment"]["team_size"])
                if "team_size" in runtime.config["environment"]
                else None
            )
            tactical = plot_top_down(
                runtime.model.schema,
                transition.state[env_index],
                predicted_states=plan.predicted_states[env_index],
                action_sequence=plan.best_action_sequence[env_index],
                uncertainty=plan.uncertainty["state"][env_index],
                actual_future=torch.stack(
                    (
                        transition.state[env_index],
                        transition.next_state[env_index],
                    )
                ),
                title=(
                    f"episode {completed} step {step} | "
                    f"R_H {plan.predicted_discounted_reward_return[env_index]:.3f} + "
                    f"V {plan.terminal_value_contribution[env_index]:.3f} | "
                    f"objective {plan.best_objective[env_index]:.3f} | "
                    f"planning {plan.planning_time_seconds:.3f}s | "
                    f"fallback {bool(plan.fallback_used[env_index])}"
                ),
                terminal_value=float(plan.terminal_state_value[env_index]),
                controlled_robot_count=team_size,
            )
            tactical_rgb = figure_to_rgb(tactical)
            plt.close(tactical)
            try:
                simulator_rgb = runtime.env.env.render("rgb_array")
                simulator_rgb = _resize_nearest(
                    np.asarray(simulator_rgb), tactical_rgb.shape[0]
                )
                tactical_resized = _resize_nearest(
                    tactical_rgb, simulator_rgb.shape[0]
                )
                combined = np.concatenate((simulator_rgb, tactical_resized), axis=1)
            except Exception:
                combined = tactical_rgb
            episode_frames.append(combined)
            if not actual_states:
                actual_states.append(
                    transition.state[env_index].detach().cpu()
                )
                predicted_states.append(
                    transition.state[env_index].detach().cpu()
                )
            (
                predicted_next,
                predicted_reward,
                _,
                event_probability,
                uncertainty,
            ) = runtime.model.predict_next(
                transition.state,
                transition.executed_action,
                deterministic=True,
            )
            actual_states.append(
                transition.next_state[env_index].detach().cpu()
            )
            predicted_states.append(
                predicted_next[env_index].detach().cpu()
            )
            actual_rewards.append(float(transition.reward[env_index]))
            predicted_rewards.append(float(predicted_reward[env_index]))
            uncertainties.append(
                float(uncertainty["mean_state_uncertainty"][env_index])
            )
            actual_events.append(
                transition.event_labels[env_index].detach().cpu()
            )
            predicted_events.append(
                event_probability[env_index].detach().cpu()
            )
            selected_actions.append(
                transition.executed_action[env_index].detach().cpu()
            )
            step_diagnostics.append(
                {
                    "step": step,
                    "fallback_used": bool(plan.fallback_used[env_index]),
                    "requested_action_modified": bool(
                        transition.requested_action_modified[env_index]
                    ),
                    "teacher_action_executed": bool(
                        transition.teacher_action_executed[env_index]
                    ),
                    "best_objective": float(plan.best_objective[env_index]),
                    "predicted_discounted_reward_return": float(
                        plan.predicted_discounted_reward_return[env_index]
                    ),
                    "planning_time_seconds": float(plan.planning_time_seconds),
                    "max_plan_state_uncertainty": float(
                        plan.uncertainty["state"][env_index].max()
                    ),
                    "objective_components": {
                        name: float(value[env_index])
                        for name, value in plan.objective_components.items()
                    },
                }
            )
            step += 1
            if bool(transition.done[env_index]):
                prediction_path = episode_dir / "prediction_vs_reality.png"
                plot_prediction_vs_reality(
                    runtime.model.schema,
                    torch.stack(predicted_states),
                    torch.stack(actual_states),
                    predicted_rewards,
                    actual_rewards,
                    uncertainties,
                    torch.stack(predicted_events),
                    torch.stack(actual_events),
                    runtime.model.event_names,
                    prediction_path,
                )
                skill_path = episode_dir / "skill_and_parameters.png"
                plot_skill_and_parameters(
                    torch.stack(selected_actions),
                    runtime.model.action_adapter.num_robots,
                    skill_path,
                )
                multi_step_errors = multi_step_prediction_errors(
                    runtime.model,
                    torch.stack(actual_states),
                    torch.stack(selected_actions),
                    actual_rewards,
                    max_horizon=runtime.mpc_config.horizon,
                )
                diagnostic_path = episode_dir / "mpc_diagnostics.png"
                plot_mpc_execution_diagnostics(
                    step_diagnostics,
                    multi_step_errors,
                    diagnostic_path,
                )
                diagnostics_json = episode_dir / "diagnostics.json"
                diagnostics = {
                    "fallback_fraction": float(
                        np.mean([row["fallback_used"] for row in step_diagnostics])
                    ),
                    "requested_action_modified_fraction": float(
                        np.mean(
                            [
                                row["requested_action_modified"]
                                for row in step_diagnostics
                            ]
                        )
                    ),
                    "teacher_action_executed_fraction": float(
                        np.mean(
                            [
                                row["teacher_action_executed"]
                                for row in step_diagnostics
                            ]
                        )
                    ),
                    "multi_step_prediction_errors": multi_step_errors,
                    "steps": step_diagnostics,
                }
                diagnostics_json.write_text(
                    json.dumps(diagnostics, indent=2, sort_keys=True)
                )
                video_path = save_video_or_frames(
                    episode_frames,
                    episode_dir / "mpc_episode.mp4",
                    args.fps,
                )
                summaries.append(
                    {
                        "episode": completed,
                        "steps": len(actual_rewards),
                        "real_return": float(sum(actual_rewards)),
                        "video": str(video_path),
                        "prediction_vs_reality": str(prediction_path),
                        "skill_and_parameters": str(skill_path),
                        "mpc_diagnostics": str(diagnostic_path),
                        "diagnostics_json": str(diagnostics_json),
                        "fallback_fraction": diagnostics["fallback_fraction"],
                        "requested_action_modified_fraction": diagnostics[
                            "requested_action_modified_fraction"
                        ],
                    }
                )
                completed += 1
                episode_frames = []
                actual_states = []
                predicted_states = []
                actual_rewards = []
                predicted_rewards = []
                uncertainties = []
                actual_events = []
                predicted_events = []
                selected_actions = []
                step_diagnostics = []
                step = 0
    finally:
        runtime.controller.close()
        runtime.env.close()
    summary = {
        "episodes": summaries,
        "simulator_camera_available_for_env": 0,
        "static_obstacle_rendering": (
            "none; checkpoint uses two dynamic robot teams"
            if not any(
                feature.name.startswith("obstacle_")
                for feature in runtime.model.schema.features
            )
            else (
                "Conservative circumscribed radius because checkpoint state "
                "omits the randomized static-box yaw."
            )
        ),
    }
    (save_dir / "visualization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mpc.yaml")
    parser.add_argument(
        "--world-model-checkpoint",
        default="wandb/run-20260725_151500-cjmpg2he/files/best.pt",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save-dir", default="outputs/mpc_visualizations")
    parser.add_argument("--profile", default="visualization")
    parser.add_argument("--candidate-diagnostics", type=int, default=0)
    parser.add_argument("--fps", type=float, default=5.0)
    add_simulator_arguments(parser)
    parser.set_defaults(num_envs=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
    # Isaac Gym's legacy native runtime may segfault while Python tears down
    # CUDA/PhysX objects even after gym.destroy_sim has completed. All figures,
    # videos, and summary files are explicitly closed/written by main().
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
