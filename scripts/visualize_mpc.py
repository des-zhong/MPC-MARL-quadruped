"""Generate compact MPC videos and episode-level diagnostic figures."""

from __future__ import annotations

import argparse
import json
import os
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

from quadruped.mpc.runtime import add_simulator_arguments, build_runtime
from quadruped.mpc.visualization import (
    figure_to_rgb,
    plot_action_selection,
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
    width = max(1, round(frame.shape[1] * height / frame.shape[0]))
    columns = np.linspace(0, frame.shape[1] - 1, width).astype(int)
    return frame[indices[:, None], columns[None, :], :3]


@torch.no_grad()
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
                action_sequence=None,
                uncertainty=plan.uncertainty["state"][env_index],
                actual_future=torch.stack(
                    (
                        transition.state[env_index],
                        transition.next_state[env_index],
                    )
                ),
                title=(
                    f"episode {completed} step {step} | "
                    f"predicted return "
                    f"{plan.predicted_discounted_reward_return[env_index]:.3f} | "
                    f"planning {plan.planning_time_seconds:.3f}s"
                ),
                controlled_robot_count=team_size,
            )
            tactical_rgb = figure_to_rgb(tactical)
            plt.close(tactical)
            selection = plot_action_selection(runtime.model.schema, plan,
                                              transition.executed_action, env_index)
            selection_rgb = figure_to_rgb(selection)
            if step == 0:
                selection.savefig(episode_dir / "action_selection.png", dpi=150)
            plt.close(selection)
            raw = runtime.env.env
            target, distance = raw._get_high_level_camera_pose()
            distance = [value * args.camera_scale for value in distance]
            simulator_rgb = np.asarray(raw.render("rgb_array", target_loc=target,
                                                 cam_distance=distance))[..., :3]
            panels = [simulator_rgb, tactical_rgb, selection_rgb]
            combined = np.concatenate([_resize_nearest(panel, 640) for panel in panels], axis=1)
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
            if bool(transition.done[env_index]) or step >= args.max_steps:
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
                reward_error = np.asarray(predicted_rewards) - np.asarray(
                    actual_rewards
                )
                reward_correlation = None
                if (
                    len(reward_error) > 1
                    and np.std(predicted_rewards) > 0
                    and np.std(actual_rewards) > 0
                ):
                    reward_correlation = float(
                        np.corrcoef(predicted_rewards, actual_rewards)[0, 1]
                    )
                diagnostics = {
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
                    "one_step_reward_prediction": {
                        "bias": float(np.mean(reward_error)),
                        "mae": float(np.mean(np.abs(reward_error))),
                        "rmse": float(np.sqrt(np.mean(np.square(reward_error)))),
                        "pearson_correlation": reward_correlation,
                    },
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
                        "terminated": bool(transition.done[env_index]),
                        "action_selection": str(episode_dir / "action_selection.png"),
                        "real_return": float(sum(actual_rewards)),
                        "video": str(video_path),
                        "prediction_vs_reality": str(prediction_path),
                        "skill_and_parameters": str(skill_path),
                        "mpc_diagnostics": str(diagnostic_path),
                        "diagnostics_json": str(diagnostics_json),
                        "requested_action_modified_fraction": diagnostics[
                            "requested_action_modified_fraction"
                        ],
                        "one_step_reward_prediction": diagnostics[
                            "one_step_reward_prediction"
                        ],
                    }
                )
                completed += 1
                if not bool(transition.done[env_index]):
                    runtime.controller.reset()
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
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "config": args.config,
        "profile": args.profile,
        "camera_scale": args.camera_scale,
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
    parser.add_argument("--config", default="configs/mpc_joint_teams.yaml")
    parser.add_argument(
        "--world-model-checkpoint",
        default="checkpoints/reproduction/world_model/best.pt",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--save-dir", default="outputs/mpc_visualizations")
    parser.add_argument("--profile", default="teacher_high_quality")
    parser.add_argument("--candidate-diagnostics", type=int, default=128)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--camera-scale", type=float, default=0.65,
                        help="Camera distance relative to full-field view (smaller is closer).")
    add_simulator_arguments(parser)
    parser.set_defaults(num_envs=1, skill_policy_source="local",
                        walk_policy_dir="checkpoints/reproduction/walk",
                        dribble_policy_dir="checkpoints/reproduction/dribble",
                        shoot_policy_dir="checkpoints/reproduction/shoot")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.fps <= 0 or args.camera_scale <= 0:
        parser.error("episodes, max-steps, fps, and camera-scale must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
    # Isaac Gym's legacy native runtime may segfault while Python tears down
    # CUDA/PhysX objects even after gym.destroy_sim has completed. All figures,
    # videos, and summary files are explicitly closed/written by main().
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
