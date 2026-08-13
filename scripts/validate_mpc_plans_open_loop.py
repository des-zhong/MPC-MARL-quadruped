"""Evaluate selected MPC plans by executing them open-loop in separate trials."""

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
import numpy as np
import torch

from dribblebot.mpc.runtime import add_simulator_arguments, build_runtime
from dribblebot.world_model.metrics import uncertainty_error_correlation


def _group_rmse(schema, prediction, actual, group, valid):
    indices = [
        index
        for feature in schema.features
        if feature.group == group
        for index in range(feature.start, feature.stop)
    ]
    error = (prediction[..., indices] - actual[..., indices]).square().mean(-1)
    selected = error[valid]
    return float(selected.mean().sqrt()) if selected.numel() else None


@torch.no_grad()
def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime(args)
    records = []
    try:
        for plan_index in range(args.num_plans):
            runtime.controller.reset()
            initial = runtime.state_adapter.extract_state(runtime.env)["tensor"]
            plan = runtime.planner.plan(initial)
            actual_states = [initial.detach().cpu()]
            real_rewards = []
            real_events = []
            executed_actions = []
            alive = torch.ones(
                runtime.env.num_envs,
                dtype=torch.bool,
                device=initial.device,
            )
            valid_steps = []
            for step in range(runtime.mpc_config.horizon):
                requested = plan.best_action_sequence[:, step]
                runtime.controller.capture.clear()
                _, reward, done, info = runtime.env.step(
                    runtime.model.action_adapter.to_wrapper_action(requested)
                )
                executed = runtime.controller._executed_action(info, initial.device)
                live = runtime.state_adapter.extract_state(runtime.env)["tensor"]
                terminal = torch.where(
                    runtime.controller.capture.valid[:, None],
                    runtime.controller.capture.states,
                    live,
                )
                event = runtime.state_adapter.extract_event_labels(
                    actual_states[-1].to(initial.device), terminal, info
                )
                actual_states.append(terminal.detach().cpu())
                real_rewards.append(reward.detach().cpu())
                real_events.append(event.detach().cpu())
                executed_actions.append(executed.detach().cpu())
                valid_steps.append(alive.detach().cpu())
                alive &= ~done.bool()
            actual = torch.stack(actual_states, dim=1)
            valid = torch.stack(valid_steps, dim=1)
            predicted = plan.predicted_states.detach().cpu()
            state_error = (
                predicted[:, 1:] - actual[:, 1:]
            ).square().mean(-1)
            uncertainty = plan.uncertainty["state"].detach().cpu()
            record = {
                "plan_index": plan_index,
                "valid_transition_count": int(valid.sum()),
                "state_rmse": (
                    float(state_error[valid].mean().sqrt())
                    if bool(valid.any())
                    else None
                ),
                "robot_position_rmse_encoded": _group_rmse(
                    runtime.model.schema,
                    predicted[:, 1:],
                    actual[:, 1:],
                    "robot_position",
                    valid,
                ),
                "ball_position_rmse_encoded": _group_rmse(
                    runtime.model.schema,
                    predicted[:, 1:],
                    actual[:, 1:],
                    "ball_position",
                    valid,
                ),
                "ball_velocity_rmse": _group_rmse(
                    runtime.model.schema,
                    predicted[:, 1:],
                    actual[:, 1:],
                    "ball_velocity",
                    valid,
                ),
                "reward_rmse": float(
                    (
                        plan.predicted_rewards.detach().cpu()[valid]
                        - torch.stack(real_rewards, 1)[valid]
                    ).square().mean().sqrt()
                ),
                "uncertainty_error_correlation": uncertainty_error_correlation(
                    uncertainty[valid], state_error[valid]
                ),
                "requested_action_modified_rate": float(
                    (
                        torch.stack(executed_actions, 1)
                        - plan.best_action_sequence.detach().cpu()
                    ).abs().gt(1.0e-5)[valid].any(-1).float().mean()
                ),
            }
            records.append(record)
            if plan_index < args.save_trajectory_plans:
                np.savez_compressed(
                    output / f"open_loop_plan_{plan_index:04d}.npz",
                    initial_state=initial.detach().cpu().numpy(),
                    requested_actions=plan.best_action_sequence.detach().cpu().numpy(),
                    executed_actions=torch.stack(executed_actions, 1).numpy(),
                    predicted_states=predicted.numpy(),
                    actual_states=actual.numpy(),
                    predicted_rewards=plan.predicted_rewards.detach().cpu().numpy(),
                    actual_rewards=torch.stack(real_rewards, 1).numpy(),
                    predicted_done=plan.predicted_done_probabilities.detach().cpu().numpy(),
                    actual_events=torch.stack(real_events, 1).numpy(),
                    valid=valid.numpy(),
                )
    finally:
        runtime.controller.close()
        runtime.env.close()
    numeric_keys = [
        key
        for key in records[0]
        if key != "plan_index" and all(row[key] is not None for row in records)
    ] if records else []
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in records])),
            "std": float(np.std([row[key] for row in records])),
        }
        for key in numeric_keys
    }
    payload = {
        "experiment": "evaluation-only open-loop execution; not teacher collection",
        "plans": records,
        "aggregate": aggregate,
        "limitation": (
            "The repository has no exact simulator clone/restore. Each plan is "
            "predicted and then executed from the same freshly reset state, "
            "rather than branched from a receding-horizon collection episode."
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mpc.yaml")
    parser.add_argument(
        "--world-model-checkpoint",
        default="checkpoints/world_model_as2/best.pt",
    )
    parser.add_argument("--profile", default="fast_validation")
    parser.add_argument("--num-plans", type=int, default=100)
    parser.add_argument("--save-trajectory-plans", type=int, default=10)
    parser.add_argument("--output", default="outputs/mpc_open_loop_validation")
    add_simulator_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
