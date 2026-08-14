"""Interactively inspect and execute one MPC macro step at a time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import isaacgym

assert isaacgym
import torch
from matplotlib import pyplot as plt

from dribblebot.mpc.runtime import add_simulator_arguments, build_runtime
from dribblebot.mpc.visualization import plot_top_down


def _print_plan(runtime, transition):
    plan = transition.plan
    skills, parameters = runtime.model.action_adapter.unpack(
        plan.first_joint_action
    )
    print(
        f"objective={float(plan.best_objective[0]):.4f} "
        f"planning={plan.planning_time_seconds:.4f}s "
        f"fallback={bool(plan.fallback_used[0])}"
    )
    print(f"skills={skills[0].tolist()} parameters={parameters[0].tolist()}")
    print(
        "components="
        + str(
            {
                name: round(float(value[0]), 5)
                for name, value in plan.objective_components.items()
            }
        )
    )
    print(
        f"max_state_uncertainty={float(plan.uncertainty['max_state'][0]):.6f}"
    )
    print(
        "first_skill_probabilities="
        + str(plan.final_skill_probabilities[0, 0].detach().cpu().tolist())
    )


def main(args):
    runtime = build_runtime(args, max_candidate_diagnostics=64)
    runtime.controller.reset()
    step = 0
    try:
        while True:
            if args.non_interactive_steps is None:
                command = input(
                    "[enter/n] next, r reset, h N horizon, u X uncertainty, "
                    "w toggle warm-start, s save-next, q quit > "
                ).strip()
            else:
                command = "n" if step < args.non_interactive_steps else "q"
            if command == "q":
                break
            if command == "r":
                runtime.controller.reset()
                step = 0
                print("environment and planner reset")
                continue
            if command.startswith("h "):
                runtime.planner.config.horizon = int(command.split()[1])
                runtime.planner.config.validate()
                runtime.controller.planner_state = None
                print(f"horizon={runtime.planner.config.horizon}")
                continue
            if command.startswith("u "):
                runtime.planner.config.uncertainty_penalty = float(
                    command.split()[1]
                )
                runtime.planner.config.validate()
                print(
                    f"uncertainty_penalty={runtime.planner.config.uncertainty_penalty}"
                )
                continue
            if command == "w":
                runtime.planner.config.warm_start = not runtime.planner.config.warm_start
                runtime.controller.planner_state = None
                print(f"warm_start={runtime.planner.config.warm_start}")
                continue
            save = command == "s"
            transition = runtime.controller.step()
            _print_plan(runtime, transition)
            if save or args.save_every_step:
                figure = plot_top_down(
                    runtime.model.schema,
                    transition.state[0],
                    transition.plan.predicted_states[0],
                    transition.plan.best_action_sequence[0],
                    transition.plan.uncertainty["state"][0],
                    actual_future=torch.stack(
                        (transition.state[0], transition.next_state[0])
                    ),
                    output=Path(args.save_dir) / f"step_{step:06d}.png",
                    controlled_robot_count=(
                        int(runtime.config["environment"]["team_size"])
                        if "team_size" in runtime.config["environment"]
                        else None
                    ),
                )
                plt.close(figure)
            step += 1
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
    parser.add_argument("--profile", default="fast_validation")
    parser.add_argument("--save-dir", default="outputs/mpc_debug")
    parser.add_argument("--save-every-step", action="store_true")
    parser.add_argument("--non-interactive-steps", type=int, default=None)
    add_simulator_arguments(parser)
    parser.set_defaults(num_envs=1)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
