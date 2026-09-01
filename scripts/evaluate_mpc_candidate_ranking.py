"""Branch candidate plans across identical simulator environments and rank them."""

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
from isaacgym import gymtorch
import numpy as np
import torch

from quadruped.mpc.runtime import add_simulator_arguments, build_runtime
from quadruped.world_model.metrics import reward_ranking_metrics


def _clone_environment_zero(runtime):
    """Clone all dynamic simulator and controller state into every env row."""

    wrapper = runtime.env
    raw = wrapper.env
    rows = [raw.robot_actor_idxs_all]
    if hasattr(raw, "object_actor_idxs"):
        rows.append(raw.object_actor_idxs[:, None])
    static = getattr(raw, "static_opponent_actor_idxs", None)
    if static is not None and static.numel():
        rows.append(static)
    actor_ids = torch.cat(rows, dim=1)
    source = raw.root_states[actor_ids[0]].clone()
    origin_delta = raw.env_origins - raw.env_origins[:1]
    replicated = source[None].expand(raw.num_envs, -1, -1).clone()
    replicated[..., :3] += origin_delta[:, None, :]
    raw.root_states[actor_ids.reshape(-1)] = replicated.reshape(-1, 13)
    flat_actor_ids = actor_ids.reshape(-1).to(torch.int32)
    raw.gym.set_actor_root_state_tensor_indexed(
        raw.sim, gymtorch.unwrap_tensor(raw.root_states),
        gymtorch.unwrap_tensor(flat_actor_ids), flat_actor_ids.numel(),
    )
    if hasattr(raw, "object_actor_idxs"):
        object_roots = raw.root_states[raw.object_actor_idxs]
        raw.object_pos_world_frame[:] = object_roots[:, 0:3]
        raw.object_lin_vel[:] = object_roots[:, 7:10]
        raw.object_ang_vel[:] = object_roots[:, 10:13]
    raw.dof_pos[:] = raw.dof_pos[:1]
    raw.dof_vel[:] = raw.dof_vel[:1]
    robot_ids = raw.robot_actor_idxs_all.reshape(-1).to(torch.int32)
    raw.gym.set_dof_state_tensor_indexed(
        raw.sim, gymtorch.unwrap_tensor(raw.dof_state),
        gymtorch.unwrap_tensor(robot_ids), robot_ids.numel(),
    )
    for owner, names in (
        (raw, (
            "gait_indices", "high_level_skill_ids",
            "high_level_requested_skill_ids", "high_level_invalid_skill_mask",
            "high_level_commands", "prev_object_pos_world_frame",
            "prev_object_lin_vel", "prev_high_level_robot_ball_distances",
        )),
        (wrapper, (
            "skill_ids", "requested_skill_ids", "invalid_skill_mask",
            "collision_avoidance_mask", "skill_commands", "low_level_actions",
            "last_low_level_actions", "low_level_obs_history_full",
        )),
    ):
        for name in names:
            value = getattr(owner, name, None)
            if torch.is_tensor(value) and value.ndim and value.shape[0] == raw.num_envs:
                value[:] = value[:1]
    wrapper._update_high_level_obs()


@torch.no_grad()
def main(args):
    args.num_envs = args.num_candidates
    args.fixed_initial_state = True
    args.disable_domain_randomization = True
    runtime = build_runtime(
        args,
        max_candidate_diagnostics=args.num_candidates,
        mpc_overrides={"max_candidate_diagnostics": args.num_candidates},
    )
    records = []
    try:
        for trial in range(args.num_trials):
            runtime.controller.reset()
            _clone_environment_zero(runtime)
            state = runtime.state_adapter.extract_state(runtime.env)["tensor"]
            spread = (state - state[:1]).abs().amax().item()
            if spread > args.initial_state_tolerance:
                raise RuntimeError(
                    "Candidate branching requires identical vector environments; "
                    f"initial compact-state spread is {spread:.3g}. Set "
                    "environment.fixed_initial_state and disable domain randomization."
                )
            fixed = mask = None
            if runtime.opponent_forecaster is not None:
                fixed, mask = runtime.opponent_forecaster.fixed_action_sequence(
                    runtime.mpc_config.horizon
                )
                fixed = fixed[:1]
            plan = runtime.planner.plan(
                state[:1], fixed_action_sequence=fixed, fixed_robot_mask=mask
            )
            diagnostics = plan.candidate_diagnostics
            requested = diagnostics["action_sequences"][0]
            count = requested.shape[0]
            alive = torch.ones(count, dtype=torch.bool, device=state.device)
            actual_return = torch.zeros(count, dtype=state.dtype, device=state.device)
            discount = 1.0
            modified = []
            for step in range(runtime.mpc_config.horizon):
                _, reward, done, info = runtime.env.step(
                    runtime.model.action_adapter.to_wrapper_action(requested[:, step])
                )
                executed = runtime.controller._executed_action(info, state.device)
                modified.append((executed - requested[:, step]).abs().gt(1e-5).any(-1))
                actual_return += discount * reward * alive
                alive &= ~done.bool()
                discount *= runtime.mpc_config.gamma
            sources = {
                "learned_reward": diagnostics["learned_reward_returns"][0],
                "analytical_reward": diagnostics["analytical_reward_returns"][0],
                "full_objective": diagnostics["objectives"][0],
            }
            records.append({
                "trial": trial,
                "candidate_count": count,
                "initial_state_max_spread": spread,
                "requested_action_modified_rate": float(torch.stack(modified, 1).float().mean()),
                "valid_candidate_fraction": float(diagnostics["valid"][0].float().mean()),
                "metrics": {
                    name: reward_ranking_metrics(value, actual_return, args.top_k)
                    for name, value in sources.items()
                },
            })
    finally:
        runtime.controller.close()
        runtime.env.close()
    aggregate = {}
    for source in records[0]["metrics"] if records else ():
        aggregate[source] = {
            metric: float(np.nanmean([row["metrics"][source][metric] for row in records]))
            for metric in records[0]["metrics"][source]
        }
    full = aggregate.get("full_objective", {})
    checks = {
        "minimum_full_objective_spearman": (
            float(full.get("spearman", float("-inf")))
            >= args.minimum_full_objective_spearman
        ),
        "minimum_full_objective_pairwise_accuracy": (
            float(full.get("pairwise_accuracy", float("-inf")))
            >= args.minimum_full_objective_pairwise_accuracy
        ),
        "maximum_full_objective_selection_regret": (
            float(full.get("selection_regret", float("inf")))
            <= args.maximum_full_objective_selection_regret
        ),
    }
    payload = {
        "trials": records,
        "aggregate": aggregate,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_threshold and not payload["acceptance"]["passed"]:
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mpc_joint_teams.yaml")
    parser.add_argument("--world-model-checkpoint", default="checkpoints/world_model_as2/best.pt")
    parser.add_argument("--profile", default="teacher_high_quality")
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-full-objective-spearman", type=float, default=0.20)
    parser.add_argument(
        "--minimum-full-objective-pairwise-accuracy", type=float, default=0.55
    )
    parser.add_argument(
        "--maximum-full-objective-selection-regret", type=float, default=0.50
    )
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--initial-state-tolerance", type=float, default=1e-5)
    parser.add_argument("--output", default="outputs/mpc_candidate_ranking/metrics.json")
    add_simulator_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
    # Isaac Gym's extension can segfault during interpreter teardown after a
    # clean env.close(). All files/stdout are complete here; bypass only the
    # extension's static destructors so automated evaluations retain exit 0.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
