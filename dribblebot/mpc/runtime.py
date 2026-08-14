"""Shared runtime construction for simulator-facing MPC scripts."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from dribblebot.world_model.state_adapter import FootballWorldModelStateAdapter
from dribblebot.world_model.trainer import load_checkpoint

from .config import load_mpc_config
from .hybrid_cem import HybridCEMMPC
from .objective import MPCObjective
from .terminal_value import load_value_checkpoint
from .local_observation import LocalObservationAdapter
from .simulator_controller import (
    MPCSimulatorController,
    validate_environment_compatibility,
)
from .teacher_dataset import file_sha256


@dataclass
class MPCRuntime:
    config: dict
    mpc_config: object
    model: object
    checkpoint: dict
    checkpoint_id: str
    env: object
    state_adapter: object
    local_adapter: object
    planner: object
    controller: object
    value_model: object = None
    value_checkpoint: object = None
    opponent_forecaster: object = None
    opponent_policy: object = None


def add_simulator_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--num-robots",
        type=int,
        default=None,
        help=(
            "Robots per team for joint-team configs; total controlled robots "
            "for legacy static-obstacle configs."
        ),
    )
    parser.add_argument("--skill-checkpoint", default="latest")
    parser.add_argument(
        "--terminal-value-checkpoint",
        default=None,
        help="Override mpc.terminal_value.checkpoint; omitted checkpoints fall back to reward-only.",
    )
    parser.add_argument(
        "--objective-mode",
        choices=("reward_only", "terminal_value_only", "reward_plus_terminal_value"),
        default=None,
    )
    parser.add_argument("--terminal-value-coefficient", type=float, default=None)
    parser.add_argument("--walk-wandb-run", default="des_zhong/as2_walking/3a6g1def")
    parser.add_argument("--dribble-wandb-run", default="des_zhong/as2_dribbling/cp9m21ay")
    parser.add_argument("--shoot-wandb-run", default="des_zhong/as2_shooting/bve3isir")
    parser.add_argument(
        "--skill-policy-source",
        choices=("wandb", "local"),
        default="wandb",
    )
    parser.add_argument("--walk-policy-dir", default=None)
    parser.add_argument("--dribble-policy-dir", default=None)
    parser.add_argument("--shoot-policy-dir", default=None)
    parser.add_argument(
        "--opponent-policy-source",
        choices=("local", "wandb", "none"),
        default="local",
        help=(
            "Frozen high-level opponent used by joint-team MPC. 'none' holds "
            "opponent robots on zero-command reposition actions."
        ),
    )
    parser.add_argument(
        "--opponent-policy-dir",
        default="checkpoints/reproduction/high_level",
    )
    parser.add_argument("--opponent-wandb-run", default=None)
    parser.add_argument("--opponent-policy-checkpoint", default="latest")
    parser.add_argument("--walk-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--dribble-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--shoot-x-speed-scale", type=float, default=3.0)
    parser.add_argument("--shoot-y-speed-scale", type=float, default=3.0)
    return parser


def build_runtime(
    args,
    *,
    capture_terminal_state: bool = True,
    max_candidate_diagnostics: Optional[int] = None,
    mpc_overrides: Optional[dict] = None,
) -> MPCRuntime:
    mpc_config, config = load_mpc_config(args.config, getattr(args, "profile", None))
    cli_overrides = {}
    if getattr(args, "objective_mode", None) is not None:
        cli_overrides["objective_mode"] = args.objective_mode
        cli_overrides["use_terminal_value"] = args.objective_mode != "reward_only"
    if getattr(args, "terminal_value_coefficient", None) is not None:
        cli_overrides["terminal_value_coefficient"] = args.terminal_value_coefficient
    effective_overrides = dict(mpc_overrides or {})
    if "objective_mode" in effective_overrides:
        effective_overrides.setdefault(
            "use_terminal_value", effective_overrides["objective_mode"] != "reward_only"
        )
    if mpc_overrides or cli_overrides:
        from .config import MPCConfig
        from dribblebot.world_model.config import deep_update

        mpc_config = MPCConfig.from_mapping(
            deep_update(deep_update(mpc_config.to_dict(), effective_overrides), cli_overrides)
        )
    if max_candidate_diagnostics is not None:
        mpc_config.max_candidate_diagnostics = int(max_candidate_diagnostics)
        mpc_config.validate()
    if args.num_envs is not None:
        config["environment"]["num_envs"] = int(args.num_envs)
    model, checkpoint = load_checkpoint(args.world_model_checkpoint, args.device)
    model.eval()
    value_model = value_checkpoint = None
    requested_value = getattr(args, "terminal_value_checkpoint", None) or mpc_config.terminal_value_checkpoint
    needs_value = mpc_config.objective_mode in {
        "terminal_value_only", "reward_plus_terminal_value"
    }
    if needs_value and requested_value:
        try:
            mpc_config.terminal_value_checkpoint = str(requested_value)
            value_model, value_checkpoint = load_value_checkpoint(requested_value, args.device)
            if value_model.schema.to_dict() != model.schema.to_dict():
                raise ValueError("terminal-value and world-model state schemas differ")
            value_gamma = float(value_checkpoint["gamma"])
            if abs(value_gamma - mpc_config.gamma) > 1.0e-9:
                raise ValueError(
                    f"terminal value gamma {value_gamma} differs from MPC gamma {mpc_config.gamma}"
                )
            if mpc_config.terminal_value_clip:
                percentiles = value_checkpoint.get("return_statistics", {}).get("percentiles", {})
                if mpc_config.terminal_value_clip_min is None and "1" in percentiles:
                    mpc_config.terminal_value_clip_min = float(percentiles["1"])
                if mpc_config.terminal_value_clip_max is None and "99" in percentiles:
                    mpc_config.terminal_value_clip_max = float(percentiles["99"])
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            if mpc_config.terminal_value_required:
                raise
            warnings.warn(f"Terminal value unavailable ({error}); using reward-only MPC")
            value_model = value_checkpoint = None
            mpc_config.objective_mode = "reward_only"
            mpc_config.use_terminal_value = False
    elif needs_value:
        if mpc_config.terminal_value_required:
            raise FileNotFoundError("Terminal-value MPC requires mpc.terminal_value.checkpoint")
        warnings.warn("No terminal value checkpoint configured; using reward-only MPC")
        mpc_config.objective_mode = "reward_only"
        mpc_config.use_terminal_value = False
    checkpoint_num_robots = model.action_adapter.num_robots
    checkpoint_num_obstacles = sum(
        feature.name.startswith("obstacle_")
        for feature in model.schema.features
    )
    environment_config = config["environment"]
    joint_teams = "team_size" in environment_config
    if args.num_robots is not None and int(args.num_robots) < 1:
        raise ValueError("--num-robots must be at least 1")
    if joint_teams:
        team_size = int(
            args.num_robots
            if args.num_robots is not None
            else environment_config["team_size"]
        )
        expected_robots = 2 * team_size
        if expected_robots != checkpoint_num_robots:
            raise ValueError(
                f"--num-robots={team_size} means {expected_robots} physical robots "
                f"for two teams, but the checkpoint contains {checkpoint_num_robots}"
            )
        if checkpoint_num_obstacles:
            raise ValueError(
                "Joint-team MPC requires a checkpoint with zero static obstacles; "
                f"this checkpoint contains {checkpoint_num_obstacles} obstacle slots"
            )
        environment_config["team_size"] = team_size
        environment_config["num_robots"] = expected_robots
        config["world_model"]["max_obstacles"] = 0
    else:
        requested_robots = int(
            checkpoint_num_robots
            if args.num_robots is None
            else args.num_robots
        )
        if requested_robots != checkpoint_num_robots:
            raise ValueError(
                f"--num-robots={requested_robots} does not match world-model "
                f"checkpoint robot count {checkpoint_num_robots}"
            )
        environment_config["num_robots"] = requested_robots
        config["world_model"]["max_obstacles"] = checkpoint_num_obstacles
    # Local import preserves Isaac Gym's required import-before-torch order in
    # each simulator-facing entry point.
    from scripts.collect_world_model_data import build_environment

    env = build_environment(args, config)
    validate_environment_compatibility(
        env,
        model,
        checkpoint,
        int(config["world_model"]["macro_action_steps"]),
    )
    state_adapter = FootballWorldModelStateAdapter(
        env,
        sum(feature.name.startswith("obstacle_") for feature in model.schema.features),
        model.schema,
        event_names=model.event_names,
        num_robots=model.action_adapter.num_robots,
    )
    local_adapter = LocalObservationAdapter(model.schema)
    controlled_robot_count = team_size if joint_teams else checkpoint_num_robots
    objective = MPCObjective(
        model.schema,
        model.action_adapter,
        model.event_names,
        mpc_config,
        terminal_value=None if value_model is None else value_model.predict,
        controlled_robot_count=controlled_robot_count,
    )
    planner = HybridCEMMPC(
        model,
        state_adapter,
        model.action_adapter,
        objective=objective,
        config=mpc_config,
        terminal_value=None if value_model is None else value_model.predict,
    )
    opponent_forecaster = opponent_policy = None
    if joint_teams:
        from .opponent_forecast import (
            FrozenPolicyOpponentForecaster,
            ZeroOpponentForecaster,
        )

        opponent_source = str(getattr(args, "opponent_policy_source", "local"))
        if opponent_source == "none":
            opponent_forecaster = ZeroOpponentForecaster(
                env, team_size, model.action_adapter
            )
        else:
            from scripts.play_high_level import load_high_level_policy

            opponent_args = SimpleNamespace(
                high_level_policy_source=opponent_source,
                high_level_policy_dir=getattr(args, "opponent_policy_dir", None),
                high_level_wandb_run=getattr(args, "opponent_wandb_run", None),
                high_level_checkpoint=getattr(
                    args, "opponent_policy_checkpoint", "latest"
                ),
                policy_device=args.policy_device,
            )
            opponent_policy = load_high_level_policy(opponent_args)
            opponent_forecaster = FrozenPolicyOpponentForecaster(
                env,
                team_size,
                model.action_adapter,
                opponent_policy,
                opponent_device=args.policy_device,
            )
    controller = MPCSimulatorController(
        env,
        planner,
        state_adapter,
        local_adapter,
        capture_terminal_state=capture_terminal_state,
        opponent_forecaster=opponent_forecaster,
    )
    checkpoint_id = file_sha256(args.world_model_checkpoint)
    return MPCRuntime(
        config,
        mpc_config,
        model,
        checkpoint,
        checkpoint_id,
        env,
        state_adapter,
        local_adapter,
        planner,
        controller,
        value_model,
        value_checkpoint,
        opponent_forecaster,
        opponent_policy,
    )
