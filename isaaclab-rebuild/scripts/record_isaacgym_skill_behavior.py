"""Record the actual legacy HighLevelSkillWrapper low-level behavior trace."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REBUILD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REBUILD_ROOT.parent
INTERPRETER_BIN = str(Path(sys.executable).parent)
os.environ["PATH"] = os.pathsep.join((INTERPRETER_BIN, os.environ.get("PATH", "")))
for import_root in (REBUILD_ROOT, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from parity.behavior_excitation import SKILL_NAMES, generate_coordinator_sequence
from parity.behavior_schema import BEHAVIOR_SCHEMA_VERSION, save_behavior_trace
from parity.policy_contract import make_legacy_policy_contract
from record_isaacgym_rollout import (
    _configure_legacy_env,
    _enforce_actor_dynamics,
    _install_deterministic_ball_asset_profile,
    _install_deterministic_command_sampler,
    _to_numpy,
    _write_deterministic_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint_root", type=Path, default=REPOSITORY_ROOT / "checkpoints/reproduction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=(*SKILL_NAMES, "switch"), default="switch")
    parser.add_argument("--switch_interval", type=int, default=8)
    parser.add_argument("--command_input", type=float, nargs=3, default=(0.25, -0.15, 0.10))
    parser.add_argument("--ball_position", type=float, nargs=3, default=(0.6, 0.0, 0.10))
    parser.add_argument("--allow_resets", action="store_true")
    return parser.parse_args()


def _load_policy_records(root: Path, device: str) -> tuple[dict, dict]:
    import torch
    import yaml

    records = {}
    checkpoint_contract = {}
    for skill in SKILL_NAMES:
        directory = root / skill
        body_path = directory / "body_latest.jit"
        adaptation_path = directory / "adaptation_module_latest.jit"
        config_path = directory / "config.yaml"
        for path in (body_path, adaptation_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing reproduction artifact: {path}")
        body = torch.jit.load(str(body_path), map_location=device).eval()
        adaptation = torch.jit.load(str(adaptation_path), map_location=device).eval()
        history_dim = _first_linear_input_dim(adaptation)
        action_clip = _action_clip(config_path, yaml)

        def policy(observation, body=body, adaptation=adaptation):
            history = observation["obs_history"].to(device)
            latent = adaptation.forward(history)
            return body.forward(torch.cat((history, latent), dim=-1))

        records[skill] = {
            "policy": policy,
            "policy_type": "walk" if skill == "walk" else "ball",
            "expected_history_dim": history_dim,
            "source": str(directory),
            "action_clip": action_clip,
        }
        checkpoint_contract[skill] = {
            "body_sha256": _sha256(body_path),
            "adaptation_sha256": _sha256(adaptation_path),
            "action_clip": action_clip,
            "history_dim": history_dim,
        }
    return records, checkpoint_contract


def _configure_wrapper_env(args: argparse.Namespace) -> object:
    legacy_args = SimpleNamespace(
        task="dribble",
        num_envs=args.num_envs,
        steps=args.steps,
        device=args.device,
        ball_position=args.ball_position,
    )
    cfg = _configure_legacy_env(legacy_args)
    cfg.env.num_robots = 1
    cfg.env.num_team_robots = 1
    cfg.env.control_all_robots = True
    cfg.env.high_level_control = True
    cfg.env.num_actions = 12
    cfg.env.high_level_num_actions = 6
    cfg.env.high_level_num_observations = 31
    cfg.env.high_level_control_interval = 1
    cfg.env.high_level_history_length = 4
    cfg.env.high_level_use_geometric_skill_fallback = False
    cfg.env.num_static_opponents = 0
    cfg.env.add_field_markers = False
    cfg.env.add_goalposts = False
    cfg.env.add_field_texture = False
    return cfg, legacy_args


def _snapshot(raw_env: object) -> dict[str, np.ndarray]:
    robot_ids = raw_env.robot_actor_idxs_all[:, 0]
    return {
        "root_pos_w": _to_numpy(raw_env.root_states[robot_ids, :3]),
        "root_quat_w": _xyzw_to_wxyz(_to_numpy(raw_env.root_states[robot_ids, 3:7])),
        "joint_pos": _to_numpy(raw_env.dof_pos[:, :12]),
        "ball_pos_w": _to_numpy(raw_env.root_states[raw_env.object_actor_idxs, :3]),
    }


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.switch_interval <= 0:
        raise ValueError("num_envs, steps, and switch_interval must be positive")

    import isaacgym
    import torch

    from dribblebot.envs.as2.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    from dribblebot.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg, legacy_args = _configure_wrapper_env(args)
    _install_deterministic_ball_asset_profile()
    raw_env = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=True, cfg=cfg)
    wrapper = None
    try:
        _enforce_actor_dynamics(raw_env, include_ball=True)
        _install_deterministic_command_sampler(raw_env)
        policies, checkpoint_contract = _load_policy_records(args.checkpoint_root, args.device)
        wrapper = HighLevelSkillWrapper(raw_env, policies, control_interval=1, history_length=4)
        wrapper.reset()
        _write_deterministic_state(raw_env, legacy_args)
        wrapper.low_level_obs_history_full.zero_()
        wrapper.low_level_actions.zero_()
        wrapper.last_low_level_actions.zero_()

        coordinator = generate_coordinator_sequence(
            args.steps,
            raw_env.num_envs,
            args.mode,
            command_input=tuple(args.command_input),
            switch_interval=args.switch_interval,
        )
        frames = {name: [value] for name, value in _snapshot(raw_env).items()}
        policy_observations = []
        policy_histories = []
        policy_actions = []
        processed_targets = []
        skill_ids = []
        skill_commands = []
        done_frames = []
        with torch.inference_mode():
            for step in range(args.steps):
                action = torch.from_numpy(coordinator[step]).to(raw_env.device)
                _, _, done, _ = wrapper.step(action)
                history = wrapper.low_level_obs_history_full[:, 0, :]
                policy_histories.append(_to_numpy(history))
                policy_observations.append(_to_numpy(history[:, -75:]))
                policy_actions.append(_to_numpy(wrapper.low_level_actions[:, 0, :]))
                processed_targets.append(_to_numpy(raw_env.joint_pos_target[:, :12]))
                skill_ids.append(_to_numpy(wrapper.skill_ids[:, 0]).astype(np.int64, copy=False))
                skill_commands.append(_to_numpy(wrapper.skill_commands[:, 0, :]))
                done_numpy = _to_numpy(done.to(dtype=torch.bool)).astype(np.bool_, copy=False)
                done_frames.append(done_numpy)
                if done.any() and not args.allow_resets:
                    raise RuntimeError(f"Legacy wrapper reset at behavior step {step}")
                for name, value in _snapshot(raw_env).items():
                    frames[name].append(value)

        arrays = {name: np.stack(values, axis=0) for name, values in frames.items()}
        arrays.update(
            {
                "time_s": np.arange(args.steps + 1, dtype=np.float64) * float(raw_env.dt),
                "coordinator_action": coordinator,
                "skill_id": np.stack(skill_ids, axis=0),
                "skill_command": np.stack(skill_commands, axis=0),
                "policy_observation": np.stack(policy_observations, axis=0),
                "policy_history": np.stack(policy_histories, axis=0),
                "policy_action": np.stack(policy_actions, axis=0),
                "processed_joint_target": np.stack(processed_targets, axis=0),
                "done": np.stack(done_frames, axis=0),
                "env_origin_w": _to_numpy(raw_env.env_origins),
            }
        )
        metadata = {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "source": "isaacgym",
            "source_version": getattr(isaacgym, "__version__", "Preview 4"),
            "task": "Legacy-AS2-HighLevelSkillWrapper",
            "step_dt": float(raw_env.dt),
            "physics_dt": float(cfg.sim.dt),
            "decimation": int(cfg.control.decimation),
            "joint_names": list(raw_env.dof_names[:12]),
            "skill_names": list(SKILL_NAMES),
            "policy_contract": make_legacy_policy_contract(include_ball=True),
            "checkpoint_contract": checkpoint_contract,
            "action_history_semantics": "duplicate_previous_policy_output",
            "contract_horizon_steps": 10,
            "mode": args.mode,
            "switch_interval": args.switch_interval,
        }
        output = save_behavior_trace(args.output, metadata, arrays)
        print(f"[INFO] Saved Isaac Gym frozen-skill behavior trace to {output}")
    finally:
        if wrapper is not None:
            wrapper.close()
        else:
            raw_env.close()


def _first_linear_input_dim(module: object) -> int:
    for parameter in module.parameters():
        if parameter.ndim == 2:
            return int(parameter.shape[1])
    raise ValueError("TorchScript adaptation module has no linear weight")


def _action_clip(config_path: Path, yaml_module: object) -> float:
    payload = yaml_module.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = _unwrap(payload.get("Cfg", payload))
    normalization = _unwrap(cfg["normalization"])
    clip = float(_unwrap(normalization["clip_actions"]))
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"Invalid action clip {clip!r} in {config_path}")
    return clip


def _unwrap(value: object) -> object:
    if isinstance(value, dict) and "value" in value and len(value) <= 2:
        return value["value"]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    return np.concatenate((quaternion[..., 3:4], quaternion[..., :3]), axis=-1)


if __name__ == "__main__":
    main()
