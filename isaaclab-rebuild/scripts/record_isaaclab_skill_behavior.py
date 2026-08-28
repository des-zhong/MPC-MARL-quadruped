"""Record the manager-based FrozenSkillPolicyAction behavior trace."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REBUILD_ROOT = Path(__file__).resolve().parents[1]
if str(REBUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(REBUILD_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Skill-Flat-Play-v0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", choices=("walk", "dribble", "shoot", "switch"), default="switch")
parser.add_argument("--switch_interval", type=int, default=8)
parser.add_argument("--command_input", type=float, nargs=3, default=(0.25, -0.15, 0.10))
parser.add_argument("--ball_position", type=float, nargs=3, default=(0.6, 0.0, 0.10))
parser.add_argument("--allow_resets", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401
import isaaclab  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from parity.behavior_excitation import SKILL_NAMES, generate_coordinator_sequence  # noqa: E402
from parity.behavior_schema import BEHAVIOR_SCHEMA_VERSION, save_behavior_trace  # noqa: E402
from parity.policy_contract import make_legacy_policy_contract  # noqa: E402
from parity.runtime_profiles import configure_isaaclab_env_for_parity  # noqa: E402


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def _write_deterministic_state(raw_env: object) -> None:
    robot = raw_env.scene["robot"]
    ball = raw_env.scene["ball"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += raw_env.scene.env_origins
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))
    ball_state = ball.data.default_root_state.clone()
    ball_state[:, :3] = raw_env.scene.env_origins + torch.tensor(
        args_cli.ball_position,
        device=raw_env.device,
        dtype=ball_state.dtype,
    )
    ball_state[:, 3:7] = ball_state.new_tensor((1.0, 0.0, 0.0, 0.0))
    ball_state[:, 7:] = 0.0
    ball.write_root_state_to_sim(ball_state)
    raw_env.scene.write_data_to_sim()
    raw_env.sim.forward()
    raw_env.scene.update(dt=0.0)
    raw_env.episode_length_buf.zero_()
    raw_env.common_step_counter = 0
    raw_env.action_manager.get_term("skill_policy").reset()


def _snapshot(raw_env: object, joint_ids: list[int]) -> dict[str, np.ndarray]:
    robot = raw_env.scene["robot"]
    ball = raw_env.scene["ball"]
    return {
        "root_pos_w": _to_numpy(robot.data.root_pos_w),
        "root_quat_w": _to_numpy(robot.data.root_quat_w),
        "joint_pos": _to_numpy(robot.data.joint_pos[:, joint_ids]),
        "ball_pos_w": _to_numpy(ball.data.root_pos_w),
    }


def _checkpoint_contract(action_term: object) -> dict:
    result = {}
    for skill_id, skill in enumerate(SKILL_NAMES):
        policy = action_term._policies.policies[skill_id]
        result[skill] = {
            "body_sha256": _sha256(Path(policy.spec.body_path)),
            "adaptation_sha256": _sha256(Path(policy.spec.adaptation_path)),
            "action_clip": float(policy.spec.action_clip),
            "history_dim": int(policy.history_dim),
        }
    return result


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0 or args_cli.switch_interval <= 0:
        raise ValueError("num_envs, steps, and switch_interval must be positive")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    configure_isaaclab_env_for_parity(env_cfg, seed=args_cli.seed, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    try:
        env.reset(seed=args_cli.seed)
        action_term = raw_env.action_manager.get_term("skill_policy")
        joint_ids = list(action_term._joint_ids)
        joint_names = list(action_term._joint_names)
        _write_deterministic_state(raw_env)
        coordinator = generate_coordinator_sequence(
            args_cli.steps,
            raw_env.num_envs,
            args_cli.mode,
            command_input=tuple(args_cli.command_input),
            switch_interval=args_cli.switch_interval,
        )
        frames = {name: [value] for name, value in _snapshot(raw_env, joint_ids).items()}
        policy_observations = []
        policy_histories = []
        policy_actions = []
        processed_targets = []
        skill_ids = []
        skill_commands = []
        done_frames = []
        with torch.inference_mode():
            for step in range(args_cli.steps):
                action = torch.from_numpy(coordinator[step]).to(raw_env.device)
                _, _, terminated, truncated, _ = env.step(action)
                done = torch.logical_or(terminated, truncated)
                history = action_term.policy_history
                policy_histories.append(_to_numpy(history))
                policy_observations.append(_to_numpy(history[:, -75:]))
                policy_actions.append(_to_numpy(action_term.policy_action))
                processed_targets.append(_to_numpy(action_term.processed_actions))
                skill_ids.append(_to_numpy(action_term.skill_ids).astype(np.int64, copy=False))
                skill_commands.append(_to_numpy(action_term.skill_commands))
                done_numpy = _to_numpy(done).astype(np.bool_, copy=False)
                done_frames.append(done_numpy)
                if done.any() and not args_cli.allow_resets:
                    raise RuntimeError(f"Isaac Lab skill task reset at behavior step {step}")
                for name, value in _snapshot(raw_env, joint_ids).items():
                    frames[name].append(value)

        arrays = {name: np.stack(values, axis=0) for name, values in frames.items()}
        arrays.update(
            {
                "time_s": np.arange(args_cli.steps + 1, dtype=np.float64) * float(raw_env.step_dt),
                "coordinator_action": coordinator,
                "skill_id": np.stack(skill_ids, axis=0),
                "skill_command": np.stack(skill_commands, axis=0),
                "policy_observation": np.stack(policy_observations, axis=0),
                "policy_history": np.stack(policy_histories, axis=0),
                "policy_action": np.stack(policy_actions, axis=0),
                "processed_joint_target": np.stack(processed_targets, axis=0),
                "done": np.stack(done_frames, axis=0),
                "env_origin_w": _to_numpy(raw_env.scene.env_origins),
            }
        )
        metadata = {
            "schema_version": BEHAVIOR_SCHEMA_VERSION,
            "source": "isaaclab",
            "source_version": getattr(isaaclab, "__version__", "unknown"),
            "task": args_cli.task,
            "step_dt": float(raw_env.step_dt),
            "physics_dt": float(raw_env.physics_dt),
            "decimation": int(raw_env.cfg.decimation),
            "joint_names": joint_names,
            "skill_names": list(SKILL_NAMES),
            "policy_contract": make_legacy_policy_contract(include_ball=True),
            "checkpoint_contract": _checkpoint_contract(action_term),
            "action_history_semantics": str(action_term.cfg.action_history_semantics),
            "contract_horizon_steps": 10,
            "mode": args_cli.mode,
            "switch_interval": args_cli.switch_interval,
        }
        output = save_behavior_trace(args_cli.output, metadata, arrays)
        print(f"[INFO] Saved Isaac Lab frozen-skill behavior trace to {output}")
    finally:
        env.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
