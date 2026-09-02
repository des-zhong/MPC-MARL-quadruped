"""Instantiate a migrated task and run bounded random actions as a smoke test."""

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Velocity-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from dribblebot_isaaclab.tasks.manager_based.football import mdp  # noqa: E402


def _assert_finite(value: object, name: str) -> None:
    """Raise a useful error when a manager term emits NaN/Inf values."""

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values")
    elif isinstance(value, dict):
        for key, nested in value.items():
            _assert_finite(nested, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            _assert_finite(nested, f"{name}[{index}]")


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] observation_space={env.observation_space}")
    print(f"[INFO] action_space={env.action_space}")
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    if len(robot.joint_names) != 12:
        raise RuntimeError(f"AS2 must expose 12 joints, got {robot.joint_names}")

    # Isaac Lab/URDF may expose articulation DOFs grouped by joint type
    # (all hips, then thighs, then calves).  The frozen legacy policies use
    # the leg-major order instead.  Never compare ``robot.joint_names``
    # directly: resolve the canonical policy names explicitly and verify the
    # returned mapping instead.
    legacy_joint_ids, resolved_legacy_names = robot.find_joints(
        list(mdp.LEGACY_JOINT_NAMES),
        preserve_order=True,
    )
    if list(resolved_legacy_names) != list(mdp.LEGACY_JOINT_NAMES):
        raise RuntimeError(
            "AS2 legacy joint mapping could not be resolved in policy order; "
            f"expected {list(mdp.LEGACY_JOINT_NAMES)}, got {list(resolved_legacy_names)} "
            f"from articulation order {list(robot.joint_names)}"
        )
    if len(set(legacy_joint_ids)) != len(mdp.LEGACY_JOINT_NAMES):
        raise RuntimeError(f"AS2 legacy joint mapping contains duplicate indices: {legacy_joint_ids}")
    action_dim = raw_env.action_manager.total_action_dim
    if action_dim not in (3, 4, 12):
        raise RuntimeError(
            "Migrated AS2 smoke tasks must expose 12D joint actions, the 4D hybrid "
            f"skill coordinator, or a 3D fixed-skill command; got {action_dim}"
        )
    action_term_name = "skill_policy" if action_dim in (3, 4) else "joint_pos"
    action_term = raw_env.action_manager.get_term(action_term_name)
    resolved_action_names = getattr(action_term, "_joint_names", None)
    if resolved_action_names is not None and list(resolved_action_names) != list(mdp.LEGACY_JOINT_NAMES):
        raise RuntimeError(
            f"{action_term_name} action term is not in legacy joint order; "
            f"expected {list(mdp.LEGACY_JOINT_NAMES)}, got {list(resolved_action_names)}"
        )
    reset_result = env.reset()
    _assert_finite(reset_result[0] if isinstance(reset_result, tuple) else reset_result, "reset observation")
    try:
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                if not simulation_app.is_running():
                    break
                actions = 2.0 * torch.rand(
                    (raw_env.num_envs, action_dim),
                    device=raw_env.device,
                ) - 1.0
                if action_dim == 4:
                    actions[:, 0] = torch.randint(0, 3, (raw_env.num_envs,), device=raw_env.device)
                if not torch.isfinite(actions).all():
                    raise RuntimeError("generated random action contains non-finite values")
                step_result = env.step(actions)
                _assert_finite(step_result[0], "step observation")
                joint_target = robot.data.joint_pos_target
                _assert_finite(joint_target, "joint_pos_target")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
