"""Train the manager-based four-AS2 coordinator with a frozen opponent."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from time import perf_counter

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument(
    "--num_steps_per_env",
    type=int,
    default=None,
    help="Override rollout length (use 4 for a quick end-to-end smoke test).",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="")
parser.add_argument(
    "--resume_checkpoint",
    type=Path,
    default=None,
    help="Resume actor, optimizer, and iteration state from an RSL-RL model_*.pt checkpoint.",
)
parser.add_argument("--opponent_checkpoint_root", default=None)
parser.add_argument("--opponent_policy_device", default="cpu")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401
from dribblebot_isaaclab.tasks.manager_based.football.rsl import (  # noqa: E402
    MatchSelfPlayRslRlVecEnvWrapper,
    install_opponent_snapshot_schedule,
)
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


_START_TIME = perf_counter()


def _stage(message: str) -> None:
    """Print an unbuffered startup marker for slow Isaac Sim phases."""

    print(f"[TRAIN][{perf_counter() - _START_TIME:8.1f}s] {message}", flush=True)


def main() -> None:
    _stage("解析环境和 PPO 配置")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.num_steps_per_env is not None:
        if args_cli.num_steps_per_env <= 0:
            raise ValueError("--num_steps_per_env must be positive")
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.opponent_checkpoint_root:
        env_cfg.opponent_checkpoint_root = args_cli.opponent_checkpoint_root
        env_cfg.opponent_policy_device = args_cli.opponent_policy_device

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        stamp = f"{stamp}_{agent_cfg.run_name}"
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs" / "rsl_rl" / agent_cfg.experiment_name / stamp
    log_dir.mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(log_dir)

    _stage(f"创建四机环境（physical matches={env_cfg.scene.num_envs}）")
    env = gym.make(args_cli.task, cfg=env_cfg)
    _stage("环境创建完成；执行一次初始 reset")
    env.reset()
    _stage("初始 reset 完成；初始化 self-play RSL-RL wrapper")
    _stage(f"gym wrapper 类型={type(env).__name__}, inner={type(env.env).__name__}")
    _stage(f"gym wrapper num_envs={env.get_wrapper_attr('num_envs')}")
    # Gymnasium's outer OrderEnforcing proxy recursively resolves custom
    # ``num_envs`` attributes.  Pass the actual MatchSelfPlayWrapper to the
    # RSL-RL adapter to keep attribute access deterministic.
    env = MatchSelfPlayRslRlVecEnvWrapper(env.env, clip_actions=agent_cfg.clip_actions, auto_reset=False)
    _stage(
        f"wrapper 初始化完成（training samples={env.num_envs}, "
        f"rollout steps={agent_cfg.num_steps_per_env}）"
    )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    _stage("PPO runner 初始化完成")
    if args_cli.resume_checkpoint is not None:
        checkpoint = args_cli.resume_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        runner.load(str(checkpoint), load_optimizer=True, map_location=agent_cfg.device)
        _stage(f"已恢复 checkpoint={checkpoint.name}, iteration={runner.current_learning_iteration}")
    _stage("安装 opponent snapshot schedule")
    runner.add_git_repo_to_log(__file__)
    # Install an archived opponent when configured, then refresh snapshots at
    # the configured optimizer-update cadence.
    install_opponent_snapshot_schedule(
        runner,
        env,
        interval=int(getattr(env.env, "opponent_snapshot_interval", 500)),
        policy_device=args_cli.opponent_policy_device,
        # Fresh jobs use a zero opponent to avoid a synchronous actor deepcopy
        # during Isaac Sim startup.  A resumed job snapshots its restored actor
        # immediately so it does not silently switch back to a zero opponent.
        initial_snapshot=args_cli.resume_checkpoint is not None,
    )
    _stage("对手快照 schedule 已安装")
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    try:
        remaining_iterations = max(int(agent_cfg.max_iterations) - int(runner.current_learning_iteration), 0)
        _stage(
            f"开始训练 remaining={remaining_iterations}, "
            f"target_iteration={agent_cfg.max_iterations}"
        )
        runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True)
        _stage("训练完成")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
