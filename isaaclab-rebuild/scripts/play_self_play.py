"""Visualize a trained four-AS2 self-play checkpoint for several episodes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher


REBUILD_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REBUILD_ROOT / "source" / "dribblebot_isaaclab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="RSL-RL model_*.pt; default selects the most recently modified checkpoint.",
)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1, help="Physical matches shown in parallel.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--frame_delay", type=float, default=0.03, help="Wall-time delay after each 200 ms action.")
parser.add_argument("--max_steps", type=int, default=0, help="Safety limit; zero derives it from episode length.")
parser.add_argument(
    "--opponent",
    choices=("self", "zero", "rule_based"),
    default="self",
    help="Use the same checkpoint, a stationary opponent, or the deterministic soccer controller.",
)
parser.add_argument("--opponent_policy_device", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")
# This entry point is intentionally interactive. Closing the Isaac Sim window
# exits the episode loop cleanly.
args_cli.headless = False

if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
    raise ValueError("--episodes and --num_envs must be positive")
if args_cli.frame_delay < 0.0 or args_cli.max_steps < 0:
    raise ValueError("--frame_delay and --max_steps cannot be negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import dribblebot_isaaclab  # noqa: E402, F401
from dribblebot_isaaclab.tasks.manager_based.football.rsl import (  # noqa: E402
    FrozenRslRlOpponentPolicy,
    HybridOnPolicyRunner,
    MatchSelfPlayRslRlVecEnvWrapper,
)
from dribblebot_isaaclab.tasks.manager_based.football.rule_based_opponent import (  # noqa: E402
    RuleBasedOpponent,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402


def _latest_checkpoint() -> Path:
    root = REBUILD_ROOT / "logs" / "rsl_rl" / "dribblebot_as2_match_self_play"
    candidates = list(root.glob("*/model_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No model_*.pt found under {root}; pass --checkpoint /absolute/path/to/model.pt"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _resolve_checkpoint() -> Path:
    checkpoint = args_cli.checkpoint if args_cli.checkpoint is not None else _latest_checkpoint()
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def _match_wrapper(env):
    current = env
    while not hasattr(current, "_history") and hasattr(current, "env"):
        current = current.env
    if not hasattr(current, "_history"):
        raise RuntimeError("Could not locate MatchSelfPlayWrapper in the Gym wrapper chain")
    return current


def _zero_opponent(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    history = observation["obs_history"]
    return torch.zeros(history.shape[0], 4, device=history.device)


def main() -> None:
    checkpoint = _resolve_checkpoint()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.wait_for_textures = False
    env_cfg.viewer.eye = (7.5, -8.5, 7.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device

    print(f"[PLAY] checkpoint={checkpoint}", flush=True)
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")
    match_env = _match_wrapper(base_env)
    try:
        observation, _ = base_env.reset()
        if observation is None:
            raise RuntimeError("Initial environment reset returned no observation")
        vec_env = MatchSelfPlayRslRlVecEnvWrapper(
            match_env,
            clip_actions=agent_cfg.clip_actions,
            auto_reset=False,
        )
        runner = HybridOnPolicyRunner(
            vec_env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(checkpoint), load_optimizer=False, map_location=agent_cfg.device)
        actor = runner.get_inference_policy(device=agent_cfg.device)

        if args_cli.opponent == "self":
            opponent_device = args_cli.opponent_policy_device or args_cli.device
            match_env.set_opponent_callable(
                FrozenRslRlOpponentPolicy(runner.alg.policy, device=opponent_device)
            )
        elif args_cli.opponent == "zero":
            match_env.set_opponent_callable(_zero_opponent)
        else:
            match_env.set_opponent_action_provider(RuleBasedOpponent(match_env))

        raw_env = base_env.unwrapped
        num_matches = int(match_env.match_count)
        team_size = int(match_env.team_size)
        returns = torch.zeros(num_matches, device=raw_env.device)
        lengths = torch.zeros(num_matches, dtype=torch.long, device=raw_env.device)
        skill_counts = torch.zeros(num_matches, 3, dtype=torch.long, device=raw_env.device)
        completed = 0
        step = 0
        max_steps = args_cli.max_steps or (
            args_cli.episodes * int(match_env.max_episode_length) * 2
        )

        for _ in range(3):
            raw_env.sim.render()
            simulation_app.update()
        print(
            f"[PLAY] window ready: matches={num_matches}, target_episodes={args_cli.episodes}, "
            f"opponent={args_cli.opponent}",
            flush=True,
        )

        observations = vec_env.get_observations().to(agent_cfg.device)
        with torch.inference_mode():
            while simulation_app.is_running() and completed < args_cli.episodes and step < max_steps:
                actions = actor(observations)
                observations, rewards, _, extras = vec_env.step(actions)
                observations = observations.to(agent_cfg.device)
                match_rewards = rewards.reshape(num_matches, team_size).mean(dim=1)
                returns += match_rewards
                lengths += 1
                for robot_index in range(2 * team_size):
                    ids = raw_env.action_manager.get_term(f"skill_policy_{robot_index}").skill_ids
                    skill_counts += torch.nn.functional.one_hot(ids, num_classes=3).to(torch.long)

                match_done = torch.as_tensor(
                    extras.get("match_done", torch.zeros(num_matches, dtype=torch.bool)),
                    dtype=torch.bool,
                    device=raw_env.device,
                )
                for match_index in match_done.nonzero(as_tuple=False).flatten().tolist():
                    completed += 1
                    counts = skill_counts[match_index].detach().cpu().tolist()
                    print(
                        f"[PLAY] episode={completed}/{args_cli.episodes} match={match_index} "
                        f"length={int(lengths[match_index])} return={float(returns[match_index]):.3f} "
                        f"skills(walk/dribble/shoot)={counts}",
                        flush=True,
                    )
                    returns[match_index] = 0.0
                    lengths[match_index] = 0
                    skill_counts[match_index] = 0
                    if completed >= args_cli.episodes:
                        break

                raw_env.sim.render()
                simulation_app.update()
                if args_cli.frame_delay > 0.0:
                    time.sleep(args_cli.frame_delay)
                step += 1

        if completed < args_cli.episodes and simulation_app.is_running():
            raise RuntimeError(
                f"Reached max_steps={max_steps} after {completed}/{args_cli.episodes} episodes"
            )
        print(f"[PLAY] finished {completed} episode(s) in {step} coordinator steps", flush=True)
    finally:
        base_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
