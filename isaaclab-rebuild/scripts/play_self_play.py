"""Visualize a trained four-AS2 self-play checkpoint for several episodes."""

from __future__ import annotations

import argparse
import json
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
parser.add_argument(
    "--video_output",
    type=Path,
    default=None,
    help="Optional MP4 path. When set, capture the first physical match camera.",
)
parser.add_argument("--video_width", type=int, default=640)
parser.add_argument("--video_height", type=int, default=360)
parser.add_argument("--video_fps", type=int, default=5)
parser.add_argument(
    "--metrics_output",
    type=Path,
    default=None,
    help="Optional JSON file containing episode returns, lengths, and skill counts.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    if args_cli.video_output is not None:
        # Do not inherit the training-only base experience: AppLauncher must
        # select its camera-enabled headless experience. An explicit video
        # override remains available for custom Isaac Sim installations.
        configured_experience = os.environ.get("DRIBBLEBOT_VIDEO_EXPERIENCE")
        if configured_experience:
            args_cli.experience = configured_experience
    else:
        args_cli.experience = os.environ.get(
            "DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit"
        )
if args_cli.video_output is not None:
    if args_cli.video_width <= 0 or args_cli.video_height <= 0 or args_cli.video_fps <= 0:
        raise ValueError("video dimensions and fps must be positive")
    args_cli.enable_cameras = True

if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
    raise ValueError("--episodes and --num_envs must be positive")
if args_cli.frame_delay < 0.0 or args_cli.max_steps < 0:
    raise ValueError("--frame_delay and --max_steps cannot be negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
import dribblebot_isaaclab  # noqa: E402, F401
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
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


def _reset_without_view_forward(match_env) -> None:
    """Reset a camera-enabled match without blocking in ``sim.forward``.

    Isaac Sim's base-python experience can return from the app while waiting
    for a viewport during ``ManagerBasedRLEnv.reset`` when no X display is
    available.  A direct manager reset performs the same tensor/bookkeeping
    work and one non-rendering physics tick initializes the camera safely.
    """

    raw_env = match_env.unwrapped
    env_ids = torch.arange(raw_env.num_envs, dtype=torch.long, device=raw_env.device)
    raw_env.recorder_manager.record_pre_reset(env_ids)
    raw_env._reset_idx(env_ids)
    raw_env.scene.write_data_to_sim()
    raw_env.sim.step(render=False)
    raw_env.scene.update(dt=raw_env.physics_dt)
    raw_env.recorder_manager.record_post_reset(env_ids)
    raw_env.obs_buf = raw_env.observation_manager.compute(update_history=True)
    match_env._history.zero_()
    match_env._last_team_actions.zero_()
    match_env._update_observations(raw_env.obs_buf, reset_mask=None)


def main() -> None:
    checkpoint = _resolve_checkpoint()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        # Camera pose writes use the USD xform view. Disable Fabric only for
        # video eval so those writes remain synchronized with the renderer.
        use_fabric=args_cli.video_output is None,
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device
    env_cfg.wait_for_textures = False
    env_cfg.viewer.eye = (7.5, -8.5, 7.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    if args_cli.video_output is not None:
        env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
        # Use a dedicated tiled camera instead of the Kit viewport.  The
        # viewport's ``rgb_array`` path is not reliable in headless base
        # experiences (it may shut the app down before the first step), while
        # sensor RGB is deterministic and works both with and without X.
        env_cfg.scene.match_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/MatchCamera",
            update_period=0.0,
            height=args_cli.video_height,
            width=args_cli.video_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        )
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device

    print(f"[PLAY] checkpoint={checkpoint}", flush=True)
    # RGB frames are read from the explicit MatchCamera below.  Keep the
    # environment render mode unset so no viewport synchronization is needed.
    render_mode = None if args_cli.headless else "human"
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    match_env = _match_wrapper(base_env)
    raw_env = base_env.unwrapped
    camera = raw_env.scene["match_camera"] if args_cli.video_output is not None else None
    try:
        if args_cli.video_output is not None:
            _reset_without_view_forward(match_env)
        else:
            observation, _ = base_env.reset()
            if observation is None:
                raise RuntimeError("Initial environment reset returned no observation")
        if camera is not None:
            # Reset events restore the sensor offset, so apply the recording
            # view after reset and immediately before the first render.
            origins = raw_env.scene.env_origins
            camera.set_world_poses_from_view(
                origins + origins.new_tensor((7.5, -8.5, 7.0)),
                origins + origins.new_tensor((0.0, 0.0, 0.0)),
            )
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

        num_matches = int(match_env.match_count)
        team_size = int(match_env.team_size)
        returns = torch.zeros(num_matches, device=raw_env.device)
        lengths = torch.zeros(num_matches, dtype=torch.long, device=raw_env.device)
        skill_counts = torch.zeros(num_matches, 3, dtype=torch.long, device=raw_env.device)
        completed = 0
        episode_records = []
        step = 0
        max_steps = args_cli.max_steps or (
            args_cli.episodes * int(match_env.max_episode_length) * 2
        )
        video_writer = None
        if args_cli.video_output is not None:
            args_cli.video_output.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                str(args_cli.video_output),
                fps=args_cli.video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            )

        for _ in range(5):
            raw_env.sim.render()
            if camera is not None:
                raw_env.scene.update(dt=raw_env.physics_dt)
            if not args_cli.headless:
                simulation_app.update()
        print(
            f"[PLAY] window ready: matches={num_matches}, target_episodes={args_cli.episodes}, "
            f"opponent={args_cli.opponent}",
            flush=True,
        )

        observations = vec_env.get_observations().to(agent_cfg.device)
        with torch.inference_mode():
            while (
                (args_cli.headless or simulation_app.is_running())
                and completed < args_cli.episodes
                and step < max_steps
            ):
                if video_writer is not None:
                    raw_env.sim.render()
                    frame = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                    if frame.ndim != 3 or frame.shape[-1] != 3 or float(frame.std()) <= 1.0:
                        raise RuntimeError(f"invalid eval RGB frame: shape={frame.shape}, std={frame.std():.3f}")
                    video_writer.append_data(frame)
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
                    episode_records.append(
                        {
                            "episode": completed,
                            "match_index": match_index,
                            "length": int(lengths[match_index]),
                            "return": float(returns[match_index]),
                            "skill_counts": {
                                "walk": int(counts[0]),
                                "dribble": int(counts[1]),
                                "shoot": int(counts[2]),
                            },
                        }
                    )
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

                if video_writer is not None:
                    raw_env.sim.render()
                    raw_env.scene.update(dt=raw_env.physics_dt)
                if not args_cli.headless:
                    simulation_app.update()
                if args_cli.frame_delay > 0.0:
                    time.sleep(args_cli.frame_delay)
                step += 1

        if completed < args_cli.episodes and (args_cli.headless or simulation_app.is_running()):
            raise RuntimeError(
                f"Reached max_steps={max_steps} after {completed}/{args_cli.episodes} episodes"
            )
        print(f"[PLAY] finished {completed} episode(s) in {step} coordinator steps", flush=True)
        if video_writer is not None:
            video_writer.close()
            video_writer = None
            if not args_cli.video_output.is_file() or args_cli.video_output.stat().st_size == 0:
                raise RuntimeError(f"video writer produced no file: {args_cli.video_output}")
            print(f"[PLAY] video={args_cli.video_output}", flush=True)
        if args_cli.metrics_output is not None:
            args_cli.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            total_skills = {
                name: sum(row["skill_counts"][name] for row in episode_records)
                for name in ("walk", "dribble", "shoot")
            }
            report = {
                "checkpoint": str(checkpoint),
                "opponent": args_cli.opponent,
                "completed_episodes": completed,
                "mean_return": (
                    sum(row["return"] for row in episode_records) / completed if completed else 0.0
                ),
                "mean_length": (
                    sum(row["length"] for row in episode_records) / completed if completed else 0.0
                ),
                "skill_counts": total_skills,
                "episodes": episode_records,
            }
            args_cli.metrics_output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"[PLAY] metrics={args_cli.metrics_output}", flush=True)
    finally:
        if "video_writer" in locals() and video_writer is not None:
            video_writer.close()
        base_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
