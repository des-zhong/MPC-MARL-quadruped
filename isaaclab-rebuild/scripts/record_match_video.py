"""Record a short four-AS2 football match from the Isaac Lab viewport.

The task is the same four-robot match/self-play environment used by the
contract smoke tests.  The recorder deliberately uses the environment's
``rgb_array`` viewport so the resulting MP4 exercises the normal Isaac Lab
capture path while retaining explicit frame validation and progress output.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


REBUILD_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REBUILD_ROOT / "source" / "dribblebot_isaaclab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120, help="Number of coordinator steps to record.")
parser.add_argument(
    "--output",
    type=Path,
    default=REBUILD_ROOT / "outputs" / "videos" / "match-selfplay-scripted.mp4",
)
parser.add_argument("--video_width", type=int, default=640)
parser.add_argument("--video_height", type=int, default=360)
parser.add_argument("--fps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# This entry point exists only to record RGB, so cameras are mandatory even if
# the caller omitted the flag.  Leave the experience empty unless explicitly
# overridden so AppLauncher selects its rendering experience for headless/UI.
args_cli.enable_cameras = True
if not args_cli.experience and os.environ.get("DRIBBLEBOT_EXPERIENCE"):
    args_cli.experience = os.environ["DRIBBLEBOT_EXPERIENCE"]

if (
    args_cli.num_envs <= 0
    or args_cli.steps <= 0
    or args_cli.video_width <= 0
    or args_cli.video_height <= 0
    or args_cli.fps <= 0
):
    raise ValueError("num_envs, steps, video_width, video_height, and fps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import dribblebot_isaaclab  # noqa: E402, F401
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _action_from_skill(
    *,
    skill_id: int,
    command_xy: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build one hybrid coordinator action from a skill and command."""

    action = torch.zeros(command_xy.shape[0], 4, device=device)
    action[:, 0] = int(skill_id)
    # The action term applies tanh to these values. Clamp the input
    # to avoid accidentally requesting commands outside the configured range.
    action[:, 1:3] = command_xy.clamp(-2.0, 2.0)
    return action


def _body_frame_direction(manager_env, robot_name: str, ball_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distance and normalized ball direction in one robot's body frame."""

    robot = manager_env.scene[robot_name]
    delta_w = ball_xy - robot.data.root_pos_w[:, :2]
    delta_w3 = torch.cat((delta_w, torch.zeros_like(delta_w[:, :1])), dim=-1)
    delta_b = quat_apply_inverse(robot.data.root_quat_w, delta_w3)[:, :2]
    distance = torch.linalg.vector_norm(delta_b, dim=-1)
    return distance, delta_b / distance.clamp_min(1.0e-6).unsqueeze(-1)


def _make_team_action(manager_env, device: torch.device) -> torch.Tensor:
    """Approach the ball, then request a shot when a robot is strikeable."""

    ball_xy = manager_env.scene["ball"].data.root_pos_w[:, :2]
    actions = []
    for robot_name in ("robot_0", "robot_1"):
        distance, direction = _body_frame_direction(manager_env, robot_name, ball_xy)
        # The walk policy gets a body-frame direction.  Once within shooting
        # distance, request a +x shot; the action term still applies its
        # geometric strikeability fallback if the exact setup is not valid.
        walk_command = 1.25 * direction
        shoot = (distance <= 0.70) & (direction[:, 0] >= -0.10) & (direction[:, 1].abs() <= 0.40)
        skill_action = _action_from_skill(skill_id=0, command_xy=walk_command, device=device)
        if bool(shoot.any()):
            shoot_action = _action_from_skill(
                skill_id=2,
                command_xy=torch.cat((torch.ones_like(direction[:, :1]), torch.zeros_like(direction[:, :1])), dim=-1),
                device=device,
            )
            skill_action[shoot] = shoot_action[shoot]
        actions.append(skill_action)
    return torch.stack(actions, dim=1).reshape(-1, 4)


def _opponent_policy(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    """A deterministic opponent that walks toward the ball in its body frame."""

    history = observation["obs_history"]
    # Team-1 robots face -x at reset, so +x is the forward body-frame command.
    action = torch.zeros(history.shape[0], 4, device=history.device)
    action[:, 1] = 1.25
    return action


def _reset_for_headless_viewport(match_env) -> None:
    """Reset the match without Isaac Sim's blocking forward/viewport sync.

    Isaac Sim 5.1's base-python experience can block in ``sim.forward()`` when
    no X display is present but a viewport is enabled.  The manager reset has
    already written every pose, velocity, action-history, command and recorder
    buffer before that call.  One non-rendering physics tick provides the same
    initialized tensor state without waiting on the viewport.
    """

    raw_env = match_env.env.env
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


def main() -> Path:
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)
    cfg.seed = 7
    # In the Isaac Sim 5.1 pip runtime, the global texture-loading flag may
    # remain true indefinitely for the base viewport experience even though
    # all local AS2/ball assets are already available.  Do bounded frame
    # validation below instead of blocking forever in ManagerBasedEnv.reset.
    cfg.wait_for_textures = False
    cfg.scene.match_camera = TiledCameraCfg(
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

    print("[VIDEO] creating match environment", flush=True)
    base_env = gym.make(args_cli.task, cfg=cfg)
    print("[VIDEO] match environment ready", flush=True)
    match_env = base_env
    while not hasattr(match_env, "_history") and hasattr(match_env, "env"):
        match_env = match_env.env
    if not hasattr(match_env, "_history"):
        raise RuntimeError("Could not locate MatchSelfPlayWrapper in the Gym wrapper chain")
    match_env.set_opponent_callable(_opponent_policy)
    manager_env = base_env.unwrapped
    camera = manager_env.scene["match_camera"]
    origins = manager_env.scene.env_origins
    camera.set_world_poses_from_view(
        origins + origins.new_tensor((7.5, -8.5, 7.0)),
        origins + origins.new_tensor((0.0, 0.0, 0.0)),
    )
    writer = None
    try:
        _reset_for_headless_viewport(match_env)
        print("[VIDEO] warming up viewport", flush=True)
        first_frame = None
        for warmup in range(10):
            manager_env.sim.render()
            manager_env.scene.update(dt=manager_env.physics_dt)
            frame = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            if frame.shape != (args_cli.video_height, args_cli.video_width, 3):
                raise RuntimeError(
                    f"viewport returned shape {frame.shape}, expected "
                    f"({args_cli.video_height}, {args_cli.video_width}, 3)"
                )
            if frame.size > 0 and float(frame.std()) > 1.0:
                first_frame = frame
                print(
                    f"[VIDEO] viewport ready after {warmup + 1} render(s): "
                    f"mean={frame.mean():.2f} std={frame.std():.2f}",
                    flush=True,
                )
                break
        if first_frame is None:
            raise RuntimeError("viewport did not produce a non-empty RGB frame after 10 renders")

        writer = imageio.get_writer(
            str(args_cli.output),
            fps=args_cli.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        writer.append_data(first_frame)
        with torch.inference_mode():
            for step in range(args_cli.steps):
                action = _make_team_action(manager_env, manager_env.device)
                _, reward, _, _, _ = base_env.step(action)
                if not torch.isfinite(reward).all():
                    raise RuntimeError(f"non-finite reward at coordinator step {step}")
                frame = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                if frame.shape != first_frame.shape or float(frame.std()) <= 1.0:
                    raise RuntimeError(
                        f"invalid RGB frame at coordinator step {step}: shape={frame.shape}, std={frame.std():.3f}"
                    )
                writer.append_data(frame)
                if (step + 1) % 5 == 0 or step + 1 == args_cli.steps:
                    print(
                        f"[VIDEO] captured {step + 1}/{args_cli.steps} coordinator frames "
                        f"(mean={frame.mean():.2f}, std={frame.std():.2f})",
                        flush=True,
                    )
    finally:
        if writer is not None:
            writer.close()
        base_env.close()

    if not args_cli.output.is_file() or args_cli.output.stat().st_size == 0:
        raise FileNotFoundError(f"Video writer did not produce a non-empty MP4: {args_cli.output}")
    return args_cli.output


if __name__ == "__main__":
    exit_code = 1
    try:
        output = main()
        print(f"[VIDEO] wrote {output}", flush=True)
        exit_code = 0
    except BaseException:
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
