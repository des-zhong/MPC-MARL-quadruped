"""Validate and visualize AS2 walking, dribbling, and shooting independently.

Each ability is run in a fresh simulator process so that policy state, robot
state, and ball state cannot leak from one validation scenario into another.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
ABILITIES = ("walk", "dribble", "shoot")
PHASE_BY_ABILITY = {"walk": "approach", "dribble": "dribble", "shoot": "shoot"}


def direction_cosine(actual_xy, target_xy):
    """Return alignment in [-1, 1], or 0 when either vector is stationary."""

    actual_xy = np.asarray(actual_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    denominator = np.linalg.norm(actual_xy) * np.linalg.norm(target_xy)
    if denominator <= 1.0e-8:
        return 0.0
    return float(np.clip(np.dot(actual_xy, target_xy) / denominator, -1.0, 1.0))


def summarize_walk(rows, args):
    measured = rows[min(args.warmup_steps, max(0, len(rows) - 1)) :]
    target = np.array([args.walk_x, args.walk_y, args.walk_yaw], dtype=np.float64)
    actual = np.array(
        [[row["robot_vx_body"], row["robot_vy_body"], row["robot_yaw_rate"]] for row in measured],
        dtype=np.float64,
    )
    errors = actual - target
    tracking_rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else float("inf")
    start = np.array([rows[0]["robot_x"], rows[0]["robot_y"]])
    end = np.array([rows[-1]["robot_x"], rows[-1]["robot_y"]])
    command_xy = target[:2]
    command_norm = np.linalg.norm(command_xy)
    progress = (
        float(np.dot(end - start, command_xy / command_norm))
        if command_norm > 1.0e-8
        else float(np.linalg.norm(end - start))
    )
    criteria = {
        "tracking_rmse": {
            "value": tracking_rmse,
            "operator": "<=",
            "threshold": args.walk_max_rmse,
            "passed": tracking_rmse <= args.walk_max_rmse,
        },
        "forward_progress_m": {
            "value": progress,
            "operator": ">=",
            "threshold": args.walk_min_progress,
            "passed": progress >= args.walk_min_progress,
        },
        "terminations": {
            "value": int(sum(row["done"] for row in rows)),
            "operator": "==",
            "threshold": 0,
            "passed": not any(row["done"] for row in rows),
        },
    }
    return criteria


def summarize_dribble(rows, args):
    measured = rows[min(args.warmup_steps, max(0, len(rows) - 1)) :]
    target = np.array([args.dribble_x, args.dribble_y], dtype=np.float64)
    target_speed = float(np.linalg.norm(target))
    ball_vel = np.array([[row["ball_vx"], row["ball_vy"]] for row in measured])
    speeds = np.linalg.norm(ball_vel, axis=1)
    speed_mae = float(np.mean(np.abs(speeds - target_speed))) if len(speeds) else float("inf")
    alignments = [direction_cosine(velocity, target) for velocity in ball_vel if np.linalg.norm(velocity) > 0.05]
    mean_alignment = float(np.mean(alignments)) if alignments else 0.0
    control_fraction = (
        float(np.mean([row["robot_ball_dist"] <= args.dribble_control_distance for row in measured]))
        if measured
        else 0.0
    )
    start = np.array([rows[0]["ball_x"], rows[0]["ball_y"]])
    end = np.array([rows[-1]["ball_x"], rows[-1]["ball_y"]])
    travel = float(np.linalg.norm(end - start))
    criteria = {
        "ball_speed_mae": {
            "value": speed_mae,
            "operator": "<=",
            "threshold": args.dribble_max_speed_mae,
            "passed": speed_mae <= args.dribble_max_speed_mae,
        },
        "mean_direction_alignment": {
            "value": mean_alignment,
            "operator": ">=",
            "threshold": args.dribble_min_alignment,
            "passed": mean_alignment >= args.dribble_min_alignment,
        },
        "control_fraction": {
            "value": control_fraction,
            "operator": ">=",
            "threshold": args.dribble_min_control_fraction,
            "passed": control_fraction >= args.dribble_min_control_fraction,
        },
        "ball_travel_m": {
            "value": travel,
            "operator": ">=",
            "threshold": args.dribble_min_travel,
            "passed": travel >= args.dribble_min_travel,
        },
        "terminations": {
            "value": int(sum(row["done"] for row in rows)),
            "operator": "==",
            "threshold": 0,
            "passed": not any(row["done"] for row in rows),
        },
    }
    return criteria


def summarize_shoot(rows, args):
    target = np.array([args.shoot_x, args.shoot_y], dtype=np.float64)
    velocities = np.array([[row["ball_vx"], row["ball_vy"]] for row in rows])
    speeds = np.linalg.norm(velocities, axis=1)
    peak_idx = int(np.argmax(speeds))
    peak_speed = float(speeds[peak_idx])
    peak_alignment = direction_cosine(velocities[peak_idx], target)
    start = np.array([rows[0]["ball_x"], rows[0]["ball_y"]])
    end = np.array([rows[-1]["ball_x"], rows[-1]["ball_y"]])
    target_norm = np.linalg.norm(target)
    launch_distance = (
        float(np.dot(end - start, target / target_norm))
        if target_norm > 1.0e-8
        else float(np.linalg.norm(end - start))
    )
    criteria = {
        "peak_ball_speed": {
            "value": peak_speed,
            "operator": ">=",
            "threshold": args.shoot_min_speed,
            "passed": peak_speed >= args.shoot_min_speed,
        },
        "launch_alignment": {
            "value": peak_alignment,
            "operator": ">=",
            "threshold": args.shoot_min_alignment,
            "passed": peak_alignment >= args.shoot_min_alignment,
        },
        "launch_distance_m": {
            "value": launch_distance,
            "operator": ">=",
            "threshold": args.shoot_min_distance,
            "passed": launch_distance >= args.shoot_min_distance,
        },
        "terminations": {
            "value": int(sum(row["done"] for row in rows)),
            "operator": "==",
            "threshold": 0,
            "passed": not any(row["done"] for row in rows),
        },
    }
    return criteria


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, ability, rows, command_xy):
    from matplotlib import pyplot as plt

    times = np.array([row["time_s"] for row in rows])
    robot_xy = np.array([[row["robot_x"], row["robot_y"]] for row in rows])
    ball_xy = np.array([[row["ball_x"], row["ball_y"]] for row in rows])
    robot_speed = np.array(
        [np.hypot(row["robot_vx_body"], row["robot_vy_body"]) for row in rows]
    )
    ball_speed = np.array([np.hypot(row["ball_vx"], row["ball_vy"]) for row in rows])
    distances = np.array([row["robot_ball_dist"] for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(robot_xy[:, 0], robot_xy[:, 1], label="robot", color="tab:blue")
    if ability != "walk":
        axes[0, 0].plot(ball_xy[:, 0], ball_xy[:, 1], label="ball", color="tab:orange")
    axes[0, 0].scatter(robot_xy[0, 0], robot_xy[0, 1], marker="o", color="green", label="start")
    axes[0, 0].set_title("Top-down trajectory")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend()

    axes[0, 1].plot(times, robot_speed, label="robot")
    if ability != "walk":
        axes[0, 1].plot(times, ball_speed, label="ball")
    axes[0, 1].axhline(np.linalg.norm(command_xy), color="black", linestyle="--", label="command")
    axes[0, 1].set_title("Planar speed")
    axes[0, 1].set_ylabel("m/s")
    axes[0, 1].legend()

    axes[1, 0].plot(times, distances, color="tab:orange")
    axes[1, 0].set_title("Robot-ball distance")
    axes[1, 0].set_ylabel("m")
    axes[1, 0].set_xlabel("time (s)")

    axes[1, 1].plot(times, [row["reward"] for row in rows], color="tab:green")
    axes[1, 1].set_title("Environment reward")
    axes[1, 1].set_xlabel("time (s)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    fig.suptitle(f"{ability.capitalize()} ability validation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _policy_record(playback, args, phase):
    """Load only the policy under test when the source is unambiguous."""

    checkpoint = getattr(args, f"{phase}_checkpoint") or args.checkpoint
    explicit_local = getattr(args, f"{phase}_local") or getattr(args, f"{phase}_run_dir")
    explicit_wandb = getattr(args, f"{phase}_wandb_run")
    if args.skill_policy_source == "wandb" or explicit_wandb:
        return playback.load_wandb_phase_policy(args, phase, checkpoint)
    if (
        args.skill_policy_source == "local"
        or explicit_local
        or any((args.walk_policy_dir, args.dribble_policy_dir, args.shoot_policy_dir))
        or (
            not args.wandb_run
            and not args.local
            and args.run_dir == playback.DEFAULT_RUN_DIR
            and args.body is None
            and args.adaptation_module is None
        )
    ):
        return playback.load_local_phase_policy(args, phase, checkpoint)
    # Legacy single-policy/direct-file mode is intentionally retained.
    return playback.load_phase_policies(args)[phase]


def run_single(args):
    # Isaac Gym must be imported before torch; the shared playback module
    # already enforces that ordering. Keeping it out of the "all" parent also
    # guarantees a clean simulator for each child.
    from scripts import play_walk_dribble_shoot as playback
    from scripts.playback_utils import get_raw_env, get_sensor_slice, patch_obs_command, set_walking_command
    import imageio
    import torch

    phase = PHASE_BY_ABILITY[args.ability]
    if args.ball_x is None:
        args.ball_x = 3.0 if args.ability == "walk" else args.ball_distance
    policy_record = _policy_record(playback, args, phase)
    metadata_config = policy_record.get("policy_metadata", {}).get("config_path")
    config_path = Path(args.config).expanduser().resolve() if args.config else metadata_config
    if config_path is None:
        config_path = playback.find_config_path(args, policy_record["body_path"].parent)

    env = playback.make_env(
        args,
        config_path,
        raw_action_clip=policy_record["action_clip"],
    )
    raw_env = get_raw_env(env)
    command_slice = get_sensor_slice(raw_env, "RCSensor")
    object_slice = get_sensor_slice(raw_env, "ObjectSensor")
    obs = env.reset()

    if args.ability == "walk":
        command = np.array([args.walk_x, args.walk_y, args.walk_yaw], dtype=np.float32)
    elif args.ability == "dribble":
        command = np.array([args.dribble_x, args.dribble_y, args.dribble_yaw], dtype=np.float32)
    else:
        command = np.array([args.shoot_x, args.shoot_y, 0.0], dtype=np.float32)

    ability_dir = Path(args.output_dir) / args.ability
    ability_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        fps = args.fps or int(round(1.0 / raw_env.dt))
        writer = imageio.get_writer(str(ability_dir / "rollout.mp4"), fps=fps)

    rows = []
    steps = args.steps or getattr(args, f"{args.ability}_steps")
    try:
        for step in range(steps):
            if args.ability == "walk":
                set_walking_command(raw_env, command, args)
            else:
                playback.set_ball_command(raw_env, command, args)
            patch_obs_command(raw_env, obs, command_slice)
            policy_obs = playback.adapt_obs_for_policy(obs, raw_env, policy_record, object_slice)
            with torch.no_grad():
                action = policy_record["policy"](policy_obs).to(raw_env.device)
                action = playback.clip_policy_action(action, policy_record)
            obs, reward, done, _ = env.step(action)

            robot_xy = raw_env.base_pos[0, :2].detach().cpu().numpy()
            ball_xy = raw_env.object_pos_world_frame[0, :2].detach().cpu().numpy()
            ball_vel = raw_env.object_lin_vel[0, :2].detach().cpu().numpy()
            base_vel = raw_env.base_lin_vel[0, :2].detach().cpu().numpy()
            yaw_rate = float(raw_env.base_ang_vel[0, 2].item())
            rows.append(
                {
                    "step": step,
                    "time_s": float(step * raw_env.dt),
                    "ability": args.ability,
                    "command_x": float(command[0]),
                    "command_y": float(command[1]),
                    "command_yaw": float(command[2]),
                    "robot_x": float(robot_xy[0]),
                    "robot_y": float(robot_xy[1]),
                    "robot_vx_body": float(base_vel[0]),
                    "robot_vy_body": float(base_vel[1]),
                    "robot_yaw_rate": yaw_rate,
                    "ball_x": float(ball_xy[0]),
                    "ball_y": float(ball_xy[1]),
                    "ball_vx": float(ball_vel[0]),
                    "ball_vy": float(ball_vel[1]),
                    "robot_ball_dist": float(np.linalg.norm(ball_xy - robot_xy)),
                    "reward": float(reward[0].item()),
                    "action_norm": float(torch.norm(action[0]).item()),
                    "done": int(done[0].item()),
                }
            )
            if writer is not None:
                writer.append_data(env.render(mode="rgb_array"))
            if done[0].item():
                break
    finally:
        if writer is not None:
            writer.close()

    if not rows:
        raise RuntimeError("The simulator produced no validation samples.")
    if args.ability == "walk":
        criteria = summarize_walk(rows, args)
    elif args.ability == "dribble":
        criteria = summarize_dribble(rows, args)
    else:
        criteria = summarize_shoot(rows, args)
    passed = all(item["passed"] for item in criteria.values())
    summary = {
        "ability": args.ability,
        "passed": passed,
        "samples": len(rows),
        "duration_s": rows[-1]["time_s"] + float(raw_env.dt),
        "policy_source": policy_record["source"],
        "policy_metadata": policy_record.get("policy_metadata", {}),
        "criteria": criteria,
        "artifacts": {
            "csv": str(ability_dir / "metrics.csv"),
            "plot": str(ability_dir / "metrics.png"),
            "video": None if args.no_video else str(ability_dir / "rollout.mp4"),
        },
    }
    write_csv(ability_dir / "metrics.csv", rows)
    save_plot(ability_dir / "metrics.png", args.ability, rows, command[:2])
    with (ability_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    print(f"\n{args.ability.upper()}: {'PASS' if passed else 'FAIL'}")
    for name, result in criteria.items():
        print(
            f"  {name}: {result['value']:.4f} "
            f"{result['operator']} {result['threshold']} "
            f"[{'pass' if result['passed'] else 'FAIL'}]"
        )
    print(f"  artifacts: {ability_dir}")
    return 0 if passed or not args.fail_on_threshold else 1


def _without_ability(argv):
    result = []
    skip_next = False
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == "--ability":
            skip_next = index + 1 < len(argv)
            continue
        if item.startswith("--ability="):
            continue
        result.append(item)
    return result


def run_all(args, argv):
    child_args = _without_ability(argv)
    exit_code = 0
    for ability in ABILITIES:
        summary_path = Path(args.output_dir) / ability / "summary.json"
        if summary_path.exists():
            summary_path.unlink()
        command = [sys.executable, str(Path(__file__).resolve()), "--ability", ability] + child_args
        result = subprocess.run(command, cwd=str(ROOT_DIR), check=False)
        exit_code = max(exit_code, result.returncode)

    summaries = {}
    for ability in ABILITIES:
        summary_path = Path(args.output_dir) / ability / "summary.json"
        if summary_path.exists():
            with summary_path.open() as file:
                summaries[ability] = json.load(file)
    combined = {
        "passed": len(summaries) == len(ABILITIES)
        and all(summary["passed"] for summary in summaries.values()),
        "abilities": summaries,
    }
    output_path = Path(args.output_dir) / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(combined, file, indent=2)
    print(f"\nALL ABILITIES: {'PASS' if combined['passed'] else 'FAIL'}")
    print(f"Combined summary: {output_path}")
    return exit_code


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate and visualize AS2 walk, dribble, and shoot policies in isolated scenarios."
    )
    parser.add_argument("--ability", choices=("all",) + ABILITIES, default="all")
    parser.add_argument("--output-dir", default="outputs/ability_validation")
    parser.add_argument("--steps", type=int, default=None, help="Override the selected ability's step count.")
    parser.add_argument("--walk-steps", type=int, default=250)
    parser.add_argument("--dribble-steps", type=int, default=350)
    parser.add_argument("--shoot-steps", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fps", type=int, default=None)

    parser.add_argument("--walk-x", type=float, default=0.6)
    parser.add_argument("--walk-y", type=float, default=0.0)
    parser.add_argument("--walk-yaw", type=float, default=0.0)
    parser.add_argument("--dribble-speed", type=float, default=0.9, help=argparse.SUPPRESS)
    parser.add_argument("--dribble-angle-deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dribble-x", type=float, default=0.9)
    parser.add_argument("--dribble-y", type=float, default=0.1)
    parser.add_argument("--dribble-yaw", type=float, default=0.3)
    parser.add_argument("--shoot-speed", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--shoot-angle-deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shoot-x", type=float, default=3.0)
    parser.add_argument("--shoot-y", type=float, default=0.0)
    parser.add_argument("--ball-distance", type=float, default=0.55)
    parser.add_argument("--ball-x", type=float, default=0.5)
    parser.add_argument("--ball-y", type=float, default=0.5)
    parser.add_argument("--ball-z", type=float, default=0.1)

    parser.add_argument("--walk-max-rmse", type=float, default=0.45)
    parser.add_argument("--walk-min-progress", type=float, default=1.0)
    parser.add_argument("--dribble-max-speed-mae", type=float, default=0.75)
    parser.add_argument("--dribble-min-alignment", type=float, default=0.6)
    parser.add_argument("--dribble-control-distance", type=float, default=1.0)
    parser.add_argument("--dribble-min-control-fraction", type=float, default=0.65)
    parser.add_argument("--dribble-min-travel", type=float, default=0.75)
    parser.add_argument("--shoot-min-speed", type=float, default=1.2)
    parser.add_argument("--shoot-min-alignment", type=float, default=0.6)
    parser.add_argument("--shoot-min-distance", type=float, default=0.7)

    parser.add_argument("--skill-policy-source", choices=("local", "wandb"), default=None)
    parser.add_argument("--walk-policy-dir", default=None)
    parser.add_argument("--dribble-policy-dir", default=None)
    parser.add_argument("--shoot-policy-dir", default=None)
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--approach-wandb-run", default=None)
    parser.add_argument("--dribble-wandb-run", default=None)
    parser.add_argument("--shoot-wandb-run", default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--run-dir", default="runs/as2")
    parser.add_argument("--approach-local", action="store_true")
    parser.add_argument("--dribble-local", action="store_true")
    parser.add_argument("--shoot-local", action="store_true")
    parser.add_argument("--approach-run-dir", default=None)
    parser.add_argument("--dribble-run-dir", default=None)
    parser.add_argument("--shoot-run-dir", default=None)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--approach-checkpoint", default='latest')
    parser.add_argument("--dribble-checkpoint", default='latest')
    parser.add_argument("--shoot-checkpoint", default='latest')
    parser.add_argument("--body", default=None)
    parser.add_argument("--adaptation-module", default=None)
    parser.add_argument("--pt", default=None)
    parser.add_argument("--policy-type", choices=("ball", "walking"), default="ball")
    parser.add_argument("--approach-policy-type", choices=("ball", "walking"), default="walking")
    parser.add_argument("--dribble-policy-type", choices=("ball", "walking"), default="ball")
    parser.add_argument("--shoot-policy-type", choices=("ball", "walking"), default="ball")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episode-length", type=float, default=60.0)

    # Gait fields consumed by the shared command/configuration helpers.
    parser.add_argument("--gait", default="trotting")
    parser.add_argument("--body-height", type=float, default=0.0)
    parser.add_argument("--step-frequency", type=float, default=3.0)
    parser.add_argument("--gait-duration", type=float, default=0.5)
    parser.add_argument("--footswing-height", type=float, default=0.09)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--stance-width", type=float, default=0.05)
    parser.add_argument("--stance-length", type=float, default=0.05)
    parser.add_argument("--aux-reward-coef", type=float, default=0.005)
    parser.add_argument("--approach-speed", type=float, default=0.45)
    parser.add_argument("--approach-max-yaw", type=float, default=1.0)
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be positive.")
    if args.ability == "all":
        return run_all(args, argv)
    return run_single(args)


if __name__ == "__main__":
    status = main()
    # Isaac Gym's legacy native runtime can segfault while Python tears down
    # CUDA/PhysX objects (including after gym.destroy_sim). All output files are
    # explicitly closed above, so terminate at the process boundary instead.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
