"""Evaluate a Go1 high-level multi-robot soccer policy."""

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import isaacgym

assert isaacgym
import imageio
import numpy as np
import torch
from tqdm import trange

from dribblebot.envs.base.legged_robot_config import Cfg
from scripts.go1_scripts.play_walk_dribble_shoot import load_policy_record, resolve_wandb_policy_files
from scripts.go1_scripts.training_high_level import configure_high_level_cfg, load_skill_policies


SKILL_NAMES = ("walk", "dribble", "shoot")
TERMINAL_KEYS = (
    "high_level_goal",
    "high_level_ball_off_border",
    "high_level_obstacle_contact",
    "high_level_accidental_termination",
)


def set_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_domain_randomization():
    false_flags = [
        "randomize_rigids_after_start",
        "randomize_friction_indep",
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "randomize_gravity",
        "randomize_ground_friction",
        "randomize_motor_strength",
        "randomize_motor_offset",
        "randomize_ball_drag",
        "randomize_lag_timesteps",
        "push_robots",
    ]
    for name in false_flags:
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, False)
    if hasattr(Cfg.domain_rand, "lag_timesteps"):
        Cfg.domain_rand.lag_timesteps = 0


def configure_eval_cfg(args):
    configure_high_level_cfg(Cfg, args)
    Cfg.env.num_envs = args.num_envs
    Cfg.env.record_video = not args.no_video
    Cfg.env.randomize_match_init = not args.fixed_init
    Cfg.env.add_field_markers = not args.no_field_markers
    if args.camera_height is not None:
        Cfg.env.high_level_camera_height = args.camera_height
    if args.recording_fov is not None:
        Cfg.env.recording_horizontal_fov = args.recording_fov
    if not args.domain_rand:
        disable_domain_randomization()


def load_high_level_policy(args):
    body_path, adaptation_module_path, ac_weights_path, run_path = resolve_wandb_policy_files(
        args.high_level_wandb_run,
        args.high_level_checkpoint,
    )
    return load_policy_record(
        "high-level",
        body_path,
        adaptation_module_path,
        ac_weights_path,
        "high_level",
        args.policy_device,
        f"W&B {run_path}@{args.high_level_checkpoint}",
    )


def make_env(args, skill_policies):
    from dribblebot.envs.go1.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    from dribblebot.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper

    configure_eval_cfg(args)
    raw_env = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=args.headless, cfg=Cfg)
    env = HighLevelSkillWrapper(raw_env, skill_policies)
    return env, raw_env


def validate_high_level_obs_shape(policy_record, env):
    expected_dim = policy_record.get("expected_history_dim")
    if expected_dim is None or expected_dim == env.num_obs_history:
        return

    raise ValueError(
        f"High-level policy expects obs_history dim {expected_dim}, "
        f"but this eval env provides {env.num_obs_history} "
        f"({env.num_obs} obs x {env.history_length} history). "
        "Use matching --high-level-history/config settings, or retrain/load a checkpoint "
        "that matches the current high-level observation layout."
    )

def validate_low_level_skill_shapes(skill_policies, env):
    full_history_dim = env.low_level_obs_dim_full * env.low_level_history_length
    no_object_history_dim = (env.low_level_obs_dim_full - 3) * env.low_level_history_length

    valid_dims = {
        "ball": {full_history_dim},
        "walking": {full_history_dim, no_object_history_dim},
    }
    valid_dims["walk"] = valid_dims["walking"]
    valid_dims["dribble"] = valid_dims["ball"]
    valid_dims["shoot"] = valid_dims["ball"]

    errors = []
    for skill_name, policy_record in skill_policies.items():
        expected_dim = policy_record.get("expected_history_dim")
        if expected_dim is None:
            continue

        policy_type = policy_record.get("policy_type", skill_name)
        allowed = valid_dims.get(policy_type, valid_dims.get(skill_name, {full_history_dim, no_object_history_dim}))
        if expected_dim in allowed:
            continue

        errors.append(
            f"{skill_name} ({policy_record.get('source', 'unknown source')}) expects obs_history dim {expected_dim}, "
            f"but a low-level {skill_name} policy must expect one of {sorted(allowed)}. "
            f"Full low-level history is {full_history_dim}; walking/no-object history is {no_object_history_dim}."
        )

    if errors:
        raise ValueError(
            "Invalid low-level skill checkpoint(s):\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\nThis usually means a high-level coordinator run was passed as --walk-wandb-run, "
            "--dribble-wandb-run, or --shoot-wandb-run. Put the coordinator run only in "
            "--high-level-wandb-run."
        )


def skill_name(skill_id):
    skill_id = int(skill_id)
    if 0 <= skill_id < len(SKILL_NAMES):
        return SKILL_NAMES[skill_id]
    return f"unknown_{skill_id}"


def info_array(info, key, shape, default=0):
    value = info.get(key)
    if value is None:
        return np.full(shape, default)
    array = np.asarray(value)
    if array.shape == shape:
        return array
    if len(shape) > 0 and shape[0] == 1 and array.shape[:1] != ():
        sliced = array[:1]
        if sliced.shape == shape:
            return sliced
    return np.reshape(array, shape)


def collect_state(raw_env):
    roots = raw_env.root_states[raw_env.robot_actor_idxs_all.reshape(-1)].view(
        raw_env.num_envs,
        raw_env.num_robots,
        13,
    )
    robot_xy = roots[:, :, :2] - raw_env.env_origins[:, None, :2]
    robot_vel = roots[:, :, 7:9]
    ball_xy = raw_env.object_pos_world_frame[:, :2] - raw_env.env_origins[:, :2]
    ball_vel = raw_env.object_lin_vel[:, :2]
    robot_ball_dist = torch.norm(roots[:, :, :2] - raw_env.object_pos_world_frame[:, None, :2], dim=-1)
    obstacle_xy = None
    if getattr(raw_env, "num_static_opponents", 0) > 0:
        obstacle_states = raw_env.root_states[raw_env.static_opponent_actor_idxs.reshape(-1)].view(
            raw_env.num_envs,
            raw_env.num_static_opponents,
            13,
        )
        obstacle_xy = obstacle_states[:, :, :2] - raw_env.env_origins[:, None, :2]

    return {
        "robot_xy": robot_xy.detach().cpu().numpy(),
        "robot_vel": robot_vel.detach().cpu().numpy(),
        "ball_xy": ball_xy.detach().cpu().numpy(),
        "ball_vel": ball_vel.detach().cpu().numpy(),
        "robot_ball_dist": robot_ball_dist.detach().cpu().numpy(),
        "obstacle_xy": None if obstacle_xy is None else obstacle_xy.detach().cpu().numpy(),
    }


def row_from_step(step, high_level_dt, state, action, reward, done, info):
    num_robots = state["robot_xy"].shape[1]
    requested = info_array(info, "high_level_requested_skill_ids", (1, num_robots), 0).astype(np.int64)
    executed = info_array(info, "high_level_skill_ids", (1, num_robots), 0).astype(np.int64)
    invalid = info_array(info, "high_level_invalid_skill_mask", (1, num_robots), False).astype(bool)
    commands = info_array(info, "high_level_commands", (1, num_robots, 3), 0.0).astype(np.float32)
    action_np = action.detach().cpu().numpy()
    reward_np = reward.detach().cpu().numpy()
    done_np = done.detach().cpu().numpy().astype(bool)

    ball_xy = state["ball_xy"][0]
    ball_vel = state["ball_vel"][0]
    robot_xy = state["robot_xy"][0]
    robot_vel = state["robot_vel"][0]
    robot_ball_dist = state["robot_ball_dist"][0]
    obstacle_xy = state["obstacle_xy"]
    row = {
        "step": step,
        "time_s": step * high_level_dt,
        "reward": float(reward_np[0]),
        "done": int(done_np[0]),
        "action_norm": float(np.linalg.norm(action_np[0])),
        "ball_x": float(ball_xy[0]),
        "ball_y": float(ball_xy[1]),
        "ball_vx": float(ball_vel[0]),
        "ball_vy": float(ball_vel[1]),
        "ball_speed": float(np.linalg.norm(ball_vel)),
        "num_robots": num_robots,
    }

    missing_xy = np.array([np.nan, np.nan], dtype=np.float32)
    for robot_idx in range(num_robots):
        row[f"robot{robot_idx}_x"] = float(robot_xy[robot_idx, 0])
        row[f"robot{robot_idx}_y"] = float(robot_xy[robot_idx, 1])
        row[f"robot{robot_idx}_vx"] = float(robot_vel[robot_idx, 0])
        row[f"robot{robot_idx}_vy"] = float(robot_vel[robot_idx, 1])
        row[f"robot{robot_idx}_ball_dist"] = float(robot_ball_dist[robot_idx])
        row[f"robot{robot_idx}_requested_skill_id"] = int(requested[0, robot_idx])
        row[f"robot{robot_idx}_requested_skill"] = skill_name(requested[0, robot_idx])
        row[f"robot{robot_idx}_executed_skill_id"] = int(executed[0, robot_idx])
        row[f"robot{robot_idx}_executed_skill"] = skill_name(executed[0, robot_idx])
        row[f"robot{robot_idx}_invalid_skill"] = int(invalid[0, robot_idx])
        row[f"robot{robot_idx}_cmd_x"] = float(commands[0, robot_idx, 0])
        row[f"robot{robot_idx}_cmd_y"] = float(commands[0, robot_idx, 1])
        row[f"robot{robot_idx}_cmd_yaw"] = float(commands[0, robot_idx, 2])
        obstacle = missing_xy if obstacle_xy is None else obstacle_xy[0, robot_idx]
        row[f"obstacle{robot_idx}_x"] = float(obstacle[0])
        row[f"obstacle{robot_idx}_y"] = float(obstacle[1])

    for key in TERMINAL_KEYS:
        row[key] = int(bool(info_array(info, key, (1,), False)[0]))

    return row


def write_metrics_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, rows, args, show):
    if not rows:
        return

    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([row["time_s"] for row in rows], dtype=np.float32)
    ball_x = np.array([row["ball_x"] for row in rows], dtype=np.float32)
    ball_y = np.array([row["ball_y"] for row in rows], dtype=np.float32)
    reward = np.array([row["reward"] for row in rows], dtype=np.float32)
    num_robots = int(rows[0]["num_robots"])
    invalid = np.array([
        sum(row[f"robot{idx}_invalid_skill"] for idx in range(num_robots)) for row in rows
    ], dtype=np.float32)

    half_width = 0.5 * args.field_width
    goal_x = 0.5 * args.field_length

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(times, ball_x, color="tab:blue", label="ball x")
    axes[0].axhline(goal_x, color="tab:green", linestyle="--", label="goal x")
    axes[0].set_ylabel("x (m)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, ball_y, color="tab:orange", label="ball y")
    axes[1].axhline(half_width, color="black", linestyle=":", linewidth=1)
    axes[1].axhline(-half_width, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylabel("y (m)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    for robot_idx in range(num_robots):
        distance = np.array([row[f"robot{robot_idx}_ball_dist"] for row in rows], dtype=np.float32)
        axes[2].plot(times, distance, label=f"robot {robot_idx}")
    axes[2].set_ylabel("ball dist (m)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")

    for robot_idx in range(num_robots):
        skill = np.array([row[f"robot{robot_idx}_executed_skill_id"] for row in rows], dtype=np.float32)
        axes[3].step(times, skill + 0.04 * robot_idx, where="post", label=f"robot {robot_idx}")
    axes[3].set_yticks(range(len(SKILL_NAMES)))
    axes[3].set_yticklabels(SKILL_NAMES)
    axes[3].set_ylabel("skill")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right")

    axes[4].plot(times, reward, color="tab:green", label="reward")
    axes[4].step(times, invalid, where="post", color="tab:red", alpha=0.7, label="invalid requests")
    axes[4].set_ylabel("reward / count")
    axes[4].set_xlabel("time (s)")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def print_summary(rows):
    if not rows:
        print("No rollout rows were recorded.")
        return

    total_reward = sum(row["reward"] for row in rows)
    terminations = sum(row["done"] for row in rows)
    goals = sum(row["high_level_goal"] for row in rows)
    off_border = sum(row["high_level_ball_off_border"] for row in rows)
    obstacle_hits = sum(row["high_level_obstacle_contact"] for row in rows)
    accidental = sum(row["high_level_accidental_termination"] for row in rows)
    num_robots = int(rows[0]["num_robots"])
    invalid_requests = sum(
        row[f"robot{idx}_invalid_skill"] for row in rows for idx in range(num_robots)
    )
    max_ball_speed = max(row["ball_speed"] for row in rows)
    final_ball_x = rows[-1]["ball_x"]

    print(f"Total reward: {total_reward:.3f}")
    print(f"Mean reward per high-level step: {total_reward / len(rows):.3f}")
    print(f"Terminations: {terminations} | goals: {goals} | off border: {off_border} | obstacles: {obstacle_hits} | accidental: {accidental}")
    print(f"Invalid low-level skill requests: {invalid_requests}")
    print(f"Final ball x: {final_ball_x:.3f} | max ball speed: {max_ball_speed:.3f}")

    for robot_idx in range(num_robots):
        requested_counts = {name: 0 for name in SKILL_NAMES}
        executed_counts = {name: 0 for name in SKILL_NAMES}
        for row in rows:
            requested_counts[row[f"robot{robot_idx}_requested_skill"]] += 1
            executed_counts[row[f"robot{robot_idx}_executed_skill"]] += 1
        print(f"Robot {robot_idx} requested skills: {requested_counts}")
        print(f"Robot {robot_idx} executed skills:  {executed_counts}")


def run(args):
    set_seed(args.seed)
    high_level_policy = load_high_level_policy(args)
    skill_policies = load_skill_policies(args)
    env, raw_env = make_env(args, skill_policies)
    validate_high_level_obs_shape(high_level_policy, env)
    validate_low_level_skill_shapes(skill_policies, env)

    output_video = Path(args.video)
    writer = None
    if not args.no_video:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        high_level_dt = raw_env.dt * env.control_interval
        fps = args.fps or max(1, int(round(1.0 / (high_level_dt * max(args.frame_stride, 1)))))
        writer = imageio.get_writer(str(output_video), fps=fps)
    else:
        high_level_dt = raw_env.dt * env.control_interval

    obs = env.reset()
    rows = []

    try:
        if writer is not None and args.include_initial_frame:
            writer.append_data(raw_env.render(mode="rgb_array"))

        for step in trange(args.steps, desc="High-level rollout"):
            state = collect_state(raw_env)
            with torch.no_grad():
                action = high_level_policy["policy"](obs).to(raw_env.device)
            obs, reward, done, info = env.step(action)
            rows.append(row_from_step(step, high_level_dt, state, action, reward, done, info))

            if writer is not None and step % max(args.frame_stride, 1) == 0:
                writer.append_data(raw_env.render(mode="rgb_array"))

            if args.stop_on_done and bool(done[0].item()):
                break
    finally:
        if writer is not None:
            writer.close()

    metrics_csv = Path(args.csv)
    plot_path = Path(args.plot)
    write_metrics_csv(metrics_csv, rows)
    save_plot(plot_path, rows, args, args.show_plot)

    if not args.no_video:
        print(f"Saved video: {output_video}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print_summary(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and validate a trained high-level multi-robot policy from W&B.")
    parser.add_argument("--high-level-wandb-run", required=True, help="High-level W&B run URL or entity/project/run_id.")
    parser.add_argument("--high-level-checkpoint", default="latest", help="High-level checkpoint suffix, for example latest or 10000.")
    parser.add_argument("--skill-checkpoint", default="latest", help="Checkpoint suffix for walk/dribble/shoot low-level skills.")
    parser.add_argument("--walk-wandb-run", default="des_zhong/walking/fdmj3ehy")
    parser.add_argument("--dribble-wandb-run", default="des_zhong/dribbling/4m92o5tf")
    parser.add_argument("--shoot-wandb-run", default="des_zhong/shooting/finz9edm")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-robots", type=int, default=2, help="Controlled robot count; creates the same number of static obstacles.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--episode-length", type=float, default=30.0)
    parser.add_argument("--control-interval", type=int, default=10)
    parser.add_argument("--high-level-history", type=int, default=4)
    parser.add_argument("--field-length", type=float, default=8.0)
    parser.add_argument("--field-width", type=float, default=5.0)
    parser.add_argument("--goal-half-width", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--walk-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-yaw-speed-scale", type=float, default=0.0)
    parser.add_argument("--walk-yaw-reward-scale", type=float, default=1.0)
    parser.add_argument("--dribble-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--shoot-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--shoot-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--domain-rand", action="store_true", help="Keep domain randomization enabled during playback.")
    parser.add_argument("--fixed-init", action="store_true", help="Disable randomized robot/ball/obstacle match initialization.")

    parser.add_argument("--video", default="outputs/high_level_eval.mp4")
    parser.add_argument("--plot", default="outputs/high_level_eval_metrics.png")
    parser.add_argument("--csv", default="outputs/high_level_eval_metrics.csv")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--include-initial-frame", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--no-field-markers", action="store_true")
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--recording-fov", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
