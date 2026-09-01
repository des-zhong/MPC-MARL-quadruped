"""Evaluate the archived shoot checkpoint in its original Isaac Gym task."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REBUILD_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REBUILD_ROOT.parent
for import_root in (REBUILD_ROOT, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=320)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--checkpoint_dir",
    type=Path,
    default=REPOSITORY_ROOT / "checkpoints/reproduction/shoot",
)
parser.add_argument(
    "--output",
    type=Path,
    default=REBUILD_ROOT / "outputs/frozen-shooting" / timestamp / "isaacgym-report.json",
)
args = parser.parse_args()


def _apply_archived_cfg(cfg: object, config_path: Path) -> None:
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    values = payload.get("Cfg", payload)
    if isinstance(values, dict) and "value" in values:
        values = values["value"]
    for section_name, section_values in values.items():
        target = getattr(cfg, section_name, None)
        if target is None or not isinstance(section_values, dict):
            continue
        for key, value in section_values.items():
            setattr(target, key, value)


def _load_policy(checkpoint_dir: Path, device: str):
    import torch

    body = torch.jit.load(str(checkpoint_dir / "body_latest.jit"), map_location=device).eval()
    adaptation = torch.jit.load(
        str(checkpoint_dir / "adaptation_module_latest.jit"),
        map_location=device,
    ).eval()

    def policy(history: torch.Tensor) -> torch.Tensor:
        latent = adaptation.forward(history)
        return body.forward(torch.cat((history, latent), dim=-1))

    return policy


def main() -> int:
    if args.num_envs <= 0 or args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("num_envs, episodes, and max_steps must be positive")
    for name in ("body_latest.jit", "adaptation_module_latest.jit", "config.yaml"):
        path = args.checkpoint_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)

    import isaacgym
    import torch

    assert isaacgym
    from quadruped.envs.as2.as2_config import config_as2
    from quadruped.envs.as2.velocity_tracking import VelocityTrackingEasyEnv
    from quadruped.envs.base.legged_robot_config import Cfg
    from quadruped.envs.wrappers.history_wrapper import HistoryWrapper

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config_as2(Cfg)
    _apply_archived_cfg(Cfg, args.checkpoint_dir / "config.yaml")
    Cfg.robot.name = "as2"
    Cfg.env.num_envs = args.num_envs
    Cfg.env.record_video = False
    Cfg.env.num_recording_envs = 0

    raw_env = VelocityTrackingEasyEnv(sim_device=args.device, headless=True, cfg=Cfg)
    env = HistoryWrapper(raw_env)
    policy = _load_policy(args.checkpoint_dir, args.device)
    launch_step = torch.full((args.num_envs,), -1, dtype=torch.long, device=raw_env.device)
    records: list[dict[str, object]] = []
    original_reset_idx = raw_env.reset_idx

    def capture_reset(env_ids: torch.Tensor) -> None:
        ids = env_ids.to(device=raw_env.device, dtype=torch.long)
        if ids.numel() > 0:
            just_launched = raw_env.shooting_launch_buf[ids] & (launch_step[ids] < 0)
            launch_step[ids[just_launched]] = raw_env.episode_length_buf[ids[just_launched]]
            command_xy = raw_env.commands[ids, :2]
            target_speed = torch.linalg.vector_norm(command_xy, dim=-1)
            command_direction = command_xy / target_speed.clamp_min(1.0e-6).unsqueeze(-1)
            ball_velocity = raw_env.object_lin_vel[ids, :2]
            ball_speed = torch.linalg.vector_norm(ball_velocity, dim=-1)
            alignment = torch.sum(ball_velocity * command_direction, dim=-1) / ball_speed.clamp_min(1.0e-6)
            separation = torch.linalg.vector_norm(
                raw_env.object_pos_world_frame[ids, :2] - raw_env.base_pos[ids, :2],
                dim=-1,
            )
            for local_index, env_id in enumerate(ids.tolist()):
                if len(records) >= args.episodes:
                    break
                if bool(raw_env.shooting_success_buf[env_id].item()):
                    outcome = "success"
                elif bool(raw_env.shooting_failure_buf[env_id].item()):
                    outcome = "failure_after_launch" if launch_step[env_id] >= 0 else "failure_no_launch"
                else:
                    outcome = "other_termination"
                records.append(
                    {
                        "env_id": int(env_id),
                        "outcome": outcome,
                        "terminal_step": int(raw_env.episode_length_buf[env_id].item()),
                        "launch_step": int(launch_step[env_id].item()),
                        "terminal_ball_speed_mps": float(ball_speed[local_index].item()),
                        "terminal_robot_ball_separation_m": float(separation[local_index].item()),
                        "terminal_velocity_alignment": float(alignment[local_index].item()),
                    }
                )
            launch_step[ids] = -1
        original_reset_idx(env_ids)

    raw_env.reset_idx = capture_reset
    try:
        observation = env.reset()
        invalid_action_rows = 0
        simulated_steps = 0
        with torch.inference_mode():
            for step in range(args.max_steps):
                if len(records) >= args.episodes:
                    break
                action = policy(observation["obs_history"])
                invalid_action_rows += int((~torch.isfinite(action).all(dim=-1)).sum().item())
                observation, _, _, _ = env.step(action)
                simulated_steps = step + 1
                newly_launched = raw_env.shooting_launch_buf & (launch_step < 0)
                launch_step[newly_launched] = raw_env.episode_length_buf[newly_launched]

        launched = [record for record in records if int(record["launch_step"]) >= 0]
        successes = [record for record in records if record["outcome"] == "success"]
        other = [record for record in records if record["outcome"] == "other_termination"]
        denominator = max(len(records), 1)
        report = {
            "source": "isaacgym",
            "device": args.device,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "requested_episodes": args.episodes,
            "completed_episodes": len(records),
            "simulated_steps": simulated_steps,
            "launch_rate": len(launched) / denominator,
            "success_rate": len(successes) / denominator,
            "other_terminations": len(other),
            "invalid_action_rows": invalid_action_rows,
            "episodes": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"[INFO] Isaac Gym frozen shooting report written to {args.output}")
        return 0 if len(records) >= args.episodes and invalid_action_rows == 0 else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
