"""Validate the frozen shooting checkpoint inside the manager-based shooting task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


REBUILD_ROOT = Path(__file__).resolve().parents[1]
if str(REBUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(REBUILD_ROOT))

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DribbleBot-AS2-Shooting-Frozen-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=320)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_launch_rate", type=float, default=0.25)
parser.add_argument("--min_success_rate", type=float, default=0.0)
parser.add_argument(
    "--output",
    type=Path,
    default=REBUILD_ROOT / "outputs/frozen-shooting" / timestamp / "report.json",
)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac Sim started; building the frozen shooting environment", flush=True)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401
from dribblebot_isaaclab.policies import REPRODUCTION_COMMAND_SCALES, SHOOT_SKILL_ID  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _normalized_command(command: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Invert the frozen action term's tanh command map for validation."""

    nonzero = torch.abs(scale) > 1.0e-8
    if torch.any(torch.abs(command[:, ~nonzero]) > 1.0e-6):
        raise RuntimeError("A zero-scale shooting command axis received a non-zero physical command")
    normalized = torch.zeros_like(command)
    ratio = command[:, nonzero] / scale[nonzero]
    normalized[:, nonzero] = torch.atanh(ratio.clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6))
    return normalized


def _episode_record(phase_term: object, env_id: int, outcome: str) -> dict[str, object]:
    return {
        "env_id": int(env_id),
        "outcome": outcome,
        "terminal_step": int(phase_term.last_terminal_step[env_id].item()),
        "launch_step": int(phase_term.last_terminal_launch_step[env_id].item()),
        "terminal_ball_speed_mps": float(phase_term.last_terminal_ball_speed[env_id].item()),
        "terminal_robot_ball_separation_m": float(phase_term.last_terminal_separation[env_id].item()),
        "terminal_velocity_alignment": float(phase_term.last_terminal_alignment[env_id].item()),
    }


def main() -> int:
    print(f"[INFO] Validating task={args_cli.task}", flush=True)
    if args_cli.num_envs <= 0 or args_cli.episodes <= 0 or args_cli.max_steps <= 0:
        raise ValueError("num_envs, episodes, and max_steps must be positive")
    for name, value in (
        ("min_launch_rate", args_cli.min_launch_rate),
        ("min_success_rate", args_cli.min_success_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = int(args_cli.seed)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[INFO] Instantiated frozen shooting environment", flush=True)
    raw_env = env.unwrapped
    try:
        env.reset(seed=args_cli.seed)
        action_term = raw_env.action_manager.get_term("skill_policy")
        if raw_env.action_manager.total_action_dim != 3:
            raise RuntimeError("Frozen shooting validation requires a 3D fixed-skill action term")
        if int(action_term.cfg.fixed_skill_id) != SHOOT_SKILL_ID:
            raise RuntimeError("Frozen shooting task is not fixed to the shoot policy")
        expected_scales = torch.tensor(REPRODUCTION_COMMAND_SCALES, device=raw_env.device)
        torch.testing.assert_close(action_term._command_scales, expected_scales)

        phase_cfg = raw_env.termination_manager.get_term_cfg("shooting_phase")
        phase_term = phase_cfg.func
        robot = raw_env.scene["robot"]
        ball = raw_env.scene["ball"]
        scale = action_term._command_scales[SHOOT_SKILL_ID]

        completed_before = phase_term.completed_count.clone()
        episode_records: list[dict[str, object]] = []
        other_terminations = 0
        invalid_input_rows = 0
        wrong_skill_rows = 0
        command_error_max = 0.0
        launch_events = 0
        simulated_steps = 0

        with torch.inference_mode():
            for step in range(args_cli.max_steps):
                if len(episode_records) >= args_cli.episodes or not simulation_app.is_running():
                    break
                desired_command = raw_env.command_manager.get_command("base_velocity").clone()
                action = _normalized_command(desired_command, scale)
                _, _, terminated, truncated, _ = env.step(action)
                simulated_steps = step + 1
                done = torch.logical_or(terminated, truncated)

                invalid_input_rows += int(action_term.invalid_input_mask.sum().item())
                wrong_skill_rows += int((action_term.skill_ids != SHOOT_SKILL_ID).sum().item())
                active = ~done
                if torch.any(active):
                    error = torch.max(torch.abs(action_term.skill_commands[active] - desired_command[active]))
                    command_error_max = max(command_error_max, float(error.item()))

                launch_events += int(phase_term.launch_event.sum().item())
                completed_after = phase_term.completed_count.clone()
                phase_completed = completed_after > completed_before
                for env_id in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    if len(episode_records) >= args_cli.episodes:
                        break
                    if phase_completed[env_id]:
                        if bool(phase_term.last_terminal_success[env_id].item()):
                            outcome = "success"
                        elif int(phase_term.last_terminal_launch_step[env_id].item()) >= 0:
                            outcome = "failure_after_launch"
                        else:
                            outcome = "failure_no_launch"
                        episode_records.append(_episode_record(phase_term, env_id, outcome))
                    else:
                        other_terminations += 1
                        episode_records.append(
                            {
                                "env_id": int(env_id),
                                "outcome": "other_termination",
                                "terminal_step": -1,
                                "launch_step": -1,
                            }
                        )
                completed_before = completed_after

                for tensor, label in (
                    (robot.data.joint_pos_target, "joint target"),
                    (ball.data.root_pos_w, "ball position"),
                    (action_term.policy_action, "frozen policy action"),
                ):
                    if not torch.isfinite(tensor).all():
                        raise RuntimeError(f"{label} contains non-finite values")

        classified = [record for record in episode_records if record["outcome"] != "other_termination"]
        launched = [record for record in classified if int(record["launch_step"]) >= 0]
        successes = [record for record in classified if record["outcome"] == "success"]
        denominator = max(len(episode_records), 1)
        launch_rate = len(launched) / denominator
        success_rate = len(successes) / denominator
        gates = {
            "requested_episodes_completed": len(episode_records) >= args_cli.episodes,
            "all_terminations_from_shooting_phase": other_terminations == 0,
            "no_invalid_policy_inputs": invalid_input_rows == 0,
            "fixed_shoot_skill_preserved": wrong_skill_rows == 0,
            "command_injection_max_error": command_error_max <= 1.0e-4,
            "minimum_launch_rate": launch_rate >= args_cli.min_launch_rate,
            "minimum_success_rate": success_rate >= args_cli.min_success_rate,
        }
        report = {
            "task": args_cli.task,
            "device": args_cli.device,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "requested_episodes": args_cli.episodes,
            "simulated_steps": simulated_steps,
            "completed_episodes": len(episode_records),
            "launch_events": launch_events,
            "launch_rate": launch_rate,
            "success_rate": success_rate,
            "other_terminations": other_terminations,
            "invalid_input_rows": invalid_input_rows,
            "wrong_skill_rows": wrong_skill_rows,
            "command_error_max": command_error_max,
            "gates": gates,
            "passed": all(gates.values()),
            "episodes": episode_records,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"[INFO] Frozen shooting report written to {args_cli.output}")
        return 0 if report["passed"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
