"""Canonical football state snapshots for recorder/world-model adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def capture_match_snapshot(
    env: ManagerBasedEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return a stable, simulator-independent football state dictionary."""

    if env_ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, torch.Tensor):
        ids = env_ids.to(device=env.device, dtype=torch.long)
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    robots = [env.scene[name] for name in robot_names]
    action_terms = [env.action_manager.get_term(f"skill_policy_{index}") for index in range(len(robot_names))]
    gait_phases = torch.stack(
        [env.command_manager.get_term(f"gait_parameters_{index}").gait_phase[ids] for index in range(len(robot_names))],
        dim=1,
    )
    return {
        "env_origin_w": env.scene.env_origins[ids].clone(),
        "robot_root_state_w": torch.stack([robot.data.root_state_w[ids] for robot in robots], dim=1),
        "robot_joint_pos": torch.stack([robot.data.joint_pos[ids] for robot in robots], dim=1),
        "robot_joint_vel": torch.stack([robot.data.joint_vel[ids] for robot in robots], dim=1),
        "robot_joint_target": torch.stack([term.processed_actions[ids] for term in action_terms], dim=1),
        "ball_root_state_w": env.scene["ball"].data.root_state_w[ids].clone(),
        "skill_id": torch.stack([term.skill_ids[ids] for term in action_terms], dim=1),
        "requested_skill_id": torch.stack([term.requested_skill_ids[ids] for term in action_terms], dim=1),
        "invalid_skill": torch.stack([term.invalid_skill_mask[ids] for term in action_terms], dim=1),
        "skill_command": torch.stack([term.skill_commands[ids] for term in action_terms], dim=1),
        "gait_phase": gait_phases,
        "episode_length": env.episode_length_buf[ids].clone(),
    }


class MatchSnapshotRecorder(RecorderTerm):
    """Record canonical match state after every manager step and reset."""

    cfg: MatchSnapshotRecorderCfg

    def __init__(self, cfg: MatchSnapshotRecorderCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._robot_names = tuple(cfg.robot_names)

    def record_post_step(self):
        return "football/snapshot", capture_match_snapshot(self._env, self._robot_names)

    def record_post_reset(self, env_ids):
        return "football/snapshot", capture_match_snapshot(self._env, self._robot_names, env_ids)


@configclass
class MatchSnapshotRecorderCfg(RecorderTermCfg):
    """Configuration for the canonical match snapshot recorder."""

    class_type: type = MatchSnapshotRecorder
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3")


__all__ = ["MatchSnapshotRecorder", "MatchSnapshotRecorderCfg", "capture_match_snapshot"]
