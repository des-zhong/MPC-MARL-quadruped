"""Football reset events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def set_rigid_body_collider_offsets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    rest_offset: float,
    contact_offset: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Set and verify collider offsets through the initialized PhysX view.

    Isaac Sim 5.1's URDF importer creates instanceable collider prims.  USD
    schema overrides on those instance proxies are rejected, while the PhysX
    tensor API updates the actual simulation shapes without breaking asset
    instancing.  This event is intended for ``startup`` mode.
    """

    if rest_offset < 0.0 or contact_offset < 0.0:
        raise ValueError("Collider offsets must be non-negative")
    if contact_offset < rest_offset:
        raise ValueError("contact_offset must be greater than or equal to rest_offset")

    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        physx_env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device="cpu")
    elif isinstance(env_ids, slice):
        physx_env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device="cpu")[env_ids]
    else:
        physx_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")

    view = asset.root_physx_view
    rest_offsets = view.get_rest_offsets().clone()
    contact_offsets = view.get_contact_offsets().clone()
    rest_offsets[physx_env_ids] = float(rest_offset)
    contact_offsets[physx_env_ids] = float(contact_offset)
    view.set_rest_offsets(rest_offsets, physx_env_ids)
    view.set_contact_offsets(contact_offsets, physx_env_ids)

    actual_rest = view.get_rest_offsets()[physx_env_ids]
    actual_contact = view.get_contact_offsets()[physx_env_ids]
    expected_rest = torch.full_like(actual_rest, float(rest_offset))
    expected_contact = torch.full_like(actual_contact, float(contact_offset))
    if not torch.allclose(actual_rest, expected_rest, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            f"Failed to set rest_offset={rest_offset} for asset '{asset_cfg.name}': "
            f"observed range [{actual_rest.min().item()}, {actual_rest.max().item()}]"
        )
    if not torch.allclose(actual_contact, expected_contact, rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            f"Failed to set contact_offset={contact_offset} for asset '{asset_cfg.name}': "
            f"observed range [{actual_contact.min().item()}, {actual_contact.max().item()}]"
        )


def set_rigid_body_com(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    com: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Set selected rigid-body COM translations through the PhysX view.

    The legacy ``LeggedRobot`` actor callback always overwrites the root body
    COM with its ``com_displacements`` buffer.  When COM randomization is
    disabled that buffer is zero, so checkpoint-compatible AS2 dynamics use a
    root COM at the body origin rather than the non-zero URDF inertial origin.
    """

    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        physx_env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device="cpu")
    elif isinstance(env_ids, slice):
        physx_env_ids = torch.arange(env.scene.num_envs, dtype=torch.long, device="cpu")[env_ids]
    else:
        physx_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(asset.num_bodies))[body_ids]
    else:
        body_ids = list(body_ids)
    if not body_ids:
        raise ValueError(f"No bodies resolved for asset '{asset_cfg.name}'")

    view = asset.root_physx_view
    coms = view.get_coms().clone()
    expected = torch.as_tensor(com, dtype=coms.dtype, device=coms.device)
    for body_id in body_ids:
        coms[physx_env_ids, body_id, :3] = expected
    view.set_coms(coms, physx_env_ids)

    actual = view.get_coms()[physx_env_ids][:, body_ids, :3]
    if not torch.allclose(actual, expected.expand_as(actual), rtol=0.0, atol=1.0e-7):
        raise RuntimeError(
            f"Failed to set COM {tuple(com)} for asset '{asset_cfg.name}': "
            f"maximum error {torch.max(torch.abs(actual - expected)).item()}"
        )


def reset_ball_in_front(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    forward_range: tuple[float, float],
    lateral_range: tuple[float, float],
    ball_height: float,
    velocity_range: tuple[float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> None:
    """Reset the ball in front of the robot regardless of randomized yaw."""

    robot: Articulation = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    count = len(env_ids)
    if count == 0:
        return

    random = torch.rand(count, 4, device=ball.device)
    forward = forward_range[0] + random[:, 0] * (forward_range[1] - forward_range[0])
    lateral = lateral_range[0] + random[:, 1] * (lateral_range[1] - lateral_range[0])
    local_offset = torch.stack((forward, lateral, torch.zeros_like(forward)), dim=-1)
    world_offset = quat_apply(yaw_quat(robot.data.root_quat_w[env_ids]), local_offset)

    pose = ball.data.default_root_state[env_ids, :7].clone()
    pose[:, :3] = robot.data.root_pos_w[env_ids] + world_offset
    pose[:, 2] = env.scene.env_origins[env_ids, 2] + float(ball_height)
    pose[:, 3:7] = pose.new_tensor((1.0, 0.0, 0.0, 0.0))

    velocity = ball.data.default_root_state[env_ids, 7:13].clone()
    low, high = velocity_range
    velocity[:, :2] = low + random[:, 2:4] * (high - low)
    velocity[:, 2:] = 0.0
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def reset_match_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
    ball_name: str = "ball",
    team_size: int = 2,
    ball_height: float = 0.10,
) -> None:
    """Reset four AS2 actors and one ball into a deterministic match layout."""

    if len(robot_names) != 2 * int(team_size):
        raise ValueError("robot_names must contain exactly two teams")
    if len(env_ids) == 0:
        return
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    offsets = env.scene.env_origins[env_ids].new_tensor(
        ((-1.0, -0.65, 0.34), (-1.0, 0.65, 0.34), (1.0, -0.65, 0.34), (1.0, 0.65, 0.34))
    )
    yaws = env.scene.env_origins[env_ids].new_tensor((0.0, 0.0, torch.pi, torch.pi))
    for index, name in enumerate(robot_names):
        robot: Articulation = env.scene[name]
        pose = robot.data.default_root_state[env_ids, :7].clone()
        pose[:, :3] = env.scene.env_origins[env_ids] + offsets[index]
        pose[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), device=env.device, dtype=pose.dtype),
            torch.zeros(len(env_ids), device=env.device, dtype=pose.dtype),
            torch.full((len(env_ids),), yaws[index], device=env.device, dtype=pose.dtype),
        )
        velocity = torch.zeros(len(env_ids), 6, device=env.device, dtype=pose.dtype)
        robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[env_ids],
            torch.zeros_like(robot.data.default_joint_vel[env_ids]),
            env_ids=env_ids,
        )

    ball: RigidObject = env.scene[ball_name]
    ball_pose = ball.data.default_root_state[env_ids, :7].clone()
    ball_pose[:, :3] = env.scene.env_origins[env_ids] + ball_pose.new_tensor((0.0, 0.0, ball_height))
    ball_pose[:, 3:7] = ball_pose.new_tensor((1.0, 0.0, 0.0, 0.0))
    ball_velocity = torch.zeros(len(env_ids), 6, device=env.device, dtype=ball_pose.dtype)
    ball.write_root_pose_to_sim(ball_pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(ball_velocity, env_ids=env_ids)
