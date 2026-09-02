"""Minimal match-level manager terms used while the four-robot scene is staged."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_apply

from ..command_frames import world_xy_to_body_xy


WALK_SKILL_ID = 0
DRIBBLE_SKILL_ID = 1
SHOOT_SKILL_ID = 2

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def match_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a finite per-step survival signal for match smoke tests."""

    return torch.ones(env.num_envs, device=env.device)


def match_ball_out_of_bounds(
    env: ManagerBasedRLEnv,
    half_extent_xy: tuple[float, float] = (4.0, 2.5),
) -> torch.Tensor:
    """Terminate a match smoke episode when the ball leaves the field."""

    ball = env.scene["ball"]
    local_xy = ball.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (torch.abs(local_xy[:, 0]) > float(half_extent_xy[0])) | (
        torch.abs(local_xy[:, 1]) > float(half_extent_xy[1])
    )


def _field_ball_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["ball"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


def match_goal(
    env: ManagerBasedRLEnv,
    goal_x: float = 4.0,
    goal_half_width: float = 1.0,
) -> torch.Tensor:
    ball_xy = _field_ball_xy(env)
    return (ball_xy[:, 0] >= float(goal_x)) & (torch.abs(ball_xy[:, 1]) <= float(goal_half_width))


def match_opponent_goal(
    env: ManagerBasedRLEnv,
    goal_x: float = 4.0,
    goal_half_width: float = 1.0,
) -> torch.Tensor:
    ball_xy = _field_ball_xy(env)
    return (ball_xy[:, 0] <= -float(goal_x)) & (torch.abs(ball_xy[:, 1]) <= float(goal_half_width))


def match_robot_fallen(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
    min_height: float = 0.20,
) -> torch.Tensor:
    heights = torch.stack([env.scene[name].data.root_pos_w[:, 2] for name in robot_names], dim=1)
    return torch.any(heights < float(min_height), dim=1)


def match_goal_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.get_term("goal").float()


def match_opponent_goal_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.termination_manager.get_term("opponent_goal").float()


def match_accidental_termination_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current Isaac Gym accidental reset contract from the learner perspective."""

    return (
        env.termination_manager.get_term("opponent_goal")
        | env.termination_manager.get_term("ball_out_of_bounds")
        | env.termination_manager.get_term("robot_fallen")
    ).float()


def match_possession(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
) -> torch.Tensor:
    ball_xy = env.scene["ball"].data.root_pos_w[:, :2]
    # Match reward is from the learning team's (+x) perspective. Opponent
    # proximity must not award possession to the learning policy.
    team_names = robot_names[: len(robot_names) // 2]
    robot_xy = torch.stack([env.scene[name].data.root_pos_w[:, :2] for name in team_names], dim=1)
    distance = torch.linalg.vector_norm(robot_xy - ball_xy[:, None, :], dim=-1).amin(dim=1)
    return torch.exp(-2.0 * torch.square(distance))


def match_ball_goal_progress(env: ManagerBasedRLEnv, goal_x: float = 4.0) -> torch.Tensor:
    """Ball velocity projected toward the learning team's goal."""

    ball = env.scene["ball"]
    goal = env.scene.env_origins[:, :2].clone()
    goal[:, 0] += float(goal_x)
    direction = goal - ball.data.root_pos_w[:, :2]
    direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
    speed = torch.sum(ball.data.root_lin_vel_w[:, :2] * direction, dim=-1)
    return speed.clamp(-1.0, 1.0)


def match_robot_spacing(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
    min_spacing: float = 0.65,
    target_spacing: float = 1.5,
    support_min_ball_distance: float = 1.15,
) -> torch.Tensor:
    """Reward useful teammate spacing and penalize support-ball crowding."""

    team_names = robot_names[: len(robot_names) // 2]
    positions = torch.stack([env.scene[name].data.root_pos_w[:, :2] for name in team_names], dim=1)
    if positions.shape[1] < 2:
        return torch.zeros(env.num_envs, device=env.device)
    pair_distance = torch.linalg.vector_norm(positions[:, 0] - positions[:, 1], dim=-1)
    useful = torch.exp(-torch.square((pair_distance - float(target_spacing)) / float(target_spacing)))
    too_close = torch.clamp(float(min_spacing) - pair_distance, min=0.0) / float(min_spacing)
    ball_xy = env.scene["ball"].data.root_pos_w[:, None, :2]
    ball_distance = torch.linalg.vector_norm(positions - ball_xy, dim=-1)
    attacker = _team_attacker_mask(env, team_names, ball_distance)
    support_distance = torch.where(
        ~attacker,
        ball_distance,
        torch.full_like(ball_distance, float("inf")),
    ).amin(dim=1)
    crowding = torch.clamp(
        (float(support_min_ball_distance) - support_distance) / float(support_min_ball_distance),
        min=0.0,
        max=1.0,
    )
    return useful - 4.0 * too_close.square() - 2.0 * crowding


def match_robot_collision(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1", "robot_2", "robot_3"),
    collision_distance: float = 0.65,
    lookahead: float = 0.25,
) -> torch.Tensor:
    """Predict closest robot-pair separation over a short horizon."""

    positions = torch.stack([env.scene[name].data.root_pos_w[:, :2] for name in robot_names], dim=1)
    velocities = torch.stack([env.scene[name].data.root_lin_vel_w[:, :2] for name in robot_names], dim=1)
    pairs = torch.triu_indices(len(robot_names), len(robot_names), offset=1, device=env.device)
    # Exclude opponent-opponent pairs: the learning team cannot control them.
    pairs = pairs[:, pairs[0] < len(robot_names) // 2]
    relative_position = positions[:, pairs[0]] - positions[:, pairs[1]]
    relative_velocity = velocities[:, pairs[0]] - velocities[:, pairs[1]]
    current_distance = torch.linalg.vector_norm(relative_position, dim=-1)
    speed_sq = relative_velocity.square().sum(dim=-1)
    closest_time = (
        -torch.sum(relative_position * relative_velocity, dim=-1) / speed_sq.clamp_min(1.0e-6)
    ).clamp(0.0, max(float(lookahead), 0.0))
    closest_distance = torch.linalg.vector_norm(
        relative_position + closest_time.unsqueeze(-1) * relative_velocity, dim=-1
    )
    distance = torch.minimum(current_distance, closest_distance)
    overlap = torch.clamp(
        (float(collision_distance) - distance) / float(collision_distance), min=0.0, max=1.0
    )
    return overlap.square().amax(dim=1)


def match_invalid_skill(
    env: ManagerBasedRLEnv,
    action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
) -> torch.Tensor:
    masks = [env.action_manager.get_term(name).invalid_skill_mask for name in action_names]
    return torch.stack(masks, dim=1).float().sum(dim=1)


def match_approach_ball(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1"),
    action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
    dribble_distance: float = 1.0,
) -> torch.Tensor:
    scores = []
    active_masks = []
    for robot_name, action_name in zip(robot_names, action_names, strict=True):
        robot = env.scene[robot_name]
        term = env.action_manager.get_term(action_name)
        delta = env.scene["ball"].data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / distance.clamp_min(1.0e-6).unsqueeze(-1)
        progress = torch.sum(robot.data.root_lin_vel_w[:, :2] * direction, dim=-1).clamp(-1.0, 1.0)
        active = (
            term.attacker_mask
            & (term.skill_ids == WALK_SKILL_ID)
            & (term.requested_skill_ids == WALK_SKILL_ID)
            & ~term.invalid_skill_mask
            & (distance > float(dribble_distance))
        )
        scores.append(progress)
        active_masks.append(active)
    return _reduce_active_score(torch.stack(scores, dim=1), torch.stack(active_masks, dim=1))


def match_walk_command_alignment(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1"),
    action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
) -> torch.Tensor:
    scores = []
    active_masks = []
    ball_xy = env.scene["ball"].data.root_pos_w[:, :2]
    for robot_name, action_name in zip(robot_names, action_names, strict=True):
        robot = env.scene[robot_name]
        term = env.action_manager.get_term(action_name)
        target_body = world_xy_to_body_xy(ball_xy - robot.data.root_pos_w[:, :2], robot.data.root_quat_w)
        command = term.skill_commands[:, :2]
        alignment = torch.nn.functional.cosine_similarity(command, target_body, dim=-1, eps=1.0e-6)
        active = (
            term.attacker_mask
            & (term.skill_ids == WALK_SKILL_ID)
            & (term.requested_skill_ids == WALK_SKILL_ID)
            & ~term.invalid_skill_mask
        )
        scores.append(alignment)
        active_masks.append(active)
    return _reduce_active_score(torch.stack(scores, dim=1), torch.stack(active_masks, dim=1))


def match_face_ball_while_approaching(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1"),
    action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
    dribble_distance: float = 1.0,
    target_speed: float = 0.9,
) -> torch.Tensor:
    """Reward the assigned attacker for facing the ball while walking toward it.

    This is an upper-level shaping term from the Isaac Gym MARL objective.  It
    is intentionally evaluated only for a valid walk request while the attacker
    is still outside the dribble range; a stationary robot cannot farm the
    orientation reward.
    """

    ball_xy = env.scene["ball"].data.root_pos_w[:, :2]
    scores = []
    active_masks = []
    for robot_name, action_name in zip(robot_names, action_names, strict=True):
        robot = env.scene[robot_name]
        term = env.action_manager.get_term(action_name)
        ball_delta = ball_xy - robot.data.root_pos_w[:, :2]
        ball_direction = ball_delta / torch.linalg.vector_norm(
            ball_delta, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        forward_seed = torch.zeros_like(robot.data.root_pos_w)
        forward_seed[:, 0] = 1.0
        forward = quat_apply(robot.data.root_quat_w, forward_seed)[:, :2]
        forward = forward / torch.linalg.vector_norm(
            forward, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        facing = torch.sum(forward * ball_direction, dim=-1).clamp(-1.0, 1.0)
        command_speed = torch.linalg.vector_norm(term.skill_commands[:, :2], dim=-1)
        activity = (command_speed / max(float(target_speed), 1.0e-6)).clamp(0.0, 1.0)
        distance = torch.linalg.vector_norm(ball_delta, dim=-1)
        active = (
            term.attacker_mask
            & (term.skill_ids == WALK_SKILL_ID)
            & (term.requested_skill_ids == WALK_SKILL_ID)
            & ~term.invalid_skill_mask
            & (distance > float(dribble_distance))
        )
        scores.append(facing * activity)
        active_masks.append(active)
    return _reduce_active_score(torch.stack(scores, dim=1), torch.stack(active_masks, dim=1))


def match_face_goal_while_moving(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1"),
    action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
    goal_x: float = 4.0,
    target_speed: float = 0.5,
) -> torch.Tensor:
    """Reward the assigned attacker for moving forward toward the goal.

    The minimum of goalward and body-forward velocity keeps backward motion
    from receiving a positive score merely because the robot is oriented at
    the goal.  This mirrors the sign-preserving Gym high-level term while
    keeping all state access inside a manager reward term.
    """

    goal = env.scene.env_origins[:, :2].clone()
    goal[:, 0] += float(goal_x)
    scores = []
    active_masks = []
    for robot_name, action_name in zip(robot_names, action_names, strict=True):
        robot = env.scene[robot_name]
        term = env.action_manager.get_term(action_name)
        goal_delta = goal - robot.data.root_pos_w[:, :2]
        goal_direction = goal_delta / torch.linalg.vector_norm(
            goal_delta, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        forward_seed = torch.zeros_like(robot.data.root_pos_w)
        forward_seed[:, 0] = 1.0
        forward = quat_apply(robot.data.root_quat_w, forward_seed)[:, :2]
        forward = forward / torch.linalg.vector_norm(
            forward, dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        velocity = robot.data.root_lin_vel_w[:, :2]
        scale = max(float(target_speed), 1.0e-6)
        goalward = (torch.sum(velocity * goal_direction, dim=-1) / scale).clamp(-1.0, 1.0)
        forward_speed = (torch.sum(velocity * forward, dim=-1) / scale).clamp(-1.0, 1.0)
        valid = (term.skill_ids == term.requested_skill_ids) & ~term.invalid_skill_mask
        active = term.attacker_mask & valid
        scores.append(torch.minimum(goalward, forward_speed))
        active_masks.append(active)
    return _reduce_active_score(torch.stack(scores, dim=1), torch.stack(active_masks, dim=1))


def match_dribble_ball_control(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1"),
    action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
    control_distance: float = 0.8,
    target_ball_speed: float = 1.0,
) -> torch.Tensor:
    ball = env.scene["ball"]
    ball_speed = torch.linalg.vector_norm(ball.data.root_lin_vel_w[:, :2], dim=-1)
    scores = []
    active_masks = []
    for robot_name, action_name in zip(robot_names, action_names, strict=True):
        term = env.action_manager.get_term(action_name)
        distance = torch.linalg.vector_norm(
            ball.data.root_pos_w[:, :2] - env.scene[robot_name].data.root_pos_w[:, :2], dim=-1
        )
        alignment = torch.nn.functional.cosine_similarity(
            ball.data.root_lin_vel_w[:, :2], term.skill_commands[:, :2], dim=-1, eps=1.0e-6
        ).clamp_min(0.0)
        speed_score = (ball_speed / float(target_ball_speed)).clamp(0.0, 1.0)
        active = (
            term.attacker_mask
            & (term.skill_ids == DRIBBLE_SKILL_ID)
            & (term.requested_skill_ids == DRIBBLE_SKILL_ID)
            & ~term.invalid_skill_mask
            & (distance <= float(control_distance))
        )
        scores.append(alignment * speed_score)
        active_masks.append(active)
    return _reduce_active_score(torch.stack(scores, dim=1), torch.stack(active_masks, dim=1))


def match_pass(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1"),
    action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
) -> torch.Tensor:
    """Reward a shot that sends a moving ball toward an available teammate."""

    if len(robot_names) < 2:
        return torch.zeros(env.num_envs, device=env.device)
    robot_xy = torch.stack([env.scene[name].data.root_pos_w[:, :2] for name in robot_names], dim=1)
    ball = env.scene["ball"]
    ball_xy = ball.data.root_pos_w[:, :2]
    ball_velocity = ball.data.root_lin_vel_w[:, :2]
    ball_speed = torch.linalg.vector_norm(ball_velocity, dim=-1)
    ball_direction = ball_velocity / ball_speed.clamp_min(1.0e-6).unsqueeze(-1)
    scores = []
    for passer, action_name in enumerate(action_names):
        term = env.action_manager.get_term(action_name)
        passer_distance = torch.linalg.vector_norm(robot_xy[:, passer] - ball_xy, dim=-1)
        for receiver in range(len(robot_names)):
            if receiver == passer:
                continue
            receiver_vector = robot_xy[:, receiver] - ball_xy
            receiver_distance = torch.linalg.vector_norm(receiver_vector, dim=-1)
            receiver_direction = receiver_vector / receiver_distance.clamp_min(1.0e-6).unsqueeze(-1)
            alignment = torch.sum(ball_direction * receiver_direction, dim=-1).clamp(0.0, 1.0)
            score = (
                (term.skill_ids == SHOOT_SKILL_ID).float()
                * torch.exp(-4.0 * passer_distance.square())
                * torch.exp(-0.5 * (receiver_distance - 1.2).square())
                * alignment
                * (ball_speed > 0.35).float()
            )
            scores.append(score)
    return torch.stack(scores, dim=1).amax(dim=1) if scores else torch.zeros(env.num_envs, device=env.device)


def match_shoot_setup(
    env: ManagerBasedRLEnv,
    robot_names: tuple[str, ...] = ("robot_0", "robot_1"),
    action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
    shoot_distance: float = 0.75,
    min_alignment: float = 0.3,
    min_command_speed: float = 0.2,
) -> torch.Tensor:
    ball_xy = env.scene["ball"].data.root_pos_w[:, :2]
    goal = env.scene.env_origins[:, :2].clone()
    goal[:, 0] += 4.0
    goal_direction = goal - ball_xy
    goal_direction /= torch.linalg.vector_norm(goal_direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
    scores = []
    active_masks = []
    for robot_name, action_name in zip(robot_names, action_names, strict=True):
        term = env.action_manager.get_term(action_name)
        robot_to_ball = ball_xy - env.scene[robot_name].data.root_pos_w[:, :2]
        distance = torch.linalg.vector_norm(robot_to_ball, dim=-1)
        direction = robot_to_ball / distance.clamp_min(1.0e-6).unsqueeze(-1)
        alignment = torch.sum(direction * goal_direction, dim=-1).clamp(-1.0, 1.0)
        command = term.skill_commands[:, :2]
        command_speed = torch.linalg.vector_norm(command, dim=-1)
        command_alignment = torch.nn.functional.cosine_similarity(
            command, goal_direction, dim=-1, eps=1.0e-6
        ).clamp(-1.0, 1.0)
        readiness = ((alignment - float(min_alignment)) / max(1.0 - float(min_alignment), 1.0e-6)).clamp(0.0, 1.0)
        active = (
            term.attacker_mask
            & (term.skill_ids == SHOOT_SKILL_ID)
            & ~term.invalid_skill_mask
            & term.skill_transition_mask
            & (distance <= float(shoot_distance))
            & (command_speed >= float(min_command_speed))
            & (readiness > 0.0)
        )
        scores.append(command_alignment * readiness)
        active_masks.append(active)
    return _reduce_active_score(torch.stack(scores, dim=1), torch.stack(active_masks, dim=1))


class MatchShootLaunchReward(ManagerTermBase):
    """One-step reward for a valid shot that accelerates the ball along its command."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_ball_velocity = env.scene["ball"].data.root_lin_vel_w[:, :2].clone()
        action_names = tuple(cfg.params.get("action_names", ("skill_policy_0", "skill_policy_1")))
        robot_names = tuple(cfg.params.get("robot_names", ("robot_0", "robot_1")))
        ball_xy = env.scene["ball"].data.root_pos_w[:, :2]
        self.previous_distances = torch.stack(
            [torch.linalg.vector_norm(env.scene[name].data.root_pos_w[:, :2] - ball_xy, dim=-1) for name in robot_names],
            dim=1,
        )
        if len(action_names) != len(robot_names):
            raise ValueError("MatchShootLaunchReward action_names and robot_names must have equal length")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        env = self._env
        ball_xy = env.scene["ball"].data.root_pos_w[:, :2]
        self.previous_ball_velocity[ids] = env.scene["ball"].data.root_lin_vel_w[ids, :2]
        robot_names = tuple(self.cfg.params.get("robot_names", ("robot_0", "robot_1")))
        for slot, name in enumerate(robot_names):
            self.previous_distances[ids, slot] = torch.linalg.vector_norm(
                env.scene[name].data.root_pos_w[ids, :2] - ball_xy[ids], dim=-1
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_names: tuple[str, ...] = ("robot_0", "robot_1"),
        action_names: tuple[str, ...] = ("skill_policy_0", "skill_policy_1"),
        shoot_distance: float = 0.75,
        min_command_speed: float = 0.2,
        min_ball_speed: float = 0.8,
        min_delta_speed: float = 0.25,
        target_delta_speed: float = 1.5,
        min_command_alignment: float = 0.6,
    ) -> torch.Tensor:
        ball = env.scene["ball"]
        current_velocity = ball.data.root_lin_vel_w[:, :2]
        current_ball_xy = ball.data.root_pos_w[:, :2]
        per_robot = []
        current_distances = []
        for slot, (robot_name, action_name) in enumerate(zip(robot_names, action_names, strict=True)):
            term = env.action_manager.get_term(action_name)
            command = term.skill_commands[:, :2]
            command_speed = torch.linalg.vector_norm(command, dim=-1)
            command_direction = command / command_speed.clamp_min(1.0e-6).unsqueeze(-1)
            current_projected = torch.sum(current_velocity * command_direction, dim=-1)
            previous_projected = torch.sum(self.previous_ball_velocity * command_direction, dim=-1)
            delta_projected = current_projected - previous_projected
            ball_speed = torch.linalg.vector_norm(current_velocity, dim=-1)
            alignment = current_projected / ball_speed.clamp_min(1.0e-6)
            alignment_score = (
                (alignment - float(min_command_alignment))
                / max(1.0 - float(min_command_alignment), 1.0e-6)
            ).clamp(0.0, 1.0)
            launch_score = (
                (delta_projected - float(min_delta_speed))
                / max(float(target_delta_speed) - float(min_delta_speed), 1.0e-6)
            ).clamp(0.0, 1.0)
            launched = (
                (term.skill_ids == SHOOT_SKILL_ID)
                & (term.requested_skill_ids == SHOOT_SKILL_ID)
                & ~term.invalid_skill_mask
                & (self.previous_distances[:, slot] <= float(shoot_distance))
                & (command_speed >= float(min_command_speed))
                & (current_projected >= float(min_ball_speed))
                & (delta_projected >= float(min_delta_speed))
                & (alignment >= float(min_command_alignment))
            )
            per_robot.append(launch_score * alignment_score * launched.float())
            current_distances.append(torch.linalg.vector_norm(
                env.scene[robot_name].data.root_pos_w[:, :2] - current_ball_xy, dim=-1
            ))
        self.previous_ball_velocity[:] = current_velocity
        self.previous_distances[:] = torch.stack(current_distances, dim=1)
        return torch.stack(per_robot, dim=1).amax(dim=1)


def _team_attacker_mask(env, team_names, distances: torch.Tensor) -> torch.Tensor:
    masks = []
    for slot, _ in enumerate(team_names):
        term = env.action_manager.get_term(f"skill_policy_{slot}")
        masks.append(term.attacker_mask)
    mask = torch.stack(masks, dim=1)
    valid = mask.long().sum(dim=1) == 1
    nearest = torch.nn.functional.one_hot(distances.argmin(dim=1), num_classes=len(team_names)).bool()
    return torch.where(valid[:, None], mask, nearest)


def _reduce_active_score(scores: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Select the best active robot without letting inactive zeros hide penalties."""

    any_active = torch.any(active, dim=1)
    masked = torch.where(active, scores, torch.full_like(scores, -torch.inf))
    best = torch.amax(masked, dim=1)
    return torch.where(any_active, best, torch.zeros_like(best))
