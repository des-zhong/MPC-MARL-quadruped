import gym
import math
import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse


SKILL_NAMES = ("walk", "dribble", "shoot")
SKILL_TO_ID = {name: idx for idx, name in enumerate(SKILL_NAMES)}


class HighLevelSkillWrapper(gym.Wrapper):
    """High-level N-robot skill wrapper with six coordinator actions per robot."""

    def __init__(self, env, skill_policies, control_interval=None, history_length=None):
        super().__init__(env)
        self.env = env
        self.skill_policies = skill_policies
        missing_policies = set(SKILL_NAMES) - set(skill_policies)
        if missing_policies:
            raise ValueError(f"Missing low-level skill policies: {sorted(missing_policies)}")
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_train_envs = env.num_train_envs
        self.num_robots = int(getattr(env, "num_robots", getattr(env.cfg.env, "num_robots", 2)))
        if self.num_robots < 1:
            raise ValueError("num_robots must be at least 1")
        self.num_actions = int(getattr(env.cfg.env, "high_level_num_actions", 6 * self.num_robots))
        self.num_obs = int(getattr(env.cfg.env, "high_level_num_observations", 25 * self.num_robots + 6))
        if self.num_actions != 6 * self.num_robots:
            raise ValueError(
                f"high_level_num_actions must be 6*num_robots={6 * self.num_robots}, got {self.num_actions}"
            )
        if self.num_obs != 25 * self.num_robots + 6:
            raise ValueError(
                f"high_level_num_observations must be 25*num_robots+6={25 * self.num_robots + 6}, got {self.num_obs}"
            )
        self.num_privileged_obs = self.num_obs
        self.history_length = int(
            history_length if history_length is not None else getattr(env.cfg.env, "high_level_history_length", 4)
        )
        self.num_obs_history = self.num_obs * self.history_length
        self.control_interval = int(
            control_interval if control_interval is not None else getattr(env.cfg.env, "high_level_control_interval", 10)
        )
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf

        self.high_level_obs = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.high_level_obs_history = torch.zeros(
            self.num_envs,
            self.num_obs_history,
            dtype=torch.float,
            device=self.device,
        )
        self.low_level_obs_dim_full = 75
        self.low_level_history_length = int(getattr(env.cfg.env, "num_observation_history", 15))
        self.low_level_obs_history_full = torch.zeros(
            self.num_envs,
            self.num_robots,
            self.low_level_obs_dim_full * self.low_level_history_length,
            dtype=torch.float,
            device=self.device,
        )
        self.low_level_actions = torch.zeros(self.num_envs, self.num_robots, 12, dtype=torch.float, device=self.device)
        self.last_low_level_actions = torch.zeros_like(self.low_level_actions)
        self.skill_ids = torch.zeros(self.num_envs, self.num_robots, dtype=torch.long, device=self.device)
        self.requested_skill_ids = torch.zeros_like(self.skill_ids)
        self.invalid_skill_mask = torch.zeros(self.num_envs, self.num_robots, dtype=torch.bool, device=self.device)
        self.skill_commands = torch.zeros(self.num_envs, self.num_robots, 3, dtype=torch.float, device=self.device)
        configured_raw_clip = float(getattr(self.env.cfg.normalization, "clip_actions", 1.0))
        self.policy_action_clips = {}
        for skill_name in SKILL_NAMES:
            # Records created before checkpoint metadata was added retain the
            # configured raw clip for backward compatibility.  New records
            # always carry the exact training clip.
            clip = float(self.skill_policies[skill_name].get("action_clip", configured_raw_clip))
            if not math.isfinite(clip) or clip <= 0.0:
                raise ValueError(f"Invalid action_clip={clip!r} for {skill_name} policy")
            self.policy_action_clips[skill_name] = clip
        # The raw env is only a final safety ceiling.  Each policy is clipped
        # to its narrower training range before all robot actions are
        # concatenated.
        self.raw_low_level_action_clip = max(self.policy_action_clips.values())
        self.env.cfg.normalization.clip_actions = self.raw_low_level_action_clip
        self.cached_obs = self._obs_dict()

    def _sanitize_tensor(self, tensor, clip=None):
        if clip is None:
            return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)

    def _obs_clip(self):
        return float(getattr(self.env.cfg.normalization, "clip_observations", 100.0))

    def _action_clip(self):
        return float(getattr(self.env.cfg.normalization, "clip_actions", self.raw_low_level_action_clip))

    def _high_level_action_clip(self):
        return float(getattr(self.env.cfg.env, "high_level_action_input_clip", 10.0))

    def _bad_env_mask(self):
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        roots = self._robot_roots()
        bad |= ~torch.isfinite(roots).flatten(1).all(dim=1)
        bad |= ~torch.isfinite(self.env.dof_pos).all(dim=1)
        bad |= ~torch.isfinite(self.env.dof_vel).all(dim=1)
        if getattr(self.env.cfg.env, "add_balls", False):
            bad |= ~torch.isfinite(self.env.object_pos_world_frame).all(dim=1)
            bad |= ~torch.isfinite(self.env.object_lin_vel).all(dim=1)
        return bad

    def _reset_bad_envs(self):
        bad_envs = self._bad_env_mask()
        if not bool(torch.any(bad_envs).detach().cpu().item()):
            return bad_envs

        env_ids = bad_envs.nonzero(as_tuple=False).flatten()
        if hasattr(self.env, "high_level_accidental_termination_buf"):
            self.env.high_level_accidental_termination_buf[env_ids] = True
        self.env.reset_buf[env_ids] = True
        self.env.reset_idx(env_ids)
        self.low_level_obs_history_full[env_ids] = 0.0
        self.low_level_actions[env_ids] = 0.0
        self.last_low_level_actions[env_ids] = 0.0
        return bad_envs

    def _cfg_command_scale(self, name, default):
        values = getattr(self.env.cfg.env, name, default)
        if len(values) != 3:
            raise ValueError(f"cfg.env.{name} must have 3 values, got {values}")
        return torch.tensor(values, dtype=torch.float, device=self.device)

    def _command_scales(self, skill_ids):
        walk = self._cfg_command_scale("high_level_walk_command_scale", [1.2, 0.6, 0.0])
        dribble = self._cfg_command_scale("high_level_dribble_command_scale", [1.5, 1.5, 1.0])
        shoot = self._cfg_command_scale("high_level_shoot_command_scale", [1.5, 1.5, 0.0])
        scales = torch.zeros(*skill_ids.shape, 3, dtype=torch.float, device=self.device)
        scales[skill_ids == 0] = walk
        scales[skill_ids == 1] = dribble
        scales[skill_ids == 2] = shoot
        return scales

    def _command_obs_scale(self):
        return self._cfg_command_scale("high_level_command_obs_scale", [1.5, 1.5, 1.0]).clamp(min=1e-6)

    def _robot_roots(self):
        return self.env.root_states[self.env.robot_actor_idxs_all.reshape(-1)].view(
            self.num_envs,
            self.num_robots,
            13,
        )

    def _skill_affordances(self, roots=None):
        if roots is None:
            roots = self._robot_roots()

        robot_pos = roots[:, :, :3]
        robot_quat = roots[:, :, 3:7]
        ball_pos = self.env.object_pos_world_frame[:, None, :].expand(-1, self.num_robots, -1)
        ball_delta_world = ball_pos - robot_pos
        ball_delta_local = quat_rotate_inverse(
            robot_quat.reshape(-1, 4),
            ball_delta_world.reshape(-1, 3),
        ).view(self.num_envs, self.num_robots, 3)
        ball_delta_local[:, :, 2] = 0.0

        robot_to_ball = ball_delta_world[:, :, :2]
        distance = torch.norm(robot_to_ball, dim=-1)

        half_length = 0.5 * float(getattr(self.env.cfg.env, "field_length", 8.0))
        half_width = 0.5 * float(getattr(self.env.cfg.env, "field_width", 5.0))
        goal_x = float(getattr(self.env.cfg.env, "team_goal_x", half_length))
        goal_world = torch.stack(
            (
                self.env.env_origins[:, 0] + goal_x,
                self.env.env_origins[:, 1],
            ),
            dim=-1,
        )
        ball_xy_world = self.env.object_pos_world_frame[:, :2]
        ball_to_goal = goal_world[:, None, :] - ball_xy_world[:, None, :]
        robot_to_ball_dir = robot_to_ball / distance.clamp(min=1e-6).unsqueeze(-1)
        ball_to_goal_dir = ball_to_goal / torch.norm(ball_to_goal, dim=-1).clamp(min=1e-6).unsqueeze(-1)
        behind_alignment = torch.sum(robot_to_ball_dir * ball_to_goal_dir, dim=-1).clamp(min=-1.0, max=1.0)

        dribble_distance = float(getattr(self.env.cfg.rewards, "high_level_dribble_skill_distance", 1.0))
        shoot_distance = float(getattr(self.env.cfg.rewards, "high_level_shoot_skill_distance", 0.75))
        shoot_min_forward = float(getattr(self.env.cfg.rewards, "high_level_shoot_min_forward", -0.1))
        shoot_lateral_reach = float(getattr(self.env.cfg.rewards, "high_level_shoot_lateral_reach", 0.45))
        can_dribble = distance <= dribble_distance
        # A shoot command also represents passes and deliberate clearances, so
        # validity must not assume that every kick targets the opponent goal.
        # Gate only on whether the ball is physically strikeable: close to the
        # robot, roughly in front, and within lateral leg reach.  Goal alignment
        # remains in the feature vector below for policy/reward shaping.
        ball_strikeable = (
            (ball_delta_local[:, :, 0] >= shoot_min_forward)
            & (torch.abs(ball_delta_local[:, :, 1]) <= shoot_lateral_reach)
        )
        can_shoot = (distance <= shoot_distance) & ball_strikeable

        field_scale = torch.tensor([half_length, half_width], dtype=torch.float, device=self.device).clamp(min=1e-6)
        field_diag = torch.norm(field_scale).clamp(min=1e-6)
        features = torch.cat(
            (
                ball_delta_local[:, :, :2] / field_scale,
                (distance / field_diag).unsqueeze(-1),
                can_dribble.float().unsqueeze(-1),
                can_shoot.float().unsqueeze(-1),
                behind_alignment.unsqueeze(-1),
            ),
            dim=-1,
        )
        return {
            "features": features,
            "local_ball_xy": ball_delta_local[:, :, :2],
            "distance": distance,
            "can_dribble": can_dribble,
            "can_shoot": can_shoot,
            "ball_strikeable": ball_strikeable,
            "behind_alignment": behind_alignment,
        }

    def _walk_to_ball_commands(self, affordances):
        local_ball_xy = affordances["local_ball_xy"]
        distance = torch.norm(local_ball_xy, dim=-1, keepdim=True).clamp(min=1e-6)
        direction = local_ball_xy / distance
        approach_speed = float(getattr(self.env.cfg.rewards, "high_level_approach_walk_speed", 0.9))
        yaw_to_ball = torch.atan2(local_ball_xy[:, :, 1], local_ball_xy[:, :, 0])

        commands = torch.zeros(self.num_envs, self.num_robots, 3, dtype=torch.float, device=self.device)
        walk_scale = self._cfg_command_scale("high_level_walk_command_scale", [1.2, 0.6, 0.0])
        commands[:, :, 0] = torch.clamp(direction[:, :, 0] * approach_speed, -walk_scale[0], walk_scale[0])
        commands[:, :, 1] = torch.clamp(direction[:, :, 1] * approach_speed, -walk_scale[1], walk_scale[1])
        commands[:, :, 2] = torch.clamp(1.5 * yaw_to_ball, -walk_scale[2], walk_scale[2])
        return commands

    def _decode_action(self, action):
        action = self._sanitize_tensor(action.to(self.device), self._high_level_action_clip()).view(
            self.num_envs,
            self.num_robots,
            6,
        )
        requested_skill_ids = torch.argmax(action[:, :, :3], dim=-1)
        affordances = self._skill_affordances()

        skill_ids = requested_skill_ids.clone()
        use_geometric_fallback = bool(
            getattr(
                self.env.cfg.env,
                "high_level_use_geometric_skill_fallback",
                False,
            )
        )
        if use_geometric_fallback:
            dribble_invalid = (requested_skill_ids == 1) & ~affordances["can_dribble"]
            shoot_invalid = (requested_skill_ids == 2) & ~affordances["can_shoot"]
            shoot_to_dribble = shoot_invalid & affordances["can_dribble"]
            force_walk = dribble_invalid | (shoot_invalid & ~affordances["can_dribble"])
            invalid_skill_mask = dribble_invalid | shoot_invalid
            skill_ids[shoot_to_dribble] = 1
            skill_ids[force_walk] = 0
        else:
            # Do not silently replace the coordinator's decision. This keeps
            # exploration and credit assignment faithful when no affordance
            # masking is part of the experiment.
            force_walk = torch.zeros_like(requested_skill_ids, dtype=torch.bool)
            invalid_skill_mask = torch.zeros_like(requested_skill_ids, dtype=torch.bool)

        commands = torch.tanh(action[:, :, 3:6]) * self._command_scales(requested_skill_ids)
        final_scales = self._command_scales(skill_ids)
        commands = torch.maximum(torch.minimum(commands, final_scales), -final_scales)
        if use_geometric_fallback:
            walk_to_ball_commands = self._walk_to_ball_commands(affordances)
            commands[force_walk] = walk_to_ball_commands[force_walk]
        commands[:, :, 2] = torch.where(skill_ids == 2, torch.zeros_like(commands[:, :, 2]), commands[:, :, 2])
        command_clip = float(torch.max(self._command_obs_scale()).detach().cpu().item())
        commands = self._sanitize_tensor(commands, max(command_clip, 1.0))
        self.requested_skill_ids[:] = requested_skill_ids
        self.skill_ids[:] = skill_ids
        self.invalid_skill_mask[:] = invalid_skill_mask
        self.env.high_level_requested_skill_ids[:] = requested_skill_ids
        self.skill_commands[:] = commands
        self.env.high_level_skill_ids[:] = skill_ids
        self.env.high_level_invalid_skill_mask[:] = invalid_skill_mask
        self.env.high_level_commands[:] = commands

    def _full_command(self, skill_id, command):
        full = torch.zeros(self.num_envs, 15, dtype=torch.float, device=self.device)
        full[:, 0:3] = command
        full[:, 3] = 0.0
        full[:, 4] = 3.0
        full[:, 5] = 0.5
        full[:, 6] = 0.0
        full[:, 7] = 0.0
        full[:, 8] = 0.5
        full[:, 9] = 0.09
        full[:, 10] = 0.0
        full[:, 11] = 0.0
        full[:, 12] = 0.05
        full[:, 13] = 0.05
        full[:, 14] = 0.005
        shoot_mask = skill_id == 2
        full[shoot_mask, 2] = 0.0
        return full

    def _robot_root_states(self, robot_slot):
        actor_idxs = self.env.robot_actor_idxs_all[:, robot_slot]
        return self.env.root_states[actor_idxs]

    def _robot_dof_slice(self, robot_slot):
        start = robot_slot * self.env.num_robot_dof
        stop = start + self.env.num_robot_dof
        return slice(start, stop)

    def _robot_low_level_observation_full(self, robot_slot, full_command):
        root_state = self._robot_root_states(robot_slot)
        base_pos = root_state[:, :3]
        base_quat = root_state[:, 3:7]
        object_local = quat_rotate_inverse(base_quat, self.env.object_pos_world_frame - base_pos)
        object_local[:, 2] = 0.0
        projected_gravity = quat_rotate_inverse(base_quat, self.env.gravity_vec)

        dof_slice = self._robot_dof_slice(robot_slot)
        dof_pos = (self.env.dof_pos[:, dof_slice] - self.env.default_dof_pos[:, dof_slice]) * self.env.cfg.obs_scales.dof_pos
        dof_vel = self.env.dof_vel[:, dof_slice] * self.env.cfg.obs_scales.dof_vel
        command = full_command * self.env.commands_scale

        forward = quat_apply(base_quat, self.env.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        yaw = yaw - self.env.heading_offsets.unsqueeze(1)
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))

        obs = torch.cat(
            (
                object_local * self.env.cfg.obs_scales.ball_pos,
                projected_gravity,
                command,
                dof_pos,
                dof_vel,
                self.low_level_actions[:, robot_slot, :],
                self.last_low_level_actions[:, robot_slot, :],
                self.env.clock_inputs,
                yaw,
                self.env.gait_indices.unsqueeze(1),
            ),
            dim=-1,
        )
        return self._sanitize_tensor(obs, self._obs_clip())

    def _update_low_level_history(self, robot_slot, obs_full):
        width = self.low_level_obs_dim_full
        history = self.low_level_obs_history_full[:, robot_slot, :]
        obs_full = self._sanitize_tensor(obs_full, self._obs_clip())
        self.low_level_obs_history_full[:, robot_slot, :] = self._sanitize_tensor(
            torch.cat((history[:, width:], obs_full), dim=-1),
            self._obs_clip(),
        )

    def _policy_obs(self, robot_slot, policy_record):
        history = self._sanitize_tensor(self.low_level_obs_history_full[:, robot_slot, :], self._obs_clip())
        expected_dim = policy_record.get("expected_history_dim")
        if expected_dim is None or expected_dim == history.shape[1]:
            return {"obs_history": history}

        no_object_width = self.low_level_obs_dim_full - 3
        no_object_history_dim = no_object_width * self.low_level_history_length
        if expected_dim == no_object_history_dim:
            history_full = history.view(self.num_envs, self.low_level_history_length, self.low_level_obs_dim_full)
            history_no_object = history_full[:, :, 3:].reshape(self.num_envs, no_object_history_dim)
            return {"obs_history": history_no_object}

        raise ValueError(
            f"Skill policy {policy_record.get('source', '<unknown source>')} expects obs_history dim {expected_dim}, "
            f"but high-level wrapper can provide {history.shape[1]} or {no_object_history_dim}. "
            "Check that --walk-wandb-run, --dribble-wandb-run, and --shoot-wandb-run point to low-level skill runs, "
            "not the high-level coordinator run."
        )

    def _low_level_actions_from_skills(self):
        actions = torch.zeros(self.num_envs, self.num_robots, 12, dtype=torch.float, device=self.device)
        full_commands = []
        for robot_slot in range(self.num_robots):
            full_command = self._full_command(self.skill_ids[:, robot_slot], self.skill_commands[:, robot_slot, :])
            full_commands.append(full_command)
            obs_full = self._robot_low_level_observation_full(robot_slot, full_command)
            self._update_low_level_history(robot_slot, obs_full)

            for skill_name, skill_id in SKILL_TO_ID.items():
                mask = self.skill_ids[:, robot_slot] == skill_id
                if not bool(torch.any(mask).detach().cpu().item()):
                    continue
                policy_record = self.skill_policies[skill_name]
                with torch.no_grad():
                    skill_action = policy_record["policy"](self._policy_obs(robot_slot, policy_record)).to(self.device)
                skill_action = self._sanitize_tensor(skill_action, self.policy_action_clips[skill_name])
                actions[mask, robot_slot, :] = skill_action[mask]

        self.env.commands[:, :] = full_commands[0][:, : self.env.cfg.commands.num_commands]
        return actions

    def _high_level_observation(self):
        roots = self.env.root_states[self.env.robot_actor_idxs_all.reshape(-1)].view(self.num_envs, self.num_robots, 13)
        robot_xy = roots[:, :, :2] - self.env.env_origins[:, None, :2]
        robot_forward_vec = self.env.forward_vec[:, None, :].repeat(1, self.num_robots, 1).reshape(-1, 3)
        forward = quat_apply(roots[:, :, 3:7].reshape(-1, 4), robot_forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0]).view(self.num_envs, self.num_robots)
        robot_vel = roots[:, :, 7:9]
        robot_yaw_rate = roots[:, :, 12:13]

        half_length = 0.5 * float(getattr(self.env.cfg.env, "field_length", 8.0))
        half_width = 0.5 * float(getattr(self.env.cfg.env, "field_width", 5.0))
        ball_xy = self.env.object_pos_world_frame[:, :2] - self.env.env_origins[:, :2]
        ball_vel = self.env.object_lin_vel[:, :2]
        affordances = self._skill_affordances(roots)
        field_scale = torch.tensor(
            [max(half_length, 1e-6), max(half_width, 1e-6)],
            dtype=torch.float,
            device=self.device,
        )

        pieces = [
            (robot_xy / field_scale).reshape(self.num_envs, -1),
            torch.cos(yaw),
            torch.sin(yaw),
            (robot_vel / 3.0).reshape(self.num_envs, -1),
            (robot_yaw_rate / 3.0).reshape(self.num_envs, -1),
            ball_xy / field_scale,
            ball_vel / 5.0,
            affordances["features"].reshape(self.num_envs, -1),
        ]

        max_obstacles = self.num_robots
        if getattr(self.env, "num_static_opponents", 0) > 0:
            obstacle_states = self.env.root_states[self.env.static_opponent_actor_idxs.reshape(-1)].view(
                self.num_envs,
                self.env.num_static_opponents,
                13,
            )
            obstacle_xy = obstacle_states[:, :max_obstacles, :2] - self.env.env_origins[:, None, :2]
            obstacle_count = min(max_obstacles, self.env.num_static_opponents)
            obstacle_forward_vec = self.env.forward_vec[:, None, :].repeat(1, obstacle_count, 1).reshape(-1, 3)
            obstacle_forward = quat_apply(
                obstacle_states[:, :max_obstacles, 3:7].reshape(-1, 4),
                obstacle_forward_vec,
            )
            obstacle_yaw = torch.atan2(obstacle_forward[:, 1], obstacle_forward[:, 0]).view(
                self.num_envs,
                obstacle_count,
            )
            obstacle_size = torch.tensor(
                [self.env.static_opponent_size[0] / float(getattr(self.env.cfg.env, "field_length", 8.0)),
                 self.env.static_opponent_size[1] / float(getattr(self.env.cfg.env, "field_width", 5.0))],
                dtype=torch.float,
                device=self.device,
            ).repeat(self.num_envs, obstacle_count, 1)
            obstacle_obs = torch.cat(
                (
                    obstacle_xy / field_scale,
                    torch.cos(obstacle_yaw).unsqueeze(-1),
                    torch.sin(obstacle_yaw).unsqueeze(-1),
                    obstacle_size,
                ),
                dim=-1,
            )
            if self.env.num_static_opponents < max_obstacles:
                pad = torch.zeros(self.num_envs, max_obstacles - self.env.num_static_opponents, 6, device=self.device)
                obstacle_obs = torch.cat((obstacle_obs, pad), dim=1)
        else:
            obstacle_obs = torch.zeros(self.num_envs, max_obstacles, 6, device=self.device)
        pieces.append(obstacle_obs.reshape(self.num_envs, -1))

        goal_x = float(getattr(self.env.cfg.env, "team_goal_x", half_length))
        goal_half_width = float(getattr(self.env.cfg.env, "team_goal_half_width", 1.0))
        pieces.append(
            torch.stack(
                (
                    torch.full((self.num_envs,), goal_x / field_scale[0], dtype=torch.float, device=self.device),
                    torch.full((self.num_envs,), goal_half_width / field_scale[1], dtype=torch.float, device=self.device),
                ),
                dim=-1,
            )
        )

        skill_one_hot = torch.nn.functional.one_hot(self.skill_ids, num_classes=3).float().reshape(self.num_envs, -1)
        pieces.append(skill_one_hot)
        pieces.append(
            self.skill_commands.reshape(self.num_envs, -1)
            / self._command_obs_scale().repeat(self.num_robots)
        )

        obs = torch.cat(pieces, dim=-1)
        if obs.shape[1] != self.num_obs:
            raise RuntimeError(f"High-level obs dim mismatch: built {obs.shape[1]}, configured {self.num_obs}")
        return self._sanitize_tensor(obs, self._obs_clip())

    def _update_high_level_obs(self):
        self.high_level_obs[:] = self._sanitize_tensor(self._high_level_observation(), self._obs_clip())
        width = self.num_obs
        self.high_level_obs_history[:] = self._sanitize_tensor(
            torch.cat((self.high_level_obs_history[:, width:], self.high_level_obs), dim=-1),
            self._obs_clip(),
        )
        self.cached_obs = self._obs_dict()

    def _obs_dict(self):
        return {
            "obs": self.high_level_obs,
            "privileged_obs": self.high_level_obs,
            "obs_history": self.high_level_obs_history,
        }

    def get_observations(self):
        return self.cached_obs

    def _clear_high_level_state(self, env_ids):
        """Clear coordinator state after the raw env has auto-reset rows."""

        self.high_level_obs[env_ids] = 0.0
        self.high_level_obs_history[env_ids] = 0.0
        self.skill_ids[env_ids] = 0
        self.requested_skill_ids[env_ids] = 0
        self.invalid_skill_mask[env_ids] = False
        self.skill_commands[env_ids] = 0.0
        self.env.high_level_skill_ids[env_ids] = 0
        self.env.high_level_requested_skill_ids[env_ids] = 0
        self.env.high_level_invalid_skill_mask[env_ids] = False
        self.env.high_level_commands[env_ids] = 0.0

    def reset(self):
        self.env.reset()
        self.high_level_obs.zero_()
        self.high_level_obs_history.zero_()
        self.low_level_obs_history_full.zero_()
        self.low_level_actions.zero_()
        self.last_low_level_actions.zero_()
        self.skill_ids.zero_()
        self.requested_skill_ids.zero_()
        self.invalid_skill_mask.zero_()
        self.skill_commands.zero_()
        self.env.high_level_skill_ids.zero_()
        self.env.high_level_requested_skill_ids.zero_()
        self.env.high_level_invalid_skill_mask.zero_()
        self.env.high_level_commands.zero_()
        self._update_high_level_obs()
        return self.cached_obs

    def step(self, action):
        self._reset_bad_envs()
        self._decode_action(action)
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        done_total = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        elapsed_low_level_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        info = {}
        terminal_info = {
            "high_level_goal": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "high_level_opponent_goal": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "high_level_ball_off_border": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "high_level_obstacle_contact": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "high_level_accidental_termination": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "time_outs": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }
        terminal_attrs = {
            "high_level_goal": "last_high_level_goal_buf",
            "high_level_opponent_goal": "last_high_level_opponent_goal_buf",
            "high_level_ball_off_border": "last_high_level_ball_off_border_buf",
            "high_level_obstacle_contact": "last_high_level_obstacle_contact_buf",
            "high_level_accidental_termination": "last_high_level_accidental_termination_buf",
            "time_outs": "time_out_buf",
        }

        for _ in range(self.control_interval):
            active = ~done_total
            elapsed_low_level_steps += active.long()
            self.last_low_level_actions[:] = self.low_level_actions
            self.low_level_actions[:] = self._low_level_actions_from_skills()
            joint_actions = self.low_level_actions.reshape(self.num_envs, -1)
            joint_actions[~active] = 0.0
            _, low_reward, done, info = self.env.step(joint_actions)
            bad_envs = self._reset_bad_envs()
            low_reward = self._sanitize_tensor(low_reward, 1.0e4)
            done = torch.logical_or(done.bool(), bad_envs)
            for key, attr_name in terminal_attrs.items():
                if hasattr(self.env, attr_name):
                    terminal_info[key] = torch.logical_or(terminal_info[key], getattr(self.env, attr_name))
            reward += low_reward * active.float()
            done_bool = done.bool()
            done_total = torch.logical_or(done_total, done_bool)
            if bool(torch.any(done_bool).detach().cpu().item()):
                self.low_level_obs_history_full[done_bool] = 0.0
                self.low_level_actions[done_bool] = 0.0
                self.last_low_level_actions[done_bool] = 0.0
            if bool(torch.all(done_total).detach().cpu().item()):
                break

        executed_skill_ids = self.skill_ids.detach().cpu().numpy().copy()
        requested_skill_ids = self.requested_skill_ids.detach().cpu().numpy().copy()
        invalid_skill_mask = self.invalid_skill_mask.detach().cpu().numpy().copy()
        executed_commands = self.skill_commands.detach().cpu().numpy().copy()
        if bool(torch.any(done_total).detach().cpu().item()):
            self._clear_high_level_state(done_total)
        self._update_high_level_obs()
        reward = self._sanitize_tensor(reward, 1.0e4)
        info = dict(info)
        info["privileged_obs"] = self.high_level_obs
        info["high_level_skill_ids"] = executed_skill_ids
        info["high_level_requested_skill_ids"] = requested_skill_ids
        info["high_level_invalid_skill_mask"] = invalid_skill_mask
        info["high_level_commands"] = executed_commands
        info["high_level_robot_ball_distances"] = (
            self.env._high_level_robot_ball_distances().detach().cpu().numpy()
        )
        info["elapsed_low_level_steps"] = elapsed_low_level_steps.detach().cpu().numpy()
        info["low_level_action_clips"] = dict(self.policy_action_clips)
        for key, value in terminal_info.items():
            # PPO bootstraps truncated episodes from this field and expects an
            # on-device tensor, matching the raw environment/logger contract.
            # The remaining terminal fields are reporting metadata consumed by
            # NumPy-based playback and dataset tooling.
            info[key] = value.detach() if key == "time_outs" else value.detach().cpu().numpy()
        return self.cached_obs, reward, done_total, info
