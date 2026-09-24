import time
from collections import deque
import copy
import os
import shutil

import torch
# from ml_logger import logger
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook

from params_proto import PrefixProto

from .actor_critic import AC_Args, ActorCritic
from .rollout_storage import RolloutStorage


def checkpoint_next_iteration(path):
    """Return the next iteration encoded by ac_weights_<iteration>.pt."""

    if not path:
        return 0
    name = os.path.basename(path)
    prefix, suffix = "ac_weights_", ".pt"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return 0
    token = name[len(prefix) : -len(suffix)]
    return int(token) + 1 if token.isdigit() else 0


def checkpoint_wandb_base_path(checkpoint_dir, wandb_run_dir, working_directory):
    """Choose a base that preserves layout without nesting a run into itself."""

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    working_directory = os.path.abspath(working_directory)
    if wandb_run_dir:
        wandb_run_dir = os.path.abspath(wandb_run_dir)
        if os.path.commonpath((wandb_run_dir, checkpoint_dir)) == wandb_run_dir:
            return wandb_run_dir
    if os.path.commonpath((working_directory, checkpoint_dir)) == working_directory:
        return working_directory
    return checkpoint_dir


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from quadruped_learn.ppo_cse.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10
    checkpoint_dir = './tmp/legged_data'

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True
    resume_checkpoint = 'ac_weights_last.pt'
    # Warm-start only policy features and actor when the task/reward changed.
    # The critic, action noise, and iteration numbering then start fresh.
    resume_policy_only = False
    # Disabled for ordinary locomotion tasks. Competitive wrappers implement
    # ``update_opponent_policy`` and receive a frozen actor snapshot at this
    # interval during self-play training.
    self_play_update_interval = 0
    skill_entropy_initial_coef = 0.0
    skill_entropy_final_coef = 0.0
    skill_entropy_anneal_iterations = 0


class Runner:

    def __init__(self, env, device='cpu', training_extension=None):
        from .ppo import PPO

        self.device = device
        self.env = env
        # Optional, opt-in hooks for training pipelines which need to learn
        # auxiliary models from the same rollout.  The ordinary locomotion and
        # high-level runners pass no extension and retain their exact behavior.
        self.training_extension = training_extension

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)
        checkpoint_path = None
        resume_iteration = 0
        # Load weights from checkpoint 
        if RunnerArgs.resume:
            if RunnerArgs.resume_path:
                restored = wandb.restore(
                    RunnerArgs.resume_checkpoint,
                    run_path=RunnerArgs.resume_path,
                )
                checkpoint_path = restored.name
                source = f"W&B run {RunnerArgs.resume_path}"
            else:
                checkpoint_path = os.path.abspath(
                    os.path.expanduser(RunnerArgs.resume_checkpoint)
                )
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(
                        f"Local resume checkpoint does not exist: {checkpoint_path}"
                    )
                source = "local filesystem"
            try:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                # Compatibility with older PyTorch versions that predate the
                # safer weights_only loader argument.
                state_dict = torch.load(checkpoint_path, map_location=self.device)
            if RunnerArgs.resume_policy_only:
                policy_prefixes = ("adaptation_module.", "actor_body.")
                policy_state = {
                    name: value
                    for name, value in state_dict.items()
                    if name.startswith(policy_prefixes)
                }
                if not policy_state:
                    raise ValueError(
                        f"Checkpoint {checkpoint_path} contains no actor policy weights"
                    )
                incompatible = actor_critic.load_state_dict(policy_state, strict=False)
                unexpected = list(incompatible.unexpected_keys)
                invalid_missing = [
                    name
                    for name in incompatible.missing_keys
                    if not (name.startswith("critic_body.") or name == "std")
                ]
                if unexpected or invalid_missing:
                    raise ValueError(
                        "Policy-only checkpoint is incompatible: "
                        f"missing={invalid_missing}, unexpected={unexpected}"
                    )
                print(
                    f"Warm-started actor policy from {checkpoint_path} ({source}); "
                    "critic, exploration noise, and iteration numbering were reset."
                )
            else:
                actor_critic.load_state_dict(state_dict)
                resume_iteration = checkpoint_next_iteration(checkpoint_path)
                print(
                    f"Successfully loaded weights from {checkpoint_path} "
                    f"({source})."
                )
                if resume_iteration:
                    print(f"Continuing iteration numbering at {resume_iteration}.")

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = (
            resume_iteration * self.num_steps_per_env * self.env.num_envs
        )
        self.tot_time = 0
        self.current_learning_iteration = resume_iteration
        self.last_opponent_update_iteration = resume_iteration
        self.last_recording_it = -RunnerArgs.save_video_interval

        self.env.reset()

        if hasattr(self.env, "update_opponent_policy"):
            self.env.update_opponent_policy(self.alg.actor_critic, iteration=0)
            if (
                checkpoint_path
                and not RunnerArgs.resume_path
                and not RunnerArgs.resume_policy_only
            ):
                checkpoint_name = os.path.basename(checkpoint_path)
                opponent_candidates = []
                pool_candidates = []
                if (
                    checkpoint_name.startswith("ac_weights_")
                    and checkpoint_name.endswith(".pt")
                ):
                    suffix = checkpoint_name[len("ac_weights_") :]
                    if suffix != "latest.pt":
                        pool_candidates.append(
                            os.path.join(
                                os.path.dirname(checkpoint_path),
                                f"opponent_pool_{suffix}",
                            )
                        )
                        opponent_candidates.append(
                            os.path.join(
                                os.path.dirname(checkpoint_path),
                                f"opponent_ac_weights_{suffix}",
                            )
                        )
                pool_candidates.append(
                    os.path.join(
                        os.path.dirname(checkpoint_path),
                        "opponent_pool_latest.pt",
                    )
                )
                opponent_candidates.append(
                    os.path.join(
                        os.path.dirname(checkpoint_path),
                        "opponent_ac_weights_latest.pt",
                    )
                )
                pool_path = next(
                    (path for path in pool_candidates if os.path.isfile(path)),
                    pool_candidates[-1],
                )
                opponent_path = next(
                    (path for path in opponent_candidates if os.path.isfile(path)),
                    opponent_candidates[-1],
                )
                if os.path.isfile(pool_path) and hasattr(
                    self.env, "load_opponent_pool_state_dict"
                ):
                    try:
                        pool_state = torch.load(
                            pool_path, map_location=self.device, weights_only=True
                        )
                    except TypeError:
                        pool_state = torch.load(pool_path, map_location=self.device)
                    self.env.load_opponent_pool_state_dict(
                        pool_state, self.alg.actor_critic
                    )
                    print(f"Loaded opponent pool from {pool_path}.")
                elif os.path.isfile(opponent_path) and hasattr(
                    self.env, "load_opponent_policy_state_dict"
                ):
                    try:
                        opponent_state = torch.load(
                            opponent_path, map_location=self.device, weights_only=True
                        )
                    except TypeError:
                        opponent_state = torch.load(opponent_path, map_location=self.device)
                    self.env.load_opponent_policy_state_dict(
                        opponent_state,
                        self.alg.actor_critic,
                        iteration=max(resume_iteration - 1, -1),
                    )
                    print(f"Loaded frozen opponent weights from {opponent_path}.")

        if self.training_extension is not None:
            self.training_extension.bind_runner(self)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        from .ppo import PPO_Args

        trigger_sync = TriggerWandbSyncHook()
        wandb.watch(self.alg.actor_critic, log="all", log_freq=RunnerArgs.log_freq)

        if init_at_random_ep_len:
            if hasattr(self.env, "randomize_episode_lengths"):
                self.env.randomize_episode_lengths()
            else:
                self.env.episode_length_buf[:] = torch.randint_like(self.env.episode_length_buf,
                                                                 high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs

        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            if RunnerArgs.skill_entropy_anneal_iterations > 0:
                entropy_progress = min(
                    max(float(it) / RunnerArgs.skill_entropy_anneal_iterations, 0.0),
                    1.0,
                )
            else:
                entropy_progress = 1.0
            PPO_Args.skill_entropy_coef = (
                (1.0 - entropy_progress) * RunnerArgs.skill_entropy_initial_coef
                + entropy_progress * RunnerArgs.skill_entropy_final_coef
            )
            discrete_skill_direction = bool(
                self.alg.actor_critic.discrete_skill_direction_policy
            )
            high_level_skill_names = (
                ("walk", "dribble", "shoot", "stop")
                if discrete_skill_direction
                else ("walk", "dribble", "shoot")
            )
            high_level_executed_counts = torch.zeros(
                len(high_level_skill_names), dtype=torch.float64
            )
            high_level_requested_counts = torch.zeros(
                len(high_level_skill_names), dtype=torch.float64
            )
            high_level_direction_names = (
                "up",
                "up_right",
                "right",
                "down_right",
                "down",
                "down_left",
                "left",
                "up_left",
            )
            high_level_direction_counts = torch.zeros(8, dtype=torch.float64)
            high_level_direction_count = 0
            high_level_invalid_count = 0.0
            high_level_avoidance_count = 0.0
            high_level_selection_count = 0
            high_level_distance_sums = torch.zeros(2, dtype=torch.float64)
            high_level_distance_count = 0
            local_role_reward_names = (
                "attacker_ball_skill",
                "attacker_command_assist",
                "role_conflict",
                "support_ball_crowding",
            )
            high_level_local_role_sums = {
                name: 0.0 for name in local_role_reward_names
            }
            high_level_local_role_count = 0
            high_level_skill_clip_sums = torch.zeros(3, dtype=torch.float64)
            high_level_skill_clip_counts = torch.zeros(3, dtype=torch.float64)
            high_level_event_names = (
                "high_level_walk_height_failure",
                "high_level_dribble_height_failure",
                "high_level_shoot_height_failure",
                "high_level_stop_height_failure",
                "high_level_goal",
                "high_level_opponent_goal",
                "high_level_accidental_termination",
                "high_level_opponent_accidental_termination",
                "high_level_learning_team_failure",
                "high_level_opponent_team_failure",
            )
            high_level_event_counts = {
                name: 0.0 for name in high_level_event_names
            }
            high_level_event_match_steps = 0
            # These counters are deliberately episode based.  The existing
            # event-rate metrics below describe events per simulator step;
            # keep a separate denominator for the training goal/termination
            # rates requested by the learning-curve recorder.
            high_level_completed_episodes = 0
            high_level_training_goals = 0
            high_level_training_accidental_terminations = 0
            attack_diagnostic_counts = {}
            high_level_near_ball_control_count = 0.0
            high_level_near_ball_control_total = 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    if self.training_extension is not None:
                        self.training_extension.before_env_step(
                            it,
                            obs_before={
                                "obs": obs[:num_train_envs],
                                "privileged_obs": privileged_obs[:num_train_envs],
                                "obs_history": obs_history[:num_train_envs],
                            },
                            actions=actions,
                        )
                    
                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    if self.training_extension is not None:
                        self.training_extension.process_env_step(
                            it,
                            obs=obs_dict,
                            actions=actions,
                            rewards=rewards,
                            dones=dones,
                            infos=infos,
                        )
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], infos)

                    if 'train/episode' in infos:
                        wandb.log(infos['train/episode'], step=it)
                        
                    if 'curriculum' in infos:

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                    for name, values in infos.get("attack_diagnostics", {}).items():
                        attack_diagnostic_counts[name] = attack_diagnostic_counts.get(name, 0.) + float(torch.as_tensor(values).sum())
                    if "high_level_goal" in infos:
                        def match_flags(value):
                            # Logging mixes GPU match events and CPU timeout
                            # flags; aggregate them on one device.
                            flags = torch.as_tensor(value).detach().cpu().bool().reshape(-1)
                            match_count = torch.as_tensor(
                                infos["high_level_goal"]
                            ).numel()
                            if flags.numel() == match_count:
                                return flags
                            if flags.numel() % match_count == 0:
                                return flags.reshape(match_count, -1).any(dim=1)
                            return flags[:match_count]

                        goal_events = match_flags(infos["high_level_goal"])
                        high_level_event_match_steps += int(goal_events.numel())
                        for event_name in high_level_event_names:
                            if event_name in infos:
                                high_level_event_counts[event_name] += float(
                                    match_flags(infos[event_name]).sum().item()
                                )
                        opponent_goal_events = match_flags(
                            infos.get(
                                "high_level_opponent_goal",
                                torch.zeros_like(goal_events),
                            )
                        )
                        accidental_events = match_flags(
                            infos.get(
                                "high_level_accidental_termination",
                                torch.zeros_like(goal_events),
                            )
                        )
                        opponent_accidental_events = match_flags(
                            infos.get(
                                "high_level_opponent_accidental_termination",
                                torch.zeros_like(goal_events),
                            )
                        )
                        timeout_events = match_flags(
                            infos.get(
                                "time_outs", torch.zeros_like(goal_events)
                            )
                        )
                        terminal_events = (
                            goal_events
                            | opponent_goal_events
                            | accidental_events
                            | opponent_accidental_events
                            | timeout_events
                        )
                        high_level_completed_episodes += int(
                            terminal_events.sum().item()
                        )
                        high_level_training_goals += int(goal_events.sum().item())
                        high_level_training_accidental_terminations += int(
                            accidental_events.sum().item()
                        )

                    if 'high_level_skill_ids' in infos:
                        executed = torch.as_tensor(infos['high_level_skill_ids']).long()
                        requested = torch.as_tensor(
                            infos.get('high_level_requested_skill_ids', executed)
                        ).long()
                        invalid = torch.as_tensor(
                            infos.get('high_level_invalid_skill_mask', torch.zeros_like(executed))
                        ).bool()
                        avoidance = torch.as_tensor(
                            infos.get(
                                'high_level_collision_avoidance_mask',
                                torch.zeros_like(executed),
                            )
                        ).bool()
                        # Shared self-play reports [match, both teams], while
                        # PPO owns only the first team. Keep diagnostics from
                        # accidentally counting the frozen opponent.
                        team_size = getattr(self.env, 'team_size', None)
                        if executed.ndim == 2 and team_size is not None:
                            team_size = int(team_size)
                            executed = executed[:, :team_size]
                            requested = requested[:, :team_size]
                            invalid = invalid[:, :team_size]
                            avoidance = avoidance[:, :team_size]
                        else:
                            executed = executed[:num_train_envs]
                            requested = requested[:num_train_envs]
                            invalid = invalid[:num_train_envs]
                            avoidance = avoidance[:num_train_envs]
                        high_level_executed_counts += torch.bincount(
                            executed.reshape(-1).cpu(),
                            minlength=len(high_level_skill_names),
                        )[: len(high_level_skill_names)]
                        high_level_requested_counts += torch.bincount(
                            requested.reshape(-1).cpu(),
                            minlength=len(high_level_skill_names),
                        )[: len(high_level_skill_names)]
                        high_level_invalid_count += float(invalid.sum().item())
                        high_level_avoidance_count += float(avoidance.sum().item())
                        high_level_selection_count += int(executed.numel())

                        if "high_level_requested_direction_ids" in infos:
                            directions = torch.as_tensor(
                                infos["high_level_requested_direction_ids"]
                            ).long()
                            if directions.ndim == 2 and team_size is not None:
                                directions = directions[:, : int(team_size)]
                            else:
                                directions = directions[:num_train_envs]
                            active_directions = directions[directions >= 0]
                            if active_directions.numel() > 0:
                                high_level_direction_counts += torch.bincount(
                                    active_directions.reshape(-1).cpu(),
                                    minlength=8,
                                )[:8]
                                high_level_direction_count += int(
                                    active_directions.numel()
                                )

                        # SharedPolicySelfPlayWrapper adds these local terms
                        # after the cooperative match reward. Keep them
                        # visible in W&B so the actual PPO objective is
                        # auditable rather than inferred from raw episode sums.
                        for name in local_role_reward_names:
                            key = f"high_level_local_{name}_rewards"
                            if key not in infos:
                                continue
                            component = torch.as_tensor(infos[key]).float()
                            if component.ndim == 2 and team_size is not None:
                                component = component[:num_train_envs, : int(team_size)]
                            else:
                                component = component[:num_train_envs]
                            high_level_local_role_sums[name] += float(component.sum().item())
                        if "high_level_local_role_rewards" in infos:
                            local = torch.as_tensor(
                                infos["high_level_local_role_rewards"]
                            ).float()
                            if local.ndim == 2 and team_size is not None:
                                local = local[:num_train_envs, : int(team_size)]
                            else:
                                local = local[:num_train_envs]
                            high_level_local_role_count += int(local.numel())

                        if "low_level_action_clip_fraction" in infos:
                            clip = torch.as_tensor(
                                infos["low_level_action_clip_fraction"]
                            ).float()
                            if clip.ndim == 2 and team_size is not None:
                                clip = clip[:num_train_envs, : int(team_size)]
                            else:
                                clip = clip[:num_train_envs]
                            executed_for_clip = executed
                            if executed_for_clip.ndim == 1:
                                executed_for_clip = executed_for_clip.unsqueeze(-1)
                            for skill_id in range(3):
                                skill_mask = executed_for_clip == skill_id
                                if clip.shape == skill_mask.shape and bool(skill_mask.any()):
                                    high_level_skill_clip_sums[skill_id] += float(
                                        clip[skill_mask].sum().item()
                                    )
                                    high_level_skill_clip_counts[skill_id] += float(
                                        skill_mask.sum().item()
                                    )

                    if 'high_level_robot_ball_distances' in infos:
                        distances = torch.as_tensor(
                            infos['high_level_robot_ball_distances']
                        )[:num_train_envs]
                        if distances.ndim == 2 and distances.shape[1] >= 1:
                            logged_robots = min(2, distances.shape[1])
                            high_level_distance_sums[:logged_robots] += (
                                distances[:, :logged_robots].double().sum(dim=0).cpu()
                            )
                            high_level_distance_count += int(distances.shape[0])
                            learning_team_size = min(
                                int(getattr(self.env, "team_size", logged_robots)),
                                int(distances.shape[1]),
                            )
                            control_distance = float(
                                getattr(
                                    self.env.cfg.rewards,
                                    "high_level_dribble_control_distance",
                                    0.8,
                                )
                            )
                            near_control = (
                                distances[:, :learning_team_size].amin(dim=1)
                                <= control_distance
                            )
                            high_level_near_ball_control_count += float(
                                near_control.sum().item()
                            )
                            high_level_near_ball_control_total += int(
                                near_control.numel()
                            )

                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

            if self.training_extension is not None:
                self.training_extension.after_rollout(it)
            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses_dict = self.alg.update()
            extension_metrics = {}
            if self.training_extension is not None:
                extension_metrics = dict(
                    self.training_extension.after_policy_update(it) or {}
                )
            if hasattr(self.env, "update_training_curriculum"):
                extension_metrics.update(self.env.update_training_curriculum())
            if (
                RunnerArgs.self_play_update_interval > 0
                and hasattr(self.env, "update_opponent_policy")
                and (it + 1 - self.last_opponent_update_iteration) >= RunnerArgs.self_play_update_interval
                and (not hasattr(self.env, "opponent_update_ready")
                     or self.env.opponent_update_ready())
            ):
                self.env.update_opponent_policy(self.alg.actor_critic, iteration=it + 1)
                self.last_opponent_update_iteration = it + 1
            stop = time.time()
            learn_time = stop - start

            clip_actions = float(self.env.cfg.normalization.clip_actions)
            action_clip_fraction = (
                self.env.actions.abs() >= clip_actions - 1e-6
            ).float().mean().item()
            if self.alg.actor_critic.discrete_skill_direction_policy:
                # This policy has no Gaussian coordinates; keep the historic
                # metric keys present while reporting their actual value.
                policy_std = self.alg.actor_critic.std.detach().new_zeros(1)
            else:
                policy_std = self.alg.actor_critic.std.detach().clamp(
                    min=AC_Args.min_action_std,
                    max=AC_Args.max_action_std,
                )
            if (
                self.alg.actor_critic.hybrid_skill_policy
                and not self.alg.actor_critic.discrete_skill_direction_policy
            ):
                grouped_std = policy_std.reshape(
                    -1, self.alg.actor_critic.skill_action_stride
                )
                policy_std = grouped_std[
                    :, self.alg.actor_critic.num_skill_logits :
                ]

            training_metrics = {
                "time_iter": learn_time,
                # "time_iter": logger.split('epoch'),
                "adaptation_loss": mean_adaptation_module_loss,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "mean_decoder_loss": mean_decoder_loss,
                "mean_decoder_loss_student": mean_decoder_loss_student,
                "mean_decoder_test_loss": mean_decoder_test_loss,
                "mean_decoder_test_loss_student": mean_decoder_test_loss_student,
                "mean_adaptation_module_test_loss": mean_adaptation_module_test_loss,
                "ppo/learning_rate": self.alg.learning_rate,
                "ppo/kl_mean": self.alg.last_kl_mean,
                "policy/skill_entropy": self.alg.last_skill_entropy,
                "policy/skill_entropy_coef": PPO_Args.skill_entropy_coef,
                "policy/action_std_mean": policy_std.mean().item(),
                "policy/action_std_max": policy_std.max().item(),
                "policy/action_mean_abs": self.alg.last_action_mean_abs,
                "policy/action_abs_max": self.alg.last_action_abs_max,
                # This reads the frozen low-level actuator actions exposed by
                # the environment, not PPO's high-level hybrid action. Keep
                # the legacy key for existing dashboards and add an explicit
                # namespace for new runs.
                "policy/action_clip_fraction": action_clip_fraction,
                "low_level/action_clip_fraction_last_step": action_clip_fraction,
            }
            training_metrics.update(extension_metrics)
            training_metrics.update(attack_diagnostic_counts)
            if high_level_selection_count > 0:
                for skill_id, skill_name in enumerate(high_level_skill_names):
                    training_metrics[
                        f"high_level/executed_{skill_name}_fraction"
                    ] = float(high_level_executed_counts[skill_id] / high_level_selection_count)
                    training_metrics[
                        f"high_level/requested_{skill_name}_fraction"
                    ] = float(high_level_requested_counts[skill_id] / high_level_selection_count)
                training_metrics["high_level/invalid_request_fraction"] = (
                    high_level_invalid_count / high_level_selection_count
                )
                training_metrics["high_level/collision_avoidance_override_fraction"] = (
                    high_level_avoidance_count / high_level_selection_count
                )
            if high_level_direction_count > 0:
                for direction_id, direction_name in enumerate(
                    high_level_direction_names
                ):
                    training_metrics[
                        f"high_level/requested_direction_{direction_name}_fraction"
                    ] = float(
                        high_level_direction_counts[direction_id]
                        / high_level_direction_count
                    )
            if high_level_distance_count > 0:
                logged_robots = min(2, int(getattr(self.env, "num_robots", 1)))
                for robot_idx in range(logged_robots):
                    training_metrics[f"high_level/robot{robot_idx}_ball_distance"] = float(
                        high_level_distance_sums[robot_idx] / high_level_distance_count
                    )
            if high_level_local_role_count > 0:
                for name, total in high_level_local_role_sums.items():
                    training_metrics[f"high_level/local_{name}_mean"] = (
                        total / high_level_local_role_count
                    )
            for skill_id, skill_name in enumerate(("walk", "dribble", "shoot")):
                count = high_level_skill_clip_counts[skill_id].item()
                if count > 0:
                    training_metrics[
                        f"high_level/{skill_name}_low_level_action_clip_fraction"
                    ] = high_level_skill_clip_sums[skill_id].item() / count
            if high_level_event_match_steps > 0:
                for event_name, count in high_level_event_counts.items():
                    metric_name = event_name[len("high_level_") :]
                    training_metrics[f"high_level/{metric_name}_event_rate"] = (
                        count / high_level_event_match_steps
                    )
                    training_metrics[f"high_level/{metric_name}_event_count"] = count
            if high_level_completed_episodes > 0:
                training_metrics["training/goal_rate"] = (
                    high_level_training_goals / high_level_completed_episodes
                )
                training_metrics["training/termination_rate"] = (
                    high_level_training_accidental_terminations
                    / high_level_completed_episodes
                )
            training_metrics["training/completed_episodes"] = (
                high_level_completed_episodes
            )
            training_metrics["training/goals"] = high_level_training_goals
            training_metrics["training/accidental_terminations"] = (
                high_level_training_accidental_terminations
            )
            if high_level_near_ball_control_total > 0:
                training_metrics["high_level/near_ball_control_fraction"] = (
                    high_level_near_ball_control_count
                    / high_level_near_ball_control_total
                )
            snapshot_iteration = int(
                getattr(self.env, "opponent_snapshot_iteration", -1)
            )
            training_metrics["self_play/opponent_snapshot_iteration"] = (
                snapshot_iteration
            )
            training_metrics["self_play/opponent_snapshot_age"] = (
                it - snapshot_iteration if snapshot_iteration >= 0 else -1
            )
            pool_iterations = list(
                getattr(self.env, "opponent_pool_iterations", [])
            )
            training_metrics["self_play/opponent_pool_size"] = len(pool_iterations)
            if pool_iterations:
                assignments = getattr(self.env, "opponent_assignment", None)
                if assignments is not None:
                    iteration_tensor = torch.as_tensor(
                        pool_iterations,
                        device=assignments.device,
                        dtype=torch.float,
                    )
                    selected = iteration_tensor[assignments]
                    training_metrics["self_play/selected_opponent_iteration_mean"] = (
                        selected.mean().item()
                    )
                    training_metrics["self_play/selected_opponent_age_mean"] = (
                        it - selected.mean().item()
                    )
            wandb.log(training_metrics, step=it)


            
            # logger.store_metrics(**mean_adaptation_losses_dict)
            wandb.log(mean_adaptation_losses_dict, step=it)

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs

            wandb.log({"timesteps": self.tot_timesteps, "iterations": it}, step=it)
            trigger_sync()

            if it % RunnerArgs.save_interval == 0:
                print(f"Saving model at iteration {it}")

                path = os.path.abspath(os.path.expanduser(RunnerArgs.checkpoint_dir))
                os.makedirs(path, exist_ok=True)
                print(f"Checkpoint directory: {path}")

                state_dict = self.alg.actor_critic.state_dict()
                checkpoint_paths = [
                    os.path.join(path, f"ac_weights_{it}.pt"),
                    os.path.join(path, "ac_weights_latest.pt"),
                ]
                for checkpoint_path in checkpoint_paths:
                    torch.save(state_dict, checkpoint_path)

                if hasattr(self.env, "opponent_policy_state_dict"):
                    opponent_state_dict = self.env.opponent_policy_state_dict()
                    if opponent_state_dict is not None:
                        opponent_paths = [
                            os.path.join(path, f"opponent_ac_weights_{it}.pt"),
                            os.path.join(path, "opponent_ac_weights_latest.pt"),
                        ]
                        for opponent_path in opponent_paths:
                            torch.save(opponent_state_dict, opponent_path)
                    else:
                        opponent_paths = []
                else:
                    opponent_paths = []

                if hasattr(self.env, "opponent_pool_state_dict"):
                    opponent_pool_state = self.env.opponent_pool_state_dict()
                    if opponent_pool_state is not None:
                        opponent_pool_paths = [
                            os.path.join(path, f"opponent_pool_{it}.pt"),
                            os.path.join(path, "opponent_pool_latest.pt"),
                        ]
                        for opponent_pool_path in opponent_pool_paths:
                            torch.save(opponent_pool_state, opponent_pool_path)
                    else:
                        opponent_pool_paths = []
                else:
                    opponent_pool_paths = []

                adaptation_module = copy.deepcopy(
                    self.alg.actor_critic.adaptation_module
                ).to('cpu')
                traced_adaptation_module = torch.jit.script(adaptation_module)
                adaptation_paths = [
                    os.path.join(path, f"adaptation_module_{it}.jit"),
                    os.path.join(path, "adaptation_module_latest.jit"),
                ]
                for adaptation_path in adaptation_paths:
                    traced_adaptation_module.save(adaptation_path)

                body_model = copy.deepcopy(
                    self.alg.actor_critic.actor_body
                ).to('cpu')
                traced_body_module = torch.jit.script(body_model)
                body_paths = [
                    os.path.join(path, f"body_{it}.jit"),
                    os.path.join(path, "body_latest.jit"),
                ]
                for body_path in body_paths:
                    traced_body_module.save(body_path)

                config_paths = []
                wandb_run_dir = getattr(getattr(wandb, "run", None), "dir", None)
                if wandb_run_dir is not None:
                    wandb_config_path = os.path.join(wandb_run_dir, "config.yaml")
                    if os.path.isfile(wandb_config_path):
                        local_config_path = os.path.join(path, "config.yaml")
                        if os.path.abspath(wandb_config_path) != os.path.abspath(local_config_path):
                            shutil.copy2(wandb_config_path, local_config_path)
                        config_paths.append(local_config_path)

                wandb_base_path = checkpoint_wandb_base_path(
                    path,
                    wandb_run_dir,
                    os.getcwd(),
                )
                artifact_paths = (
                    adaptation_paths
                    + body_paths
                    + checkpoint_paths
                    + opponent_paths
                    + opponent_pool_paths
                    + config_paths
                )
                if self.training_extension is not None:
                    extension_paths = self.training_extension.save_checkpoint(
                        path, it
                    )
                    artifact_paths += list(extension_paths or [])
                for artifact_path in artifact_paths:
                    wandb.save(artifact_path, base_path=wandb_base_path)
                    

            self.current_learning_iteration += num_learning_iterations

        os.makedirs(
            os.path.abspath(os.path.expanduser(RunnerArgs.checkpoint_dir)),
            exist_ok=True,
        )

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            import numpy as np
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            print(video_array.shape)
            # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
            wandb.log({"video": wandb.Video(video_array, fps=1 / self.env.dt)}, step=it)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert
