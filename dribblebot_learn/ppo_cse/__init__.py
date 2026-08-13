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
        from dribblebot_learn.ppo_cse.metrics_caches import SlotCache, DistCache

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
    # Disabled for ordinary locomotion tasks. Competitive wrappers implement
    # ``update_opponent_policy`` and receive a frozen actor snapshot at this
    # interval during self-play training.
    self_play_update_interval = 0


class Runner:

    def __init__(self, env, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      ).to(self.device)
        checkpoint_path = None
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
            actor_critic.load_state_dict(state_dict)
            print(
                f"Successfully loaded weights from {checkpoint_path} "
                f"({source})."
            )

        self.alg = PPO(actor_critic, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        self.alg.init_storage(self.env.num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = -RunnerArgs.save_video_interval

        self.env.reset()

        if hasattr(self.env, "update_opponent_policy"):
            self.env.update_opponent_policy(self.alg.actor_critic, iteration=0)
            if checkpoint_path and not RunnerArgs.resume_path:
                opponent_path = os.path.join(
                    os.path.dirname(checkpoint_path),
                    "opponent_ac_weights_latest.pt",
                )
                if os.path.isfile(opponent_path) and hasattr(
                    self.env, "load_opponent_policy_state_dict"
                ):
                    try:
                        opponent_state = torch.load(
                            opponent_path, map_location=self.device, weights_only=True
                        )
                    except TypeError:
                        opponent_state = torch.load(opponent_path, map_location=self.device)
                    self.env.load_opponent_policy_state_dict(
                        opponent_state, self.alg.actor_critic, iteration=-1
                    )
                    print(f"Loaded frozen opponent weights from {opponent_path}.")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
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
            high_level_executed_counts = torch.zeros(3, dtype=torch.float64)
            high_level_requested_counts = torch.zeros(3, dtype=torch.float64)
            high_level_invalid_count = 0.0
            high_level_selection_count = 0
            high_level_distance_sums = torch.zeros(2, dtype=torch.float64)
            high_level_distance_count = 0
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    
                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
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

                    if 'high_level_skill_ids' in infos:
                        executed = torch.as_tensor(infos['high_level_skill_ids']).long()
                        requested = torch.as_tensor(
                            infos.get('high_level_requested_skill_ids', executed)
                        ).long()
                        invalid = torch.as_tensor(
                            infos.get('high_level_invalid_skill_mask', torch.zeros_like(executed))
                        ).bool()
                        executed = executed[:num_train_envs]
                        requested = requested[:num_train_envs]
                        invalid = invalid[:num_train_envs]
                        high_level_executed_counts += torch.bincount(
                            executed.reshape(-1).cpu(),
                            minlength=3,
                        )[:3]
                        high_level_requested_counts += torch.bincount(
                            requested.reshape(-1).cpu(),
                            minlength=3,
                        )[:3]
                        high_level_invalid_count += float(invalid.sum().item())
                        high_level_selection_count += int(executed.numel())

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

                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses_dict = self.alg.update()
            if (
                RunnerArgs.self_play_update_interval > 0
                and hasattr(self.env, "update_opponent_policy")
                and (it + 1) % RunnerArgs.self_play_update_interval == 0
            ):
                self.env.update_opponent_policy(self.alg.actor_critic, iteration=it + 1)
            stop = time.time()
            learn_time = stop - start

            clip_actions = float(self.env.cfg.normalization.clip_actions)
            action_clip_fraction = (
                self.env.actions.abs() >= clip_actions - 1e-6
            ).float().mean().item()
            policy_std = self.alg.actor_critic.std.detach().clamp(
                min=AC_Args.min_action_std,
                max=AC_Args.max_action_std,
            )

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
                "policy/action_std_mean": policy_std.mean().item(),
                "policy/action_std_max": policy_std.max().item(),
                "policy/action_mean_abs": self.alg.last_action_mean_abs,
                "policy/action_abs_max": self.alg.last_action_abs_max,
                "policy/action_clip_fraction": action_clip_fraction,
            }
            if high_level_selection_count > 0:
                skill_names = ("walk", "dribble", "shoot")
                for skill_id, skill_name in enumerate(skill_names):
                    training_metrics[
                        f"high_level/executed_{skill_name}_fraction"
                    ] = float(high_level_executed_counts[skill_id] / high_level_selection_count)
                    training_metrics[
                        f"high_level/requested_{skill_name}_fraction"
                    ] = float(high_level_requested_counts[skill_id] / high_level_selection_count)
                training_metrics["high_level/invalid_request_fraction"] = (
                    high_level_invalid_count / high_level_selection_count
                )
            if high_level_distance_count > 0:
                logged_robots = min(2, int(getattr(self.env, "num_robots", 1)))
                for robot_idx in range(logged_robots):
                    training_metrics[f"high_level/robot{robot_idx}_ball_distance"] = float(
                        high_level_distance_sums[robot_idx] / high_level_distance_count
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

                working_directory = os.path.abspath(os.getcwd())
                wandb_base_path = (
                    working_directory
                    if os.path.commonpath((working_directory, path)) == working_directory
                    else path
                )
                artifact_paths = (
                    adaptation_paths
                    + body_paths
                    + checkpoint_paths
                    + opponent_paths
                    + config_paths
                )
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
