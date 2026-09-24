"""Run MAPPO in the simulator while visualizing MPC as a diagnostic observer."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import isaacgym  # Must precede torch.
import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib import pyplot as plt

from scripts import play_high_level as playback
from scripts.train_high_level_online_mpc import _wrapper_actions_to_canonical
from quadruped.mpc.config import load_mpc_config
from quadruped.mpc.hybrid_cem import HybridCEMMPC
from quadruped.mpc.objective import MPCObjective
from quadruped.mpc.simulator_controller import TerminalStateCapture, validate_environment_compatibility
from quadruped.mpc.terminal_value import load_value_checkpoint
from quadruped.mpc.value_reference import resolve_value_target_settings, value_reference_returns
from quadruped.mpc.visualization import figure_to_rgb, _field_coordinates
from quadruped.mpc.prediction_dashboard import recorded_returns, plot_plan, plot_positions, plot_values, completed_goal_episodes
from quadruped.world_model.state_adapter import FootballWorldModelStateAdapter
from quadruped.world_model.trainer import load_checkpoint


class PredictionObserver:
    def __init__(self, env, args):
        self.env, self.match, self.args = env, env.env, args
        self.raw = self.match.env
        self.model, payload = load_checkpoint(args.world_model_checkpoint, args.device)
        self.model.eval()
        self.adapter = FootballWorldModelStateAdapter(
            self.match, max_obstacles=0, num_robots=self.model.action_adapter.num_robots,
            event_names=self.model.event_names)
        if self.adapter.schema.to_dict() != self.model.schema.to_dict():
            raise ValueError("MAPPO environment and world model have different state schemas")
        validate_environment_compatibility(self.match, self.model, payload)
        config, _ = load_mpc_config(args.mpc_config, args.mpc_profile)
        if args.mpc_horizon is not None:
            config.horizon = args.mpc_horizon
        config.max_candidate_diagnostics = args.candidate_diagnostics
        # Every diagnostic plan starts fresh; the simulator follows MAPPO.
        config.warm_start = False
        config.objective_mode = args.objective_mode
        config.use_terminal_value = args.objective_mode != "reward_only"
        value = None
        if config.use_terminal_value and not args.terminal_value_checkpoint:
            raise ValueError("Provide --terminal-value-checkpoint for a terminal-value objective")
        if args.terminal_value_checkpoint:
            value, value_payload = load_value_checkpoint(args.terminal_value_checkpoint, args.device)
            if value.schema.to_dict() != self.model.schema.to_dict() or abs(value_payload['gamma'] - config.gamma) > 1e-9:
                raise ValueError("Terminal-value schema/gamma differs from the world model/MPC")
            value.eval()
        config.validate()
        self.value_target_settings = resolve_value_target_settings(args.terminal_value_checkpoint)
        self.value = value
        self.config = config
        objective = MPCObjective(self.model.schema, self.model.action_adapter,
                                 self.model.event_names, config,
                                 controlled_robot_count=env.team_size,
                                 terminal_value=value.predict_with_uncertainty if config.use_terminal_value else None)
        self.planner = HybridCEMMPC(self.model, self.adapter, self.model.action_adapter, objective, config)
        self.capture = TerminalStateCapture(self.match, self.adapter)
        self.output = Path(args.video)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.camera_cache = tempfile.TemporaryDirectory(prefix='mappo_camera_')
        self.dt = self.raw.dt * self.match.control_interval
        self.rows = []
        self.snapshots = []

    @torch.no_grad()
    def before_step(self, action):
        self.capture.clear()
        self.state = self.adapter.extract_state(self.match)['tensor'].clone()
        # Preview caches the exact opponent action subsequently consumed by env.step.
        opponent = self.env.preview_opponent_actions().clone()
        batch, team = self.env.match_count, self.env.team_size
        own = _wrapper_actions_to_canonical(action.view(batch, team, -1), self.match, self.model.action_adapter)
        other = _wrapper_actions_to_canonical(opponent, self.match, self.model.action_adapter)
        self.requested = torch.cat((own, other), dim=-1)
        reference = self.requested[:, None].expand(-1, self.config.horizon, -1).clone()
        fixed_mask = torch.zeros(batch, 2 * team, dtype=torch.bool, device=action.device)
        fixed_mask[:, team:] = True
        devices = [torch.device(self.args.device).index or 0] if action.is_cuda else []
        # Diagnostic sampling must not perturb later MAPPO/opponent/simulator RNG.
        with torch.random.fork_rng(devices=devices):
            self.plan = self.planner.plan(self.state, fixed_action_sequence=reference,
                                          fixed_robot_mask=fixed_mask,
                                          reference_action_sequence=reference)
        self.value_prediction = float(self.value.predict(self.state)[0]) if self.value else None
        target, distance = self.raw._get_high_level_camera_pose()
        camera = self.raw.render('rgb_array', target_loc=target,
                                 cam_distance=[x * self.args.camera_scale for x in distance])[..., :3]
        # Isaac's overhead camera has field +x pointing down. Rotate it to the right.
        camera = np.rot90(camera)
        height = min(camera.shape[0], round(camera.shape[1] * 5 / 8))
        top = (camera.shape[0] - height) // 2
        imageio.imwrite(str(Path(self.camera_cache.name) / f'{len(self.rows):06d}.png'),
                        camera[top:top+height])

    @torch.no_grad()
    def after_step(self, done, info):
        live = self.adapter.extract_state(self.match)['tensor']
        actual = torch.where(self.capture.valid[:, None], self.capture.states, live)
        executed = self.model.action_adapter.pack(
            torch.as_tensor(info['high_level_skill_ids'], device=live.device).long(),
            torch.as_tensor(info['high_level_commands'], device=live.device).float())
        step = len(self.rows)
        # Condition the one-step model check on commands actually executed, including fallback.
        predicted = self.model.predict_next(self.state, executed, deterministic=True)[0]
        _, actual_robots, actual_ball = _field_coordinates(actual[0], self.model.schema)
        _, predicted_robots, predicted_ball = _field_coordinates(predicted[0], self.model.schema)
        self.snapshots.append((self.state[0].cpu(), self.plan.predicted_states[0].cpu()))
        self.rows.append({
            'step': step, 'controller': 'mappo', 'mpc_action_applied': False,
            'mappo_requested_action': self.requested[0].cpu().tolist(),
            'executed_action': executed[0].cpu().tolist(),
            'mpc_proposed_action': self.plan.first_joint_action[0].cpu().tolist(),
            'mpc_objective': float(self.plan.best_objective[0]),
            'learning_goal': bool(torch.as_tensor(info.get('high_level_goal', False)).reshape(-1)[0]),
            'done': bool(done.reshape(self.env.match_count, -1)[0].any()),
            'timeout': bool(torch.as_tensor(info['time_outs']).reshape(self.env.match_count, -1)[0].any()),
            'reward': float(torch.as_tensor(info['high_level_match_rewards']).reshape(self.env.match_count, -1)[0, 0]),
            'value_prediction': self.value_prediction,
            'next_value_prediction': float(self.value.predict(actual)[0]) if self.value and self.value_target_settings['bootstrap_on_timeout'] is True and bool(torch.as_tensor(info['time_outs']).any()) else None,
            'actual_positions': np.concatenate((actual_ball[None], actual_robots[:self.env.team_size])).tolist(),
            'predicted_positions': np.concatenate((predicted_ball[None], predicted_robots[:self.env.team_size])).tolist(),
        })

    def close(self):
        self.capture.restore()
        returns = recorded_returns([r['reward'] for r in self.rows],
                                   [r['done'] for r in self.rows], self.config.gamma)
        reference = value_reference_returns(self.rows, self.config.gamma,
                                            self.value_target_settings['bootstrap_on_timeout'])
        for index, row in enumerate(self.rows):
            row['realized_discounted_return'] = float(returns[index])
            row['value_reference_return'] = float(reference[index]) if np.isfinite(reference[index]) else None
        selected = completed_goal_episodes(self.rows)
        videos = []
        try:
            for episode, start, end in selected:
                episode_output = self.output.with_name(f'{self.output.stem}_episode_{episode:03d}.mp4')
                # Isaac graphics can retain the previous episode until the first physics step.
                first_frame = start + int(start > 0 and end-start > 1)
                with imageio.get_writer(str(episode_output), fps=self.args.fps or round(1 / self.dt)) as writer:
                    for index in range(first_frame, end):
                        state, trajectory = self.snapshots[index]
                        camera = imageio.imread(str(Path(self.camera_cache.name) / f'{index:06d}.png'))
                        sim, axis = plt.subplots(figsize=(8, 5), dpi=120)
                        axis.imshow(camera)
                        axis.set_title(f'MAPPO simulation · step {index-start}')
                        axis.axis('off')
                        sim.tight_layout()
                        figures = [sim, plot_plan(self.model.schema, state, trajectory, self.env.team_size, self.dt),
                                   plot_positions(self.rows, index, self.dt, self.env.team_size),
                                   plot_values(self.rows, index, reference, self.dt)]
                        panels = [figure_to_rgb(fig) for fig in figures]
                        frame = np.concatenate((np.concatenate(panels[:2], axis=1),
                                                np.concatenate(panels[2:], axis=1)), axis=0)
                        writer.append_data(frame)
                        if index == first_frame:
                            imageio.imwrite(episode_output.with_suffix('.png'), frame)
                        for fig in figures:
                            plt.close(fig)
                        self.rows[index]['realized_discounted_return'] = float(returns[index])
                videos.append(str(episode_output))
                print(f'Saved goal episode {episode}: {episode_output}', flush=True)
            if not selected:
                print('No learning-team goals; skipped video generation.', flush=True)
        finally:
            self.camera_cache.cleanup()
        self.output.with_suffix('.json').write_text(json.dumps({
            'controller': 'mappo', 'mpc_action_applied': False,
            'videos': videos,
            'completed_episodes': sum(row['done'] for row in self.rows),
            'saved_goal_episodes': len(videos),
            'seed': self.args.seed,
            'random_learning_start': self.args.random_learning_start,
            'high_level_policy_dir': self.args.high_level_policy_dir,
            'world_model_checkpoint': self.args.world_model_checkpoint,
            'opponent_forecast': 'current opponent action held over horizon',
            'camera_timing': 'before action; aligned with MPC planning state',
            'value_comparison': 'V(real state) vs discounted recorded MAPPO return; not a realized MPC counterfactual',
            'gamma': self.config.gamma,
            'mpc_horizon_steps': self.config.horizon,
            'mpc_horizon_seconds': self.config.horizon * self.dt,
            'terminal_value_checkpoint': self.args.terminal_value_checkpoint,
            'value_target_settings': self.value_target_settings,
            'steps': self.rows,
        }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to attempt; save only learning-team goals')
    parser.add_argument('--mpc-config', default='configs/mpc_joint_teams.yaml')
    parser.add_argument('--mpc-profile', default='teacher_high_quality')
    parser.add_argument('--mpc-horizon', type=int, default=None,
                        help='MPC prediction/planning horizon in high-level steps; overrides the profile (typically 0.2 seconds per step)')
    parser.add_argument('--terminal-value-checkpoint')
    parser.add_argument('--objective-mode', choices=('reward_only', 'reward_plus_terminal_value', 'terminal_value_only'), default='reward_only')
    parser.add_argument('--candidate-diagnostics', type=int, default=0)
    parser.add_argument('--camera-scale', type=float, default=0.65)
    extra, remaining = parser.parse_known_args()
    if '-h' in remaining or '--help' in remaining:
        parser.print_help()
    args = playback.parse_args(remaining, defaults={
        'high_level_policy_source': 'local', 'training_environment': True,
        'training_skills': True, 'headless': True, 'num_envs': 1, 'steps': 10000,
        'no_plot': True, 'outcomes_only': True, 'stop_after_episodes': extra.episodes,
        'video': 'outputs/mappo_mpc/predictions.mp4', 'csv': 'outputs/mappo_mpc/metrics.csv',
    })
    vars(args).update(vars(extra))
    if args.mpc_horizon is not None and args.mpc_horizon < 1:
        parser.error('--mpc-horizon must be at least 1')
    if not args.world_model_checkpoint:
        parser.error('--world-model-checkpoint is required')
    if args.no_video or args.first_episode_only or args.world_model_output or args.num_envs != 1:
        parser.error('This visualization requires video, one environment, and no dataset export/first-episode-only mode')
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.camera_scale <= 0 or args.candidate_diagnostics < 0:
        parser.error('camera-scale must be positive; candidate-diagnostics must be nonnegative')
    playback.run(args, observer_factory=PredictionObserver)


if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
