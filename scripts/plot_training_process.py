#!/usr/bin/env python3
"""Extract local W&B history and save separate paper figures (PDF + PNG).

Example:
    python scripts/plot_training_process.py --window 50
    python scripts/plot_training_process.py --figures attack_milestones skill_usage
    python scripts/plot_training_process.py --mappo-dir wandb/run-MAPPO --x-axis timesteps

Each plot_* function accepts (rows, output_dir, window=50, x_axis='iterations')
where rows is a scalar-history list or {'MPC': rows, 'MAPPO': rows}.
Optional figsize=(5.4, 4.2) controls width and height in inches. Each function
writes one standalone figure plus its raw/smoothed data CSV. Window widths
are training iterations, including when the horizontal axis shows timesteps.
Requires wandb, numpy, matplotlib; no login, network, or simulator is needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from collections.abc import Mapping
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    from .record_training_rates import local_history
except ImportError:
    from record_training_rates import local_history

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'wandb/run-20260918_145325-5iw09e94'
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')


def extract_history(run_dir):
    """Merge scalar history by W&B step; never forward-fill sparse metrics."""
    files = list(Path(run_dir).glob('run-*.wandb'))
    if len(files) != 1:
        raise ValueError(f'Expected exactly one run-*.wandb file in {run_dir}')
    merged = {}
    for record in local_history(files[0]):
        if '_step' not in record:
            continue
        row = merged.setdefault(record['_step'], {})
        row.update({key: value for key, value in record.items()
                    if isinstance(value, (float, int)) and math.isfinite(value)})
    rows = sorted((r for r in merged.values() if 'iterations' in r),
                  key=lambda r: r['iterations'])
    if not rows:
        raise ValueError('No history records with logged training iterations')
    return rows


def write_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _curve(rows, numerator, denominator, window, x_axis):
    """Ratios use sum(numerator)/sum(denominator) in trailing windows."""
    selected = [r for r in rows if numerator in r and x_axis in r and
                (denominator is None or r.get(denominator, 0) > 0)]
    if not selected:
        return None
    iteration = np.array([r['iterations'] for r in selected])
    x = np.array([r[x_axis] for r in selected])
    n = np.array([r[numerator] for r in selected], dtype=float)
    d = np.ones(len(n)) if denominator is None else np.array(
        [r[denominator] for r in selected], dtype=float)
    start = np.searchsorted(iteration, iteration - window, side='right')
    stop = np.arange(1, len(n) + 1)
    ns, ds = np.r_[0., np.cumsum(n)], np.r_[0., np.cumsum(d)]
    return iteration, x, n / d, (ns[stop] - ns[start]) / (ds[stop] - ds[start])


class MissingMetricError(ValueError):
    """A requested figure has no observations in any supplied run."""


def _runs(rows):
    return rows if isinstance(rows, Mapping) else {'MPC': rows}


def _plot(rows, output_dir, name, ylabel, series, window=50,
          x_axis='iterations', limits=None, figsize=(5.4, 4.2), font_size=10.0):
    """Color denotes metric; solid MPC / dashed MAPPO denotes method.

    Single-metric comparisons use separate method colors as well. Each run
    retains its own x coordinates; no interpolation or extrapolation is used.
    """
    if window < 1:
        raise ValueError('window must be at least 1')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = _runs(rows)
    with plt.rc_context({'font.size': font_size, 'axes.labelsize': font_size,
                         'axes.titlesize': font_size, 'xtick.labelsize': font_size,
                         'ytick.labelsize': font_size, 'legend.fontsize': font_size,
                         'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=figsize)
        exported = []
        for method_index, (method, history) in enumerate(histories.items()):
            style = ('-', '--', ':', '-.')[method_index % 4]
            for index, (label, numerator, denominator) in enumerate(series):
                curve = _curve(history, numerator, denominator, window, x_axis)
                if curve is None:
                    print(f'Skipping missing series: {name}: {method}: {label}')
                    continue
                iteration, x, raw, smooth = curve
                display_x = x / 1e6 if x_axis == 'timesteps' else x
                color_index = method_index if len(series) == 1 else index
                color = COLORS[color_index % len(COLORS)]
                legend = f'{method}: {label}' if len(histories) > 1 and method != 'MPC' else label
                ax.plot(display_x, raw, color=color, alpha=.12, linewidth=.6,
                        linestyle=style, rasterized=True)
                ax.plot(display_x, smooth, color=color, linewidth=1.8,
                        linestyle=style, label=legend)
                exported.extend(dict(method=method, series=label, iterations=it,
                                     x=xv, raw=rv, smoothed=sv, numerator=numerator,
                                     denominator=denominator or '')
                                for it, xv, rv, sv in zip(iteration, x, raw, smooth))
        if not exported:
            plt.close(fig)
            raise MissingMetricError(f'No data for figure {name}')
        ax.set_xlabel('Agent transitions (millions)' if x_axis == 'timesteps'
                      else 'Training iteration')
        ax.set_ylabel(textwrap.fill(ylabel, width=28))
        if limits is not None:
            ax.set_ylim(*limits)
        ax.grid(alpha=.2)
        # Dense comparisons put the legend outside the data area.
        if len(ax.lines) // 2 > 4:
            handles = [Line2D([], [], color=COLORS[i % len(COLORS)], label=label)
                       for i, (label, _, _) in enumerate(series)
                       if any(row['series'] == label for row in exported)]
            if len(histories) > 1:
                handles.extend(Line2D([], [], color='black',
                                      linestyle=('-', '--', ':', '-.')[i % 4], label=method)
                               for i, method in enumerate(histories)
                               if any(row['method'] == method for row in exported))
            ax.legend(handles=handles, frameon=False, fontsize=font_size, loc='upper center',
                      bbox_to_anchor=(.5, 1.3), ncol=2)
        else:
            ax.legend(frameon=False, fontsize=font_size)
        fig.tight_layout()
        for suffix in ('pdf', 'png'):
            fig.savefig(output_dir / f'{name}.{suffix}', dpi=300)
        plt.close(fig)
    write_csv(output_dir / f'{name}.csv', exported)
    return output_dir / f'{name}.pdf'


def plot_skill_usage(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    """Executed skill mix shows how the learned strategy changes."""
    return _plot(rows, output_dir, 'skill_usage', 'Fraction of executed skills',
                 [(s.title(), f'high_level/executed_{s}_fraction', None)
                  for s in ('walk', 'dribble', 'shoot')], window, x_axis, (0, 1), **plot_options)


def plot_attack_milestones(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    """Episode-weighted milestone rates; these events overlap."""
    return _plot(rows, output_dir, 'attack_milestones', 'Fraction of completed episodes',
                 [(label, f'attack/episode_{key}', 'attack/episode_count') for label, key in
                  [('Reached ball', 'reached_ball'), ('Requested shot near ball',
                   'shot_requested_near_ball'), ('Goalward launch', 'goalward_launch')]],
                 window, x_axis, (0, 1), **plot_options)


def plot_ball_progress(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    """Initial minus best ball-to-goal distance, averaged over completed episodes."""
    return _plot(rows, output_dir, 'ball_progress', 'Best goalward progress per episode (m)',
                 [('Ball progress', 'attack/episode_goal_progress_m', 'attack/episode_count')],
                 window, x_axis, **plot_options)


def plot_timeout_causes(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    """Disjoint timeout causes, each divided by ALL completed episodes."""
    return _plot(rows, output_dir, 'timeout_causes', 'Fraction of completed episodes',
                 [(label, f'attack/timeout_{key}', 'attack/episode_count') for label, key in
                  [('No contact', 'no_contact'), ('Contact, no shot', 'no_shot'),
                   ('Shot, no launch', 'shot_no_launch'), ('Launch, no goal', 'launch_no_goal')]],
                 window, x_axis, (0, 1), **plot_options)


def plot_ball_distance(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'ball_distance', 'Mean robot-to-ball distance (m)',
                 [(f'Robot {i}', f'high_level/robot{i}_ball_distance', None) for i in (0, 1)],
                 window, x_axis, **plot_options)


def plot_skill_entropy(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'skill_entropy', 'Logged skill entropy (nats)',
                 [('Policy entropy', 'policy/skill_entropy', None)], window, x_axis, **plot_options)


def plot_action_std(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'action_std', 'Continuous action standard deviation',
                 [('Mean', 'policy/action_std_mean', None)], window, x_axis, **plot_options)


def plot_ppo_kl(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'ppo_kl', 'PPO KL divergence',
                 [('PPO update KL', 'ppo/kl_mean', None)], window, x_axis, **plot_options)


def plot_value_loss(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'value_loss', 'PPO value loss',
                 [('Critic loss', 'mean_value_loss', None)], window, x_axis, **plot_options)


def plot_world_model_ball_error(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'world_model_ball_error', 'Error (m)',
                 [('Overall', 'mpc_quality/ball_error_m', None),
                  ('Shot-conditioned', 'mpc_quality/shot_ball_error_m', None)], window, x_axis, **plot_options)


def plot_world_model_velocity_error(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'world_model_velocity_error', 'Error (m/s)',
                 [('Velocity prediction error', 'mpc_quality/shot_velocity_error_mps', None)], window, x_axis, **plot_options)


def plot_teacher_acceptance(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'teacher_acceptance', 'Accepted MPC candidate fraction',
                 [('Teacher acceptance', 'mpc_quality/accepted_fraction', None)], window, x_axis, (0, 1), **plot_options)


def _performed_updates(rows):
    """Do not mistake zero placeholders from rejected/skipped updates for losses."""
    return {method: [r for r in history if r.get('mpc_replay/updates', 0) > 0]
            for method, history in _runs(rows).items()}


def plot_distillation_kl(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(_performed_updates(rows), output_dir, 'distillation_kl',
                 'Categorical teacher–student KL',
                 [('Skill selection', 'mpc_replay/categorical_kl', None)],
                 window, x_axis, **plot_options)


def plot_command_imitation_loss(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    """Bounded mean loss with quality filtering; continuous KL otherwise."""
    return _plot(_performed_updates(rows), output_dir, 'command_imitation_loss',
                 'Continuous command imitation loss',
                 [('Command imitation', 'mpc_replay/continuous_kl', None)],
                 window, x_axis, **plot_options)


def plot_terminal_value_error(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    """Pre-update return prediction MSE / return variance; not explained variance."""
    return _plot(rows, output_dir, 'terminal_value_error', 'Return prediction MSE',
                 [('Terminal value MSE', 'terminal_value/prequential_mse',
                   'terminal_value/prequential_variance')], window, x_axis, **plot_options)


def plot_invalid_requests(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'invalid_requests', 'Invalid skill request fraction',
                 [('Invalid requests', 'high_level/invalid_request_fraction', None)],
                 window, x_axis, (0, 1), **plot_options)


def plot_world_model_robot_error(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    """Mean one-step planar Euclidean error over all encoded robots, in meters."""
    return _plot(rows, output_dir, 'world_model_robot_error',
                 'Robot position prediction error (m)',
                 [('All robots', 'mpc_quality/robot_position_error_m', None)],
                 window, x_axis, **plot_options)


def plot_world_model_robot_errors_by_robot(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    """Robot IDs follow world-model schema order, including opponents."""
    keys = {key for history in _runs(rows).values() for row in history for key in row
            if re.fullmatch(r'mpc_quality/robot_\d+_position_error_m', key)}
    series = [(f'Robot {int(key.split("robot_")[1].split("_")[0])}', key, None)
              for key in sorted(keys, key=lambda k: int(k.split('robot_')[1].split('_')[0]))]
    return _plot(rows, output_dir, 'world_model_robot_errors_by_robot',
                 'Robot position prediction error (m)', series,
                 window, x_axis, **plot_options)


def plot_goal_rate(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'goal_rate', 'Goals / completed episodes',
                 [('Goal rate', 'training/goals', 'training/completed_episodes')],
                 window, x_axis, (0, 1), **plot_options)


def plot_termination_rate(rows, output_dir, window=50, x_axis='iterations', **plot_options):
    return _plot(rows, output_dir, 'termination_rate', 'Accidental terminations / completed episodes',
                 [('Termination rate', 'training/accidental_terminations', 'training/completed_episodes')],
                 window, x_axis, (0, 1), **plot_options)


FIGURES = {name[len('plot_'):]: function for name, function in list(globals().items())
           if name.startswith('plot_') and callable(function)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', '--mpc-dir', dest='run_dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--mappo-dir', type=Path, help='Optional local MAPPO W&B run to overlay')
    parser.add_argument('--figure-width', '--width', dest='figure_width', type=float,
                        default=8.0, help='Figure width in inches (default: 8)')
    parser.add_argument('--figure-height', '--height', dest='figure_height', type=float,
                        default=4.2, help='Figure height in inches (default: 4.2)')
    parser.add_argument('--font-size', type=float, default=10.0,
                        help='Font size in points (default: 10)')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/training_process_5iw09e94')
    parser.add_argument('--window', type=int, default=50, help='Trailing iteration window; 1 disables smoothing')
    parser.add_argument('--x-axis', choices=('iterations', 'timesteps'), default='iterations')
    parser.add_argument('--figures', nargs='+', choices=sorted(FIGURES), default=list(FIGURES))
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    if not all(math.isfinite(v) and v > 0 for v in
               (args.figure_width, args.figure_height, args.font_size)):
        parser.error('--figure-width, --figure-height and --font-size must be finite and positive')
    rows = extract_history(args.run_dir)
    histories = {'MPC': rows}
    sources = {'MPC': str(args.run_dir.resolve())}
    if args.mappo_dir:
        histories['MAPPO'] = extract_history(args.mappo_dir)
        sources['MAPPO'] = str(args.mappo_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'scalar_history.csv', rows)
    if args.mappo_dir:
        write_csv(args.output_dir / 'scalar_history_mappo.csv', histories['MAPPO'])
    generated, skipped = [], {}
    for name in args.figures:
        try:
            print(FIGURES[name](histories, args.output_dir, args.window, args.x_axis,
                               figsize=(args.figure_width, args.figure_height), font_size=args.font_size))
            generated.append(name)
        except MissingMetricError as error:
            skipped[name] = str(error)
            print(f'Skipping {name}: no logged observations; historical errors cannot be reconstructed.')
    metadata = dict(source=str(args.run_dir.resolve()), records=len(rows),
                    first_iteration=rows[0]['iterations'], last_iteration=rows[-1]['iterations'],
                    x_axis=args.x_axis, trailing_iteration_window=args.window, figures=generated,
                    skipped_figures=skipped, sources=sources,
                    figure_size_inches=[args.figure_width, args.figure_height], font_size=args.font_size,
                    runs={method: dict(records=len(history), first_iteration=history[0]['iterations'],
                                       last_iteration=history[-1]['iterations'])
                          for method, history in histories.items()},
                    comparison='Own x coordinates per run; no extrapolation. Solid MPC, dashed MAPPO. Verify equal transition semantics and evaluation conditions.',
                    robot_error='Mean planar Euclidean one-step prediction error in meters; excludes terminal transitions; EMA 0.95; all encoded robots including opponents.',
                    distillation='Only performed replay updates. Continuous metric is bounded mean loss when mpc_quality_filter=true, continuous KL otherwise.',
                    smoothing='(iteration-window, iteration]; ratio of sums for derived ratios, arithmetic mean otherwise',
                    raw_data='Faint curves are logged observations; solid curves are smoothed. No confidence intervals.',
                    missing_data='Missing metrics are omitted, never forward-filled or replaced with zero.',
                    interpretation='One training run per method, not held-out evaluation or multi-seed statistics. Training scenario mixture may change.',
                    units='timesteps counts logged agent transitions, not independent matches.',
                    progress='Initial minus best ball-to-goal distance; not final displacement.',
                    model_errors='Online quality estimates, not fixed held-out test errors.',
                    terminal_value='MSE / target variance, not explained variance; logged estimates already use exponential smoothing.')
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Extracted {len(rows)} iteration records; saved {len(generated)} separate figures; skipped {len(skipped)}.')


if __name__ == '__main__':
    main()
