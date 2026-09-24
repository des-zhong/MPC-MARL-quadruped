"""Compact world-model diagnostics for recorded MAPPO episodes."""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .visualization import _field_coordinates


def recorded_returns(rewards, dones, gamma):
    """Discounted observed return, stopping at every reset and recording end."""
    result = np.zeros(len(rewards), dtype=float)
    suffix = 0.0
    for index in reversed(range(len(rewards))):
        if dones[index]:
            suffix = 0.0
        suffix = float(rewards[index]) + gamma * suffix
        result[index] = suffix
    return result


def episode_slice(rows, index):
    start = index
    while start > 0 and not rows[start - 1]['done']:
        start -= 1
    end = index + 1
    while end < len(rows) and not rows[end - 1]['done']:
        end += 1
    return start, end


def plot_plan(schema, state, trajectory, team_size, dt):
    field, robots, ball = _field_coordinates(trajectory, schema)
    half_x, half_y = field[0, :2]
    figure, axis = plt.subplots(figsize=(8, 5), dpi=120)
    axis.add_patch(Rectangle((-half_x, -half_y), 2 * half_x, 2 * half_y,
                             fill=False, color='0.4'))
    axis.axvline(0, color='0.85', linewidth=1)
    for robot in range(robots.shape[1]):
        color = 'tab:blue' if robot < team_size else 'tab:red'
        points = robots[:, robot]
        axis.plot(*points.T, '.--', color=color, linewidth=2, markersize=4)
        axis.scatter(*points[0], color=color, s=60, zorder=5)
        axis.scatter(*points[-1], color=color, marker='x', s=65, zorder=6)
        axis.annotate(f'{"L" if robot < team_size else "O"}{robot % team_size}',
                      points[0], xytext=(5, 6), textcoords='offset points', fontsize=9)
        if len(points) > 1:
            axis.annotate('', xy=points[-1], xytext=points[-2],
                          arrowprops=dict(arrowstyle='->', color=color, lw=2))
    axis.plot(*ball.T, 'o--', color='darkorange', linewidth=2.5, markersize=4)
    axis.scatter(*ball[0], color='darkorange', edgecolor='black', s=65, zorder=7)
    axis.scatter(*ball[-1], color='darkorange', marker='*', edgecolor='black', s=170, zorder=8)
    axis.legend(handles=[Line2D([], [], color=c, marker='o', label=label)
                         for c, label in [('tab:blue', 'Learning'), ('tab:red', 'Opponent'),
                                          ('darkorange', 'Ball')]], loc='upper left', ncol=3, fontsize=8)
    axis.set(xlim=(-half_x-.3, half_x+.3), ylim=(-half_y-.3, half_y+.3),
             xlabel='Field x (m)', ylabel='Field y (m)', aspect='equal',
             title=f'MPC future trajectory · {(len(ball)-1)*dt:.1f} s horizon')
    axis.text(.99, .01, 'Dashed: prediction   × / ★: horizon endpoint',
              transform=axis.transAxes, ha='right', fontsize=8)
    figure.tight_layout()
    return figure


def plot_positions(rows, index, dt, team_size):
    start, end = episode_slice(rows, index)
    seen = rows[start:index+1]
    actual = np.asarray([row['actual_positions'] for row in seen])
    predicted = np.asarray([row['predicted_positions'] for row in seen])
    times = (np.arange(len(seen)) + 1) * dt
    mse = np.mean(np.square(predicted - actual), axis=-1)
    figure, axis = plt.subplots(figsize=(8, 5), dpi=120)
    for entity, label, color in [(0, 'Ball', 'darkorange')] + [
            (robot+1, f'Robot {robot}', plt.get_cmap('tab10')((2 * robot) % 10))
            for robot in range(team_size)]:
        axis.plot(times, mse[:, entity], color=color, label=label, lw=2)
        axis.scatter(times[-1], mse[-1, entity], color=color, s=24)
    axis.set(xlim=(0, max(dt, (end-start)*dt)), ylim=(0, None),
             title='Position prediction MSE', xlabel='Episode time (s)',
             ylabel='Mean squared position error (m²)')
    axis.legend(loc='best', fontsize=9)
    axis.grid(alpha=.2)
    figure.tight_layout()
    return figure


def plot_values(rows, index, returns, dt):
    start, end = episode_slice(rows, index)
    times = np.arange(end-start) * dt
    values = [row['value_prediction'] for row in rows[start:end]]
    figure, axis = plt.subplots(figsize=(8, 5), dpi=120)
    has_reference = rows[end-1]['done'] and np.isfinite(returns[start:end]).all()
    if has_reference:
        bootstrapped = rows[end-1].get('timeout', False) and rows[end-1].get('next_value_prediction') is not None
        axis.plot(times, returns[start:end], color='black', lw=2,
                  label='Bootstrapped return target' if bootstrapped else 'Discounted episode return')
    else:
        axis.text(.5, .85, 'Reference unavailable: episode unfinished or timeout target unknown',
                  transform=axis.transAxes, ha='center', fontsize=8)
    if values[0] is not None:
        values = np.array(values)*0.1+np.array(returns[start:end])
        axis.plot(times, values, '--', color='tab:purple', lw=2, label='Terminal value prediction')
        axis.scatter(times[index-start], values[index-start], color='tab:purple', s=40)
    else:
        axis.text(.5, .9, 'No terminal-value checkpoint loaded', transform=axis.transAxes, ha='center')
    axis.axvline(times[index-start], color='tab:blue', alpha=.6, lw=1)
    if has_reference:
        axis.scatter(times[index-start], returns[index], color='black', s=40)
    axis.set(title='Terminal-value model vs realized MAPPO return',
             xlabel='Episode time (s)', ylabel='Discounted reward')
    axis.text(.02, .02, 'Evaluated on real states · return computed after recording',
              transform=axis.transAxes, fontsize=8)
    axis.legend(fontsize=9)
    axis.grid(alpha=.2)
    figure.tight_layout()
    return figure


def completed_goal_episodes(rows):
    """Return episode number and exclusive bounds only for completed learner goals."""
    start, episode = 0, 0
    selected = []
    for index, row in enumerate(rows):
        if row['done']:
            if any(item.get('learning_goal', False) for item in rows[start:index+1]):
                selected.append((episode, start, index+1))
            start = index+1
            episode += 1
    return selected
