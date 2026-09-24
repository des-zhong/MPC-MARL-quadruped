"""Summarize first episodes only; run from repository root after diagnostics."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.compare_high_level_learning_curves import summarize_rollout


def main():
    output = Path(__file__).resolve().parent
    source = ROOT / 'outputs/core_diagnosis'
    results = {'matches': {}, 'isolated': {}}
    for path in sorted(source.glob('*.csv')):
        rows = list(csv.DictReader(path.open()))
        first = []
        ended = set()
        for row in rows:
            env = int(row['env_id'])
            if env in ended:
                continue
            first.append(row)
            if int(row['done']):
                ended.add(env)
        durations, alignments, speeds = [], [], []
        near_rows = 0
        for env in range(32):
            episode = [r for r in first if int(r['env_id']) == env]
            for robot in range(2):
                length = 0
                for row in episode:
                    near_rows += float(row[f'robot{robot}_ball_dist']) <= .8
                    if row[f'robot{robot}_executed_skill'] == 'shoot':
                        length += 1
                        command = np.array([float(row[f'robot{robot}_cmd_x']), float(row[f'robot{robot}_cmd_y'])])
                        goal = np.array([4. - float(row['ball_x']), -float(row['ball_y'])])
                        speed = np.linalg.norm(command)
                        speeds.append(speed)
                        alignments.append(np.dot(goal, command) / max(np.linalg.norm(goal) * speed, 1e-9))
                    elif length:
                        durations.append(.2 * length)
                        length = 0
                if length:
                    durations.append(.2 * length)
        summary = summarize_rollout(path, expected_envs=32, first_episode_per_env=True)
        summary.update(shoot_bursts=len(durations), shoot_ticks=len(speeds), near_robot_ticks=near_rows,
                       median_burst_seconds=float(np.median(durations)) if durations else None,
                       one_tick_burst_fraction=float(np.mean(np.array(durations) <= .20001)) if durations else None,
                       mean_shoot_command_speed=float(np.mean(speeds)) if speeds else None,
                       command_below_training_min_fraction=float(np.mean(np.array(speeds) < 1.5)) if speeds else None,
                       command_alignment_below_0_6_fraction=float(np.mean(np.array(alignments) < .6)) if alignments else None)
        probability_path = path.with_suffix('.probabilities.json')
        if probability_path.exists():
            probabilities = json.loads(probability_path.read_text())
            n = sum(r['active_near_rows'] for r in probabilities)
            summary['near_mean_shoot_probability'] = sum(r['shoot_probability_sum'] for r in probabilities) / n if n else None
            summary['near_shoot_argmax_count'] = sum(r['shoot_argmax_count'] for r in probabilities)
        results['matches'][path.stem] = summary
    for path in sorted(source.glob('fixed_*/shoot/summary.json')):
        saved = json.loads(path.read_text())
        rows = list(csv.DictReader(path.with_name('metrics.csv').open()))
        sign = 1. if float(path.parents[1].name.split('_')[1]) > 0 else -1.
        launches = [r for r in rows if sign * float(r['ball_vx_world']) >= .8 and not int(r['done'])]
        results['isolated'][path.parents[1].name] = {
            'criteria': saved['criteria'],
            'first_directed_0_8_mps_seconds': float(launches[0]['time_s']) if launches else None}
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    for name, result in results['matches'].items():
        print(name, {k: result[k] for k in ['episodes', 'goals', 'accidental_terminations', 'shoot_bursts', 'median_burst_seconds']})


if __name__ == '__main__':
    main()
