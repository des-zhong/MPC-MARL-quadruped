"""Evaluate a frozen shooting checkpoint across the option initiation region."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validated_policy_dir(output_dir='outputs/shooting_option_validation'):
    import hashlib
    report = json.loads((Path(output_dir) / 'summary.json').read_text())
    if not report.get('ready'):
        raise ValueError('Shooting readiness checks did not pass; inspect the validation summary before training the coordinator.')
    directory = Path(report['validated_policy_dir'])
    for name, expected in report['hashes'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Validated shooting snapshot changed: ' + name)
    return str(directory.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shoot-policy-dir', default='tmp/legged_data/shoot_option')
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--output-dir', default='outputs/shooting_option_validation')
    parser.add_argument('--print-policy-dir', action='store_true', help='Print the verified snapshot from a passing readiness report.')
    args = parser.parse_args()
    if args.print_policy_dir:
        print(validated_policy_dir(args.output_dir))
        return 0
    # Explicit override accepts the just-trained generation. Snapshot once so
    # a concurrently saving trainer cannot change weights between scenarios.
    from scripts.playback_utils import resolve_policy_files, resolve_ac_weights_file, find_policy_config_path
    import hashlib
    source = Path(args.shoot_policy_dir).resolve()
    from scripts.train_high_level import newest_complete_local_checkpoint
    checkpoint = newest_complete_local_checkpoint(source) or 'latest'
    body, adaptation = resolve_policy_files(source, checkpoint)
    weights = resolve_ac_weights_file(source, checkpoint)
    config = find_policy_config_path(body, (source, source.parent))
    files = {'body_latest.jit': body, 'adaptation_module_latest.jit': adaptation,
             'ac_weights_latest.pt': weights, 'config.yaml': config}
    out = Path(args.output_dir).resolve()
    hashes = {}
    contents = {}
    for name, path in files.items():
        if path is None:
            raise FileNotFoundError(name)
        data = Path(path).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        contents[name] = data
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    snapshot = out / ('checkpoint_' + str(checkpoint) + '_' + identity)
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (snapshot / name).write_bytes(data)
    environment = dict(os.environ)
    environment.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/dribblebot_torch_extensions')
    results = []
    # Includes lateral and heading errors inside the initiation envelope.
    cases = [(0.45, 0., 0.), (.3, .1, .25), (.6, -.15, -.25), (.5, .15, .35)]
    for seed in args.seeds:
        for index, (x, y, angle) in enumerate(cases):
            case_dir = out / f'seed_{seed}_case_{index}'
            command = [sys.executable, str(ROOT / 'scripts/validate_robot_abilities.py'),
                       '--ability', 'shoot', '--skill-policy-source', 'local',
                       '--shoot-policy-dir', str(snapshot), '--command-frame', 'world',
                       '--fixed-skill-init', '--seed', str(seed), '--ball-x', str(x), '--ball-y', str(y),
                       '--shoot-x', str(3*math.cos(angle)), '--shoot-y', str(3*math.sin(angle)),
                       '--steps', '160', '--headless', '--no-video', '--device', args.device,
                       '--policy-device', args.device, '--output-dir', str(case_dir)]
            case_dir.mkdir(parents=True, exist_ok=True)
            with (case_dir / 'process.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = list(csv.DictReader((case_dir / 'shoot/metrics.csv').open()))
            launched = any(float(r['time_s']) <= 2.4 and not int(r['done']) and
                           float(r['ball_vx_world'])*math.cos(angle) + float(r['ball_vy_world'])*math.sin(angle) >= .8
                           for r in rows)
            stable = not any(int(r['done']) and not int(r.get('timeout', 0)) for r in rows)
            results.append({'seed': seed, 'case': index, 'directed_launch_by_deadline': launched,
                            'stable': stable, 'passed': launched and stable})
            print(results[-1], flush=True)
    rate = sum(r['passed'] for r in results) / len(results)
    report = {'checkpoint': checkpoint, 'hashes': hashes, 'cases': results,
              'validated_policy_dir': str(snapshot),
              'pass_rate': rate, 'ready': rate >= .8,
              'note': 'Readiness threshold is a research criterion, not proof of match success.'}
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print('Pass rate:', rate, 'report:', out / 'summary.json')
    print('Use this frozen directory for both high-level conditions:', snapshot)
    return 0 if report['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
