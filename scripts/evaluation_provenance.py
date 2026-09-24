"""Immutable low-level skill inputs and content identities for evaluation."""
import hashlib
import json
from pathlib import Path
import shutil

import yaml

FILENAMES = {'body': 'body_latest.jit', 'adaptation_module': 'adaptation_module_latest.jit',
             'ac_weights': 'ac_weights_latest.pt', 'config': 'config.yaml'}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def recorded_skills(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    metadata = config.get('skill_policy_metadata', {})
    metadata = metadata.get('value', metadata)
    result = {skill: metadata.get(skill, {}).get('artifacts', {})
              for skill in ('walk', 'dribble', 'shoot')}
    if any(not all(key in artifacts for key in FILENAMES) for artifacts in result.values()):
        raise ValueError(f'{config_path} lacks recorded skill artifacts; provide explicit skill directories')
    return result


def freeze_skills(config_paths, output_root, overrides=None):
    """Use exact recorded artifacts by default; never silently substitute latest."""
    from scripts.playback_utils import resolve_policy_files, resolve_ac_weights_file, find_policy_config_path
    records = [recorded_skills(path) for path in config_paths]
    selected = records[0]
    overrides = overrides or {}
    for skill in selected:
        if overrides.get(skill):
            directory = Path(overrides[skill])
            body, adaptation = resolve_policy_files(directory, 'latest')
            paths = dict(body=body, adaptation_module=adaptation,
                         ac_weights=resolve_ac_weights_file(directory, 'latest'),
                         config=find_policy_config_path(body, search_roots=(directory,)))
            selected[skill] = {key: {'path': str(path), 'sha256': file_hash(path)}
                               for key, path in paths.items()}
        elif any({k: a['sha256'] for k, a in record[skill].items()}
                 != {k: a['sha256'] for k, a in selected[skill].items()}
                 for record in records[1:]):
            raise ValueError(f'Training runs used different {skill} weights; select an explicit common evaluation bundle')
    hashes = {skill: {key: entry['sha256'] for key, entry in artifacts.items()}
              for skill, artifacts in selected.items()}
    root = Path(output_root) / identity(hashes)
    for skill, artifacts in selected.items():
        for key, filename in FILENAMES.items():
            entry = artifacts[key]
            target = root / skill / filename
            if target.is_file() and file_hash(target) == entry['sha256']:
                continue
            source = Path(entry['path'])
            if not source.is_file() or file_hash(source) != entry['sha256']:
                raise ValueError(f'Recorded {skill}/{key} is missing or changed: {source}. '
                                 'Restore that artifact; do not silently substitute latest.')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_hash(target) != entry['sha256']:
                raise ValueError(f'Skill export changed during snapshot: {source}')
    matches = []
    for path in config_paths:
        original = recorded_skills(path)
        matches.append(all(original[s][k]['sha256'] == hashes[s][k]
                           for s in hashes for k in FILENAMES))
    return {skill: root / skill for skill in hashes}, {'hashes': hashes, 'matches_training': matches}
