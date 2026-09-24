"""Resolve online value-target semantics and build comparable evaluation targets."""
from pathlib import Path

import numpy as np

from quadruped.world_model.config import load_config


def resolve_value_target_settings(checkpoint_path):
    """Read the online run settings, not the generic defaults in old value payloads."""
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser().resolve()
        parents = list(path.parents)
        parents.sort(key=lambda parent: parent.name != 'files')
        for parent in parents:
            candidate = parent / 'config.yaml'
            if not candidate.is_file():
                continue
            payload = load_config(candidate)
            online = payload.get('online_mpc', {})
            online = online.get('value', online)
            if 'terminal_bootstrap_on_timeout' in online:
                return {
                    'bootstrap_on_timeout': bool(online['terminal_bootstrap_on_timeout']),
                    'source': str(candidate),
                    'reward': 'high_level_match_rewards (learning team)',
                }
    return {'bootstrap_on_timeout': None, 'source': None,
            'reward': 'high_level_match_rewards (learning team)'}


def value_reference_returns(rows, gamma, bootstrap_on_timeout=None):
    """Match completed-episode training targets; unfinished recordings have no target.

    Bootstrapped targets use the frozen value model at the captured pre-reset
    terminal state. They follow the training convention, but are not independent
    Monte Carlo ground truth or the original historical replay labels.
    """
    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0, 1]')
    targets = np.full(len(rows), np.nan)
    start = 0
    for end, row in enumerate(rows):
        if not row['done']:
            continue
        continuation = 0.0
        if row.get('timeout', False):
            if bootstrap_on_timeout is None:
                start = end + 1
                continue
            if bootstrap_on_timeout:
                continuation = row.get('next_value_prediction')
                if continuation is None or not np.isfinite(continuation):
                    start = end + 1
                    continue
        for index in range(end, start - 1, -1):
            continuation = float(rows[index]['reward']) + gamma * continuation
            targets[index] = continuation
        start = end + 1
    return targets
