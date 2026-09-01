"""YAML configuration loading without simulator coupling."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    with Path(path).open("r") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"World-model config must be a mapping: {path}")
    return payload


def deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
