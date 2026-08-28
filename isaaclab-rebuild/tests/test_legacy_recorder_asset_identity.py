"""Regression checks for the legacy parity recorder's asset selection."""

from __future__ import annotations

import ast
from pathlib import Path


RECORDER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record_isaacgym_rollout.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} was not found in {RECORDER_PATH}")


def test_legacy_recorder_explicitly_selects_as2() -> None:
    """Do not let ``config_as2`` silently retain the base config's Go1 adapter."""

    tree = ast.parse(RECORDER_PATH.read_text(encoding="utf-8"), filename=str(RECORDER_PATH))
    configure = _function(tree, "_configure_legacy_env")

    assignments = [
        node
        for node in ast.walk(configure)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "name"
        and isinstance(node.targets[0].value, ast.Attribute)
        and node.targets[0].value.attr == "robot"
        and isinstance(node.targets[0].value.value, ast.Name)
        and node.targets[0].value.value.id == "Cfg"
    ]

    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant)
    assert assignments[0].value.value == "as2"
