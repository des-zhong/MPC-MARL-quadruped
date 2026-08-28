"""Verify that the migration asset adapter resolves the current repository."""

import importlib.util
from pathlib import Path


PATHS_MODULE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
    / "assets"
    / "paths.py"
)
SPEC = importlib.util.spec_from_file_location("dribblebot_asset_paths", PATHS_MODULE)
paths = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(paths)


def test_default_asset_root_contains_as2_and_ball() -> None:
    root = paths.default_asset_root()
    assert (root / "robots/as2/urdf/as2.urdf").is_file()
    assert (root / "objects/ball.urdf").is_file()


def test_asset_path_reports_existing_file() -> None:
    assert paths.asset_path("robots/as2/urdf/as2.urdf").is_file()
