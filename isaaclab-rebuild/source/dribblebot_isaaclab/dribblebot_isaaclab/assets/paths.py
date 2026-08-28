"""Resolve legacy repository assets through one replaceable seam.

The first migration stage deliberately reuses the validated URDF and meshes in
``resources/``. Set ``DRIBBLEBOT_ASSET_ROOT`` to test a converted USD/asset tree
without changing task configuration code.
"""

from __future__ import annotations

import os
from pathlib import Path


ASSET_ROOT_ENV = "DRIBBLEBOT_ASSET_ROOT"


def default_asset_root() -> Path:
    """Return the original repository's ``resources`` directory."""

    return Path(__file__).resolve().parents[5] / "resources"


def asset_root() -> Path:
    """Return the configured asset root without requiring Isaac Sim imports."""

    override = os.environ.get(ASSET_ROOT_ENV)
    return Path(override).expanduser().resolve() if override else default_asset_root()


def asset_path(relative_path: str) -> Path:
    """Resolve an asset and fail with an actionable error when it is missing."""

    path = asset_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"DribbleBot asset not found: {path}. "
            f"Set {ASSET_ROOT_ENV} to the directory containing robots/, objects/, and textures/."
        )
    return path
