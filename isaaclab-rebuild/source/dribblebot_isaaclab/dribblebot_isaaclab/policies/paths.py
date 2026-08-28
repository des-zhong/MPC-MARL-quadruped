"""Resolve trusted low-level policy bundles through one configurable seam."""

from __future__ import annotations

import os
from pathlib import Path


CHECKPOINT_ROOT_ENV = "DRIBBLEBOT_CHECKPOINT_ROOT"


def default_checkpoint_root() -> Path:
    """Return the repository's curated reproduction checkpoint directory."""

    return Path(__file__).resolve().parents[5] / "checkpoints" / "reproduction"


def checkpoint_root() -> Path:
    """Return the configured checkpoint root."""

    override = os.environ.get(CHECKPOINT_ROOT_ENV)
    return Path(override).expanduser().resolve() if override else default_checkpoint_root()


def skill_checkpoint_path(skill: str, filename: str) -> Path:
    """Resolve one checkpoint artifact and fail with an actionable message."""

    path = checkpoint_root() / skill / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"DribbleBot checkpoint not found: {path}. "
            f"Set {CHECKPOINT_ROOT_ENV} to a directory containing walk/, dribble/, and shoot/."
        )
    return path
