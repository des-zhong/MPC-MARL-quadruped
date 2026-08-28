"""Ensure the rebuild root is importable without installing simulator packages."""

import sys
from pathlib import Path


REBUILD_ROOT = Path(__file__).resolve().parents[1]
if str(REBUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(REBUILD_ROOT))
