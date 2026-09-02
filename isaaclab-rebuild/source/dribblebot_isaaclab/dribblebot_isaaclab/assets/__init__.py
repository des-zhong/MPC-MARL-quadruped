"""Robot and object configurations owned by the Isaac Lab rebuild."""

from .objects import *  # noqa: F401, F403
from .robots import AS2_CFG

__all__ = [name for name in globals() if name.endswith("_CFG")]
