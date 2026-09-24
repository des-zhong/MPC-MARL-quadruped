"""Football object configurations."""

from .ball import SOCCER_BALL_CFG
from .field import (
    GOAL_VISUAL_CFG,
    GOAL_EAST_CROSSBAR_CFG,
    GOAL_EAST_NORTH_CFG,
    GOAL_EAST_SOUTH_CFG,
    GOAL_WEST_CROSSBAR_CFG,
    GOAL_WEST_NORTH_CFG,
    GOAL_WEST_SOUTH_CFG,
    SOCCER_FIELD_VISUAL_CFG,
)

__all__ = [name for name in globals() if name.endswith("_CFG")]
