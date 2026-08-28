"""DribbleBot tasks rebuilt on Isaac Lab's manager-based environment interface."""

from .runtime_compat import patch_urdf_importer_if_needed


# Importing the package after AppLauncher starts Isaac Sim registers all Gym
# tasks.  The compatibility hook is runtime-only and is a no-op in processes
# that import individual simulator-free modules without Kit.
patch_urdf_importer_if_needed()
from .tasks import *  # noqa: E402,F401,F403
