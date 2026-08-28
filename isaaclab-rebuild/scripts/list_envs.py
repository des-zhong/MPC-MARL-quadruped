"""List DribbleBot tasks registered by the external Isaac Lab extension."""

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--keyword", default="DribbleBot", help="Substring used to filter task ids.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.experience:
    args_cli.experience = os.environ.get("DRIBBLEBOT_EXPERIENCE", "isaacsim.exp.base.python.kit")

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import dribblebot_isaaclab  # noqa: E402, F401


def main() -> None:
    matches = [spec for spec in gym.registry.values() if args_cli.keyword in spec.id]
    if not matches:
        raise RuntimeError(f"No registered tasks contain {args_cli.keyword!r}")
    for spec in sorted(matches, key=lambda item: item.id):
        print(f"{spec.id}: {spec.kwargs.get('env_cfg_entry_point')}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
