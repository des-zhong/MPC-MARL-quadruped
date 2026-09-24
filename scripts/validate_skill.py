"""Default local body-frame shoot validation profile."""

from scripts.validate_robot_abilities import main


DEFAULTS = {
    "ability": "shoot",
    "skill_policy_source": "local",
    "walk_x": 1.1811809539794922,
    "walk_y": 0.2554563879966736,
    "walk_yaw": 0.3178353011608124,
    "headless": True,
}


if __name__ == "__main__":
    raise SystemExit(main(defaults=DEFAULTS))
