"""Default validation profile with a deterministic rule-based opponent."""

from scripts.play_high_level import parse_args, run


DEFAULTS = {
    "high_level_policy_source": "local",
    "high_level_checkpoint": "latest",
    "opponent_rule_based": True,
    "headless": True,
    "collision_avoidance": True,
    "seed": 50,
}


if __name__ == "__main__":
    run(parse_args(defaults=DEFAULTS))
