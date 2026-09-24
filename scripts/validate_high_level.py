"""Default validation profile for the learned high-level soccer policy."""

from scripts.play_high_level import parse_args, run_cli


DEFAULTS = {
    "high_level_policy_source": "local",
    "high_level_checkpoint": "latest",
    # The launcher supplies an explicit opponent pool checkpoint.  Keeping the
    # default empty prevents an accidental self-play evaluation when this
    # module is invoked directly.
    "opponent_high_level_checkpoint": None,
    "opponent_pool_checkpoint": None,
    "headless": True,
    "seed": 1,
    "num_robots": 2,
    "training_skills": True,
    "training_environment": True,
    "skill_policy_source": "local",
}


if __name__ == "__main__":
    run_cli(parse_args(defaults=DEFAULTS))
