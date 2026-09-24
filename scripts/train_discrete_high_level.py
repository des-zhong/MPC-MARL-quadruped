"""Train the categorical four-skill/eight-direction AS2 coordinator.

This is a thin entry point over :mod:`scripts.train_high_level` so all of the
project's checkpoint loading, self-play, telemetry, and simulator setup stay
identical to the hybrid coordinator.  The discrete action mode is enabled by
default and can still be overridden only by editing the shared training code.
"""

from scripts.train_high_level import build_arg_parser, train_robot


def parse_args():
    parser = build_arg_parser()
    parser.description = (
        "Train an AS2 discrete high-level coordinator with four skills and "
        "eight-way directions."
    )
    parser.set_defaults(discrete_skill_direction=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Keep this entry point unambiguously discrete even if a caller supplied a
    # shared parser option intended for the legacy hybrid trainer.
    args.discrete_skill_direction = True
    train_robot(args)
    import wandb
    wandb.finish()
    from scripts.train_high_level import successful_training_exit
    successful_training_exit()
