"""Train the full method with only the terminal continuation value removed.

Short-horizon planner guidance, PPO, fictitious self-play, and planner-to-policy
KL distillation remain enabled. This makes comparison with the full method a
controlled ablation of the terminal value alone.
"""

from scripts.train_high_level_online_mpc import build_arg_parser, train_robot


def parse_args():
    parser = build_arg_parser()
    parser.description = (
        "Train the full planner-guided method without terminal continuation value."
    )
    parser.set_defaults(
        project="as2_ablation_ours_no_terminal",
        checkpoint_subdir="high_level_online_mpc_no_terminal",
        mpc_terminal_value="disabled",
        mpc_kl_coefficient=0.05,
        mpc_guidance_reward_coefficient=1.0,
    )
    args = parser.parse_args()
    if args.mpc_terminal_value != "disabled":
        raise ValueError(
            "The no-terminal ablation requires --mpc-terminal-value=disabled"
        )
    if float(args.mpc_kl_coefficient) <= 0.0:
        raise ValueError(
            "The no-terminal ablation keeps distillation enabled and requires "
            "a positive --mpc-kl-coefficient"
        )
    if float(args.mpc_guidance_reward_coefficient) <= 0.0:
        raise ValueError(
            "The no-terminal ablation requires a positive "
            "--mpc-guidance-reward-coefficient"
        )
    args.mpc_terminal_value = "disabled"
    return args


if __name__ == "__main__":
    train_robot(parse_args())
