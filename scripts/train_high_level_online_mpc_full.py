"""Train the full terminal-value MPC + policy-distillation method."""

from scripts.train_high_level_online_mpc import build_arg_parser, train_robot


def parse_args():
    parser = build_arg_parser()
    parser.description = (
        "Train terminal-value MPC with dense PPO guidance and KL distillation."
    )
    parser.set_defaults(
        project="as2_ablation_ours",
        checkpoint_subdir="high_level_online_mpc_full",
        mpc_terminal_value="enabled",
        mpc_kl_coefficient=0.05,
        mpc_guidance_reward_coefficient=1.0,
    )
    args = parser.parse_args()
    if float(args.mpc_kl_coefficient) <= 0.0:
        raise ValueError(
            "The full method requires a positive --mpc-kl-coefficient"
        )
    if float(args.mpc_guidance_reward_coefficient) <= 0.0:
        raise ValueError(
            "The full method requires a positive "
            "--mpc-guidance-reward-coefficient"
        )
    args.mpc_terminal_value = "enabled"
    return args


if __name__ == "__main__":
    train_robot(parse_args())
