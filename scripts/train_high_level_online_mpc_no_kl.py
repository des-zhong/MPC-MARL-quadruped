"""Train online world-model MPC with PPO guidance but no KL distillation.

The world model, terminal continuation value, MPC planner, dense planner
guidance reward, PPO, and fictitious self-play remain enabled.  Only the
MPC-to-policy KL objective is removed.
"""

from scripts.train_high_level_online_mpc import build_arg_parser, train_robot


def parse_args():
    parser = build_arg_parser()
    parser.description = (
        "Train terminal-value MPC with dense PPO guidance and no KL distillation."
    )
    parser.set_defaults(
        project="as2_high_level_online_mpc_no_kl",
        checkpoint_subdir="high_level_online_mpc_no_kl",
        mpc_terminal_value="enabled",
        mpc_kl_coefficient=0.0,
        mpc_guidance_reward_coefficient=1.0,
    )
    args = parser.parse_args()
    if float(args.mpc_kl_coefficient) != 0.0:
        raise ValueError(
            "The no-KL ablation requires --mpc-kl-coefficient=0"
        )
    if args.mpc_terminal_value != "enabled":
        raise ValueError(
            "The no-KL ablation keeps terminal value enabled and requires "
            "--mpc-terminal-value=enabled"
        )
    if float(args.mpc_guidance_reward_coefficient) <= 0.0:
        raise ValueError(
            "The no-KL ablation keeps dense MPC guidance and requires a "
            "positive --mpc-guidance-reward-coefficient"
        )
    args.mpc_kl_coefficient = 0.0
    args.mpc_terminal_value = "enabled"
    return args


if __name__ == "__main__":
    train_robot(parse_args())
