"""Train the MAPPO fictitious-self-play baseline used by the ablation table."""

from scripts.train_high_level import build_arg_parser, train_robot


def parse_args():
    parser = build_arg_parser()
    parser.description = "Train the MAPPO-FSP high-level soccer baseline."
    parser.set_defaults(
        project="as2_ablation_mappo_fsp",
        opponent_pool_size=8,
        opponent_latest_probability=0.5,
    )
    args = parser.parse_args()
    # FSP needs a genuine historical mixture. Keep the numerical values
    # configurable so the shared ablation launcher can apply the same pool
    # settings to FSP and every MPC-derived condition.
    if args.opponent_pool_size < 2:
        parser.error("MAPPO-FSP requires --opponent-pool-size >= 2")
    if args.opponent_latest_probability >= 1.0:
        parser.error(
            "MAPPO-FSP requires --opponent-latest-probability < 1"
        )
    return args


if __name__ == "__main__":
    train_robot(parse_args())
