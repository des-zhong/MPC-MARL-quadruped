"""Train the shared high-level policy with privileged world-model MPC guidance."""

from scripts.train_high_level import build_arg_parser, train_robot


def parse_args():
    parser = build_arg_parser()
    parser.description = (
        "Train an AS2 self-play high-level policy with privileged MPC guidance."
    )
    parser.add_argument(
        "--world-model-checkpoint",
        required=True,
        help="Joint two-team world-model checkpoint used by the MPC teacher.",
    )
    parser.add_argument(
        "--mpc-config",
        default="configs/mpc_joint_teams.yaml",
        help="MPC configuration for joint two-team planning.",
    )
    parser.add_argument("--mpc-profile", default="teacher_training")
    parser.add_argument(
        "--teacher-reward-coefficient",
        type=float,
        default=1.0,
        help="Scale for the dense per-agent MPC action-agreement reward.",
    )
    # Online CEM is substantially more expensive than ordinary PPO rollout.
    parser.set_defaults(num_envs=32, project="as2_high_level_mpc_teacher")
    return parser.parse_args()


if __name__ == "__main__":
    train_robot(parse_args())
