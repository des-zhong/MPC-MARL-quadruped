"""MAPPO with replayed CEM teacher targets from an offline frozen world model."""
from scripts import train_high_level_mpc_replay as replay


def main():
    parser = replay.build_arg_parser()
    parser.description = __doc__
    parser.set_defaults(freeze_world_model=True, world_model_checkpoint=None,
                        mpc_terminal_value='disabled', mpc_warmup_steps=48,
                        mpc_min_replay_size=1, mpc_min_world_model_updates=0,
                        checkpoint_subdir='high_level_frozen_world_model',
                        auto_resume_online_replay=False)
    args = parser.parse_args()
    if not args.world_model_checkpoint:
        parser.error('--world-model-checkpoint is required')
    if args.mpc_terminal_value != 'disabled':
        parser.error('This offline teacher uses no online terminal-value model; use --mpc-terminal-value disabled')
    if args.on_policy_ppo_epochs < 1:
        parser.error('MAPPO requires --on-policy-ppo-epochs >= 1')
    replay.train_robot(args)
    from scripts.train_high_level import successful_training_exit
    successful_training_exit()


if __name__ == '__main__':
    main()
