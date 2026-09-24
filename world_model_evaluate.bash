python scripts/evaluate_world_model.py \
    --checkpoint wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay/world_model_online_800.pt \
    --value-checkpoint wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay/terminal_value_online_800.pt \
    --dataset data/world_model_ppo_eval_20260916_192612/checkpoint_latest \
    --horizons 1 2 4 10 --device cuda:0 "$@"
