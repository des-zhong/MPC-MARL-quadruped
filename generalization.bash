OPPONENT_POOL_DIR_1="${1:-wandb/run-20260920_145323-1s2mw9rb/files/tmp/legged_data/high_level}"
OPPONENT_POOL_DIR_2="${2:-wandb/run-20260920_145325-mn957gzl/files/tmp/legged_data/high_level_discrete}"
OPPONENT_POOL_DIR_3="${3:-wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay}"

python scripts/evaluate_high_level_pool.py \
    --policy wandb/run-20260920_145324-clcesqb3/files/tmp/legged_data/high_level_mpc_replay/ac_weights_latest.pt \
    --opponent-pool-dir \
        "${OPPONENT_POOL_DIR_1}" \
        "${OPPONENT_POOL_DIR_2}" \
        "${OPPONENT_POOL_DIR_3}" \
    --device cuda:2 \
    --policy-device cuda:2 \
    --output outputs/high_level_pool_eval_d \
    --metric win-rate
