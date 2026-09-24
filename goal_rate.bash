python scripts/record_training_rates.py \
    --ppo-run wandb/run-20260915_212353-9ujw6ubq \
    --mpc-run wandb/run-20260914_192707-w96oikyf \
    --x-axis timesteps --period 500000 \
    --output-dir outputs/training_rates