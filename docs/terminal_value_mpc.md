# Terminal-value model utilities

The terminal-value model is an optional continuation-value term in MPC. The
recommended workflow trains an ensemble on real states plus the imagined
terminal states recorded by MPC, using the realized continuation return as the
target. Ranking metrics and ensemble disagreement are used during validation
and planning.

These utilities remain available for offline experiments:

```bash
./train_world_model_pipeline.bash all --wandb-mode offline

# Or run only the final value stage after MPC teacher data exists:
./train_world_model_pipeline.bash terminal_final
./train_world_model_pipeline.bash validate --candidate-ranking
```
