# Terminal-value-augmented MPC

The learned dynamics/reward ensemble remains unchanged. `TerminalValueModel` is
a separate centralized MLP over the same global state schema and the same state
normalization. Its target is a Monte-Carlo return computed from ordered real
simulator episode shards. It estimates continuation under the behavior mixture
in its replay data; it is not claimed to be `V*`.

True terminal episodes use zero continuation beyond their final reward. Until a
reliable bootstrap critic exists, truncated episodes exclude their configurable
final suffix from value fitting. Earlier rows retain finite recorded returns.
No imagined world-model reward is used as a value target.

## Commands

Compute value targets:

```bash
python scripts/compute_terminal_value_targets.py --config configs/terminal_value.yaml --dataset data/mpc_teacher --output data/terminal_value
```

Train and evaluate:

```bash
python scripts/train_terminal_value.py --config configs/terminal_value.yaml --dataset data/terminal_value --world-model-checkpoint checkpoints/world_model_as2/best.pt --output checkpoints/terminal_value
python scripts/evaluate_terminal_value.py --checkpoint checkpoints/terminal_value/best.pt --dataset data/terminal_value --split test
```

Reward-only and value-augmented MPC use the same entry points. Reward-only is an
explicit override; value augmentation loads the checkpoint from the CLI (or
`mpc.terminal_value.checkpoint`):

```bash
python scripts/evaluate_mpc.py --methods mpc --objective-mode reward_only --world-model-checkpoint checkpoints/world_model_as2/best.pt
python scripts/evaluate_mpc.py --methods mpc --objective-mode reward_plus_terminal_value --world-model-checkpoint checkpoints/world_model_as2/best.pt --terminal-value-checkpoint checkpoints/terminal_value/best.pt
python scripts/evaluate_mpc.py --methods mpc --world-model-checkpoint checkpoints/world_model_as2/best.pt --terminal-value-checkpoint checkpoints/terminal_value/best.pt --run-value-ablations
```

Collect new real trajectories and iterate both models:

```bash
python scripts/collect_mpc_teacher_rollouts.py --config configs/mpc.yaml --world-model-checkpoint checkpoints/world_model_as2/best.pt --terminal-value-checkpoint checkpoints/terminal_value/best.pt --output data/mpc_teacher_value --world-model-expansion-output data/world_model_iterations/value_000 --num-episodes 1000
python scripts/run_value_augmented_mpc_iteration.py --config configs/iterative_mpc_world_model.yaml
```

Visualize and decompose errors:

```bash
python scripts/visualize_mpc.py --config configs/mpc.yaml --world-model-checkpoint checkpoints/world_model_as2/best.pt --terminal-value-checkpoint checkpoints/terminal_value/best.pt --profile visualization
python scripts/analyze_terminal_value_rollouts.py --dataset data/mpc_teacher_value --checkpoint checkpoints/terminal_value/best.pt
python scripts/visualize_terminal_value_map.py --checkpoint checkpoints/terminal_value/best.pt --dataset data/terminal_value
python scripts/smoke_test_terminal_value.py --run-simulator
```

If a value checkpoint is absent or fails to load, MPC warns and falls back to
reward-only unless `terminal_value.required: true`. A checkpoint schema or gamma
mismatch is rejected. Optional clipping and uncertainty gating protect against
value exploitation; both are disabled by default so experiments remain explicit.
