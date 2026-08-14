# Reproduction checkpoints

This directory contains one curated checkpoint bundle for each component used
by the AS2 hierarchical soccer pipeline. Binary `.pt` and `.jit` files are
stored with Git LFS.

## Policy bundle contract

Each policy directory contains:

- `body_latest.jit`: exported deterministic actor body used for inference;
- `adaptation_module_latest.jit`: exported adaptation/history encoder;
- `ac_weights_latest.pt`: complete actor-critic state dictionary for validation
  metadata and training resume;
- `config.yaml`: run-level W&B configuration defining observation dimensions,
  action clipping, command scales, rewards, and environment settings.

The high-level bundle also contains `opponent_ac_weights_latest.pt`, the frozen
self-play opponent snapshot associated with the coordinator checkpoint. It is
not required for inference, but is needed to resume self-play from the same
opponent rather than initializing the opponent from the current actor.

The world-model bundle contains:

- `best.pt.part-*`: Git LFS parts of the self-contained model ensemble,
  normalizer, state/action schemas, event schema, and training metadata;
- `config.yaml`: collection/training configuration retained for provenance and
  compatible reruns.

Reassemble the exact `best.pt` file after `git lfs pull`:

```bash
./checkpoints/reproduction/assemble_world_model.bash
```

The script verifies the final SHA-256 digest before installing the checkpoint.
The parts are necessary because a single 115 MB LFS upload is unreliable on
slow or time-limited connections.

The curated aliases are pinned to these training snapshots:

| Component | Snapshot |
| --- | ---: |
| Walk skill | iteration 215200 |
| Dribble skill | iteration 92000 |
| Shoot skill | iteration 228000 |
| High-level policy | iteration 6400 |
| World model | best validation checkpoint (epoch 19) |

The PPO `.pt` files contain model weights, but the current trainer does not
save optimizer or learning-rate scheduler state. They support inference and
weight-based fine-tuning/resume, not bit-for-bit continuation of an interrupted
PPO run. Re-training the world model from scratch additionally requires the
collected dataset, which is intentionally excluded from Git because of its
size.

## Directory layout

```text
reproduction/
├── walk/
├── dribble/
├── shoot/
├── high_level/
└── world_model/
```

After cloning, install Git LFS and materialize the binaries:

```bash
git lfs install
git lfs pull
./checkpoints/reproduction/assemble_world_model.bash
```

Use the policy directories directly with `--*-policy-dir`. For example:

```bash
./validate_skill.bash \
  --walk-policy-dir checkpoints/reproduction/walk \
  --dribble-policy-dir checkpoints/reproduction/dribble \
  --shoot-policy-dir checkpoints/reproduction/shoot

./validate_high_level.bash \
  --high-level-policy-dir checkpoints/reproduction/high_level \
  --walk-policy-dir checkpoints/reproduction/walk \
  --dribble-policy-dir checkpoints/reproduction/dribble \
  --shoot-policy-dir checkpoints/reproduction/shoot

./mpc_visualize.bash \
  --world-model-checkpoint checkpoints/reproduction/world_model/best.pt \
  --walk-policy-dir checkpoints/reproduction/walk \
  --dribble-policy-dir checkpoints/reproduction/dribble \
  --shoot-policy-dir checkpoints/reproduction/shoot
```

`manifest.sha256` records the hash of every checkpoint and configuration file
in the bundle. Verify it with:

```bash
cd checkpoints/reproduction
sha256sum --check manifest.sha256
```
