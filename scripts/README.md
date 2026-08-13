# Script layout

Top-level entry points target the AS2 robot. The four low-level and coordinator
trainers are consolidated as:

- `training_walking.py`
- `training_dribbling.py`
- `training_shooting.py`
- `training_high_level.py`

For example, start AS2 walking training with:

```bash
python scripts/training_walking.py --headless
```

Dribbling and shooting use different checkpoint directories by default, so
they can train concurrently:

```bash
python scripts/train_dribbling.py --headless
python scripts/train_shooting.py --headless
```

Their checkpoints are written to `tmp/legged_data/dribble/` and
`tmp/legged_data/shoot/`, respectively. For multiple runs of the same skill,
give every process a distinct directory:

```bash
python scripts/train_dribbling.py \
  --checkpoint-dir tmp/legged_data/dribble-experiment-2 \
  --headless
```

New W&B run folders preserve those nested paths. For local collection, point
`--dribble-policy-dir` or `--shoot-policy-dir` at the corresponding nested
folder below `wandb/<run>/files/tmp/legged_data/`.

AS2 playback and world-model tools also live at the top level and use AS2
defaults. Robot-neutral helper code remains top level when it is shared by
those entry points.

Validate walking, dribbling, and shooting in separate simulator processes:

```bash
python scripts/validate_robot_abilities.py \
  --ability all \
  --skill-policy-source local \
  --headless
```

Use `--ability walk`, `--ability dribble`, or `--ability shoot` to isolate one
policy. Each scenario writes a rollout video, metrics CSV, diagnostic plot, and
pass/fail JSON under `outputs/ability_validation/<ability>/`. Add
`--fail-on-threshold` when the result should be reflected in the process exit
code.

GO1 entry points, legacy configuration, and the Unitree actuator-network tools
live in `go1_scripts/`. The GO1 training scripts use the same `training_*`
naming inside that package:

```bash
python scripts/go1_scripts/training_walking.py --headless
```

## Joint-team world model and MPC teacher

The AS2 world-model collector now creates two equal robot teams and records one
global state and joint hybrid action for all robots. `--num-robots` is the
number of robots per team:

```bash
./collect.bash --num-robots 2 --num-episodes 20000
python scripts/train_world_model.py \
  --dataset data/world_model_as2 \
  --output checkpoints/world_model_as2 \
  --num-robots 2
```

After training that model, start self-play PPO with privileged MPC guidance:

```bash
./train_high_level_with_mpc_teacher.bash \
  --world-model-checkpoint checkpoints/world_model_as2/best.pt
```

The student and frozen opponent keep their decentralized shared-policy
observations. MPC sees the joint world-model state, optimizes only the learning
team, holds the frozen opponent's current policy action over its horizon, and
adds a dense per-agent action-agreement reward. MPC is used only during
training and is not required when evaluating the learned policy.
