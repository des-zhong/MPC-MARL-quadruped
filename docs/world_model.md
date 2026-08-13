# Skill-level football world model

This package learns one transition per high-level skill interval. In the current AS2 configuration, a low-level policy update spans four 5 ms physics steps (20 ms), and `macro_action_steps: 10` spans ten low-level updates (0.2 s). Reward is the sum of the existing environment reward over that interval. If termination occurs early, the collector records the terminal state immediately before Isaac Gym's automatic reset and records the actual elapsed low-level step count.

## Repository integration

The simulator is `TwoRobotVelocityTrackingEasyEnv`, a GPU-vectorized Isaac Gym environment. `HighLevelSkillWrapper` selects one frozen low-level policy per robot and accepts 12 values: three skill logits and three raw command values per robot. The canonical dataset action is more direct:

```text
[robot0_skill_id, robot0_param0, robot0_param1, robot0_param2,
 robot1_skill_id, robot1_param0, robot1_param1, robot1_param2]
```

Skill 0 is called `walk` in the repository and is the requested `reposition` skill. Skills 1 and 2 are `dribble` and `shoot`. Bounds are read from `Cfg.env.high_level_*_command_scale`; they are not hard-coded into training data or checkpoints. Shoot yaw and any other zero-range parameters have a zero validity mask. `JointActionAdapter.to_wrapper_action` is the only conversion to wrapper logits and inverse-tanh command inputs.

The fixed team coordinate system attacks along +x. Planar positions are relative to each vectorized environment origin and divided by field half-length/half-width. Absolute yaw uses `[sin(yaw), cos(yaw)]`. Velocities remain in m/s and rad/s and are standardized using the training split.

## State definition

`FootballWorldModelStateAdapter` extracts, for both robots: normalized xyz position (z remains metres), yaw sine/cosine, roll/pitch, world-frame linear and angular velocity, fallen/contact flags, current skill one-hot, previous normalized command and mask, and gait phase as sine/cosine. Ball state includes position, linear/angular velocity, possession/possessor estimates, goal flags, and out-of-bounds state.

Static opponents in this repository are box actors, despite older task language calling them cylinders. Each set element stores normalized xy, two planar half-extents, height, and validity. A DeepSets encoder sums masked element embeddings, making obstacle order irrelevant. Static obstacle and field slots are copied unchanged by every imagined transition and are never output targets.

The exact indices, units, kinds, dynamic/static status, event names, action bounds, field timing, and collection config are written to `metadata.json` and every checkpoint. No model code uses anonymous hard-coded state indices.

Robot-ball contact and possession are distance-derived because Isaac Gym's net-contact tensor does not identify the opposing body. Goal, border, ball-obstacle collision, timeout, and accidental termination labels use the environment's own buffers. A `pass` is labeled when the old possessor executes shoot and possession transfers directly to another robot over that macro transition. New event labels are appended, and readers use the ordered event list in dataset/checkpoint metadata so older datasets remain readable.

The multi-robot entry points accept `--num-robots N` for any `N >= 1`; the scenario creates exactly `N` static obstacles. High-level actions have width `6N`, raw joint actions have width `12N`, and world-model joint actions have width `4N`. World-model and high-level checkpoints retain their robot count, so playback/MPC rejects an explicit incompatible count instead of silently reshaping it.

## Data collection and splitting

Collection reuses match reset randomization and domain randomization configured by `configure_high_level_cfg`. The default behavior mixture combines conditioned random actions with scripted approach, support, dribble, shoot, pass, obstacle, clearance, and failure behaviors. Script geometry is computed in metres. The configured uniform, nominal-Gaussian, boundary-focused, and goal-directed sampling weights are honored, and action persistence is cleared at episode boundaries.

Half of newly reset environments are targeted by default. A targeted reset first performs the normal randomized reset, then places only the relevant actors one short transition from a goal, own goal, sideline crossing, ball/robot obstacle contact, teammate contact, possession change, shot result, or pass. The transition still runs through Isaac Gym and the frozen low-level policies. `minimum_event_counts` requires at least 25 goals, passes, ball-obstacle hits, and out-of-bounds transitions. Configuration-driven collection may add up to `max_extra_episodes` to meet those quotas. An explicit `--num-episodes` is always a hard cap; unmet required quotas are reported after that exact number rather than extending collection. Use `--no-coverage-quota` for short smoke tests where missing rare events are acceptable.

```bash
python3 scripts/collect_world_model_data.py \
  --config configs/world_model_as2.yaml \
  --output data/world_model_as2 \
  --device cuda:0
```

Low-level checkpoint selection is explicit. The default `--skill-policy-source wandb` downloads fresh files from each `--walk-wandb-run`, `--dribble-wandb-run`, and `--shoot-wandb-run` into process-local temporary directories; it does not reuse the repository's persistent W&B cache. For offline use, pass `--skill-policy-source local` together with `--walk-policy-dir`, `--dribble-policy-dir`, and `--shoot-policy-dir`, pointing each one at a directory such as `./wandb/<run>/files/tmp/legged_data`. Each policy carries its source, training-time action clip (the current AS2 runs use walk=1, dribble=10, shoot=1), SHA-256 checksums, and observation-history size into dataset metadata. The collector does not render.

```bash
# Fresh online W&B checkpoints (the default).
python scripts/training_high_level.py --skill-policy-source wandb

# Direct local files; no W&B download and no cache lookup.
python scripts/training_high_level.py \
  --skill-policy-source local \
  --walk-policy-dir ./wandb/<walk-run>/files/tmp/legged_data \
  --dribble-policy-dir ./wandb/<dribble-run>/files/tmp/legged_data \
  --shoot-policy-dir ./wandb/<shoot-run>/files/tmp/legged_data
```

The local directories must contain independent checkpoint files. W&B run folders created during training can contain symlinks into a shared `tmp/legged_data`; if two skill folders resolve to the same underlying file, the loader stops with a diagnostic instead of silently loading the wrong policy.

Each completed episode is one compressed NumPy shard under `episodes/`. It contains ordered transition IDs, state/action/next-state, accumulated reward, termination and truncation separately, elapsed steps, event labels, behavior source, targeted scenario, and termination reason. `joint_action` is the skill and command actually executed after wrapper affordance checks; `requested_joint_action` and downgrade flags are retained for auditing. Atomic shard writes and SHA-256 entries make interrupted collection inspectable. Splits are seeded and made only from whole episodes:

```bash
python3 scripts/inspect_world_model_dataset.py --dataset data/world_model_as2
```

The report checks finiteness, constant features, duplicates, skill imbalance, missing/configured rare-event coverage, targeted-scenario counts, requested-versus-executed action changes, terminal robot falls, source/event distributions, and episode leakage.

## Normalization, model, and losses

Statistics are fit only on the training manifest. Continuous state inputs, dynamic deltas, and optionally rewards are standardized with standard deviations clamped to `1e-6`. Binary flags, categorical IDs/one-hot values, masks, and already normalized skill commands remain untouched.

Each independently initialized member has a learned skill embedding, masked parameter encoder, DeepSets obstacle encoder, residual SiLU MLP, Gaussian state-delta and reward heads, and separate binary, termination, truncation, and event logits. Continuous heads use Gaussian NLL; classification heads use BCE. Feature-group weighting gives ball position/velocity higher default weight. The rare portion of each training epoch samples event types uniformly before sampling transitions, preventing frequent failure/proximity labels from drowning out goals, passes, boundaries, or collisions. Members receive independent bootstrap minibatch resamples.

Scheduled multi-step training starts from real contiguous sequences, never treats imagined transitions as ground truth, stops loss after real terminal transitions, discounts later errors, and linearly schedules teacher forcing. Set `multi_step_loss_weight: 0` to disable it.

```bash
python3 scripts/train_world_model.py \
  --config configs/world_model_as2.yaml \
  --dataset data/world_model_as2 \
  --output checkpoints/world_model_as2 \
  --wandb-project as2_world_model
```

Training logs each epoch to W&B, including the total and component train/validation
losses, gradient norm, validation ensemble/member losses, learning rate, epoch time,
early-stopping state, and best validation loss. The run config records the full world-model config,
dataset sizes/schema, device, mixed-precision setting, and model parameter counts.
At completion, `best.pt`, `final.pt`, and `history.json` are uploaded to the run.
Use `--wandb-mode offline` for a disconnected machine, `--wandb-mode disabled`
to turn logging off, or `--no-wandb-save-checkpoints` to keep large checkpoints local.
For a resumed training job, pass the same `--wandb-id` together with `--resume`.

`latest.pt`, `best.pt`, and `final.pt` contain model/optimizer/scheduler states, all schemas and bounds, normalization, config, metrics, seed, epoch, and Git commit. Resume with `--resume checkpoints/world_model_as2/latest.pt`.

## Evaluation

```bash
python3 scripts/evaluate_world_model.py \
  --checkpoint checkpoints/world_model_as2/best.pt \
  --dataset data/world_model_as2 \
  --split test

python3 scripts/validate_world_model_in_env.py \
  --checkpoint checkpoints/world_model_as2/best.pt \
  --num-trajectories 100 \
  --horizon 10 \
  --device cuda:0
```

Offline evaluation reports state/feature-group/reward errors, NLL, termination and event classification, uncertainty/error correlation, and contiguous rollouts at horizons 1, 3, 5, 10, and 20. It saves metrics plus an error-versus-horizon plot when Matplotlib is installed. Environment validation executes the same random valid open-loop actions in the simulator and model; this is evaluation, not MPC.

## Loading and future MPC use

```python
import torch
from dribblebot.world_model.trainer import load_checkpoint

model, checkpoint = load_checkpoint("checkpoints/world_model_as2/best.pt", "cuda")
result = model.rollout(
    initial_states,       # [batch, state_dim]
    candidate_actions,   # [batch, candidates, horizon, 8]
    deterministic=True,
)
scores = model.evaluate_action_sequences(
    initial_states,
    candidate_actions,
    gamma=0.99,
    uncertainty_penalty=0.1,
)
```

Candidates are flattened into the batch dimension, so there is no Python loop over candidates; only horizon is looped. Results include `[B,C,H+1,D]` states, rewards, done probabilities, event probabilities, and separate state/reward uncertainty. Epistemic variance is disagreement of member means; aleatoric variance is the mean predicted member variance. The model never folds uncertainty into reward; `evaluate_action_sequences` applies an optional external penalty.

## Known limitations

- The root workflow targets the repository's configurable multi-robot AS2 environment. The legacy GO1 entry points live under `scripts/go1_scripts/`.
- Current team sides do not switch. A canonical side-flip transform must be added if side switching is introduced.
- Contact/possession identity is approximate where Isaac Gym exposes only net force, not a contact-pair tensor.
- “Static cylinders” are physically boxes in the present environment; their real box geometry is encoded.
- Targeted reset distributions improve support for rare outcomes but intentionally do not represent their natural match frequency. Use `behavior_source` and `targeted_scenario` when weighting downstream analyses.
- Simulator collection and environment validation require the repository's Python 3.8 Isaac Gym/PyTorch environment and an NVIDIA GPU. Offline training/inference can run on CPU, but defaults target CUDA.
