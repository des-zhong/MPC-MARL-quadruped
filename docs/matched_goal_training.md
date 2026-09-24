# Matched goal-learning experiments

The three launchers now use one argument builder in
`scripts/run_mpc_efficiency_ablation.py`, through `scripts/train_goal_policy.py`.
Their common defaults are 256 environments, seed 42, 48 rollout steps, five PPO
epochs, centralized critics, balanced kickoffs, the soccer curriculum, shooting
options, goalward launch reward, attack-position reward, and the same frozen
walking/dribbling/shooting directories. Self-play uses snapshots every 1200
iterations with probability 0.2 for the latest snapshot. Discrete PPO retains
its categorical skill/direction representation (including Stop); the other two
use the hybrid action representation. These are task-matched configurations,
not identical networks or action spaces.

Start fresh runs from the repository root:

```bash
bash train_high_level.bash --cuda 5
bash train_discrete_high_level.bash --cuda 6
bash train_high_level_mpc_replay.bash --cuda 7
```

Each defaults to 3000 iterations. `--seed`, `--iterations`, `--num-envs`,
`--shoot-policy-dir`, and `--dry-run` are supported. Additional trainer flags
are forwarded, but overriding shared task settings breaks the matched setup.
Do not replace `checkpoints/reproduction/shoot` while these runs are training.
Recorded artifact hashes let evaluation verify that the skill bundles match.

For sequential repeated-seed experiments:

```bash
python scripts/run_mpc_efficiency_ablation.py --condition all \
  --seeds 42 43 44 --iterations 3000 --device cuda:7 --execute
```

## MPC failure and changes

The earlier MPC run initially could not plan because terminal value failed its
prediction-quality gate. Once distillation began, performance declined. That
coincidence is evidence to investigate, not proof that every decline came from
MPC. The previous update guard bounded policy KL on teacher states but did not
check whether the update opposed the real rollout's learning signal.

The new profile:

- Uses four predicted macro steps and reward-only planning while terminal value
  fails validation. The dynamics warmup and model-quality filters still apply.
  Terminal value is added only after its validation and update gates pass.
- Preserves five real PPO epochs for all conditions.
- Keeps teacher labels for at most 256 macro steps, starts updates at 64
  accepted labels, and performs at most one replay update per rollout, with
  a policy KL limit of 0.003.
- Copies up to 2048 real transitions after advantages are computed, before PPO
  clears storage. After PPO, this fixed batch defines the reference score and
  policy distribution. Teacher proposals must preserve the clipped PPO
  surrogate on that batch and obey its cumulative KL budget as well as the
  teacher-batch KL limit. Rejected proposals restore parameters and optimizer
  state. The real batch is excluded from the distillation loss.

The rollout guard uses an empirical advantage estimate; it is not a guarantee
of future goals. Reward-only MPC is short-sighted and model error can still
affect labels. These changes address identified failure paths without treating
MPC activation itself as proof of benefit.

Monitor normal/finishing/possession goal and timeout rates, accepted and rejected
teacher steps, `mpc/terminal_value_active`, and
`mpc_replay/real_rollout_guard_enabled`. Establish efficiency with fixed-opponent
evaluations at equal environment steps and wall time across multiple seeds.
The existing hybrid cross-play evaluator cannot directly mix the discrete
12-output policy with hybrid 6-output opponents; do not bypass its compatibility
checks to compare those representations. Use a common rule-based opponent or
an evaluator that explicitly supports each policy's observation/action layout.

Verification: 75 regression tests passed, including a real replay optimizer
update with the rollout guard enabled, rejection of a small harmful proposal,
terminal-value fallback/warmup behavior, and shared argument consistency.
The GPU smoke tests also exposed mixed CPU/GPU event flags in the training
logger; event aggregation now moves those flags to a common device.

## Extra-compute MPC preset

MPC starts from `checkpoints/offline_teacher/best.pt` with a 192-decision-step
warmup (four 48-step rollouts), instead of 2400. The model matches the four-robot
shooting-option schema, action bounds, skill fingerprint and 10-step control
interval. Online dynamics updates and live quality/replay-size gates remain
enabled. This avoids learning dynamics from scratch; simulator startup and
terminal-value learning still take time. Override the checkpoint using
`--world-model-checkpoint PATH` on the launcher.

The preset now stops after 250 consecutive rollouts without a teacher update
after warmup. It logs the skip reason every 50 idle rollouts and saves recovery
actor-critic weights before stopping. The recovery file excludes optimizer and
replay state. Use `--mpc-max-idle-rollouts 0` for warnings without stopping.
This detects teacher starvation; it does not guarantee better football results.

Online return normalization now rescales each value-network output head to
preserve its unnormalized predictions when the mean and standard deviation
change. This corrects prediction jumps introduced by the previous scaling-only
update. It preserves predictions at rescaling time, not future learning quality.

```bash
bash train_high_level_mpc_replay.bash --cuda 7 --mpc-performance
```

This preset queries up to 256 matches rather than 128, searches 128 candidates
rather than 64, and uses three CEM refinement rounds rather than two. Its
teacher ramp lasts 2400 decision steps (50 rollouts) rather than 40000. It
uses between 64 and 256 distinct accepted replay samples per update, depending
on availability, so 101 fresh labels no longer cause a skipped update.
The real-rollout guard and model-quality filters remain enabled. The W&B
project is `as2_efficiency_mpc_extra_compute` to identify this configuration.

This is a performance experiment with a larger planning compute budget, not
an established improvement. It can improve search coverage but also increases
wall time and exposure to model error. Report the extra compute, use the same
fixed evaluation protocol, and compare repeated seeds before claiming a gain.
