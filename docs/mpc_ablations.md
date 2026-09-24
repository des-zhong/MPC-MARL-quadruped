# MPC uncertainty and terminal-value ablations

Run the two ablations with the same MPC replay training profile used by
`train_high_level_mpc_replay.bash`:

```bash
bash train_mpc_ablations.bash --cuda 7
```

Defaults: seeds 42/43/44, 6000 iterations per run, 256 environments, sequential
execution on one GPU. Walk/dribble/shoot policies stay frozen; the world model
and high-level policy train online. Each run starts from the same world-model
checkpoint, `checkpoints/offline_teacher/best.pt`.

| Condition | Uncertainty in planning | Terminal continuation value |
| --- | --- | --- |
| `no_uncertainty` | Disabled | Enabled |
| `no_terminal_value` | Enabled | Disabled |
| `full` (optional baseline) | Enabled | Enabled |

`no_uncertainty` removes state/return uncertainty penalties, state/return/value
uncertainty rejection thresholds, and terminal-value uncertainty gating.
The ensembles and their uncertainty diagnostics remain intact. This tests the
combined effect of uncertainty-aware planning, not ensemble size.

`no_terminal_value` uses the imagined reward sum alone and disables online
terminal-value training. It preserves uncertainty handling and the world-model
and policy training settings. Episode termination handling is unchanged.

The value-enabled conditions retain the existing value-readiness validation
and reward-only fallback. Check `mpc/terminal_value_active` and
`terminal_value/validation_ready` in W&B: if value never becomes active, those
runs cannot establish its benefit. Search and iteration budgets are matched;
wall-clock cost can differ because the disabled value model is not trained.

Include a matched full-MPC baseline for measuring each component's effect:

```bash
bash train_mpc_ablations.bash --cuda 7 \
  --conditions full no_uncertainty no_terminal_value --seeds 42 43 44
```

Preview commands without training or writing output:

```bash
bash train_mpc_ablations.bash --dry-run --seeds 42
```

Run either ablation separately, or override the training budget/checkpoint:

```bash
bash train_mpc_ablations.bash --conditions no_terminal_value --cuda 4 \
  --seeds 42 --iterations 6000 --num-envs 256 \
  --world-model-checkpoint checkpoints/offline_teacher/best.pt \
  --output-root outputs/mpc_ablations_trial2
```

Checkpoints and `launch.json` commands are saved under
`outputs/mpc_ablations/<condition>/seed_<seed>/`. W&B projects are
`as2_mpc_ablation_<condition>`. Nonempty run directories are rejected to avoid
mixing experiments. Training stops if a run fails. Use `WANDB_MODE=offline`
for local W&B logging. `--mpc-performance` applies the existing extra-compute
preset to all selected conditions; compare runs using the same preset.

For analysis, evaluate each condition against the same opponent pool and
evaluation seeds at matching training iterations. Compare goal/win rates and
returns against `full`, and report variation across training seeds.

## Fast learning-curve evaluation

The Bash variable `MAX_OPPONENT_ITERATION` defaults to 2000 and caps both
opponent and candidate iterations, matching `compare_learning_curves.bash`.
Override it with `MAX_OPPONENT_ITERATION=1200 bash compare_mpc_ablation_learning_curves.bash`.
The forwarded options `--opponent-max-iteration` and `--max-iteration` can
override the two caps independently.

```bash
bash compare_mpc_ablation_learning_curves.bash --cuda 4
```

Pass checkpoint directories as the first three arguments, in this order:
full MPC, without terminal value, without uncertainty.

```bash
bash compare_mpc_ablation_learning_curves.bash \
  /path/to/full_mpc /path/to/no_terminal_value /path/to/no_uncertainty \
  --cuda 4 --opponent-dir /path/to/opponent_pool
```

The named options `--mpc-dir`, `--no-terminal-value-dir`, and
`--no-uncertainty-dir` also remain supported.
Like `evaluate_goal_rate.bash`, the launcher defines editable directory
variables and passes them as named Python arguments. Each positional override
is optional: passing only the MPC directory preserves both ablation defaults.
Remaining options are forwarded through `"$@"`.

This compares `MPC`, `MPC w.o. terminal value`, and `MPC w.o. uncertainty`
using the existing learning-curve evaluation engine. Defaults use the full MPC
run from `compare_learning_curves.bash` and each ablation's `seed_42` directory.
The evaluated policies are the trained high-level actors; this does not run
the MPC teacher online during evaluation.

Only complete numbered checkpoints shared by all three runs are evaluated.
At most eight evenly spaced points are selected, including the first and last.
A shared pool of up to three MAPPO opponents is selected within the common
training range and held fixed across all points. Ten episodes per opponent run
in parallel environments on the selected GPU, with video and plots inside the
simulator disabled. Evaluation stops after each environment's first episode
finishes (1000 steps is a safety cap). Complete cached rollouts are reused when
the benchmark configuration, selected checkpoints, code, and skills match.

With common iterations 0/400/800/1200 this is 36 simulator runs, versus 144
for the same candidates against the old 12-opponent suite. This reduces work,
not the cost of each simulator startup. The smaller pool and 30 episodes per
curve point are intended for quick comparisons; increase the pool/episode
counts for final results. The default compares one training seed per method;
confidence intervals describe evaluation outcomes, not training-seed variance.

```bash
# Preview checkpoint selection and the exact evaluation command.
bash compare_mpc_ablation_learning_curves.bash --dry-run

# A faster preliminary check: first/last checkpoints against one opponent.
bash compare_mpc_ablation_learning_curves.bash --cuda 4 \
  --max-points 2 --opponents 1 --output-dir outputs/mpc_ablation_quick

# Explicit paths and a larger evaluation budget.
bash compare_mpc_ablation_learning_curves.bash --cuda 4 \
  --mpc-dir /path/to/full_mpc \
  --no-terminal-value-dir outputs/mpc_ablations/no_terminal_value/seed_42 \
  --no-uncertainty-dir outputs/mpc_ablations/no_uncertainty/seed_42 \
  --max-points 0 --opponents 6 --episodes-per-opponent 32
```

Outputs go to `outputs/mpc_ablation_learning_curves/`: `learning_curves.png`,
`termination_rates.png`, `learning_curve.csv`, `evaluations.csv`, rollout logs,
and benchmark metadata. The default metric is win rate (wins divided by all
completed episodes, including draws and accidental terminations). Use
`--metric goal-rate` for goals divided by both teams' goals, or
`--metric expected-score` for the existing match-score metric.

If training continues, pin `--max-iteration 1200 --opponent-checkpoints 0,800,1200`
to keep the benchmark stable for repeat evaluations. Changing the selected
checkpoint set or pool invalidates the existing engine's benchmark-wide cache.
Skill compatibility and provenance checks are inherited from the existing
evaluator; explicit `--walk-policy-dir`, `--dribble-policy-dir`, and
`--shoot-policy-dir` overrides apply equally to all three methods.

## Replot saved CSVs as win rate

No simulation is needed:

```bash
python scripts/replot_mpc_ablation_learning_curves.py \
  --input outputs/mpc_ablation_learning_curves/learning_curve.csv \
  --metric win-rate
```

This writes `replot/learning_curves.png` and `replot/learning_curves.pdf` next
to the CSV. Use `--output-dir` to choose another location. `evaluations.csv`
is also accepted and its counts are pooled by method/checkpoint before rates
and Wilson 95% intervals are calculated. All method labels and recorded
checkpoint positions are retained. Use `--x-axis iteration` for iterations
instead of environment steps. Existing evaluation results are unchanged.
