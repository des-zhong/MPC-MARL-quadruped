# MAPPO versus MPC sample efficiency

Run the fixed-benchmark learning-curve evaluator first. It evaluates every
numbered checkpoint against the same opponent suite and writes
`learning_curve.csv` (or the raw `evaluations.csv`) for each method. Then run:

```bash
/home/zhz/anaconda3/envs/legged_env/bin/python scripts/evaluate_sample_efficiency.py \
  --method MAPPO=outputs/mappo_learning_curve \
  --method MPC=outputs/mpc_learning_curve \
  --target-score 55 \
  --consecutive-checkpoints 3 \
  --output-dir outputs/sample_efficiency
```

The metric is expected match score: a win is 1, a draw is 0.5, and a loss is
0. This is measured on the shared benchmark, so it does not include shaped
rewards or auxiliary losses that differ between MAPPO and MPC. A method reaches
the target only when the Wilson 95% lower confidence bound of its score is at
least 55% for three consecutive checkpoints. The result reports the first
checkpoint, PPO iteration, and training environment-transition count that
satisfy that rule. `sample_efficiency.csv` contains one row per method;
`protocol.json` records the metric and stability rule.

A method directory may be passed directly; the script automatically selects
`learning_curve.csv`, falling back to `evaluations.csv`.

When both methods are in one combined `learning_curve.csv`, pass that same file
for both entries using the exact method labels stored in its `method` column.

For a one-command run with explicit high-level and low-level policy paths, use
`evaluate_mappo_mpc_sample_efficiency.bash`:

```bash
./evaluate_mappo_mpc_sample_efficiency.bash \
  /path/to/mappo_policy \
  /path/to/mpc_policy \
  /path/to/walk_skill \
  /path/to/dribble_skill \
  /path/to/shoot_skill \
  outputs/mappo_mpc_efficiency
```

The script also accepts environment overrides such as `DEVICE`, `POLICY_DEVICE`,
`MAX_ITERATION`, `EVAL_STEPS`, `EPISODES_PER_OPPONENT`, `TARGET_SCORE`,
and `CONSECUTIVE_CHECKPOINTS`. Run it with `--help` to show the positional
arguments.
