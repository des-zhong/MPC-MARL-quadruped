# Paired MPC teacher versus student test

This diagnostic tests whether a teacher's first action improves real simulator
return over the student's action. It uses a frozen checkpoint from the run
launched by `train_high_level_mpc_replay.bash`. It does not train or modify any
checkpoint, start a W&B run, or execute MPC for the entire episode.

```bash
bash test_mpc_teacher.bash \
  --run-dir wandb/run-20260917_011907-32aec9oq \
  --checkpoint 800 --device cuda:4 \
  --num-envs 1 --seeds 50 51 52 53 54 \
  --warmup-steps 20 --rollout-steps 150 \
  --output outputs/mpc_teacher_paired_800
```

Use an available GPU. The script runs three subprocesses sequentially per seed;
one environment and five seeds provide five independent paired decisions.
Accepted teacher targets may be a much smaller subset. Use a new output directory for
each experiment. A quick installation check uses `--num-envs 1 --seeds 50
--warmup-steps 4 --rollout-steps 8`; that is not enough to assess teacher quality.

The default is deliberately **one match per simulator process**. The vectorized
pilot passed exact student-repeat checks but unchanged matches still drifted when
other matches took teacher actions, even after isolating reset randomness.
Its long-return effects are therefore inconclusive. Single-match processes remove
this cross-match interference. To investigate vectorized execution explicitly,
pass e.g. `--num-envs 8 --control-matches 1`; a failed unchanged-match check
invalidates that seed batch. Never relax the checks to obtain a favorable result.

## What is held constant

The script loads the numbered actor, online world model, terminal value, and
opponent pool from the same checkpoint directory. The opponent-pool checkpoint
also restores curriculum state. Saved environment and actor settings are
restored, and current low-level skills must match the world model's recorded
fingerprint. Planning uses the current MPC YAML plus the run's saved overrides;
the YAML checksum and resolved planner configuration are recorded. Numbered
checkpoints are required so a running trainer cannot silently replace `latest`.

Each seed uses three fresh simulator processes:

1. **Student:** replay the frozen student for the common warm-up prefix, execute
   its sampled action at the branch point, then continue the frozen student.
2. **Repeat:** independently repeat the complete student trial with the same seed.
3. **Teacher:** reconstruct the same prefix, substitute the selected MPC action
   for the eligible attacker at the branch point, then continue the same frozen
   student. Support actions and active shooting options retain the student's
   request, following the training teacher mask. A shooting request can start a
   persistent option; its duration is part of the action's real consequence.

The opponent continues its frozen feedback policy in every branch. Policy
sampling is reseeded identically at each decision to share random draws between
branches. Reset draws use a separate RNG context so one match's reset cannot
shift another match's observation noise. Opponent policies sample the full batch
before selecting pool assignments, keeping random draw indices stable when other
matches reset. These are evaluation-only changes to random-number coupling;
neither the reward nor the policy is changed. The default `--policy-mode sample` matches training action selection;
`--policy-mode mode` changes only the learning team's selection to deterministic
actions. Opponent sampling remains as in training.

There is no exact PhysX clone/restore in this repository. Seeded reconstruction
is therefore checked, not assumed: the script compares all exposed tensor state
on the raw environment and both wrappers, including joints, low-level histories,
high-level histories, option timers, commands, and the proposed actions. It also
checks the entire repeat trajectory's compact states, requested actions, rewards,
and termination flags. A failed check excludes the seed batch and exits with
code 2 after saving results. Hidden PhysX caches cannot be directly compared;
the repeat is a necessary empirical control, not a proof of exact cloning.
The default absolute state tolerance is `1e-6`; contact-force tensors separately
allow `1e-3` because GPU force reductions showed float32 roundoff even when the
entire repeated state/reward trajectory was identical. Other state checks are
not relaxed. These thresholds are recorded in the experiment manifest.
For vectorized trials, reserved controls and matches with no eligible intervention must
retain identical trajectories through their first terminal transition inside the
teacher branch itself. This detects cross-match interference that a completely
separate student-repeat process would miss.

## Returns and acceptance

Real return is the sum of discounted **match rewards** from the intervention,
including the first terminal transition and excluding all subsequent auto-reset
episodes. The discount matches MPC. No learned value bootstrap is added. A
nonterminated branch at `--rollout-steps` is marked censored, so short tests measure
truncated return. `short_return` also records the real return over the MPC horizon.

The predicted advantage reproduces the training comparison: teacher first action
versus student first action with the same MPC continuation and fixed opponent
forecast. Real branches instead use student feedback after the first action.
This difference is intentional: it tests whether the training acceptance score
transfers to how the student actually behaves.

All eligible teacher proposals are executed, including rejected proposals, to
avoid losing diagnostic information. `accepted_teacher` filters the results using
the training advantage margin, valid/OOD checks, shooting readiness, and world
model readiness recorded at or before the checkpoint iteration. These global
readiness values are historical gates, not freshly estimated model accuracy.
Terminal value defaults to its checkpoint-time active/inactive status.

## Reading the output

- `summary.json`: pairing checks, effect summaries, accepted-target summaries,
  start-kind breakdowns, censoring rates, and predicted/real advantage correlation.
- `pairs.csv`: per-state predicted advantage, actual teacher-minus-student return,
  short-horizon returns, goal/concession/failure outcomes, and repeat differences.
- `experiment.json`: saved configuration, command settings, checkpoint hashes,
  and checkpoint-time quality metrics.
- `seed_*/`: logs, branch results, exposed start-state snapshots, and trajectories.

Positive `accepted_teacher.mean_real_advantage` supports the teacher's utility;
negative values indicate harmful accepted advice under this continuation policy.
Intervals bootstrap whole seed batches to retain within-batch dependence. One
seed has no confidence interval, and a handful of seeds/accepted targets is only
a pilot. Inconclusive replay checks invalidate causal interpretation regardless
of the apparent return difference. This test does not establish whether policy
distillation itself is beneficial; that requires a separate training ablation.

## Separating terminal-value and search effects

Repeat the same seeds and prefix with terminal value disabled:

```bash
bash test_mpc_teacher.bash --checkpoint 800 --device cuda:4 \
  --num-envs 1 --seeds 50 51 52 53 54 \
  --terminal-value disabled --output outputs/mpc_teacher_paired_no_value
```

Then try a larger search budget:

```bash
bash test_mpc_teacher.bash --checkpoint 800 --device cuda:4 \
  --num-envs 1 --seeds 50 51 52 53 54 \
  --candidates 512 --cem-iterations 5 \
  --output outputs/mpc_teacher_paired_large_search
```

Use both all-eligible and accepted-subset results, because the accepted population
can change between ablations. Better predicted scores without better real returns
suggest objective/model ranking problems; better real returns with more search
support a search-budget limitation. These are diagnostic interpretations, not
unique causal identifications. Repeat with different warm-up lengths to sample
different parts of the state distribution.
