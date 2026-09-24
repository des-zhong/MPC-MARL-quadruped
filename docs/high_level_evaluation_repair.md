# High-level evaluation and training repairs

Longer training alone has not produced a sustained normal-start goal-rate gain.
These changes address verified evaluation mismatches and insufficient fresh MPC
teacher data; they do not establish convergence or an MPC advantage over MAPPO.

## Exact skills and action execution

`validate_high_level.bash` now uses `--training-skills`. The evaluator reads
the high-level run's recorded artifact paths and hashes and makes immutable
copies. It does not substitute a skill run's mutable `latest` file.
For the September 12 runs, the recorded shooter is checkpoint 61600, and all
recorded artifacts were recovered and verified successfully.

Collision avoidance and near-ball repositioning now inherit the saved training
settings. Explicit incompatible overrides are checked by the existing evaluation
contract validation. Evaluation retains ordinary match starts rather than the
finishing curriculum, so training and evaluation goal rates need not match.

`compare_learning_curves.bash` also uses exact recorded skills by default. Its
goal-rate metric is goals / **all completed episodes**, including accidental
terminations in the denominator. `conditional_goal_rate` retains the previous
non-accidental denominator in the CSV. Expected-score mode retains its existing
conditional definition. The horizontal axis is agent environment transitions,
accounting for the different rollout sizes.

Rollout cache paths now include hashes of checkpoints, skill bundles, relevant
evaluation/controller code, and evaluation arguments. Old mismatched results
are not reused. Explicit skill-directory overrides are available for transfer
experiments and are marked in the protocol when they differ from training.

Run the corrected evaluations:

```bash
bash validate_high_level.bash
bash compare_learning_curves.bash
```

The full comparison launches many GPU evaluations. Existing aggregate plots
are replaced as corrected evaluations complete; previous rollout CSVs remain.
The default 100 episodes per checkpoint is an initial diagnostic, not a precise
estimate at a goal rate of only a few percent.

The evaluation CLI also uses the repository's existing post-output process-exit
workaround for legacy Isaac Gym teardown crashes. Exceptions during evaluation
still fail; only a normally completed rollout takes this exit.

## More teacher data and a matched baseline

The MPC launcher and matched ablation use a query budget of 128 instead of 64.
This targets the observed fresh-label shortage while keeping the advantage
filter, full distinct batches, replay expiration, and policy-KL bound intact.
Monitor `mpc_replay/eligible_size`, `samples_per_update`, update frequency, and
whole-iteration time. More queries are not evidence of improved efficiency.

Use `scripts/run_mpc_efficiency_ablation.py --shoot-policy-dir PATH` to give both
conditions the same frozen shooter. It applies the same curriculum, rewards,
centralized critic, rollout length and opponent settings to MAPPO and MPC.
Its default only prints commands and writes a manifest; `--execute` runs them.
For the recovered September 12 bundle, PATH is:

```
outputs/evaluation_skills/e8ce2e5dc8ee8114cc3646ff8f25d7273a7de0cd7ca4024bc0544060182708d4/shoot
```

Restart training to change query coverage. Existing high-level policies are
unchanged. Skill switching and the persistent launch bottleneck remain hypotheses
to test; no unvalidated action-hold rule has been added to PPO or MPC.
