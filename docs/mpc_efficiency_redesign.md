# MPC efficiency redesign

The updated `train_high_level_mpc_replay.bash` starts a new experiment. This is a
model-based policy-improvement teacher for PPO/MAPPO, not a claim of a measured
win over MAPPO. Demonstrating that win requires matched evaluation.

## Problems addressed

The mijpeot4 run spent about 48 seconds per iteration after MPC activated,
compared with 33 seconds before it. Teacher imitation improved while finishing
success declined. Imitating CEM's final distribution was not evidence that its
best action was better than the student's action. In particular, the search
variance is not a desired exploration standard deviation for the actor.

Timeout is the existing 30-second episode limit. The earlier logs cannot
separate failure to reach the ball, failure to attempt a shot, and failure to
launch it. The wrapper also forced near-ball Walk requests into Dribble. That
removed the ability to walk into a useful shooting position, while approach
reward stopped inside the ball-skill range. Shot-launch reward could credit a
command-aligned kick even when it went away from the attacking goal.

## Budgeted, filtered teacher

- Query 128 of 256 matches every eight decisions, rotating to cover every match.
- Use a two-decision horizon, 64 candidates and two CEM iterations. This is
  roughly 48 times fewer candidate-step evaluations per rollout than the prior
  256-match / every-two-step / horizon-four / 128-candidate / three-iteration
  configuration. This is a workload calculation, **not** a wall-clock speedup.
- Seed the search with the student's sampled action, held over the short horizon,
  and include that reference as an actual candidate in every CEM iteration.
- When role-aware fallback is active, hold support commands and opponent
  forecasts fixed. Only the attacker receives teacher labels. Forecasts remain
  an approximation over the short horizon; they are not opponent-policy rollouts.
- Compare the selected teacher first action against the student's first action
  with the same future continuation and opponent/support forecasts. Retain only
  finite, in-distribution improvements above `--mpc-advantage-margin` (0.1).
- Before spending the planning budget, require 512 observations and an EMA ball
  position error at most 0.25 m. These are one-step **prequential** errors on live
  nonterminal transitions, measured before insertion into replay. This catches
  poor current dynamics predictions but is not held-out counterfactual validation.
- Distill the winning action, not the final search-distribution average. Use
  categorical loss and bounded command-mean regression, weighted by predicted
  improvement; do not regress the actor's exploration standard deviation.
- Reject replay older than 1,024 high-level steps. The new query budget uses a
  256-sample minimum and batch, rather than waiting for 20,000 accepted labels.
  Expired targets are removed **before** checking the minimum and sampling.
  Each guarded update samples 256 distinct eligible rows; if fewer are available,
  the update waits. It never reduces a 256-row sample to a handful of fresh rows.
  Replay updates also stop when current model quality fails the gate. Loading
  replay under quality filtering discards legacy rows without age and weight.
- Apply the coefficient ramp to the optimizer's parameter displacement, then
  backtrack until the old-to-new hybrid policy KL on the sampled histories is
  at most 0.003. The nominal coefficient 0.01 corresponds to a full proposed
  step; smaller coefficients reduce the displacement. On rejection restore
  parameters and the shared optimizer state. This is an empirical minibatch
  trust limit, not a bound on all states.
- Keep terminal value disabled by default for this short-horizon experiment.
  It remains configurable; unvalidated terminal predictions should not be used
  to justify longer plans. Real PPO returns still bootstrap through the critic.

This design is informed by policy-guided short-horizon planning, including
[TD-MPC2](https://arxiv.org/abs/2310.16828). It uses this repository's existing
explicit world model and hybrid CEM; it is not an implementation of TD-MPC2.

## Making attacks learnable

`--attack-position-reward 1` enables signed progress toward a point 0.55 m
behind a slow ball, evaluated for the assigned attacker. The same geometry is
used by the simulator and the analytical MPC reward, with their respective time
intervals. This is bounded signed shaping, not policy-invariant potential shaping.
It provides no stationary proximity bonus. It also enables near-ball walking
instead of forcibly converting it to Dribble. Both experiment conditions use it.

`--goalward-launch-reward` gates launch credit by ball velocity alignment with
the attacking goal. Existing goal rewards, timeout duration, failure conditions,
and the once-per-episode setup bonus remain in place. The existing analytical
successful-shot event already uses goalward motion; learned event prediction and
its magnitude are still approximations to low-level launch shaping.

`--attack-diagnostics` samples every low-level tick before automatic resets and
logs per-rollout counts:

- `attack/timeout_no_contact`: never within 0.8 m (proximity, not measured contact).
- `attack/timeout_no_shot`: reached that range but no Shoot execution there.
- `attack/timeout_shot_no_launch`: Shoot was executed, but no near-ball goalward
  launch at >=0.8 m/s was observed.
- `attack/timeout_launch_no_goal`: such a launch was observed, but time ran out.
- `attack/timeout_no_goal_progress`: less than 0.25 m best improvement toward goal;
  this overlaps the four mutually exclusive categories above.
- `attack/episode_count`, `attack/episode_reached_ball`,
  `attack/episode_shot_requested_near_ball`, `attack/episode_goalward_launch`,
  and `attack/episode_goal_progress_m` (sum of best observed progress).

Divide counts by completed episodes, not environment transitions. Launch/proximity
are diagnostic proxies rather than force-based contact attribution. Early
termination is deliberately not used to cosmetically lower timeout rates.

## Matched MAPPO experiment

The previous FSP baseline's critic saw only a local observation/history.
`--centralized-critic` supplies both teams' observations to the critic while the
actor still uses only its own observation history. Enable it in **both** conditions.
It changes network input dimensions: use fresh runs, not a full resume of the
old decentralized-critic checkpoint. Both conditions have identical actor and
critic dimensions, reward settings, reset curriculum, environment count, rollout
length, and self-play settings.

Print a six-run manifest (three seeds, two conditions) without starting training:

```bash
/home/zhz/anaconda3/envs/legged_env/bin/python scripts/run_mpc_efficiency_ablation.py
```

Run either condition explicitly:

```bash
/home/zhz/anaconda3/envs/legged_env/bin/python scripts/run_mpc_efficiency_ablation.py --condition mappo --execute
/home/zhz/anaconda3/envs/legged_env/bin/python scripts/run_mpc_efficiency_ablation.py --condition mpc --execute
```

These launch separate processes; the MAPPO condition constructs no world model
or planner. The manifest is `/tmp/mpc_efficiency_experiments.json` by default.

Compare fixed-opponent, normal-start evaluation goal rates at equal environment
transitions and equal elapsed time, across seeds. Keep frozen evaluation starting
states independent of training curriculum progression. The existing
`scripts/compare_high_level_learning_curves.py` supports checkpoint comparisons;
its scenario/opponent suite must be kept identical across conditions. Training
curriculum rates alone are not a controlled efficiency result.

Monitor `mpc_quality/ball_error_m`, `prequential_samples`, `model_gate_open`,
`predicted_advantage`, `accepted_fraction`, and `query_matches`, plus
`mpc_replay/actual_policy_kl`, `rejected_policy_steps`, updates and wall time.
Acceptance/advantage diagnostics describe the latest query in that rollout;
`actual_policy_kl` is averaged over accepted teacher updates. A closed model gate
or zero accepted labels is a useful failure signal, not a reason to silently
remove the filter. Existing `planning_seconds_total` counts CEM time, so use
whole-iteration time to include validation, replay, and distillation costs.

## Replay freshness and changing skill checkpoints

Monitor `mpc_replay/eligible_size`, `expired_targets_removed`, and
`samples_per_update`. An accepted update should report the configured batch size
(256 by default); an insufficient fresh buffer reports a skipped update.
The eligible buffer is now intentionally much smaller than its capacity.

Dynamics/value replay and teacher replay checkpoints record the loaded skill
artifact hashes. A different bundle, or a legacy replay with no identity, is
discarded on load. World-model weights with changed or unknown skill provenance
are retained as initialization, with optimizer state reset and at least
`--mpc-min-world-model-updates` new gradient updates required before planning.
Restored counters cannot satisfy that refresh requirement. The existing warmup,
replay minimum and model prediction-error gates still apply. Check
`mpc/model_refresh_required` and `mpc/model_updates_since_start`.

Restart training to use these changes. A running process continues using the
skills it loaded; select the intended exported skill directory before launch.
A matched MPC/MAPPO comparison remains necessary to establish improvement in
scoring efficiency. These changes do not alter the shooting policy or its trainer.

## Validation

An eight-iteration, 16-match GPU smoke run completed successfully, exercising
15 planner queries and one guarded distillation update. It retained 47 of 120
teacher targets. The smoke used a permissive model-error threshold and zero
advantage margin to exercise the path; these are not the production defaults.
It does not establish an improvement in scoring or training efficiency.
