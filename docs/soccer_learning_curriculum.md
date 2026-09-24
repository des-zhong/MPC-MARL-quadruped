# Soccer learning curriculum

`train_high_level_mpc_replay.bash` enables the attacking curriculum, slower gated
self-play, 48-step rollouts, and a bounded shot-setup event reward. Launch a new
run with:

```bash
./train_high_level_mpc_replay.bash --cuda 7
```

The launcher does not resume or modify an existing process. The Python entry
points retain their previous defaults; the new behavior is selected explicitly
by the launcher flags.

## Attacking starts

| Stage | Finishing | Possession | Normal | Finishing distance from goal |
|---|---:|---:|---:|---|
| 0 | 50% | 30% | 20% | 1–2 m |
| 1 | 40% | 30% | 30% | 1.3–2.5 m |
| 2 | 30% | 30% | 40% | 1.6–3 m |
| 3 | 20% | 20% | 60% | 1.9–3.5 m |

Possession starts begin 2.5–4 m from goal, with distances increasing by stage and
clamped to the field. The attacker starts 0.55 m behind the ball, facing it;
the attacker slot is randomized. The teammate starts behind and to the side.
Defenders initially start to the sides of the attacking lane and move closer
as stages advance. All robot positions remain inside the field. Normal starts
retain the existing balanced reset distribution, including its near-ball option.
Supported curriculum layouts are 1v1 and 2v2 on fields at least 6 m by 4 m.

Each start category maintains the most recent 1,024 completed episodes by default.
Advancement requires at least 512 episodes **in each** of finishing and possession,
and a Wilson 95% lower success-rate bound of at least 35% for both. Failures,
concessions and timeouts remain in the denominator. Windows clear on advancement;
episodes begun in older stages cannot promote the new stage. This is a training
curriculum, not a held-out evaluation or a promise of monotonic improvement.

## Opponents

The launcher changes snapshot checks from 400 to 1,200 PPO iterations and latest
opponent probability from 0.5 to 0.2. Existing pool logic preserves the initial
frozen policy as an anchor (the anchor is only weak if its initialization is weak).

A scheduled update now also requires at least 512 normal-start episodes against
that **anchor**, a Wilson 95% lower goal-rate bound of at least 10%, and an own-team
failure rate no greater than 25%. Easy starts and matches against other pool
members cannot unlock an opponent update. The old pool stays frozen until this
gate passes. Evidence windows clear when an opponent is added; pool diversity
and episode-boundary opponent sampling remain in place.

Curriculum stage and evidence windows are stored in the existing opponent-pool
checkpoint and restored on a local full resume. A policy-only resume starts a new
curriculum. Existing remote/policy-only resume limitations still apply. Initial
in-flight reset labels may belong to stage zero after a full restore and are
excluded from later-stage advancement until those episodes end.

## Rewards and rollout length

48 high-level decisions cover 9.6 simulated seconds at the default 0.2-second
control interval. This doubles samples per PPO update, so compare experiments
by environment transitions or simulated time as well as iteration number.

`--shoot-setup-event-reward 1.0` replaces the timestep-scaled setup term with an
unscaled event reward of at most +1 per episode. The existing validity, transition,
behind-ball, distance and goal-alignment gates still apply. Once the positive
bonus has been paid, toggling skills cannot earn it again until reset. Negative
setup feedback remains signed. Goals remain +75 and concessions -75.

Existing signed ball-progress, approach, dribble and launch rewards remain active.
No new potential-based reward is introduced: it was an alternative to consider,
and adding it correctly would also require matching macro-step discounting,
terminal handling and the MPC objective. The MPC analytical teacher still does
not model the setup bonus; real PPO outcomes provide that signal.

## Controls and metrics

The common Python training parser exposes:

```text
--soccer-curriculum
--curriculum-min-episodes 512
--curriculum-success-threshold 0.35
--curriculum-opponent-threshold 0.10
--curriculum-max-failure-rate 0.25
--rollout-steps 48
--self-play-update-interval 1200
--opponent-latest-probability 0.2
--shoot-setup-event-reward 1.0
```

Watch `curriculum/stage`, and `curriculum/{normal,finishing,possession}_` metrics:
`episodes`, `goal_rate`, `concession_rate`, `failure_rate`, and `timeout_rate`.
These rates use **completed episodes**, not transitions. With no completed
episodes, rates display zero; check the count before interpreting them. They are
recent-window rates, not lifetime totals. Opponent-gate diagnostics are
`curriculum/anchor_normal_episodes`, `curriculum/opponent_goal_rate_lower95`, and
`curriculum/opponent_ready`. The latter reports gate readiness before that
iteration's possible snapshot update.

For a reliable external comparison, evaluate checkpoints against the same
frozen opponents and starting states, separately from training curriculum rates.
