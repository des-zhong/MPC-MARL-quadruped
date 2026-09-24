# Shooting and coordination diagnosis: 40ius30q

Examined on 2026-09-14. High-level checkpoint **2800**, frozen shooting checkpoint
**121600**, exact training skill fingerprint
`2058c0be72dc60b1eef71b5c77037b1278ba625d29e06bb3f2fb90dd97bde13e`.
Shooting body SHA256: `5f4e22015c4e6c411bc2c5949df3d7bdb033213cd069828e3fa20d96b55a7290`.

## Findings supported by experiments

The evidence points to a coordination/skill compatibility problem, compounded
by reward inconsistencies. It does not identify a single parameter responsible
for the whole training plateau, or establish the causal effect of MPC versus MAPPO.

### 1. The deployed policy does not reliably initiate or sustain shooting

In 32 ordinary-match evaluations, both learner robots requested **zero shots**
with deterministic action selection. There were 1,258 learner decision rows
within 0.8 m of the ball; average shooting probability on these rows was **1.98%**,
and shooting was never the most probable skill. Proximity alone is not proof
that every row was a feasible shooting opportunity.

With categorical skill sampling, the same checkpoint executed 35 shot bursts
in the first episodes of 32 matches. **33/35 bursts lasted one decision (0.2 s)**.
Across 37 executed shooting decision rows, the average command speed was
**0.975 m/s**, 89.2% were below 1.5 m/s, and 37.8% had command-to-goal cosine
alignment below 0.6. These are shot-row statistics, not success probabilities.

Holding each sampled shooting request and its command for 0.8 s increased
median executed burst length to 0.8 s and produced 2 goals versus 0 without
the hold (32 matches each). Concessions increased from 1 to 3 and learner
accidental terminations from 4 to 6. This small intervention is suggestive of
a commitment problem, not evidence that a universal hold improves match play.
See `results.json` for the complete intervention results.

| Skill selection and intervention | Goals / 32 | Concessions / 32 | Learner accidental terminations / 32 |
|---|---:|---:|---:|
| Deterministic, unchanged | 0 | 0 | 8 |
| Sampled skills, unchanged | 0 | 1 | 4 |
| Sampled skills, hold request 0.8 s | 2 | 3 | 6 |
| Sampled skills, goal-directed command + 0.8 s hold | 2 | 3 | 7 |
| Sampled skills, goal-directed command + 2.4 s hold | 3 | 1 | 8 |

The 2.4-second request hold produced median **executed** shooting bursts of
1.2 seconds because geometric fallback and terminations can interrupt them.
The direction interventions use approximately 3 m/s (axis clipping caps the
normalized command at 0.95), so they change magnitude as well as direction.
Their effects cannot be attributed exclusively to aim. All deterministic
hold/aim variants remained identical to baseline because they only modify
shoot requests and none were made. These matched conditions share initial seed
and weights but diverge in state trajectories; 32 matches are too few to establish
a reliable goal-rate improvement. No long training ablation was performed.

### 2. The low-level skill works in a favorable setup but is direction-sensitive

Controlled single-environment tests used the same seed (42), zero initial yaw,
ball 0.45 m ahead, fixed world commands, and 4 seconds of uninterrupted shooting.
Training reset randomization and shooting-specific early termination were
disabled for these tests. The physical simulator, skill observation interface,
and saved skill weights were used. These are cold-start probes, not a
multi-seed estimate of skill reliability or exact snapshots from failed matches.

| Command | Peak ball speed | Alignment at peak | Directed travel | Result |
|---|---:|---:|---:|---|
| +3 m/s | 1.73 m/s | 0.913 | 1.84 m | Pass |
| +1 m/s | 1.99 m/s | 0.951 | 2.40 m | Pass |
| -3 m/s, same front-ball placement | 0.57 m/s | -0.283 | -0.34 m | Fail |

None of these three probes terminated. The +3 m/s case first exceeded 0.8 m/s
in the requested direction at **2.16 s**. This does not imply every shot needs
2.16 s, but it makes 0.2-second commitment and 0.4-second prediction horizons
questionable for this skill.

The recorded shooting training distribution used target speeds 1.5–3.0 m/s,
ball longitudinal offsets 0.35–0.60 m, and lateral offsets ±0.20 m. High-level
shoot eligibility allows distance up to 0.75 m, lateral offset up to ±0.45 m,
and local forward position down to -0.10 m, without requiring goal-directed
strike geometry. This exposes the skill to harder situations. Nevertheless,
the successful +1 m/s test **rules out low command speed as a sufficient
explanation** of failure in a favorable pose.

Exploratory `isolated_positive` and `world_*` outputs inherited randomized
shooting-training resets; they are not paired direction tests. The original
validator's 5-second terminal flag is not sufficient to label a fall. Use the
controlled `fixed_*` outputs for the comparison above.

### 3. High-level launch credit is much weaker than its nominal weight suggests

`scripts/train_high_level.py` sets a nominal launch weight of 5, but excludes
launch from `unscaled_reward_names`. At dt=0.02, the maximum per-tick launch
reward is **0.1**, multiplied by acceleration/alignment gates. Setup can pay
up to **1** once per episode. A goal pays 75. Launch acceleration is transient,
so scaling it as a sustained rate substantially reduces this intermediate
credit. Multiple acceleration events can still receive multiple rewards;
0.1 is not an episode-level cap.

This establishes the reward magnitude and a plausible credit-assignment
weakness. Only a controlled retraining ablation can establish how much it
contributes to the plateau. Simply raising all dense rewards is not justified.

### 4. MPC has concrete objective discrepancies and limited foresight

The production run uses an analytical objective, horizon 2, macro dt=0.2 s,
and no terminal value. It therefore evaluates **0.4 s** of predicted outcomes.
More query matches increase supervision volume without extending that horizon.

The analytical dribble reward is an older positive-only combination of
command tracking and goalward velocity. The real environment uses signed
goalward progress and penalizes controlled backward movement. The CPU probe
`scripts/diagnose_mpc_reward_parity.py` reproduces:

| Controlled backward dribble component over 0.2 s | Reward |
|---|---:|
| Real environment, constant-state rate integrated over interval | -0.20 |
| MPC analytical component | +0.10 |

This discrepancy concerns one component; other components such as ball progress
can still penalize backward trajectories. It does not by itself prove that
the full planner chooses backward movement.

MPC **does have a launch proxy**: weight × 0.02 × predicted `successful_shot`
probability. It is not the same acceleration/alignment computation used by
the simulator. The once-per-episode setup term is absent from reconstruction;
matching it requires tracking its paid/transition state, not adding an
unconditional stationary bonus.

The quality gate measures average one-step ball-position error on live
transitions. This is not a check of shot-conditional error, launch timing,
multi-step accuracy, or actual teacher-versus-student return advantage.

MPC is not simply refusing to teach Shoot: checkpoint 2800's fresh replay
contains **95 shooting targets out of 604**, and the student assigns mean
shooting probability **76.1%** on those teacher-shoot states. The same student
chooses Shoot as its mode on 146/604 replay states. Thus shooting exists on
some training states but does not transfer reliably to the ordinary-match
states visited in these evaluations. See `teacher_buffer.json`.

## What should change first

1. Define a shooting option with a measured initiation region, explicit target
   direction, and completion/abort conditions. Train and evaluate the same
   option execution. A timeout alone or blindly holding every shot is inadequate.
2. Use failed-match placements to validate the skill across headings, offsets,
   motion, and command speeds. Reposition before shooting when the target
   direction is outside its demonstrated capability. Retrain the skill on
   the residual failures rather than assuming another checkpoint solves them.
3. Give actual directed launches clear, bounded event credit. Share reward
   semantics between simulation and MPC, including signed dribbling; test
   forward, backward, stationary, and repeated-event cases.
4. Make MPC evaluate the shooting option over a useful duration or use a
   validated terminal value. Increasing horizon with an inaccurate model can
   worsen targets. Validate candidate ranking on real held-out shot outcomes.
5. Compare deterministic and stochastic normal-start evaluations alongside
   curriculum rates. Then run matched multi-seed MAPPO/MPC training to measure
   MPC's contribution. These diagnostics are not that causal ablation.

## Artifacts and scope

`../../outputs/core_diagnosis/` contains per-decision CSVs, immutable skill
copies, isolated-skill metrics, and probability traces. `results.json` counts
only the first completed episode from each environment, including accidental
terminations. Training-mode skill sampling tests retain deterministic continuous
commands and a deterministic checkpoint-0 opponent, so they are not exact
reproductions of the stochastic training distribution. Ordinary evaluation
uses no finishing curriculum. There are 32 matches per condition, one seed.

Recompute summaries with:

```bash
python analysis/shoot_core_40ius30q/summarize.py
python scripts/diagnose_mpc_reward_parity.py
```

New diagnostic controls are in `scripts/diagnose_shoot_coordination.py` and
`scripts/validate_robot_abilities.py` (`--command-frame world`,
`--fixed-skill-init`, `--seed`). Training rewards, MPC settings, running training,
and checkpoint weights were not changed by this investigation.

Validation: 16 targeted regression tests passed; diagnostic scripts compiled;
the completed simulator checks exited successfully. Initial diagnostic setup
errors (native cache permissions and an inherited shooting-reset cache) were
resolved before the controlled results above were collected.
