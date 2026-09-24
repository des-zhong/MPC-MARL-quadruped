# Retraining after the September 14 diagnosis

For the current shared PPO, discrete PPO, and MPC training launchers and MPC
update safeguards, use [matched goal training](matched_goal_training.md).
That profile uses the shooting bundle installed in `checkpoints/reproduction/shoot`.
The validation-first sequence below describes the earlier launcher defaults.

The fixes change the execution contract, observation size, reward, and world-model
state. Start fresh high-level runs; do not resume `40ius30q` or reuse its replay.
The existing walking and dribbling policies can remain in use.

## What changed

- A Shoot request first positions and turns the robot behind the ball, then
  commits to a forward strike for at most 2.4 seconds. Launch, unsafe posture,
  nearby robots, loss of the ball, or the deadline terminate the strike. A short
  cooldown prevents immediate retriggering. The coordinator observes the timer.
- Shooting retraining now samples the ball positions and heading errors used
  at strike initiation. The new mode trains within the same 2.4-second deadline.
- A goalward launch earns a threshold-crossing event reward, without the old
  0.02 timestep attenuation. Setup-state reward is disabled in option mode.
  Simulator and MPC share the signed dribbling and launch scoring functions.
- MPC uses the same shooting execution helper and includes its timer in the
  modeled state. It starts a fresh online world model. Terminal value supplies
  information beyond the short dynamics horizon, but teacher queries wait for
  evidence that value predicts newly completed trajectories better than a
  constant predictor. Timeout targets use finite-horizon returns.
- Shooting teacher labels also require sufficient real shooting transitions
  and low observed ball-position/velocity prediction errors. Locked strikes
  are excluded from teacher updates. Existing improvement, uncertainty, and
  policy-KL protections remain active.
- Checkpoints save a local evaluation configuration. Evaluation restores the
  shooting-option flag and exact recorded skill files. Normal training exits
  avoid the observed Isaac Gym shutdown crash.

## Commands, in order

Run from the repository root with the `legged_env` environment activated.

1. Train a new shooting policy:

   ```bash
   bash train_shooting_option.bash
   ```

   Default GPU is `cuda:7`; use `DEVICE=cuda:4 bash train_shooting_option.bash`
   to change it. Checkpoints go to `tmp/legged_data/shoot_option`.

   To continue from a saved option-training checkpoint:

   ```bash
   bash train_shooting_option.bash --resume \
     --resume-checkpoint /path/to/shoot_option/ac_weights_latest.pt
   ```

   Use the actual checkpoint directory printed by the previous run (it can
   be inside the W&B run directory). This loads learned weights into a new
   training run; optimizer state is not restored, and exploration standard
   deviation is reset to `--init-noise-std`. Use a compatible body-frame
   shooting checkpoint. Repeat readiness validation after further training.

2. Validate and freeze its latest complete checkpoint:

   ```bash
   python scripts/validate_shooting_option.py --device cuda:4
   ```

   This tests four initiation configurations with three seeds. Passing requires
   a directed launch by 2.4 seconds and stability in at least 10 of 12 cases.
   Inspect `outputs/shooting_option_validation/summary.json`. If it fails,
   continue improving the shooting policy and repeat this step before training
   the coordinator. Passing establishes isolated readiness, not match success.

3. Start a fresh MPC run:

   ```bash
   bash train_high_level_mpc_replay.bash --cuda 7
   ```

   The launcher uses the frozen, hash-verified skill from step 2. It rebuilds
   the online world model automatically; no separate old world-model training
   script is needed. Record the checkpoint directory printed by training.

   To test the MPC contribution, use this matched experiment **instead of**
   the single-run command above:

   ```bash
   python scripts/run_mpc_efficiency_ablation.py --condition both \
     --seeds 42 43 44 --iterations 3000 --device cuda:7 --execute
   ```

   This launches six runs sequentially with the same skill snapshot, shooting
   options, reward, curriculum, and self-play configuration. Both conditions
   receive the execution repairs, so their difference measures MPC assistance.

4. Evaluate the new checkpoints:

   ```bash
   bash validate_high_level.bash NEW_MPC_CHECKPOINT_DIR
   bash compare_learning_curves.bash NEW_MAPPO_CHECKPOINT_DIR NEW_MPC_CHECKPOINT_DIR
   ```

   Replace the uppercase placeholders with checkpoint directories printed by
   the new runs. Comparing against the old baseline would confound MPC with
   the shooting and reward repairs.

## Interpreting results

Check directed launches, goals, timeouts and falls against fixed opponents,
alongside reward. Compare both environment steps and wall time across seeds.
MPC can remain inactive while its model/value gates fail; inspect terminal-value
`prequential_mse`, `prequential_variance`, `validation_ready`, shot-model readiness,
and accepted teacher labels before attributing changes to MPC.

The launch formula is shared, but MPC still approximates a macro step using
predicted endpoints; it does not reproduce every intermediate physics frame.
Readiness thresholds are research heuristics. These repairs do not establish
convergence, monotonically increasing reward, or an MPC efficiency advantage.

## Verification completed

- 63 regression tests passed, including timer/abort behavior, real/MPC execution
  parity, signed reward behavior, value gating, and an actual CPU CEM plan.
- Eight-iteration GPU shooting-training smoke run completed successfully.
- Eight-iteration high-level training/model-update loop completed; after fixing
  shutdown handling, a fresh two-iteration GPU run exited successfully and
  saved its configuration and checkpoints.
- The new high-level checkpoint loaded with its recorded skills and completed
  a four-environment, 20-step GPU evaluation, exiting successfully.
- Smoke runs use temporary output directories. They are integration checks;
  the new shooting skill has not yet been trained to readiness, and no long
  matched MAPPO/MPC experiment has been completed for this revision.
