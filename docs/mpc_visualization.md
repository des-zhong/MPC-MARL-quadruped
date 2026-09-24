# MPC action selection visualization

Run `bash visualize_mpc.bash` from the repository. It uses the local reproduction
world model and skill checkpoints, two teams, and the `teacher_high_quality` CEM
profile. Outputs go to `outputs/mpc_visualizations/episode_000/` by default.

The video contains a closer simulation view, the selected predicted trajectory
against the next real transition, and CEM diagnostics: iteration objectives,
first-step skill probabilities, sampled ball endpoints colored by objective,
and planned versus executed commands. The red star marks the selected plan,
which can come from an earlier CEM iteration; dots are a diagnostic subset of
the final iteration, not the entire search. Gray crosses indicate rejected
candidates. Only the first action is executed before replanning.

```bash
bash visualize_mpc.bash --device cuda:0 --episodes 1 --max-steps 60 \
  --camera-scale 0.65 --save-dir outputs/mpc_action_selection
```

`--camera-scale` multiplies the usual full-field camera distance: smaller values
move closer and may crop field edges. `--max-steps` caps each recorded episode;
`terminated` in the summary distinguishes a natural ending from this cap. The
simulation camera shows the live state after execution (the reset state on a
terminal transition); the tactical plot retains the captured terminal state.

Each episode also includes an initial `action_selection.png`, prediction error
and skill plots, and `diagnostics.json`. Use `--world-model-checkpoint`,
`--config`, and `--profile` to select another model and compatible environment.
These diagnostics visualize the supplied model; they do not establish its
prediction accuracy or policy quality.

## MAPPO control with MPC predictions

`bash visualize_mappo_mpc.bash` runs the saved MAPPO policy directly, with its
saved opponent pool and recorded training skills/environment. Edit
`HIGH_LEVEL_POLICY_DIR`, `HIGH_LEVEL_CHECKPOINT`, `OPPONENT_POOL_CHECKPOINT`,
`WORLD_MODEL_CHECKPOINT`, `TERMINAL_VALUE_CHECKPOINT`, `OBJECTIVE_MODE`,
`DEVICE`, and `CAMERA_SCALE` in that launcher. CLI overrides are accepted:

```bash
bash visualize_mappo_mpc.bash --steps 60 --camera-scale 0.65
```

The four panels show:

- Top left: the simulation rotated so field +x points right, aligned with the tactical plot.
- Top right: MPC's predicted robot and ball trajectories, with horizon endpoints.
- Bottom left: one-step position MSE, averaged over x/y in square metres, with curves for Ball, Robot 0 and Robot 1. Predictions use executed commands, including fallback.
- Bottom right: the terminal-value model evaluated on real states against realized discounted MAPPO returns. A cursor marks the current frame.

The value comparison is computed after recording, so video assembly happens at
the end. Recorded returns stop at episode boundaries. Incomplete episodes and
timeouts are labeled truncated; they are not presented as complete returns.
MPC's hypothetical terminal states have no observed counterfactual returns:
the value chart compares values and returns on the actual MAPPO state sequence.
The camera is captured before execution, aligning it with the MPC planning state.

The launcher attempts five episodes by default (`EPISODES=5` or `--episodes 5`).
Only completed episodes with a learning-team goal produce videos, named
`outputs/mappo_mpc/predictions_episode_000.mp4`, etc., with matching first-frame
PNGs. Failed and incomplete episodes are skipped. `predictions.json` lists saved
videos and contains diagnostic records for every attempt; `metrics.csv` contains
the rollout outcomes. `--steps` is an optional overall safety cap (default 10000). Only the
six-component hybrid MAPPO action encoding is supported. Opponent commands are
held fixed during MPC predictions, while real opponents continue acting each step.

The MAPPO launcher uses its editable `SEED` setting and `--random-learning-start`.
Use `bash visualize_mappo_mpc.bash --seed 17` (or edit `SEED` in the launcher).
Robot poses are randomized, and the ball is sampled 0.4–0.95 m in front of a
random learning-team robot, within ±0.35 radians of its heading. Unsafe samples
are retried; a failed placement raises an error instead of silently starting
with the ball elsewhere. This reset mode overrides saved curriculum/fair starts
while preserving the policy's execution settings. Change the near-ball bounds
with `--near-ball-init-min-distance`, `--near-ball-init-max-distance`, and
`--near-ball-init-max-angle`. The seed is saved in the prediction JSON.

### Terminal-value reference for the current checkpoint

For `run-20260920_145324-clcesqb3`, `files/config.yaml` records
`online_mpc.terminal_bootstrap_on_timeout: false`. The latest value checkpoint
is epoch 3200 with gamma 0.99. The online trainer buffers real pre-action states
and learning-team `high_level_match_rewards`, then computes discounted
return-to-go when the environment episode ends. These labels are stored in
`online_replay_latest.pt`'s value replay as `return_to_go`.

Thus a completed goal/failure episode **or an actual environment timeout** uses
`sum(gamma**k * reward[t+k])` with zero continuation after the episode boundary.
Stopping a video early using `--steps` is different: the training path never
adds that unfinished episode to value replay, so its recorded partial return
is not the full target. The chart now omits the reference for unfinished runs.

Old online value payloads serialize a generic `ValueModelConfig` whose
`bootstrap_on_truncation=True` default can disagree with the actual run flag.
The visualization resolves timeout semantics from the saved run configuration,
not that generic field. For a run with timeout bootstrapping enabled, it adds
`gamma**remaining_steps * V(final_pre_reset_state)` and labels the curve
“Bootstrapped return target”; this is not independent Monte Carlo ground truth.
Unknown timeout semantics are left without a reference rather than guessed.

The current MAPPO policy comes from a different training run than the value
model. Their saved environment/reward-scale settings match, but their behavior
policies need not. New MAPPO episode returns are evaluation samples using the
same reward/discount convention, not the original historical training labels.
The exact training references remain the value replay's stored `return_to_go`.

Set `--mpc-horizon 10` to predict/plan ten high-level steps ahead (2 seconds at
0.2 seconds per step). The MAPPO launcher exposes `MPC_HORIZON=10`; command-line
arguments override it, for example `bash visualize_mappo_mpc.bash --mpc-horizon 20`.
This overrides the MPC profile horizon and updates the top-right trajectory and
its duration label. MAPPO remains the controller. The actual horizon in steps
and seconds is also recorded in the output JSON. Longer horizons increase
planning time and accumulated world-model prediction error.
