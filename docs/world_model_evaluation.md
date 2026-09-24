# World model evaluation

The main value columns report **Value rollout RMSE** and **Value rollout NRMSE**.
For horizon H, rollout RMSE is
`sqrt(mean((V(predicted_state_t+H) - V(real_state_t+H))**2))`, in return units.
This isolates value distortion caused by dynamics rollout from the critic's
existing return-prediction error. NRMSE divides by the standard deviation of
real-endpoint values at the shortest requested horizon, held fixed across all
horizons. It is null when this scale is <=1e-6. Changing the shortest requested
horizon changes the scale. Original return-target MSE remains in JSON/CSV.
A constant or insensitive critic can still produce small rollout distortion
despite poor dynamics; this is not evidence of accurate return prediction and
the metric is not forced to increase monotonically.

The table reports `Ball accumulated SD(m)`; robot and value uncertainty are
omitted from all reports. Position errors and value error remain available.

### Uncertainty definition for reporting

Accumulated ball uncertainty is estimated by propagating stochastic state
trajectories through the learned dynamics for H transitions under the recorded
actions. Each transition samples a diagonal Gaussian whose variance combines
ensemble disagreement and the members' learned variances. Subsequent predictions
use each sampled state, propagating uncertainty from earlier transitions.
The reported statistic averages endpoint sample standard deviations across XYZ
coordinates and starting states, in metres. This differs from position RMSE,
which uses Euclidean distances from deterministic mean rollouts.

This is a particle approximation of the model's moment-matched transition law;
it does not retain a fixed ensemble member per trajectory or model temporal
correlation in epistemic errors. Predicted termination does not freeze particles,
consistent with position-error evaluation. Uncertainty often increases with
horizon, but nonlinear contraction and sampling variation can cause decreases.

Use `--uncertainty-particles` (default 128, minimum 2) to control Monte Carlo
precision and cost. Particle batches contain at most eight starting states.
JSON/CSV `ball_accumulated_uncertainty_*` fields include mean SD, variance/error
correlation, Gaussian 95% coverage, and Gaussian NLL about the particle endpoint
mean. These Gaussian diagnostics approximate a potentially non-Gaussian endpoint
distribution. Constant error or variance gives null correlation. Only NLL floors
variance at 1e-12. Scoring is excluded from model and MPC inference timers.

## Collect fresh PPO evaluation episodes

```bash
# Three complete episodes per ac_weights_<suffix>.pt in this run directory:
bash collect_world_model_data.bash /path/to/high_level_mpc_replay

# Select particular checkpoints and choose the output location:
OUTPUT_ROOT=data/ppo_eval DEVICE=cuda:0 ROLLOUTS=3 \
  bash collect_world_model_data.bash /path/to/high_level_mpc_replay 0 400 latest
```

The default run is `wandb/run-20260914_192707-w96oikyf/files/tmp/legged_data/high_level_mpc_replay`.
Each checkpoint gets a separate `checkpoint_<suffix>` dataset, including
`metadata.json`, `episodes/`, and `test_manifest.json`. Every exported episode
is evaluation data; none is assigned to training. `latest` is collected separately
even if it duplicates a numbered checkpoint; specify suffixes to avoid duplication.
The same seed is used for each checkpoint for comparison.

Collection uses deterministic PPO inference, recorded training skills, and the
matching `opponent_ac_weights_<suffix>.pt`. Set `OPPONENT_CHECKPOINT=latest` or
an absolute opponent checkpoint path to use a fixed opponent across checkpoints.
`WORLD_MODEL_CHECKPOINT` defaults to `world_model_online_latest.pt` beside the PPO
files and defines the required schema. Collection rejects mismatched live state
or action schemas. `NUM_ROBOTS` is the number per team (default 2).

The launcher runs one match at a time and saves exactly `ROLLOUTS` complete
episodes per checkpoint. `MAX_STEPS` (default 10000) caps simulator steps per
checkpoint; reaching it before the episode quota raises an error and retains
only completed episodes. Nonempty output directories are never overwritten.
Timeouts are retained as truncations, with terminal states captured before reset.
The evaluator bootstraps timeout returns using the loaded value model, as training does.
Three episodes are a smoke evaluation; use more for stable error estimates.

```bash
python scripts/evaluate_world_model.py \
  --checkpoint /path/to/world_model_online_latest.pt \
  --value-checkpoint /path/to/terminal_value_online_latest.pt \
  --dataset data/ppo_eval/checkpoint_latest --device cuda:0
```

## Evaluate predictions

The console's `MPC ms/inference` is a complete CEM planning call for **one
state**, including candidate generation, model rollouts, objective scoring,
elite updates, and final plan selection. Each horizon uses the same sampled
states. Timing excludes loading and transfers and synchronizes CUDA. It uses
cold starts and jointly optimizes all robots; online warm-started planning with
a fixed opponent can have different latency.

Use `--mpc-config configs/mpc.yaml` and optional `--mpc-profile` to select the
search workload. The default resolves to 128 candidates, 3 iterations, and
candidate chunks of 64. Terminal-value scoring follows the selected config;
merely supplying `--value-checkpoint` does not enable it in a reward-only
planner. JSON stores the resolved config for every horizon, and JSON/CSV include
mean and p95 milliseconds per inference. `--mpc-timing-samples` (default 3),
`--mpc-timing-repeats` (3), and `--mpc-timing-warmup` (1) control benchmarking.
Use `--skip-mpc-timing` for prediction errors alone. The earlier model-only
batch/throughput timing fields remain in JSON/CSV for compatibility; they are
not full planning latency.

`Value MSE/Var` matches `plot_terminal_value_error`: return prediction MSE
divided by return variance, not RMSE or explained variance. Timeout targets
include the loaded model's predicted continuation. This is a bootstrapped
estimate, not an independently observed full return.

Run from the repository root (no simulator required):

```bash
python scripts/evaluate_world_model.py \
  --checkpoint /path/to/world_model/best.pt \
  --value-checkpoint /path/to/terminal_value/best.pt \
  --dataset data/world_model_env_v2 \
  --horizons 1 2 4 10 --device cuda:0
```

The script prints a table and writes `metrics.json` and `metrics.csv` under
`outputs/world_model_evaluation`. Omit `--value-checkpoint` to evaluate dynamics
alone; unavailable value metrics are null, not zero.

For legacy datasets lacking `robot_N.shoot_option_remaining`, use
`--allow-missing-shoot-timers` to explicitly assume inactive (zero) timers.
The evaluator maps existing fields by name, preserving their values despite
shifted offsets, and records zero-filled fields in JSON. Other missing or
incompatible fields remain errors. This compatibility evaluation does not
validate active shooting options or eliminate differences in collection dynamics;
use matching newly collected held-out data for that purpose.

Each horizon is measured in recorded high-level transitions. Predictions start
from the recorded state and use the recorded action sequence without subsequent
state corrections. Predicted termination does not freeze the dynamics rollout.
All horizons share the same seeded sample of test-split origins with at least
the maximum horizon available, never crossing a termination or truncation.
This conditions evaluation on trajectories lasting that long. Use a separate
`--horizons 1` run to include short episodes in one-step evaluation.

Ball and robot errors compare the endpoint at step H, excluding the initial
state. XYZ positions are converted to metres using each state's field geometry.
Mean distance is mean Euclidean error; RMSE is the square root of mean squared
Euclidean error, pooled over samples and, for robots, all robots.

Terminal value error compares `V(predicted_state_t+H)` with recorded returns
starting at the real endpoint t+H, using the value checkpoint's gamma.
The action rewards before t+H are excluded. True termination has zero bootstrap;
timeouts bootstrap from `V(real_final_next_state)`. The loaded model is assumed
ready, so no uninitialized-model tail exclusion applies. Use
`--no-terminal-bootstrap-on-timeout` to match training with bootstrapping disabled.
The same sampled origins used for position errors are used at each horizon;
incomplete episodes are excluded from value scoring. Raw MSE is the mean squared
endpoint value error; normalized MSE divides it by the variance of endpoint
return targets (floored at 1e-6). JSON/CSV retain both metrics and also include
target variance and sample count. Aggregation pools samples without a temporal
EMA. This retains the training plot's MSE/variance normalization and return-target
convention, but evaluates predicted endpoints instead of real input states.
Errors can now reflect accumulated dynamics error; neither raw nor normalized
MSE is guaranteed to increase monotonically with horizon. Target return variance
also changes with horizon, so inspect raw MSE alongside the normalized ratio.

Timing includes deterministic ensemble rollout and separately the value forward
pass, after warmup, with CUDA synchronization. Loading, transfers, metrics and
warmup are excluded. Per-batch time is average batch latency; per-sample time is
amortized throughput cost, not single-request latency. Use `--batch-size 1` for
single-trajectory latency. Control cost with `--max-samples`, `--warmup`, and
`--repeats`; the final batch may be smaller. JSON records all run arguments.
