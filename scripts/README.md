# Script layout

Top-level entry points target the AS2 robot. The four low-level and coordinator
trainers are consolidated as:

- `training_walking.py`
- `training_dribbling.py`
- `training_shooting.py`
- `training_high_level.py`

For example, start AS2 walking training with:

```bash
python scripts/training_walking.py --headless
```

Dribbling and shooting use different checkpoint directories by default, so
they can train concurrently:

```bash
python scripts/train_dribbling.py --headless
python scripts/train_shooting.py --headless
```

Their checkpoints are written to `tmp/legged_data/dribble/` and
`tmp/legged_data/shoot/`, respectively. For multiple runs of the same skill,
give every process a distinct directory:

```bash
python scripts/train_dribbling.py \
  --checkpoint-dir tmp/legged_data/dribble-experiment-2 \
  --headless
```

New W&B run folders preserve those nested paths. For local collection, point
`--dribble-policy-dir` or `--shoot-policy-dir` at the corresponding nested
folder below `wandb/<run>/files/tmp/legged_data/`.

AS2 playback and world-model tools also live at the top level and use AS2
defaults. Robot-neutral helper code remains top level when it is shared by
those entry points.

Validate walking, dribbling, and shooting in separate simulator processes:

```bash
python scripts/validate_robot_abilities.py \
  --ability all \
  --skill-policy-source local \
  --headless
```

Use `--ability walk`, `--ability dribble`, or `--ability shoot` to isolate one
policy. Each scenario writes a rollout video, metrics CSV, diagnostic plot, and
pass/fail JSON under `outputs/ability_validation/<ability>/`. Add
`--fail-on-threshold` when the result should be reflected in the process exit
code.

GO1 entry points, legacy configuration, and the Unitree actuator-network tools
live in `go1_scripts/`. The GO1 training scripts use the same `training_*`
naming inside that package:

```bash
python scripts/go1_scripts/training_walking.py --headless
```

## Joint-team world model and MPC teacher

The AS2 world-model collector now creates two equal robot teams and records one
global state and joint hybrid action for all robots. `--num-robots` is the
number of robots per team:

```bash
./collect.bash initial as2 --episodes 20000
python scripts/train_world_model.py \
  --dataset data/world_model_as2 \
  --output checkpoints/world_model_as2 \
  --num-robots 2
```

After training that model, start self-play PPO with privileged MPC guidance:

```bash
./train_high_level_with_mpc_teacher.bash \
  --world-model-checkpoint checkpoints/world_model_as2/best.pt
```

The student keeps its decentralized shared-policy observations. The opponent
uses a deterministic slow walk-to-ball action in both the MPC forecast and the
simulator request. MPC sees the joint world-model state, optimizes only the learning
team, and adds a dense per-agent action-agreement reward. Candidate plans
combine the analytical short-horizon reward with a conservatively weighted
terminal value; world-model and value uncertainty are used as risk gates. MPC
is used only during training and is not required when evaluating the learned
policy.

### Terminal-value ablation table

Run every numbered policy checkpoint on one fixed benchmark, then generate a
LaTeX table and explicit checks of the terminal-value claim:

```bash
./run_ablation_study.bash \
  /path/to/mappo_fsp/checkpoints \
  /path/to/ours_without_terminal_value/checkpoints \
  /path/to/ours/checkpoints \
  --cuda 6
```

`--cuda 6` selects simulator GPU 6 (equivalently, use `--cuda-device 6` or
`--device cuda:6`).
The same device option is forwarded to every rollout subprocess. If that GPU
still cannot hold `--eval-num-envs 16` matches, reduce the parallel count, for
example with `--eval-num-envs 4`.

The paper-facing table contains only `MAPPO-FSP`, `Ours w/o Terminal Value`,
and `Ours`. The controlled terminal-value comparison is the last two rows:
they keep planner guidance, PPO, fictitious self-play, KL distillation,
hyperparameters, initialization, and training budget fixed. `MAPPO-FSP` is a
reference baseline; its comparison with `Ours` measures the overall method and
must not be presented as evidence for one individual component. Use repeated
training seeds for every condition.

Produce the three policy directories with:

```bash
ABLATION_TRAINING_SEED=0 ./train_ablation_method.bash mappo_fsp
ABLATION_TRAINING_SEED=0 ./train_ablation_method.bash no_terminal
ABLATION_TRAINING_SEED=0 ./train_ablation_method.bash ours
```

This writes:

```text
checkpoints/ablation/MAPPO-FSP/seed_0
checkpoints/ablation/Ours-no-terminal/seed_0
checkpoints/ablation/Ours/seed_0
```

`no_terminal` and `ours` both retain the same positive guidance and
distillation coefficients; only terminal continuation value use changes.
Override `WORLD_MODEL_CHECKPOINT`, `TERMINAL_VALUE_CHECKPOINT`,
`ABLATION_ITERATIONS`, and the low-level policy directories through
environment variables. The controlled default trains all three high-level
policies from random initialization while sharing the same pretrained
low-level skills. `ABLATION_INITIAL_POLICY` may warm-start all three conditions
from one neutral high-level initialization, but it must not be a policy already
trained by one of the compared methods.

The launcher evaluates all methods against the same early/middle/final frozen
MAPPO-FSP opponents and initialization seeds. It writes raw rollouts,
`evaluations.csv`, `learning_curve.csv`, and `learning_curves.png`, followed by:

- `ablation_table.tex` and `ablation_table.csv`;
- `ablation_results.json` with machine-readable paired deltas;
- `claim_report.md` with verdicts for the overall baseline comparison and the
  controlled terminal-value comparison.

The table uses final expected match score, mean curve score (normalized area
under the learning curve), and steps to a pre-declared expected-score target.
Expected score counts a win as 1 and a draw as 0.5. Elo is omitted because,
against a single fixed reference population, it is only a log-odds transform
of the same score. Set the target before examining results, for example:

```bash
ABLATION_TARGET_SCORE=65 ABLATION_EVAL_SEEDS=0,1,2,3,4 \
  ./run_ablation_study.bash FSP_DIR NO_TERMINAL_DIR OURS_DIR
```

If evaluations already exist, rebuild only the statistics and table with:

```bash
python scripts/analyze_ablation_study.py \
  --evaluations outputs/ablation_study/evaluations.csv \
  --target-score 65
```

Bootstrap intervals resample paired opponent/initialization-seed blocks. They
do not establish training-seed robustness when each condition contains only
one trained run; paper claims should include multiple independently trained
runs per condition. To combine repeated exports, concatenate the three-method
evaluation CSVs with a `training_run` column, or pass repeated labeled inputs
such as `--evaluations seed0=results_seed0/evaluations.csv
--evaluations seed1=results_seed1/evaluations.csv`.

Sample efficiency means real agent-environment transitions, not wall-clock
planner cost. If a method is warm-started or uses method-specific real data to
train its world/value model, include those transitions with repeated
`--step-offset 'LABEL=N'` options to the launcher. The evaluator already
infers the rollout batch size of each policy run, so methods trained with
different numbers of parallel environments remain on the same transition
axis. Use `--fail-on-unsupported` with the analysis script in automated checks.

### MPC reward-ranking and terminal-value evaluation

`scripts/evaluate_mpc_candidate_ranking.py` evaluates candidate ordering, not
just scalar reward RMSE. It disables initialization/domain randomization,
branches sampled CEM sequences across identical vector environments, and
reports Pearson/Spearman correlation, pairwise accuracy, top-k overlap, and
selection regret for the learned reward, analytical reconstruction, and full
objective.

The default MPC configs reconstruct the known high-level reward from imagined
state transitions/events, apply geometric skill fallback inside every imagined
step, reject state uncertainty above the measured maximum deployment threshold,
and load the consolidated pipeline's `checkpoints/terminal_value_mpc/best.pt`
for terminal continuation value. The value ensemble is trained on both real
and MPC-imagined terminal states; use `--fail-on-threshold` for ranking gates.

### World-model collection modes

Use a new run name whenever environment dynamics, observations, actions,
rewards, timing, field geometry, robot count, or opponent behavior changes:

```bash
./collect.bash initial env_v2
./train_world_model_pipeline.bash all --run env_v2
```

The first command writes `data/world_model_env_v2`. It refuses to reuse an
existing directory, preventing old and new environment transitions from being
mixed. Rendering-only changes do not require recollection.

The training pipeline automatically performs MPC collection. For a manual MPC
collection, use `./collect.bash mpc env_v2`; optional controls are limited to
`--episodes`, `--num-envs`, `--iteration`, and `--device`. Lower-level Python
scripts remain available for experiments that need detailed overrides.
