I already have a working robot-football world-model data collector, trained world model, and MPC implementation in this repository.

Do NOT rewrite these systems from scratch.

Your task is to inspect the existing implementation and modify it to support **terminal-value-augmented MPC**.

The purpose is to compensate for the limited prediction horizon of the learned world model.

The new MPC objective should be:

\[
J(a_{t:t+H-1})
=
\sum_{h=0}^{H-1}
\gamma^h
\hat r_{t+h}
+
\alpha_V \gamma^H V_\psi(\hat s_{t+H})
-
\lambda_u U
-
C,
\]

where:

- \(H\) is the finite MPC planning horizon;
- \(\hat r_{t+h}\) is predicted reward from the existing world model;
- \(\hat s_{t+H}\) is the predicted terminal state after the MPC horizon;
- \(V_\psi(\hat s_{t+H})\) estimates long-term discounted return after the planning horizon;
- \(U\) is the existing world-model uncertainty penalty;
- \(C\) contains the existing MPC constraint and regularization penalties;
- \(\alpha_V\) controls the strength of the terminal value term.

The intention is:

```text
short-term consequences
        =
world-model rollout rewards

long-term consequences
        =
terminal value model

MPC objective
        =
short-term reward
+ terminal state value
- uncertainty
- constraints
```

The terminal value model must be trained from **real simulator trajectory returns**, not imagined world-model rollouts.

# 1. Inspect the current repository first

Before changing code, inspect and understand the existing implementations of:

- world-model data collection;
- world-model dataset format;
- episode storage;
- world-model state representation;
- state normalization;
- world-model training;
- world-model rollout;
- ensemble uncertainty;
- MPC planner;
- CEM sampling, if CEM is used;
- MPC objective;
- MPC teacher rollout collection;
- environment reset and termination handling;
- reward accumulation over one macro action;
- configuration;
- checkpoint loading and saving;
- visualization and validation tools.

Do not invent replacement APIs.

Reuse the current implementation wherever possible.

Identify exactly where the following modifications belong:

1. storing full episode information required for return calculation;
2. calculating return-to-go targets;
3. training a terminal value network;
4. loading the value model into MPC;
5. adding terminal value to candidate evaluation;
6. collecting better trajectories using value-augmented MPC;
7. updating the value model from newly collected real trajectories;
8. validating whether terminal value actually improves MPC.

Before implementing, summarize the current relevant code structure and state which files will be modified.

# 2. Preserve the existing world model

The existing world model should continue predicting approximately:

\[
f_\phi(s_t,a_t)
\rightarrow
(
\hat s_{t+1},
\hat r_t,
\hat d_t,
\text{uncertainty},
\ldots
).
\]

Do not merge the terminal value function into the dynamics model.

The new architecture should be:

```text
                 ┌──→ predicted reward
state + action ─→ world model
                 └──→ predicted next state
                            │
                            │ repeat H times
                            ↓
                     terminal state s_H
                            │
                            ↓
                       value model
                            │
                            ↓
                         V(s_H)
```

Keep:

```text
world model f_phi
```

and

```text
terminal value model V_psi
```

as separately trainable components.

# 3. Modify the world-model/MPC data collector

Inspect the existing collector.

The collector must preserve complete trajectories so discounted return-to-go can be calculated correctly.

For every macro-level transition, ensure the dataset contains:

```text
episode_id
step_id

global_state
joint_action

real_reward
real_next_state

terminated
truncated

elapsed_macro_steps or elapsed_low_level_steps

event information, if already available
behavior_source
```

If the collector already stores these fields, reuse them.

Do not duplicate existing fields.

The collector must preserve strict temporal ordering inside each episode.

Never randomly shuffle transitions before calculating returns.

# 4. Add return-to-go calculation

For every real simulator trajectory, calculate:

\[
G_t
=
r_t
+
\gamma r_{t+1}
+
\gamma^2 r_{t+2}
+
\cdots
+
\gamma^{T-t-1}r_{T-1}.
\]

Implement a reusable utility such as:

```python
def compute_discounted_returns(
    rewards,
    terminated,
    truncated,
    gamma,
):
    ...
```

or adapt to the existing repository style.

Return calculation must operate independently for every episode.

Do not allow return propagation across episode boundaries.

Store or dynamically compute:

```text
return_to_go
```

for each transition.

Configuration:

```yaml
value_model:
  gamma: 0.99
```

Use the same gamma later in MPC unless explicitly configured otherwise.

# 5. Handle termination correctly

If an episode ends with a true terminal state:

```text
goal
failure
terminal game condition
```

then:

\[
V(s_T)=0
\]

unless the existing reward convention explicitly defines another terminal value.

For true termination:

```python
bootstrap_value = 0
```

If an episode is truncated only because of:

- time limit;
- collector horizon;
- artificial rollout cutoff;

then do not automatically interpret the final state as having zero value.

Support:

```yaml
value_model:
  bootstrap_on_truncation: true
```

Initially, if no reliable bootstrap model exists, allow truncated episodes to be excluded from value training near their final steps.

Document whichever strategy is selected based on the existing environment.

# 6. Add optional lambda-return support

Implement ordinary Monte Carlo return-to-go first.

Also structure the code so future critic-based bootstrapping can support TD(lambda):

\[
G_t^\lambda.
\]

Configuration:

```yaml
value_model:
  target_type: monte_carlo
```

Possible future values:

```text
monte_carlo
td_lambda
```

Do not require TD(lambda) for the initial implementation.

Monte Carlo return from real trajectories is the baseline target.

# 7. Terminal value model

Implement a centralized terminal value network:

\[
V_\psi(s)
\rightarrow
\hat V.
\]

It should use the same global task-level state representation as the world model wherever reasonable.

Reuse:

- state schema;
- state adapter;
- normalizer;
- coordinate conventions.

Do not create a second incompatible state encoding without a strong reason.

Suggested interface:

```python
class TerminalValueModel(nn.Module):

    def forward(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            states: [batch, state_dim]

        Returns:
            values: [batch]
        """
        ...

    @torch.no_grad()
    def predict(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

# 8. Value model architecture

Keep the initial value model simple.

Use an MLP compatible with the existing project architecture.

Suggested default:

```yaml
value_model:
  hidden_dims: [512, 512, 256]
  activation: silu
  layer_norm: true
  dropout: 0.0
```

Input:

```text
normalized global football state
```

Output:

```text
scalar predicted discounted return
```

Do not output an action.

Do not make this a Q-function.

The model estimates:

\[
V(s)
\]

rather than:

\[
Q(s,a).
\]

# 9. Value-model training loss

Initially use:

\[
\mathcal L_V
=
\frac{1}{N}
\sum_t
\left(
V_\psi(s_t)-G_t
\right)^2.
\]

Allow Huber loss:

```yaml
value_model:
  loss: huber
```

Huber should be the preferred default if return targets have high variance.

Support optional value-target normalization.

If target normalization is used:

```text
normalize return targets during training
denormalize V(s) before inserting it into MPC objective
```

This is critical.

The MPC objective must use value estimates on the same reward scale as predicted cumulative reward.

# 10. Do not train value from imagined states initially

The initial terminal value model must be trained only from states visited in real simulator trajectories and their real simulator returns.

Do NOT generate:

```text
imagined world-model trajectory
→ predicted reward
→ use as value target
```

as the primary value dataset.

The initial value target source must be:

```text
real simulator rewards
→ real episode return
→ V target
```

This prevents compounding world-model bias into both dynamics prediction and terminal value.

# 11. Value dataset

Implement or adapt the dataset so value-model samples contain:

```text
global_state
return_to_go

episode_id
step_id

terminated
truncated

behavior_source
```

Optional diagnostic fields:

```text
episode_return
remaining_episode_length
goal_scored
possession
shot_event
```

Split train, validation, and test data by complete episode.

Do not split individual transitions randomly across datasets.

Avoid episode leakage.

# 12. Value-model training script

Add or adapt a command such as:

```bash
python scripts/train_terminal_value.py \
    --config configs/terminal_value.yaml \
    --dataset data/mpc_teacher \
    --output checkpoints/terminal_value
```

Reuse existing training utilities where possible.

Training must support:

- GPU;
- checkpointing;
- validation;
- early stopping;
- resume;
- deterministic seeds;
- mixed precision if already used;
- training curves.

Save:

```text
model weights
optimizer state
config
state schema
state normalization
return normalization, if used
gamma
training dataset manifest
validation metrics
repository commit hash when available
```

# 13. Initial value-model training source

The system may initially not have a good terminal value model.

Implement the following progression.

## Phase 0

Use the existing MPC without terminal value:

\[
J_0
=
\sum_{h=0}^{H-1}
\gamma^h\hat r_h
-
\lambda_u U
-
C.
\]

This is:

```text
reward-only MPC
```

Use it to collect real simulator episodes.

From those real episodes, calculate:

\[
G_t.
\]

Train:

\[
V_0(s_t)\approx G_t.
\]

## Phase 1+

Load \(V_0\) into MPC and use:

\[
J_1
=
\sum_{h=0}^{H-1}
\gamma^h\hat r_h
+
\alpha_V\gamma^H V_0(\hat s_H)
-
\lambda_u U
-
C.
\]

Collect improved trajectories.

Use the new real trajectories to update:

- the world model;
- the value model.

Then repeat.

# 14. Modify the MPC objective

Locate the existing candidate-scoring function.

Extend it without breaking existing behavior.

The planner must support:

```yaml
mpc:
  objective_mode: reward_plus_terminal_value
```

Supported modes:

```text
reward_only
terminal_value_only
reward_plus_terminal_value
```

Default:

```text
reward_plus_terminal_value
```

`terminal_value_only` is an ablation mode and should not be the normal configuration.

Implement:

```python
predicted_return = discounted_predicted_rewards

terminal_value = value_model(predicted_terminal_state)

objective = (
    predicted_return
    + terminal_value_coefficient
      * terminal_discount
      * terminal_value
    - uncertainty_penalty
    - constraint_penalties
)
```

where:

```python
terminal_discount = gamma ** effective_horizon
```

Do not forget the \(\gamma^H\) factor.

# 15. Correct terminal value under early predicted termination

If a candidate trajectory predicts termination before horizon \(H\), do not add the value of a state after termination.

For candidate \(n\), if predicted termination occurs at step \(k < H\):

\[
J_n
=
\sum_{h=0}^{k}
\gamma^h\hat r_h
\]

plus any applicable constraint terms.

Terminal value contribution should become:

\[
0
\]

after true predicted termination.

Implement proper masks.

Do not evaluate:

```text
V(predicted_state_after_goal)
```

as continuation value.

# 16. Effective horizon masking

If the existing world-model rollout returns:

```text
done probabilities
```

support configurable handling.

For example:

```yaml
mpc:
  terminal_handling: probability_weighted
```

Possible implementations:

## Hard threshold

If:

\[
P(done_h) > p_{\text{threshold}}
\]

consider the trajectory terminated.

## Survival weighting

Compute probability of surviving to the terminal horizon:

\[
P_{\text{survive},H}
=
\prod_{h=0}^{H-1}
(1-p_h).
\]

Then use approximately:

\[
J =
\sum_h
\gamma^h
P_{\text{survive},h}
\hat r_h
+
\gamma^H
P_{\text{survive},H}
V(s_H).
\]

Implement the simpler hard-threshold version if that matches the current MPC design.

If practical, add probability-weighted survival as an optional mode.

# 17. Terminal-value coefficient

Make terminal value strength configurable:

```yaml
mpc:
  terminal_value_coefficient: 1.0
```

Support values including:

```text
0.0
0.25
0.5
1.0
2.0
```

so it can be ablated.

Mathematically, `1.0` is the natural value when:

- value targets use the same gamma;
- reward scales match;
- value is calibrated correctly.

Do not automatically tune this coefficient inside the MPC code.

# 18. Value clipping and safety

A poorly trained value model can produce extreme estimates that dominate MPC.

Add optional robust protection:

```yaml
mpc:
  terminal_value:
    enabled: true
    clip_value: true
    clip_min: null
    clip_max: null
```

Prefer deriving reasonable clipping bounds from training return statistics.

For example, store training-return percentiles:

```text
1%
5%
50%
95%
99%
```

and optionally clamp value predictions to a configurable percentile range.

Log every clipping event.

Do not silently clip without diagnostics.

# 19. Value uncertainty / ensemble support

Initially one value model is acceptable.

However, design the interface so value ensembles can be added later.

Optional configuration:

```yaml
value_model:
  ensemble_size: 1
```

If `ensemble_size > 1`, support:

```text
mean terminal value
terminal value disagreement
```

and optionally:

\[
V_{\text{robust}}
=
\mathbb E[V]
-
\beta_V\operatorname{Std}[V].
\]

Do not make value-ensemble support block the initial implementation.

# 20. Distribution-shift protection

The MPC will evaluate predicted states that may differ from the states used to train \(V\).

This creates a second extrapolation problem.

At minimum:

- reuse world-model uncertainty;
- log terminal-state world-model uncertainty;
- log terminal-value magnitude;
- identify states with high model uncertainty and large terminal value.

Add an optional terminal-value attenuation:

\[
\tilde V(s_H)
=
w(U_H)V(s_H),
\]

where for example:

\[
w(U_H)
=
\exp(-\beta U_H).
\]

Configuration:

```yaml
mpc:
  terminal_value:
    uncertainty_gating: false
    uncertainty_beta: 1.0
```

Keep disabled initially.

Provide it as an experimental safeguard.

# 21. Preserve the existing MPC reward objective

Do not replace finite-horizon reward with terminal value.

Normal operation must remain:

\[
\boxed{
\text{finite-horizon predicted rewards}
+
\text{terminal value}
}
\]

not:

\[
\boxed{
\text{terminal value only}
}
\]

Reason:

Two plans can reach similar terminal states while having very different intermediate behavior.

Example:

```text
Plan A:
collision
lose ball
recover
reach good state

Plan B:
maintain possession
avoid collision
reach same good state
```

A terminal-value-only objective may treat these similarly.

Finite-horizon reward preserves preference for Plan B.

# 22. Planner result diagnostics

Extend the existing MPC result object to include:

```text
predicted_discounted_reward_return
terminal_state_value
discounted_terminal_value
terminal_value_coefficient
terminal_value_contribution

world_model_uncertainty_penalty

constraint_penalty
skill_switch_penalty
command_change_penalty

total_objective
```

Do not return only a scalar objective.

These components are required for debugging.

For the selected trajectory, log:

```text
reward contribution
terminal-value contribution
uncertainty contribution
constraint contribution
```

# 23. Teacher rollout collection

Modify the existing MPC teacher collector.

For each executed MPC step, store both prediction and real outcome.

Add:

```text
predicted_finite_horizon_return
predicted_terminal_value
discounted_terminal_value
terminal_value_contribution

selected_plan_total_objective

real_episode_return, once episode completes
real_return_to_go, during postprocessing
```

Continue storing:

```text
real state
real executed action
real reward
real next state
termination
```

Do not store predicted terminal values as value training targets.

Only real return-to-go is the supervised target for \(V\).

# 24. Iterative update loop

Modify or add an orchestration script such as:

```bash
python scripts/run_value_augmented_mpc_iteration.py \
    --config configs/value_augmented_mpc.yaml
```

The intended loop is:

```text
load current world model
load current terminal value model, if available

run MPC in simulator
collect real episodes

append real transitions to world-model training data

calculate real return-to-go
append state/return pairs to value-model training data

update world model

update value model

validate both models

evaluate updated MPC

accept/reject updated checkpoints
```

The value model and world model do not need to be updated at every environment step.

Use batch iterations.

# 25. Preserve old data

When updating the value model, do not train only on the newest MPC trajectories.

Maintain a replay mixture of:

```text
older reward-only MPC trajectories
older value-MPC trajectories
new value-MPC trajectories
rare successful episodes
rare failure episodes
```

Suggested configuration:

```yaml
value_model_update:
  sampling:
    old_data: 0.60
    newest_data: 0.30
    rare_events: 0.10
```

Adapt based on the existing dataset implementation.

This reduces catastrophic forgetting and excessive policy-specific value fitting.

# 26. Consider behavior-policy dependence

Remember that:

\[
V^\pi(s)
=
\mathbb E_\pi[
\sum_{k=0}^\infty
\gamma^k r_{t+k}
| s_t=s
].
\]

The value model trained from MPC trajectories estimates approximately:

\[
V^{\pi_{\text{MPC}}}(s).
\]

Document this explicitly.

The interpretation of terminal-value MPC is therefore:

```text
MPC explicitly optimizes the next H actions.

At the terminal predicted state,
V(s_H) estimates the continuation return
under the behavior represented in the value dataset.
```

This is acceptable.

Do not incorrectly document the learned model as guaranteed \(V^*\).

# 27. Optional fitted-value improvement

Prepare, but do not require, future support for more RL-like fitted value updates.

For example:

\[
V(s_t)
\leftarrow
r_t+\gamma V(s_{t+1}).
\]

Do not use this as the primary implementation yet.

Initial implementation:

```text
full real trajectory
→ Monte Carlo return
→ supervised V fitting
```

is preferred because it is simpler and less coupled to its own prediction errors.

# 28. Validate the terminal value model separately

Implement:

```bash
python scripts/evaluate_terminal_value.py \
    --checkpoint checkpoints/terminal_value/best.pt \
    --dataset data/mpc_teacher/test
```

Report:

- MSE;
- RMSE;
- MAE;
- Huber loss;
- explained variance;
- Pearson correlation between predicted value and real return-to-go;
- Spearman rank correlation;
- predicted versus actual return plots.

Also break down errors by:

```text
distance of ball to goal
possession status
skill currently active
goal-scoring episode vs non-scoring episode
episode stage
behavior source
```

when those labels are available.

# 29. Value calibration visualization

Create plots of:

```text
predicted V(s)
versus
actual return-to-go
```

Include:

- scatter plot;
- identity line;
- binned calibration curve;
- return target distribution;
- prediction distribution;
- residual distribution.

This is important because MPC is sensitive not just to ranking but to value scale.

# 30. Visualize terminal value spatially

For selected simulator states, create useful football-specific value visualizations where feasible.

For example:

- fix other state variables;
- vary ball position over a 2D field grid;
- evaluate \(V(s)\);
- display top-down value heatmap.

Possible visualizations:

```text
value vs ball position
value vs robot position
value vs ball distance to goal
value vs shot angle
value vs possession state
```

Do not fabricate physically inconsistent states if doing so breaks the environment representation.

Use only valid state perturbations.

# 31. Modify MPC visualization

Extend the existing MPC visualizer.

For the selected plan, show at least:

```text
Predicted immediate/future reward return: X
Terminal state value: Y
Discounted terminal value: Z
World-model uncertainty penalty: U
Constraint penalty: C

Total MPC objective: J
```

On the top-down field visualization, mark:

```text
current state
predicted trajectory
predicted terminal state
```

Clearly highlight the terminal state whose value is being evaluated.

Annotate:

```text
V(s_H) = ...
```

# 32. Plot value along the candidate trajectory

For the selected trajectory, optionally evaluate:

\[
V(\hat s_1),V(\hat s_2),...,V(\hat s_H)
\]

for diagnostics only.

The MPC objective should still normally use only:

\[
V(\hat s_H).
\]

Plot:

```text
step
predicted reward
predicted cumulative reward
predicted state value
world-model uncertainty
```

This helps identify whether the planner is moving toward increasingly valuable states.

# 33. Candidate visualization

For selected planning steps, show candidate trajectories with:

```text
finite-horizon reward
terminal value
total objective
uncertainty
```

This should make it possible to identify examples where:

```text
Candidate A:
higher immediate reward
lower terminal value

Candidate B:
lower immediate reward
much higher terminal value
```

and terminal-value-augmented MPC correctly selects B.

# 34. Validate the actual hypothesis

Implement an evaluation specifically testing whether terminal value helps limited-horizon MPC.

Compare:

```text
A. reward-only MPC
B. terminal-value-only MPC
C. reward + terminal-value MPC
```

Use identical:

- world model;
- environments;
- seeds;
- planning horizon;
- CEM candidate count;
- CEM iterations.

Test at horizons:

```text
1
3
5
8
10
```

Measure:

- average real simulator return;
- goal rate;
- successful-shot rate;
- possession;
- ball progress;
- collision rate;
- out-of-bounds rate;
- planning latency.

Expected hypothesis:

```text
terminal-value augmentation should provide
larger benefit when MPC horizon is short.
```

Do not assume this is true.

Measure it.

# 35. Horizon-vs-value ablation

Generate a result table conceptually like:

```text
Horizon | Reward MPC | Value-only MPC | Reward + Value MPC
-----------------------------------------------------------
1       | ...        | ...            | ...
3       | ...        | ...            | ...
5       | ...        | ...            | ...
8       | ...        | ...            | ...
10      | ...        | ...            | ...
```

Report:

```text
return
goal rate
collision rate
planning time
```

This is one of the main experiments.

# 36. Terminal coefficient ablation

Evaluate:

```text
alpha_V:
0.0
0.25
0.5
1.0
2.0
```

where:

```text
alpha_V = 0
```

is exactly reward-only MPC.

Plot:

```text
alpha_V vs real return
alpha_V vs goal rate
alpha_V vs collision rate
alpha_V vs predicted-value contribution
```

# 37. Check value exploitation

Add diagnostics for world-model/value exploitation.

Flag candidate plans where:

```text
terminal value is extremely high
AND
world-model uncertainty is high
```

Log:

```text
V(s_H)
terminal world-model uncertainty
distance from terminal state to training distribution, if available
```

Provide configurable warnings.

This is important because MPC may discover predicted states that cause unrealistically high value estimates.

# 38. Compare predicted terminal value with realized return

During real MPC execution, suppose planning occurs at time \(t\), and the selected predicted terminal state is at \(t+H\).

When sufficient real trajectory data later becomes available, compare:

\[
V(\hat s_{t+H})
\]

with realized continuation return.

Also compare value on the actual state:

\[
V(s_{t+H}^{real})
\]

against:

\[
G_{t+H}^{real}.
\]

This separates two error sources:

```text
world-model error:
predicted terminal state differs from real terminal state

value error:
V(real terminal state) differs from real future return
```

Report both separately.

# 39. Error decomposition

For evaluation episodes, calculate:

```text
world-model terminal-state prediction error

value error on actual terminal state

value difference caused by state prediction error:

V(predicted_terminal_state)
-
V(actual_terminal_state)
```

This decomposition is important for understanding why value-augmented MPC succeeds or fails.

# 40. Value model checkpoint selection

Select the best value checkpoint using held-out episode validation.

Primary validation metric:

```text
Huber or MSE on return-to-go
```

Also log:

```text
explained variance
rank correlation
```

Do not select based solely on training loss.

# 41. Iterative acceptance gate

When updating the value model, do not automatically replace the active model.

Compare new and old models on a fixed validation set.

Possible acceptance rules:

```yaml
value_model_acceptance:
  require_validation_improvement: true
  max_rmse_degradation_fraction: 0.02
  require_finite_predictions: true
```

Optionally also evaluate actual MPC performance on a small fixed validation environment set before accepting the new value model.

If rejected:

```text
keep previous active value checkpoint
save rejected checkpoint for debugging
```

# 42. MPC fallback

The planner should remain usable if:

- no terminal value checkpoint exists;
- the terminal value model cannot load;
- terminal value is explicitly disabled.

In these cases, fall back cleanly to:

```text
reward-only MPC
```

Do not crash unless the configuration explicitly requires a terminal value model.

# 43. Configuration

Add a configuration similar to:

```yaml
value_model:
  enabled: true

  gamma: 0.99

  target_type: monte_carlo
  bootstrap_on_truncation: true

  hidden_dims: [512, 512, 256]
  activation: silu
  layer_norm: true

  loss: huber
  learning_rate: 0.0003
  weight_decay: 0.00001

  batch_size: 1024
  max_epochs: 200

  normalize_targets: true

  ensemble_size: 1

mpc:
  objective_mode: reward_plus_terminal_value

  terminal_value:
    enabled: true

    checkpoint: null

    coefficient: 1.0

    clip_value: false

    uncertainty_gating: false
    uncertainty_beta: 1.0
```

Integrate this into the existing config system rather than creating conflicting config systems.

# 44. Testing

Add tests for:

1. discounted return calculation;
2. episode boundaries do not leak returns;
3. terminal returns correctly become zero after true termination;
4. truncation handling works;
5. value dataset split has no episode leakage;
6. value model output shape;
7. value checkpoint save/load;
8. return normalization/denormalization;
9. MPC with coefficient `0` exactly reproduces reward-only objective;
10. terminal value is multiplied by \(\gamma^H\);
11. terminal value is removed after predicted termination;
12. objective decomposition sums correctly;
13. value clipping works when enabled;
14. value augmentation changes candidate ranking in a toy example;
15. terminal-value-only mode works;
16. reward-plus-value mode works;
17. missing value checkpoint falls back to reward-only MPC when allowed;
18. visualization scripts run headlessly.

Create one particularly important toy test:

```text
Candidate A:
larger H-step reward
poor terminal state

Candidate B:
smaller H-step reward
excellent terminal state
```

Verify:

```text
reward-only MPC selects A

reward + terminal-value MPC selects B
```

when the configured value difference is sufficiently large.

# 45. Smoke test

Implement a smoke test that:

1. loads existing real trajectory data;
2. computes return-to-go;
3. trains a tiny terminal value model;
4. reloads the checkpoint;
5. loads the existing world model;
6. loads the existing MPC;
7. runs reward-only MPC;
8. runs value-augmented MPC from the same state;
9. prints objective decomposition;
10. executes several value-augmented MPC actions in the real simulator;
11. records real transitions;
12. creates a diagnostic plot.

# 46. Do not unnecessarily modify the world-model collector

Only change the collector where required to preserve value-learning information.

If the current collector already stores:

```text
episode boundaries
state
action
real reward
next state
done
```

then avoid rewriting it.

Prefer adding:

```text
return postprocessing
value-dataset export
```

instead of duplicating the collector.

# 47. Do not unnecessarily modify the world model

The existing world model does not need to predict state value.

Keep:

```text
f_phi = dynamics + reward prediction
```

and:

```text
V_psi = long-term return prediction
```

separate.

This allows the world model and value model to be:

- trained on different targets;
- updated at different rates;
- validated independently;
- replaced independently.

# 48. Recommended implementation order

Implement in this order.

## Stage A — inspect existing code

Identify the current:

```text
collector
dataset
world model
MPC planner
objective function
visualizer
```

## Stage B — return processing

Implement:

```text
episode return calculation
return-to-go calculation
value dataset
```

Validate thoroughly.

## Stage C — value model

Implement:

```text
TerminalValueModel
training
validation
checkpointing
```

## Stage D — modify MPC

Extend objective from:

\[
J =
\sum_{h=0}^{H-1}\gamma^h\hat r_h
-\lambda U
-C
\]

to:

\[
J =
\sum_{h=0}^{H-1}\gamma^h\hat r_h
+
\alpha_V\gamma^H V(\hat s_H)
-\lambda U
-C.
\]

## Stage E — diagnostics

Add:

```text
reward contribution
terminal-value contribution
uncertainty
constraints
```

to logs and plots.

## Stage F — iterative collection

Collect new real MPC trajectories and update:

```text
world model
terminal value model
```

from real simulator data.

## Stage G — evaluation

Compare:

```text
reward-only MPC
terminal-value-only MPC
reward + terminal-value MPC
```

especially for short planning horizons.

# 49. Acceptance criteria

This task is complete when:

1. existing trajectory datasets can produce correct return-to-go targets;
2. a terminal value network can be trained;
3. the value network predicts scalar continuation return from global state;
4. value checkpoints save and load correctly;
5. MPC can load the terminal value model;
6. reward-only MPC still works;
7. terminal-value-only MPC works for ablation;
8. reward-plus-terminal-value MPC works;
9. terminal value is discounted by \(\gamma^H\);
10. predicted terminated trajectories do not receive continuation value;
11. MPC logs objective decomposition;
12. teacher rollouts store terminal-value diagnostics;
13. real collected trajectories can update the value model;
14. real collected transitions can still update the world model;
15. reward-only and value-augmented MPC can be compared under identical seeds;
16. horizon ablations run;
17. terminal-value coefficient ablations run;
18. predicted value can be compared with realized return;
19. world-model error and value-model error can be separated;
20. visualization clearly marks the predicted terminal state and its value;
21. tests pass;
22. smoke test passes.

At the end, provide:

- the relevant existing repository architecture discovered;
- every modified file;
- every newly created file;
- exact command for computing value targets;
- exact command for training the terminal value model;
- exact command for reward-only MPC;
- exact command for value-augmented MPC;
- exact command for collecting new trajectories;
- exact command for evaluating value prediction;
- exact command for comparing MPC variants;
- exact command for visualization;
- assumptions made;
- known limitations;
- recommended next step for using these MPC trajectories to initialize MAPPO actor and critic.