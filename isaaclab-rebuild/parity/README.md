# Physics parity archive

The recorders write compressed schema-versioned `.npz` files. Metadata is a
scalar JSON string and all signal arrays load with `allow_pickle=False`.

## Time and shape contract

- `time_s`: `(T+1,)`
- `action`: `(T, N, A)`
- `done`: `(T, N)`
- `env_origin_w`: `(N, 3)`
- root position, world/body velocity: `(T+1, N, 3)`
- root quaternion: `(T+1, N, 4)`, always `wxyz`
- joint position, velocity, target, torque: `(T+1, N, J)`
- foot position and force: `(T+1, N, F, 3)`
- foot contact: `(T+1, N, F)`
- optional ball pose and velocity: `(T+1, N, 3|4)`
- optional `legacy_policy_observation`: `(T, N, 72|75)`
- optional `legacy_policy_history`: `(T, N, 1080|1125)`

Metadata carries separate `joint_names`, `action_names`, and `foot_names`.
The comparison tool reorders the candidate by name before computing errors.
World positions are compared relative to `env_origin_w`, so simulator-specific
parallel-environment layouts do not affect the result.

Schema v2 also stores a required `physics_profile`. Both recorders force the
same gravity, solver/contact offsets, ground material, and—when present—ball
mass, radius, damping, velocity limits, material, and gyroscopic-force flag.
Comparison stops before computing trajectory metrics if these profiles differ.
This is necessary because the legacy environment otherwise randomizes ball
mass unconditionally during actor creation.

For the instanceable AS2 URDF, Isaac Lab applies `contact_offset` and
`rest_offset` with a manager-based startup event through the initialized PhysX
tensor view. The event immediately reads the values back, and the Lab recorder
performs a second check before saving. This avoids invalid USD edits on
instance proxies while retaining instancing for large vectorized scenes.

Schema-v2 metadata also captures the runtime body names, mass, COM, inertia,
joint limits, velocity/effort limits, drive gains, armature, friction, and the
Isaac Lab actuator model. The legacy recorder explicitly selects
`Cfg.robot.name = "as2"` and checks for `base_link`; this is separate from
`config_as2(Cfg)` in the original entry-point design. Both recorders now expose
the legacy-compatible root COM `(0, 0, 0)` rather than comparing Isaac Gym's
runtime override with Isaac Lab's untouched URDF inertial origin.

The optional `policy_contract` metadata describes every observation slice,
the fixed unscaled 15D command, and oldest-to-newest zero-padding. Policy
signals are step-aligned because the old sensor stack computes them after each
environment action. Both policy arrays must be present together; older
physics-only schema-v2 archives remain readable without them.

## Metrics and gates

The default acceptance gates cover:

- identical normalized action excitation and done state;
- root position, orientation, and world/body velocity;
- joint position, velocity, target, and applied torque;
- foot position and contact-state disagreement;
- optional ball position and linear velocity.
- complete legacy policy observation and 15-frame history;
- exact command, current/previous action, gait clock, and gait phase slices.

Foot normal impulse and absolute actuator work are always reported as
diagnostics. They are not default hard gates because Isaac Gym reports the full
net contact force while Isaac Lab v2.3.2 `ContactSensor.net_forces_w` reports
normal contact force only.

Pass a JSON object with `--thresholds` to replace all default gates. Metric
names and their measured values are present in every comparison report.

## Accepted host archive

`outputs/host-validation/20260825T034900Z-policy-contract/` is the accepted RTX 4090 baseline
for the currently installed runtime: Isaac Sim 5.1.0, PhysX, and an editable
Isaac Lab checkout at `v2.3.2-35-gbde69bb24ef-dirty` with the final
`IdealPDActuatorCfg` implementation. Its manifest reports four physics plus
policy-contract passes:

- `velocity-zero`
- `velocity-sine`
- `dribble-zero`
- `dribble-sine`

The robot zero/sine metrics are identical between the velocity and dribble
runs because the deterministic ball does not contact the robot in this parity
fixture. Ball position RMSE is `5.96e-7 m` in both dribble comparisons.
Zero-rollout policy observation/history RMSE stays below `0.00164`; sine
rollouts stay below `0.03050`, inside the `0.05` gates. Deterministic command,
action, clock, and phase slices pass their `1e-6` gates.

This archive is suitable for continued development on the workstation. Before
calling it the release-pinned baseline, repeat the matrix in a clean exact-tag
Isaac Lab v2.3.2 environment and archive that interpreter/commit identity.

Do not use `outputs/host-validation/20260824T150305Z/`. Its Gym archives were
recorded with the default Go1 adapter instead of AS2, so its sine failures do
not represent a simulator-backend difference.

`20260824T151700Z` correctly compares AS2 physics but contains no
`legacy_policy_observation` or `legacy_policy_history` arrays. It cannot prove
checkpoint-input parity. The current policy-contract archive also does not
exercise frozen TorchScript inference; that requires the separate behavior
trace described in `MIGRATION.md`.

## Frozen behavior trace

`behavior_schema.py` defines the separate frozen-policy contract. Each archive
records the coordinator action, selected skill, physical command, exact 75D
frame and 15-frame history consumed by TorchScript, raw 12D policy action,
processed joint target, checkpoint SHA256 identities, and root/joint/ball state.
`compare_skill_behaviors.py` applies a 10-policy-tick contract horizon; longer
traces remain diagnostics so accepted interface parity is not turned into a
long-horizon PhysX gate.

The accepted matrix is `outputs/skill-behavior/20260825T042000Z/` and passes
walk, dribble, shoot, and 3-tick switch modes. The action-history metadata is
explicit:

- `duplicate_previous_policy_output`: high-level wrapper compatibility,
  `[A(t-1), A(t-1)]`;
- `successive_policy_outputs`: original low-level PPO/HistoryWrapper contract,
  `[A(t-1), A(t-2)]`.

The frozen Shooting manager task selects the second mode and composes shooting
command/reset/reward/termination managers. Its RTX4090 report is
`outputs/frozen-shooting/20260825T044000Z/report.json`: all 8 episodes launched
without invalid input or non-shooting termination, but none met the current
success distance/speed gate. That is an open behavioral-quality gate, not an
ActionManager wiring failure.

## Coordinator macro rate

`football/macro.py` keeps the rate seam outside the manager implementation. The
raw `ManagerBasedRLEnv` continues to advance one 20 ms policy tick with its
normal ActionManager, RewardManager, TerminationManager, and reset order.
`MacroActionWrapper` calls that raw step ten times, sums only rewards from rows
that were active at the beginning of each low-level tick, ORs terminated and
truncated signals, and reports `elapsed_low_level_steps`. The accepted smoke
task is `Isaac-DribbleBot-AS2-Coordinator-Macro-Flat-Play-v0`.

The match staging task adds four independent AS2 terms and emits a canonical
`(N,4,34)` local-state tensor. `MatchSelfPlayWrapper` reshapes team 0 into
`N*2` agents with four-frame histories and accepts an opponent callable; it is
the manager-based replacement seam for the legacy shared-policy wrapper. It
does not yet claim parity for opponent snapshots or high-level match rewards.

The match config enables `MatchSnapshotRecorder` with export disabled by
default, so collectors can consume the in-memory `EpisodeData` without forcing
HDF5 output. The smoke archive records stable keys for robot root/joint state,
joint targets, ball root state, skill IDs, requested IDs, invalid masks, and
physical skill commands.

`world_model_adapter.py` consumes the same snapshot dictionary and maps it into
the repository's existing `default_state_schema(num_robots=4)`; this keeps
model code independent of Isaac Gym actor tensors. The adapter is intentionally
separate from RecorderManager so offline datasets and live MPC can share the
same canonical fields.

The host contract report `outputs/macro-validation/match-contract.json` now
also verifies live encoding into the existing 128D four-robot world-model
state schema.
