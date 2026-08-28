# Migration status and next gates

## Architecture mapping

| Isaac Gym implementation | Isaac Lab manager-based module | Status |
| --- | --- | --- |
| `BaseTask.create_sim/_create_envs` | `InteractiveSceneCfg` | AS2, plane and ball implemented |
| `_compute_torques` | `ActionManager` + `IdealPDActuatorCfg` | Legacy PD law and static effort clipping implemented |
| custom `Sensor` classes | `ObservationManager` terms | Core terms plus exact 72D/75D legacy groups implemented |
| `SoccerRewards` method discovery | `RewardManager` terms | dribbling subset and shooting rewards implemented |
| `check_termination` | `TerminationManager` terms | ball limits and shooting phase state implemented |
| `reset_idx` and randomizers | `EventManager` + custom command term | dribble reset and command-relative shooting reset implemented |
| command curriculum | `CommandManager` / `CurriculumManager` | velocity and shooting commands implemented; curriculum pending |
| named physics rollout export | `parity/` plus Gym/Lab recorder adapters | schema-v2 physics + policy observation matrix passes 4/4 |
| `HistoryWrapper` | custom stateful observation term | 15-frame zero-padding and host rollout parity verified |
| `HighLevelSkillWrapper` | frozen-policy action term plus skill router | single-AS2 smoke works; 10-tick behavior matrix passes |
| low-level shooting + frozen checkpoint | `AS2FrozenShootingFlatEnvCfg` + `FrozenSkillPolicyAction` | manager composition and launch/timeout integration pass; success quality pending |
| 10-tick coordinator rate | `MacroActionWrapper` around `ManagerBasedRLEnv` | 10×20 ms aggregation and four-robot action contract smoke pass |
| four-AS2 match scene | `AS2MatchSceneCfg` + `AS2MatchActionsCfg` | 4 articulations, 24D manager action, `(4,34)` canonical local state smoke passes |
| shared-policy self-play | `MatchSelfPlayWrapper` + team-frame adapter | team-0 two-agent interface, π-mirrored opponent frame, checkpoint loader, and replaceable callable smoke pass |
| RSL-RL self-play boundary | `MatchSelfPlayRslRlVecEnvWrapper` + match PPO cfg | agent-expanded TensorDict adapter and static-opponent launcher implemented; RTX4090 resume from iteration 450 through 500 crossed the periodic detached snapshot cadence and wrote `model_500.pt`; resumed actors now snapshot at their restored iteration |
| canonical football snapshot | `MatchSnapshotRecorder` + `RecorderManager` | reset plus 10 low-level ticks archived with stable robot/ball/action shapes |
| geometric skill fallback | `FrozenSkillPolicyAction._apply_geometric_skill_fallback` | out-of-range shoot requests downgrade to walk and expose invalid mask |
| world-model state extraction | `IsaacLabFootballWorldModelStateAdapter` | canonical snapshot → existing 128D `StateSchema` encoder passes simulator-free and RTX4090 live smoke |
| `SharedPolicySelfPlayWrapper` | `AS2MatchSceneCfg` + `MatchSelfPlayWrapper` | four-AS2 scene, snapshot seam, team-frame parity seam, and basic match rewards smoke pass; exact reward/training parity pending |
| `FootballWorldModelStateAdapter` | `IsaacLabFootballWorldModelStateAdapter` | canonical snapshot encoding and recorder-episode transition seam implemented; MPC collector hook pending |

## Required parity gate for the current slice

The named archive format, deterministic excitation, both simulator adapters,
runtime asset-property capture, and offline comparison gates are implemented.
The final host run used Isaac Sim 5.1.0, PhysX, an RTX 4090, driver 570.169,
and the workstation's editable Isaac Lab checkout at
`v2.3.2-35-gbde69bb24ef-dirty` (`isaaclab==0.54.3`).
`outputs/host-validation/20260825T034900Z-policy-contract/` passes all four
zero/sine velocity/dribble comparisons with the final `IdealPDActuatorCfg`
model. It supersedes `20260824T151700Z` as the development baseline because it
also records the low-level policy observation and 15-frame history. This
validates the installed descendant of v2.3.2; a clean exact-tag v2.3.2 rerun is
still required for the release-reproducibility claim.

The two simulator actors now agree on 17 AS2 bodies, total mass
`17.6400001645 kg`, joint limits, velocity/effort limits, inertia, armature, and
friction. The manager startup events set and verify `contact_offset=0.01`,
`rest_offset=0.0`, and the legacy-compatible `base_link` COM `(0, 0, 0)` through
the initialized PhysX tensor views. The COM override is required because the
legacy actor callback writes its zero `com_displacements` buffer when COM
randomization is disabled, replacing the non-zero URDF inertial origin.

The invalid `20260824T150305Z` run must not be used as a regression baseline.
The old recorder called `config_as2(Cfg)` but did not perform the legacy entry
point's separate `Cfg.robot.name = "as2"` selection, so Gym loaded Go1 (23
bodies, about 11.31 kg). The recorder now sets the name explicitly and refuses
to continue unless the runtime body set contains `base_link`.

Completed parity checks:

1. Twelve actuated joints, four feet, body set, mass, inertia, COM, and joint properties.
2. Zero-action drift for base, joints, torque, contacts, and the ball.
3. Fixed sinusoidal joint-target phase lag, torque response, and contacts.
4. Matching 5 ms physics and 20 ms policy/control steps.
5. Ball state under the deterministic no-contact parity setup.
6. Legacy 72D/75D observations and zero-padded 1080D/1125D histories under direct joint actions.
7. Exact deterministic command, action-history, gait-clock, and gait-phase slices.

Still required before broad training comparisons:

1. Inspect the standing pose and self-collision visually in an interactive run.
2. Compare randomized ball reset distributions and longer rolling/contact decay.
3. Validate shooting success timing and post-strike decay against the original low-level task.
4. Only then run short PPO convergence comparisons for each migrated skill.

The first direct Isaac Gym comparison attempt is currently blocked by the
legacy Preview 4 runtime rather than by the migrated task: its Torch 1.10/cu113
NVRTC rejects the RTX4090 `sm_89` architecture (`invalid value for
--gpu-architecture`). The existing frozen behavior archives remain valid
because they were recorded before this runtime limitation surfaced; rerun the
Gym side in its supported CUDA/Torch environment or with a compatible legacy
GPU before treating the new success-rate report as a Gym-vs-Lab conclusion.

## Legacy observation compatibility gate

The compatibility path is intentionally separate from the current actor input:

- `base_velocity` remains the shared 3D task command used by rewards.
- `gait_parameters` owns the other 12 legacy command values and the gait phase.
- `legacy_policy` concatenates sensors in the old checkpoint order.
- `legacy_history` owns a `(num_envs, 15, obs_dim)` zero buffer instead of using
  Isaac Lab's circular history, which fills all slots with the first frame.
- The first external reset returns zero history. A normal step appends once,
  and an internally reset environment clears history before appending its reset
  observation.

Simulator-free tests and the host parity archive cover dimensions, ordering, scaling, noise masks, clock
mapping, phase wrapping, reset/append behavior, real TorchScript contract
inspection, and zero-history inference. A two-step Isaac Sim 5.1 smoke also
verifies finite legacy observations and frozen-policy joint targets for the 6D
Skill coordinator. Direct joint-action observation/history parity now passes
on the RTX4090. Still required against Isaac Gym:

1. Repeat with fixed seeds and noise enabled to compare distributions rather than
   individual random samples.
2. Run the frozen-policy skill action term and compare its actual consumed history/action sequence with
   the Isaac Gym wrapper for identical histories and coordinator inputs.

The behavior recorder resolved the action-history ambiguity as two explicit
contracts. `HighLevelSkillWrapper` copies
`last_low_level_actions = low_level_actions` immediately before constructing
each policy observation, so coordinator execution sees `[A(t-1), A(t-1)]`; the
manager action term defaults to this compatibility mode. The original low-level
Shooting/Dribble PPO tasks use `HistoryWrapper` and see successive outputs
`[A(t-1), A(t-2)]`; the frozen Shooting composition opts into
`action_history_semantics="successive_policy_outputs"`.

The `AS2-Skill` smoke task accepts a six-value action: three skill logits plus
three normalized command values. It selects walking, dribbling, or shooting,
updates the legacy command/history contract, executes the corresponding frozen
policy, and emits 12 joint-position targets. Invalid coordinator or policy rows
fail safe to zero low-level action. The default bundle is
`checkpoints/reproduction`; `DRIBBLEBOT_CHECKPOINT_ROOT` overrides that root.
This task deliberately keeps the dribbling rewards, so it is an integration
fixture rather than the final coordinator objective.

The gait parameters currently use the legacy fixed/default ranges with uniform
sampling. Their reward-threshold curriculum and gait/contact-shaped rewards are
not yet migrated.

## Next implementation order

1. Compare frozen Shooting success/failure timing and post-strike decay against Isaac Gym.
2. Validate match-specific goal, possession acquired/lost, pass, successful/failed
   shot, teammate-collision, and accidental-termination rewards.
3. Feed `extract_episode_transitions()` into the existing world-model dataset
   writer, then expose the same state adapter to the MPC runtime/controller.
4. Add historical opponent-pool persistence and sampling beyond the validated
   single periodic detached snapshot.
5. Convert validated URDF assets to versioned USD assets and rerun the exact clean
   Isaac Lab v2.3.2 release environment.

## Version policy

The production migration remains pinned to Isaac Sim 5.1.0, Isaac Lab v2.3.2,
and PhysX. Isaac Lab v2.3.2 is the stable 2.x baseline and supports Isaac Sim
4.5, 5.0, and 5.1; this repository chooses 5.1.0 so one exact runtime can be
reproduced. When validating, use the `v2.3.2` tag or its matching Isaac Lab
release environment: the local Isaac Lab checkout may be on a newer development
commit and should not be treated as the pinned runtime without an API check.
That distinction applies to the current workstation: its successful GPU run
is 35 commits past v2.3.2 and has local modifications.
Isaac Lab 3.0/Newton is still a separate, broader physics-backend transition.
The Warp 1.8.2 `warp.sim` deprecation warning announces a future library move;
it does not invalidate the current PhysX runtime. A Newton experiment should
therefore live in a separate branch and is not a production or parity target
until the manager-based PhysX pipeline reproduces every original skill and the
Newton backend has task-relevant feature/performance evidence.
