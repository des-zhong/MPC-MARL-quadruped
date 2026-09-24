"""Evaluate an AS2 high-level multi-robot soccer policy."""

import argparse
import csv
import pickle
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import isaacgym

assert isaacgym
import imageio
import numpy as np
import torch
from tqdm import trange

from quadruped.envs.base.legged_robot_config import Cfg
try:
    from scripts.play_walk_dribble_shoot import (
        load_policy_record,
        resolve_wandb_policy_files,
    )
except ModuleNotFoundError as error:
    if error.name != "scripts.play_walk_dribble_shoot":
        raise
    # Support lightweight checkouts where the legacy loader lives in discard/.
    from discard.play_walk_dribble_shoot import (
        load_policy_record,
        resolve_wandb_policy_files,
    )
from scripts.train_high_level import (
    add_skill_policy_source_args,
    configure_high_level_cfg,
    high_level_checkpoint_contract,
    validate_high_level_evaluation_contract,
    load_skill_policies,
)


SKILL_NAMES = ("walk", "dribble", "shoot", "stop")
TERMINAL_KEYS = (
    "high_level_goal",
    "high_level_opponent_goal",
    "high_level_ball_off_border",
    "high_level_obstacle_contact",
    "high_level_accidental_termination",
    "high_level_opponent_accidental_termination",
    "high_level_learning_team_failure",
    "high_level_opponent_team_failure",
)


def set_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_domain_randomization():
    false_flags = [
        "randomize_rigids_after_start",
        "randomize_friction_indep",
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "randomize_gravity",
        "randomize_ground_friction",
        "randomize_motor_strength",
        "randomize_motor_offset",
        "randomize_ball_drag",
        "randomize_lag_timesteps",
        "push_robots",
    ]
    for name in false_flags:
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, False)
    if hasattr(Cfg.domain_rand, "lag_timesteps"):
        Cfg.domain_rand.lag_timesteps = 0


def configure_eval_cfg(args):
    configure_high_level_cfg(Cfg, args)
    # Restore the saved environment contract before applying playback-only
    # controls. In particular, do not evaluate old policies with today's reset
    # ranges or curriculum defaults.
    for section, values in getattr(args, "saved_training_cfg", {}).items():
        target = getattr(Cfg, section)
        for name, value in values.items():
            if section == "env" and name in {
                "num_envs", "record_video", "num_train_envs", "num_eval_envs"
            }:
                continue
            setattr(target, name, value)
    Cfg.env.high_level_allow_near_ball_reposition = bool(
        getattr(args, "allow_near_ball_reposition", False))
    Cfg.env.num_envs = args.num_envs
    Cfg.env.record_video = not args.no_video
    Cfg.env.randomize_match_init = not args.fixed_init
    if getattr(args, "random_init", False):
        # Apply after restoring training settings: both reset mechanisms can
        # otherwise replace normal randomized starts with easy attacking starts.
        Cfg.env.randomize_match_init = True
        Cfg.env.high_level_near_ball_init_probability = 0.0
        Cfg.env.soccer_curriculum = False
    Cfg.env.require_learning_near_ball_init = bool(getattr(args, "random_learning_start", False))
    if Cfg.env.require_learning_near_ball_init:
        Cfg.env.randomize_match_init = True
        Cfg.env.deterministic_match_init = False
        Cfg.env.fair_match_init = False
        Cfg.env.soccer_curriculum = False
        Cfg.env.high_level_near_ball_init_probability = 1.0
        Cfg.env.high_level_near_ball_init_team = "learning"
        Cfg.env.high_level_near_ball_init_distance_range = [
            args.near_ball_init_min_distance, args.near_ball_init_max_distance]
        Cfg.env.high_level_near_ball_init_angle_range = [
            -args.near_ball_init_max_angle, args.near_ball_init_max_angle]
    Cfg.env.add_field_markers = not args.no_field_markers
    Cfg.env.add_boundary_walls = bool(
        getattr(args, "boundary_walls", True)
    )
    if args.camera_height is not None:
        Cfg.env.high_level_camera_height = args.camera_height
    if args.recording_fov is not None:
        Cfg.env.recording_horizontal_fov = args.recording_fov
    if not args.domain_rand:
        disable_domain_randomization()
    if getattr(args, "fair_match_init", False):
        from quadruped.envs.fair_match_init import configure_fair_match
        configure_fair_match(Cfg)


def add_high_level_policy_source_args(parser):
    """Add explicit high-level coordinator checkpoint source options."""

    parser.add_argument(
        "--high-level-policy-source",
        choices=("wandb", "local"),
        default=None,
        help=(
            "Load the high-level coordinator from --high-level-wandb-run or "
            "directly from --high-level-policy-dir."
        ),
    )
    parser.add_argument(
        "--high-level-wandb-run",
        default=None,
        help=(
            "High-level W&B run URL or entity/project/run_id. Required when "
            "--high-level-policy-source=wandb."
        ),
    )
    parser.add_argument(
        "--high-level-policy-dir",
        default=None,
        help=(
            "Local high-level checkpoint directory, W&B run directory, or "
            "files/tmp/legged_data directory. Required when "
            "--high-level-policy-source=local."
        ),
    )
    return parser


def load_high_level_policy(args):
    source_kind = getattr(args, "high_level_policy_source", None)
    if source_kind is None:
        source_kind = "local" if args.high_level_policy_dir else "wandb"
        args.high_level_policy_source = source_kind
    source_kind = str(source_kind)
    if source_kind == "wandb":
        if not args.high_level_wandb_run:
            raise ValueError(
                "--high-level-policy-source wandb requires --high-level-wandb-run"
            )
        (
            body_path,
            adaptation_module_path,
            ac_weights_path,
            run_path,
            policy_metadata,
        ) = resolve_wandb_policy_files(
            args.high_level_wandb_run,
            args.high_level_checkpoint,
            skill="high_level",
            return_metadata=True,
        )
        source_label = f"W&B {run_path}@{args.high_level_checkpoint}"
    elif source_kind == "local":
        if not args.high_level_policy_dir:
            raise ValueError(
                "--high-level-policy-source local requires --high-level-policy-dir"
            )
        from scripts.playback_utils import (
            build_policy_metadata,
            find_policy_config_path,
            resolve_ac_weights_file,
            resolve_policy_files,
        )

        policy_dir = Path(args.high_level_policy_dir).expanduser().resolve()
        if not policy_dir.is_dir():
            raise FileNotFoundError(
                f"Local high-level policy directory does not exist: {policy_dir}"
            )
        local_checkpoint = (
            "latest"
            if args.high_level_checkpoint in ("latest", "last")
            else str(args.high_level_checkpoint)
        )
        body_path, adaptation_module_path = resolve_policy_files(
            policy_dir,
            local_checkpoint,
        )
        ac_weights_path = resolve_ac_weights_file(
            policy_dir,
            local_checkpoint,
            policy_dir=body_path.parent,
        )
        config_path = find_policy_config_path(body_path, (policy_dir.parent,))
        policy_metadata = build_policy_metadata(
            body_path,
            adaptation_module_path,
            ac_weights_path,
            checkpoint=args.high_level_checkpoint,
            config_path=config_path,
        )
        source_label = (
            f"local directory {policy_dir}@{args.high_level_checkpoint}"
        )
    else:
        raise ValueError(
            f"Unsupported high_level_policy_source={source_kind!r}; "
            "expected 'wandb' or 'local'."
        )

    return load_policy_record(
        "high-level",
        body_path,
        adaptation_module_path,
        ac_weights_path,
        "high_level",
        args.policy_device,
        source_label,
        policy_metadata=policy_metadata,
    )


def resolve_opponent_weights_path(args, high_level_policy):
    """Resolve a frozen self-play opponent checkpoint from a path or suffix."""

    checkpoint = getattr(args, "opponent_high_level_checkpoint", None)
    if checkpoint is None:
        return None

    requested = Path(str(checkpoint)).expanduser()
    if requested.is_file():
        return requested.resolve()

    suffix = "latest" if str(checkpoint) in ("latest", "last") else str(checkpoint)
    filename = (
        suffix
        if suffix.startswith("opponent_ac_weights_") and suffix.endswith(".pt")
        else f"opponent_ac_weights_{suffix}.pt"
    )
    configured_dir = getattr(args, "opponent_high_level_policy_dir", None)
    roots = []
    if configured_dir:
        roots.append(Path(configured_dir).expanduser().resolve())
    else:
        roots.append(Path(high_level_policy["body_path"]).resolve().parent)
        high_level_dir = getattr(args, "high_level_policy_dir", None)
        if high_level_dir:
            roots.append(Path(high_level_dir).expanduser().resolve())

    candidate_dirs = []
    for root in roots:
        candidate_dirs.extend(
            (
                root,
                root / "high_level",
                root / "tmp" / "legged_data" / "high_level",
                root / "files" / "tmp" / "legged_data" / "high_level",
            )
        )
    for directory in candidate_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(path / filename) for path in candidate_dirs)
    raise FileNotFoundError(
        f"Could not find frozen opponent checkpoint {checkpoint!r}. Searched: {searched}"
    )


def load_opponent_high_level_policy(args, high_level_policy):
    """Load a frozen opponent snapshot, or mirror the learning-side policy."""

    weights_path = resolve_opponent_weights_path(args, high_level_policy)
    if weights_path is None:
        return high_level_policy

    device = args.policy_device
    # The frozen opponent may have been trained with a different observation
    # history layout.  In particular, old checkpoints can use a flattened
    # four-frame history (136 inputs), while the current policy export may use
    # a single-frame history (34 inputs).  Loading the current JIT module and
    # then copying the opponent state into it produces an opaque size-mismatch
    # error.  Prefer exports next to the opponent weights so the module shape
    # always matches the checkpoint that is being loaded.
    opponent_dir = weights_path.parent
    prefix = "opponent_ac_weights_" if weights_path.stem.startswith("opponent_ac_weights_") else "ac_weights_"
    token = weights_path.stem[len(prefix) :]
    opponent_adaptation = opponent_dir / f"adaptation_module_{token}.jit"
    opponent_body = opponent_dir / f"body_{token}.jit"
    if not opponent_adaptation.is_file():
        opponent_adaptation = opponent_dir / "adaptation_module_latest.jit"
    if not opponent_body.is_file():
        opponent_body = opponent_dir / "body_latest.jit"
    adaptation_path = (
        opponent_adaptation
        if opponent_adaptation.is_file()
        else Path(high_level_policy["adaptation_module_path"])
    )
    body_path = (
        opponent_body if opponent_body.is_file() else Path(high_level_policy["body_path"])
    )
    adaptation_module = torch.jit.load(str(adaptation_path), map_location=device).eval()
    body = torch.jit.load(str(body_path), map_location=device).eval()
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Opponent checkpoint must contain a state dictionary, got {type(state_dict).__name__}"
        )

    adaptation_state = {
        key[len("adaptation_module.") :]: value
        for key, value in state_dict.items()
        if key.startswith("adaptation_module.")
    }
    body_state = {
        key[len("actor_body.") :]: value
        for key, value in state_dict.items()
        if key.startswith("actor_body.")
    }
    if not adaptation_state or not body_state:
        raise ValueError(
            "Opponent checkpoint is missing adaptation_module.* or actor_body.* weights: "
            f"{weights_path}"
        )
    adaptation_module.load_state_dict(adaptation_state, strict=True)
    body.load_state_dict(body_state, strict=True)
    expected_history_dim = next(
        value.shape[1] for value in adaptation_state.values() if value.ndim == 2
    )
    from scripts.playback_utils import find_policy_config_path
    opponent_config = find_policy_config_path(body_path, (opponent_dir.parent,))
    opponent_metadata = {"config_path": str(opponent_config)}
    opponent_contract = high_level_checkpoint_contract({"policy_metadata": opponent_metadata})
    candidate_contract = high_level_checkpoint_contract(high_level_policy) or {}
    # The simulator shares these execution settings across teams. Encoding and
    # history length, however, are per-policy and can differ without approximation.
    for key in ("shooting_options", "control_interval", "walk_scale", "dribble_scale",
                "shoot_scale", "discrete_command_fraction"):
        left, right = candidate_contract.get(key), opponent_contract.get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"Candidate/opponent execution setting {key} differs: {left} vs {right}")
    encoding = opponent_contract.get("action_encoding", "hybrid")
    obs_width = 34 + int(bool(opponent_contract.get("shooting_options"))) + int(encoding == "discrete_skill_direction")
    history_length = opponent_contract.get("history_length") or args.high_level_history
    if expected_history_dim != obs_width * history_length:
        raise ValueError(f"Opponent config predicts {obs_width * history_length} history values, weights expect {expected_history_dim}")

    def policy(obs):
        obs_history = obs["obs_history"].to(device)
        latent = adaptation_module(obs_history)
        return body(torch.cat((obs_history, latent), dim=-1))

    print("opponent high-level:")
    print(f"  frozen actor-critic weights: {weights_path}")
    return {
        **high_level_policy,
        "policy": policy,
        "ac_weights_path": weights_path,
        "expected_history_dim": expected_history_dim,
        "body_path": body_path,
        "adaptation_module_path": adaptation_path,
        "source": f"frozen opponent checkpoint {weights_path}",
        "policy_metadata": opponent_metadata,
        "opponent_layout": (encoding, history_length),
    }


def load_opponent_pool(args, env, pool_path):
    """Load a serialized self-play opponent pool into ``env``.

    Pool snapshots use the same inference state-dict format as the training
    wrapper.  The actor template is created from the already constructed
    evaluation environment, so the pool's history/action dimensions are
    checked by ``load_opponent_pool_state_dict``.
    """
    from quadruped_learn.ppo_cse.actor_critic import AC_Args, ActorCritic
    from quadruped.envs.wrappers.shared_self_play_wrapper import FrozenOpponentPolicy

    try:
        payload = torch.load(str(pool_path), map_location=args.policy_device,
                             weights_only=True)
    except (TypeError, RuntimeError, pickle.UnpicklingError):
        payload = torch.load(str(pool_path), map_location=args.policy_device)
    if not isinstance(payload, dict) or not payload.get("policies"):
        raise ValueError(f"Opponent pool contains no policies: {pool_path}")
    policies = payload["policies"]
    first = policies[0]
    history_dim = next(
        value.shape[1] for key, value in first.items()
        if key.startswith("adaptation_module.") and value.ndim == 2
    )
    action_biases = [
        value for key, value in first.items()
        if key.startswith("actor_body.") and key.endswith(".bias")
    ]
    if not action_biases:
        raise ValueError(f"Opponent pool has no actor output bias: {pool_path}")
    action_dim = action_biases[-1].numel()
    opponent_obs_dim = history_dim // env.history_length
    opponent_discrete = action_dim == 12
    AC_Args.discrete_skill_direction_policy = opponent_discrete
    AC_Args.hybrid_skill_policy = not opponent_discrete
    AC_Args.skill_action_stride = 12 if opponent_discrete else 6
    AC_Args.num_skill_logits = 4 if opponent_discrete else 3
    AC_Args.num_direction_logits = 8 if opponent_discrete else 0
    AC_Args.stop_skill_id = 3
    template = ActorCritic(
        opponent_obs_dim,
        history_dim,
        history_dim,
        action_dim,
    ).to(args.policy_device).eval()
    frozen = []
    for state_dict in policies:
        snapshot = FrozenOpponentPolicy(template).to(args.policy_device).eval()
        snapshot.load_state_dict(state_dict, strict=False)
        frozen.append(snapshot)

    # The normal wrapper can execute only the candidate's action encoding. For
    # a discrete opponent, pad its extra observation feature and translate the
    # sampled one-hot action into the candidate's regular six-value layout.
    env.opponent_pool_size = len(frozen)
    env.opponent_pool = frozen
    env.opponent_pool_iterations = list(payload.get("iterations", range(len(frozen))))
    env._sample_opponent_assignments()
    if opponent_obs_dim != env.num_obs or action_dim != env.num_actions:
        def opponent_callable(observation):
            history = observation["obs_history"].to(args.policy_device)
            if opponent_obs_dim != env.num_obs:
                history = history.view(-1, env.history_length, env.num_obs)
                if opponent_obs_dim > env.num_obs:
                    extra = history.new_zeros(
                        history.shape[0], env.history_length,
                        opponent_obs_dim - env.num_obs,
                    )
                    history = torch.cat((history, extra), dim=-1)
                else:
                    history = history[..., :opponent_obs_dim]
                history = history.reshape(-1, history_dim)
            selected = []
            for index, policy in enumerate(frozen):
                mask = env.opponent_assignment.repeat_interleave(env.team_size) == index
                actions = torch.zeros(history.shape[0], action_dim, device=history.device)
                if bool(mask.any()):
                    actions[mask] = policy.act_training(history[mask])
                selected.append(actions)
            actions = selected[-1]
            for index, candidate in enumerate(selected):
                mask = env.opponent_assignment.repeat_interleave(env.team_size) == index
                actions = torch.where(mask[:, None], candidate, actions)
            if action_dim == 12 and env.num_actions == 6:
                grouped = actions.view(-1, 12)
                skill = grouped[:, :4].argmax(-1)
                direction = grouped[:, 4:12].argmax(-1)
                regular = torch.zeros(grouped.shape[0], 6, device=grouped.device)
                regular[:, :3].scatter_(1, skill.clamp_max(2)[:, None], 1.0)
                vectors = grouped.new_tensor(((1, 0), (0.7071, 0.7071), (0, 1),
                                               (-0.7071, 0.7071), (-1, 0),
                                               (-0.7071, -0.7071), (0, -1),
                                               (0.7071, -0.7071)))
                regular[:, 3:5] = vectors[direction] * 0.5
                return regular
            if action_dim == 6 and env.num_actions == 12:
                # Regular opponent -> discrete candidate environment.
                grouped = actions.view(-1, 6)
                discrete = torch.zeros(grouped.shape[0], 12, device=grouped.device)
                discrete[:, :3] = grouped[:, :3]
                planar = grouped[:, 3:5]
                vectors = grouped.new_tensor(((1, 0), (0.7071, 0.7071), (0, 1),
                                               (-0.7071, 0.7071), (-1, 0),
                                               (-0.7071, -0.7071), (0, -1),
                                               (0.7071, -0.7071)))
                discrete[:, 4:12] = planar @ vectors.t()
                return discrete
            return actions
        env.set_opponent_callable(opponent_callable)
    else:
        env.opponent_policy_callable = None
    print(f"opponent pool: {pool_path} ({len(env.opponent_pool)} policies)")


def make_env(args, skill_policies):
    from quadruped.envs.as2.two_robot_velocity_tracking import TwoRobotVelocityTrackingEasyEnv
    from quadruped.envs.wrappers.high_level_skill_wrapper import HighLevelSkillWrapper
    from quadruped.envs.wrappers.shared_self_play_wrapper import SharedPolicySelfPlayWrapper

    configure_eval_cfg(args)
    raw_env = TwoRobotVelocityTrackingEasyEnv(sim_device=args.device, headless=args.headless, cfg=Cfg)
    match_env = HighLevelSkillWrapper(raw_env, skill_policies)
    env = SharedPolicySelfPlayWrapper(
        match_env,
        team_size=args.num_robots,
        opponent_device=args.policy_device,
    )
    return env, raw_env


def validate_high_level_obs_shape(policy_record, env):
    expected_dim = policy_record.get("expected_history_dim")
    if expected_dim is None or expected_dim == env.num_obs_history:
        return

    raise ValueError(
        f"High-level policy expects obs_history dim {expected_dim}, "
        f"but this eval env provides {env.num_obs_history} "
        f"({env.num_obs} obs x {env.history_length} history). "
        "Use matching --high-level-history/config settings, or retrain/load a checkpoint "
        "that matches the current high-level observation layout."
    )

def validate_low_level_skill_shapes(skill_policies, env):
    from scripts.playback_utils import validate_ball_skill_command_frame

    full_history_dim = env.low_level_obs_dim_full * env.low_level_history_length
    no_object_history_dim = (env.low_level_obs_dim_full - 3) * env.low_level_history_length

    valid_dims = {
        "ball": {full_history_dim},
        "walking": {full_history_dim, no_object_history_dim},
    }
    valid_dims["walk"] = valid_dims["walking"]
    valid_dims["dribble"] = valid_dims["ball"]
    valid_dims["shoot"] = valid_dims["ball"]

    errors = []
    for skill_name, policy_record in skill_policies.items():
        validate_ball_skill_command_frame(
            skill_name,
            policy_record.get("policy_metadata"),
            source=policy_record.get("source"),
        )
        expected_dim = policy_record.get("expected_history_dim")
        if expected_dim is None:
            continue

        policy_type = policy_record.get("policy_type", skill_name)
        allowed = valid_dims.get(policy_type, valid_dims.get(skill_name, {full_history_dim, no_object_history_dim}))
        if expected_dim in allowed:
            continue

        errors.append(
            f"{skill_name} ({policy_record.get('source', 'unknown source')}) expects obs_history dim {expected_dim}, "
            f"but a low-level {skill_name} policy must expect one of {sorted(allowed)}. "
            f"Full low-level history is {full_history_dim}; walking/no-object history is {no_object_history_dim}."
        )

    if errors:
        raise ValueError(
            "Invalid low-level skill checkpoint(s):\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\nThis usually means a high-level coordinator run was passed as --walk-wandb-run, "
            "--dribble-wandb-run, or --shoot-wandb-run. Put the coordinator run only in "
            "--high-level-wandb-run."
        )


def skill_name(skill_id):
    skill_id = int(skill_id)
    if 0 <= skill_id < len(SKILL_NAMES):
        return SKILL_NAMES[skill_id]
    return f"unknown_{skill_id}"


def info_array(info, key, shape, default=0, env_index=0):
    value = info.get(key)
    if value is None:
        return np.full(shape, default)
    array = np.asarray(value)
    if array.shape == shape and env_index == 0:
        return array
    if len(shape) > 0 and shape[0] == 1 and array.shape[:1] != ():
        if env_index >= array.shape[0]:
            raise IndexError(
                f"{key} has {array.shape[0]} environment rows; "
                f"cannot select env_index={env_index}"
            )
        sliced = array[env_index : env_index + 1]
        if sliced.shape == shape:
            return sliced
    return np.reshape(array, shape)


def learning_team_array(info, key, num_robots, default=0.0, env_index=0):
    """Return a learning-team field aligned with all physical robot slots.

    Self-play exposes some coordinator telemetry for every robot in the match,
    but ``high_level_local_role_rewards`` is intentionally computed only for
    the learning team.  Keep the opponent slots at the neutral default when
    exporting per-robot evaluation rows.
    """
    shape = (1, num_robots)
    value = info.get(key)
    result = np.full(shape, default, dtype=np.asarray(default).dtype)
    if value is None:
        return result

    array = np.asarray(value)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(
            f"{key} must be a one- or two-dimensional array, got shape {array.shape}"
        )

    if env_index >= array.shape[0]:
        raise IndexError(
            f"{key} has {array.shape[0]} environment rows; "
            f"cannot select env_index={env_index}"
        )
    array = array[env_index : env_index + 1]
    rows = min(result.shape[0], array.shape[0])
    cols = min(result.shape[1], array.shape[1])
    result[:rows, :cols] = array[:rows, :cols]
    return result


def collect_state(raw_env):
    roots = raw_env.root_states[raw_env.robot_actor_idxs_all.reshape(-1)].view(
        raw_env.num_envs,
        raw_env.num_robots,
        13,
    )
    robot_xy = roots[:, :, :2] - raw_env.env_origins[:, None, :2]
    robot_vel = roots[:, :, 7:9]
    ball_xy = raw_env.object_pos_world_frame[:, :2] - raw_env.env_origins[:, :2]
    ball_vel = raw_env.object_lin_vel[:, :2]
    robot_ball_dist = torch.norm(roots[:, :, :2] - raw_env.object_pos_world_frame[:, None, :2], dim=-1)
    obstacle_xy = None
    if getattr(raw_env, "num_static_opponents", 0) > 0:
        obstacle_states = raw_env.root_states[raw_env.static_opponent_actor_idxs.reshape(-1)].view(
            raw_env.num_envs,
            raw_env.num_static_opponents,
            13,
        )
        obstacle_xy = obstacle_states[:, :, :2] - raw_env.env_origins[:, None, :2]

    return {
        "robot_xy": robot_xy.detach().cpu().numpy(),
        "robot_vel": robot_vel.detach().cpu().numpy(),
        "ball_xy": ball_xy.detach().cpu().numpy(),
        "ball_vel": ball_vel.detach().cpu().numpy(),
        "robot_ball_dist": robot_ball_dist.detach().cpu().numpy(),
        "obstacle_xy": None if obstacle_xy is None else obstacle_xy.detach().cpu().numpy(),
    }


def row_from_step(
    step,
    high_level_dt,
    state,
    action,
    reward,
    done,
    info,
    env_index=0,
):
    num_robots = state["robot_xy"].shape[1]
    requested = info_array(
        info,
        "high_level_requested_skill_ids",
        (1, num_robots),
        0,
        env_index,
    ).astype(np.int64)
    executed = info_array(
        info, "high_level_skill_ids", (1, num_robots), 0, env_index
    ).astype(np.int64)
    invalid = info_array(
        info, "high_level_invalid_skill_mask", (1, num_robots), False, env_index
    ).astype(bool)
    avoidance = info_array(
        info,
        "high_level_collision_avoidance_mask",
        (1, num_robots),
        False,
        env_index,
    ).astype(bool)
    attacker = info_array(
        info, "high_level_attacker_mask", (1, num_robots), False, env_index
    ).astype(bool)
    role_conflict = info_array(
        info, "high_level_role_conflict_mask", (1, num_robots), False, env_index
    ).astype(bool)
    command_assist = info_array(
        info,
        "high_level_attacker_command_assist_mask",
        (1, num_robots),
        False,
        env_index,
    ).astype(bool)
    local_role_reward = learning_team_array(
        info, "high_level_local_role_rewards", num_robots, 0.0, env_index
    ).astype(np.float32)
    commands = info_array(
        info, "high_level_commands", (1, num_robots, 3), 0.0, env_index
    ).astype(np.float32)
    action_np = action.detach().cpu().numpy()
    reward_np = reward.detach().cpu().numpy()
    done_np = done.detach().cpu().numpy().astype(bool)

    team_size = max(1, num_robots // 2)
    learner_index = env_index * team_size
    ball_xy = state["ball_xy"][env_index]
    ball_vel = state["ball_vel"][env_index]
    robot_xy = state["robot_xy"][env_index]
    robot_vel = state["robot_vel"][env_index]
    robot_ball_dist = state["robot_ball_dist"][env_index]
    obstacle_xy = state["obstacle_xy"]
    row = {
        "step": step,
        "env_id": env_index,
        "time_s": step * high_level_dt,
        "reward": float(reward_np[learner_index]),
        "done": int(done_np[learner_index]),
        "action_norm": float(np.linalg.norm(action_np[learner_index])),
        "ball_x": float(ball_xy[0]),
        "ball_y": float(ball_xy[1]),
        "ball_vx": float(ball_vel[0]),
        "ball_vy": float(ball_vel[1]),
        "ball_speed": float(np.linalg.norm(ball_vel)),
        "num_robots": num_robots,
    }

    missing_xy = np.array([np.nan, np.nan], dtype=np.float32)
    for robot_idx in range(num_robots):
        row[f"robot{robot_idx}_x"] = float(robot_xy[robot_idx, 0])
        row[f"robot{robot_idx}_y"] = float(robot_xy[robot_idx, 1])
        row[f"robot{robot_idx}_vx"] = float(robot_vel[robot_idx, 0])
        row[f"robot{robot_idx}_vy"] = float(robot_vel[robot_idx, 1])
        row[f"robot{robot_idx}_ball_dist"] = float(robot_ball_dist[robot_idx])
        row[f"robot{robot_idx}_requested_skill_id"] = int(requested[0, robot_idx])
        row[f"robot{robot_idx}_requested_skill"] = skill_name(requested[0, robot_idx])
        row[f"robot{robot_idx}_executed_skill_id"] = int(executed[0, robot_idx])
        row[f"robot{robot_idx}_executed_skill"] = skill_name(executed[0, robot_idx])
        row[f"robot{robot_idx}_invalid_skill"] = int(invalid[0, robot_idx])
        row[f"robot{robot_idx}_collision_avoidance"] = int(
            avoidance[0, robot_idx]
        )
        row[f"robot{robot_idx}_attacker"] = int(attacker[0, robot_idx])
        row[f"robot{robot_idx}_role_conflict"] = int(role_conflict[0, robot_idx])
        row[f"robot{robot_idx}_command_assist"] = int(command_assist[0, robot_idx])
        row[f"robot{robot_idx}_local_role_reward"] = float(
            local_role_reward[0, robot_idx]
        )
        row[f"robot{robot_idx}_cmd_x"] = float(commands[0, robot_idx, 0])
        row[f"robot{robot_idx}_cmd_y"] = float(commands[0, robot_idx, 1])
        row[f"robot{robot_idx}_cmd_yaw"] = float(commands[0, robot_idx, 2])
        row[f"robot{robot_idx}_cmd_xy_frame"] = (
            "body"
            if executed[0, robot_idx] == 0
            else "team_canonical_field"
        )

        obstacle = (
            missing_xy
            if obstacle_xy is None
            else obstacle_xy[env_index, robot_idx]
        )
        row[f"obstacle{robot_idx}_x"] = float(obstacle[0])
        row[f"obstacle{robot_idx}_y"] = float(obstacle[1])

    for key in TERMINAL_KEYS:
        row[key] = int(
            bool(info_array(info, key, (1,), False, env_index)[0])
        )

    return row


def outcome_rows_from_step(step, reward, done, info, num_envs):
    """Export only terminal outcomes for fast batch evaluation."""

    done_np = done.detach().cpu().numpy().astype(bool).reshape(-1)
    team_size = max(1, done_np.size // max(1, int(num_envs)))
    reward_np = reward.detach().cpu().numpy().reshape(-1)
    terminal = {
        key: np.asarray(info.get(key, 0)).reshape(-1)
        for key in TERMINAL_KEYS
    }

    def env_flag(key, env_index):
        array = terminal[key]
        return int(bool(array[0 if array.size == 1 else env_index]))

    return [
        {
            "step": step,
            "env_id": env_index,
            "done": int(done_np[env_index * team_size]),
            "high_level_goal": env_flag("high_level_goal", env_index),
            "high_level_opponent_goal": env_flag(
                "high_level_opponent_goal", env_index
            ),
            "high_level_ball_off_border": env_flag(
                "high_level_ball_off_border", env_index
            ),
            "high_level_obstacle_contact": env_flag(
                "high_level_obstacle_contact", env_index
            ),
            "high_level_accidental_termination": env_flag(
                "high_level_accidental_termination", env_index
            ),
            "high_level_opponent_accidental_termination": env_flag(
                "high_level_opponent_accidental_termination", env_index
            ),
            "high_level_learning_team_failure": env_flag(
                "high_level_learning_team_failure", env_index
            ),
            "high_level_opponent_team_failure": env_flag(
                "high_level_opponent_team_failure", env_index
            ),
            "reward": float(reward_np[env_index * team_size]),
        }
        for env_index in range(num_envs)
    ]


def write_metrics_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path, rows, args, show):
    if not rows:
        return

    # Multi-environment metric export is intended for statistical evaluation.
    # Keep the diagnostic time-series readable by plotting the first match.
    rows = [row for row in rows if int(row.get("env_id", 0)) == 0]

    from matplotlib import pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([row["time_s"] for row in rows], dtype=np.float32)
    ball_x = np.array([row["ball_x"] for row in rows], dtype=np.float32)
    ball_y = np.array([row["ball_y"] for row in rows], dtype=np.float32)
    reward = np.array([row["reward"] for row in rows], dtype=np.float32)
    num_robots = int(rows[0]["num_robots"])
    invalid = np.array([
        sum(row[f"robot{idx}_invalid_skill"] for idx in range(num_robots)) for row in rows
    ], dtype=np.float32)

    half_width = 0.5 * args.field_width
    goal_x = 0.5 * args.field_length

    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(times, ball_x, color="tab:blue", label="ball x")
    axes[0].axhline(goal_x, color="tab:green", linestyle="--", label="learning target +x")
    axes[0].axhline(-goal_x, color="tab:red", linestyle="--", label="opponent target -x")
    axes[0].set_ylabel("x (m)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, ball_y, color="tab:orange", label="ball y")
    axes[1].axhline(half_width, color="black", linestyle=":", linewidth=1)
    axes[1].axhline(-half_width, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylabel("y (m)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    for robot_idx in range(num_robots):
        distance = np.array([row[f"robot{robot_idx}_ball_dist"] for row in rows], dtype=np.float32)
        axes[2].plot(times, distance, label=f"robot {robot_idx}")
    axes[2].set_ylabel("ball dist (m)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")

    for robot_idx in range(num_robots):
        requested = np.array(
            [row[f"robot{robot_idx}_requested_skill_id"] for row in rows],
            dtype=np.float32,
        )
        executed = np.array(
            [row[f"robot{robot_idx}_executed_skill_id"] for row in rows],
            dtype=np.float32,
        )
        axes[3].step(
            times,
            requested + 0.04 * robot_idx,
            where="post",
            linestyle=":",
            alpha=0.75,
            label=f"robot {robot_idx} requested",
        )
        axes[3].step(
            times,
            executed + 0.04 * robot_idx,
            where="post",
            label=f"robot {robot_idx} executed",
        )
    axes[3].set_yticks(range(len(SKILL_NAMES)))
    axes[3].set_yticklabels(SKILL_NAMES)
    axes[3].set_ylabel("skill")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right")

    axes[4].plot(times, reward, color="tab:green", label="reward")
    axes[4].step(times, invalid, where="post", color="tab:red", alpha=0.7, label="invalid requests")
    axes[4].set_ylabel("reward / count")
    axes[4].set_xlabel("time (s)")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def print_summary(rows):
    if not rows:
        print("No rollout rows were recorded.")
        return

    total_reward = sum(row["reward"] for row in rows)
    terminations = sum(row["done"] for row in rows)
    goals = sum(row["high_level_goal"] for row in rows)
    opponent_goals = sum(row["high_level_opponent_goal"] for row in rows)
    off_border = sum(row["high_level_ball_off_border"] for row in rows)
    obstacle_hits = sum(row["high_level_obstacle_contact"] for row in rows)
    accidental = sum(row["high_level_accidental_termination"] for row in rows)
    opponent_accidental = sum(
        row.get("high_level_opponent_accidental_termination", 0) for row in rows
    )
    if "num_robots" not in rows[0]:
        print(f"Total reward: {total_reward:.3f}")
        print(
            f"Terminations: {terminations} | learning-team goals (+x): {goals} | "
            f"opponent-team goals (-x): {opponent_goals} | off border: "
            f"{off_border} | learner accidental: {accidental} | "
            f"opponent accidental: {opponent_accidental}"
        )
        return
    num_robots = int(rows[0]["num_robots"])
    invalid_requests = sum(
        row[f"robot{idx}_invalid_skill"] for row in rows for idx in range(num_robots)
    )
    avoidance_overrides = sum(
        row[f"robot{idx}_collision_avoidance"]
        for row in rows
        for idx in range(num_robots)
    )
    role_conflicts = sum(
        row[f"robot{idx}_role_conflict"]
        for row in rows
        for idx in range(num_robots)
    )
    command_assists = sum(
        row[f"robot{idx}_command_assist"]
        for row in rows
        for idx in range(num_robots)
    )
    max_ball_speed = max(row["ball_speed"] for row in rows)
    final_ball_x = rows[-1]["ball_x"]

    print(f"Total reward: {total_reward:.3f}")
    goal_count = goals + opponent_goals
    goal_rate = goals / goal_count if goal_count else float("nan")
    print(f"Goal rate (ours / total goals): {goal_rate:.2%} ({goals}/{goal_count})")
    print(f"Mean reward per high-level step: {total_reward / len(rows):.3f}")
    print(
        f"Terminations: {terminations} | learning-team goals (+x): {goals} | "
        f"opponent-team goals (-x): {opponent_goals} | off border: {off_border} | "
        f"obstacles: {obstacle_hits} | learner accidental: {accidental} | "
        f"opponent accidental: {opponent_accidental}"
    )
    print(f"Invalid low-level skill requests: {invalid_requests}")
    print(f"Predictive collision-avoidance overrides: {avoidance_overrides}")
    print(f"Role conflicts: {role_conflicts} | attacker command assists: {command_assists}")
    print(f"Final ball x: {final_ball_x:.3f} | max ball speed: {max_ball_speed:.3f}")

    for robot_idx in range(num_robots):
        requested_counts = {name: 0 for name in SKILL_NAMES}
        executed_counts = {name: 0 for name in SKILL_NAMES}
        for row in rows:
            requested_counts[row[f"robot{robot_idx}_requested_skill"]] += 1
            executed_counts[row[f"robot{robot_idx}_executed_skill"]] += 1
        print(f"Robot {robot_idx} requested skills: {requested_counts}")
        print(f"Robot {robot_idx} executed skills:  {executed_counts}")


def apply_saved_execution_settings(args, contract):
    args.shooting_options = bool(contract.get("shooting_options", False))
    for argument, key in (("collision_avoidance", "collision_avoidance"),
                          ("allow_near_ball_reposition", "near_ball_reposition")):
        if getattr(args, argument, None) is None:
            setattr(args, argument, bool(contract.get(key, False)))


def run_first_episode_outcomes(args, env, raw_env, policy):
    """Minimal rollout: retain first-episode goals on device, export once."""
    obs = env.reset()
    finished = torch.zeros(args.num_envs, dtype=torch.bool, device=raw_env.device)
    goals = torch.zeros((args.num_envs, 2), dtype=torch.bool, device=raw_env.device)
    terminal_steps = torch.full((args.num_envs,), -1, device=raw_env.device, dtype=torch.long)
    with torch.no_grad():
        for step in range(args.steps):
            action = policy(obs).to(raw_env.device)
            obs, _, done, info = env.step(action)
            active = ~finished
            for column, key in enumerate(("high_level_goal", "high_level_opponent_goal")):
                flag = torch.as_tensor(info.get(key, False), device=raw_env.device, dtype=torch.bool).reshape(-1)
                goals[:, column] |= active & flag
            ended = done.reshape(args.num_envs, -1).any(dim=1) & active
            terminal_steps[ended] = step
            finished |= ended
            if bool(finished.all()):
                break
    # Include unfinished environments so downstream code can reject incomplete runs.
    result = torch.cat((terminal_steps[:, None], finished[:, None], goals), dim=1).cpu().tolist()
    rows = [dict(step=int(row[0]), env_id=index, done=int(row[1]),
                 high_level_goal=int(row[2]), high_level_opponent_goal=int(row[3]))
            for index, row in enumerate(result)]
    write_metrics_csv(Path(args.csv), rows)
    print(f"Saved first-episode outcomes: {args.csv}")


def run(args, observer_factory=None):
    set_seed(args.seed)
    high_level_policy = load_high_level_policy(args)
    contract = high_level_checkpoint_contract(high_level_policy) or {}
    if getattr(args, "training_environment", False):
        from quadruped.world_model.config import load_config
        path = high_level_policy["policy_metadata"]["config_path"]
        payload = load_config(path)
        saved = payload.get("Cfg", {})
        saved = saved.get("value", saved)
        args.saved_training_cfg = {}
        for section in ("env", "rewards", "reward_scales", "domain_rand"):
            values = saved.get(section, {})
            args.saved_training_cfg[section] = values.get("value", values)
        args.domain_rand = True
        args.fixed_init = not args.saved_training_cfg["env"].get("randomize_match_init", True)
        for argument, key in (
            ("control_interval", "control_interval"),
            ("high_level_history", "history_length"),
            ("role_aware_fallback", "role_aware_fallback"),
            ("attacker_switch_margin", "attacker_switch_margin"),
            ("support_command_deadband", "support_command_deadband"),
            ("boundary_walls", "boundary_walls"),
        ):
            if contract.get(key) is not None:
                setattr(args, argument, contract[key])
        print(f"Restoring training environment from {path}")
    apply_saved_execution_settings(args, contract)
    # Follow the checkpoint's training semantics unless the caller explicitly
    # requests a deployment override. This keeps validate_high_level.bash
    # compatible with both old fallback-trained and new faithful-action runs.
    if args.use_geometric_skill_fallback is None:
        saved_fallback = contract.get("geometric_fallback")
        args.use_geometric_skill_fallback = (
            bool(saved_fallback) if saved_fallback is not None else False
        )
    if args.near_ball_init_probability is None:
        saved_near_ball = contract.get("near_ball_probability")
        args.near_ball_init_probability = (
            float(saved_near_ball) if saved_near_ball is not None else 0.8
        )
    if getattr(args, "near_ball_init_team", None) is None:
        saved_kickoff_team = contract.get("near_ball_init_team")
        # Checkpoints produced before balanced kickoffs used learner-only
        # resets; preserve that contract when it is explicitly recorded while
        # making new evaluations balanced by default.
        args.near_ball_init_team = (
            str(saved_kickoff_team)
            if saved_kickoff_team is not None
            else "balanced"
        )
    # Discrete coordinator checkpoints persist their action encoding in the
    # run config. Carry it into environment construction so a 12-value packed
    # categorical action is not silently reshaped as the legacy six-value
    # hybrid action.
    args.discrete_skill_direction = (
        contract.get("action_encoding") == "discrete_skill_direction"
    )
    if getattr(args, "fair_match_init", False):
        args.random_init = True
    if getattr(args, "random_init", False):
        args.fixed_init = False
        args.near_ball_init_probability = 0.0
        print("Evaluation reset: random initialization (no near-ball or curriculum starts)")
    if getattr(args, "random_learning_start", False):
        args.fixed_init = False
        args.near_ball_init_probability = 1.0
        args.near_ball_init_team = "learning"
    validate_high_level_evaluation_contract(high_level_policy, args)
    opponent_high_level_policy = None
    opponent_pool_path = getattr(args, "opponent_pool_checkpoint", None)
    if not getattr(args, "opponent_rule_based", False) and not opponent_pool_path:
        opponent_high_level_policy = load_opponent_high_level_policy(
            args, high_level_policy
        )
    if getattr(args, "training_skills", False):
        from scripts.evaluation_provenance import freeze_skills
        config_path = high_level_policy["policy_metadata"]["config_path"]
        directories, provenance = freeze_skills(
            [config_path], Path(args.csv).resolve().parent / "evaluation_skills")
        args.skill_policy_source = "local"
        args.skill_checkpoint = "latest"
        for skill, directory in directories.items():
            setattr(args, skill + "_policy_dir", str(directory))
        print(f"Evaluation uses recorded training skills: {provenance['hashes']}")
    skill_policies = load_skill_policies(args)
    env, raw_env = make_env(args, skill_policies)
    if opponent_pool_path:
        pool_path = Path(opponent_pool_path).expanduser().resolve()
        if not pool_path.is_file():
            raise FileNotFoundError(f"Opponent pool checkpoint does not exist: {pool_path}")
        load_opponent_pool(args, env, pool_path)
    elif getattr(args, "opponent_rule_based", False):
        from quadruped.envs.wrappers.rule_based_opponent import RuleBasedOpponent

        rule_opponent = RuleBasedOpponent(
            env,
            shoot_distance=getattr(args, "rule_opponent_shoot_distance", 0.75),
            dribble_distance=getattr(args, "rule_opponent_dribble_distance", 1.0),
            walk_speed=getattr(args, "rule_opponent_walk_speed", 0.9),
            dribble_speed=getattr(args, "rule_opponent_dribble_speed", 1.0),
            shoot_speed=getattr(args, "rule_opponent_shoot_speed", 1.5),
            block_distance=getattr(args, "rule_opponent_block_distance", 0.75),
            collision_distance=getattr(args, "rule_opponent_collision_distance", 0.65),
            collision_strength=getattr(args, "rule_opponent_collision_strength", 1.5),
        )
        env.set_opponent_action_provider(rule_opponent)
        print("opponent high-level:")
        print("  deterministic rule-based attacker/blocker")
    else:
        layout = opponent_high_level_policy.get("opponent_layout")
        if layout is not None:
            env.configure_opponent_layout(*layout)
        env.set_opponent_callable(opponent_high_level_policy["policy"])
    validate_high_level_obs_shape(high_level_policy, env)
    if opponent_high_level_policy is not None:
        expected = opponent_high_level_policy.get("expected_history_dim")
        provided = env._opponent_history.shape[-1] if hasattr(env, "_opponent_history") else env.num_obs_history
        if expected is not None and expected != provided:
            raise ValueError(f"Opponent expects {expected} history values, environment provides {provided}")
    validate_low_level_skill_shapes(skill_policies, env)

    if getattr(args, "first_episode_only", False):
        run_first_episode_outcomes(args, env, raw_env, high_level_policy["policy"])
        return

    output_video = Path(args.video)
    writer = None
    if not args.no_video and observer_factory is None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        high_level_dt = raw_env.dt * env.control_interval
        fps = args.fps or max(1, int(round(1.0 / (high_level_dt * max(args.frame_stride, 1)))))
        writer = imageio.get_writer(str(output_video), fps=fps)
    else:
        high_level_dt = raw_env.dt * env.control_interval

    obs = env.reset()
    observer = observer_factory(env, args) if observer_factory is not None else None
    recorder = None
    if getattr(args, "world_model_output", None):
        from quadruped.world_model.rollout_recorder import EvaluationRolloutRecorder
        recorder = EvaluationRolloutRecorder(env, args, high_level_policy, skill_policies)
    rows = []
    completed_episodes = torch.zeros(
        args.num_envs, dtype=torch.long, device=raw_env.device
    )

    try:
        if writer is not None and args.include_initial_frame:
            writer.append_data(raw_env.render(mode="rgb_array"))

        for step in trange(
            args.steps,
            desc="High-level rollout",
            disable=getattr(args, "no_progress", False),
        ):
            state = None if args.outcomes_only else collect_state(raw_env)
            with torch.no_grad():
                action = high_level_policy["policy"](obs).to(raw_env.device)
            if recorder is not None:
                recorder.before_step()
            if observer is not None:
                observer.before_step(action)
            obs, reward, done, info = env.step(action)
            if observer is not None:
                observer.after_step(done, info)
            if recorder is not None:
                recorder.after_step(done, info)
            export_count = args.num_envs if args.export_all_envs else 1
            if args.outcomes_only:
                rows.extend(
                    outcome_rows_from_step(
                        step, reward, done, info, args.num_envs
                    )[:export_count]
                )
            else:
                rows.extend(
                    row_from_step(
                        step,
                        high_level_dt,
                        state,
                        action,
                        reward,
                        done,
                        info,
                        env_index=env_index,
                    )
                    for env_index in range(export_count)
                )
            if args.stop_after_episodes is not None:
                match_done = done.reshape(args.num_envs, -1).any(dim=1)
                completed_episodes += match_done.long()
                if bool(
                    torch.all(completed_episodes >= args.stop_after_episodes).item()
                ):
                    break

            if writer is not None and step % max(args.frame_stride, 1) == 0:
                writer.append_data(raw_env.render(mode="rgb_array"))

            if args.stop_on_done and bool(done[0].item()):
                break
    finally:
        if observer is not None:
            observer.close()
        if recorder is not None:
            recorder.close()
        if writer is not None:
            writer.close()

    if recorder is not None:
        recorder.verify_complete()

    metrics_csv = Path(args.csv)
    plot_path = Path(args.plot)
    write_metrics_csv(metrics_csv, rows)
    if not args.no_plot:
        save_plot(plot_path, rows, args, args.show_plot)

    if not args.no_video and observer is None:
        print(f"Saved video: {output_video}")
    if not args.no_plot:
        print(f"Saved plot: {plot_path}")
    print(f"Saved metrics CSV: {metrics_csv}")
    print_summary(rows)


def parse_args(argv=None, defaults=None):
    parser = argparse.ArgumentParser(description="Visualize and validate a trained high-level multi-robot policy.")
    parser.add_argument("--world-model-output", help="Export complete episodes as a world-model test dataset")
    parser.add_argument("--world-model-checkpoint", help="Checkpoint defining the required collection schema")
    parser.add_argument("--training-skills", action="store_true",
                        help="Use frozen copies of the exact skill artifacts recorded by training.")
    add_high_level_policy_source_args(parser)
    parser.add_argument("--high-level-checkpoint", default="latest", help="High-level checkpoint suffix, for example latest or 10000.")
    parser.add_argument(
        "--opponent-high-level-checkpoint",
        default=None,
        help=(
            "Frozen opponent checkpoint suffix (for example 12800, resolving "
            "opponent_ac_weights_12800.pt) or a direct .pt path. If omitted, "
            "the opponent uses the learning-side high-level policy."
        ),
    )
    parser.add_argument(
        "--opponent-high-level-policy-dir",
        default=None,
        help=(
            "Directory in which to find opponent_ac_weights_<checkpoint>.pt. "
            "Defaults to the loaded high-level policy directory."
        ),
    )
    parser.add_argument(
        "--opponent-pool-checkpoint",
        default=None,
        help="Direct opponent_pool_*.pt file to load into the self-play wrapper.",
    )
    parser.add_argument(
        "--opponent-rule-based",
        action="store_true",
        help=(
            "Use a deterministic rule-based opponent instead of loading an "
            "opponent high-level checkpoint. The nearest opponent attacks; "
            "the other marks the nearest learning robot."
        ),
    )
    parser.add_argument("--rule-opponent-shoot-distance", type=float, default=0.75)
    parser.add_argument("--rule-opponent-dribble-distance", type=float, default=1.0)
    parser.add_argument("--rule-opponent-walk-speed", type=float, default=0.9)
    parser.add_argument("--rule-opponent-dribble-speed", type=float, default=1.0)
    parser.add_argument("--rule-opponent-shoot-speed", type=float, default=1.5)
    parser.add_argument("--rule-opponent-block-distance", type=float, default=0.75)
    parser.add_argument("--rule-opponent-collision-distance", type=float, default=0.65)
    parser.add_argument("--rule-opponent-collision-strength", type=float, default=1.5)
    parser.add_argument("--skill-checkpoint", default="latest", help="Checkpoint suffix for walk/dribble/shoot low-level skills.")
    parser.add_argument("--walk-wandb-run", default="des_zhong/as2_walking/3a6g1def")
    parser.add_argument("--dribble-wandb-run", default="des_zhong/as2_dribbling/cp9m21ay")
    parser.add_argument("--shoot-wandb-run", default="des_zhong/as2_shooting/bve3isir")
    add_skill_policy_source_args(parser)

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default="cuda:0",
        help="Simulator device string, for example cuda:6 or cpu.",
    )
    device_group.add_argument(
        "--cuda",
        "--cuda-device",
        dest="cuda_index",
        type=int,
        metavar="N",
        help="Use CUDA device N (equivalent to --device cuda:N).",
    )
    parser.add_argument("--policy-device", default="cpu")
    headless_group = parser.add_mutually_exclusive_group()
    headless_group.add_argument("--headless", dest="headless", action="store_true")
    headless_group.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=False)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--export-all-envs",
        action="store_true",
        help=(
            "Write one CSV row per parallel match and high-level step. "
            "Without this flag only match zero is exported."
        ),
    )
    parser.add_argument("--num-robots", type=int, default=2, help="Number of shared-policy AS2 actors per team.")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--episode-length", type=float, default=15.0)
    parser.add_argument("--control-interval", type=int, default=10)
    parser.add_argument("--high-level-history", type=int, default=4)
    parser.add_argument("--field-length", type=float, default=8.0)
    parser.add_argument("--field-width", type=float, default=5.0)
    parser.add_argument("--goal-half-width", type=float, default=1.0)
    parser.add_argument(
        "--no-boundary-walls",
        dest="boundary_walls",
        action="store_false",
        default=True,
        help="Disable physical rebound walls and restore legacy out-of-bounds termination.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--walk-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--walk-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--walk-yaw-reward-scale", type=float, default=1.0)
    parser.add_argument("--dribble-x-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-y-speed-scale", type=float, default=1.5)
    parser.add_argument("--dribble-yaw-speed-scale", type=float, default=1.0)
    parser.add_argument("--shoot-x-speed-scale", type=float, default=3.0)
    parser.add_argument("--shoot-y-speed-scale", type=float, default=3.0)
    parser.add_argument(
        "--use-geometric-skill-fallback",
        dest="use_geometric_skill_fallback",
        action="store_true",
        default=None,
        help="Replace invalid dribble/shoot requests instead of using the checkpoint default.",
    )
    parser.add_argument(
        "--no-geometric-skill-fallback",
        dest="use_geometric_skill_fallback",
        action="store_false",
        help="Execute invalid skill requests directly (out-of-distribution evaluation).",
    )
    parser.add_argument(
        "--no-role-aware-fallback",
        dest="role_aware_fallback",
        action="store_false",
        default=True,
        help="Disable attacker/support arbitration for a legacy comparison.",
    )
    parser.add_argument(
        "--attacker-switch-margin",
        type=float,
        default=0.0,
        help="Metres by which a teammate must be closer before taking the attacker role.",
    )
    parser.add_argument(
        "--support-command-deadband",
        type=float,
        default=0.08,
        help="Set support Walk commands to zero inside this target-distance deadband.",
    )
    collision_group = parser.add_mutually_exclusive_group()
    collision_group.add_argument(
        "--collision-avoidance",
        dest="collision_avoidance",
        action="store_true",
        help="Enable the short-horizon emergency escape override during playback.",
    )
    collision_group.add_argument(
        "--no-collision-avoidance",
        dest="collision_avoidance",
        action="store_false",
        help="Evaluate without the emergency escape override.",
    )
    parser.set_defaults(collision_avoidance=None)
    reposition_group = parser.add_mutually_exclusive_group()
    reposition_group.add_argument("--allow-near-ball-reposition", dest="allow_near_ball_reposition", action="store_true")
    reposition_group.add_argument("--no-near-ball-reposition", dest="allow_near_ball_reposition", action="store_false")
    parser.set_defaults(allow_near_ball_reposition=None)
    parser.add_argument("--collision-avoidance-distance", type=float, default=0.55)
    parser.add_argument("--collision-avoidance-lookahead", type=float, default=0.25)
    parser.add_argument("--collision-avoidance-speed", type=float, default=0.5)
    parser.add_argument(
        "--near-ball-init-probability",
        type=float,
        default=None,
        help="Override the checkpoint's randomized near-ball reset probability.",
    )
    parser.add_argument(
        "--near-ball-init-team",
        choices=("balanced", "learning", "opponent"),
        default=None,
        help=(
            "Team used for near-ball kickoffs. Defaults to the checkpoint's "
            "recorded mode, or balanced for new/legacy checkpoints."
        ),
    )
    parser.add_argument("--near-ball-init-min-distance", type=float, default=0.4)
    parser.add_argument("--near-ball-init-max-distance", type=float, default=0.95)
    parser.add_argument("--near-ball-init-max-angle", type=float, default=0.35)
    parser.add_argument(
        "--allow-training-config-mismatch",
        action="store_true",
        help="Allow an intentional evaluation that differs from the saved training contract.",
    )
    parser.add_argument("--training-environment", action="store_true",
                        help="Restore saved reset, curriculum, reward and randomization settings.")
    parser.add_argument("--fair-match-init", action="store_true",
                        help="Centered ball, paired team poses rotated by pi, nominal joints and physics; no attacking curriculum.")
    parser.add_argument("--domain-rand", action="store_true", help="Keep domain randomization enabled during playback.")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--random-learning-start", action="store_true",
                            help="Randomize starts and require the ball near a learning-team robot.")
    init_group.add_argument("--fixed-init", action="store_true", help="Disable randomized robot/ball/obstacle match initialization.")
    init_group.add_argument("--random-init", action="store_true",
                            help="Force randomized match initialization without near-ball or curriculum starts, including with --training-environment.")

    parser.add_argument("--video", default="outputs/high_level_eval.mp4")
    parser.add_argument("--plot", default="outputs/high_level_eval_metrics.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--first-episode-only", action="store_true",
                        help="Fast first-match-only export; requires headless, no video/plot, outcomes-only.")
    parser.add_argument("--csv", default="outputs/high_level_eval_metrics.csv")
    parser.add_argument(
        "--outcomes-only",
        action="store_true",
        help="Export only episode, goal, and reward fields for faster evaluation.",
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--include-initial-frame", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the per-rollout progress bar.",
    )
    parser.add_argument(
        "--stop-after-episodes",
        type=int,
        default=None,
        help=(
            "Stop once every parallel environment has completed this many "
            "episodes. Useful for exact-size evaluation batches."
        ),
    )
    parser.add_argument("--no-field-markers", action="store_true")
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--recording-fov", type=float, default=None)
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.random_learning_start and args.fair_match_init:
        parser.error("--random-learning-start conflicts with --fair-match-init")
    if args.first_episode_only and (
        not (args.headless and args.no_video and args.no_plot and args.outcomes_only)
        or args.world_model_output or args.stop_after_episodes != 1
    ):
        parser.error("--first-episode-only requires --headless --no-video --no-plot "
                     "--outcomes-only --stop-after-episodes 1 and no world-model export")
    if args.world_model_output and (not args.world_model_checkpoint or args.stop_after_episodes is None):
        parser.error("--world-model-output requires --world-model-checkpoint and --stop-after-episodes")
    if args.stop_after_episodes is not None and args.stop_after_episodes < 1:
        parser.error("--stop-after-episodes must be positive")
    if args.outcomes_only and not args.no_plot:
        parser.error("--outcomes-only requires --no-plot")
    if args.cuda_index is not None:
        if args.cuda_index < 0:
            parser.error("--cuda must be a non-negative device index")
        args.device = f"cuda:{args.cuda_index}"
    return args


def run_cli(args):
    run(args)
    # The legacy Isaac Gym runtime can crash during interpreter teardown.
    # As in validate_robot_abilities.py, exit only after run returned normally
    # and closed its CSV/video outputs. Evaluation exceptions still propagate.
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    run_cli(parse_args())
