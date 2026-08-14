from scripts.train_dribbling import build_arg_parser as build_dribble_parser
from scripts.train_high_level import (
    build_arg_parser as build_high_level_parser,
    find_run_level_config,
    newest_complete_local_checkpoint,
    resolve_high_level_checkpoint_dir,
)
from scripts.train_shooting import build_arg_parser as build_shoot_parser
from dribblebot_learn.ppo_cse import (
    checkpoint_next_iteration,
    checkpoint_wandb_base_path,
)


def test_dribble_and_shoot_checkpoint_defaults_are_isolated():
    dribble_args = build_dribble_parser().parse_args([])
    shoot_args = build_shoot_parser().parse_args([])
    high_level_args = build_high_level_parser().parse_args([])

    assert dribble_args.checkpoint_dir == "tmp/legged_data/dribble"
    assert shoot_args.checkpoint_dir == "tmp/legged_data/shoot"
    assert high_level_args.checkpoint_dir is None
    assert dribble_args.checkpoint_dir != shoot_args.checkpoint_dir


def test_resume_defaults_follow_isolated_checkpoint_directories():
    dribble_args = build_dribble_parser().parse_args([])
    shoot_args = build_shoot_parser().parse_args([])

    assert dribble_args.resume_checkpoint.startswith(
        dribble_args.checkpoint_dir + "/"
    )
    assert shoot_args.resume_checkpoint.startswith(
        shoot_args.checkpoint_dir + "/"
    )


def test_checkpoint_directory_can_be_overridden_per_run():
    args = build_dribble_parser().parse_args(
        ["--checkpoint-dir", "checkpoints/dribble-experiment-2"]
    )

    assert args.checkpoint_dir == "checkpoints/dribble-experiment-2"


def test_high_level_checkpoint_directory_defaults_inside_new_wandb_run(tmp_path):
    run_dir = tmp_path / "wandb" / "run-20260814_120000-abc123" / "files"

    resolved = resolve_high_level_checkpoint_dir(None, run_dir)

    assert resolved == str(
        (run_dir / "tmp" / "legged_data" / "high_level").resolve()
    )


def test_wandb_base_path_does_not_nest_run_scoped_checkpoints(tmp_path):
    working = tmp_path / "project"
    run_dir = working / "wandb" / "run-abc" / "files"
    checkpoint_dir = run_dir / "tmp" / "legged_data" / "high_level"

    assert checkpoint_wandb_base_path(
        checkpoint_dir, run_dir, working
    ) == str(run_dir.resolve())


def test_high_level_pins_only_complete_numbered_skill_checkpoint(tmp_path):
    policy = tmp_path / "run" / "files" / "tmp" / "legged_data" / "shoot"
    policy.mkdir(parents=True)
    for stem in ("body", "adaptation_module", "ac_weights"):
        suffix = ".pt" if stem == "ac_weights" else ".jit"
        (policy / f"{stem}_400{suffix}").write_bytes(b"complete")
    (policy / "ac_weights_800.pt").write_bytes(b"partial")
    (policy / "adaptation_module_800.jit").write_bytes(b"partial")

    assert newest_complete_local_checkpoint(policy) == "400"


def test_high_level_prefers_run_level_wandb_config(tmp_path):
    files = tmp_path / "run" / "files"
    policy = files / "tmp" / "legged_data" / "shoot"
    policy.mkdir(parents=True)
    config = files / "config.yaml"
    config.write_text("Cfg: {}\n", encoding="utf-8")

    assert find_run_level_config(policy) == config.resolve()


def test_high_level_defaults_protect_discrete_skill_exploration():
    args = build_high_level_parser().parse_args([])

    assert not args.resume
    assert args.resume_run is None
    assert args.resume_checkpoint == "tmp/legged_data/high_level/ac_weights_latest.pt"
    assert args.skill_entropy_coef > 0.0
    assert args.ppo_epochs == 2
    assert args.max_kl_factor > 1.0
    assert args.use_geometric_skill_fallback


def test_high_level_resume_accepts_explicit_checkpoint():
    checkpoint = "wandb/run/files/tmp/legged_data/high_level/ac_weights_latest.pt"
    args = build_high_level_parser().parse_args(
        ["--resume", "True", "--resume-checkpoint", checkpoint]
    )

    assert args.resume
    assert args.resume_checkpoint == checkpoint


def test_numbered_high_level_resume_continues_iteration_numbering():
    assert checkpoint_next_iteration("/run/ac_weights_6800.pt") == 6801
    assert checkpoint_next_iteration("/run/ac_weights_latest.pt") == 0


def test_dribble_defaults_keep_policy_distribution_inside_action_range():
    args = build_dribble_parser().parse_args([])

    assert args.action_clip == 1.0
    assert args.action_mean_bound == args.action_clip
    assert args.init_noise_std == 0.3
    assert args.max_noise_std == 0.5
    assert args.entropy_coef == 0.003


def test_shooting_defaults_stabilize_policy_updates_and_contact_setup():
    args = build_shoot_parser().parse_args([])

    assert args.schedule == "adaptive"
    assert args.action_clip == 1.0
    assert args.action_mean_bound < args.action_clip
    assert args.init_noise_std == 0.20
    assert args.max_noise_std == 0.25
    assert args.entropy_coef == 0.001
    assert args.desired_kl == 0.01
    assert args.reset_longitudinal_min <= 0.45 <= args.reset_longitudinal_max
    assert args.reset_lateral_range == 0.20
    assert args.reset_yaw_error_range == 0.25
    assert not args.randomize_ball_physics
