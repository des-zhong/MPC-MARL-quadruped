"""Offline regression tests for low-level policy checkpoint isolation and clips."""

import argparse
import hashlib
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import pytest

from scripts.playback_utils import (
    build_policy_metadata,
    find_local_wandb_config,
    find_policy_config_path,
    policy_action_clip_from_config,
    restore_wandb_file,
    wandb_run_cache_dir,
)
from scripts.train_high_level import add_skill_policy_source_args


def test_skill_policy_source_cli_is_explicit():
    parser = add_skill_policy_source_args(argparse.ArgumentParser())

    online = parser.parse_args([])
    assert online.skill_policy_source == "wandb"
    assert online.walk_policy_dir is None

    local = parser.parse_args(
        [
            "--skill-policy-source",
            "local",
            "--walk-policy-dir",
            "walk",
            "--dribble-policy-dir",
            "dribble",
            "--shoot-policy-dir",
            "shoot",
        ]
    )
    assert local.skill_policy_source == "local"
    assert local.walk_policy_dir == "walk"


def test_online_skill_source_forces_fresh_process_temporary_downloads(monkeypatch):
    calls = []

    def fake_resolve(run, checkpoint, **kwargs):
        calls.append((run, checkpoint, dict(kwargs)))
        root = Path(kwargs["cache_root"])
        return root / "body.jit", root / "adaptation.jit", None, run, {
            "config_path": str(root / "config.yaml"),
            "action_clip": 1.0,
            "artifacts": {},
        }

    def fake_load(label, body, adaptation, weights, policy_type, device, source, *, policy_metadata):
        return {
            "policy": object(),
            "policy_type": policy_type,
            "expected_history_dim": 1080 if policy_type == "walking" else 1125,
            "body_path": Path(body),
            "adaptation_module_path": Path(adaptation),
            "ac_weights_path": None,
            "source": source,
            "action_clip": 10.0 if label == "dribble" else 1.0,
            "policy_metadata": policy_metadata,
        }

    fake_module = types.SimpleNamespace(
        load_policy_record=fake_load,
        resolve_wandb_policy_files=fake_resolve,
    )
    monkeypatch.setitem(sys.modules, "scripts.play_walk_dribble_shoot", fake_module)
    from scripts.train_high_level import load_skill_policies

    args = argparse.Namespace(
        skill_policy_source="wandb",
        skill_checkpoint="latest",
        policy_device="cpu",
        walk_wandb_run="entity/project/walk",
        dribble_wandb_run="entity/project/dribble",
        shoot_wandb_run="entity/project/shoot",
        walk_policy_dir=None,
        dribble_policy_dir=None,
        shoot_policy_dir=None,
    )
    policies = load_skill_policies(args)

    assert len(calls) == 3
    for _, _, kwargs in calls:
        assert kwargs["refresh"] is True
        assert kwargs["local_config_fallback"] is False
        assert Path(kwargs["cache_root"]).parent == Path("/tmp")
        assert "wandb_restore_cache" not in kwargs["cache_root"]
    assert all(record["policy_metadata"]["source_kind"] == "wandb" for record in policies.values())
    for record in policies.values():
        record["_temporary_policy_download"].cleanup()


def test_policy_config_discovery_is_python38_compatible(tmp_path):
    body = tmp_path / "run" / "files" / "tmp" / "legged_data" / "body_latest.jit"
    body.parent.mkdir(parents=True)
    body.write_bytes(b"body")
    config = tmp_path / "run" / "files" / "config.yaml"
    config.write_text("Cfg: {}\n", encoding="utf-8")

    assert find_policy_config_path(body) == config.resolve()


def test_local_wandb_config_is_matched_by_run_id(tmp_path):
    run_id = "abc123"
    config = tmp_path / f"run-20260722_120000-{run_id}" / "files" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("Cfg: {}\n", encoding="utf-8")

    assert find_local_wandb_config(f"entity/project/{run_id}", tmp_path) == config.resolve()
    assert find_local_wandb_config("entity/project/missing", tmp_path) is None


def test_wandb_restore_cache_is_scoped_by_run_and_reused(monkeypatch, tmp_path):
    calls = []
    fail_refresh = {"enabled": False}
    remote_contents = {
        ("entity/project/run-one", "tmp/legged_data/body_latest.jit"): b"walk-policy",
        ("entity/project/run-two", "tmp/legged_data/body_latest.jit"): b"shoot-policy",
    }

    def fake_restore(candidate, *, run_path, root, replace=False):
        calls.append((run_path, candidate, root, replace))
        destination = Path(root) / candidate
        destination.parent.mkdir(parents=True, exist_ok=True)
        if fail_refresh["enabled"]:
            destination.write_bytes(b"partial-download")
            raise OSError("simulated network interruption")
        destination.write_bytes(remote_contents[(run_path, candidate)])
        return types.SimpleNamespace(name=str(destination))

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(restore=fake_restore))
    candidate = "tmp/legged_data/body_latest.jit"

    first = restore_wandb_file("entity/project/run-one", [candidate], cache_root=tmp_path)
    second = restore_wandb_file("entity/project/run-two", [candidate], cache_root=tmp_path)

    assert first != second
    assert first.read_bytes() == b"walk-policy"
    assert second.read_bytes() == b"shoot-policy"
    assert first == (wandb_run_cache_dir("entity/project/run-one", tmp_path) / candidate).resolve()
    assert second == (wandb_run_cache_dir("entity/project/run-two", tmp_path) / candidate).resolve()

    # A cache hit for run one must neither contact W&B again nor see run two's
    # identically named artifact.
    assert restore_wandb_file("entity/project/run-one", [candidate], cache_root=tmp_path) == first
    assert len(calls) == 2

    remote_contents[("entity/project/run-one", candidate)] = b"walk-policy-v2"
    refreshed = restore_wandb_file(
        "entity/project/run-one",
        [candidate],
        cache_root=tmp_path,
        refresh=True,
    )
    assert refreshed == first
    assert refreshed.read_bytes() == b"walk-policy-v2"
    assert calls[-1][-1] is True

    # Refreshes use a staging directory: a partial failed download must not
    # damage the last complete cached policy.
    fail_refresh["enabled"] = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fallback = restore_wandb_file(
            "entity/project/run-one",
            [candidate],
            cache_root=tmp_path,
            refresh=True,
        )
    assert fallback.read_bytes() == b"walk-policy-v2"
    assert any("using cached file" in str(item.message) for item in caught)
    run_cache = wandb_run_cache_dir("entity/project/run-one", tmp_path)
    assert not any(path.name.startswith(".refresh-") for path in run_cache.iterdir())


def test_run_cache_key_handles_urls_and_sanitization_without_collisions(tmp_path):
    url = "https://wandb.ai/entity/project/runs/run-one"
    assert wandb_run_cache_dir(url, tmp_path) == wandb_run_cache_dir(
        "entity/project/run-one",
        tmp_path,
    )
    assert wandb_run_cache_dir("entity/project/run-one", tmp_path) != wandb_run_cache_dir(
        "entity_project/run-one",
        tmp_path,
    )


def test_policy_metadata_reads_wandb_clip_and_checksums_every_input(tmp_path):
    body = tmp_path / "body_latest.jit"
    adaptation = tmp_path / "adaptation_module_latest.jit"
    weights = tmp_path / "ac_weights_latest.pt"
    config = tmp_path / "config.yaml"
    body.write_bytes(b"body")
    adaptation.write_bytes(b"adaptation")
    weights.write_bytes(b"weights")
    config.write_text(
        """\
Cfg:
  desc: null
  value:
    normalization:
      clip_observations: 100.0
      clip_actions: 10.0
""",
        encoding="utf-8",
    )

    assert policy_action_clip_from_config(config) == 10.0
    metadata = build_policy_metadata(
        body,
        adaptation,
        weights,
        run_path="https://wandb.ai/entity/project/runs/run-id",
        checkpoint="latest",
        config_path=config,
    )

    assert metadata["run_path"] == "entity/project/run-id"
    assert metadata["checkpoint"] == "latest"
    assert metadata["action_clip"] == 10.0
    assert metadata["action_clip_source"].startswith("config:")
    expected = {
        "body": body,
        "adaptation_module": adaptation,
        "ac_weights": weights,
        "config": config,
    }
    for name, path in expected.items():
        artifact = metadata["artifacts"][name]
        assert artifact["path"] == str(path.resolve())
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def _load_wrapper_module(monkeypatch):
    """Load the wrapper without importing the simulator for an offline test."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("gym")
    fake_torch_utils = types.ModuleType("isaacgym.torch_utils")
    fake_torch_utils.quat_apply = lambda quat, vector: vector
    fake_torch_utils.quat_rotate_inverse = lambda quat, vector: vector
    fake_isaacgym = types.ModuleType("isaacgym")
    fake_isaacgym.torch_utils = fake_torch_utils
    monkeypatch.setitem(sys.modules, "isaacgym", fake_isaacgym)
    monkeypatch.setitem(sys.modules, "isaacgym.torch_utils", fake_torch_utils)

    path = (
        Path(__file__).resolve().parents[1]
        / "dribblebot"
        / "envs"
        / "wrappers"
        / "high_level_skill_wrapper.py"
    )
    spec = importlib.util.spec_from_file_location("_offline_high_level_skill_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, torch


def test_wrapper_clips_each_selected_policy_to_its_own_training_range(monkeypatch):
    module, torch = _load_wrapper_module(monkeypatch)
    wrapper = module.HighLevelSkillWrapper.__new__(module.HighLevelSkillWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 3
    wrapper.num_robots = 2
    wrapper.skill_ids = torch.tensor([[0, 1], [2, 0], [1, 2]])
    wrapper.skill_commands = torch.zeros(3, 2, 3)
    wrapper.policy_action_clips = {"walk": 1.0, "dribble": 10.0, "shoot": 1.0}

    def policy(value):
        return lambda obs: torch.full((3, 12), value)

    wrapper.skill_policies = {
        "walk": {"policy": policy(5.0)},
        "dribble": {"policy": policy(5.0)},
        "shoot": {"policy": policy(-5.0)},
    }
    wrapper._full_command = lambda skill_id, command: torch.zeros(3, 15)
    wrapper._robot_low_level_observation_full = lambda robot_slot, command: torch.zeros(3, 75)
    wrapper._update_low_level_history = lambda robot_slot, obs: None
    wrapper._policy_obs = lambda robot_slot, record: {"obs_history": torch.zeros(3, 1)}
    wrapper.env = types.SimpleNamespace(
        commands=torch.zeros(3, 15),
        cfg=types.SimpleNamespace(commands=types.SimpleNamespace(num_commands=15)),
    )

    actions = wrapper._low_level_actions_from_skills()

    assert torch.all(actions[wrapper.skill_ids == 0] == 1.0)
    assert torch.all(actions[wrapper.skill_ids == 1] == 5.0)
    assert torch.all(actions[wrapper.skill_ids == 2] == -1.0)


def test_wrapper_clears_high_level_state_only_for_reset_rows(monkeypatch):
    module, torch = _load_wrapper_module(monkeypatch)
    wrapper = module.HighLevelSkillWrapper.__new__(module.HighLevelSkillWrapper)
    wrapper.high_level_obs = torch.ones(3, 4)
    wrapper.high_level_obs_history = torch.ones(3, 8)
    wrapper.skill_ids = torch.ones(3, 2, dtype=torch.long)
    wrapper.requested_skill_ids = torch.ones(3, 2, dtype=torch.long)
    wrapper.invalid_skill_mask = torch.ones(3, 2, dtype=torch.bool)
    wrapper.skill_commands = torch.ones(3, 2, 3)
    wrapper.env = types.SimpleNamespace(
        high_level_skill_ids=torch.ones(3, 2, dtype=torch.long),
        high_level_requested_skill_ids=torch.ones(3, 2, dtype=torch.long),
        high_level_invalid_skill_mask=torch.ones(3, 2, dtype=torch.bool),
        high_level_commands=torch.ones(3, 2, 3),
    )

    wrapper._clear_high_level_state(torch.tensor([False, True, False]))

    for value in (
        wrapper.high_level_obs,
        wrapper.high_level_obs_history,
        wrapper.skill_ids,
        wrapper.requested_skill_ids,
        wrapper.invalid_skill_mask,
        wrapper.skill_commands,
        wrapper.env.high_level_skill_ids,
        wrapper.env.high_level_requested_skill_ids,
        wrapper.env.high_level_invalid_skill_mask,
        wrapper.env.high_level_commands,
    ):
        assert torch.all(value[1] == 0)
        assert torch.all(value[[0, 2]] == 1)


def test_wrapper_derives_raw_clip_without_reusing_it_for_high_level_input(monkeypatch):
    module, torch = _load_wrapper_module(monkeypatch)
    fake_env = types.SimpleNamespace(
        action_space=None,
        observation_space=None,
        reward_range=(-float("inf"), float("inf")),
        metadata={},
        device=torch.device("cpu"),
        num_envs=2,
        num_train_envs=2,
        max_episode_length=100,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        cfg=types.SimpleNamespace(
            env=types.SimpleNamespace(
                high_level_num_actions=12,
                high_level_num_observations=56,
                high_level_history_length=4,
                high_level_control_interval=10,
                high_level_action_input_clip=7.0,
                num_observation_history=15,
            ),
            normalization=types.SimpleNamespace(
                clip_actions=1.0,
                clip_observations=100.0,
            ),
        ),
    )
    records = {
        "walk": {"action_clip": 1.0},
        "dribble": {"action_clip": 10.0},
        "shoot": {"action_clip": 1.0},
    }

    wrapper = module.HighLevelSkillWrapper(fake_env, records)

    assert wrapper.policy_action_clips == {"walk": 1.0, "dribble": 10.0, "shoot": 1.0}
    assert wrapper.raw_low_level_action_clip == 10.0
    assert fake_env.cfg.normalization.clip_actions == 10.0
    assert wrapper._high_level_action_clip() == 7.0


def test_shoot_affordance_uses_strikeability_not_goal_alignment(monkeypatch):
    module, torch = _load_wrapper_module(monkeypatch)
    wrapper = module.HighLevelSkillWrapper.__new__(module.HighLevelSkillWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 3
    wrapper.num_robots = 2
    wrapper.env = types.SimpleNamespace(
        # All robots are at the origin and face +x in the offline quaternion
        # stub.  The three balls are respectively reachable in front, behind,
        # and too far to the side.
        object_pos_world_frame=torch.tensor(
            [[0.5, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.3, 0.6, 0.0]],
            dtype=torch.float,
        ),
        env_origins=torch.zeros(3, 3),
        cfg=types.SimpleNamespace(
            env=types.SimpleNamespace(
                field_length=8.0,
                field_width=5.0,
                # Put the nominal goal behind the ball so the first case has
                # negative goal alignment yet must still be a valid kick.
                team_goal_x=-4.0,
            ),
            rewards=types.SimpleNamespace(
                high_level_dribble_skill_distance=1.0,
                high_level_shoot_skill_distance=0.75,
                high_level_shoot_min_forward=-0.1,
                high_level_shoot_lateral_reach=0.45,
                high_level_shoot_alignment=0.35,
            ),
        ),
    )
    roots = torch.zeros(3, 2, 13)

    affordances = wrapper._skill_affordances(roots)

    assert torch.all(affordances["behind_alignment"][0] < 0.0)
    assert torch.all(affordances["can_shoot"][0])
    assert not torch.any(affordances["can_shoot"][1])
    assert not torch.any(affordances["can_shoot"][2])


def test_decode_executes_requested_skills_when_geometric_fallback_is_disabled(monkeypatch):
    module, torch = _load_wrapper_module(monkeypatch)
    wrapper = module.HighLevelSkillWrapper.__new__(module.HighLevelSkillWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 1
    wrapper.num_robots = 2
    wrapper.requested_skill_ids = torch.zeros(1, 2, dtype=torch.long)
    wrapper.skill_ids = torch.zeros(1, 2, dtype=torch.long)
    wrapper.invalid_skill_mask = torch.ones(1, 2, dtype=torch.bool)
    wrapper.skill_commands = torch.zeros(1, 2, 3)
    wrapper.env = types.SimpleNamespace(
        cfg=types.SimpleNamespace(
            env=types.SimpleNamespace(
                high_level_use_geometric_skill_fallback=False,
                high_level_action_input_clip=10.0,
                high_level_walk_command_scale=[1.0, 1.0, 1.0],
                high_level_dribble_command_scale=[1.0, 1.0, 1.0],
                high_level_shoot_command_scale=[1.0, 1.0, 0.0],
                high_level_command_obs_scale=[1.0, 1.0, 1.0],
            )
        ),
        high_level_requested_skill_ids=torch.zeros(1, 2, dtype=torch.long),
        high_level_skill_ids=torch.zeros(1, 2, dtype=torch.long),
        high_level_invalid_skill_mask=torch.ones(1, 2, dtype=torch.bool),
        high_level_commands=torch.zeros(1, 2, 3),
    )
    wrapper._skill_affordances = lambda: {
        "can_dribble": torch.zeros(1, 2, dtype=torch.bool),
        "can_shoot": torch.zeros(1, 2, dtype=torch.bool),
    }

    # Robot 0 requests dribble and robot 1 requests shoot even though the
    # geometric helper marks both unavailable.
    action = torch.tensor([[[0.0, 2.0, 0.0, 0.2, 0.0, 0.0],
                            [0.0, 0.0, 2.0, 0.3, 0.0, 0.0]]])
    wrapper._decode_action(action)

    assert torch.equal(wrapper.skill_ids, torch.tensor([[1, 2]]))
    assert torch.equal(wrapper.requested_skill_ids, wrapper.skill_ids)
    assert not torch.any(wrapper.invalid_skill_mask)
