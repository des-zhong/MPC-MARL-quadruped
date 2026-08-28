"""Validate the curated legacy policy bundle without launching a simulator."""

import importlib.util
import sys
from pathlib import Path

import torch


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


paths = _load_module("dribblebot_policy_paths_test", PACKAGE_ROOT / "policies" / "paths.py")
policy_module = _load_module(
    "dribblebot_frozen_policy_bundle_test",
    PACKAGE_ROOT / "policies" / "frozen_low_level.py",
)


def test_checkpoint_root_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "policy-bundle"
    monkeypatch.setenv(paths.CHECKPOINT_ROOT_ENV, str(override))
    assert paths.checkpoint_root() == override.resolve()


def test_reproduction_bundle_matches_low_level_policy_contract() -> None:
    root = paths.default_checkpoint_root()
    expected_history_dims = {"walk": 1080, "dribble": 1125, "shoot": 1125}
    policies = {}
    for skill_id, skill in enumerate(policy_module.SKILL_NAMES):
        spec = policy_module.FrozenPolicySpec(
            name=skill,
            body_path=root / skill / "body_latest.jit",
            adaptation_path=root / skill / "adaptation_module_latest.jit",
            action_clip=1.0,
        )
        policies[skill_id] = policy_module.FrozenLowLevelPolicy.load(spec)
        assert policies[skill_id].history_dim == expected_history_dims[skill]
        assert policies[skill_id].action_dim == 12

    policy_set = policy_module.FrozenSkillPolicySet(policies)
    action = policy_set.route(torch.zeros(3, 1125), torch.tensor([0, 1, 2]))
    assert action.shape == (3, 12)
    assert torch.isfinite(action).all()
    assert torch.max(torch.abs(action)) <= 1.0


def test_reproduction_high_level_bundle_matches_34d_four_frame_contract() -> None:
    module_path = PACKAGE_ROOT / "policies" / "high_level.py"
    module = _load_module("dribblebot_high_level_policy_test", module_path)
    root = paths.default_checkpoint_root() / "high_level"
    policy = module.FrozenCoordinatorPolicy.load(root, device="cpu")
    action = policy(torch.zeros(2, 136))
    assert action.shape == (2, 6)
    assert torch.isfinite(action).all()
