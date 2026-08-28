"""Structural checks for the manager-based frozen-shoot composition task."""

import ast
from pathlib import Path


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
    / "tasks"
    / "manager_based"
)


def _class_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def test_frozen_shooting_composes_actions_observations_and_state_machine() -> None:
    path = PACKAGE_ROOT / "as2_shooting" / "shooting_env_cfg.py"
    source = path.read_text(encoding="utf-8")
    classes = _class_names(path)
    assert "AS2FrozenShootingActionsCfg" in classes
    assert "AS2FrozenShootingFlatEnvCfg" in classes
    assert "AS2FrozenShootingFlatEnvCfg_PLAY" in classes
    assert "fixed_skill_id=mdp.SHOOT_SKILL_ID" in source
    assert "command_scales=REPRODUCTION_COMMAND_SCALES" in source
    assert 'action_history_semantics="successive_policy_outputs"' in source
    assert "class AS2FrozenShootingFlatEnvCfg(AS2ShootingFlatEnvCfg)" in source
    assert "observations: AS2FrozenSkillObservationsCfg" in source


def test_frozen_shooting_tasks_are_registered() -> None:
    source = (PACKAGE_ROOT / "as2_shooting" / "__init__.py").read_text(encoding="utf-8")
    assert "Isaac-DribbleBot-AS2-Shooting-Frozen-Flat-v0" in source
    assert "Isaac-DribbleBot-AS2-Shooting-Frozen-Flat-Play-v0" in source


def test_coordinator_macro_task_uses_reproduction_scales_and_ten_tick_interval() -> None:
    path = PACKAGE_ROOT / "as2_skill" / "skill_env_cfg.py"
    source = path.read_text(encoding="utf-8")
    classes = _class_names(path)
    assert "AS2CoordinatorActionsCfg" in classes
    assert "AS2CoordinatorFlatEnvCfg" in classes
    assert "AS2CoordinatorFlatEnvCfg_PLAY" in classes
    assert "REPRODUCTION_COMMAND_SCALES" in source
    assert "macro_control_interval: int = 10" in source

    registration = (PACKAGE_ROOT / "as2_skill" / "__init__.py").read_text(encoding="utf-8")
    assert "Isaac-DribbleBot-AS2-Coordinator-Macro-Flat-Play-v0" in registration
    assert "football.macro:make_manager_based_macro_env" in registration


def test_four_robot_match_and_self_play_seams_are_registered() -> None:
    path = PACKAGE_ROOT / "as2_match" / "match_env_cfg.py"
    source = path.read_text(encoding="utf-8")
    classes = _class_names(path)
    for name in (
        "AS2MatchSceneCfg",
        "AS2MatchCommandsCfg",
        "AS2MatchActionsCfg",
        "AS2MatchObservationsCfg",
        "AS2MatchFlatEnvCfg",
        "AS2MatchFlatEnvCfg_PLAY",
    ):
        assert name in classes
    for robot_name in ("robot_0", "robot_1", "robot_2", "robot_3"):
        assert robot_name in source
    assert "expected 34" in (
        PACKAGE_ROOT / "football" / "mdp" / "observations.py"
    ).read_text(encoding="utf-8")

    registration = (PACKAGE_ROOT / "as2_match" / "__init__.py").read_text(encoding="utf-8")
    assert "Isaac-DribbleBot-AS2-Match-SelfPlay-Flat-Play-v0" in registration
    assert "football.self_play:make_match_self_play_env" in registration
    match_terms = (PACKAGE_ROOT / "football" / "mdp" / "match.py").read_text(encoding="utf-8")
    assert "def match_goal" in match_terms
    assert "def match_opponent_goal" in match_terms
    assert "def match_possession" in match_terms
    self_play = (PACKAGE_ROOT / "football" / "self_play.py").read_text(encoding="utf-8")
    assert "def update_opponent_snapshot" in self_play
    assert "opponent_snapshot_interval" in self_play
    snapshot = (PACKAGE_ROOT / "football" / "snapshot.py").read_text(encoding="utf-8")
    assert "def capture_match_snapshot" in snapshot
    assert "class MatchSnapshotRecorder" in snapshot
    action_terms = (PACKAGE_ROOT / "football" / "mdp" / "actions.py").read_text(encoding="utf-8")
    assert "_apply_geometric_skill_fallback" in action_terms
    assert "invalid_skill_mask" in action_terms
