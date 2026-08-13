from types import SimpleNamespace

import pytest

from scripts.train_high_level import validate_high_level_evaluation_contract


def _args(**overrides):
    values = dict(
        num_robots=2,
        control_interval=10,
        high_level_history=4,
        walk_x_speed_scale=1.5,
        walk_y_speed_scale=1.5,
        walk_yaw_speed_scale=1.0,
        dribble_x_speed_scale=1.5,
        dribble_y_speed_scale=1.5,
        dribble_yaw_speed_scale=1.0,
        shoot_x_speed_scale=3.0,
        shoot_y_speed_scale=3.0,
        use_geometric_skill_fallback=True,
        near_ball_init_probability=0.6,
        allow_training_config_mismatch=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _record(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
Cfg:
  value:
    env:
      high_level_control_interval: 10
      high_level_history_length: 4
      high_level_walk_command_scale: [1.5, 1.5, 1.0]
      high_level_dribble_command_scale: [1.5, 1.5, 1.0]
      high_level_shoot_command_scale: [3.0, 3.0, 0.0]
      high_level_use_geometric_skill_fallback: true
      high_level_near_ball_init_probability: 0.6
      num_team_robots: 2
self_play:
  value:
    team_size: 2
""".strip()
    )
    return {"policy_metadata": {"config_path": str(config)}}


def test_matching_high_level_evaluation_contract_is_accepted(tmp_path):
    validate_high_level_evaluation_contract(_record(tmp_path), _args())


def test_high_level_evaluation_contract_rejects_semantic_mismatch(tmp_path):
    with pytest.raises(ValueError, match="walk_scale"):
        validate_high_level_evaluation_contract(
            _record(tmp_path), _args(walk_yaw_speed_scale=0.0)
        )
