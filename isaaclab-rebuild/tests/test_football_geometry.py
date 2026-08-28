"""Simulator-free parity tests for migrated football geometry."""

import importlib.util
from pathlib import Path

import torch


GEOMETRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "dribblebot_isaaclab"
    / "dribblebot_isaaclab"
    / "tasks"
    / "manager_based"
    / "football"
    / "mdp"
    / "geometry.py"
)
SPEC = importlib.util.spec_from_file_location("dribblebot_football_geometry", GEOMETRY_PATH)
geometry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(geometry)


def test_velocity_tracking_is_one_at_target() -> None:
    actual = torch.tensor([[1.0, -0.5]])
    result = geometry.planar_velocity_tracking_exp(actual, actual.clone(), (1.5, 1.5), std=0.5)
    torch.testing.assert_close(result, torch.ones(1))


def test_dribbling_setup_prefers_ball_ahead_of_robot() -> None:
    positions = torch.tensor([[0.35, 0.0], [0.0, 0.0], [0.35, 0.4]])
    result = geometry.dribbling_setup_score(positions, target_forward=0.35, position_gain=10.0)
    assert result[0] > result[1]
    assert result[0] > result[2]


def test_direction_alignment_handles_forward_reverse_and_zero_command() -> None:
    actual = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    result = geometry.direction_alignment_score(actual, target)
    torch.testing.assert_close(result, torch.tensor([1.0, 0.0, 0.0]))


def test_out_of_bounds_uses_rectangular_field_extent() -> None:
    positions = torch.tensor([[2.49, 0.0], [2.51, 0.0], [0.0, -2.51]])
    result = geometry.out_of_bounds_mask(positions, (2.5, 2.5))
    assert result.tolist() == [False, True, True]


def test_shooting_setup_pose_is_behind_ball_in_command_direction() -> None:
    base = torch.tensor([[0.0, 0.0]])
    ball = torch.tensor([[1.0, 0.0]])
    command = torch.tensor([[2.0, 0.0]])
    direction, target, error = geometry.shooting_setup_geometry(base, ball, command, setup_distance=0.45)
    torch.testing.assert_close(direction, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(target, torch.tensor([[0.55, 0.0]]))
    torch.testing.assert_close(error, torch.tensor([0.55]))


def test_shooting_setup_progress_is_signed() -> None:
    previous = torch.tensor([[0.0, 0.0], [0.5, 0.0]])
    current = torch.tensor([[0.1, 0.0], [0.4, 0.0]])
    ball = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    command = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    progress = geometry.shooting_setup_progress(
        previous,
        current,
        ball,
        command,
        setup_distance=0.45,
        dt=0.1,
        speed_scale=1.0,
    )
    torch.testing.assert_close(progress, torch.tensor([1.0, -1.0]))


def test_forward_velocity_rewards_partial_correct_strike() -> None:
    ball_velocity = torch.tensor([[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]])
    command = torch.tensor([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    score = geometry.shooting_forward_velocity_score(ball_velocity, command, min_command_speed=0.3)
    torch.testing.assert_close(score, torch.tensor([0.25, 0.0, 0.0]))


def test_shooting_phase_launch_is_a_one_step_event() -> None:
    common = {
        "base_xy": torch.tensor([[0.0, 0.0]]),
        "ball_xy": torch.tensor([[0.5, 0.0]]),
        "ball_velocity_xy": torch.tensor([[0.5, 0.0]]),
        "command_xy": torch.tensor([[2.0, 0.0]]),
        "elapsed_s": torch.tensor([1.0]),
        "min_command_speed": 0.3,
        "launch_speed_fraction": 0.20,
        "launch_alignment": 0.50,
        "success_distance": 0.70,
        "success_speed_fraction": 0.25,
        "success_alignment": 0.65,
        "max_attempt_time_s": 5.0,
    }
    launch, launched, success, failure = geometry.compute_shooting_phase(
        previously_launched=torch.tensor([False]), **common
    )
    assert launch.tolist() == [True]
    assert launched.tolist() == [True]
    assert success.tolist() == [False]
    assert failure.tolist() == [False]

    launch_again, launched_again, _, _ = geometry.compute_shooting_phase(
        previously_launched=launched, **common
    )
    assert launch_again.tolist() == [False]
    assert launched_again.tolist() == [True]


def test_shooting_phase_detects_success_and_timeout_failure() -> None:
    base = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ball = torch.tensor([[0.8, 0.0], [0.5, 0.0], [0.5, 0.0]])
    velocity = torch.tensor([[0.6, 0.0], [0.0, 0.0], [0.0, 0.0]])
    command = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.1, 0.0]])
    launch, launched, success, failure = geometry.compute_shooting_phase(
        base,
        ball,
        velocity,
        command,
        previously_launched=torch.tensor([True, False, False]),
        elapsed_s=torch.tensor([2.0, 5.02, 5.02]),
        min_command_speed=0.3,
        launch_speed_fraction=0.20,
        launch_alignment=0.50,
        success_distance=0.70,
        success_speed_fraction=0.25,
        success_alignment=0.65,
        max_attempt_time_s=5.0,
    )
    assert launch.tolist() == [False, False, False]
    assert launched.tolist() == [True, False, False]
    assert success.tolist() == [True, False, False]
    assert failure.tolist() == [False, True, False]
