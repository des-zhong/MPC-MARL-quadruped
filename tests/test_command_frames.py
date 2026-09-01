import math
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "dribblebot"
    / "command_frames.py"
)
SPEC = importlib.util.spec_from_file_location("_offline_command_frames", MODULE_PATH)
COMMAND_FRAMES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMAND_FRAMES)
body_xy_to_world_xy = COMMAND_FRAMES.body_xy_to_world_xy
world_xy_to_body_xy = COMMAND_FRAMES.world_xy_to_body_xy


def yaw_quaternion(yaw):
    half = 0.5 * torch.as_tensor(yaw, dtype=torch.float)
    zeros = torch.zeros_like(half)
    return torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)


@pytest.mark.parametrize(
    "yaw, expected_world",
    [
        (0.0, [1.0, 0.0]),
        (math.pi / 2.0, [0.0, 1.0]),
        (-math.pi / 2.0, [0.0, -1.0]),
        (math.pi, [-1.0, 0.0]),
    ],
)
def test_planar_command_frame_cardinal_rotations(yaw, expected_world):
    body = torch.tensor([1.0, 0.0])
    quat = yaw_quaternion(yaw)

    world = body_xy_to_world_xy(body, quat)

    assert torch.allclose(world, torch.tensor(expected_world), atol=1e-6)
    assert torch.allclose(world_xy_to_body_xy(world, quat), body, atol=1e-6)


def test_planar_command_frame_ignores_roll_and_pitch():
    # This normalized quaternion has zero yaw but non-zero roll. A planar body
    # vector must remain planar and unchanged.
    quat = torch.tensor([math.sin(0.3), 0.0, 0.0, math.cos(0.3)])
    vector = torch.tensor([0.4, -0.7])

    assert torch.allclose(body_xy_to_world_xy(vector, quat), vector, atol=1e-6)
    assert torch.allclose(world_xy_to_body_xy(vector, quat), vector, atol=1e-6)
