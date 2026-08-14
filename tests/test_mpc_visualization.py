import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from matplotlib import pyplot as plt

from dribblebot.mpc.visualization import (
    plot_mpc_execution_diagnostics,
    plot_prediction_vs_reality,
    plot_skill_and_parameters,
    plot_top_down,
)
from dribblebot.world_model.schema import default_state_schema


def test_top_down_visualization_runs_headlessly(tmp_path):
    schema = default_state_schema(1)
    state = torch.zeros(schema.state_dim)
    state[schema.slice("field.geometry")] = torch.tensor(
        [4.0, 2.5, -4.0, 4.0, 1.0, 0.09]
    )
    for _, cos_index in schema.yaw_pairs:
        state[cos_index] = 1.0
    predicted = state[None].expand(4, -1).clone()
    actions = torch.zeros(3, 8)
    output = tmp_path / "diagnostic.png"
    figure = plot_top_down(
        schema,
        state,
        predicted,
        actions,
        torch.zeros(3),
        output=output,
    )
    plt.close(figure)
    assert output.exists()
    assert output.stat().st_size > 0


def test_single_robot_prediction_vs_reality_runs_headlessly(tmp_path):
    schema = default_state_schema(max_obstacles=1, num_robots=1)
    states = torch.zeros(4, schema.state_dim)
    states[:, schema.slice("field.geometry")] = torch.tensor(
        [4.0, 2.5, -4.0, 4.0, 1.0, 0.09]
    )
    output = tmp_path / "prediction_vs_reality.png"

    plot_prediction_vs_reality(
        schema,
        states,
        states.clone(),
        torch.zeros(3),
        torch.zeros(3),
        torch.zeros(3),
        output=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_skill_and_parameter_timeline_runs_headlessly(tmp_path):
    actions = torch.tensor(
        [
            [0, 0.5, 0.0, 0.2, 1, 1.0, -0.5, 0.1],
            [1, 1.0, 0.2, 0.0, 2, 2.0, 0.0, 0.0],
            [2, 2.0, 0.0, 0.0, 0, -0.5, 0.1, -0.2],
        ],
        dtype=torch.float,
    )
    output = tmp_path / "skill_and_parameters.png"

    returned = plot_skill_and_parameters(actions, num_robots=2, output=output)

    assert returned == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_mpc_execution_diagnostics_runs_headlessly(tmp_path):
    rows = [
        {
            "fallback_used": False,
            "requested_action_modified": False,
            "best_objective": 1.0,
            "planning_time_seconds": 0.02,
        },
        {
            "fallback_used": True,
            "requested_action_modified": True,
            "best_objective": 0.2,
            "planning_time_seconds": 0.03,
        },
    ]
    errors = [
        {
            "horizon": 1,
            "robot_position_rmse_m": 0.1,
            "ball_position_rmse_m": 0.05,
        },
        {
            "horizon": 2,
            "robot_position_rmse_m": 0.2,
            "ball_position_rmse_m": 0.12,
        },
    ]
    output = tmp_path / "mpc_diagnostics.png"

    returned = plot_mpc_execution_diagnostics(rows, errors, output)

    assert returned == output
    assert output.exists()
    assert output.stat().st_size > 0
