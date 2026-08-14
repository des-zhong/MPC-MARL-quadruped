from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from dribblebot.mpc.simulator_controller import MPCSimulatorController


def test_controller_passes_fixed_opponent_forecast_to_planner():
    fixed = torch.zeros(2, 3, 16)
    mask = torch.tensor([False, False, True, True])

    class Forecaster:
        def fixed_action_sequence(self, horizon):
            assert horizon == 3
            return fixed, mask

    class Planner:
        action_adapter = object()
        config = SimpleNamespace(horizon=3)

        def plan(self, states, planner_state, **kwargs):
            self.call = (states, planner_state, kwargs)
            return SimpleNamespace(planner_state="updated")

    planner = Planner()
    controller = MPCSimulatorController(
        env=object(),
        planner=planner,
        state_adapter=object(),
        local_observation_adapter=object(),
        capture_terminal_state=False,
        opponent_forecaster=Forecaster(),
    )
    states = torch.zeros(2, 10)

    controller.act(states)

    assert planner.call[0] is states
    assert planner.call[1] is None
    assert planner.call[2]["fixed_action_sequence"] is fixed
    assert planner.call[2]["fixed_robot_mask"] is mask
    assert controller.planner_state == "updated"
