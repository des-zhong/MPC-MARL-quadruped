import unittest
from contextlib import redirect_stdout
from io import StringIO

import torch

from dribblebot_learn.ppo_cse.actor_critic import AC_Args, ActionMeanBound, ActorCritic


class ActorCriticActionBoundTest(unittest.TestCase):
    def test_action_mean_bound_limits_outputs(self):
        raw_mean = torch.tensor([[-100.0, -1.0, 0.0, 1.0, 100.0]])

        bounded = ActionMeanBound(1.0)(raw_mean)

        self.assertTrue(torch.all(bounded <= 1.0))
        self.assertTrue(torch.all(bounded >= -1.0))
        self.assertAlmostEqual(float(bounded[0, 2]), 0.0)
        self.assertGreater(float(bounded[0, 4]), 0.99)

    def test_bound_is_preserved_by_jit_export(self):
        module = torch.jit.script(ActionMeanBound(0.75))
        output = module(torch.tensor([[-100.0, 100.0]]))

        self.assertTrue(torch.all(output <= 0.75))
        self.assertTrue(torch.all(output >= -0.75))

    def test_bound_adds_no_checkpoint_parameters(self):
        original_bound = AC_Args.action_mean_bound
        try:
            with redirect_stdout(StringIO()):
                AC_Args.action_mean_bound = None
                unbounded = ActorCritic(4, 2, 4, 3)
                AC_Args.action_mean_bound = 1.0
                bounded = ActorCritic(4, 2, 4, 3)

            self.assertEqual(
                set(unbounded.state_dict().keys()),
                set(bounded.state_dict().keys()),
            )
            self.assertIsInstance(bounded.actor_body[-1], ActionMeanBound)
            scripted_actor = torch.jit.script(bounded.actor_body)
            scripted_output = scripted_actor(torch.zeros(2, 6))
            self.assertTrue(torch.all(scripted_output.abs() <= 1.0))
        finally:
            AC_Args.action_mean_bound = original_bound

    def test_configured_action_std_ceiling_is_enforced(self):
        original_max_std = AC_Args.max_action_std
        try:
            AC_Args.max_action_std = 0.5
            with redirect_stdout(StringIO()):
                actor_critic = ActorCritic(4, 2, 4, 3)
            with torch.no_grad():
                actor_critic.std.fill_(2.0)

            actor_critic.update_distribution(torch.zeros(2, 4))

            self.assertTrue(torch.all(actor_critic.action_std <= 0.5))
        finally:
            AC_Args.max_action_std = original_max_std


if __name__ == "__main__":
    unittest.main()
