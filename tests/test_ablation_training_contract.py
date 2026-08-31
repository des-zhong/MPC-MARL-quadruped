import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from scripts.train_high_level import build_arg_parser as build_base_parser
from scripts.train_high_level_mappo_fsp import parse_args as parse_mappo_fsp
from scripts.train_high_level_online_mpc import (
    configure_online_mpc_objective,
    mpc_action_agreement_reward,
)
from scripts.train_high_level_online_mpc_full import parse_args as parse_full
from scripts.train_high_level_online_mpc_no_kl import parse_args as parse_no_kl
from scripts.train_high_level_online_mpc_no_terminal import (
    parse_args as parse_no_terminal,
)

import torch

from dribblebot.mpc.config import MPCConfig
from dribblebot.world_model.action_adapter import JointActionAdapter, SkillBounds


class AblationTrainingContractTests(unittest.TestCase):
    def test_base_training_exposes_reproducible_seed(self):
        args = build_base_parser().parse_args(["--seed", "17"])
        self.assertEqual(args.seed, 17)

    def test_mappo_fsp_preserves_shared_launcher_pool_settings(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "train_high_level_mappo_fsp.py",
                "--opponent-pool-size",
                "12",
                "--opponent-latest-probability",
                "0.25",
            ],
        ):
            args = parse_mappo_fsp()

        self.assertEqual(args.opponent_pool_size, 12)
        self.assertEqual(args.opponent_latest_probability, 0.25)

    def test_mpc_objective_toggle_changes_only_terminal_ranking_mode(self):
        for mode, expected in (
            ("disabled", (False, "reward_only")),
            ("enabled", (True, "reward_plus_terminal_value")),
        ):
            config = MPCConfig(
                objective_mode="reward_plus_terminal_value",
                use_terminal_value=True,
                terminal_value_required=True,
                terminal_value_checkpoint="unused.pt",
            )
            enabled = configure_online_mpc_objective(
                config,
                SimpleNamespace(mpc_terminal_value=mode, seed=9),
            )
            self.assertEqual(enabled, expected[0])
            self.assertEqual(config.objective_mode, expected[1])
            self.assertEqual(config.use_terminal_value, expected[0])
            self.assertFalse(config.terminal_value_required)
            self.assertIsNone(config.terminal_value_checkpoint)
            self.assertEqual(config.seed, 9)

    def test_controlled_entry_points_enforce_component_matrix(self):
        with mock.patch.object(sys, "argv", ["no_kl"]):
            no_kl = parse_no_kl()
        with mock.patch.object(sys, "argv", ["no_terminal"]):
            no_terminal = parse_no_terminal()
        with mock.patch.object(sys, "argv", ["full"]):
            full = parse_full()

        self.assertEqual(no_kl.mpc_terminal_value, "enabled")
        self.assertEqual(no_kl.mpc_kl_coefficient, 0.0)
        self.assertEqual(no_terminal.mpc_terminal_value, "disabled")
        self.assertGreater(no_terminal.mpc_kl_coefficient, 0.0)
        self.assertEqual(full.mpc_terminal_value, "enabled")
        self.assertGreater(full.mpc_kl_coefficient, 0.0)
        for args in (no_kl, no_terminal, full):
            self.assertEqual(args.mpc_guidance_reward_coefficient, 1.0)

    def test_no_kl_rejects_nonzero_distillation_coefficient(self):
        with mock.patch.object(
            sys,
            "argv",
            ["no_kl", "--mpc-kl-coefficient", "0.05"],
        ):
            with self.assertRaisesRegex(ValueError, "coefficient=0"):
                parse_no_kl()

    def test_no_terminal_rejects_enabling_terminal_value(self):
        with mock.patch.object(
            sys,
            "argv",
            ["no_terminal", "--mpc-terminal-value", "enabled"],
        ):
            with self.assertRaisesRegex(ValueError, "terminal-value=disabled"):
                parse_no_terminal()

    def test_mpc_stages_preserve_one_shared_guidance_coefficient(self):
        parsers = (parse_no_kl, parse_no_terminal, parse_full)
        for parser in parsers:
            with self.subTest(parser=parser.__module__), mock.patch.object(
                sys,
                "argv",
                ["trainer", "--mpc-guidance-reward-coefficient", "0.3"],
            ):
                args = parser()
            self.assertEqual(args.mpc_guidance_reward_coefficient, 0.3)

    def test_exact_teacher_agreement_has_higher_reward_than_mismatch(self):
        bounds = {
            0: SkillBounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            1: SkillBounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
            2: SkillBounds((-1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
        }
        adapter = JointActionAdapter(bounds, num_robots=2)
        parameters = torch.zeros(1, 2, 3)
        teacher = adapter.pack(torch.tensor([[1, 1]]), parameters)
        exact = teacher.clone()
        mismatch = adapter.pack(torch.tensor([[0, 1]]), parameters)

        exact_reward, exact_error = mpc_action_agreement_reward(
            exact, teacher, adapter, 2, 1.0
        )
        mismatch_reward, mismatch_error = mpc_action_agreement_reward(
            mismatch, teacher, adapter, 2, 1.0
        )

        self.assertGreater(float(exact_reward[0, 0]), float(mismatch_reward[0, 0]))
        self.assertGreater(float(exact_reward.mean()), float(mismatch_reward.mean()))
        self.assertEqual(float(exact_error.max()), 0.0)
        self.assertEqual(float(mismatch_error[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
