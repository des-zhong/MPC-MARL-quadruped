"""CPU-only regression tests for the high-level field perimeter."""

import unittest
from types import SimpleNamespace

import isaacgym

assert isaacgym
import torch
from isaacgym import gymapi

from dribblebot.envs.base.legged_robot_two import (
    boundary_wall_layout,
    high_level_match_reset_flags,
    high_level_outside_field,
)
from dribblebot.world_model.state_adapter import FootballWorldModelStateAdapter
from scripts.train_high_level import build_arg_parser


class BoundaryWallTests(unittest.TestCase):
    def test_cpu_physx_sphere_rebounds_from_wall(self):
        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 120.0
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.use_gpu_pipeline = False
        sim_params.physx.use_gpu = False
        sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
        if sim is None:
            self.skipTest("CPU PhysX simulator is unavailable")

        try:
            layout = boundary_wall_layout(8.0, 5.0, 0.12, 0.50, 0.05)
            fixed_options = gymapi.AssetOptions()
            fixed_options.fix_base_link = True
            fixed_options.disable_gravity = True
            long_asset = gym.create_box(
                sim, *layout["long_dimensions"], fixed_options
            )
            short_asset = gym.create_box(
                sim, *layout["short_dimensions"], fixed_options
            )
            for asset in (long_asset, short_asset):
                props = gym.get_asset_rigid_shape_properties(asset)
                props[0].friction = 0.35
                props[0].restitution = 0.85
                gym.set_asset_rigid_shape_properties(asset, props)

            env = gym.create_env(
                sim,
                gymapi.Vec3(-5.0, -4.0, 0.0),
                gymapi.Vec3(5.0, 4.0, 1.0),
                1,
            )
            assets = {"long": long_asset, "short": short_asset}
            for index, (kind, position) in enumerate(layout["walls"]):
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(*position)
                gym.create_actor(
                    env, assets[kind], pose, f"wall_{index}", 0, 0, 0
                )

            ball_options = gymapi.AssetOptions()
            ball_options.disable_gravity = True
            ball_asset = gym.create_sphere(sim, 0.0889, ball_options)
            ball_props = gym.get_asset_rigid_shape_properties(ball_asset)
            ball_props[0].restitution = 0.85
            gym.set_asset_rigid_shape_properties(ball_asset, ball_props)
            ball_pose = gymapi.Transform()
            ball_pose.p = gymapi.Vec3(0.0, 2.0, 0.10)
            ball = gym.create_actor(env, ball_asset, ball_pose, "ball", 0, 0, 0)
            ball_state = gym.get_actor_rigid_body_states(
                env, ball, gymapi.STATE_ALL
            )
            ball_state["vel"]["linear"]["y"] = 3.0
            gym.set_actor_rigid_body_states(
                env, ball, ball_state, gymapi.STATE_ALL
            )

            end_ball_pose = gymapi.Transform()
            end_ball_pose.p = gymapi.Vec3(3.30, 1.50, 0.10)
            end_ball = gym.create_actor(
                env, ball_asset, end_ball_pose, "end_ball", 0, 0, 0
            )
            end_ball_state = gym.get_actor_rigid_body_states(
                env, end_ball, gymapi.STATE_ALL
            )
            end_ball_state["vel"]["linear"]["x"] = 3.0
            gym.set_actor_rigid_body_states(
                env, end_ball, end_ball_state, gymapi.STATE_ALL
            )
            gym.prepare_sim(sim)

            max_y = float("-inf")
            min_y_velocity = float("inf")
            max_end_x = float("-inf")
            min_end_x_velocity = float("inf")
            for _ in range(90):
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                state = gym.get_actor_rigid_body_states(
                    env, ball, gymapi.STATE_ALL
                )
                max_y = max(max_y, float(state["pose"]["p"]["y"][0]))
                min_y_velocity = min(
                    min_y_velocity,
                    float(state["vel"]["linear"]["y"][0]),
                )
                end_state = gym.get_actor_rigid_body_states(
                    env, end_ball, gymapi.STATE_ALL
                )
                max_end_x = max(
                    max_end_x, float(end_state["pose"]["p"]["x"][0])
                )
                min_end_x_velocity = min(
                    min_end_x_velocity,
                    float(end_state["vel"]["linear"]["x"][0]),
                )

            self.assertGreater(max_y, 2.4)
            self.assertLess(min_y_velocity, -0.2)
            # Outside the goal mouth, the end wall reverses the ball before
            # its centre reaches the +/-4 m goal/out-of-bounds plane.
            self.assertLess(max_end_x, 4.0)
            self.assertLess(min_end_x_velocity, -0.2)
        finally:
            gym.destroy_sim(sim)

    def test_layout_is_closed_and_outside_painted_boundary(self):
        layout = boundary_wall_layout(8.0, 5.0, 0.12, 0.50, 0.05)

        self.assertEqual(len(layout["walls"]), 6)
        self.assertEqual(layout["long_dimensions"], (8.34, 0.12, 0.50))
        self.assertAlmostEqual(layout["short_dimensions"][0], 0.12)
        self.assertAlmostEqual(layout["short_dimensions"][1], 1.55)
        self.assertAlmostEqual(layout["short_dimensions"][2], 0.50)
        self.assertAlmostEqual(layout["goal_opening_half_width"], 1.0)

        wall_positions = {kind: [] for kind in ("long", "short")}
        for kind, position in layout["walls"]:
            wall_positions[kind].append(position)
            self.assertAlmostEqual(position[2], 0.25)

        # Sideline and end-wall collision faces are anchored to the painted
        # boundary. End walls are split around the goal mouth.
        for _, y, _ in wall_positions["long"]:
            self.assertAlmostEqual(abs(y) - 0.06, 2.55)
        for x, y, _ in wall_positions["short"]:
            self.assertAlmostEqual(abs(x) - 0.06, 4.05)
            self.assertAlmostEqual(abs(y), 1.775)

    def test_wall_mode_removes_border_reset_flag(self):
        base = torch.tensor([False, False])
        timeout = torch.tensor([False, False])
        goal = torch.tensor([False, True])
        opponent_goal = torch.tensor([False, False])
        border = torch.tensor([True, False])
        obstacle = torch.tensor([False, False])

        reset, accidental = high_level_match_reset_flags(
            base,
            timeout,
            goal,
            opponent_goal,
            border,
            obstacle,
            True,
        )

        self.assertEqual(reset.tolist(), [False, True])
        self.assertEqual(accidental.tolist(), [False, False])

        legacy_reset, _ = high_level_match_reset_flags(
            base,
            timeout,
            goal,
            opponent_goal,
            border,
            obstacle,
            False,
        )
        self.assertEqual(legacy_reset.tolist(), [True, True])

    def test_wall_mode_suppresses_legacy_border_event(self):
        positions = torch.tensor(
            [[0.0, 2.6], [4.2, 0.0], [0.0, 0.0]], dtype=torch.float
        )

        legacy = high_level_outside_field(positions, 4.0, 2.5, 0.0, False)
        walled = high_level_outside_field(positions, 4.0, 2.5, 0.0, True)

        self.assertEqual(legacy.tolist(), [True, True, False])
        self.assertEqual(walled.tolist(), [False, False, False])

    def test_training_cli_enables_walls_by_default(self):
        parser = build_arg_parser()

        self.assertTrue(parser.parse_args([]).boundary_walls)
        self.assertFalse(
            parser.parse_args(["--no-boundary-walls"]).boundary_walls
        )

    def test_world_model_does_not_relabel_wall_rebound_as_out_of_bounds(self):
        env_cfg = SimpleNamespace(
            num_robots=1,
            field_length=8.0,
            field_width=5.0,
            team_goal_x=4.0,
            team_goal_half_width=1.0,
            add_boundary_walls=True,
            high_level_walk_command_scale=(1.2, 0.6, 0.0),
            high_level_dribble_command_scale=(1.5, 1.5, 1.0),
            high_level_shoot_command_scale=(3.0, 3.0, 0.0),
        )
        cfg = SimpleNamespace(
            env=env_cfg,
            ball=SimpleNamespace(radius=0.09),
            rewards=SimpleNamespace(
                terminal_body_height=0.2,
                high_level_dribble_skill_distance=1.0,
            ),
        )
        roots = torch.zeros(1, 13)
        roots[:, 2] = 0.34
        roots[:, 6] = 1.0
        raw = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=1,
            num_robots=1,
            root_states=roots,
            robot_actor_idxs_all=torch.tensor([[0]]),
            env_origins=torch.zeros(1, 3),
            object_pos_world_frame=torch.tensor([[0.0, 2.60, 0.09]]),
            object_lin_vel=torch.zeros(1, 3),
            object_ang_vel=torch.zeros(1, 3),
            cfg=cfg,
            gait_indices=torch.zeros(1),
            num_static_opponents=0,
            high_level_skill_ids=torch.zeros(1, 1, dtype=torch.long),
            high_level_commands=torch.zeros(1, 1, 3),
        )
        wrapper = SimpleNamespace(
            env=raw,
            skill_ids=raw.high_level_skill_ids,
            skill_commands=raw.high_level_commands,
        )
        adapter = FootballWorldModelStateAdapter(wrapper, max_obstacles=0)

        walled_state = adapter.extract_state()["ball.out_of_bounds"]
        env_cfg.add_boundary_walls = False
        legacy_state = adapter.extract_state()["ball.out_of_bounds"]

        self.assertEqual(walled_state.item(), 0.0)
        self.assertEqual(legacy_state.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
