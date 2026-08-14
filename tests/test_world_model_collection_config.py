import unittest

from dribblebot.world_model.config import load_config


class WorldModelCollectionConfigTests(unittest.TestCase):
    def test_as2_collection_uses_goal_oriented_symmetric_mixture(self):
        config = load_config("configs/world_model_as2.yaml")
        mixture = config["data_collection"]["behavior_mixture"]

        self.assertAlmostEqual(mixture["scripted"], 0.90)
        self.assertAlmostEqual(mixture["random_valid"], 0.10)
        self.assertAlmostEqual(sum(mixture.values()), 1.0)
        self.assertEqual(config["environment"]["num_robots"], 4)
        self.assertEqual(config["environment"]["team_size"], 2)
        self.assertEqual(config["data_collection"]["num_episodes"], 20000)
        self.assertEqual(config["world_model"]["max_obstacles"], 0)
        self.assertTrue(config["data_collection"]["targeted_rare_events"]["enabled"])
        self.assertAlmostEqual(
            config["data_collection"]["random_sampling"]["goal_directed"], 0.40
        )
        self.assertTrue(config["data_collection"]["require_minimum_event_counts"])
