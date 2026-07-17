"""Fast, simulator-free checks for the V4 fixed-medium-terrain campaign."""

import unittest

from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict


V4_TASKS = (
    "go2_v4_mlp",
    "go2_v4_sysid",
    "go2_v4_rma",
    "go2_v4_dreamwaq",
    "go2_v4_him_fixed",
    "go2_v4_superset_oracle",
)


class TestV4TrainingContract(unittest.TestCase):
    def _cfgs(self, name):
        return task_registry.get_cfgs(name)

    def test_all_v4_tasks_are_registered(self):
        for name in V4_TASKS:
            env_cfg, train_cfg = self._cfgs(name)
            self.assertIsNotNone(env_cfg)
            self.assertIsNotNone(train_cfg)

    def test_shared_terrain_and_dynamic_physics_contract(self):
        for name in V4_TASKS:
            env_cfg, _ = self._cfgs(name)
            terrain = env_cfg.terrain
            dr = env_cfg.domain_rand
            self.assertEqual(terrain.mesh_type, "heightfield")
            self.assertTrue(terrain.curriculum)
            self.assertFalse(terrain.selected)
            self.assertEqual(terrain.fixed_terrain_level, 5)
            self.assertEqual((terrain.num_rows, terrain.num_cols), (10, 10))
            self.assertTrue(terrain.measure_heights)
            self.assertEqual(
                len(terrain.measured_points_x) * len(terrain.measured_points_y), 17 * 11
            )
            self.assertEqual(dr.added_mass_range, [-2.0, 5.0])
            self.assertTrue(dr.resample_physics_within_episode)
            self.assertTrue(dr.physics_resample_mass)
            self.assertTrue(dr.physics_resample_com)

    def test_height_map_is_privileged_to_oracle_and_rma_teacher_only(self):
        mlp, _ = self._cfgs("go2_v4_mlp")
        rma, _ = self._cfgs("go2_v4_rma")
        oracle, _ = self._cfgs("go2_v4_superset_oracle")
        self.assertEqual(mlp.env.num_observations, 45)
        self.assertEqual(mlp.env.num_privileged_obs, 48)
        self.assertFalse(hasattr(mlp.env, "oracle_height_map"))
        self.assertTrue(rma.env.rma_teacher_height_map)
        self.assertEqual(rma.env.height_map_dim, 17 * 11)
        self.assertEqual(rma.env.num_privileged_obs, 5 + 3 + 17 * 11)
        self.assertEqual(rma.env.num_latent_dims, 8)
        self.assertTrue(oracle.env.oracle_height_map)
        self.assertEqual(oracle.env.height_map_dim, 17 * 11)
        self.assertEqual(oracle.env.num_observations, 20 * 45 + 3 + 5 + 17 * 11)
        self.assertEqual(oracle.env.num_privileged_obs, oracle.env.num_observations)

    def test_method_shapes_and_runner_protocol_are_preserved(self):
        sysid, sysid_train = self._cfgs("go2_v4_sysid")
        rma, _ = self._cfgs("go2_v4_rma")
        dreamwaq, _ = self._cfgs("go2_v4_dreamwaq")
        him, him_train = self._cfgs("go2_v4_him_fixed")
        self.assertEqual(sysid.env.num_estimator_features, 20 * 45)
        self.assertEqual(sysid.env.num_privileged_obs, 5 * 53)
        self.assertEqual(rma.env.num_history_obs, 20 * 45)
        self.assertEqual(rma.env.num_privileged_obs, 5 + 3 + 17 * 11)
        self.assertEqual(rma.env.num_latent_dims, 8)
        self.assertEqual(rma.env.num_critic_obs, 5 * 53)
        self.assertEqual(dreamwaq.env.num_history_obs, 5 * 45)
        self.assertEqual(him.env.num_observations, 6 * 45)
        self.assertEqual(him_train.runner.policy_class_name, "HIMActorCritic")
        self.assertEqual(him_train.runner.algorithm_class_name, "PPO_HIM")
        self.assertEqual(him_train.runner_class_name, "HIMRunner")
        for name in V4_TASKS:
            _, train_cfg = self._cfgs(name)
            runner = class_to_dict(train_cfg)["runner"]
            self.assertEqual(runner["experiment_name"], "go2_v4_medium_terrain")
            self.assertEqual(runner["max_iterations"], 3000)
            self.assertEqual(runner["eval_seed"], 12345)

    def test_v3_remains_flat_and_map_free(self):
        v3, _ = self._cfgs("go2_v3_superset_oracle")
        self.assertEqual(v3.terrain.mesh_type, "plane")
        self.assertFalse(v3.terrain.measure_heights)
        self.assertFalse(hasattr(v3.env, "oracle_height_map"))


if __name__ == "__main__":
    unittest.main()
