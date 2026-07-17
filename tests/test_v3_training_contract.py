"""Fast, simulator-free checks for the V3 training contract."""

import unittest

from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict


V3_TASKS = (
    "go2_v3_mlp",
    "go2_v3_sysid",
    "go2_v3_rma",
    "go2_v3_dreamwaq",
    "go2_v3_him_fixed",
    "go2_v3_superset_oracle",
)


class TestV3TrainingContract(unittest.TestCase):
    def _cfgs(self, name):
        return task_registry.get_cfgs(name)

    def test_all_v3_tasks_are_registered(self):
        for name in V3_TASKS:
            env_cfg, train_cfg = self._cfgs(name)
            self.assertIsNotNone(env_cfg)
            self.assertIsNotNone(train_cfg)

    def test_shared_dynamic_physics_contract(self):
        for name in V3_TASKS:
            env_cfg, _ = self._cfgs(name)
            dr = env_cfg.domain_rand
            self.assertEqual(dr.added_mass_range, [-2.0, 5.0])
            self.assertEqual(dr.com_pos_x_range, [-0.08, 0.08])
            self.assertEqual(dr.com_pos_y_range, [-0.08, 0.08])
            self.assertEqual(dr.com_pos_z_range, [-0.08, 0.08])
            # Low-dose dynamic physics: exactly one switch per episode, fired
            # at a random step inside the middle fraction window of the episode.
            self.assertTrue(dr.resample_physics_within_episode)
            self.assertTrue(dr.physics_resample_mass)
            self.assertTrue(dr.physics_resample_com)
            self.assertEqual(dr.physics_resample_switch_window, [0.2, 0.8])
            # PD gain is intentionally not a V3 latent: P stays P5.
            self.assertFalse(dr.randomize_pd_gain)
            self.assertEqual(env_cfg.commands.ranges.lin_vel_x, [-1.0, 2.0])

    def test_observation_contracts(self):
        mlp, _ = self._cfgs("go2_v3_mlp")
        sysid, _ = self._cfgs("go2_v3_sysid")
        rma, _ = self._cfgs("go2_v3_rma")
        dreamwaq, _ = self._cfgs("go2_v3_dreamwaq")
        him_fixed, him_fixed_train = self._cfgs("go2_v3_him_fixed")
        oracle, _ = self._cfgs("go2_v3_superset_oracle")
        self.assertEqual(mlp.env.num_observations, 45)
        self.assertEqual(mlp.env.num_privileged_obs, 48)
        self.assertEqual(sysid.env.num_estimator_features, 20 * 45)
        self.assertEqual(sysid.env.num_estimator_labels, 8)
        self.assertEqual(sysid.env.num_privileged_obs, 5 * 53)
        self.assertEqual(rma.env.num_observations, 45)
        self.assertEqual(rma.env.num_history_obs, 20 * 45)
        self.assertEqual(rma.env.num_privileged_obs, 8)
        self.assertEqual(rma.env.num_critic_obs, 5 * 53)
        self.assertEqual(dreamwaq.env.num_observations, 45)
        self.assertEqual(dreamwaq.env.num_history_obs, 5 * 45)
        self.assertEqual(dreamwaq.env.num_privileged_obs, 5 * 53)
        self.assertEqual(him_fixed.env.num_observations, 6 * 45)
        self.assertEqual(him_fixed.env.num_privileged_obs, 5 * 53)
        self.assertEqual(him_fixed_train.runner.policy_class_name, "HIMActorCritic")
        self.assertEqual(him_fixed_train.runner.algorithm_class_name, "PPO_HIM")
        self.assertEqual(him_fixed_train.runner_class_name, "HIMRunner")
        self.assertEqual(oracle.env.num_observations, 20 * 45 + 3 + 5)
        self.assertEqual(oracle.env.num_privileged_obs, 20 * 45 + 3 + 5)

    def test_schedule_and_budget_match(self):
        expected_schedule = [
            {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
            {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
            {"start_iteration": 1500, "lin_vel_x": [-1.0, 2.0]},
        ]
        for name in V3_TASKS:
            _, train_cfg = self._cfgs(name)
            runner = class_to_dict(train_cfg)["runner"]
            self.assertEqual(runner["max_iterations"], 3000)
            self.assertEqual(runner["command_schedule"], expected_schedule)
            self.assertEqual(runner["eval_seed"], 12345)
            self.assertEqual(runner["eval_fall_guard"], 0.05)

    def test_legacy_wave1_contract_is_unchanged(self):
        legacy, _ = self._cfgs("go2_bench_mlp")
        self.assertEqual(legacy.domain_rand.added_mass_range, [-1.0, 1.0])
        self.assertEqual(legacy.commands.ranges.lin_vel_x, [-1.0, 1.0])
        self.assertFalse(hasattr(legacy.domain_rand, "resample_physics_within_episode"))


if __name__ == "__main__":
    unittest.main()
