"""V5 UED allowlist contract tests (solun_plani.md §12, subplans/05 §Kontrat).

The four arms (`handcrafted_v4`, `uniform`, `lp_acrl`, `alp`) must share an
IDENTICAL PPO / reward / DR / actor-critic / task-support / budget contract.
Only three mechanisms may differ, and only in the documented direction:

  1. the episode-task sampler itself (none for handcrafted_v4 vs.
     uniform/lp_acrl/alp), which necessarily also carries the terrain
     promotion mechanism (§2: the sampler REPLACES terrain promotion +
     command curriculum for the UED arms);
  2. command_schedule / command-curriculum: ON only for handcrafted_v4
     (§14.5);
  3. standstill mechanism: batch-wide (handcrafted_v4) vs. per-env (UED arms)
     (§14.4).

Any config diff NOT covered by these three buckets must fail this test.
"""
from __future__ import annotations

import unittest

from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict


V5_TASKS = ("go2_v5_handcrafted", "go2_v5_uniform", "go2_v5_lpacrl", "go2_v5_alp")
V5_UED_TASKS = ("go2_v5_uniform", "go2_v5_lpacrl", "go2_v5_alp")

# Dotted-path keys allowed to differ across arms, bucketed by the allowlisted
# mechanism responsible (see module docstring). Any OTHER key that differs
# between two arms' resolved config dicts fails the test.
ALLOWED_ENV_DIFF_KEYS = {
    # (1) episode-task sampler presence / identity, and the terrain-promotion
    # mechanism it replaces for the UED arms.
    "env.ued_enabled",
    "curriculum.algorithm",
    "terrain.curriculum",
    "terrain.num_rows",
    "terrain.num_cols",
    "terrain.ued_training_grid",
    "terrain.ued_training_seed",
    # (2) command_schedule / command-curriculum: only handcrafted_v4 keeps
    # the legacy performance-based widening switched on.
    "commands.legacy_performance_command_curriculum_enabled",
    # (3) standstill mechanism: batch-wide vs. per-env.
    "commands.per_env_standstill",
}
ALLOWED_TRAIN_DIFF_KEYS = {
    # naming only, not a training-semantic parameter
    "runner.run_name",
    # (2) command_schedule: only handcrafted_v4 has one.
    "runner.command_schedule",
}


def _flatten(value, prefix=""):
    """Flatten a class_to_dict()-produced nested dict into {dotted_key: value}."""
    flat = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            flat.update(_flatten(sub, f"{prefix}.{key}" if prefix else str(key)))
    else:
        flat[prefix] = value
    return flat


def _diff_keys(flat_a, flat_b):
    keys = set(flat_a) | set(flat_b)
    return {k for k in keys if flat_a.get(k, _MISSING) != flat_b.get(k, _MISSING)}


_MISSING = object()


class TestV5TrainingContract(unittest.TestCase):
    def _cfgs(self, name):
        return task_registry.get_cfgs(name)

    def test_all_v5_arms_are_registered(self):
        for name in V5_TASKS:
            env_cfg, train_cfg = self._cfgs(name)
            self.assertIsNotNone(env_cfg)
            self.assertIsNotNone(train_cfg)

    def test_env_cfg_allowlist_across_all_four_arms(self):
        flats = {}
        for name in V5_TASKS:
            env_cfg, _ = self._cfgs(name)
            flats[name] = _flatten(class_to_dict(env_cfg))
        names = list(V5_TASKS)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                diff = _diff_keys(flats[a], flats[b])
                unexpected = diff - ALLOWED_ENV_DIFF_KEYS
                self.assertFalse(
                    unexpected,
                    f"{a} vs {b}: non-allowlisted env_cfg diff at {sorted(unexpected)}",
                )

    def test_train_cfg_allowlist_across_all_four_arms(self):
        flats = {}
        for name in V5_TASKS:
            _, train_cfg = self._cfgs(name)
            flats[name] = _flatten(class_to_dict(train_cfg))
        names = list(V5_TASKS)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                diff = _diff_keys(flats[a], flats[b])
                unexpected = diff - ALLOWED_TRAIN_DIFF_KEYS
                self.assertFalse(
                    unexpected,
                    f"{a} vs {b}: non-allowlisted train_cfg diff at {sorted(unexpected)}",
                )

    def test_handcrafted_v4_keeps_command_schedule_and_curriculum_on(self):
        env_cfg, train_cfg = self._cfgs("go2_v5_handcrafted")
        self.assertTrue(env_cfg.terrain.curriculum)
        self.assertTrue(
            getattr(env_cfg.commands, "legacy_performance_command_curriculum_enabled", True)
        )
        self.assertFalse(getattr(env_cfg.commands, "per_env_standstill", False))
        self.assertFalse(env_cfg.env.ued_enabled)
        schedule = class_to_dict(train_cfg)["runner"]["command_schedule"]
        self.assertIsNotNone(schedule)
        self.assertEqual(schedule[-1]["lin_vel_x"], [0.0, 2.0])

    def test_ued_arms_disable_legacy_curriculum_and_use_per_env_standstill(self):
        for name in V5_UED_TASKS:
            env_cfg, train_cfg = self._cfgs(name)
            self.assertTrue(env_cfg.env.ued_enabled)
            self.assertFalse(env_cfg.terrain.curriculum)
            self.assertFalse(env_cfg.commands.legacy_performance_command_curriculum_enabled)
            self.assertTrue(env_cfg.commands.per_env_standstill)
            self.assertIsNone(class_to_dict(train_cfg)["runner"]["command_schedule"])
            self.assertEqual((env_cfg.terrain.num_rows, env_cfg.terrain.num_cols), (4, 6))
            self.assertTrue(env_cfg.terrain.ued_training_grid)

    def test_shared_forward_command_support_and_v_scale(self):
        # All four arms share the same [0, 2] m/s support -> SPNTE v_scale=2.0.
        for name in V5_TASKS:
            env_cfg, _ = self._cfgs(name)
            self.assertEqual(env_cfg.commands.ranges.lin_vel_x, [0.0, 2.0])
            v_scale = max(abs(env_cfg.commands.ranges.lin_vel_x[0]), abs(env_cfg.commands.ranges.lin_vel_x[1]))
            self.assertEqual(v_scale, 2.0)

    def test_shared_ppo_reward_dr_actor_critic_and_budget(self):
        rewards_by_arm = set()
        dr_by_arm = set()
        for name in V5_TASKS:
            env_cfg, train_cfg = self._cfgs(name)
            self.assertEqual(env_cfg.env.num_observations, 45)
            self.assertEqual(env_cfg.env.num_privileged_obs, 48)
            self.assertEqual(env_cfg.env.num_actions, 12)
            self.assertFalse(hasattr(env_cfg.env, "oracle_height_map"))
            self.assertFalse(hasattr(env_cfg.env, "rma_teacher_height_map"))
            self.assertEqual(env_cfg.env.num_envs, 4096)
            self.assertEqual(env_cfg.env.episode_length_s, 20)
            self.assertTrue(env_cfg.domain_rand.resample_physics_within_episode)
            self.assertTrue(env_cfg.domain_rand.physics_resample_mass)
            self.assertTrue(env_cfg.domain_rand.physics_resample_com)
            self.assertEqual(env_cfg.domain_rand.added_mass_range, [-2.0, 5.0])
            rewards_by_arm.add(repr(sorted(class_to_dict(env_cfg.rewards).items())))
            dr_by_arm.add(repr(sorted(class_to_dict(env_cfg.domain_rand).items())))
            runner = class_to_dict(train_cfg)["runner"]
            self.assertEqual(runner["max_iterations"], 3000)
            self.assertEqual(runner["num_steps_per_env"], 24)
            self.assertEqual(runner["save_interval"], 200)
            self.assertEqual(runner["eval_seed"], 12345)
            algorithm = class_to_dict(train_cfg)["algorithm"]
            policy = class_to_dict(train_cfg)["policy"]
            self.assertEqual(policy["actor_hidden_dims"], [512, 256, 128])
            self.assertEqual(policy["critic_hidden_dims"], [512, 256, 128])
            self.assertEqual(algorithm["learning_rate"], 1e-3)
        # Rewards and DR must be BYTE-IDENTICAL across arms (single shared dict).
        self.assertEqual(len(rewards_by_arm), 1, "reward config drifted across V5 arms")
        self.assertEqual(len(dr_by_arm), 1, "domain_rand config drifted across V5 arms")

    def test_geometry_command_bank_and_eval_seed_shared_across_ued_arms(self):
        from legged_gym.envs.go2.go2_v5_config import build_ued_teacher

        fingerprints = set()
        eval_seeds = set()
        for name in V5_TASKS:
            env_cfg, train_cfg = self._cfgs(name)
            eval_seeds.add(class_to_dict(train_cfg)["runner"]["eval_seed"])
        self.assertEqual(len(eval_seeds), 1, "eval_seed diverged across V5 arms")
        for name in V5_UED_TASKS:
            env_cfg, _ = self._cfgs(name)
            _, task_space = build_ued_teacher(env_cfg)
            fingerprints.add(task_space.fingerprint())
        self.assertEqual(
            len(fingerprints), 1,
            "UED-teacher arms must share one deterministic training-grid geometry/task-space fingerprint",
        )

    def test_curriculum_checkpoint_refuses_foreign_fingerprint(self):
        from legged_gym.envs.go2.go2_v5_config import Go2V5LPACRLCfg, Go2V5UniformCfg, build_ued_teacher

        lp_curriculum, _ = build_ued_teacher(Go2V5LPACRLCfg())
        state = lp_curriculum.state_dict()

        # Different algorithm -> refused.
        uniform_curriculum, _ = build_ued_teacher(Go2V5UniformCfg())
        with self.assertRaises(ValueError):
            uniform_curriculum.load_state_dict(state)

        # Different task-space fingerprint (different deterministic terrain
        # builder seed) -> refused, even for the same algorithm.
        other_cfg = Go2V5LPACRLCfg()
        other_cfg.terrain.ued_training_seed = 999
        other_curriculum, _ = build_ued_teacher(other_cfg)
        with self.assertRaises(ValueError):
            other_curriculum.load_state_dict(state)

        # Different curriculum config (stage length) -> refused even with the
        # same algorithm and task space.
        other_stage_cfg = Go2V5LPACRLCfg()
        other_stage_cfg.curriculum.stage_length_control_steps = 4000
        other_stage_curriculum, _ = build_ued_teacher(other_stage_cfg)
        with self.assertRaises(ValueError):
            other_stage_curriculum.load_state_dict(state)

    def test_handcrafted_v4_builds_no_teacher(self):
        from legged_gym.envs.go2.go2_v5_config import Go2V5HandcraftedCfg, build_ued_teacher

        curriculum, task_space = build_ued_teacher(Go2V5HandcraftedCfg())
        self.assertIsNone(curriculum)
        self.assertIsNone(task_space)

    def test_unknown_algorithm_fails_closed(self):
        from legged_gym.envs.go2.go2_v5_config import Go2V5UniformCfg, build_ued_teacher

        bad_cfg = Go2V5UniformCfg()
        bad_cfg.curriculum.algorithm = "not_a_real_algorithm"
        with self.assertRaises(ValueError):
            build_ued_teacher(bad_cfg)


if __name__ == "__main__":
    unittest.main()
