"""Static / unit tests for the P5 privileged-policy headroom plan (codex_plan.md sec. 5).

These tests are CPU-only: they exercise config wiring, the command-schedule helper,
seed plumbing and the sweep isolation registry WITHOUT building a simulator env
(no GPU / genesis runtime required). Runtime smoke tests live elsewhere.

Run:  SIMULATOR=genesis .venv/bin/python -m unittest tests.test_bench_p5 -v
(or just: .venv/bin/python -m unittest tests/test_bench_p5.py)
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import copy
import types
import unittest

import legged_gym.envs  # noqa: F401  (side effect: registers benchmark tasks)
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import class_to_dict, update_cfg_from_args
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


SCHEDULE = [
    {"start_iteration": 0,   "lin_vel_x": [-0.5, 0.5]},
    {"start_iteration": 500, "lin_vel_x": [-1.0, 1.0]},
]


def _flatten(d, prefix=""):
    """Flatten a nested dict into {dotted.key: leaf_value}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _fresh_cfgs(task):
    """Deep copies of the registered cfgs so a test can mutate them freely."""
    env_cfg, train_cfg = task_registry.get_cfgs(task)
    return copy.deepcopy(env_cfg), copy.deepcopy(train_cfg)


def _args(**overrides):
    base = dict(
        num_envs=None, debug=False, motion_file=None, num_student=None,
        max_iterations=None, resume=False, sync_wandb=False, ckpt=None,
        load_run=None, seed=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class TestSeedOverride(unittest.TestCase):
    def test_seed_reaches_both_cfgs(self):
        env_cfg, train_cfg = _fresh_cfgs("go2_bench_mlp")
        update_cfg_from_args(env_cfg, train_cfg, _args(seed=7))
        self.assertEqual(train_cfg.seed, 7)
        self.assertEqual(env_cfg.seed, 7)

    def test_no_seed_leaves_default(self):
        env_cfg, train_cfg = _fresh_cfgs("go2_bench_mlp")
        default = train_cfg.seed
        update_cfg_from_args(env_cfg, train_cfg, _args(seed=None))
        self.assertEqual(train_cfg.seed, default)


class TestCommandSchedule(unittest.TestCase):
    def _rng(self, it):
        stage = OnPolicyRunner.command_stage_for_iter(SCHEDULE, it)
        return None if stage is None else stage["lin_vel_x"]

    def test_boundaries(self):
        self.assertEqual(self._rng(0),   [-0.5, 0.5])
        self.assertEqual(self._rng(499), [-0.5, 0.5])
        self.assertEqual(self._rng(500), [-1.0, 1.0])

    def test_resume_past_boundary(self):
        # a run resumed at any iteration >= 500 must land on the wide stage
        self.assertEqual(self._rng(1500), [-1.0, 1.0])
        self.assertEqual(self._rng(2999), [-1.0, 1.0])

    def test_unsorted_schedule_is_ordered(self):
        rng = OnPolicyRunner.command_stage_for_iter(list(reversed(SCHEDULE)), 500)
        self.assertEqual(rng["lin_vel_x"], [-1.0, 1.0])

    def test_none_or_empty_schedule(self):
        self.assertIsNone(OnPolicyRunner.command_stage_for_iter(None, 100))
        self.assertIsNone(OnPolicyRunner.command_stage_for_iter([], 100))

    def test_configured_schedule_matches_plan(self):
        _, train_cfg = _fresh_cfgs("go2_bench_mlp")
        sched = class_to_dict(train_cfg)["runner"]["command_schedule"]
        self.assertEqual(sched, SCHEDULE)


class TestMlpOracleCfgDiff(unittest.TestCase):
    def test_diff_limited_to_obs_and_runname(self):
        mlp_env, mlp_tr = _fresh_cfgs("go2_bench_mlp")
        orc_env, orc_tr = _fresh_cfgs("go2_bench_oracle_id")

        env_a, env_b = _flatten(class_to_dict(mlp_env)), _flatten(class_to_dict(orc_env))
        tr_a, tr_b = _flatten(class_to_dict(mlp_tr)), _flatten(class_to_dict(orc_tr))

        def diff(a, b):
            keys = set(a) | set(b)
            return {k for k in keys if a.get(k) != b.get(k)}

        env_diff = diff(env_a, env_b)
        tr_diff = diff(tr_a, tr_b)

        # Actor width and critic width differ: Oracle appends P5 to the actor,
        # and each asymmetric critic has its corresponding velocity append.
        self.assertTrue(
            env_diff <= {"env.num_observations", "env.num_privileged_obs"},
            f"unexpected env cfg diffs: {env_diff}",
        )
        # ...and only the run name in the train cfg.
        self.assertTrue(
            tr_diff <= {"runner.run_name"},
            f"unexpected train cfg diffs: {tr_diff}",
        )

    def test_dr_band_identical(self):
        mlp_env, _ = _fresh_cfgs("go2_bench_mlp")
        orc_env, _ = _fresh_cfgs("go2_bench_oracle_id")
        self.assertEqual(mlp_env.domain_rand.friction_range, orc_env.domain_rand.friction_range)
        self.assertEqual(mlp_env.domain_rand.added_mass_range, orc_env.domain_rand.added_mass_range)


class TestObservationDims(unittest.TestCase):
    def test_mlp_and_oracle_dims(self):
        mlp_env, _ = _fresh_cfgs("go2_bench_mlp")
        orc_env, _ = _fresh_cfgs("go2_bench_oracle_id")
        self.assertEqual(mlp_env.env.num_observations, 45)
        self.assertEqual(orc_env.env.num_observations, 50)
        self.assertEqual(mlp_env.env.num_privileged_obs, 48)
        self.assertEqual(orc_env.env.num_privileged_obs, 53)
        # P5 append is exactly 5 dims = [friction, added_mass, com_x, com_y, com_z]
        self.assertEqual(orc_env.env.num_observations - mlp_env.env.num_observations, 5)


class TestOracleVelVariant(unittest.TestCase):
    """go2_bench_oracle_id_vel = oracle_id + true base_lin_vel(3), else identical."""

    def test_obs_dim_53(self):
        env_cfg, _ = _fresh_cfgs("go2_bench_oracle_id_vel")
        self.assertEqual(env_cfg.env.num_observations, 53)  # 45 proprio + 5 P + 3 vel

    def test_diff_vs_oracle_id_limited(self):
        orc_env, orc_tr = _fresh_cfgs("go2_bench_oracle_id")
        vel_env, vel_tr = _fresh_cfgs("go2_bench_oracle_id_vel")
        env_a, env_b = _flatten(class_to_dict(orc_env)), _flatten(class_to_dict(vel_env))
        tr_a, tr_b = _flatten(class_to_dict(orc_tr)), _flatten(class_to_dict(vel_tr))

        def diff(a, b):
            return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}

        self.assertTrue(diff(env_a, env_b) <= {"env.num_observations"},
                        f"unexpected env diff: {diff(env_a, env_b)}")
        self.assertTrue(diff(tr_a, tr_b) <= {"runner.run_name"},
                        f"unexpected train diff: {diff(tr_a, tr_b)}")

    def test_vel_adds_exactly_3(self):
        orc_env, _ = _fresh_cfgs("go2_bench_oracle_id")
        vel_env, _ = _fresh_cfgs("go2_bench_oracle_id_vel")
        self.assertEqual(vel_env.env.num_observations - orc_env.env.num_observations, 3)

    def test_dr_band_and_schedule_match_oracle_id(self):
        orc_env, orc_tr = _fresh_cfgs("go2_bench_oracle_id")
        vel_env, vel_tr = _fresh_cfgs("go2_bench_oracle_id_vel")
        self.assertEqual(orc_env.domain_rand.friction_range, vel_env.domain_rand.friction_range)
        self.assertEqual(orc_env.domain_rand.added_mass_range, vel_env.domain_rand.added_mass_range)
        self.assertEqual(class_to_dict(orc_tr)["runner"]["command_schedule"],
                         class_to_dict(vel_tr)["runner"]["command_schedule"])


class TestInDistValidationField(unittest.TestCase):
    def test_full_field_lin_vel_x(self):
        from legged_gym.scripts.eval.indist import in_dist_command_ranges
        for task in ("go2_bench_mlp", "go2_bench_oracle_id"):
            env_cfg, _ = _fresh_cfgs(task)
            fake_env = types.SimpleNamespace(cfg=env_cfg)
            ranges = in_dist_command_ranges(fake_env)
            self.assertEqual(ranges["lin_vel_x"], [-1.0, 1.0], f"{task} validation field")

    def test_curriculum_disabled(self):
        for task in ("go2_bench_mlp", "go2_bench_oracle_id"):
            env_cfg, _ = _fresh_cfgs(task)
            self.assertFalse(env_cfg.commands.curriculum, f"{task} should not use perf curriculum")


class TestSweepAxisIsolation(unittest.TestCase):
    def test_registry_covers_p5(self):
        import legged_gym.scripts.eval.dr_axes as dr
        # every sweep axis maps to a privileged param
        self.assertEqual(set(dr._AXIS_TO_PRIV), set(dr.AXES))
        # P5 privileged params = friction, added_mass, com_bias(3)
        self.assertEqual(set(dr._PRIVILEGED_PARAMS), {"friction", "added_mass", "com_bias"})

    def test_pin_others_skips_swept_axis(self):
        import legged_gym.scripts.eval.dr_axes as dr
        called = []
        original = dr._PRIVILEGED_PARAMS
        # replace setters with recorders, keep nominals
        dr._PRIVILEGED_PARAMS = {
            name: ((lambda n: (lambda env, v: called.append(n)))(name), nom)
            for name, (setter, nom) in original.items()
        }
        try:
            fake_env = types.SimpleNamespace(num_envs=4, device="cpu")
            dr.pin_others_to_nominal(fake_env, "friction")
        finally:
            dr._PRIVILEGED_PARAMS = original
        # friction is the swept axis -> must NOT be pinned; the others must be
        self.assertNotIn("friction", called)
        self.assertEqual(set(called), {"added_mass", "com_bias"})

    def test_unknown_axis_raises(self):
        import legged_gym.scripts.eval.dr_axes as dr
        with self.assertRaises(KeyError):
            dr.pin_others_to_nominal(types.SimpleNamespace(num_envs=1, device="cpu"), "bogus")


if __name__ == "__main__":
    unittest.main(verbosity=2)
