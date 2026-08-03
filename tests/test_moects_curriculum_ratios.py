"""Ratio-based curriculum tests for ``go2_moects`` / ``go2_moects_him``.

Budget decision: the vendored absolute iteration thresholds (1500 / 5000 /
20000 / 50000 of the vendored 150k budget) are stored as their FRACTIONS of
150k, and the planned total budget defaults to 30000
(``env.curriculum_total_iterations`` == ``CfgPPO.runner.max_iterations``).
The mixin resolves every ratio against the effective total, which the runner
syncs at learn() time via ``set_wty_total_iterations`` (incl. any
``--max_iterations`` CLI override).

CPU-only checks (no simulator):
1. config pins: 30k budget on both arms, ratio values == vendored fractions,
   no stale ``iter``/``start_iter``/``end_iter`` keys.
2. mixin math: ``_wty_progress`` and ``_get_current_scale`` on a mock self,
   incl. vendored equivalence at total=150000 (ratios reproduce the
   vendored absolute thresholds exactly).
3. runner sync: both learn() implementations call set_wty_total_iterations.
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import inspect
import unittest
from types import SimpleNamespace

TOTAL_DEFAULT = 30000
VENDORED = 150000


def _mock_self(total, step):
    m = SimpleNamespace(
        common_step_counter=step,
        num_steps_per_iter=24,
        _wty_total_iterations=total)
    m._wty_progress = lambda: (m.common_step_counter // m.num_steps_per_iter) / m._wty_total_iterations
    return m


class TestCurriculumRatioConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import legged_gym.envs  # noqa: F401  (registers the tasks)
        from legged_gym.utils import task_registry
        cls.get_cfgs = staticmethod(task_registry.get_cfgs)

    def test_budget_defaults_30k_both_arms(self):
        for task in ("go2_moects", "go2_moects_him"):
            env_cfg, train_cfg = self.get_cfgs(task)
            self.assertEqual(train_cfg.runner.max_iterations, TOTAL_DEFAULT, task)
            self.assertEqual(env_cfg.env.curriculum_total_iterations, TOTAL_DEFAULT, task)

    def test_ratios_match_vendored_fractions(self):
        env_cfg, _ = self.get_cfgs("go2_moects")
        zc = env_cfg.commands.zero_command_curriculum
        self.assertAlmostEqual(zc["end_ratio"], 1500 / VENDORED)
        self.assertNotIn("start_iter", zc)
        self.assertNotIn("end_iter", zc)
        ranges = env_cfg.commands.command_range_curriculum
        self.assertEqual(len(ranges), 2)
        self.assertAlmostEqual(ranges[0]["ratio"], 20000 / VENDORED)
        self.assertAlmostEqual(ranges[1]["ratio"], 50000 / VENDORED)
        for entry in ranges:
            self.assertNotIn("iter", entry)
        cr = env_cfg.rewards.curriculum_rewards
        self.assertAlmostEqual(cr[0]["end_ratio"], 1500 / VENDORED)
        self.assertAlmostEqual(cr[1]["end_ratio"], 5000 / VENDORED)
        for entry in cr:
            self.assertNotIn("start_iter", entry)
            self.assertNotIn("end_iter", entry)


class TestRatioMath(unittest.TestCase):

    def test_progress(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        m = _mock_self(TOTAL_DEFAULT, 24 * 150)
        self.assertAlmostEqual(WtyCurriculumMixin._wty_progress(m), 150 / TOTAL_DEFAULT)

    def test_ramp_completes_at_ratio_times_total(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        cfg = {"start_ratio": 0.0, "end_ratio": 1500 / VENDORED,
               "start_value": 0.0, "end_value": 0.1}
        # at 30k the zero-command ramp completes at iter 300 (= 0.01 * 30000)
        m = _mock_self(TOTAL_DEFAULT, 24 * 300)
        self.assertAlmostEqual(WtyCurriculumMixin._get_current_scale(m, cfg), 0.1)
        m = _mock_self(TOTAL_DEFAULT, 24 * 150)
        self.assertAlmostEqual(WtyCurriculumMixin._get_current_scale(m, cfg), 0.05)

    def test_vendored_equivalence_at_150k(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        cfg = {"start_ratio": 0.0, "end_ratio": 1500 / VENDORED,
               "start_value": 0.0, "end_value": 0.1}
        # at the vendored 150k budget the ratios reproduce the vendored
        # absolute thresholds exactly (ramp completes at iter 1500)
        m = _mock_self(VENDORED, 24 * 1500)
        self.assertAlmostEqual(WtyCurriculumMixin._get_current_scale(m, cfg), 0.1)
        m = _mock_self(VENDORED, 24 * 750)
        self.assertAlmostEqual(WtyCurriculumMixin._get_current_scale(m, cfg), 0.05)

    def test_command_pop_thresholds_30k(self):
        # the two command-range pops fire at iter 4000 and 10000 of 30000
        self.assertEqual(int((20000 / VENDORED) * TOTAL_DEFAULT), 4000)
        self.assertEqual(int((50000 / VENDORED) * TOTAL_DEFAULT), 10000)

    def test_set_total_iterations_guard(self):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        resyncs = []
        m = SimpleNamespace(_wty_total_iterations=TOTAL_DEFAULT)
        m._wty_resync_curriculum = lambda: resyncs.append(m._wty_total_iterations)
        WtyCurriculumMixin.set_wty_total_iterations(m, 50000)
        self.assertEqual(m._wty_total_iterations, 50000)
        # the budget sync is also the resume resync point: derived curriculum
        # state is rebuilt against the effective budget
        self.assertEqual(resyncs, [50000])
        with self.assertRaises(AssertionError):
            WtyCurriculumMixin.set_wty_total_iterations(m, 0)
        self.assertEqual(resyncs, [50000])


class TestRunnerBudgetSync(unittest.TestCase):

    def test_both_learn_impls_sync_total_iterations(self):
        from rsl_rl.runners.moe_cts_runner import MoECTSRunner
        from rsl_rl.runners.him_runner import HIMRunner
        for cls in (MoECTSRunner, HIMRunner):
            self.assertIn("set_wty_total_iterations",
                          inspect.getsource(cls.learn), cls.__name__)


if __name__ == "__main__":
    unittest.main()
