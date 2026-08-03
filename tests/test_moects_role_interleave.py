"""Focused tests for the MoE-CTS interleaved teacher/student role mapping.

Reference parity (go2_rl_gym moe_cts.py:96-102): every
int(1/student_env_ratio)-th env is a student, the rest are teachers -- NOT
the host CTS contiguous [0, num_teacher) / [num_teacher, num_envs) blocks.
Because terrain columns are assigned contiguously by env id
(arange(N) // (N/num_cols)), the interleave is what gives the deployable
student policy coverage of EVERY terrain type.

CPU-only, pure tensor arithmetic (plus one ActorCriticMoECTS construction for
the act() env-order check); no simulator is built.

Run:  .venv/bin/python -m unittest tests.test_moects_role_interleave -v
(or:  .venv/bin/python -m pytest tests/test_moects_role_interleave.py -q)
"""

import unittest
from unittest import mock

import torch

from rsl_rl.algorithms.ppo_moe_cts import PPO_MOE_CTS, compute_role_env_idxs
from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageMoECTS

NUM_OBS = 45            # vendored student obs
NUM_PRIVILEGED = 263    # vendored teacher/privileged obs
FRAME_STACK = 5
NUM_HISTORY = NUM_OBS * FRAME_STACK   # 225
NUM_LATENT = 32
NUM_CRITIC = 263
NUM_ACTIONS = 12
EXPERT_NUM = 8

NUM_TERRAIN_COLS = 20   # go2_moects moe_grid: terrain.num_cols


def _make_ac():
    torch.manual_seed(0)
    return ActorCriticMoECTS(
        NUM_OBS, NUM_ACTIONS, NUM_PRIVILEGED, NUM_HISTORY, NUM_LATENT, NUM_CRITIC,
        expert_num=EXPERT_NUM, student_encoder_hidden_dims=[512, 256, 256],
        norm_type="l2norm", init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        privilege_encoder_hidden_dims=[512, 256],
    )


def _terrain_columns(num_envs):
    # Same formula as WtyCurriculumMixin._wty_setup_terrain_curriculum
    # (vendored _get_env_origins): contiguous column assignment by env id.
    return torch.div(torch.arange(num_envs), num_envs / NUM_TERRAIN_COLS,
                     rounding_mode="floor").to(torch.long)


class TestRoleIdxFormula(unittest.TestCase):
    def test_counts_and_student_set_8192(self):
        ti, si = compute_role_env_idxs(8192, 0.75, "cpu")
        # (a) exact reference counts
        self.assertEqual(len(ti), 6144)
        self.assertEqual(len(si), 2048)
        # (b) students are exactly the i % 4 == 0 envs
        self.assertTrue(torch.equal(si, torch.arange(0, 8192, 4)))
        self.assertTrue(bool((ti % 4 != 0).all()))
        # the two sets partition [0, 8192)
        combined = torch.cat((ti, si)).sort().values
        self.assertTrue(torch.equal(combined, torch.arange(8192)))

    def test_reference_combos(self):
        # (num_envs, ratio) -> expected student idxs (reference formula:
        # students = {i : i % int(1/(1-ratio)) == 0})
        cases = [
            (1024, 0.75, list(range(0, 1024, 4))),
            (16, 0.75, [0, 4, 8, 12]),
            (8, 0.5, [0, 2, 4, 6]),
            (6, 0.75, [0, 4]),
            (2, 0.75, [0]),
        ]
        for num_envs, ratio, expected_students in cases:
            with self.subTest(num_envs=num_envs, ratio=ratio):
                ti, si = compute_role_env_idxs(num_envs, ratio, "cpu")
                self.assertEqual(si.tolist(), expected_students)
                self.assertEqual(len(ti), max(int(num_envs * ratio), 1))
                self.assertEqual(len(si), num_envs - len(ti))
                combined = torch.cat((ti, si)).sort().values
                self.assertTrue(torch.equal(combined, torch.arange(num_envs)))

    def test_fail_loud_on_impossible_split(self):
        # num_envs too small: a student idx exists but the count guard says 0
        with self.assertRaises(AssertionError):
            compute_role_env_idxs(1, 0.75, "cpu")
        # ratio not of the form 1 - 1/k: idx counts cannot match the
        # max(int(num_envs*ratio), 1) guard (reference asserts fire the same way)
        with self.assertRaises(AssertionError):
            compute_role_env_idxs(8192, 0.7, "cpu")


class TestTerrainColumnCoverage(unittest.TestCase):
    """(c) With the interleave, EVERY terrain column trains both roles."""

    def _assert_column_coverage(self, num_envs):
        ti, si = compute_role_env_idxs(num_envs, 0.75, "cpu")
        cols = _terrain_columns(num_envs)
        self.assertEqual(cols.unique().numel(), NUM_TERRAIN_COLS)
        env_ids = torch.arange(num_envs)
        for c in range(NUM_TERRAIN_COLS):
            in_col = env_ids[cols == c]
            n_student = int(torch.isin(in_col, si).sum())
            n_teacher = int(torch.isin(in_col, ti).sum())
            with self.subTest(column=c):
                self.assertGreater(n_student, 0)
                self.assertGreater(n_teacher, 0)
                frac = n_student / (n_student + n_teacher)
                # 75/25 per column up to the column-width rounding (columns
                # are 409-410 envs at 8192, so exact quarters are impossible)
                self.assertGreater(frac, 0.20)
                self.assertLess(frac, 0.30)

    def test_every_column_has_both_roles_8192(self):
        self._assert_column_coverage(8192)

    def test_every_column_has_both_roles_1024(self):
        self._assert_column_coverage(1024)

    def test_contiguous_split_would_fail_this(self):
        # Guard against regressing to the old contiguous layout: with teacher
        # envs [0, 6144) and students [6144, 8192), whole terrain columns see
        # only one role (columns 16-19 unambiguously have zero teachers; the
        # 14/15 boundary is float32-rounding dependent and not asserted).
        num_envs = 8192
        cols = _terrain_columns(num_envs)
        is_teacher = torch.arange(num_envs) < 6144
        for c in (16, 17, 18, 19):
            self.assertEqual(int(is_teacher[cols == c].sum()), 0)
        # student envs never touch the early columns at all
        self.assertLess(cols[~is_teacher].unique().numel(), NUM_TERRAIN_COLS)
        self.assertEqual(int((~is_teacher)[cols == 0].sum()), 0)


class TestAlgorithmRoleWiring(unittest.TestCase):
    def test_num_teacher_inconsistent_with_ratio_fails_loud(self):
        ac = _make_ac()
        alg = PPO_MOE_CTS(ac, device="cpu", num_teacher=6)  # ratio default 0.75
        with self.assertRaises(AssertionError):
            # 16 envs * 0.75 = 12 teachers, not 6
            alg.init_storage(16, 4, [NUM_OBS], [NUM_PRIVILEGED],
                             [NUM_HISTORY], [NUM_CRITIC], [NUM_ACTIONS])

    def test_storage_requires_interleaved_idxs(self):
        # idx kwargs are mandatory: no silent contiguous fallback
        with self.assertRaises(TypeError):
            RolloutStorageMoECTS(8, 4, 2, (3,), (4,), (5,), (6,), (2,), device="cpu")
        # count mismatch vs num_teacher fails loudly (ratio 0.75 -> 6/2, so a
        # teacher/student swap cannot slip past the count asserts)
        ti, si = compute_role_env_idxs(8, 0.75, "cpu")
        with self.assertRaises(AssertionError):
            RolloutStorageMoECTS(8, 6, 2, (3,), (4,), (5,), (6,), (2,), device="cpu",
                                 teacher_env_idxs=si, student_env_idxs=ti)

    def test_act_returns_actions_in_env_order(self):
        # Sentinel role outputs: teacher rows = +obs, student rows = -obs. The
        # actions handed to env.step must be back in original env order.
        ac = _make_ac()
        num_envs, num_teacher = 8, 6   # ratio 0.75 -> students {0, 4}
        alg = PPO_MOE_CTS(ac, device="cpu", num_teacher=num_teacher)
        ti, si = compute_role_env_idxs(num_envs, alg.teacher_env_ratio, "cpu")
        self.assertEqual(si.tolist(), [0, 4])

        state = {}

        def fake_act(obs, obs_history, privileged_obs, act_type=None, **kwargs):
            sign = 1.0 if act_type == "teacher" else -1.0
            state["sign"], state["n"] = sign, obs.shape[0]
            return sign * obs[:, :NUM_ACTIONS]

        with mock.patch.object(ac, "act", side_effect=fake_act), \
             mock.patch.object(ac, "get_actions_log_prob",
                               side_effect=lambda a: a.sum(dim=-1)), \
             mock.patch.object(type(ac), "action_mean",
                               new_callable=mock.PropertyMock,
                               side_effect=lambda *a: state["sign"] * torch.full(
                                   (state["n"], NUM_ACTIONS), 2.0)), \
             mock.patch.object(type(ac), "action_std",
                               new_callable=mock.PropertyMock,
                               side_effect=lambda *a: torch.full(
                                   (state["n"], NUM_ACTIONS), 3.0)):
            obs = torch.randn(num_envs, NUM_OBS)
            priv = torch.randn(num_envs, NUM_PRIVILEGED)
            hist = torch.randn(num_envs, NUM_HISTORY)
            critic = torch.randn(num_envs, NUM_CRITIC)
            actions = alg.act(obs, priv, hist, critic)

        expected = obs[:, :NUM_ACTIONS].clone()
        expected[si] *= -1.0
        self.assertTrue(torch.equal(actions, expected))
        self.assertTrue(torch.equal(alg.transition.actions, expected))
        # every scattered transition tensor is env-ordered, not just actions
        expected_log_prob = obs[:, :NUM_ACTIONS].sum(dim=-1)
        expected_log_prob[si] *= -1.0
        self.assertTrue(torch.allclose(alg.transition.actions_log_prob, expected_log_prob))
        expected_mean = torch.full((num_envs, NUM_ACTIONS), 2.0)
        expected_mean[si] *= -1.0
        self.assertTrue(torch.equal(alg.transition.action_mean, expected_mean))
        self.assertTrue(torch.equal(
            alg.transition.action_sigma, torch.full((num_envs, NUM_ACTIONS), 3.0)))
        # values are role-aware (real evaluate path) and env-ordered too
        with torch.no_grad():
            expected_values = torch.empty(num_envs, 1)
            expected_values[ti] = ac.evaluate(critic[ti], is_teacher=True)
            expected_values[si] = ac.evaluate(critic[si], hist[si], is_teacher=False)
        self.assertTrue(torch.allclose(alg.transition.values, expected_values, atol=1e-6))


class TestStorageGAEInterleaved(unittest.TestCase):
    """RolloutStorageMoECTS.compute_returns on interleaved role gathers."""

    @staticmethod
    def _manual_gae(values, rewards, dones, last_values, gamma, lam):
        steps = values.shape[0]
        returns = torch.empty_like(values)
        advantage = 0
        for step in reversed(range(steps)):
            next_values = last_values if step == steps - 1 else values[step + 1]
            not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + not_terminal * gamma * next_values - values[step]
            advantage = delta + not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]
        return returns

    def test_gae_matches_manual_per_role_computation(self):
        torch.manual_seed(3)
        num_envs, num_teacher, steps = 8, 4, 5   # ratio 0.5 -> students {0,2,4,6}
        ti, si = compute_role_env_idxs(num_envs, 0.5, "cpu")
        storage = RolloutStorageMoECTS(
            num_envs, num_teacher, steps, (3,), (4,), (5,), (6,), (2,), device="cpu",
            teacher_env_idxs=ti, student_env_idxs=si)
        gamma, lam = 0.99, 0.95
        storage.values = torch.randn(steps, num_envs, 1)
        storage.rewards = torch.randn(steps, num_envs, 1)
        storage.dones = (torch.rand(steps, num_envs, 1) < 0.2).byte()
        last_values = torch.randn(num_envs, 1)   # env order, as PPO_MOE_CTS passes it

        storage.compute_returns(last_values, gamma, lam)

        returns_t = self._manual_gae(storage.values[:, ti], storage.rewards[:, ti],
                                     storage.dones[:, ti], last_values[ti], gamma, lam)
        returns_s = self._manual_gae(storage.values[:, si], storage.rewards[:, si],
                                     storage.dones[:, si], last_values[si], gamma, lam)
        # returns are scattered back into the env-order buffer
        expected_returns = torch.empty(steps, num_envs, 1)
        expected_returns[:, ti] = returns_t
        expected_returns[:, si] = returns_s
        self.assertTrue(torch.allclose(storage.returns, expected_returns, atol=1e-6))
        # advantages are normalized GLOBALLY over both roles (vendored
        # go2_rl_gym rollout_storage_cts.py:141-143 parity), not per role
        adv_t = returns_t - storage.values[:, ti]
        adv_s = returns_s - storage.values[:, si]
        adv_all = torch.cat((adv_t, adv_s), dim=1)
        adv_mean, adv_std = adv_all.mean(), adv_all.std()
        adv_t = (adv_t - adv_mean) / (adv_std + 1e-8)
        adv_s = (adv_s - adv_mean) / (adv_std + 1e-8)
        self.assertTrue(torch.allclose(storage.teacher_advantages, adv_t, atol=1e-6))
        self.assertTrue(torch.allclose(storage.student_advantages, adv_s, atol=1e-6))
        # global normalization: only the concatenated mean is zero
        self.assertAlmostEqual(float(adv_all.mean() - adv_mean), 0.0, places=6)
        self.assertAlmostEqual(
            float(torch.cat((storage.teacher_advantages,
                             storage.student_advantages), dim=1).mean()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
