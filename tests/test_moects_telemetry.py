"""Training-telemetry tests for the go2_moects MoE-CTS port.

Group 1 (algorithm, rsl_rl/algorithms/ppo_moe_cts.py): PPO_MOE_CTS.update()
carries the standard PPO health keys in moe_stats -- entropy / kl_mean /
clip_frac / explained_variance -- with formula-level checks on synthetic
data (known ratios, known mu/sigma pairs, known values/returns).
Group 2 (env, legged_gym/envs/go2/go2_moects/wty_curriculum_mixin.py):
WtyCurriculumMixin._write_episode_telemetry emits the Episode/* extras keys
with the right names and per-episode means (bare mixin instance, no
simulator build).
Runner (rsl_rl/runners/moe_cts_runner.py): _log_moe_stats channel mapping
and the Loss/reconstruction dedup proxy.

CPU-only, no simulator env. Run:
  .venv/bin/python -m unittest tests/test_moects_telemetry.py -v
(or:  .venv/bin/python -m pytest tests/test_moects_telemetry.py -q)
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import math
import unittest
from collections import namedtuple
from types import SimpleNamespace

import torch

from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.algorithms import PPO_MOE_CTS
from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import WtyCurriculumMixin

NUM_OBS = 45            # vendored student obs
NUM_PRIVILEGED = 263    # vendored teacher/privileged obs
FRAME_STACK = 5
NUM_HISTORY = NUM_OBS * FRAME_STACK   # 225
NUM_LATENT = 32
NUM_CRITIC = 263        # == privileged (c_frame_stack = 1)
NUM_ACTIONS = 12
EXPERT_NUM = 8


def _make_ac():
    torch.manual_seed(0)
    return ActorCriticMoECTS(
        NUM_OBS, NUM_ACTIONS, NUM_PRIVILEGED, NUM_HISTORY, NUM_LATENT, NUM_CRITIC,
        expert_num=EXPERT_NUM, student_encoder_hidden_dims=[512, 256, 256],
        norm_type="l2norm", init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        privilege_encoder_hidden_dims=[512, 256],
    )


# Mirror of the storage track's CriticMiniBatch namedtuple (pinned contract,
# same as tests/test_moects_contract.py):
# (critic_observations, observation_histories, target_values, returns,
#  advantages, old_actions_log_prob, old_mu, old_sigma)
CriticMiniBatch = namedtuple("CriticMiniBatch", [
    "critic_observations", "observation_histories", "target_values", "returns",
    "advantages", "old_actions_log_prob", "old_mu", "old_sigma",
])


class _FakeTelemetryStorage:
    """_FakeMoECTSStorage (test_moects_contract) plus the values/returns
    buffers the explained-variance telemetry reads from the real
    RolloutStorageCTS/MoECTS contract ([T, num_envs, 1], env order).

    expose_value_buffers=False simulates a storage without the buffers, so
    the explained_variance stat must degrade to None instead of crashing.
    """

    def __init__(self, num_envs, num_teacher, num_transitions_per_env,
                 values=None, returns=None, expose_value_buffers=True):
        self.num_envs = num_envs
        self.num_teacher = num_teacher
        self.num_transitions_per_env = num_transitions_per_env
        self.cleared = False
        if expose_value_buffers:
            T = num_transitions_per_env
            self.values = values if values is not None else torch.randn(T, num_envs, 1)
            self.returns = returns if returns is not None else torch.randn(T, num_envs, 1)

    def clear(self):
        self.cleared = True

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        g = torch.Generator().manual_seed(0)
        T = self.num_transitions_per_env
        t_n = self.num_teacher * T // num_mini_batches
        s_n = (self.num_envs - self.num_teacher) * T // num_mini_batches

        def policy_batch(n, with_history):
            batch = [torch.randn(n, NUM_OBS, generator=g),
                     torch.randn(n, NUM_PRIVILEGED, generator=g)]
            if with_history:
                batch.append(torch.randn(n, NUM_HISTORY, generator=g))
            batch += [torch.randn(n, NUM_ACTIONS, generator=g),       # actions
                      torch.randn(n, 1, generator=g),                 # old log prob
                      torch.randn(n, 1, generator=g)]                 # advantages
            if not with_history:
                batch += [torch.randn(n, NUM_ACTIONS, generator=g),   # old mu
                          torch.rand(n, NUM_ACTIONS, generator=g) + 0.5]  # old sigma
            return tuple(batch)

        def critic_batch(n):
            return CriticMiniBatch(
                critic_observations=torch.randn(n, NUM_CRITIC, generator=g),
                observation_histories=torch.randn(n, NUM_HISTORY, generator=g),
                target_values=torch.randn(n, 1, generator=g),
                returns=torch.randn(n, 1, generator=g),
                advantages=torch.randn(n, 1, generator=g),
                old_actions_log_prob=torch.randn(n, 1, generator=g),
                old_mu=torch.randn(n, NUM_ACTIONS, generator=g),
                old_sigma=torch.rand(n, NUM_ACTIONS, generator=g) + 0.5,
            )

        for _ in range(num_epochs):
            for _ in range(num_mini_batches):
                yield (policy_batch(t_n, with_history=False),
                       policy_batch(s_n, with_history=True),
                       critic_batch(t_n), critic_batch(s_n))


class TestUpdateHealthStats(unittest.TestCase):
    """update() moe_stats carries the Group-1 PPO health keys."""

    def setUp(self):
        self.ac = _make_ac()
        self.num_envs, self.num_teacher, self.steps = 8, 6, 4
        self.alg = PPO_MOE_CTS(self.ac, device="cpu", num_teacher=self.num_teacher,
                               num_learning_epochs=1, num_mini_batches=2,
                               load_balance_coef=0.01)
        # returns = 2 * values + 1  ->  explained variance is exactly 0.75
        # (1 - Var(v + 1) / Var(2v + 1) = 1 - Var(v) / (4 Var(v))).
        v = torch.arange(self.steps * self.num_envs, dtype=torch.float
                         ).reshape(self.steps, self.num_envs, 1)
        self.storage = _FakeTelemetryStorage(
            self.num_envs, self.num_teacher, self.steps,
            values=v, returns=2.0 * v + 1.0)
        self.alg.storage = self.storage

    def test_health_keys_present_and_sane(self):
        out = self.alg.update()
        self.assertEqual(len(out), 5)
        stats = out[4]
        for key in ("entropy", "kl_mean", "clip_frac", "explained_variance"):
            self.assertIn(key, stats)
        self.assertTrue(math.isfinite(stats["entropy"]))
        self.assertGreaterEqual(stats["clip_frac"], 0.0)
        self.assertLessEqual(stats["clip_frac"], 1.0)
        # kl_mean is captured even with the default fixed LR schedule (the
        # override records whenever desired_kl is set), as a plain float.
        self.assertIsInstance(stats["kl_mean"], float)
        self.assertTrue(math.isfinite(stats["kl_mean"]))
        self.assertEqual(stats["kl_mean"], self.alg.last_kl_mean)

    def test_explained_variance_formula(self):
        stats = self.alg.update()[4]
        self.assertAlmostEqual(stats["explained_variance"], 0.75, places=5)

    def test_explained_variance_none_when_storage_lacks_buffers(self):
        self.alg.storage = _FakeTelemetryStorage(
            self.num_envs, self.num_teacher, self.steps, expose_value_buffers=False)
        stats = self.alg.update()[4]
        self.assertIsNone(stats["explained_variance"])

    def test_diagnostic_keys_present_and_sane(self):
        stats = self.alg.update()[4]
        for key in ("entropy_teacher", "entropy_student",
                    "kl_teacher", "kl_student",
                    "clip_frac_teacher", "clip_frac_student", "ratio_dev_max",
                    "value_loss_teacher", "value_loss_student",
                    "grad_norm", "encoder_grad_norm",
                    "latent_cosine", "latent_ev", "action_gap_mse",
                    "gating_max_weight", "experts_alive"):
            self.assertIn(key, stats)
            self.assertTrue(math.isfinite(stats[key]), key)
        for role in ("teacher", "student"):
            self.assertGreaterEqual(stats[f"clip_frac_{role}"], 0.0)
            self.assertLessEqual(stats[f"clip_frac_{role}"], 1.0)
            self.assertGreaterEqual(stats[f"value_loss_{role}"], 0.0)
        self.assertGreaterEqual(stats["grad_norm"], 0.0)
        self.assertGreaterEqual(stats["action_gap_mse"], 0.0)
        # gating weights are a softmax over EXPERT_NUM experts
        self.assertGreaterEqual(stats["gating_max_weight"], 1.0 / EXPERT_NUM)
        self.assertLessEqual(stats["gating_max_weight"], 1.0)
        self.assertGreaterEqual(stats["experts_alive"], 1.0)
        self.assertLessEqual(stats["experts_alive"], EXPERT_NUM)
        self.assertLessEqual(stats["latent_cosine"], 1.0 + 1e-6)

    def test_role_explained_variance_split(self):
        # values/returns are role-independent here (returns = 2v + 1 on every
        # env), so both role EVs must land on the aggregate 0.75.
        stats = self.alg.update()[4]
        self.assertAlmostEqual(stats["explained_variance_teacher"], 0.75, places=5)
        self.assertAlmostEqual(stats["explained_variance_student"], 0.75, places=5)

    def test_role_explained_variance_none_without_role_idxs(self):
        # A storage without value buffers (or an algorithm whose role mapping
        # was never built) degrades to None instead of crashing the run.
        self.alg.storage = _FakeTelemetryStorage(
            self.num_envs, self.num_teacher, self.steps, expose_value_buffers=False)
        stats = self.alg.update()[4]
        self.assertIsNone(stats["explained_variance_teacher"])
        self.assertIsNone(stats["explained_variance_student"])

    def test_raw_advantage_stats_forwarded_from_storage(self):
        self.alg.storage.raw_adv_stats = {"adv_std_teacher": 1.5,
                                          "adv_std_student": 4.5,
                                          "adv_std_ratio": 3.0}
        stats = self.alg.update()[4]
        self.assertEqual(stats["adv_std_ratio"], 3.0)
        self.assertEqual(stats["adv_std_teacher"], 1.5)
        self.assertEqual(stats["adv_std_student"], 4.5)

    def test_per_role_telemetry_is_not_the_aggregate(self):
        """The split must actually be computed per role, not copied."""
        alg = PPO_MOE_CTS(_make_ac(), device="cpu", num_teacher=6, clip_param=0.2)
        tel = alg._compute_minibatch_telemetry(
            torch.tensor([1.0, 1.0]),           # teacher: nothing clipped
            torch.tensor([1.5, 0.5]),           # student: both clipped
            torch.tensor([0.5, 0.5, 1.5, 1.5]),
            teacher_entropy=torch.tensor([0.5, 0.5]),
            student_entropy=torch.tensor([1.5, 1.5]))
        self.assertAlmostEqual(tel["clip_frac"], 0.5)
        self.assertAlmostEqual(tel["clip_frac_teacher"], 0.0)
        self.assertAlmostEqual(tel["clip_frac_student"], 1.0)
        self.assertAlmostEqual(tel["entropy"], 1.0)
        self.assertAlmostEqual(tel["entropy_teacher"], 0.5)
        self.assertAlmostEqual(tel["entropy_student"], 1.5)
        self.assertAlmostEqual(tel["ratio_dev_max"], 0.5, places=6)


class TestMinibatchTelemetryFormulas(unittest.TestCase):
    """Formula-level checks on the per-minibatch helpers."""

    def setUp(self):
        self.ac = _make_ac()
        self.alg = PPO_MOE_CTS(self.ac, device="cpu", num_teacher=6, clip_param=0.2)

    def test_clip_frac_known_ratios(self):
        # |1.3 - 1| = 0.3 and |0.7 - 1| = 0.3 exceed clip_param 0.2;
        # 1.0 and 1.1 do not -> 2 of 4 concatenated samples clipped.
        teacher_ratio = torch.tensor([1.0, 1.3, 0.7])
        student_ratio = torch.tensor([1.1])
        entropy = torch.tensor([0.5, 1.5])
        tel = self.alg._compute_minibatch_telemetry(
            teacher_ratio, student_ratio, entropy)
        self.assertAlmostEqual(tel["clip_frac"], 0.5)
        self.assertAlmostEqual(tel["entropy"], 1.0)

    def test_kl_capture_matches_base_formula(self):
        sigma = torch.full((4, NUM_ACTIONS), 1.2)
        old_sigma = torch.ones(4, NUM_ACTIONS)
        mu = torch.full((4, NUM_ACTIONS), 0.3)
        old_mu = torch.zeros(4, NUM_ACTIONS)
        self.alg._adjust_learning_rate(sigma, old_sigma, mu, old_mu)
        # PPO._adjust_learning_rate's estimator: sum over the action dim of
        # log(sigma/old + 1e-5) + (old_sigma^2 + (old_mu - mu)^2)/(2 sigma^2) - 0.5
        per_dim = math.log(1.2 + 1e-5) + (1.0 + 0.09) / (2.0 * 1.44) - 0.5
        self.assertAlmostEqual(self.alg.last_kl_mean, NUM_ACTIONS * per_dim, places=5)

    def test_kl_capture_preserves_adaptive_lr_update(self):
        # The override must still run the base's adaptive LR adjustment.
        self.alg.schedule = "adaptive"
        self.alg.desired_kl = 0.01
        lr_before = self.alg.learning_rate
        sigma = torch.full((4, NUM_ACTIONS), 0.05)
        old_sigma = torch.ones(4, NUM_ACTIONS)
        mu = torch.zeros(4, NUM_ACTIONS)
        old_mu = torch.zeros(4, NUM_ACTIONS)
        self.alg._adjust_learning_rate(sigma, old_sigma, mu, old_mu)
        # huge KL -> decay branch of PPO._adjust_learning_rate
        self.assertGreater(self.alg.last_kl_mean, 2.0 * self.alg.desired_kl)
        self.assertAlmostEqual(self.alg.learning_rate, max(1e-5, lr_before / 1.5))
        for group in self.alg.optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], self.alg.learning_rate)

    def test_kl_none_when_desired_kl_unset(self):
        self.alg.desired_kl = None
        sigma = torch.ones(2, NUM_ACTIONS)
        mu = torch.zeros(2, NUM_ACTIONS)
        self.alg._adjust_learning_rate(sigma, sigma.clone(), mu, mu.clone())
        self.assertIsNone(self.alg.last_kl_mean)


class TestEpisodeTelemetryExtras(unittest.TestCase):
    """Group-2 env extras: key names and per-episode means."""

    def _make_fake_env(self, curriculum_active=True):
        env = WtyCurriculumMixin.__new__(WtyCurriculumMixin)
        env.extras = {"episode": {}}
        env.zero_command_proba = 0.3
        env.command_ranges = {"lin_vel_x": [-1.0, 2.5], "ang_vel_yaw": [-0.5, 1.5]}
        env.reward_curriculum_scales = {"lin_vel_z": 0.7, "correct_base_height": 0.2}
        env._wty_curriculum_active = curriculum_active
        # per-env accumulators for 4 envs; envs 1 and 3 finish (env 2 is
        # zeroed so untouched-buffer checks can use env 0 only)
        env._tele_lin_vel_err_sum = torch.tensor([2.0, 4.0, 0.0, 8.0])
        env._tele_ang_vel_err_sum = torch.tensor([1.0, 2.0, 0.0, 4.0])
        env._tele_torque_sat_sum = torch.tensor([0.5, 1.0, 0.0, 2.0])
        env._tele_torque_sat_max = torch.tensor([0.9, 0.8, 0.0, 0.95])
        env._tele_terrain_promotions = 3
        env._tele_terrain_demotions = 1
        env.simulator = SimpleNamespace(terrain_levels=torch.tensor([0, 2, 3, 5]))
        return env

    def test_key_names_and_values(self):
        env = self._make_fake_env()
        env_ids = torch.tensor([1, 3])
        env._write_episode_telemetry(env_ids, torch.tensor([2.0, 4.0]))
        ep = env.extras["episode"]
        expected_keys = {"zero_command_proba", "cmd_range_lin_vel_x_max",
                         "cmd_range_ang_vel_yaw_max", "rewscale_lin_vel_z",
                         "rewscale_correct_base_height", "tracking_lin_vel_err",
                         "tracking_ang_vel_err", "torque_sat_mean", "torque_sat_max",
                         "terrain_level_max", "terrain_promotions", "terrain_demotions"}
        self.assertTrue(expected_keys <= set(ep.keys()))
        # instantaneous curriculum state
        self.assertAlmostEqual(ep["zero_command_proba"], 0.3)
        self.assertAlmostEqual(ep["cmd_range_lin_vel_x_max"], 2.5)
        self.assertAlmostEqual(ep["cmd_range_ang_vel_yaw_max"], 1.5)
        self.assertAlmostEqual(ep["rewscale_lin_vel_z"], 0.7)
        self.assertAlmostEqual(ep["rewscale_correct_base_height"], 0.2)
        # per-episode means over the true episode lengths, then the batch mean:
        # lin err env1 4/2 = 2, env3 8/4 = 2 -> 2.0
        self.assertAlmostEqual(float(ep["tracking_lin_vel_err"]), 2.0)
        self.assertAlmostEqual(float(ep["tracking_ang_vel_err"]), 1.0)
        self.assertAlmostEqual(float(ep["torque_sat_mean"]), 0.5)
        # worst-joint running max averaged over the batch: (0.8 + 0.95) / 2
        self.assertAlmostEqual(float(ep["torque_sat_max"]), 0.875)
        # terrain distribution / counters
        self.assertAlmostEqual(float(ep["terrain_level_max"]), 5.0)
        self.assertAlmostEqual(ep["terrain_promotions"], 3.0)
        self.assertAlmostEqual(ep["terrain_demotions"], 1.0)

    def test_counters_and_accumulators_reset_after_emit(self):
        env = self._make_fake_env()
        env_ids = torch.tensor([1, 3])
        env._write_episode_telemetry(env_ids, torch.tensor([2.0, 4.0]))
        self.assertEqual(env._tele_terrain_promotions, 0)
        self.assertEqual(env._tele_terrain_demotions, 0)
        for buf in (env._tele_lin_vel_err_sum, env._tele_ang_vel_err_sum,
                    env._tele_torque_sat_sum, env._tele_torque_sat_max):
            self.assertTrue(torch.all(buf[env_ids] == 0.0))
        # env 0 did not reset: its accumulator is untouched
        self.assertEqual(float(env._tele_lin_vel_err_sum[0]), 2.0)

    def test_terrain_level_max_skipped_when_curriculum_inactive(self):
        env = self._make_fake_env(curriculum_active=False)
        env._write_episode_telemetry(torch.tensor([0]), torch.tensor([3.0]))
        self.assertNotIn("terrain_level_max", env.extras["episode"])
        # the rest of the telemetry is still emitted
        self.assertIn("tracking_lin_vel_err", env.extras["episode"])
        self.assertIn("terrain_promotions", env.extras["episode"])


class TestRunnerTelemetryLogging(unittest.TestCase):
    """MoECTSRunner: channel mapping of the new stats + dedup proxy."""

    def _make_runner(self):
        from rsl_rl.runners.moe_cts_runner import MoECTSRunner
        written = {}
        writer = SimpleNamespace(
            add_scalar=lambda tag, val, it: written.setdefault(tag, val))
        runner = MoECTSRunner.__new__(MoECTSRunner)
        runner.writer = writer
        return runner, written

    def test_moe_stats_channel_mapping(self):
        runner, written = self._make_runner()
        stats = {"latent_mse": 0.5, "entropy": 1.1, "kl_mean": 0.02,
                 "clip_frac": 0.05, "explained_variance": 0.6,
                 "gating_entropy": 2.0, "expert_usage": [0.5, 0.5]}
        runner._log_moe_stats(stats, it=7)
        self.assertEqual(written["Policy/entropy"], 1.1)
        self.assertEqual(written["Policy/kl_mean"], 0.02)
        self.assertEqual(written["Policy/clip_frac"], 0.05)
        self.assertEqual(written["Loss/explained_variance"], 0.6)
        self.assertEqual(written["Loss/latent_mse"], 0.5)
        self.assertEqual(written["MoE/gating_entropy"], 2.0)

    def test_diagnostic_channel_mapping(self):
        """Per-role / gradient / advantage-scale / distillation channels."""
        runner, written = self._make_runner()
        stats = {
            "entropy_teacher": 1.0, "entropy_student": 0.9,
            "kl_teacher": 0.01, "kl_student": 0.03,
            "clip_frac_teacher": 0.04, "clip_frac_student": 0.09,
            "ratio_dev_max": 0.8,
            "value_loss_teacher": 0.2, "value_loss_student": 0.7,
            "explained_variance_teacher": 0.8, "explained_variance_student": 0.1,
            "grad_norm": 3.2, "encoder_grad_norm": 0.4,
            "adv_std_teacher": 1.5, "adv_std_student": 4.5, "adv_std_ratio": 3.0,
            "adv_mean_teacher": 0.1, "adv_mean_student": -0.2, "adv_std_global": 3.0,
            "latent_cosine": 0.95, "latent_ev": 0.88, "action_gap_mse": 0.03,
            "gating_max_weight": 0.9, "experts_alive": 2.0,
        }
        runner._log_moe_stats(stats, it=7)
        for key in ("entropy_teacher", "entropy_student", "kl_teacher", "kl_student",
                    "clip_frac_teacher", "clip_frac_student", "ratio_dev_max"):
            self.assertEqual(written["Policy/" + key], stats[key])
        for key in ("value_loss_teacher", "value_loss_student", "grad_norm",
                    "encoder_grad_norm", "explained_variance_teacher",
                    "explained_variance_student"):
            self.assertEqual(written["Loss/" + key], stats[key])
        for key in ("adv_std_teacher", "adv_std_student", "adv_std_ratio",
                    "adv_mean_teacher", "adv_mean_student", "adv_std_global"):
            self.assertEqual(written["Train/" + key], stats[key])
        for key in ("latent_cosine", "latent_ev", "action_gap_mse",
                    "gating_max_weight", "experts_alive"):
            self.assertEqual(written["MoE/" + key], stats[key])

    def test_none_stats_are_skipped(self):
        runner, written = self._make_runner()
        runner._log_moe_stats({"kl_mean": None, "explained_variance": None}, it=3)
        self.assertNotIn("Policy/kl_mean", written)
        self.assertNotIn("Loss/explained_variance", written)

    def test_drop_scalar_tags_filters_reconstruction_only(self):
        from rsl_rl.runners.moe_cts_runner import _DropScalarTags
        written = []
        base = SimpleNamespace(
            add_scalar=lambda tag, val, it: written.append(tag), other_attr=42)
        proxy = _DropScalarTags(base, {"Loss/reconstruction"})
        proxy.add_scalar("Loss/reconstruction", 1.0, 0)
        proxy.add_scalar("Loss/latent_mse", 1.0, 0)
        self.assertEqual(written, ["Loss/latent_mse"])
        # attribute passthrough keeps the rest of the writer API reachable
        self.assertEqual(proxy.other_attr, 42)


if __name__ == "__main__":
    unittest.main()
