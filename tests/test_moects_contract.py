"""Contract tests for the go2_rl_gym MoE-CTS port (go2_moects / go2_moects_him).

CPU-only: they exercise the MoE modules, the PPO_MOE_CTS training step, the
moe_grid terrain builder and the config/registry wiring WITHOUT building a
simulator env (no GPU / genesis runtime). Runtime smoke tests (training /
eval) live elsewhere.

Run:  .venv/bin/python -m unittest tests/test_moects_contract.py -v
(or:  .venv/bin/python -m pytest tests/test_moects_contract.py -q)
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import math
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.algorithms import PPO_MOE_CTS

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


def _make_dense_ac():
    torch.manual_seed(0)
    return ActorCriticMoECTS(
        NUM_OBS, NUM_ACTIONS, NUM_PRIVILEGED, NUM_HISTORY, NUM_LATENT, NUM_CRITIC,
        student_encoder_type="dense", student_encoder_hidden_dims=[1024, 810],
        norm_type="l2norm", init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        privilege_encoder_hidden_dims=[512, 256],
    )


# Mirror of the storage track's CriticMiniBatch namedtuple (pinned contract):
# (critic_observations, observation_histories, target_values, returns,
#  advantages, old_actions_log_prob, old_mu, old_sigma)
CriticMiniBatch = namedtuple("CriticMiniBatch", [
    "critic_observations", "observation_histories", "target_values", "returns",
    "advantages", "old_actions_log_prob", "old_mu", "old_sigma",
])


class _FakeMoECTSStorage:
    """Minimal stand-in for RolloutStorageMoECTS's pinned generator contract.

    Yields (teacher_batch, student_batch, teacher_critic, student_critic) with
    the per-role field order of RolloutStorageCTS's flat yield:
      teacher_batch = (obs, privileged_obs, actions, old_actions_log_prob,
                       advantages, old_mu, old_sigma)
      student_batch = (obs, privileged_obs, obs_histories, actions,
                       old_actions_log_prob, advantages)
    Deterministic: every generator call replays the same synthetic data.
    """

    def __init__(self, num_envs, num_teacher, num_transitions_per_env):
        self.num_envs = num_envs
        self.num_teacher = num_teacher
        self.num_transitions_per_env = num_transitions_per_env
        self.added = 0
        self.cleared = False
        self.last_bootstrap_values = None

    def add_transitions(self, transition):
        self.added += 1

    def compute_returns(self, last_values, gamma, lam):
        self.last_bootstrap_values = last_values.detach().clone()

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


class TestActorCriticMoECTS(unittest.TestCase):
    def setUp(self):
        self.ac = _make_ac()

    def test_network_input_dims(self):
        # actor: obs(45) + latent(32) = 77
        self.assertEqual(self.ac.actor[0].in_features, NUM_OBS + NUM_LATENT)
        # critic: teacher latent(32) + critic_obs(263) = 295 (paper layout)
        self.assertEqual(self.ac.critic.network[0].in_features, NUM_CRITIC + NUM_LATENT)
        # teacher encoder: 263 -> 32 with L2Norm tail
        self.assertEqual(self.ac.privilege_encoder[0][-1].out_features, NUM_LATENT)
        # student MoE encoder: 225 -> 8 experts -> 32
        self.assertEqual(self.ac.history_encoder.moe.experts.experts.groups, EXPERT_NUM)

    def test_gating_weights_softmax_and_exposure(self):
        latent, weights = self.ac.history_encoder.forward_with_weights(
            torch.randn(7, NUM_HISTORY))
        self.assertEqual(tuple(latent.shape), (7, NUM_LATENT))
        self.assertEqual(tuple(weights.shape), (7, EXPERT_NUM))
        self.assertTrue(torch.allclose(
            weights.sum(dim=-1), torch.ones(7), atol=1e-5))
        # L2-normalized latent
        self.assertTrue(torch.allclose(
            latent.norm(dim=-1), torch.ones(7), atol=1e-4))

    def test_act_contracts(self):
        B = 6
        obs, hist, priv = (torch.randn(B, NUM_OBS), torch.randn(B, NUM_HISTORY),
                           torch.randn(B, NUM_PRIVILEGED))
        a_t = self.ac.act(obs, None, priv, act_type="teacher")
        a_s = self.ac.act(obs, hist, None, act_type="student")
        self.assertEqual(tuple(a_t.shape), (B, NUM_ACTIONS))
        self.assertEqual(tuple(a_s.shape), (B, NUM_ACTIONS))
        mean = self.ac.act_student(obs, hist)  # deploy path (CTSRunner.get_inference_policy)
        self.assertEqual(tuple(mean.shape), (B, NUM_ACTIONS))
        value = self.ac.evaluate(priv)
        self.assertEqual(tuple(value.shape), (B, 1))

    def test_student_encoder_blocked_from_rl_grad(self):
        # Paper: the student encoder is trained only by the distillation loss;
        # the RL pass must not put gradients into it.
        obs, hist = torch.randn(4, NUM_OBS), torch.randn(4, NUM_HISTORY)
        actions = self.ac.act(obs, hist, None, act_type="student")
        self.ac.get_actions_log_prob(actions).sum().backward()
        for p in self.ac.history_encoder.parameters():
            self.assertIsNone(p.grad)
        self.assertIsNotNone(self.ac.actor[0].weight.grad)


class TestActorCriticDenseCTS(unittest.TestCase):
    def test_parameter_matched_encoder_contract(self):
        moe = _make_ac().history_encoder
        dense = _make_dense_ac().history_encoder
        moe_params = sum(p.numel() for p in moe.parameters())
        dense_params = sum(p.numel() for p in dense.parameters())
        self.assertEqual(moe_params, 1_088_264)
        self.assertEqual(dense_params, 1_087_626)
        self.assertLess(abs(moe_params - dense_params) / moe_params, 0.001)

        latent, weights = dense.forward_with_weights(torch.randn(7, NUM_HISTORY))
        self.assertEqual(tuple(latent.shape), (7, NUM_LATENT))
        self.assertEqual(tuple(weights.shape), (7, 1))
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))
        self.assertTrue(torch.allclose(
            latent.norm(dim=-1), torch.ones(7), atol=1e-4))

    def test_dense_encoder_loss_has_zero_load_balance(self):
        ac = _make_dense_ac()
        alg = PPO_MOE_CTS(ac, device="cpu", num_teacher=6,
                          num_learning_epochs=1, num_mini_batches=2,
                          load_balance_coef=0.01)
        total, latent, balance, weights, _, _ = alg._compute_encoder_losses(
            torch.randn(8, NUM_HISTORY), torch.randn(8, NUM_PRIVILEGED))
        self.assertTrue(torch.allclose(total, latent))
        self.assertEqual(float(balance), 0.0)
        self.assertEqual(tuple(weights.shape), (8, 1))


class TestPPOMoECTS(unittest.TestCase):
    def setUp(self):
        self.ac = _make_ac()
        self.num_envs, self.num_teacher = 8, 6
        self.alg = PPO_MOE_CTS(self.ac, device="cpu", num_teacher=self.num_teacher,
                               num_learning_epochs=1, num_mini_batches=2,
                               load_balance_coef=0.01)
        # PPO_MOE_CTS consumes the RolloutStorageMoECTS generator contract
        # (storage track); inject the pinned fake here.
        self.alg.storage = _FakeMoECTSStorage(self.num_envs, self.num_teacher, 4)

    def _rollout(self):
        for _ in range(4):
            obs, priv, hist, critic = (torch.randn(self.num_envs, NUM_OBS),
                                       torch.randn(self.num_envs, NUM_PRIVILEGED),
                                       torch.randn(self.num_envs, NUM_HISTORY),
                                       torch.randn(self.num_envs, NUM_CRITIC))
            actions = self.alg.act(obs, priv, hist, critic)
            self.assertEqual(tuple(actions.shape), (self.num_envs, NUM_ACTIONS))
            self.alg.process_env_step(torch.randn(self.num_envs),
                                      torch.zeros(self.num_envs), {})
        self.alg.compute_returns(torch.randn(self.num_envs, NUM_CRITIC),
                                 torch.randn(self.num_envs, NUM_HISTORY))

    def test_update_returns_5tuple(self):
        # New contract: (value, teacher_surrogate, student_surrogate,
        # latent_mse, moe_stats); the runner unpacks five values.
        self._rollout()
        out = self.alg.update()
        self.assertEqual(len(out), 5)
        for v in out[:4]:
            self.assertTrue(torch.isfinite(torch.tensor(float(v))))
        self.assertIsInstance(out[4], dict)

    def test_encoder_loss_includes_load_balance(self):
        coef = 1.7  # exaggerated so the term is distinguishable
        alg = PPO_MOE_CTS(self.ac, device="cpu", num_teacher=2, load_balance_coef=coef)
        hist_b, priv_b = torch.randn(6, NUM_HISTORY), torch.randn(6, NUM_PRIVILEGED)
        loss = alg._compute_encoder_loss(hist_b, priv_b)

        pred, weights = self.ac.history_encoder.forward_with_weights(hist_b)
        target = self.ac.privilege_encoder(priv_b).detach()
        expected = F.mse_loss(pred, target) + coef * torch.mean(
            (weights.mean(dim=0) - 1.0 / EXPERT_NUM).pow(2))
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))


class TestMoECTSRoleAwareEvaluate(unittest.TestCase):
    """Reference parity: evaluate(privileged_obs, history, is_teacher)."""

    def setUp(self):
        self.ac = _make_ac()

    def test_teacher_path_ignores_history_student_path_uses_it(self):
        critic_obs = torch.randn(5, NUM_CRITIC)
        h1, h2 = torch.randn(5, NUM_HISTORY), torch.randn(5, NUM_HISTORY)
        v_t1 = self.ac.evaluate(critic_obs, h1, is_teacher=True)
        v_t2 = self.ac.evaluate(critic_obs, h2, is_teacher=True)
        self.assertTrue(torch.allclose(v_t1, v_t2))  # teacher latent from critic_obs only
        v_s1 = self.ac.evaluate(critic_obs, h1, is_teacher=False)
        v_s2 = self.ac.evaluate(critic_obs, h2, is_teacher=False)
        self.assertFalse(torch.allclose(v_s1, v_s2))  # student latent follows the history
        # student value == critic([detached MoE history latent, critic_obs])
        with torch.no_grad():
            latent = self.ac.history_encoder(h1)
            manual = self.ac.critic(torch.cat((latent, critic_obs), dim=-1))
        self.assertTrue(torch.allclose(v_s1, manual, atol=1e-6))

    def test_student_path_requires_history_and_single_arg_compat(self):
        with self.assertRaises(ValueError):
            self.ac.evaluate(torch.randn(3, NUM_CRITIC), is_teacher=False)
        # single-arg evaluate(critic_observations) keeps working (teacher default)
        v = self.ac.evaluate(torch.randn(3, NUM_CRITIC))
        self.assertEqual(tuple(v.shape), (3, 1))

    def test_critic_loss_detached_from_both_encoders(self):
        critic_obs = torch.randn(4, NUM_CRITIC)
        # teacher role: no critic-loss gradient into the privilege encoder
        self.ac.evaluate(critic_obs).sum().backward()
        for p in self.ac.privilege_encoder.parameters():
            self.assertIsNone(p.grad)
        self.assertIsNotNone(self.ac.critic.network[0].weight.grad)
        self.ac.zero_grad()
        # student role: no critic-loss gradient into either encoder
        self.ac.evaluate(critic_obs, torch.randn(4, NUM_HISTORY),
                         is_teacher=False).sum().backward()
        for p in self.ac.history_encoder.parameters():
            self.assertIsNone(p.grad)
        for p in self.ac.privilege_encoder.parameters():
            self.assertIsNone(p.grad)
        self.assertIsNotNone(self.ac.critic.network[0].weight.grad)


class TestMoECTSGradientFlow(unittest.TestCase):
    """Optimizer partition and per-loss gradient routing."""

    def setUp(self):
        self.ac = _make_ac()
        self.alg = PPO_MOE_CTS(self.ac, device="cpu", num_teacher=6,
                               num_learning_epochs=1, num_mini_batches=1,
                               load_balance_coef=0.01)
        self.storage = _FakeMoECTSStorage(8, 6, 4)
        self.alg.storage = self.storage

    def test_optimizer_partition(self):
        rl_ids = {id(p) for g in self.alg.optimizer.param_groups for p in g["params"]}
        enc_ids = {id(p) for g in self.alg.history_encoder_optimizer.param_groups
                   for p in g["params"]}
        he_ids = {id(p) for p in self.ac.history_encoder.parameters()}
        priv_ids = {id(p) for p in self.ac.privilege_encoder.parameters()}
        # student MoE encoder is trained ONLY by the encoder optimizer
        self.assertTrue(he_ids.isdisjoint(rl_ids))
        self.assertEqual(enc_ids, he_ids)
        # reference optimizer1 == {teacher_encoder, critic, actor, std}
        self.assertTrue(priv_ids <= rl_ids)
        self.assertTrue({id(p) for p in self.ac.actor.parameters()} <= rl_ids)
        self.assertTrue({id(p) for p in self.ac.critic.parameters()} <= rl_ids)
        self.assertIn(id(self.ac.std), rl_ids)

    def test_rl_loss_backward_isolates_student_encoder(self):
        tb, sb, tc, sc = next(self.storage.mini_batch_generator(1, 1))
        loss, teacher_surr, student_surr, value_loss = self.alg._compute_rl_loss(tb, sb, tc, sc)
        for v in (loss, teacher_surr, student_surr, value_loss):
            self.assertTrue(torch.isfinite(v.detach()))
        loss.backward()
        for p in self.ac.history_encoder.parameters():
            self.assertIsNone(p.grad)
        # grads reach actor/critic and (via the teacher surrogate arm) the
        # privilege encoder, matching the reference optimizer1 graph
        self.assertIsNotNone(self.ac.actor[0].weight.grad)
        self.assertIsNotNone(self.ac.critic.network[0].weight.grad)
        self.assertTrue(any(p.grad is not None
                            for p in self.ac.privilege_encoder.parameters()))

    def test_encoder_loss_reaches_only_history_encoder(self):
        total = self.alg._compute_encoder_loss(torch.randn(6, NUM_HISTORY),
                                               torch.randn(6, NUM_PRIVILEGED))
        total.backward()
        for p in self.ac.history_encoder.parameters():
            self.assertIsNotNone(p.grad)
        for p in self.ac.privilege_encoder.parameters():
            self.assertIsNone(p.grad)
        for p in self.ac.actor.parameters():
            self.assertIsNone(p.grad)
        for p in self.ac.critic.parameters():
            self.assertIsNone(p.grad)


class TestMoECTSBootstrap(unittest.TestCase):
    """Role-aware bootstrap in compute_returns and rollout-time values in act.

    Roles are the reference's INTERLEAVED env ids (moe_cts.py:96-102): with
    num_envs=8 / ratio 0.75 the students are {0, 4} and the teachers the rest;
    tensors crossing the storage boundary stay in env order.
    """

    def setUp(self):
        self.ac = _make_ac()
        self.num_envs, self.num_teacher = 8, 6
        self.alg = PPO_MOE_CTS(self.ac, device="cpu", num_teacher=self.num_teacher)
        self.storage = _FakeMoECTSStorage(self.num_envs, self.num_teacher, 4)
        self.alg.storage = self.storage
        from rsl_rl.algorithms.ppo_moe_cts import compute_role_env_idxs
        self.teacher_idxs, self.student_idxs = compute_role_env_idxs(
            self.num_envs, self.alg.teacher_env_ratio, "cpu")

    def test_compute_returns_role_aware_calls(self):
        critic_obs = torch.randn(self.num_envs, NUM_CRITIC)
        hist = torch.randn(self.num_envs, NUM_HISTORY)
        with mock.patch.object(self.ac, "evaluate", wraps=self.ac.evaluate) as m:
            self.alg.compute_returns(critic_obs, hist)
        self.assertEqual(len(m.call_args_list), 2)
        t_call, s_call = m.call_args_list
        self.assertIs(t_call.kwargs["is_teacher"], True)
        self.assertEqual(tuple(t_call.args[0].shape), (self.num_teacher, NUM_CRITIC))
        self.assertIs(s_call.kwargs["is_teacher"], False)
        self.assertEqual(tuple(s_call.args[0].shape),
                         (self.num_envs - self.num_teacher, NUM_CRITIC))
        # the role arms receive the interleaved role gathers
        self.assertTrue(torch.equal(t_call.args[0], critic_obs[self.teacher_idxs]))
        self.assertTrue(torch.equal(s_call.args[0], critic_obs[self.student_idxs]))
        self.assertTrue(torch.equal(s_call.args[1], hist[self.student_idxs]))

    def test_compute_returns_bootstrap_values_sentinel(self):
        n_student = self.num_envs - self.num_teacher
        sentinel_t = torch.full((self.num_teacher, 1), 3.0)
        sentinel_s = torch.full((n_student, 1), -2.0)

        def fake_evaluate(critic_observations, obs_history=None, is_teacher=True, **kwargs):
            return sentinel_t if is_teacher else sentinel_s

        with mock.patch.object(self.ac, "evaluate", side_effect=fake_evaluate):
            self.alg.compute_returns(torch.randn(self.num_envs, NUM_CRITIC),
                                     torch.randn(self.num_envs, NUM_HISTORY))
        # last_values cross the storage boundary in env order
        expected = torch.empty(self.num_envs, 1)
        expected[self.teacher_idxs] = sentinel_t
        expected[self.student_idxs] = sentinel_s
        self.assertTrue(torch.allclose(self.storage.last_bootstrap_values, expected))

    def test_compute_returns_requires_history(self):
        with self.assertRaises(TypeError):
            self.alg.compute_returns(torch.randn(self.num_envs, NUM_CRITIC))

    def test_act_stores_role_aware_values(self):
        obs = torch.randn(self.num_envs, NUM_OBS)
        priv = torch.randn(self.num_envs, NUM_PRIVILEGED)
        hist = torch.randn(self.num_envs, NUM_HISTORY)
        critic = torch.randn(self.num_envs, NUM_CRITIC)
        actions = self.alg.act(obs, priv, hist, critic)
        self.assertEqual(tuple(actions.shape), (self.num_envs, NUM_ACTIONS))
        # transition.values are role-aware and in env order
        with torch.no_grad():
            expected = torch.empty(self.num_envs, 1)
            expected[self.teacher_idxs] = self.ac.evaluate(
                critic[self.teacher_idxs], is_teacher=True)
            expected[self.student_idxs] = self.ac.evaluate(
                critic[self.student_idxs], hist[self.student_idxs], is_teacher=False)
        self.assertTrue(torch.allclose(self.alg.transition.values, expected, atol=1e-6))
        # and they differ from the old teacher-latent-for-all-envs evaluation
        self.assertFalse(torch.allclose(self.alg.transition.values,
                                        self.ac.evaluate(critic).detach()))


class TestMoECTSUpdateStats(unittest.TestCase):
    """update() 5-tuple and moe_stats aggregation over ALL minibatches."""

    def setUp(self):
        self.ac = _make_ac()
        self.num_envs, self.num_teacher = 8, 6
        self.alg = PPO_MOE_CTS(self.ac, device="cpu", num_teacher=self.num_teacher,
                               num_learning_epochs=2, num_mini_batches=2,
                               num_encoder_epochs=1, load_balance_coef=0.01)
        self.storage = _FakeMoECTSStorage(self.num_envs, self.num_teacher, 4)
        self.alg.storage = self.storage

    def test_update_5tuple_and_pinned_stats(self):
        out = self.alg.update()
        self.assertEqual(len(out), 5)
        mvl, tsl, ssl, mll, stats = out
        for v in (mvl, tsl, ssl, mll):
            self.assertTrue(torch.isfinite(torch.tensor(float(v))))
        pinned = {"latent_mse", "load_balance", "student_encoder_total",
                  "gating_entropy", "effective_experts", "expert_usage_min",
                  "expert_usage_max", "expert_usage_std", "expert_usage"}
        self.assertTrue(pinned <= set(stats.keys()))
        self.assertEqual(len(stats["expert_usage"]), EXPERT_NUM)
        for k in pinned - {"expert_usage"}:
            self.assertTrue(math.isfinite(stats[k]), k)
        self.assertAlmostEqual(stats["effective_experts"],
                               math.exp(stats["gating_entropy"]))
        self.assertAlmostEqual(sum(stats["expert_usage"]), 1.0, places=5)
        self.assertLessEqual(stats["expert_usage_min"], stats["expert_usage_max"])
        self.assertGreaterEqual(stats["expert_usage_std"], 0.0)
        self.assertAlmostEqual(mll, stats["latent_mse"])
        self.assertAlmostEqual(stats["student_encoder_total"],
                               stats["latent_mse"] + 0.01 * stats["load_balance"],
                               places=6)
        self.assertTrue(self.storage.cleared)

    def test_stats_aggregated_over_all_minibatches_weighted(self):
        records = []
        orig = self.alg._compute_encoder_losses

        def spy(hist_b, priv_b):
            total, latent, lb, w, pred, target = orig(hist_b, priv_b)
            records.append((latent.item(), lb.item(), total.item(),
                            w.detach(), hist_b.shape[0]))
            return total, latent, lb, w, pred, target

        self.alg._compute_encoder_losses = spy
        stats = self.alg.update()[4]
        # every epoch x minibatch (x encoder epoch) computation was aggregated
        self.assertEqual(len(records), 2 * 2 * 1)
        wsum = sum(r[4] for r in records)
        exp_latent = sum(r[0] * r[4] for r in records) / wsum
        exp_lb = sum(r[1] * r[4] for r in records) / wsum
        exp_total = sum(r[2] * r[4] for r in records) / wsum
        exp_entropy = sum((-torch.xlogy(r[3], r[3]).sum(dim=-1).mean().item()) * r[4]
                          for r in records) / wsum
        exp_usage = sum(r[3].mean(dim=0) * r[4] for r in records) / wsum
        self.assertAlmostEqual(stats["latent_mse"], exp_latent, places=6)
        self.assertAlmostEqual(stats["load_balance"], exp_lb, places=6)
        self.assertAlmostEqual(stats["student_encoder_total"], exp_total, places=6)
        self.assertAlmostEqual(stats["gating_entropy"], exp_entropy, places=6)
        for got, exp in zip(stats["expert_usage"], exp_usage.tolist()):
            self.assertAlmostEqual(got, exp, places=6)
        self.assertAlmostEqual(stats["expert_usage_min"], exp_usage.min().item())
        self.assertAlmostEqual(stats["expert_usage_max"], exp_usage.max().item())
        # not just the last minibatch (deterministic data: distinct records)
        self.assertNotAlmostEqual(stats["latent_mse"], records[-1][0], places=4)


class TestMoECTSRealStorageIntegration(unittest.TestCase):
    """End-to-end CPU check against the storage track's RolloutStorageMoECTS:
    init_storage wiring + act/process_env_step/compute_returns/update all flow
    through the real generator contract (no fake)."""

    def test_full_cycle_with_real_storage(self):
        from rsl_rl.storage import RolloutStorageMoECTS

        ac = _make_ac()
        num_envs, num_teacher = 8, 6
        alg = PPO_MOE_CTS(ac, device="cpu", num_teacher=num_teacher,
                          num_learning_epochs=1, num_mini_batches=2,
                          load_balance_coef=0.01)
        alg.init_storage(num_envs, 4, [NUM_OBS], [NUM_PRIVILEGED],
                         [NUM_HISTORY], [NUM_CRITIC], [NUM_ACTIONS])
        self.assertIsInstance(alg.storage, RolloutStorageMoECTS)

        for _ in range(4):
            obs, priv, hist, critic = (torch.randn(num_envs, NUM_OBS),
                                       torch.randn(num_envs, NUM_PRIVILEGED),
                                       torch.randn(num_envs, NUM_HISTORY),
                                       torch.randn(num_envs, NUM_CRITIC))
            actions = alg.act(obs, priv, hist, critic)
            self.assertEqual(tuple(actions.shape), (num_envs, NUM_ACTIONS))
            alg.process_env_step(torch.randn(num_envs), torch.zeros(num_envs), {})
        alg.compute_returns(torch.randn(num_envs, NUM_CRITIC),
                            torch.randn(num_envs, NUM_HISTORY))

        out = alg.update()
        self.assertEqual(len(out), 5)
        for v in out[:4]:
            self.assertTrue(torch.isfinite(torch.tensor(float(v))))
        stats = out[4]
        self.assertEqual(len(stats["expert_usage"]), EXPERT_NUM)
        self.assertTrue(math.isfinite(stats["gating_entropy"]))


class TestMoEGRidTerrain(unittest.TestCase):
    def test_small_grid_layout(self):
        from legged_gym.utils.terrain import Terrain
        cfg = SimpleNamespace(
            mesh_type="heightfield", simplify_mesh=True,
            terrain_length=8.0, terrain_width=8.0, platform_size=3.0,
            terrain_proportions=[0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15],
            num_rows=10, num_cols=6, horizontal_scale=0.1, vertical_scale=0.005,
            border_size=5, moe_grid=True, terrain_spacing=0.5,
        )
        t = Terrain(cfg)
        exp_rows = int(10 * 80 + 9 * 5) + 2 * 50
        exp_cols = int(6 * 80 + 5 * 5) + 2 * 50
        self.assertEqual(t.height_field_raw.shape, (exp_rows, exp_cols))
        self.assertEqual(t.env_origins.shape, (10, 6, 3))
        # semantic ids for this proportions table (6 columns)
        self.assertEqual(t.cols2id, [0, 1, 3, 3, 5, 5])
        # 0.5 m inter-tile gap bands stay at zero height
        gap_band = t.height_field_raw[50:850, 50 + 80:50 + 85]
        self.assertTrue(bool((gap_band == 0).all()))
        # origins are spaced by tile length + spacing
        dx = t.env_origins[1, 0, 0] - t.env_origins[0, 0, 0]
        self.assertAlmostEqual(float(dx), 8.5)

    def test_task_cfg_grid(self):
        import legged_gym.envs  # noqa: F401  (registers tasks)
        from legged_gym.utils import task_registry
        from legged_gym.utils.terrain import Terrain

        env_cfg, _ = task_registry.get_cfgs("go2_moects")
        t = Terrain(env_cfg.terrain)  # 10 levels x 20 type columns, border 25m
        self.assertEqual(t.env_origins.shape, (10, 20, 3))
        self.assertEqual(
            t.cols2id,
            [0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 8, 8, 8])
        self.assertEqual(set(t.name2cols), {"wave", "slope", "rough_slope",
                                            "stairs_up", "stairs_down",
                                            "obstacles", "flat"})


class TestMoECTSRegistryAndConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import legged_gym.envs  # noqa: F401  (registers tasks)
        from legged_gym.utils import task_registry
        cls.env_cfg, cls.train_cfg = task_registry.get_cfgs("go2_moects")
        cls.env_cfg_dense, cls.train_cfg_dense = task_registry.get_cfgs("go2_dense_cts")
        cls.env_cfg_him, cls.train_cfg_him = task_registry.get_cfgs("go2_moects_him")

    def test_cts_arm_wiring(self):
        env, train = self.env_cfg.env, self.train_cfg
        self.assertEqual(env.num_observations, NUM_OBS)
        self.assertEqual(env.num_privileged_obs, NUM_PRIVILEGED)
        self.assertEqual(env.num_history_obs, NUM_OBS * env.frame_stack)
        self.assertEqual(env.num_latent_dims, NUM_LATENT)
        self.assertEqual(env.num_critic_obs, env.num_privileged_obs)
        self.assertEqual(env.num_teacher, int(env.num_envs * 0.75))
        self.assertEqual(env.teacher_env_ratio, 0.75)
        self.assertEqual(train.runner_class_name, "MoECTSRunner")  # MoE arm runner (5-tuple update + moe_stats logging)
        self.assertEqual(train.runner.policy_class_name, "ActorCriticMoECTS")
        self.assertEqual(train.runner.algorithm_class_name, "PPO_MOE_CTS")
        # iteration-based curricula assume the runner's rollout length
        self.assertEqual(env.wty_steps_per_iteration, train.runner.num_steps_per_env)

    def test_cts_arm_terrain_cfg(self):
        terr = self.env_cfg.terrain
        self.assertTrue(terr.moe_grid)
        self.assertFalse(terr.curriculum)  # the mixin owns the game curriculum
        self.assertEqual(terr.mesh_type, "heightfield")
        self.assertAlmostEqual(terr.terrain_spacing, 0.5)
        self.assertTrue(terr.measure_heights)
        self.assertEqual(len(terr.measured_points_x) * len(terr.measured_points_y), 187)

    def test_dense_arm_changes_only_student_encoder_and_run_identity(self):
        from legged_gym.envs.go2.go2_moects.go2_moects_config import (
            Go2MoECTSCfg, Go2DenseCTSCfg)
        from legged_gym.utils.helpers import class_to_dict
        moe = class_to_dict(self.train_cfg)
        dense = class_to_dict(self.train_cfg_dense)
        # Use fresh configs: an earlier terrain-builder test deliberately adds
        # derived fields to the registry's singleton go2_moects terrain cfg.
        self.assertEqual(class_to_dict(Go2MoECTSCfg()),
                         class_to_dict(Go2DenseCTSCfg()))
        self.assertEqual(dense["policy"]["student_encoder_type"], "dense")
        self.assertEqual(dense["policy"]["student_encoder_hidden_dims"], [1024, 810])
        self.assertEqual(dense["runner"]["experiment_name"], "go2_dense_cts")
        self.assertEqual(dense["runner"]["eval_interval"], 0)
        self.assertEqual(dense["runner"]["eval_num_envs"], 0)
        self.assertEqual(dense["runner"]["terrain_gate_log_interval"], 0)
        self.assertEqual(moe["algorithm"], dense["algorithm"])
        for key, value in moe["policy"].items():
            if key not in {"student_encoder_hidden_dims", "student_encoder_type"}:
                self.assertEqual(value, dense["policy"][key])

    def test_eval_resolution_in_cts_runner_namespace(self):
        # CTSRunner._init_agent_and_algo resolves class names with eval() in
        # its module globals; both MoE classes must be importable there.
        import rsl_rl.runners.cts_runner as cts_runner_mod
        self.assertIs(eval("ActorCriticMoECTS", vars(cts_runner_mod)), ActorCriticMoECTS)
        self.assertIs(eval("PPO_MOE_CTS", vars(cts_runner_mod)), PPO_MOE_CTS)
        from rsl_rl.utils.runner_registry import runner_registry
        self.assertIs(runner_registry.get_runner_class("CTSRunner"),
                      cts_runner_mod.CTSRunner)

    def test_policy_alg_cfg_roundtrip(self):
        # Construct exactly like CTSRunner._init_agent_and_algo does.
        from legged_gym.utils.helpers import class_to_dict
        train_cfg_dict = class_to_dict(self.train_cfg)
        ac = ActorCriticMoECTS(NUM_OBS, NUM_ACTIONS, NUM_PRIVILEGED, NUM_HISTORY,
                               NUM_LATENT, NUM_CRITIC, **train_cfg_dict["policy"])
        alg = PPO_MOE_CTS(ac, device="cpu", num_teacher=6,
                          **train_cfg_dict["algorithm"])
        self.assertAlmostEqual(alg.load_balance_coef, 0.01)
        self.assertAlmostEqual(alg.encoder_lr, 1e-3)

    def test_him_arm_wiring(self):
        env, train = self.env_cfg_him.env, self.train_cfg_him
        self.assertEqual(env.num_observations, env.frame_stack * env.num_one_step_obs)
        self.assertEqual(env.num_observations, 270)
        self.assertEqual(env.num_privileged_obs, 265)
        self.assertEqual(train.runner_class_name, "HIMRunner")
        self.assertEqual(train.runner.policy_class_name, "HIMActorCritic")
        self.assertEqual(train.runner.algorithm_class_name, "PPO_HIM")
        # shared substrate: identical terrain/curriculum flags on both arms
        self.assertEqual(env.wty_steps_per_iteration, train.runner.num_steps_per_env)
        for field in ("moe_grid", "terrain_spacing", "num_rows", "num_cols",
                      "terrain_proportions", "measure_heights"):
            self.assertEqual(getattr(self.env_cfg.terrain, field),
                             getattr(self.env_cfg_him.terrain, field), field)


class TestPPOMoECTSKLConcat(unittest.TestCase):
    """Adaptive-LR KL is computed over the CONCATENATED teacher+student
    batch (reference go2_rl_gym moe_cts.py:131-149 parity) -- the host
    PPO_CTS computes it over the teacher arm only. Pinned here so the MoE
    arm never silently falls back to teacher-only KL."""

    def setUp(self):
        self.ac = _make_ac()
        self.num_envs, self.num_teacher, self.steps = 8, 6, 4
        self.alg = PPO_MOE_CTS(self.ac, device="cpu", num_teacher=self.num_teacher)
        self.alg.schedule = "adaptive"
        self.alg.desired_kl = 0.01

    def test_kl_inputs_are_concatenated_teacher_student(self):
        alg = self.alg
        alg.storage = _FakeMoECTSStorage(self.num_envs, self.num_teacher, self.steps)
        t_n = self.num_teacher * self.steps // alg.num_mini_batches
        s_n = (self.num_envs - self.num_teacher) * self.steps // alg.num_mini_batches

        calls = []
        alg._adjust_learning_rate = lambda *args: calls.append(args)
        alg.update()

        # one KL evaluation per RL minibatch per epoch
        self.assertEqual(len(calls),
                         alg.num_learning_epochs * alg.num_mini_batches)
        for sigma, old_sigma, mu, old_mu in calls:
            for tensor in (sigma, old_sigma, mu, old_mu):
                self.assertEqual(tensor.shape[0], t_n + s_n,
                                 "KL batch must span teacher+student samples")
                self.assertEqual(tensor.shape[1], NUM_ACTIONS)
            # sigma/old_sigma are std devs: positivity distinguishes them
            # from the mu tensors and confirms argument order
            self.assertTrue(torch.all(sigma > 0))
            self.assertTrue(torch.all(old_sigma > 0))


if __name__ == "__main__":
    unittest.main()
