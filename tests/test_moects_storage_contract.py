"""Contract tests for RolloutStorageMoECTS (go2_moects MoE-CTS port).

CPU-only, pure torch: no simulator env is built. They pin the fixed
minibatch contract shared with the algorithm/runner ports:

    (teacher_batch, student_batch, teacher_critic, student_critic)

with role-pure, sample-aligned CriticMiniBatch batches, and role tuples that
are field-for-field identical to the host RolloutStorageCTS output.

Run:  .venv/bin/python -m unittest tests.test_moects_storage_contract -v
(or:  .venv/bin/python -m unittest tests/test_moects_storage_contract.py -v)
"""

import unittest

import torch

from rsl_rl.storage import CriticMiniBatch, RolloutStorageCTS, RolloutStorageMoECTS

NUM_ENVS = 8
NUM_TEACHER = 4                      # teacher envs: indices [0, 4)
NUM_STUDENT = NUM_ENVS - NUM_TEACHER
T = 6                                # num_transitions_per_env
NUM_OBS = 12
NUM_PRIVILEGED = 16
HISTORY_LENGTH = 5
NUM_HISTORY = HISTORY_LENGTH * NUM_OBS   # 60
NUM_CRITIC = 20
NUM_ACTIONS = 4

OBS_SHAPE = (NUM_OBS,)
PRIV_SHAPE = (NUM_PRIVILEGED,)
HIST_SHAPE = (NUM_HISTORY,)
CRITIC_SHAPE = (NUM_CRITIC,)
ACTION_SHAPE = (NUM_ACTIONS,)


def _tid(env, step):
    """Unique transition id; doubles as role bit: tid < NUM_TEACHER*T <=> teacher."""
    return float(env * T + step)


def _make_storage(storage_cls):
    return storage_cls(
        NUM_ENVS, NUM_TEACHER, T, OBS_SHAPE, PRIV_SHAPE, HIST_SHAPE,
        CRITIC_SHAPE, ACTION_SHAPE, device="cpu",
    )


def _fill_with_sentinels(storage):
    """Fill every stored field with the transition id (sentinel).

    compute_returns is intentionally NOT called here: it rewrites returns /
    teacher_advantages / student_advantages from rewards+values (GAE), which
    would destroy the sentinels. Its semantics are inherited unchanged from
    RolloutStorageCTS and covered by a separate smoke test below.
    """
    for step in range(T):
        tr = storage.Transition()
        rows = torch.tensor([_tid(e, step) for e in range(NUM_ENVS)])
        tr.observations = rows.unsqueeze(1).expand(NUM_ENVS, NUM_OBS).clone()
        tr.privileged_observations = rows.unsqueeze(1).expand(NUM_ENVS, NUM_PRIVILEGED).clone()
        tr.observation_histories = rows.unsqueeze(1).expand(NUM_ENVS, NUM_HISTORY).clone()
        tr.critic_observations = rows.unsqueeze(1).expand(NUM_ENVS, NUM_CRITIC).clone()
        tr.actions = rows.unsqueeze(1).expand(NUM_ENVS, NUM_ACTIONS).clone()
        tr.rewards = torch.zeros(NUM_ENVS)
        tr.dones = torch.zeros(NUM_ENVS)
        tr.values = rows.unsqueeze(1).clone()
        tr.actions_log_prob = rows.clone()
        tr.action_mean = rows.unsqueeze(1).expand(NUM_ENVS, NUM_ACTIONS).clone()
        tr.action_sigma = rows.unsqueeze(1).expand(NUM_ENVS, NUM_ACTIONS).clone()
        storage.add_transitions(tr)
    # GAE outputs, sentinel-encoded by hand (compute_returns not run).
    for step in range(T):
        for e in range(NUM_ENVS):
            storage.returns[step, e, 0] = _tid(e, step)
        for e in range(NUM_TEACHER):
            storage.teacher_advantages[step, e, 0] = _tid(e, step)
        for e in range(NUM_STUDENT):
            storage.student_advantages[step, e, 0] = _tid(NUM_TEACHER + e, step)


def _critic_tids(critic):
    """Recover the per-field transition-id channels of a CriticMiniBatch."""
    return {
        "critic_observations": critic.critic_observations[:, 0],
        "observation_histories": critic.observation_histories[:, 0],
        "target_values": critic.target_values[:, 0],
        "returns": critic.returns[:, 0],
        "advantages": critic.advantages[:, 0],
        "old_actions_log_prob": critic.old_actions_log_prob[:, 0],
        "old_mu": critic.old_mu[:, 0],
        "old_sigma": critic.old_sigma[:, 0],
    }


class TestCriticMiniBatchContract(unittest.TestCase):
    def setUp(self):
        self.storage = _make_storage(RolloutStorageMoECTS)
        _fill_with_sentinels(self.storage)

    def test_yield_structure_and_shapes(self):
        gen = self.storage.mini_batch_generator(num_mini_batches=2, num_epochs=1)
        batches = list(gen)
        self.assertEqual(len(batches), 2)
        teacher_batch, student_batch, teacher_critic, student_critic = batches[0]
        self.assertIsInstance(teacher_batch, tuple)
        self.assertIsInstance(student_batch, tuple)
        self.assertEqual(len(teacher_batch), 7)   # host CTS teacher field count
        self.assertEqual(len(student_batch), 6)   # host CTS student field count
        self.assertIsInstance(teacher_critic, CriticMiniBatch)
        self.assertIsInstance(student_critic, CriticMiniBatch)
        b = NUM_TEACHER * T // 2
        self.assertEqual(teacher_critic.critic_observations.shape, (b, NUM_CRITIC))
        self.assertEqual(teacher_critic.observation_histories.shape, (b, NUM_HISTORY))
        self.assertEqual(teacher_critic.target_values.shape, (b, 1))
        self.assertEqual(teacher_critic.returns.shape, (b, 1))
        self.assertEqual(teacher_critic.advantages.shape, (b, 1))
        self.assertEqual(teacher_critic.old_actions_log_prob.shape, (b, 1))
        self.assertEqual(teacher_critic.old_mu.shape, (b, NUM_ACTIONS))
        self.assertEqual(teacher_critic.old_sigma.shape, (b, NUM_ACTIONS))

    def test_critic_sample_alignment_all_epochs(self):
        # Same index of every CriticMiniBatch field == same transition, for
        # both roles, in every epoch and every minibatch.
        for num_mini_batches, num_epochs in ((2, 3), (4, 2)):
            with self.subTest(num_mini_batches=num_mini_batches, num_epochs=num_epochs):
                gen = self.storage.mini_batch_generator(num_mini_batches, num_epochs)
                for teacher_batch, student_batch, teacher_critic, student_critic in gen:
                    for critic in (teacher_critic, student_critic):
                        tids = _critic_tids(critic)
                        ref = tids["critic_observations"]
                        for name, values in tids.items():
                            self.assertTrue(
                                torch.equal(ref, values),
                                f"misaligned field {name}: {ref} vs {values}",
                            )

    def test_critic_batches_are_role_pure(self):
        gen = self.storage.mini_batch_generator(num_mini_batches=2, num_epochs=3)
        for teacher_batch, student_batch, teacher_critic, student_critic in gen:
            teacher_tids = teacher_critic.critic_observations[:, 0]
            student_tids = student_critic.critic_observations[:, 0]
            self.assertTrue(bool((teacher_tids < NUM_TEACHER * T).all()))
            self.assertTrue(bool((student_tids >= NUM_TEACHER * T).all()))
            self.assertTrue(bool((student_tids < NUM_ENVS * T).all()))
            # teacher_batch / student_batch role purity as well (obs channel)
            self.assertTrue(bool((teacher_batch[0][:, 0] < NUM_TEACHER * T).all()))
            self.assertTrue(bool((student_batch[0][:, 0] >= NUM_TEACHER * T).all()))

    def test_coverage_exactly_once_per_epoch_per_role(self):
        num_mini_batches, num_epochs = 4, 2
        gen = self.storage.mini_batch_generator(num_mini_batches, num_epochs)
        teacher_expected = sorted(float(t) for t in range(0, NUM_TEACHER * T))
        student_expected = sorted(float(t) for t in range(NUM_TEACHER * T, NUM_ENVS * T))
        for epoch in range(num_epochs):
            seen = {k: [] for k in (
                "teacher_batch", "student_batch", "teacher_critic", "student_critic")}
            for _ in range(num_mini_batches):
                teacher_batch, student_batch, teacher_critic, student_critic = next(gen)
                seen["teacher_batch"].extend(teacher_batch[0][:, 0].tolist())
                seen["student_batch"].extend(student_batch[0][:, 0].tolist())
                seen["teacher_critic"].extend(teacher_critic.critic_observations[:, 0].tolist())
                seen["student_critic"].extend(student_critic.critic_observations[:, 0].tolist())
            with self.subTest(epoch=epoch):
                self.assertEqual(sorted(seen["teacher_batch"]), teacher_expected)
                self.assertEqual(sorted(seen["student_batch"]), student_expected)
                self.assertEqual(sorted(seen["teacher_critic"]), teacher_expected)
                self.assertEqual(sorted(seen["student_critic"]), student_expected)


class TestRoleBatchesMatchHostCTS(unittest.TestCase):
    def test_teacher_student_tuples_identical_to_host(self):
        host = _make_storage(RolloutStorageCTS)
        _fill_with_sentinels(host)
        mine = _make_storage(RolloutStorageMoECTS)
        _fill_with_sentinels(mine)

        num_mini_batches = 2
        # Both generators draw their teacher randperm first and the student
        # randperm second (same sizes, same order, permutation drawn once);
        # identical seeds must reproduce identical role batches field-for-field.
        # Materialize each generator fully before seeding the next one: the
        # randperm calls run lazily on first next(), so interleaving two live
        # generators would scramble the shared RNG stream.
        torch.manual_seed(1234)
        host_data = list(host.mini_batch_generator(num_mini_batches, num_epochs=1))
        torch.manual_seed(1234)
        my_data = list(mine.mini_batch_generator(num_mini_batches, num_epochs=1))
        self.assertEqual(len(host_data), len(my_data))

        for host_sample, my_sample in zip(host_data, my_data):
            host_teacher = host_sample[0:7]
            host_student = host_sample[7:13]
            my_teacher, my_student = my_sample[0], my_sample[1]
            self.assertEqual(len(my_teacher), len(host_teacher))
            self.assertEqual(len(my_student), len(host_student))
            for i, (h, m) in enumerate(zip(host_teacher, my_teacher)):
                self.assertEqual(h.shape, m.shape, f"teacher field {i} shape")
                self.assertTrue(torch.equal(h, m), f"teacher field {i} content")
            for i, (h, m) in enumerate(zip(host_student, my_student)):
                self.assertEqual(h.shape, m.shape, f"student field {i} shape")
                self.assertTrue(torch.equal(h, m), f"student field {i} content")


class TestComputeReturnsIntegration(unittest.TestCase):
    """compute_returns is inherited from RolloutStorageCTS (same signature and
    semantics: caller supplies bootstrap values); smoke-test the full path."""

    def test_compute_returns_then_generate(self):
        torch.manual_seed(7)
        storage = _make_storage(RolloutStorageMoECTS)
        for _ in range(T):
            tr = storage.Transition()
            tr.observations = torch.randn(NUM_ENVS, NUM_OBS)
            tr.privileged_observations = torch.randn(NUM_ENVS, NUM_PRIVILEGED)
            tr.observation_histories = torch.randn(NUM_ENVS, NUM_HISTORY)
            tr.critic_observations = torch.randn(NUM_ENVS, NUM_CRITIC)
            tr.actions = torch.randn(NUM_ENVS, NUM_ACTIONS)
            tr.rewards = torch.randn(NUM_ENVS)
            tr.dones = torch.zeros(NUM_ENVS)
            tr.values = torch.randn(NUM_ENVS, 1)
            tr.actions_log_prob = torch.randn(NUM_ENVS)
            tr.action_mean = torch.randn(NUM_ENVS, NUM_ACTIONS)
            tr.action_sigma = torch.rand(NUM_ENVS, NUM_ACTIONS) + 0.1
            storage.add_transitions(tr)

        last_values = torch.randn(NUM_ENVS, 1)
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)  # base signature

        gen = storage.mini_batch_generator(num_mini_batches=2, num_epochs=1)
        for teacher_batch, student_batch, teacher_critic, student_critic in gen:
            for critic in (teacher_critic, student_critic):
                for field in critic:
                    self.assertTrue(torch.isfinite(field).all())
            # advantages come from the role-specific normalized GAE tensors
            self.assertEqual(teacher_batch[4].shape, student_batch[5].shape)
            self.assertEqual(
                teacher_critic.advantages.shape,
                (NUM_TEACHER * T // 2, 1),
            )
            self.assertEqual(
                student_critic.advantages.shape,
                (NUM_STUDENT * T // 2, 1),
            )


if __name__ == "__main__":
    unittest.main()
