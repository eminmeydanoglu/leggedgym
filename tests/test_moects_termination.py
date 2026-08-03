"""Termination-contract tests for the ``go2_moects`` / ``go2_moects_him`` tasks.

The moects tasks replace the host termination semantics with the vendored
go2_rl_gym ones (WtyCurriculumMixin.check_termination, a REPLACEMENT
override):

* terminate the SAME step any termination body (["base"] only) has a
  contact-force norm > cfg.env.base_contact_terminate_threshold (2.5 N;
  vendored go2_rl_gym hardcodes 1.0 N),
* NO tilt / projected-gravity termination (commented out in the reference),
* NO consecutive-failure counter (host: fail_to_terminal_time_s / dt ~= 5
  consecutive steps above 10 N),
* time-out termination unchanged,
* termination-reason telemetry via extras["episode"] ->
  Episode/termination_base_contact and Episode/termination_timeout.

CPU-only unit tests pin the override logic on fabricated tensors (no
simulator) plus the config / MRO wiring of both arms, and guard that the
host's other tasks are unaffected. The Genesis integration class at the
bottom is opt-in (GPU) and validates the real env end to end.

Run CPU suite:
    .venv/bin/python -m unittest tests.test_moects_termination -v
(or:  .venv/bin/python -m pytest tests/test_moects_termination.py -q)

Real run (GPU node; keep it in its own process -- gs.init is once-only):
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        .venv/bin/python -m unittest tests.test_moects_termination -v
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "Genesis termination test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 and "
    "SIMULATOR=genesis (requires a GPU node; run in its own process)")


def _make_fake_env(base_forces, episode_lengths, max_episode_length=25,
                   threshold=2.5, projected_gravity_z=-1.0):
    """Partially-mocked env for WtyCurriculumMixin.check_termination.

    The override is self-contained (no super() call), so a SimpleNamespace
    with the consumed attributes is enough. Link layout: 0 = base (the only
    termination body), 1 = a penalized body, 2/3 = feet.
    """
    base_forces = torch.as_tensor(base_forces, dtype=torch.float)
    num_envs = base_forces.shape[0]
    forces = torch.zeros(num_envs, 4, 3)
    forces[:, 0] = base_forces
    projected_gravity = torch.zeros(num_envs, 3)
    projected_gravity[:, 2] = projected_gravity_z
    return SimpleNamespace(
        simulator=SimpleNamespace(
            link_contact_forces=forces,
            termination_contact_indices=[0],
            penalized_contact_indices=[1],
            feet_contact_indices=[2, 3],
            # read by the HOST check_termination only; the override must
            # ignore it (vendored tilt check is commented out)
            projected_gravity=projected_gravity,
        ),
        base_contact_terminate_threshold=threshold,
        episode_length_buf=torch.tensor(episode_lengths, dtype=torch.int),
        max_episode_length=max_episode_length,
        fail_buf=torch.zeros(num_envs, dtype=torch.long),
    )


class TestMoECTSCheckTerminationLogic(unittest.TestCase):
    """Override logic on fabricated tensors (CPU, no simulator)."""

    def _check(self, env):
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        WtyCurriculumMixin.check_termination(env)
        return env

    def test_above_threshold_terminates_same_step(self):
        # env 1: 3.0 N on the base (> 2.5)
        env = _make_fake_env(
            base_forces=[[0, 0, 0], [3.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            episode_lengths=[3, 3, 3, 3])
        self._check(env)
        self.assertEqual(env.reset_buf.tolist(), [False, True, False, False])
        self.assertEqual(env.terminated_by_base_contact.tolist(),
                         [False, True, False, False])
        # 0/1 same-step flag, NOT a consecutive counter: the host would need
        # fail_buf > fail_to_terminal_time_s/dt (~5) before resetting
        self.assertEqual(env.fail_buf.tolist(), [0, 1, 0, 0])
        self.assertFalse(bool(env.time_out_buf.any()))
        # shared force-norm buffers still populated (rewards / privileged obs)
        self.assertEqual(tuple(env.feet_force_norm.shape), (4, 2))
        self.assertEqual(tuple(env.penalized_bodies_force_norm.shape), (4, 1))
        self.assertEqual(tuple(env.terminated_bodies_force_norm.shape), (4, 1))

    def test_no_consecutive_counter(self):
        env = _make_fake_env(base_forces=[[0, 0, 0], [3.0, 0, 0]],
                             episode_lengths=[3, 3])
        self._check(env)
        self._check(env)  # sustained contact across two checks
        # the flag stays 0/1 (host would count 2); reset fired on the 1st call
        self.assertEqual(env.fail_buf.tolist(), [0, 1])
        self.assertEqual(env.reset_buf.tolist(), [False, True])

    def test_below_threshold_does_not_terminate(self):
        env = _make_fake_env(
            base_forces=[[2.0, 0, 0], [2.49, 0, 0], [0, 0, 2.5]],  # strict >
            episode_lengths=[3, 3, 3])
        self._check(env)
        self.assertFalse(bool(env.reset_buf.any()))
        self.assertFalse(bool(env.terminated_by_base_contact.any()))
        self.assertEqual(env.fail_buf.tolist(), [0, 0, 0])

    def test_tilt_without_contact_does_not_terminate(self):
        # fully flipped robot (projected gravity z -> +1 would trip the host's
        # max_projected_gravity = -0.1) with negligible base contact
        env = _make_fake_env(base_forces=[[0, 0, 0], [0, 0, 0]],
                             episode_lengths=[3, 3],
                             projected_gravity_z=1.0)
        self._check(env)
        self.assertFalse(bool(env.reset_buf.any()))
        self.assertFalse(bool(env.terminated_by_base_contact.any()))

    def test_timeout_path_intact(self):
        # env 1: contact termination; env 3: time-out (episode_length > max)
        env = _make_fake_env(
            base_forces=[[0, 0, 0], [3.0, 0, 0], [0, 0, 0], [0, 0, 0]],
            episode_lengths=[3, 3, 3, 26], max_episode_length=25)
        self._check(env)
        self.assertEqual(env.reset_buf.tolist(), [False, True, False, True])
        self.assertEqual(env.time_out_buf.tolist(), [False, False, False, True])
        self.assertEqual(env.terminated_by_base_contact.tolist(),
                         [False, True, False, False])
        # _reward_termination semantics (reset_buf * ~time_out_buf) unchanged:
        # terminal reward only for the contact-terminated env
        terminal = env.reset_buf * ~env.time_out_buf
        self.assertEqual(terminal.tolist(), [False, True, False, False])

    def test_threshold_comes_from_env_attr(self):
        # a different threshold must be honored (no hardcoded 2.5)
        env = _make_fake_env(base_forces=[[2.0, 0, 0]], episode_lengths=[3],
                             threshold=1.0)
        self._check(env)
        self.assertEqual(env.reset_buf.tolist(), [True])


class TestMoECTSTerminationWiring(unittest.TestCase):
    """Config / MRO wiring for both arms; host tasks unaffected (CPU)."""

    @classmethod
    def setUpClass(cls):
        import legged_gym.envs  # noqa: F401  (registers tasks)
        from legged_gym.utils import task_registry
        cls.env_cfg, _ = task_registry.get_cfgs("go2_moects")
        cls.env_cfg_him, _ = task_registry.get_cfgs("go2_moects_him")
        cls.env_cfg_go2, _ = task_registry.get_cfgs("go2")

    def test_termination_bodies_base_only(self):
        self.assertEqual(self.env_cfg.asset.terminate_after_contacts_on, ["base"])
        self.assertEqual(self.env_cfg_him.asset.terminate_after_contacts_on, ["base"])

    def test_threshold_in_cfg(self):
        self.assertEqual(self.env_cfg.env.base_contact_terminate_threshold, 2.5)
        self.assertEqual(self.env_cfg_him.env.base_contact_terminate_threshold, 2.5)

    def test_check_termination_resolves_to_mixin(self):
        from legged_gym.envs.go2.go2_moects.go2_moects import Go2MoECTS
        from legged_gym.envs.go2.go2_moects.go2_moects_him import Go2MoECTSHIM
        from legged_gym.envs.go2.go2_moects.wty_curriculum_mixin import (
            WtyCurriculumMixin)
        self.assertIs(Go2MoECTS.check_termination,
                      WtyCurriculumMixin.check_termination)
        self.assertIs(Go2MoECTSHIM.check_termination,
                      WtyCurriculumMixin.check_termination)

    def test_host_tasks_unaffected(self):
        from legged_gym.envs.base.legged_robot import LeggedRobot
        from legged_gym.envs.go2.go2 import GO2
        # the plain go2 task keeps the host semantics and config
        self.assertIs(GO2.check_termination, LeggedRobot.check_termination)
        self.assertEqual(self.env_cfg_go2.asset.terminate_after_contacts_on,
                         ["base", "Head"])
        self.assertFalse(hasattr(self.env_cfg_go2.env,
                                 "base_contact_terminate_threshold"))


@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSTerminationGenesis(unittest.TestCase):
    """Real go2_moects env (Genesis, 16 envs): injection + extras keys."""

    NUM_ENVS = 16
    EPISODE_LENGTH_S = 0.5   # test-only override -> 25 control steps @ dt 0.02

    @classmethod
    def setUpClass(cls):
        import genesis as gs
        import legged_gym.envs  # noqa: F401  (registers go2_moects)
        from legged_gym.utils import task_registry

        gs.init(backend=gs.gpu, logging_level="warning")
        cfg, _ = task_registry.get_cfgs("go2_moects")
        cfg.env.num_envs = cls.NUM_ENVS
        cfg.env.episode_length_s = cls.EPISODE_LENGTH_S
        args = SimpleNamespace(
            task="go2_moects", seed=7, debug=False, headless=True, cpu=False,
            num_envs=cls.NUM_ENVS, max_iterations=None, resume=False,
            sync_wandb=False, ckpt=None, load_run=None, export_onnx=False,
            motion_file=None, num_student=None)
        cls.env, _ = task_registry.make_env("go2_moects", args=args, env_cfg=cfg)
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        env = getattr(cls, "env", None)
        if env is not None and hasattr(env, "destroy"):
            env.destroy()

    def test_termination_indices_base_only(self):
        idx = self.env.simulator.termination_contact_indices
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx[0], self.env.simulator._base_link_index)

    def test_threshold_wired_from_cfg(self):
        self.assertEqual(self.env.base_contact_terminate_threshold, 2.5)

    def test_contact_injection_same_step_and_telemetry(self):
        env = self.env
        base_idx = env.simulator.termination_contact_indices[0]
        forces = env.simulator.link_contact_forces
        env.episode_length_buf[:] = 0  # keep the time-out path out of the way

        # (b) below threshold: no termination
        forces[:] = 0.0
        forces[5, base_idx, :] = 2.0
        env.check_termination()
        self.assertFalse(bool(env.reset_buf.any()))

        # (a) above threshold: SAME-step termination of exactly env 5
        forces[5, base_idx, :] = 10.0
        env.check_termination()
        self.assertTrue(bool(env.reset_buf[5]))
        self.assertEqual(int(env.reset_buf.sum()), 1)
        self.assertTrue(bool(env.terminated_by_base_contact[5]))
        self.assertFalse(bool(env.time_out_buf.any()))
        # telemetry on the reset batch: 100% base contact, 0% time-out
        ids = env.reset_buf.nonzero(as_tuple=False).flatten()
        env.reset_idx(ids)
        episode = env.extras["episode"]
        self.assertAlmostEqual(float(episode["termination_base_contact"]), 1.0)
        self.assertAlmostEqual(float(episode["termination_timeout"]), 0.0)

        # (d) time-out path intact: all envs past max_episode_length
        forces[:] = 0.0
        env.episode_length_buf[:] = env.max_episode_length + 1
        env.check_termination()
        self.assertTrue(bool(env.reset_buf.all()))
        self.assertTrue(bool(env.time_out_buf.all()))
        self.assertFalse(bool(env.terminated_by_base_contact.any()))
        ids = env.reset_buf.nonzero(as_tuple=False).flatten()
        env.reset_idx(ids)
        episode = env.extras["episode"]
        self.assertAlmostEqual(float(episode["termination_timeout"]), 1.0)
        self.assertAlmostEqual(float(episode["termination_base_contact"]), 0.0)

    def test_rollout_extras_keys_present(self):
        # two zero-action episodes: resets must be time-outs, and every
        # extras["episode"] batch must carry both termination metrics in [0,1]
        env = self.env
        last_episode = None
        saw_timeout = False
        for _ in range(int(2 * env.max_episode_length)):
            actions = torch.zeros(self.NUM_ENVS, env.num_actions, device=env.device)
            *_, extras = env.step(actions)
            if extras.get("episode"):
                episode = extras["episode"]
                self.assertIn("termination_base_contact", episode)
                self.assertIn("termination_timeout", episode)
                for key in ("termination_base_contact", "termination_timeout"):
                    value = float(episode[key])
                    self.assertGreaterEqual(value, 0.0, key)
                    self.assertLessEqual(value, 1.0, key)
                saw_timeout |= float(episode["termination_timeout"]) > 0.0
                last_episode = episode
        self.assertIsNotNone(last_episode, "no reset batch observed")
        self.assertTrue(saw_timeout, "expected time-out resets in a 2-episode rollout")


if __name__ == "__main__":
    unittest.main()
