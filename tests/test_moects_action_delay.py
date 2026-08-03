"""Action-delay model tests for ``go2_moects`` / ``go2_moects_him``.

The moects tasks replicate the reference go2_rl_gym action delay
(``go2_rl_gym/legged_gym/envs/base/legged_robot.py::step``):

* EVERY control step, each env draws a delay k ~ U{0,1,2,3,4} in SIM SUBSTEP
  units (sim dt 0.005 s x decimation 4 -> 0/5/10/15/20 ms),
* during the control step the simulator blends the PREVIOUS and CURRENT action
  targets: substeps i < k use the previous target, substeps i >= k the current,
* at episode start the previous target is zeroed (reference zeroes
  ``last_actions`` in ``reset_idx``).

The host's legacy model (one delay per EPISODE from ``ctrl_delay_step_range``
CONTROL steps, applied via an action queue) remains the default
(``domain_rand.ctrl_delay_mode == "queue"``) for all other tasks; the moects
configs opt into ``ctrl_delay_mode == "substep"``.

Four layers of checks:

1. ``TestMoECTSActionDelayConfig`` (CPU-only, always runs): both moects cfgs
   carry the substep-delay contract and the host default / other tasks are
   untouched (the change is config-gated, hence inert elsewhere).
2. ``TestCtrlDelaySubstepRangeValidation`` (CPU-only, always runs): the
   substep range validator accepts in-range and rejects out-of-range values.
3. ``TestMoECTSSubstepDelayBlend`` (opt-in, GPU): builds the real
   ``go2_moects`` env (8 envs, headless), forces the RNG draw to k = 0, 2, 4
   and checks which target each sim substep applies, plus the previous-target
   bookkeeping across control steps.
4. ``TestMoECTSSubstepDelayDistribution`` (opt-in, GPU): over 250 control
   steps the drawn delays cover {0..4}, stay near-uniform, and change from
   step to step (not constant within an episode).
5. ``TestLegacyQueueDelayUnchanged`` (opt-in, GPU): with the mode overridden
   back to ``"queue"`` the per-episode queue behaviour is exactly as before
   (guard for all non-moects tasks).

GATING for the GPU layers: both env vars must be set:
    MOECTS_GENESIS_INTEGRATION=1   (opt-in; keeps the default CPU suite light)
    SIMULATOR=genesis              (Genesis-only; needs a GPU)

Skip-path / collect check (GPU classes report "skipped"):
    .venv/bin/python -m pytest tests/test_moects_action_delay.py -q

Real run:
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        .venv/bin/python -m pytest tests/test_moects_action_delay.py -q
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "Genesis action-delay test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 "
    "and SIMULATOR=genesis (requires a GPU node; see module docstring)")


def _ensure_genesis():
    """Init Genesis once per process (test classes may share a pytest run)."""
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")
    return gs


def _make_moects_env(num_envs, ctrl_delay_mode=None):
    """Build the real go2_moects env (small, headless) for delay tests."""
    import legged_gym.envs  # noqa: F401  (registers the tasks)
    from legged_gym.utils import task_registry

    cfg, _ = task_registry.get_cfgs("go2_moects")
    cfg.env.num_envs = num_envs
    if ctrl_delay_mode is not None:
        cfg.domain_rand.ctrl_delay_mode = ctrl_delay_mode
    args = SimpleNamespace(
        task="go2_moects", seed=7, debug=False, headless=True, cpu=False,
        num_envs=num_envs, max_iterations=None, resume=False,
        sync_wandb=False, ckpt=None, load_run=None, export_onnx=False,
        motion_file=None, num_student=None)
    env, _ = task_registry.make_env("go2_moects", args=args, env_cfg=cfg)
    return env


def _install_torque_recorder(sim):
    """Wrap sim._compute_torques so every substep's applied target is recorded.

    Returns (records, original); the caller MUST restore the original.
    """
    records = []
    original = sim._compute_torques

    def recording(actions):
        records.append(actions.detach().clone())
        return original(actions)

    sim._compute_torques = recording
    return records, original


# ----------------------------------------------------------------------
# 1. config contract (CPU-only)
# ----------------------------------------------------------------------

class TestMoECTSActionDelayConfig(unittest.TestCase):
    """Both moects arms opt into the reference-style substep delay; the host
    default and unrelated tasks keep the legacy per-episode queue model."""

    @classmethod
    def setUpClass(cls):
        import legged_gym.envs  # noqa: F401  (registers the tasks)
        from legged_gym.utils import task_registry
        cls.get_cfgs = staticmethod(task_registry.get_cfgs)

    def test_moects_substep_mode_enabled(self):
        for task in ("go2_moects", "go2_moects_him"):
            cfg, _ = self.get_cfgs(task)
            dr = cfg.domain_rand
            self.assertTrue(dr.randomize_ctrl_delay, task)
            self.assertEqual(dr.ctrl_delay_mode, "substep", task)
            self.assertEqual(list(dr.ctrl_delay_substep_range), [0, 4], task)
            # reference timing: 0.005 s sim dt x 4 substeps = 0/5/10/15/20 ms
            self.assertEqual(cfg.control.decimation, 4, task)
            self.assertEqual(cfg.sim.dt, 0.005, task)

    def test_other_tasks_keep_legacy_default(self):
        from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
        self.assertEqual(LeggedRobotCfg.domain_rand.ctrl_delay_mode, "queue")
        self.assertEqual(
            list(LeggedRobotCfg.domain_rand.ctrl_delay_substep_range), [0, 4])
        for task in ("go2", "k1"):
            cfg, _ = self.get_cfgs(task)
            self.assertEqual(
                getattr(cfg.domain_rand, "ctrl_delay_mode", "queue"), "queue",
                task)


# ----------------------------------------------------------------------
# 2. substep-range validation (CPU-only)
# ----------------------------------------------------------------------

class TestCtrlDelaySubstepRangeValidation(unittest.TestCase):
    """The validator enforces 0 <= min <= max <= decimation (substep units)."""

    @classmethod
    def setUpClass(cls):
        from legged_gym.simulator.genesis_simulator import (
            _validate_ctrl_delay_substep_range)
        cls.validate = staticmethod(_validate_ctrl_delay_substep_range)

    def test_valid_ranges(self):
        for r in ([0, 4], [0, 0], [2, 2], [1, 3], [4, 4]):
            self.validate(r, 4)

    def test_out_of_range_rejected(self):
        for r in ([0, 5], [-1, 2], [3, 1], [1, 5]):
            with self.assertRaises(ValueError, msg=f"range {r} must be rejected"):
                self.validate(r, 4)


# ----------------------------------------------------------------------
# 3. blend semantics with forced draws (opt-in, GPU)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSSubstepDelayBlend(unittest.TestCase):
    NUM_ENVS = 8

    @classmethod
    def setUpClass(cls):
        _ensure_genesis()
        cls.env = _make_moects_env(cls.NUM_ENVS)
        cls.sim = cls.env.simulator
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        env = getattr(cls, "env", None)
        if env is not None and hasattr(env, "destroy"):
            env.destroy()

    def _force_draw(self, k):
        sim = self.sim

        def fake_draw():
            return torch.full((sim._num_envs, 1), k, dtype=torch.long,
                              device=sim._device)
        return fake_draw

    def test_blend_rule_for_forced_delays(self):
        sim = self.sim
        self.assertTrue(sim._ctrl_delay_substep_mode)
        self.assertFalse(hasattr(sim, "_action_queue"),
                         "substep mode must not build the legacy action queue")
        dec = self.env.cfg.control.decimation
        n, a = self.env.num_envs, self.env.num_actions
        dev = self.env.device
        A0 = torch.full((n, a), 0.10, device=dev)
        A1 = torch.full((n, a), -0.20, device=dev)
        A2 = torch.linspace(0.05, 0.30, n * a, device=dev).reshape(n, a)
        A3 = torch.full((n, a), 0.40, device=dev)
        records, orig_torques = _install_torque_recorder(sim)
        orig_draw = sim._draw_substep_delay
        try:
            sim._draw_substep_delay = self._force_draw(0)
            self.env.step(A0)  # establishes A0 as the previous target
            # k = 0: every substep applies the CURRENT target
            records.clear()
            self.env.step(A1)
            self.assertEqual(len(records), dec)
            for i in range(dec):
                torch.testing.assert_close(records[i], A1)
            # the current target becomes the previous one for the next step
            torch.testing.assert_close(sim._prev_action_targets, A1)
            # k = 2: substeps 0,1 keep the PREVIOUS target (A1); 2,3 the CURRENT (A2)
            sim._draw_substep_delay = self._force_draw(2)
            records.clear()
            self.env.step(A2)
            self.assertEqual(len(records), dec)
            for i in range(dec):
                torch.testing.assert_close(records[i], A1 if i < 2 else A2)
            # k = 4 (= decimation): every substep keeps the PREVIOUS target (A2)
            sim._draw_substep_delay = self._force_draw(4)
            records.clear()
            self.env.step(A3)
            self.assertEqual(len(records), dec)
            for i in range(dec):
                torch.testing.assert_close(records[i], A2)
            torch.testing.assert_close(sim._prev_action_targets, A3)
        finally:
            sim._compute_torques = orig_torques
            sim._draw_substep_delay = orig_draw

    def test_reset_zeroes_previous_target(self):
        sim = self.sim
        env_ids = torch.arange(self.NUM_ENVS, device=self.env.device)
        sim._prev_action_targets[:] = 1.0
        sim.reset_idx(env_ids)
        self.assertTrue(bool((sim._prev_action_targets == 0.0).all()))


# ----------------------------------------------------------------------
# 4. per-step redraw + uniformity (opt-in, GPU)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSSubstepDelayDistribution(unittest.TestCase):
    NUM_ENVS = 8
    N_STEPS = 250

    @classmethod
    def setUpClass(cls):
        _ensure_genesis()
        cls.env = _make_moects_env(cls.NUM_ENVS)
        cls.sim = cls.env.simulator
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        env = getattr(cls, "env", None)
        if env is not None and hasattr(env, "destroy"):
            env.destroy()

    def test_draws_uniform_and_changing_every_control_step(self):
        sim = self.sim
        zero = torch.zeros(self.env.num_envs, self.env.num_actions,
                           device=self.env.device)
        draws = []
        for _ in range(self.N_STEPS):
            self.env.step(zero)
            draws.append(sim._action_delay.clone())
        D = torch.stack(draws)  # (N_STEPS, NUM_ENVS), substep units
        # range and full coverage of {0..4}
        self.assertGreaterEqual(int(D.min()), 0)
        self.assertLessEqual(int(D.max()), self.env.cfg.control.decimation)
        for k in range(self.env.cfg.control.decimation + 1):
            self.assertTrue(bool((D == k).any()), f"delay {k} never drawn")
        # loose uniformity over 250*8 = 2000 draws (expected 400 per value;
        # [0.12, 0.28] is about +/-9 sigma, so no flakiness)
        frac = torch.stack(
            [(D == k).float().mean() for k in range(5)])
        self.assertTrue(bool(((frac > 0.12) & (frac < 0.28)).all()),
                        f"draws far from uniform: {frac.tolist()}")
        # re-drawn every control step: no env keeps a constant delay
        per_env_nunique = [int(torch.unique(D[:, e]).numel())
                           for e in range(D.shape[1])]
        self.assertTrue(all(u > 1 for u in per_env_nunique),
                        f"constant per-env delays: {per_env_nunique}")
        same_as_prev = float((D[1:] == D[:-1]).float().mean())
        self.assertLess(same_as_prev, 0.5,
                        f"delay redrawn too rarely (P(same)={same_as_prev:.3f})")


# ----------------------------------------------------------------------
# 5. legacy per-episode queue unchanged (opt-in, GPU; guard for other tasks)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestLegacyQueueDelayUnchanged(unittest.TestCase):
    NUM_ENVS = 4

    @classmethod
    def setUpClass(cls):
        _ensure_genesis()
        cls.env = _make_moects_env(cls.NUM_ENVS, ctrl_delay_mode="queue")
        cls.sim = cls.env.simulator
        cls.env.reset()

    @classmethod
    def tearDownClass(cls):
        env = getattr(cls, "env", None)
        if env is not None and hasattr(env, "destroy"):
            env.destroy()

    def test_per_episode_queue_semantics(self):
        sim = self.sim
        self.assertFalse(sim._ctrl_delay_substep_mode)
        self.assertTrue(hasattr(sim, "_action_queue"))
        self.assertFalse(hasattr(sim, "_prev_action_targets"),
                         "queue mode must not build the substep blend buffer")
        dec = self.env.cfg.control.decimation
        n, a = self.env.num_envs, self.env.num_actions
        dev = self.env.device
        A1 = torch.full((n, a), 0.15, device=dev)
        A2 = torch.full((n, a), -0.25, device=dev)
        d0 = sim._action_delay.clone()
        self.assertTrue(bool(((d0 >= 0) & (d0 <= 1)).all()),
                        "legacy delay range is [0, 1] control steps")
        records, orig_torques = _install_torque_recorder(sim)
        try:
            self.env.step(A1)  # queue: [A1, 0]
            records.clear()
            self.env.step(A2)  # queue: [A2, A1]
            # the per-episode delay does NOT change between control steps
            self.assertTrue(bool(torch.equal(sim._action_delay, d0)))
            # applied target is the action from d control steps ago, constant
            # across all substeps of the control step
            self.assertEqual(len(records), dec)
            for e in range(n):
                expected = A2[e] if int(d0[e]) == 0 else A1[e]
                for i in range(dec):
                    torch.testing.assert_close(records[i][e], expected)
        finally:
            sim._compute_torques = orig_torques


if __name__ == "__main__":
    unittest.main()
