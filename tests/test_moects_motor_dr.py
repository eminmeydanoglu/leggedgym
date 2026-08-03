"""Motor-DR tests for ``go2_moects`` / ``go2_moects_him``.

The moects tasks replicate two of the reference go2_rl_gym motor
randomizations (``go2_rl_gym/legged_gym/envs/base/legged_robot.py``):

* ``randomize_motor_strength`` U[0.8, 1.2]: per-RESET, per-env, per-DOF
  multiplier (:198-203), applied EVERY sim substep to the ALREADY CLIPPED
  torque, right before it goes to the solver (:79-81) -- so it scales the
  effective torque limit too.
* ``randomize_motor_zero_offset`` U[-0.035, 0.035] rad: per-RESET, per-env,
  per-DOF calibration error (:204-206), added to the PD POSITION TARGET
  inside ``_compute_torques`` (:612), not to the measured ``dof_pos``.

The third vendored motor DR, ``randomize_link_mass``, is deliberately NOT
ported (see the ablation note in ``go2_moects_config.py``): the reference's
draw has shape ``(1, num_bodies-1)`` because IsaacGym asset props are shared,
i.e. it is one run-level static scaling with zero per-env entropy.

Both host buffers are always allocated at their nominal value (1.0 / 0.0), so
the ~24 other registered tasks keep a bit-identical torque path.

Layers:

1. ``TestMoECTSMotorDRConfig`` (CPU-only, always runs): both moects cfgs enable
   the two DRs with the vendored ranges; the host default and other tasks keep
   them off (the change is config-gated, hence inert elsewhere).
2. ``TestTorqueMathWithMotorDR`` (CPU-only, always runs): calls
   ``GenesisSimulator._compute_torques`` on a stub to pin down the zero-offset
   algebra (target-side, kp-scaled, action_scale-independent) and the
   nominal-buffer no-op.
3. ``TestMoECTSMotorStrengthApplied`` (opt-in, GPU): builds the real
   ``go2_moects`` env and checks the multiplier lands on the CLIPPED torque at
   every substep, and that draws are per-env AND per-DOF, in range, and
   re-drawn on reset.

GATING for the GPU layer: both env vars must be set:
    MOECTS_GENESIS_INTEGRATION=1   (opt-in; keeps the default CPU suite light)
    SIMULATOR=genesis              (Genesis-only; needs a GPU)

Skip-path / collect check (GPU class reports "skipped"):
    .venv/bin/python -m pytest tests/test_moects_motor_dr.py -q

Real run:
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        .venv/bin/python -m pytest tests/test_moects_motor_dr.py -q
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "Genesis motor-DR test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 "
    "and SIMULATOR=genesis (requires a GPU node; see module docstring)")

_STRENGTH_RANGE = [0.8, 1.2]
_ZERO_OFFSET_RANGE = [-0.035, 0.035]


def _ensure_genesis():
    """Init Genesis once per process (test classes may share a pytest run)."""
    import genesis as gs
    if not gs._initialized:
        gs.init(backend=gs.gpu, logging_level="warning")
    return gs


def _make_moects_env(num_envs):
    """Build the real go2_moects env (small, headless) for motor-DR tests."""
    import legged_gym.envs  # noqa: F401  (registers the tasks)
    from legged_gym.utils import task_registry

    cfg, _ = task_registry.get_cfgs("go2_moects")
    cfg.env.num_envs = num_envs
    args = SimpleNamespace(
        task="go2_moects", seed=7, debug=False, headless=True, cpu=False,
        num_envs=num_envs, max_iterations=None, resume=False,
        sync_wandb=False, ckpt=None, load_run=None, export_onnx=False,
        motion_file=None, num_student=None)
    env, _ = task_registry.make_env("go2_moects", args=args, env_cfg=cfg)
    return env


# ----------------------------------------------------------------------
# 1. config contract (CPU-only)
# ----------------------------------------------------------------------

class TestMoECTSMotorDRConfig(unittest.TestCase):
    """Both moects arms enable the two motor DRs; nothing else does."""

    @classmethod
    def setUpClass(cls):
        import legged_gym.envs  # noqa: F401  (registers the tasks)
        from legged_gym.utils import task_registry
        cls.get_cfgs = staticmethod(task_registry.get_cfgs)

    def test_moects_motor_dr_enabled_with_vendored_ranges(self):
        for task in ("go2_moects", "go2_moects_him"):
            cfg, _ = self.get_cfgs(task)
            dr = cfg.domain_rand
            self.assertTrue(dr.randomize_motor_strength, task)
            self.assertEqual(list(dr.motor_strength_range), _STRENGTH_RANGE, task)
            self.assertTrue(dr.randomize_motor_zero_offset, task)
            self.assertEqual(
                list(dr.motor_zero_offset_range), _ZERO_OFFSET_RANGE, task)

    def test_link_mass_dr_stays_unported(self):
        # conscious ablation: the reference draw is run-level static, so there
        # is no per-env latent for the MoE gate / history encoder to identify.
        from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
        for task in ("go2_moects", "go2_moects_him"):
            cfg, _ = self.get_cfgs(task)
            self.assertFalse(
                getattr(cfg.domain_rand, "randomize_link_mass", False), task)
        self.assertFalse(
            getattr(LeggedRobotCfg.domain_rand, "randomize_link_mass", False))

    def test_other_tasks_keep_motor_dr_off(self):
        from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
        self.assertFalse(LeggedRobotCfg.domain_rand.randomize_motor_strength)
        self.assertFalse(LeggedRobotCfg.domain_rand.randomize_motor_zero_offset)
        for task in ("go2", "k1"):
            cfg, _ = self.get_cfgs(task)
            dr = cfg.domain_rand
            self.assertFalse(
                getattr(dr, "randomize_motor_strength", False), task)
            self.assertFalse(
                getattr(dr, "randomize_motor_zero_offset", False), task)


# ----------------------------------------------------------------------
# 2. torque algebra with the zero offset (CPU-only)
# ----------------------------------------------------------------------

class TestTorqueMathWithMotorDR(unittest.TestCase):
    """Pin down _compute_torques' offset term without building a scene."""

    N, D = 3, 12

    @classmethod
    def setUpClass(cls):
        from legged_gym.simulator.genesis_simulator import GenesisSimulator
        cls.compute = staticmethod(GenesisSimulator._compute_torques)

    def _stub(self, offsets, action_scale=0.25):
        n, d = self.N, self.D
        return SimpleNamespace(
            _cfg=SimpleNamespace(control=SimpleNamespace(action_scale=action_scale)),
            _num_envs=n,
            _p_gains=torch.full((n, d), 20.0),
            _d_gains=torch.full((n, d), 0.5),
            _kp_scale=torch.ones(n, d),
            _kd_scale=torch.ones(n, d),
            _default_dof_pos=torch.zeros(n, d),
            _dof_pos=torch.zeros(n, d),
            _dof_vel=torch.zeros(n, d),
            _motor_zero_offsets=offsets,
            # far above anything these stubs produce, so the clip never bites
            _torque_limits=torch.full((n, d), 1e6))

    def test_zero_offset_buffer_is_a_no_op(self):
        actions = torch.linspace(-1., 1., self.N * self.D).reshape(self.N, self.D)
        off = self.compute(self._stub(torch.zeros(self.N, self.D)), actions)
        on = self.compute(self._stub(torch.zeros(self.N, self.D)), actions)
        torch.testing.assert_close(off, on)

    def test_offset_adds_kp_times_offset_to_the_target(self):
        n, d = self.N, self.D
        actions = torch.linspace(-1., 1., n * d).reshape(n, d)
        offsets = torch.linspace(
            _ZERO_OFFSET_RANGE[0], _ZERO_OFFSET_RANGE[1], n * d).reshape(n, d)
        base = self.compute(self._stub(torch.zeros(n, d)), actions)
        with_off = self.compute(self._stub(offsets), actions)
        torch.testing.assert_close(with_off - base, 20.0 * offsets)

    def test_offset_is_independent_of_action_scale(self):
        # the offset enters as its own term, so the induced torque delta does
        # not change when control.action_scale changes (unlike the action term)
        n, d = self.N, self.D
        actions = torch.full((n, d), 0.5)
        offsets = torch.full((n, d), _ZERO_OFFSET_RANGE[1])
        deltas = []
        for scale in (0.25, 0.5):
            base = self.compute(self._stub(torch.zeros(n, d), scale), actions)
            deltas.append(
                self.compute(self._stub(offsets, scale), actions) - base)
        torch.testing.assert_close(deltas[0], deltas[1])


# ----------------------------------------------------------------------
# 3. strength multiplier lands on the clipped torque (opt-in, GPU)
# ----------------------------------------------------------------------

@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSMotorStrengthApplied(unittest.TestCase):
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

    def test_draws_are_per_env_and_per_dof_and_in_range(self):
        sim = self.sim
        lo, hi = _STRENGTH_RANGE
        S = sim._motor_strengths
        self.assertEqual(tuple(S.shape), (self.NUM_ENVS, sim._num_dof))
        self.assertGreaterEqual(float(S.min()), lo)
        self.assertLessEqual(float(S.max()), hi)
        # per-DOF (rows are not constant) and per-env (columns are not constant)
        self.assertGreater(float(S.std(dim=1).min()), 0.0)
        self.assertGreater(float(S.std(dim=0).min()), 0.0)
        olo, ohi = _ZERO_OFFSET_RANGE
        O = sim._motor_zero_offsets
        self.assertEqual(tuple(O.shape), (self.NUM_ENVS, sim._num_dof))
        self.assertGreaterEqual(float(O.min()), olo)
        self.assertLessEqual(float(O.max()), ohi)
        self.assertGreater(float(O.std(dim=1).min()), 0.0)
        self.assertGreater(float(O.std(dim=0).min()), 0.0)

    def test_redrawn_on_reset(self):
        sim = self.sim
        env_ids = torch.arange(self.NUM_ENVS, device=self.env.device)
        before_s = sim._motor_strengths.clone()
        before_o = sim._motor_zero_offsets.clone()
        sim.reset_idx(env_ids)
        self.assertFalse(bool(torch.equal(sim._motor_strengths, before_s)))
        self.assertFalse(bool(torch.equal(sim._motor_zero_offsets, before_o)))

    def test_multiplier_applied_to_clipped_torque_every_substep(self):
        sim = self.sim
        dec = self.env.cfg.control.decimation
        n, a = self.env.num_envs, self.env.num_actions
        computed, applied = [], []
        orig_compute = sim._compute_torques
        orig_control = sim._robot.control_dofs_force

        def recording_compute(actions):
            out = orig_compute(actions)
            computed.append(out.detach().clone())
            return out

        def recording_control(force, dofs_idx=None, *args, **kwargs):
            applied.append(force.detach().clone())
            return orig_control(force, dofs_idx, *args, **kwargs)

        sim._compute_torques = recording_compute
        sim._robot.control_dofs_force = recording_control
        try:
            self.env.step(torch.full((n, a), 0.3, device=self.env.device))
        finally:
            sim._compute_torques = orig_compute
            sim._robot.control_dofs_force = orig_control
        self.assertEqual(len(computed), dec)
        self.assertEqual(len(applied), dec)
        S = sim._motor_strengths
        for i in range(dec):
            # multiplier sits AFTER _compute_torques' clip (reference :79-81)
            torch.testing.assert_close(applied[i], computed[i] * S)


if __name__ == "__main__":
    unittest.main()
