"""Opt-in Genesis test for the vendored (go2_rl_gym) push-robot domain
randomization on the ``go2_moects`` task.

The moects substrate (WtyCurriculumMixin + GenesisSimulator.push_robots_overwrite)
reproduces go2_rl_gym/legged_gym/envs/base/legged_robot.py::_push_robots:

* per-env trigger: an env is pushed when its OWN episode_length_buf hits
  push_interval (ceil(push_interval_s / dt) = 200 steps = 4 s) -- no global
  lockstep;
* the push OVERWRITES world-frame linear x/y with U(+-max_push_vel_xy = 0.4)
  and world-frame angular x/y/z with U(+-max_push_ang_vel = 0.6);
* linear z and the joint DOF velocities are left untouched.

This module checks:
1. overwrite semantics: an env artificially moving at 5 m/s / 5 rad/s comes
   OUT of the push slower than 0.4 / 0.6 (additive would stay >= 4.6), with
   linear z, joint DOFs and non-pushed envs untouched;
2. the angular component is actually randomized (not silently zero);
3. the trigger is per-env: after de-synchronizing a subset's episode clock,
   its pushes no longer coincide with the rest;
4. push_robots = False disables pushes entirely.

GATING: the whole module is skipped unless BOTH env vars are set:
    MOECTS_GENESIS_INTEGRATION=1   (opt-in; keeps the default CPU suite light)
    SIMULATOR=genesis              (the test is Genesis-only; needs a GPU)

Skip-path / collect check (no GPU work, must report "skipped"):
    .venv/bin/python -m unittest tests.test_moects_push -v

Real run:
    SIMULATOR=genesis MOECTS_GENESIS_INTEGRATION=1 \
        .venv/bin/python -m unittest tests.test_moects_push -v
"""

import os
import unittest
from types import SimpleNamespace

import torch

_INTEGRATION = os.environ.get("MOECTS_GENESIS_INTEGRATION") == "1"
_GENESIS = os.environ.get("SIMULATOR") == "genesis"
_SKIP_REASON = (
    "moects push test is opt-in: set MOECTS_GENESIS_INTEGRATION=1 and "
    "SIMULATOR=genesis (requires a GPU node; see module docstring)")


@unittest.skipUnless(_INTEGRATION and _GENESIS, _SKIP_REASON)
class TestMoECTSPush(unittest.TestCase):
    NUM_ENVS = 16
    # steps 1..205 in lockstep, then envs 0..7 de-synced to episode age 100;
    # their next push lands at step 305, the rest at step 400 -> stop at 410.
    TIMING_STEPS = 410
    DESYNC_STEP = 205
    DESYNC_AGE = 100

    @classmethod
    def setUpClass(cls):
        import genesis as gs
        import legged_gym.envs  # noqa: F401  (registers go2_moects)
        from legged_gym.utils import task_registry

        try:
            gs.init(backend=gs.gpu, logging_level="warning")
        except Exception:
            pass  # already initialized by another test module in this process
        cfg, _ = task_registry.get_cfgs("go2_moects")
        cfg.env.num_envs = cls.NUM_ENVS
        # default episode_length_s (20 s): no timeout resets during the test
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

    def _zero_actions(self):
        return torch.zeros(self.NUM_ENVS, self.env.num_actions,
                           device=self.env.device)

    # ------------------------------------------------------------------
    # 1+2. overwrite semantics and nonzero angular component (direct call)
    # ------------------------------------------------------------------

    def test_01_overwrite_semantics_direct(self):
        from legged_gym.utils.math_utils import quat_rotate_inverse

        env, sim = self.env, self.env.simulator
        cfg_dr = env.cfg.domain_rand
        # vendored numbers wired through the config (go2_rl_gym go2_config.py)
        self.assertEqual(cfg_dr.max_push_vel_xy, 0.4)
        self.assertEqual(cfg_dr.max_push_ang_vel, 0.6)
        self.assertEqual(cfg_dr.push_interval_s, 4)
        self.assertEqual(int(cfg_dr.push_interval), 200)  # ceil(4 / 0.02)

        robot = sim._robot
        dofs_vel = robot.get_dofs_velocity()
        # known pre-push state: fast base (5 m/s xy, 5 rad/s), distinct z,
        # and a snapshot of the joint DOF velocities
        dofs_vel[:, 0:6] = torch.tensor(
            [5.0, -5.0, 0.7, 5.0, -5.0, 5.0], device=env.device)
        joint_vel_before = dofs_vel[:, 6:].clone()
        state_before = dofs_vel.clone()
        robot.set_dofs_velocity(dofs_vel)

        push_env_ids = torch.tensor([1, 5, 9], device=env.device)
        sim.push_robots_overwrite(push_env_ids)

        after = robot.get_dofs_velocity()
        tol = 1e-4
        pushed = after[push_env_ids]
        # overwrite: |xy| <= 0.4 although the env was moving at 5 m/s
        self.assertLessEqual(float(pushed[:, :2].abs().max()), 0.4 + tol)
        # angular overwrite: |ang xyz| <= 0.6 although it was 5 rad/s
        self.assertLessEqual(float(pushed[:, 3:6].abs().max()), 0.6 + tol)
        # ... and the push is real: xy and angular draws are not all ~zero
        self.assertGreater(float(pushed[:, :2].abs().max()), 0.05)
        self.assertGreater(float(pushed[:, 3:6].abs().max()), 0.05)
        # linear z untouched by the push
        self.assertTrue(torch.allclose(
            after[push_env_ids, 2], state_before[push_env_ids, 2], atol=1e-6))
        # joint DOF velocities untouched everywhere
        self.assertTrue(torch.allclose(after[:, 6:], joint_vel_before, atol=1e-6))
        # non-pushed envs completely untouched
        keep = torch.ones(self.NUM_ENVS, dtype=torch.bool, device=env.device)
        keep[push_env_ids] = False
        self.assertTrue(torch.allclose(after[keep], state_before[keep], atol=1e-6))
        # simulator base-velocity buffers refreshed to the pushed state
        self.assertTrue(torch.allclose(
            sim.base_lin_vel,
            quat_rotate_inverse(sim.base_quat, robot.get_vel()), atol=tol))
        self.assertTrue(torch.allclose(
            sim.base_ang_vel,
            quat_rotate_inverse(sim.base_quat, robot.get_ang()), atol=tol))
        # empty call is a no-op
        sim.push_robots_overwrite(torch.empty(
            0, dtype=torch.long, device=env.device))

    # ------------------------------------------------------------------
    # 3. per-env trigger timing through env.step
    # ------------------------------------------------------------------

    def test_02_per_env_timing(self):
        env, sim = self.env, self.env.simulator
        interval = int(env.cfg.domain_rand.push_interval)  # 200
        half = self.NUM_ENVS // 2

        orig = sim.push_robots_overwrite
        calls = []  # env_ids pushed at each env.step

        def spy(env_ids):
            calls.append(env_ids.detach().cpu().clone())
            return orig(env_ids)

        sim.push_robots_overwrite = spy
        try:
            for step in range(1, self.TIMING_STEPS + 1):
                buf_before = env.episode_length_buf.clone()
                env.step(self._zero_actions())
                # host order: episode_length_buf += 1 at the start of
                # post_physics_step, then the mixin's push callback
                expected = ((buf_before + 1) % interval == 0).nonzero(
                    as_tuple=False).flatten().cpu()
                self.assertEqual(len(calls), step)
                self.assertEqual(
                    set(calls[-1].tolist()), set(expected.tolist()),
                    f"push set mismatch at step {step}")
                if step == self.DESYNC_STEP:
                    # de-sync envs [0, half): pretend they reset 100 steps ago
                    env.episode_length_buf[:half] = self.DESYNC_AGE
        finally:
            del sim.push_robots_overwrite  # drop the instance attribute

        # no lockstep: after the de-sync, some pushes cover a strict subset
        subset_pushes = [c for c in calls if 0 < len(c) < self.NUM_ENVS]
        self.assertTrue(subset_pushes,
                        "pushes stayed in global lockstep after de-sync")
        # the de-synced half pushed on its own schedule (step 305), the other
        # half on the original one (step 400)
        pushed_sets = {frozenset(c.tolist()) for c in subset_pushes}
        self.assertIn(frozenset(range(half)), pushed_sets)
        self.assertIn(frozenset(range(half, self.NUM_ENVS)), pushed_sets)

    # ------------------------------------------------------------------
    # 4. push_robots = False disables pushes
    # ------------------------------------------------------------------

    def test_03_push_disabled_flag(self):
        env, sim = self.env, self.env.simulator
        orig = sim.push_robots_overwrite
        calls = []

        def spy(env_ids):
            calls.append(env_ids)
            return orig(env_ids)

        env.cfg.domain_rand.push_robots = False
        sim.push_robots_overwrite = spy
        try:
            for _ in range(15):
                env.step(self._zero_actions())
            self.assertEqual(calls, [], "push fired with push_robots=False")
        finally:
            env.cfg.domain_rand.push_robots = True
            del sim.push_robots_overwrite


if __name__ == "__main__":
    unittest.main()
