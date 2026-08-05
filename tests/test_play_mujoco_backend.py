"""play.py's backend plumbing for the MuJoCo option.

Three things are pinned here, all of them cheap and none of them requiring a
physics engine:

* ``--sim`` reaches ``os.environ["SIMULATOR"]`` BEFORE ``legged_gym`` is
  imported.  ``legged_gym/__init__.py`` freezes the backend at import time, so
  if this bootstrap ever regresses the flag silently does nothing and play comes
  up on the wrong simulator -- the single worst failure mode of this feature.
* Which backends resolve to a real heightfield.  MuJoCo joins Genesis because
  ``simulator/mujoco_scene.py`` maps ``Terrain.height_field_raw`` onto an MJCF
  hfield asset; Isaac* still build the trimesh they always did.
* ``--terrain moe`` is gated on that same set rather than on Genesis by name,
  which is what makes the MoE showcase reachable from MuJoCo at all.
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

# Repo root on path so `legged_gym` imports resolve without install.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("SIMULATOR", "genesis")

from legged_gym.scripts import play as play_mod


class TestSimulatorArgvBootstrap(unittest.TestCase):
    """``_bootstrap_simulator_from_argv`` is the pre-import half of ``--sim``."""

    def setUp(self):
        self._argv = sys.argv
        self._env = os.environ.get("SIMULATOR")

    def tearDown(self):
        sys.argv = self._argv
        if self._env is None:
            os.environ.pop("SIMULATOR", None)
        else:
            os.environ["SIMULATOR"] = self._env

    def _run(self, argv):
        sys.argv = ["play.py"] + argv
        play_mod._bootstrap_simulator_from_argv()
        return os.environ.get("SIMULATOR")

    def test_space_separated_form(self):
        self.assertEqual(self._run(["--task", "go2_moects", "--sim", "mujoco"]), "mujoco")

    def test_equals_form(self):
        self.assertEqual(self._run(["--sim=mujoco", "--terrain", "moe"]), "mujoco")

    def test_absent_flag_leaves_the_environment_untouched(self):
        os.environ["SIMULATOR"] = "genesis"
        self.assertEqual(self._run(["--task", "go2_moects", "--terrain", "moe"]), "genesis")

    def test_trailing_flag_with_no_value_does_not_raise(self):
        # argparse would reject this later with a proper message; the bootstrap
        # must not blow up with an IndexError before that error can be printed.
        os.environ["SIMULATOR"] = "genesis"
        self.assertEqual(self._run(["--sim"]), "genesis")


class TestHeightfieldBackendSet(unittest.TestCase):
    def test_membership(self):
        self.assertIn("genesis", play_mod.HEIGHTFIELD_SIMULATORS)
        self.assertIn("mujoco", play_mod.HEIGHTFIELD_SIMULATORS)
        self.assertNotIn("isaacgym", play_mod.HEIGHTFIELD_SIMULATORS)
        self.assertNotIn("isaaclab", play_mod.HEIGHTFIELD_SIMULATORS)

    def test_mesh_type_constant_tracks_the_live_backend(self):
        expected = ("heightfield"
                    if play_mod.SIMULATOR in play_mod.HEIGHTFIELD_SIMULATORS
                    else "trimesh")
        self.assertEqual(play_mod.PLAY_MESH_TYPE, expected)


def _make_env_cfg():
    """Minimal duck-typed cfg with just what the moe branch reads/writes."""
    return SimpleNamespace(terrain=SimpleNamespace(
        mesh_type="plane",
        curriculum=True,
        selected=True,
        terrain_kwargs={"foo": 1},
        ued_training_grid=True,
        taxonomy_showcase=True,
        v6_frontier_showcase=True,
        moe_grid=False,
        moe_showcase=False,
        measure_heights=True,
        horizontal_scale=0.1,
        terrain_proportions=None,
        terrain_length=None,
        terrain_width=None,
        terrain_spacing=None,
        num_rows=0,
        num_cols=0,
        border_size=25,
        max_init_terrain_level=0,
    ))


class TestMoeTerrainGate(unittest.TestCase):
    """``--terrain moe`` must open for MuJoCo and stay shut for the trimesh backends."""

    def _configure_as(self, backend):
        """Run the moe branch as if play had been imported under ``backend``."""
        env_cfg = _make_env_cfg()
        mesh = ("heightfield" if backend in play_mod.HEIGHTFIELD_SIMULATORS
                else "trimesh")
        old = (play_mod.SIMULATOR, play_mod.PLAY_MESH_TYPE)
        play_mod.SIMULATOR, play_mod.PLAY_MESH_TYPE = backend, mesh
        try:
            play_mod.configure_play_terrain(env_cfg, "moe")
        finally:
            play_mod.SIMULATOR, play_mod.PLAY_MESH_TYPE = old
        return env_cfg

    def test_mujoco_builds_the_same_showcase_as_genesis(self):
        mujoco_cfg = self._configure_as("mujoco")
        genesis_cfg = self._configure_as("genesis")
        self.assertEqual(mujoco_cfg.terrain.mesh_type, "heightfield")
        # The showcase geometry must not depend on the backend at all -- that is
        # the premise of comparing the two runs.
        self.assertEqual(vars(mujoco_cfg.terrain), vars(genesis_cfg.terrain))
        self.assertTrue(mujoco_cfg.terrain.moe_grid)
        self.assertTrue(mujoco_cfg.terrain.moe_showcase)
        self.assertFalse(mujoco_cfg.terrain.ued_training_grid)

    def test_trimesh_backends_are_still_rejected(self):
        for backend in ("isaacgym", "isaaclab"):
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError) as ctx:
                    self._configure_as(backend)
                # The message must name the backends that DO work, not just the
                # one that failed, or the user has nothing to act on.
                self.assertIn("genesis", str(ctx.exception))
                self.assertIn("mujoco", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
