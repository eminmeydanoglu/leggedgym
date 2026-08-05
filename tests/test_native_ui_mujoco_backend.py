"""NativeArenaUI against the MuJoCo viewer backend.

Nothing here imports mujoco, opens a window or builds an env: the whole
backend seam is four setter methods on a duck-typed object
(``set_key_callback`` / ``set_scroll_callback`` / ``set_drag_callback`` /
``set_hud_lines``), so a fake viewer plus the same minimal fake env the
Genesis HUD tests use exercises every ingress point.

The invariants pinned here are the ones that would otherwise only fail with a
robot on screen:

* backend detection reads ``simulator._viewer`` and reports "mujoco" (and does
  it by attribute provenance, so the module still imports with neither Genesis
  nor MuJoCo installed);
* both key edges reach ``KeyboardSource`` -- a press that never releases is the
  stuck-key failure this whole module exists to remove;
* arena actions fire on the press edge ONLY, and Shift+Tab still resolves to
  ``tab_prev`` through the hand-tracked shift flag rather than a modifier
  bitmask the MuJoCo viewer does not even report;
* drag/scroll land in the same accumulators, in the same units and with the
  same signs, as the pyglet handlers -- the orbit rig maths is shared and must
  not learn about backends;
* the viewer is handed the EXACT list object ``update_hud`` later mutates in
  place (it keeps the reference and repaints from it, so a reallocation would
  freeze the HUD on its startup text).
"""
import unittest

import numpy as np

from legged_gym.scripts.input_source import DriveEnvelope, KeyboardSource, MergedSource
from legged_gym.utils import native_ui
from legged_gym.utils.native_ui import NativeArenaUI


class FakeMujocoViewer:
    """The four setters ``MujocoViewer`` exposes to the arena, and nothing else.

    Deliberately NOT a subclass of anything and deliberately without
    ``set_camera`` / ``sync`` / ``is_running``: the arena must reach this object
    purely through the setters, and a missing attribute here is a real defect
    rather than a fixture gap.
    """

    def __init__(self):
        self.key_cb = None
        self.scroll_cb = None
        self.drag_cb = None
        self.hud_lines = None

    def set_key_callback(self, cb):
        self.key_cb = cb

    def set_scroll_callback(self, cb):
        self.scroll_cb = cb

    def set_drag_callback(self, cb):
        self.drag_cb = cb

    def set_hud_lines(self, lines):
        self.hud_lines = lines


class FakeTerrainCfg:
    moe_showcase = False
    num_cols = 3
    num_rows = 3


class FakeCfg:
    terrain = FakeTerrainCfg()


class FakeMujocoSimulator:
    """``_scene`` is absent (that is the Genesis hook); ``_viewer`` is the window."""

    _terrain_origins = None

    def __init__(self, viewer):
        self._viewer = viewer
        self.base_pos = np.array([[10.0, 20.0, 0.5]], dtype=np.float32)
        self.base_euler = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)


class FakeEnv:
    def __init__(self, viewer):
        self.cfg = FakeCfg()
        self.simulator = FakeMujocoSimulator(viewer)
        self.commands = None
        self.camera_poses = []

    def set_viewer_camera(self, eye, lookat):
        self.camera_poses.append((np.asarray(eye), np.asarray(lookat)))


def make_arena():
    """A fully wired arena: construction alone must find and bind the viewer."""
    viewer = FakeMujocoViewer()
    source = MergedSource(keyboard=KeyboardSource(DriveEnvelope()), gamepad=None)
    arena = NativeArenaUI(
        FakeEnv(viewer), source,
        tabs=["wave", "stairs_up", "gap"], levels=[0, 4, 9], quiet=True,
    )
    return arena, viewer


class TestBackendDetection(unittest.TestCase):
    def test_simulator_viewer_is_found_and_labelled_mujoco(self):
        arena, viewer = make_arena()
        self.assertIs(arena.viewer, viewer)
        self.assertEqual(arena._viewer_backend, "mujoco")

    def test_every_ingress_point_bound_during_construction(self):
        _, viewer = make_arena()
        self.assertTrue(callable(viewer.key_cb))
        self.assertTrue(callable(viewer.drag_cb))
        self.assertTrue(callable(viewer.scroll_cb))
        self.assertIsNotNone(viewer.hud_lines)

    def test_headless_simulator_yields_no_viewer_and_no_backend(self):
        viewer = FakeMujocoViewer()
        env = FakeEnv(viewer)
        env.simulator._viewer = None  # headless run
        source = MergedSource(keyboard=KeyboardSource(), gamepad=None)
        arena = NativeArenaUI(env, source, tabs=["a"], levels=[0], quiet=True)
        self.assertIsNone(arena.viewer)
        self.assertIsNone(arena._viewer_backend)
        self.assertIsNone(viewer.key_cb)


class TestKeyRouting(unittest.TestCase):
    def test_drive_key_press_and_release_reach_the_keyboard_source(self):
        arena, viewer = make_arena()
        held = arena.source.keyboard.held
        self.assertFalse(held()["forward"])
        viewer.key_cb("UP", True)
        self.assertTrue(held()["forward"])
        # The release edge is the whole reason the viewer reports both: without
        # it the robot would drive on with nobody touching the keyboard.
        viewer.key_cb("UP", False)
        self.assertFalse(held()["forward"])

    def test_every_drive_key_maps_to_its_documented_action(self):
        arena, viewer = make_arena()
        for key_name, action in native_ui.DRIVE_KEYMAP.values():
            with self.subTest(key=key_name):
                viewer.key_cb(key_name, True)
                self.assertTrue(arena.source.keyboard.held()[action])
                viewer.key_cb(key_name, False)
                self.assertFalse(arena.source.keyboard.held()[action])

    def test_tab_fires_tab_next_on_the_press_edge_only(self):
        arena, viewer = make_arena()
        viewer.key_cb("TAB", True)
        viewer.key_cb("TAB", False)
        self.assertEqual(arena.source.drain_events(), ["tab_next"])

    def test_shift_tab_fires_tab_prev(self):
        # Shift is tracked by hand, not read off a modifier bitmask -- the
        # MuJoCo viewer reports no modifier state at all, so the flag is the
        # only thing that can distinguish the two.
        arena, viewer = make_arena()
        viewer.key_cb("LSHIFT", True)
        viewer.key_cb("TAB", True)
        viewer.key_cb("TAB", False)
        self.assertEqual(arena.source.drain_events(), ["tab_prev"])

        viewer.key_cb("LSHIFT", False)
        viewer.key_cb("TAB", True)
        self.assertEqual(arena.source.drain_events(), ["tab_next"])

    def test_right_shift_tracks_the_same_flag(self):
        arena, viewer = make_arena()
        viewer.key_cb("RSHIFT", True)
        viewer.key_cb("TAB", True)
        self.assertEqual(arena.source.drain_events(), ["tab_prev"])

    def test_action_keys_map_to_their_events(self):
        arena, viewer = make_arena()
        for key_name, event in native_ui.ACTION_KEYMAP.values():
            with self.subTest(key=key_name):
                viewer.key_cb(key_name, True)
                self.assertEqual(arena.source.drain_events(), [event])

    def test_unknown_key_is_ignored(self):
        arena, viewer = make_arena()
        viewer.key_cb("F7", True)   # bound by nothing in either keymap
        viewer.key_cb("F7", False)
        self.assertEqual(arena.source.drain_events(), [])


class TestLookAccumulators(unittest.TestCase):
    def test_drag_accumulates_pixels_with_the_pyglet_sign_convention(self):
        arena, viewer = make_arena()
        viewer.drag_cb(30.0, 10.0)
        self.assertEqual(arena._mouse_look, [30.0, 10.0])
        viewer.drag_cb(-5.0, 2.0)      # deltas accumulate, they do not replace
        self.assertEqual(arena._mouse_look, [25.0, 12.0])

    def test_drag_moves_the_rig_by_the_documented_per_pixel_rate(self):
        # The rig maths is shared with Genesis and must stay backend-blind, so
        # assert the same numbers the pyglet drag test asserts.
        arena, viewer = make_arena()
        arena.update_follow_camera(dt=0.0)   # snap behind the robot first
        start_azimuth = arena._camera_azimuth
        viewer.drag_cb(30.0, 10.0)
        # Nothing has moved yet: the callback only accumulated pixels.
        self.assertEqual(arena._camera_azimuth, start_azimuth)
        arena.update_follow_camera(dt=1 / 60.0)
        self.assertAlmostEqual(
            arena._camera_azimuth,
            start_azimuth - 30.0 * native_ui.GTA_MOUSE_AZIMUTH_PER_PX, places=6)
        self.assertAlmostEqual(
            arena._camera_pitch,
            native_ui.GTA_PITCH + 10.0 * native_ui.GTA_MOUSE_PITCH_PER_PX,
            places=6)

    def test_scroll_accumulates_notches_and_zooms_in_on_positive(self):
        arena, viewer = make_arena()
        arena.update_follow_camera(dt=0.0)
        viewer.scroll_cb(1.0)
        self.assertEqual(arena._mouse_scroll, 1.0)
        arena.update_follow_camera(dt=1 / 60.0)
        self.assertAlmostEqual(
            arena._camera_distance,
            native_ui.GTA_DISTANCE * (1.0 - native_ui.GTA_SCROLL_STEP), places=6)

    def test_look_input_is_dropped_outside_gta(self):
        # In free/rear/front the MuJoCo viewer owns the mouse; banking pixels
        # nobody drains would whip the camera on the next switch back to gta.
        arena, viewer = make_arena()
        arena.next_camera_mode()   # rear
        viewer.drag_cb(50.0, 50.0)
        viewer.scroll_cb(3.0)
        self.assertEqual(arena._mouse_look, [0.0, 0.0])
        self.assertEqual(arena._mouse_scroll, 0.0)


class TestHudHandover(unittest.TestCase):
    def test_viewer_holds_the_exact_list_update_hud_mutates_in_place(self):
        arena, viewer = make_arena()
        self.assertIs(viewer.hud_lines, arena._hud_lines)
        self.assertEqual(len(viewer.hud_lines), 4)

        arena.update_hud(command=(1.25, 0.0, -0.5), fps=50.0, force=True)
        # Same object, new strings: the viewer repaints from the reference it
        # was handed at install time, so a reallocation would freeze the HUD.
        self.assertIs(viewer.hud_lines, arena._hud_lines)
        self.assertIn("+1.25", viewer.hud_lines[1])
        self.assertIn("-0.50", viewer.hud_lines[1])

    def test_tab_and_level_changes_show_up_through_the_same_list(self):
        arena, viewer = make_arena()
        arena.tab_index, arena.level_index = 1, 2
        arena.update_hud(force=True)
        self.assertIn("stairs_up", viewer.hud_lines[0])
        self.assertIn("L9", viewer.hud_lines[0])

    def test_lines_stay_single_line_ascii(self):
        _, viewer = make_arena()
        for text in viewer.hud_lines:
            self.assertNotIn("\n", text)
            self.assertTrue(all(32 <= ord(c) < 127 for c in text), text)


class TestDegradation(unittest.TestCase):
    def test_a_viewer_without_a_hud_prints_and_keeps_the_session_alive(self):
        class NoHudViewer(FakeMujocoViewer):
            def set_hud_lines(self, lines):
                raise AttributeError("no overlay on this build")

        viewer = NoHudViewer()
        source = MergedSource(keyboard=KeyboardSource(), gamepad=None)
        arena = NativeArenaUI(FakeEnv(viewer), source, tabs=["a"], levels=[0],
                              quiet=True)
        self.assertIsNone(arena._hud_lines)
        self.assertFalse(arena.update_hud(force=True))
        # Drive still works; only the on-screen text is gone.
        viewer.key_cb("UP", True)
        self.assertTrue(arena.source.keyboard.held()["forward"])

    def test_a_viewer_without_key_support_disables_drive_not_the_session(self):
        class NoKeysViewer(FakeMujocoViewer):
            def set_key_callback(self, cb):
                raise AttributeError("no key hook")

        source = MergedSource(keyboard=KeyboardSource(), gamepad=None)
        arena = NativeArenaUI(FakeEnv(NoKeysViewer()), source, tabs=["a"],
                              levels=[0], quiet=True)
        self.assertIsNone(arena.viewer)

    def test_unregister_is_safe_and_idempotent(self):
        arena, viewer = make_arena()
        arena.unregister()
        arena.unregister()
        self.assertIsNone(arena._hud_lines)
        # The callbacks are intentionally left attached (the contract has no
        # removers) and must still be harmless after teardown.
        viewer.key_cb("UP", True)
        viewer.drag_cb(1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
