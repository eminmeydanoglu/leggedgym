"""Unit coverage for the native viewer's on-screen HUD.

Nothing here opens a window: the HUD is a duck-typed plugin appended to
``viewer._pyrender_viewer.plugins`` that only ever *reads* the shared line
list, so a plain object with a ``plugins`` list is enough to exercise
install / update / teardown, and the plugin's pure geometry math
(``_backplate`` / ``_measure`` fallback) runs without a GL context.

The invariants under test are the ones that would otherwise only fail at
runtime, on the *viewer* thread, inside Genesis's draw loop:

* every HUD string is a single line of ASCII (``pyrender``'s glyph map is
  built from ``chr(0..127)`` and ``render_string`` walks the string character
  by character, so ``\\n`` or any non-ASCII glyph raises ``KeyError`` mid-draw);
* the line *list* is allocated once and only its strings change (the plugin
  iterates that exact list every frame);
* no two lines share a horizontal band: status owns TOP_CENTER, telemetry
  BOTTOM_LEFT, the short hint TOP_RIGHT, and Genesis keeps TOP_LEFT /
  BOTTOM_RIGHT for its own text;
* ``viewer_flags["caption"]`` is never touched (the plugin draws text itself,
  after the caption loop, so the backplate can sit behind the text);
* a viewer without ``_pyrender_viewer`` -- headless, or a Genesis build
  without the plugin hook -- is a silent no-op;
* ``on_draw`` can never raise into the viewer thread: one failure prints once
  and disables the HUD layer.
"""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from legged_gym.scripts.input_source import DriveEnvelope, KeyboardSource, MergedSource
from legged_gym.utils import native_ui
from legged_gym.utils.native_ui import (
    HUD_PAD_X,
    HUD_PAD_Y,
    HUD_REFRESH_HZ,
    NativeArenaUI,
    format_hud_lines,
)


class FakePyrenderViewer:
    """Stands in for ``genesis.ext.pyrender.Viewer``."""

    def __init__(self):
        self.viewer_flags = {"caption": None}
        self.plugins = []
        self.trackball_drags = []
        self.trackball_scrolls = []

    def on_deactivate(self):
        self.deactivated = True

    # Genesis's trackball handlers; the arena shadows these as instance
    # attributes and must chain to them in every mode except gta.
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.trackball_drags.append((dx, dy))
        return "chained"

    def on_mouse_scroll(self, x, y, dx, dy):
        self.trackball_scrolls.append(dy)
        return "chained"


class FakeViewer:
    """Stands in for ``genesis.vis.viewer.Viewer``."""

    def __init__(self, with_pyrender=True):
        self._pyrender_viewer = FakePyrenderViewer() if with_pyrender else None


class FakeTerrainCfg:
    moe_showcase = False
    num_cols = 3
    num_rows = 3


class FakeCfg:
    terrain = FakeTerrainCfg()


class FakeSimulator:
    _scene = None
    _terrain_origins = None

    def __init__(self):
        self.base_pos = np.array([[10.0, 20.0, 0.5]], dtype=np.float32)
        self.base_euler = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)


class FakeEnv:
    """Minimal env surface NativeArenaUI touches during construction."""

    def __init__(self):
        self.cfg = FakeCfg()
        self.simulator = FakeSimulator()
        self.commands = None
        self.camera_poses = []

    def set_viewer_camera(self, eye, lookat):
        self.camera_poses.append((np.asarray(eye), np.asarray(lookat)))


class FakePad:
    """Minimal GamepadSource surface the HUD device label reads."""

    available = True
    name = "Logitech F310 Gamepad (XInput)"
    stick_active = False


def make_arena(with_pyrender=True, tabs=None, levels=None):
    """Build a NativeArenaUI whose viewer is a fake, with keybinds bypassed."""
    source = MergedSource(keyboard=KeyboardSource(DriveEnvelope()), gamepad=None)
    arena = NativeArenaUI(
        FakeEnv(), source,
        tabs=tabs or ["wave", "stairs_up", "gap"],
        levels=levels or [0, 4, 9],
        quiet=True,
    )
    # NativeArenaUI._find_viewer returned None (no scene), so wire the fake in
    # and run only the HUD half of the registration path.
    arena.viewer = FakeViewer(with_pyrender=with_pyrender)
    arena._install_look_handlers()
    arena._install_hud()
    return arena


class TestFormatHudLines(unittest.TestCase):
    def test_four_single_ascii_lines(self):
        lines = format_hud_lines(
            ["wave", "gap"], [0, 9], 1, 1,
            command=(1.5, -0.2, 0.75), fps=48.3, realtime=0.97,
            device="keyboard", following=True,
        )
        self.assertEqual(len(lines), 4)
        for line in lines:
            self.assertNotIn("\n", line)
            self.assertTrue(all(32 <= ord(c) < 127 for c in line), line)

    def test_status_reports_tab_level_and_camera(self):
        status, _, _, _ = format_hud_lines(
            ["wave", "stairs_up", "gap"], [0, 4, 9], 1, 2, following=False)
        self.assertIn("tab 2/3", status)
        self.assertIn("stairs_up", status)
        self.assertIn("L9", status)
        self.assertIn("row 3/3", status)
        self.assertIn("cam free", status)

    def test_hint_is_short_and_points_at_the_terminal_legend(self):
        # The long key list used to sit at BOTTOM_CENTER and painted over the
        # BOTTOM_LEFT telemetry on the same baseline.  The hint is now a stub
        # at TOP_RIGHT; it only stays collision-free with the centered status
        # line BECAUSE it is a handful of characters, so pin that down.
        _, _, _, hint = format_hud_lines(["a"], [0], 0, 0)
        self.assertIn("M", hint)
        self.assertLessEqual(len(hint), 24)
        self.assertNotIn("strafe", hint)

    def test_follow_state_is_shown(self):
        status, _, _, _ = format_hud_lines(["a"], [0], 0, 0, following=True)
        self.assertIn("cam front", status)


class FakeLookPad:
    """Minimal GamepadSource surface MergedSource.look reads."""

    available = True
    name = "Fake F310"

    def __init__(self, look=(0.0, 0.0)):
        self.look = look
        self.stick_active = False

    def poll(self, dt=0.0):
        return (0.0, 0.0, 0.0)

    def drain_events(self):
        return []


def gta_arena(look=(0.0, 0.0)):
    """An arena parked in gta mode with a scripted right-stick position."""
    arena = make_arena()
    arena.source.gamepad = FakeLookPad(look)
    return arena


class TestCameraModes(unittest.TestCase):
    def test_t_cycle_is_gta_rear_front_then_free(self):
        arena = make_arena()
        # gta is the startup default, and because it writes a pose from frame
        # one the arena owns the camera immediately (play.py's --follow_robot
        # must not also write one).
        self.assertEqual(arena.camera_mode, "gta")
        self.assertTrue(arena.camera_override_active)

        self.assertEqual(arena.next_camera_mode(), "rear")
        self.assertTrue(arena.camera_override_active)
        self.assertTrue(arena.following)
        eye, lookat = arena.env.camera_poses[-1]
        np.testing.assert_allclose(eye, [5.5, 20.0, 2.05])
        np.testing.assert_allclose(lookat, [10.7, 20.0, 0.88])

        self.assertEqual(arena.next_camera_mode(), "front")
        eye, lookat = arena.env.camera_poses[-1]
        np.testing.assert_allclose(eye, [13.0, 20.0, 1.65])
        np.testing.assert_allclose(lookat, [10.0, 20.0, 0.78])

        self.assertEqual(arena.next_camera_mode(), "free")
        self.assertFalse(arena.following)
        self.assertEqual(arena.next_camera_mode(), "gta")  # wraps back round
        self.assertTrue(arena.following)                   # gta writes poses

    def test_follow_camera_heavily_filters_vertical_body_bounce(self):
        arena = make_arena()
        arena.next_camera_mode()  # rear; initializes the camera anchor at z=0.5
        arena.env.simulator.base_pos[0, 2] = 1.5
        self.assertTrue(arena.update_follow_camera())
        eye, lookat = arena.env.camera_poses[-1]
        # A one-metre instantaneous body bounce leaves the world-level camera
        # rig untouched; the robot moves inside the frame instead.
        np.testing.assert_allclose(eye, [5.5, 20.0, 2.05])
        np.testing.assert_allclose(lookat, [10.7, 20.0, 0.88])

    def test_follow_camera_stride_skips_only_non_forced_updates(self):
        arena = make_arena()
        arena._follow_camera_stride = 2
        arena.next_camera_mode()  # immediate forced rear pose
        poses_after_switch = len(arena.env.camera_poses)

        self.assertFalse(arena.update_follow_camera())
        self.assertEqual(len(arena.env.camera_poses), poses_after_switch)
        self.assertTrue(arena.update_follow_camera())
        self.assertEqual(len(arena.env.camera_poses), poses_after_switch + 1)

    def test_follow_camera_ignores_tiny_yaw_jitter_then_turns_slowly(self):
        arena = make_arena()
        arena.next_camera_mode()
        initial_eye, _ = arena.env.camera_poses[-1]

        arena.env.simulator.base_euler[0, 2] = 0.01
        self.assertTrue(arena.update_follow_camera())
        tiny_jitter_eye, _ = arena.env.camera_poses[-1]
        np.testing.assert_allclose(tiny_jitter_eye, initial_eye)

        arena.env.simulator.base_euler[0, 2] = 0.4
        self.assertTrue(arena.update_follow_camera())
        turning_eye, _ = arena.env.camera_poses[-1]
        # The camera follows the real turn, but it has accepted only 5% of it.
        self.assertLess(turning_eye[1], 20.0)
        self.assertGreater(turning_eye[1], 19.85)

    def test_gta_snaps_behind_the_robot_on_the_first_update(self):
        arena = gta_arena()
        self.assertTrue(arena.update_follow_camera(dt=0.0))
        eye, lookat = arena.env.camera_poses[-1]
        # Robot at (10, 20, 0.5) facing +x, so the eye sits at -x from it, one
        # default distance away, lifted by the default pitch.
        horizontal = native_ui.GTA_DISTANCE * np.cos(native_ui.GTA_PITCH)
        np.testing.assert_allclose(
            eye,
            [10.0 - horizontal, 20.0,
             0.5 + native_ui.GTA_DISTANCE * np.sin(native_ui.GTA_PITCH)],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            lookat, [10.0, 20.0, 0.5 + native_ui.GTA_LOOKAT_HEIGHT], atol=1e-6)

    def test_gta_azimuth_is_absolute_and_never_follows_the_robot(self):
        """The whole point of the mode: orbit round and see the robot's front."""
        arena = gta_arena()
        arena.update_follow_camera(dt=0.0)
        eye_before, _ = arena.env.camera_poses[-1]

        # The robot turns a full quarter turn; the camera must not swing with
        # it (rear mode would).
        for yaw in (0.4, 0.9, 1.57):
            arena.env.simulator.base_euler[0, 2] = yaw
            arena.update_follow_camera(dt=1 / 60.0)
        eye_after, _ = arena.env.camera_poses[-1]
        np.testing.assert_allclose(eye_after, eye_before, atol=1e-6)

    def test_right_stick_x_orbits_at_the_documented_rate(self):
        arena = gta_arena(look=(1.0, 0.0))
        arena.update_follow_camera(dt=0.0)   # snap; look is ignored this frame
        start = arena._camera_azimuth
        arena.update_follow_camera(dt=0.1)
        # Stick right orbits the view right, i.e. the eye bearing decreases.
        self.assertAlmostEqual(
            arena._camera_azimuth, start - native_ui.GTA_AZIMUTH_RATE * 0.1,
            places=6)

    def test_right_stick_y_pitches_and_clamps_at_both_ends(self):
        arena = gta_arena(look=(0.0, -1.0))  # stick up
        arena.update_follow_camera(dt=0.0)
        for _ in range(200):
            arena.update_follow_camera(dt=0.05)
        self.assertAlmostEqual(arena._camera_pitch, native_ui.GTA_PITCH_LIMITS[1])
        arena.source.gamepad.look = (0.0, 1.0)  # stick down
        for _ in range(200):
            arena.update_follow_camera(dt=0.05)
        self.assertAlmostEqual(arena._camera_pitch, native_ui.GTA_PITCH_LIMITS[0])

    def test_look_dt_is_clamped_so_a_stall_cannot_whip_the_camera(self):
        arena = gta_arena(look=(1.0, 0.0))
        arena.update_follow_camera(dt=0.0)
        start = arena._camera_azimuth
        arena.update_follow_camera(dt=5.0)  # a five-second hitch
        self.assertAlmostEqual(
            arena._camera_azimuth,
            start - native_ui.GTA_AZIMUTH_RATE * native_ui.GTA_MAX_LOOK_DT,
            places=6)

    def test_snap_recentres_behind_the_robot_on_demand(self):
        arena = gta_arena(look=(1.0, 0.0))
        arena.update_follow_camera(dt=0.0)
        for _ in range(20):
            arena.update_follow_camera(dt=0.05)
        arena.env.simulator.base_euler[0, 2] = 1.0
        self.assertNotAlmostEqual(arena._camera_azimuth, 1.0 + np.pi, places=3)

        arena.apply("camera_snap")
        arena.update_follow_camera(dt=0.05)
        self.assertAlmostEqual(arena._camera_azimuth, 1.0 - np.pi, places=6)

    def test_snap_is_a_noop_outside_gta(self):
        arena = make_arena()
        arena.next_camera_mode()  # rear
        self.assertFalse(arena.snap_camera_behind())

    def test_teleport_recentres_but_driving_does_not(self):
        arena = gta_arena()
        arena.update_follow_camera(dt=0.0)
        arena._camera_azimuth = 0.25

        # Driving is not a teleport: nothing recentres.
        arena.env.simulator.base_pos[0, :2] += 0.2
        arena.update_follow_camera(dt=1 / 60.0)
        self.assertAlmostEqual(arena._camera_azimuth, 0.25)

        with mock.patch.object(native_ui, "teleport_env_to_taxonomy_tile",
                               return_value=True):
            arena._goto(0, 0)
        # A teleport is the one place recentring is wanted: the user asked to
        # be put somewhere else and needs to see which way the robot faces.
        self.assertAlmostEqual(arena._camera_azimuth, np.pi)


class TestGtaMouseLook(unittest.TestCase):
    """Viewer-thread callbacks may only stash deltas; the rig moves on poll."""

    def test_drag_in_gta_is_swallowed_and_applied_on_the_sim_thread(self):
        arena = gta_arena()
        arena.update_follow_camera(dt=0.0)
        start_azimuth = arena._camera_azimuth
        window = arena.viewer._pyrender_viewer

        self.assertIs(window.on_mouse_drag(0, 0, 30.0, 10.0, 1, 0), True)
        self.assertEqual(window.trackball_drags, [])   # trackball not chained
        # Nothing has moved yet: the callback only accumulated pixels.
        self.assertEqual(arena._camera_azimuth, start_azimuth)

        arena.update_follow_camera(dt=1 / 60.0)
        self.assertAlmostEqual(
            arena._camera_azimuth,
            start_azimuth - 30.0 * native_ui.GTA_MOUSE_AZIMUTH_PER_PX,
            places=6)
        self.assertAlmostEqual(
            arena._camera_pitch,
            native_ui.GTA_PITCH + 10.0 * native_ui.GTA_MOUSE_PITCH_PER_PX,
            places=6)

    def test_drag_outside_gta_chains_to_the_genesis_trackball(self):
        arena = make_arena()
        arena.next_camera_mode()  # rear
        window = arena.viewer._pyrender_viewer
        self.assertEqual(window.on_mouse_drag(0, 0, 5.0, 5.0, 1, 0), "chained")
        self.assertEqual(window.trackball_drags, [(5.0, 5.0)])
        self.assertEqual(window.on_mouse_scroll(0, 0, 0.0, 1.0), "chained")
        self.assertEqual(window.trackball_scrolls, [1.0])

    def test_scroll_zooms_and_clamps(self):
        arena = gta_arena()
        arena.update_follow_camera(dt=0.0)
        window = arena.viewer._pyrender_viewer

        self.assertIs(window.on_mouse_scroll(0, 0, 0.0, 1.0), True)
        arena.update_follow_camera(dt=1 / 60.0)
        self.assertAlmostEqual(
            arena._camera_distance,
            native_ui.GTA_DISTANCE * (1.0 - native_ui.GTA_SCROLL_STEP),
            places=6)

        for _ in range(200):
            window.on_mouse_scroll(0, 0, 0.0, 1.0)
            arena.update_follow_camera(dt=1 / 60.0)
        self.assertAlmostEqual(arena._camera_distance,
                               native_ui.GTA_DISTANCE_LIMITS[0])
        for _ in range(200):
            window.on_mouse_scroll(0, 0, 0.0, -1.0)
            arena.update_follow_camera(dt=1 / 60.0)
        self.assertAlmostEqual(arena._camera_distance,
                               native_ui.GTA_DISTANCE_LIMITS[1])

    def test_unregister_restores_the_original_handlers(self):
        arena = gta_arena()
        window = arena.viewer._pyrender_viewer
        self.assertIn("on_mouse_drag", vars(window))
        arena.unregister()
        self.assertNotIn("on_mouse_drag", vars(window))
        self.assertNotIn("on_mouse_scroll", vars(window))
        self.assertEqual(window.on_mouse_drag(0, 0, 1.0, 1.0, 1, 0), "chained")

    def test_install_is_a_noop_without_a_pyrender_window(self):
        arena = make_arena(with_pyrender=False)
        arena._install_look_handlers()  # must not raise
        self.assertIsNone(arena._look_window)


class TestHudCameraDetail(unittest.TestCase):
    def test_gta_status_reports_the_orbit_distance(self):
        arena = gta_arena()
        self.assertEqual(arena.camera_detail(), "d4.5")
        status = format_hud_lines(["a"], [0], 0, 0, camera_mode="gta",
                                  camera_detail=arena.camera_detail())[0]
        self.assertIn("cam gta d4.5", status)
        self.assertTrue(all(32 <= ord(c) < 127 for c in status))

    def test_other_modes_have_no_detail(self):
        arena = make_arena()
        arena.next_camera_mode()  # rear
        self.assertEqual(arena.camera_detail(), "")
        status = format_hud_lines(["a"], [0], 0, 0, camera_mode="rear")[0]
        self.assertIn("cam rear", status)


class TestMoreHudLines(unittest.TestCase):
    def test_command_is_signed_and_device_shown(self):
        _, telem, _, _ = format_hud_lines(
            ["a"], [0], 0, 0, command=(-1.0, 0.25, -0.5), device="pad F310")
        self.assertIn("-1.00", telem)
        self.assertIn("+0.25", telem)
        self.assertIn("-0.50", telem)
        self.assertIn("pad F310", telem)

    def test_fps_and_realtime_are_optional(self):
        _, telem, achieved, _ = format_hud_lines(["a"], [0], 0, 0)
        self.assertNotIn("FPS", telem)
        self.assertNotIn("FPS", achieved)
        _, _, achieved2, _ = format_hud_lines(["a"], [0], 0, 0, fps=30.0, realtime=0.6)
        self.assertIn("FPS", achieved2)
        self.assertIn("0.60x RT", achieved2)

    def test_non_ascii_input_is_scrubbed_not_raised(self):
        # A tab name coming from a config could contain anything; it must never
        # reach the glyph map unfiltered.
        status, _, _, _ = format_hud_lines(["stairs→up"], [0], 0, 0)
        self.assertTrue(all(32 <= ord(c) < 127 for c in status))

    def test_out_of_range_indices_are_clamped(self):
        status, _, _, _ = format_hud_lines(["a", "b"], [0, 4], 99, -5)
        self.assertIn("tab 2/2", status)
        self.assertIn("row 1/2", status)


class TestHudFontGuard(unittest.TestCase):
    """The HUD must never install a font/size that crashes the draw loop.

    pyrender builds a GL texture for every glyph chr(0..127) on first draw.  A
    glyph that rasterises exactly one pixel tall is squeezed to a 1-D array and
    Texture._add_to_context raises IndexError on source.shape[1] -- on the
    viewer thread, killing rendering.  '_' in UbuntuMono at 20 pt does exactly
    that on this Genesis build, so it is the regression anchor.  The plugin
    path draws through the same renderer.render_text, so the probe still
    applies unchanged.
    """

    def test_known_bad_pair_is_rejected(self):
        self.assertFalse(native_ui.hud_font_is_safe("UbuntuMono-Regular", 20))

    def test_shipped_candidates_are_all_usable(self):
        for name, pt in native_ui.HUD_FONT_CANDIDATES:
            with self.subTest(font=name, pt=pt):
                self.assertTrue(native_ui.hud_font_is_safe(name, pt))

    def test_missing_font_is_rejected_not_raised(self):
        self.assertFalse(native_ui.hud_font_is_safe("NoSuchFontHere", 26))

    def test_installed_plugin_uses_a_safe_font(self):
        arena = make_arena()
        plugin = arena._hud_plugin
        self.assertIsNotNone(plugin)
        self.assertTrue(
            native_ui.hud_font_is_safe(plugin._font_name, plugin._font_pt))

    def test_hud_is_skipped_when_no_font_is_safe(self):
        original = native_ui.hud_font_is_safe
        native_ui.hud_font_is_safe = lambda *a, **k: False
        try:
            arena = make_arena()
            self.assertIsNone(arena._hud_plugin)
            self.assertIsNone(arena._hud_lines)
            # No plugin may be appended, and the viewer's caption slot must be
            # left exactly as it was found.
            self.assertEqual(arena.viewer._pyrender_viewer.plugins, [])
            self.assertIsNone(
                arena.viewer._pyrender_viewer.viewer_flags["caption"])
        finally:
            native_ui.hud_font_is_safe = original


class TestHudInstall(unittest.TestCase):
    def test_plugin_is_appended_to_the_viewer(self):
        arena = make_arena()
        window = arena.viewer._pyrender_viewer
        self.assertEqual(window.plugins, [arena._hud_plugin])
        # The plugin reads the exact list update_hud swaps strings into.
        self.assertIs(arena._hud_plugin._lines, arena._hud_lines)

    def test_caption_slot_is_left_untouched(self):
        # The plugin draws both backplate and text AFTER the caption loop, so
        # viewer_flags["caption"] is no longer the HUD's business.
        arena = make_arena()
        self.assertIsNone(arena.viewer._pyrender_viewer.viewer_flags["caption"])

    def test_anchors_are_distinct_and_avoid_genesis_own_text(self):
        from genesis.ext.pyrender.constants import TextAlign

        arena = make_arena()
        anchors = arena._hud_plugin._anchors
        # Commanded and achieved intentionally share the left column; the
        # plugin offsets them vertically without using an unsafe newline.
        self.assertEqual(len(anchors), 4)
        # Genesis draws its key-instruction list at TOP_LEFT and its transient
        # messages at BOTTOM_RIGHT (_render_help_text); do not collide.
        self.assertNotIn(TextAlign.TOP_LEFT, anchors)
        self.assertNotIn(TextAlign.BOTTOM_RIGHT, anchors)

    def test_line_anchor_layout_keeps_every_line_in_its_own_band(self):
        # The bug this guards: telemetry (bottom-left) and the old long hint
        # (bottom-center) shared the bottom band and overlapped.  The hint now
        # lives at the top right, far from the centered status line.
        from genesis.ext.pyrender.constants import TextAlign

        arena = make_arena()
        anchors = arena._hud_plugin._anchors
        self.assertEqual(anchors[0], TextAlign.TOP_CENTER)    # status
        self.assertEqual(anchors[1], TextAlign.BOTTOM_LEFT)   # telemetry
        self.assertEqual(anchors[2], TextAlign.BOTTOM_LEFT)   # achieved
        self.assertEqual(anchors[3], TextAlign.TOP_RIGHT)     # short hint

    def test_missing_pyrender_viewer_is_a_silent_noop(self):
        arena = make_arena(with_pyrender=False)
        self.assertIsNone(arena._hud_plugin)
        self.assertFalse(arena.hud_due())
        # Must not raise, and must report "nothing rebuilt".
        self.assertFalse(arena.update_hud(command=(1.0, 0.0, 0.0), force=True))

    def test_no_viewer_at_all_is_a_silent_noop(self):
        source = MergedSource(keyboard=KeyboardSource(), gamepad=None)
        arena = NativeArenaUI(FakeEnv(), source, tabs=["a"], levels=[0], quiet=True)
        self.assertIsNone(arena.viewer)
        self.assertIsNone(arena._hud_plugin)
        self.assertFalse(arena.update_hud(force=True))


class TestHudUpdate(unittest.TestCase):
    def test_text_updates_in_place_without_reallocating_the_list(self):
        arena = make_arena()
        lines = arena._hud_lines
        arena.update_hud(command=(1.0, 0.0, 0.0), fps=50.0, realtime=1.0, force=True)
        self.assertIs(arena._hud_lines, lines)
        self.assertIs(arena._hud_plugin._lines, lines)
        self.assertIn("+1.00", lines[1])

    def test_tab_change_is_reflected_in_the_status_line(self):
        arena = make_arena()
        arena.update_hud(force=True)
        self.assertIn("wave", arena._hud_lines[0])

        arena.tab_index = 1
        arena.update_hud(force=True)
        self.assertIn("stairs_up", arena._hud_lines[0])
        self.assertNotIn("wave", arena._hud_lines[0])

    def test_level_change_is_reflected_in_the_status_line(self):
        arena = make_arena()
        arena.level_index = 2
        arena.update_hud(force=True)
        self.assertIn("L9", arena._hud_lines[0])
        self.assertIn("row 3/3", arena._hud_lines[0])

    def test_command_change_is_reflected_in_the_telemetry_line(self):
        arena = make_arena()
        arena.update_hud(command=(1.25, 0.0, -0.5), force=True)
        self.assertIn("+1.25", arena._hud_lines[1])
        self.assertIn("-0.50", arena._hud_lines[1])

    def test_update_is_throttled_unless_forced(self):
        arena = make_arena()
        arena.update_hud(command=(1.0, 0.0, 0.0), force=True)
        first = arena._hud_lines[1]
        # Immediately afterwards the throttle window is still open.
        self.assertFalse(arena.hud_due())
        self.assertFalse(arena.update_hud(command=(9.0, 0.0, 0.0)))
        self.assertEqual(arena._hud_lines[1], first)
        # Rewinding past the window lets the next call through.
        arena._hud_last -= (1.0 / HUD_REFRESH_HZ) * 2
        self.assertTrue(arena.hud_due())
        self.assertTrue(arena.update_hud(command=(9.0, 0.0, 0.0)))
        self.assertNotEqual(arena._hud_lines[1], first)

    def test_every_rendered_line_stays_single_line_ascii(self):
        arena = make_arena(tabs=["wave", "stépping"], levels=[0, 4])
        for tab in range(2):
            for level in range(2):
                arena.tab_index, arena.level_index = tab, level
                arena.update_hud(command=(1.0, -1.0, 1.0), fps=12.3,
                                 realtime=0.25, force=True)
                for text in arena._hud_lines:
                    self.assertNotIn("\n", text)
                    self.assertTrue(all(32 <= ord(c) < 127 for c in text), text)


class TestHudTeardown(unittest.TestCase):
    def test_unregister_removes_the_plugin_and_leaves_caption_alone(self):
        arena = make_arena()
        window = arena.viewer._pyrender_viewer
        self.assertIn(arena._hud_plugin, window.plugins)
        arena.unregister()
        self.assertEqual(window.plugins, [])
        self.assertIsNone(window.viewer_flags["caption"])
        self.assertIsNone(arena._hud_plugin)
        self.assertIsNone(arena._hud_lines)

    def test_unregister_is_idempotent(self):
        arena = make_arena()
        arena.unregister()
        arena.unregister()  # must not raise


class TestHudDeviceLabel(unittest.TestCase):
    def test_pad_named_but_keyboard_credited_while_sticks_centred(self):
        arena = make_arena()  # MergedSource defaults to gamepad enabled
        arena.source.gamepad = FakePad()
        arena.update_hud(force=True)
        self.assertIn("keyboard", arena._hud_lines[1])
        self.assertIn("F310", arena._hud_lines[1])

    def test_pad_credited_while_a_stick_is_off_centre(self):
        arena = make_arena()
        pad = FakePad()
        pad.stick_active = True
        arena.source.gamepad = pad
        arena.update_hud(force=True)
        self.assertIn("pad ", arena._hud_lines[1])
        self.assertIn("F310", arena._hud_lines[1])

    def test_pad_present_but_toggled_off_reads_pad_off(self):
        arena = make_arena()
        arena.source.gamepad = FakePad()
        arena.source._gamepad_enabled = False  # gamepad_enabled is a property
        arena.update_hud(force=True)
        self.assertIn("keyboard (pad off)", arena._hud_lines[1])

    def test_no_pad_at_all_is_plain_keyboard(self):
        arena = make_arena()
        arena.update_hud(force=True)
        self.assertIn("in: keyboard", arena._hud_lines[1])
        self.assertNotIn("pad", arena._hud_lines[1])

    def test_missing_toggle_attribute_keeps_crediting_the_pad(self):
        # input_source gained gamepad_enabled in the same change as the J key;
        # against a source that predates the toggle the label must fall back
        # to the old behaviour (pad credited whenever one is available).
        class StubSource:
            gamepad = FakePad()

        arena = make_arena()
        arena.source = StubSource()  # no gamepad_enabled attribute at all
        arena.update_hud(force=True)
        self.assertIn("F310", arena._hud_lines[1])


class TestGamepadToggle(unittest.TestCase):
    def _install_toggle(self, arena, state):
        """Stand-in for MergedSource.toggle_gamepad (owned by input_source).

        The real method also flips the read-only ``gamepad_enabled`` property
        (backing field ``_gamepad_enabled``) and, when enabling, runs a pyglet
        controller rescan -- the mock mirrors the flag flip but stays hermetic.
        """
        def toggle():
            arena.source._gamepad_enabled = state
            return state
        arena.source.toggle_gamepad = toggle

    def test_keymap_and_event_are_registered(self):
        self.assertEqual(
            native_ui.ACTION_KEYMAP["arena_gamepad_toggle"], ("J", "gamepad_toggle"))
        self.assertIn("gamepad_toggle", native_ui.ARENA_EVENTS)

    def test_toggle_on_announces_the_pad_and_refreshes_the_hud(self):
        arena = make_arena()
        arena.source.gamepad = FakePad()
        self._install_toggle(arena, True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            arena.apply("gamepad_toggle")
        self.assertIn("gamepad ON", out.getvalue())
        self.assertIn("F310", out.getvalue())
        # apply forces a HUD refresh: idle pad -> keyboard credited, pad named.
        self.assertIn("keyboard (pad:", arena._hud_lines[1])
        self.assertIn("F310", arena._hud_lines[1])

    def test_toggle_off_falls_back_to_keyboard_only(self):
        arena = make_arena()  # gamepad_enabled defaults to True
        arena.source.gamepad = FakePad()
        self._install_toggle(arena, False)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            arena.apply("gamepad_toggle")
        self.assertIn("gamepad OFF (keyboard only)", out.getvalue())
        self.assertIn("keyboard (pad off)", arena._hud_lines[1])

    def test_toggle_on_without_a_pad_claims_no_name(self):
        arena = make_arena()
        self._install_toggle(arena, True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            arena.apply("gamepad_toggle")
        self.assertIn("gamepad ON", out.getvalue())
        self.assertNotIn("F310", out.getvalue())
        self.assertIn("in: keyboard", arena._hud_lines[1])

    def test_drain_survives_a_failing_toggle(self):
        # A source whose toggle_gamepad raises (e.g. a stale input_source
        # without the method) must degrade to a console error, not a crash.
        arena = make_arena()
        def boom():
            raise AttributeError("toggle_gamepad")
        arena.source.toggle_gamepad = boom
        arena.source.push_event("gamepad_toggle")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            events = arena.drain_and_apply()
        self.assertEqual(events, ["gamepad_toggle"])
        self.assertIn("failed", out.getvalue())


class TestHudPluginDraw(unittest.TestCase):
    """The viewer-thread surface: no GL here, only the failure contracts."""

    def test_plugin_surface_matches_what_the_viewer_calls(self):
        # viewer.py calls exactly on_draw / update_on_sim_step / on_close on
        # entries of window.plugins; the latter two are deliberate no-ops.
        arena = make_arena()
        plugin = arena._hud_plugin
        plugin.update_on_sim_step()
        plugin.on_close()

    def test_on_draw_never_raises_and_disables_itself_after_one_failure(self):
        arena = make_arena()
        plugin = arena._hud_plugin
        plugin._window = object()  # no _renderer / _location_to_x_y: explode
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            plugin.on_draw()
            plugin.on_draw()  # second failure must stay silent
        self.assertTrue(plugin._broken)
        self.assertEqual(out.getvalue().count("disabled after draw failure"), 1)

    def test_measure_falls_back_to_a_monospace_estimate(self):
        # A renderer without a usable font cache must yield an estimate, not
        # an exception (the real path needs a live GL context, tested nowhere
        # here by design).
        arena = make_arena()
        plugin = arena._hud_plugin
        width, asc, desc = plugin._measure(renderer=object(), text="abc")
        self.assertAlmostEqual(width, 3 * plugin._font_pt * 0.6)
        self.assertAlmostEqual(asc, plugin._font_pt)
        self.assertAlmostEqual(desc, plugin._font_pt * 0.3)

    def _plugin_with_stub_measure(self):
        arena = make_arena()
        plugin = arena._hud_plugin
        plugin._measure = lambda renderer, text: (100.0, 20.0, 6.0)
        return plugin

    def test_backplate_geometry_bottom_left(self):
        from genesis.ext.pyrender.constants import TextAlign

        plugin = self._plugin_with_stub_measure()
        x, y, w, h = plugin._backplate(object(), 20.0, 20.0, "txt",
                                       TextAlign.BOTTOM_LEFT)
        # BOTTOM_*: x pins the left edge, y is the baseline; the rect covers
        # descender to ascender plus padding on every side.
        self.assertAlmostEqual(x, 20.0 - HUD_PAD_X)
        self.assertAlmostEqual(y, 20.0 - 6.0 - HUD_PAD_Y)
        self.assertAlmostEqual(w, 100.0 + 2 * HUD_PAD_X)
        self.assertAlmostEqual(h, 20.0 + 6.0 + 2 * HUD_PAD_Y)

    def test_backplate_geometry_top_right(self):
        from genesis.ext.pyrender.constants import TextAlign

        plugin = self._plugin_with_stub_measure()
        x, y, w, h = plugin._backplate(object(), 1000.0, 760.0, "txt",
                                       TextAlign.TOP_RIGHT)
        # TOP_*: y is the glyph top (baseline at y - ascender); RIGHT: x pins
        # the right edge.
        self.assertAlmostEqual(x, 1000.0 - 100.0 - HUD_PAD_X)
        self.assertAlmostEqual(y, 760.0 - 20.0 - 6.0 - HUD_PAD_Y)
        self.assertAlmostEqual(w, 100.0 + 2 * HUD_PAD_X)
        self.assertAlmostEqual(h, 20.0 + 6.0 + 2 * HUD_PAD_Y)

    def test_backplate_geometry_top_center(self):
        from genesis.ext.pyrender.constants import TextAlign

        plugin = self._plugin_with_stub_measure()
        x, y, w, h = plugin._backplate(object(), 500.0, 760.0, "txt",
                                       TextAlign.TOP_CENTER)
        self.assertAlmostEqual(x, 500.0 - 50.0 - HUD_PAD_X)
        self.assertAlmostEqual(y, 760.0 - 20.0 - 6.0 - HUD_PAD_Y)


class TestArenaLegend(unittest.TestCase):
    def test_legend_lists_j_and_the_autodetect_wording(self):
        text = native_ui.format_arena_legend(["wave"], [0])
        self.assertIn("toggle gamepad on/off", text)
        self.assertIn("auto-detected", text)
        self.assertIn("~2 s", text)
        # The gamepad controls are always listed now (auto-detection means a
        # pad can show up at any time), even with none connected.
        self.assertIn("connected: none", text)
        self.assertIn("left stick", text)

    def test_connected_pad_is_named(self):
        text = native_ui.format_arena_legend(["wave"], [0], gamepad="F310")
        self.assertIn("connected: F310", text)


if __name__ == "__main__":
    unittest.main()
