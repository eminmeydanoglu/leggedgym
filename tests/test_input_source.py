"""Unit tests for the play-arena input layer (GPU-free, display-free)."""
from __future__ import annotations

import inspect
import io
import math
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# input_source itself is stdlib-only, but importing it walks legged_gym/__init__,
# which hard-fails without a simulator selected.  No GPU or display is touched.
os.environ.setdefault("SIMULATOR", "genesis")

import legged_gym.scripts.input_source as input_source  # noqa: E402
from legged_gym.scripts.input_source import (  # noqa: E402
    DEFAULT_TAU_ACCEL,
    DEFAULT_TAU_DECEL,
    DriveEnvelope,
    GamepadSource,
    KeyboardSource,
    MergedSource,
    apply_deadzone,
    apply_expo,
    ramp_towards,
)


class FakeController:
    """Stand-in for a pyglet Controller: pollable attributes, no device."""

    def __init__(self, leftx=0.0, lefty=0.0, rightx=0.0, **buttons):
        self.name = "Fake F310"
        self.device = None
        self.leftx = leftx
        self.lefty = lefty
        self.rightx = rightx
        self.open_calls = 0
        for name in ("a", "b", "x", "y", "leftshoulder", "rightshoulder",
                     "start", "back"):
            setattr(self, name, bool(buttons.get(name, False)))

    def open(self):
        self.open_calls += 1

    def close(self):
        pass


def make_pad(**kwargs):
    controller = FakeController(**kwargs)
    return GamepadSource(controller=controller, auto_open=False), controller


def fake_pyglet_input(controllers):
    """A stand-in ``pyglet.input`` module with scripted enumeration results."""
    module = types.ModuleType("pyglet.input")
    module.get_controllers = lambda: list(controllers)
    return module


class FakeGamepad:
    """Stand-in for GamepadSource: records rescans, touches no pyglet/hardware.

    ``becomes_available`` scripts what a real rescan would discover: set it to
    simulate a pad that has just been plugged in.
    """

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.available = False
        self.becomes_available = False
        self.stick_active = False
        self.name = ""
        self.cmd = (0.0, 0.0, 0.0)
        self.rescan_calls = 0
        self.poll_calls = 0
        FakeGamepad.instances.append(self)

    def rescan(self):
        self.rescan_calls += 1
        if self.becomes_available:
            self.available = True
        return self.available

    def poll(self, dt=0.0):
        self.poll_calls += 1
        return self.cmd

    def drain_events(self):
        return []


class FakeTime:
    """Deterministic monotonic clock, so throttle tests need no sleeping."""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class TestEnvelope(unittest.TestCase):
    def test_defaults_match_viser_values(self):
        e = DriveEnvelope()
        self.assertEqual((e.forward, e.backward, e.lateral, e.yaw),
                         (1.5, 1.0, 0.7, 1.5))

    def test_clamp(self):
        e = DriveEnvelope()
        self.assertEqual(e.clamp(9.0, 9.0, 9.0), (1.5, 0.7, 1.5))
        self.assertEqual(e.clamp(-9.0, -9.0, -9.0), (-1.0, -0.7, -1.5))

    def test_scale_is_asymmetric_in_x(self):
        e = DriveEnvelope()
        self.assertAlmostEqual(e.scale(1.0, 0.0, 0.0)[0], 1.5)
        self.assertAlmostEqual(e.scale(-1.0, 0.0, 0.0)[0], -1.0)
        self.assertAlmostEqual(e.scale(0.0, 1.0, 0.0)[1], 0.7)
        self.assertAlmostEqual(e.scale(0.0, 0.0, -1.0)[2], -1.5)

    def test_scale_output_always_inside_envelope(self):
        e = DriveEnvelope()
        for ax in (-2.0, -1.0, 0.0, 1.0, 2.0):
            vx, vy, yaw = e.scale(ax, 2.0, -2.0)
            self.assertLessEqual(vx, e.forward + 1e-9)
            self.assertGreaterEqual(vx, -e.backward - 1e-9)
            self.assertLessEqual(abs(vy), e.lateral + 1e-9)
            self.assertLessEqual(abs(yaw), e.yaw + 1e-9)


class TestResponseCurves(unittest.TestCase):
    def test_deadzone_kills_drift(self):
        # F310 resting drift must produce exactly zero, not "almost zero".
        for drift in (0.0, 0.05, 0.1, -0.11, 0.12):
            self.assertEqual(apply_deadzone(drift, 0.12), 0.0)

    def test_deadzone_renormalises(self):
        # Just past the deadzone -> just past zero (no step), full -> full.
        self.assertAlmostEqual(apply_deadzone(0.12 + 1e-9, 0.12), 0.0, places=6)
        self.assertAlmostEqual(apply_deadzone(1.0, 0.12), 1.0)
        self.assertAlmostEqual(apply_deadzone(-1.0, 0.12), -1.0)
        self.assertAlmostEqual(apply_deadzone(0.56, 0.12), (0.56 - 0.12) / 0.88)

    def test_deadzone_zero_is_pass_through_and_clamped(self):
        self.assertAlmostEqual(apply_deadzone(0.5, 0.0), 0.5)
        self.assertAlmostEqual(apply_deadzone(3.0, 0.0), 1.0)

    def test_expo_preserves_endpoints_and_softens_centre(self):
        self.assertAlmostEqual(apply_expo(1.0, 0.35), 1.0)
        self.assertAlmostEqual(apply_expo(-1.0, 0.35), -1.0)
        self.assertAlmostEqual(apply_expo(0.0, 0.35), 0.0)
        self.assertLess(apply_expo(0.5, 0.35), 0.5)
        self.assertAlmostEqual(apply_expo(0.5, 0.0), 0.5)  # linear
        self.assertAlmostEqual(apply_expo(0.5, 1.0), 0.125)  # pure cubic

    def test_expo_is_monotonic(self):
        prev = -1.0
        for i in range(101):
            v = apply_expo(-1.0 + i * 0.02, 0.35)
            self.assertGreaterEqual(v + 1e-12, prev)
            prev = v

    def test_ramp_accelerates_faster_than_it_decelerates(self):
        up = ramp_towards(0.0, 1.0, 0.05)
        down = ramp_towards(1.0, 0.0, 0.05)
        self.assertGreater(up, 1.0 - down)  # closed more of the gap going up
        self.assertAlmostEqual(up, 1.0 - math.exp(-0.05 / DEFAULT_TAU_ACCEL))
        self.assertAlmostEqual(down, math.exp(-0.05 / DEFAULT_TAU_DECEL))

    def test_ramp_zero_dt_is_a_noop(self):
        self.assertEqual(ramp_towards(0.3, 1.0, 0.0), 0.3)


class TestKeyboardSource(unittest.TestCase):
    def test_press_release_is_known_not_inferred(self):
        kb = KeyboardSource()
        self.assertFalse(kb.active)
        kb.press("forward")
        self.assertTrue(kb.held()["forward"])
        # No repeat events, no grace window: the key stays held indefinitely.
        for _ in range(1000):
            kb.poll(0.016)
        self.assertAlmostEqual(kb.poll(0.0)[0], DriveEnvelope().forward, places=3)
        kb.release("forward")
        self.assertFalse(kb.held()["forward"])

    def test_ramp_up_then_coast_to_zero(self):
        kb = KeyboardSource()
        kb.press("forward")
        vx = 0.0
        for _ in range(60):
            vx = kb.poll(1 / 60.0)[0]
        self.assertGreater(vx, 1.4)
        kb.release("forward")
        for _ in range(120):
            vx = kb.poll(1 / 60.0)[0]
        self.assertLess(abs(vx), 0.01)

    def test_decel_is_slower_than_accel(self):
        kb = KeyboardSource()
        kb.press("forward")
        after_accel = kb.poll(0.05)[0]
        kb2 = KeyboardSource()
        kb2.press("forward")
        for _ in range(200):
            kb2.poll(0.05)
        top = kb2.poll(0.0)[0]
        kb2.release("forward")
        after_decel = top - kb2.poll(0.05)[0]
        self.assertGreater(after_accel, after_decel)

    def test_multi_axis_chord(self):
        """The whole point of real key-up: chords just work, no sticky latch."""
        kb = KeyboardSource()
        kb.press("forward")
        kb.press("yaw_left")
        kb.press("strafe_right")
        vx, vy, yaw = kb.target()
        self.assertAlmostEqual(vx, 1.5)
        self.assertAlmostEqual(vy, -0.7)
        self.assertAlmostEqual(yaw, 1.5)
        kb.release("yaw_left")
        vx, vy, yaw = kb.target()
        self.assertAlmostEqual(vx, 1.5)  # forward survives releasing yaw
        self.assertAlmostEqual(yaw, 0.0)

    def test_opposite_keys_cancel(self):
        kb = KeyboardSource()
        kb.press("forward")
        kb.press("backward")
        self.assertAlmostEqual(kb.target()[0], 1.5 - 1.0)
        kb.press("strafe_left")
        kb.press("strafe_right")
        self.assertAlmostEqual(kb.target()[1], 0.0)

    def test_target_stays_inside_envelope(self):
        kb = KeyboardSource()
        for action in ("forward", "backward", "strafe_left", "strafe_right",
                       "yaw_left", "yaw_right"):
            kb.press(action)
        vx, vy, yaw = kb.target()
        e = kb.envelope
        self.assertLessEqual(vx, e.forward)
        self.assertLessEqual(abs(vy), e.lateral)
        self.assertLessEqual(abs(yaw), e.yaw)

    def test_clear_is_a_hard_stop(self):
        kb = KeyboardSource()
        kb.press("forward")
        for _ in range(60):
            kb.poll(1 / 60.0)
        kb.clear()
        self.assertEqual(kb.poll(0.0), (0.0, 0.0, 0.0))
        self.assertFalse(kb.active)

    def test_unknown_action_is_ignored(self):
        kb = KeyboardSource()
        kb.press("nope")
        kb.release("nope")
        self.assertEqual(kb.target(), (0.0, 0.0, 0.0))


class TestGamepadSource(unittest.TestCase):
    def test_signs_match_the_old_pygame_mapping(self):
        # play.py used vx = -ly, vy = -lx, yaw = -rx.  Stick fully forward on an
        # SDL-style pad is lefty = -1.
        pad, _ = make_pad(lefty=-1.0)
        self.assertAlmostEqual(pad.poll()[0], 1.5)
        pad, _ = make_pad(lefty=1.0)
        self.assertAlmostEqual(pad.poll()[0], -1.0)   # backward envelope
        pad, _ = make_pad(leftx=-1.0)
        self.assertAlmostEqual(pad.poll()[1], 0.7)    # stick left -> +y (left)
        pad, _ = make_pad(rightx=1.0)
        self.assertAlmostEqual(pad.poll()[2], -1.5)   # stick right -> yaw right

    def test_drift_inside_deadzone_produces_no_motion(self):
        pad, _ = make_pad(leftx=0.08, lefty=-0.05, rightx=0.1)
        self.assertEqual(pad.poll(), (0.0, 0.0, 0.0))
        self.assertFalse(pad.stick_active)

    def test_stick_active_flag(self):
        pad, controller = make_pad()
        pad.poll()
        self.assertFalse(pad.stick_active)
        controller.lefty = -0.9
        pad.poll()
        self.assertTrue(pad.stick_active)

    def test_output_clamped_to_envelope(self):
        pad, _ = make_pad(leftx=-5.0, lefty=-5.0, rightx=-5.0)
        vx, vy, yaw = pad.poll()
        self.assertAlmostEqual(vx, 1.5)
        self.assertAlmostEqual(vy, 0.7)
        self.assertAlmostEqual(yaw, 1.5)

    def test_no_controller_is_safe(self):
        pad = GamepadSource(controller=None, auto_open=False)
        self.assertFalse(pad.available)
        self.assertEqual(pad.poll(0.016), (0.0, 0.0, 0.0))
        self.assertEqual(pad.drain_events(), [])

    def test_buttons_emit_edges_once(self):
        pad, controller = make_pad()
        pad.poll()
        controller.a = True
        pad.poll()
        self.assertEqual(pad.drain_events(), ["respawn"])
        pad.poll()  # still held -> no repeat
        self.assertEqual(pad.drain_events(), [])
        controller.a = False
        pad.poll()
        controller.a = True
        pad.poll()
        self.assertEqual(pad.drain_events(), ["respawn"])

    def test_shoulder_buttons_map_to_tabs(self):
        pad, controller = make_pad()
        pad.poll()
        controller.leftshoulder = True
        controller.rightshoulder = True
        pad.poll()
        self.assertEqual(sorted(pad.drain_events()), ["tab_next", "tab_prev"])


class TestMergedSource(unittest.TestCase):
    def test_gamepad_wins_while_sticks_are_off_centre(self):
        pad, controller = make_pad()
        merged = MergedSource(gamepad=pad)
        merged.keyboard.press("forward")
        for _ in range(60):
            merged.poll(1 / 60.0)
        self.assertGreater(merged.poll(0.0)[0], 1.4)  # keyboard owns it

        controller.lefty = 1.0  # stick pulled back
        cmd = merged.poll(1 / 60.0)
        self.assertAlmostEqual(cmd[0], -1.0)  # pad overrides the held key

    def test_keyboard_regains_control_when_sticks_recentre(self):
        pad, controller = make_pad(lefty=-1.0)
        merged = MergedSource(gamepad=pad)
        merged.keyboard.press("yaw_left")
        for _ in range(60):
            merged.poll(1 / 60.0)
        self.assertAlmostEqual(merged.poll(1 / 60.0)[0], 1.5)  # pad forward

        controller.lefty = 0.0
        cmd = merged.poll(1 / 60.0)
        # Keyboard ramp kept advancing under the pad, so yaw is already there.
        self.assertAlmostEqual(cmd[0], 0.0)
        self.assertGreater(cmd[2], 1.4)

    def test_keyboard_only_without_a_pad(self):
        merged = MergedSource()
        self.assertIsNone(merged.gamepad)
        merged.keyboard.press("backward")
        for _ in range(120):
            merged.poll(1 / 60.0)
        self.assertLess(merged.poll(0.0)[0], -0.99)

    def test_events_are_queued_and_drained_once(self):
        merged = MergedSource()
        merged.push_event("tab_next")
        merged.push_event("respawn")
        self.assertEqual(merged.drain_events(), ["tab_next", "respawn"])
        self.assertEqual(merged.drain_events(), [])

    def test_gamepad_events_flow_through_merge(self):
        pad, controller = make_pad()
        merged = MergedSource(gamepad=pad)
        merged.poll(0.016)
        controller.y = True
        merged.poll(0.016)
        self.assertEqual(merged.drain_events(), ["follow_toggle"])

    def test_clear_stops_the_keyboard(self):
        merged = MergedSource()
        merged.keyboard.press("forward")
        for _ in range(60):
            merged.poll(1 / 60.0)
        merged.clear()
        self.assertEqual(merged.poll(0.0), (0.0, 0.0, 0.0))


class TestGamepadRescan(unittest.TestCase):
    """rescan() is the hot-plug entry point: quiet, never raises, re-openable."""

    def test_rescan_is_a_noop_when_a_controller_is_held(self):
        pad, _ = make_pad()
        with mock.patch.object(GamepadSource, "_open_first_controller") as m:
            self.assertTrue(pad.rescan())
        m.assert_not_called()

    def test_rescan_success_prints_the_same_line_as_open(self):
        pad = GamepadSource(controller=None, auto_open=False)
        fake_input = fake_pyglet_input([FakeController()])
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"pyglet.input": fake_input}):
            with redirect_stdout(buf):
                self.assertTrue(pad.rescan())
        self.assertTrue(pad.available)
        self.assertIn("[input] gamepad: Fake F310", buf.getvalue())

    def test_rescan_failure_is_silent(self):
        # The throttled hot-plug retry must not spam "no gamepad detected".
        pad = GamepadSource(controller=None, auto_open=False)
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"pyglet.input": fake_pyglet_input([])}):
            with redirect_stdout(buf):
                self.assertFalse(pad.rescan())
        self.assertFalse(pad.available)
        self.assertEqual(buf.getvalue(), "")

    def test_open_still_reports_failures(self):
        pad = GamepadSource(controller=None, auto_open=False)
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"pyglet.input": fake_pyglet_input([])}):
            with redirect_stdout(buf):
                self.assertFalse(pad.open())
        self.assertIn("no gamepad detected", buf.getvalue())

    def test_rescan_never_raises(self):
        pad = GamepadSource(controller=None, auto_open=False)
        with mock.patch.object(GamepadSource, "_open_first_controller",
                               side_effect=RuntimeError("usb exploded")):
            self.assertFalse(pad.rescan())

    def test_rescan_recovers_after_pump_drops_the_controller(self):
        pad, _ = make_pad()
        pad.controller = None  # what pump() does on a disconnect exception
        self.assertFalse(pad.available)
        fake_input = fake_pyglet_input([FakeController()])
        with mock.patch.dict(sys.modules, {"pyglet.input": fake_input}):
            self.assertTrue(pad.rescan())
        self.assertTrue(pad.available)


class TestMergedGamepadToggle(unittest.TestCase):
    """Runtime on/off switch: poll skips the pad branch while disabled."""

    def test_gamepad_enabled_by_default(self):
        self.assertTrue(MergedSource().gamepad_enabled)

    def test_toggle_flips_the_flag_and_returns_the_new_state(self):
        merged = MergedSource(gamepad=make_pad()[0])
        self.assertIs(merged.toggle_gamepad(), False)
        self.assertFalse(merged.gamepad_enabled)
        self.assertIs(merged.toggle_gamepad(), True)
        self.assertTrue(merged.gamepad_enabled)

    def test_disabled_pad_is_not_polled_and_keyboard_owns_the_command(self):
        pad, _ = make_pad(lefty=1.0)  # stick held back: -1.0 m/s
        merged = MergedSource(gamepad=pad)
        self.assertAlmostEqual(merged.poll(0.016)[0], -1.0)  # pad owns it
        with mock.patch.object(pad, "poll", wraps=pad.poll) as pad_poll:
            merged.set_gamepad_enabled(False)
            merged.keyboard.press("forward")
            for _ in range(60):
                cmd = merged.poll(1 / 60.0)
            pad_poll.assert_not_called()
        self.assertGreater(cmd[0], 1.4)  # keyboard forward, pad ignored
        self.assertFalse(pad.stick_active)  # cleared so the HUD stays honest

    def test_disable_keeps_the_device_open_so_reenable_is_instant(self):
        pad, controller = make_pad(lefty=-1.0)
        merged = MergedSource(gamepad=pad)
        merged.set_gamepad_enabled(False)
        self.assertTrue(pad.available)  # not closed
        self.assertIs(pad.controller, controller)
        with mock.patch.object(pad, "rescan", wraps=pad.rescan) as rescan:
            merged.set_gamepad_enabled(True)  # pad already there: no rescan
            rescan.assert_not_called()
        self.assertAlmostEqual(merged.poll(0.016)[0], 1.5)


class TestMergedHotPlug(unittest.TestCase):
    """poll() picks up pads that appear (or reappear) after startup."""

    def setUp(self):
        FakeGamepad.instances = []

    def _patch_world(self, clock):
        """Deterministic time + scripted GamepadSource construction."""
        return (
            mock.patch.object(input_source, "time", clock),
            mock.patch.object(input_source, "GamepadSource", FakeGamepad),
        )

    def test_lazy_creation_when_no_pad_at_boot(self):
        clock = FakeTime()
        time_patch, pad_patch = self._patch_world(clock)
        with time_patch, pad_patch:
            merged = MergedSource(deadzone=0.2, expo=0.5)
            self.assertIsNone(merged.gamepad)
            merged.poll(0.016)
            self.assertEqual(len(FakeGamepad.instances), 1)
            pad = FakeGamepad.instances[0]
            self.assertIs(merged.gamepad, pad)  # kept even while unavailable
            self.assertEqual(pad.rescan_calls, 1)
            self.assertEqual(pad.kwargs["deadzone"], 0.2)
            self.assertEqual(pad.kwargs["expo"], 0.5)
            self.assertFalse(pad.kwargs["auto_open"])

    def test_rescan_is_throttled_to_once_per_period(self):
        clock = FakeTime()
        time_patch, pad_patch = self._patch_world(clock)
        with time_patch, pad_patch:
            merged = MergedSource()
            merged.poll(0.016)  # t=1000.0: first attempt
            pad = FakeGamepad.instances[0]
            clock.advance(1.0)
            merged.poll(0.016)  # inside the 2 s window: skipped
            self.assertEqual(pad.rescan_calls, 1)
            clock.advance(1.1)  # t=1002.1: past the window
            merged.poll(0.016)
            self.assertEqual(pad.rescan_calls, 2)
            clock.advance(1.9)  # throttle re-armed at the last attempt
            merged.poll(0.016)
            self.assertEqual(pad.rescan_calls, 2)

    def test_hotplug_starts_using_a_pad_that_appears(self):
        clock = FakeTime()
        time_patch, pad_patch = self._patch_world(clock)
        with time_patch, pad_patch:
            merged = MergedSource()
            merged.poll(0.016)
            pad = FakeGamepad.instances[0]
            pad.becomes_available = True  # the replugged pad is now visible
            pad.cmd = (0.5, 0.0, 0.0)
            pad.stick_active = True
            clock.advance(2.5)
            self.assertEqual(merged.poll(0.016), (0.5, 0.0, 0.0))
            self.assertEqual(pad.rescan_calls, 2)
            self.assertTrue(pad.available)
            self.assertGreater(pad.poll_calls, 0)

    def test_enable_forces_an_immediate_rescan(self):
        clock = FakeTime()
        time_patch, pad_patch = self._patch_world(clock)
        with time_patch, pad_patch:
            merged = MergedSource()
            merged.poll(0.016)  # attempt #1
            pad = FakeGamepad.instances[0]
            merged.set_gamepad_enabled(False)
            clock.advance(0.1)  # well inside the throttle window...
            merged.set_gamepad_enabled(True)  # ...but the toggle must not wait
            self.assertEqual(pad.rescan_calls, 2)
            clock.advance(1.0)  # the forced attempt re-arms the throttle
            merged.poll(0.016)
            self.assertEqual(pad.rescan_calls, 2)

    def test_disabled_pad_is_never_rescanned_or_created(self):
        clock = FakeTime()
        time_patch, pad_patch = self._patch_world(clock)
        with time_patch, pad_patch:
            merged = MergedSource()
            merged.set_gamepad_enabled(False)
            merged.poll(0.016)
            clock.advance(10.0)
            merged.poll(0.016)
            self.assertEqual(FakeGamepad.instances, [])
            self.assertIsNone(merged.gamepad)

    def test_reconnect_after_drop_recovers_on_the_same_object(self):
        pad, _ = make_pad(lefty=1.0)
        merged = MergedSource(gamepad=pad)
        self.assertAlmostEqual(merged.poll(0.016)[0], -1.0)
        pad.controller = None  # pump() does this on a disconnect exception
        self.assertFalse(pad.available)
        clock = FakeTime()
        fake_input = fake_pyglet_input([FakeController(lefty=1.0)])
        with mock.patch.object(input_source, "time", clock), \
                mock.patch.dict(sys.modules, {"pyglet.input": fake_input}):
            merged.poll(0.016)  # throttled rescan reopens the same source
        self.assertIs(merged.gamepad, pad)
        self.assertTrue(pad.available)
        self.assertAlmostEqual(merged.poll(0.016)[0], -1.0)


class TestBuildInputSource(unittest.TestCase):
    """The factory enables the pad unless the caller explicitly opts out."""

    def setUp(self):
        FakeGamepad.instances = []

    def test_gamepad_is_on_by_default(self):
        params = inspect.signature(input_source.build_input_source).parameters
        self.assertIs(params["use_gamepad"].default, True)

    def test_default_build_probes_and_attaches_a_present_pad(self):
        def factory(**kwargs):
            pad = FakeGamepad(**kwargs)
            pad.available = True  # the boot probe "finds" it
            return pad
        with mock.patch.object(input_source, "GamepadSource", factory):
            src = input_source.build_input_source()
        self.assertIs(src.gamepad, FakeGamepad.instances[0])
        self.assertTrue(src.gamepad_enabled)

    def test_default_build_without_a_pad_hotplugs_later(self):
        clock = FakeTime()
        with mock.patch.object(input_source, "GamepadSource", FakeGamepad), \
                mock.patch.object(input_source, "time", clock):
            src = input_source.build_input_source()
            self.assertIsNone(src.gamepad)  # boot probe found nothing...
            self.assertTrue(src.gamepad_enabled)  # ...but the pad stays enabled
            src.poll(0.016)
            # One instance from the boot probe, one from the lazy hot-plug.
            self.assertEqual(len(FakeGamepad.instances), 2)
            self.assertIs(src.gamepad, FakeGamepad.instances[1])
            self.assertEqual(FakeGamepad.instances[1].rescan_calls, 1)

    def test_explicit_opt_out_skips_the_boot_probe(self):
        with mock.patch.object(input_source, "GamepadSource", FakeGamepad):
            src = input_source.build_input_source(use_gamepad=False)
        self.assertIsNone(src.gamepad)
        self.assertEqual(FakeGamepad.instances, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
