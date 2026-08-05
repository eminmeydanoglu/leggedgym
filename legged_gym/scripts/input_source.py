"""Velocity-command input sources for interactive play.

Single source of truth for the drive envelope and for turning raw human input
(gamepad sticks, keyboard key-down / key-up) into a ``(vx, vy, yaw)`` command in
the robot base frame.

Why this module exists
----------------------
The Viser keyboard path (``legged_gym/utils/viser_viewer.py``) has to *infer*
"key is held" from browser auto-repeat, because Viser only forwards key-down.
Browsers auto-repeat only the last key, so it needs a sticky-latch /
grace-window heuristic that gets keys stuck or drops them.  The Genesis native
viewer delivers real ``KeyAction.PRESS`` / ``KeyAction.RELEASE`` events, so held
state is *known*.  ``KeyboardSource`` therefore keeps only the smooth ramp
(the part that is about feel) and none of the repeat heuristic.

Everything here is GPU-free and display-free so it is unit testable; ``pyglet``
is imported lazily and only by ``GamepadSource``.

Gamepad default-on and hot-plug
-------------------------------
The pad used to be opt-in (``--use_joystick``) and probed exactly once at
startup, so a pad unplugged at launch or plugged in later never appeared.
``build_input_source()`` now enables it by default, and ``MergedSource`` keeps
retrying enumeration from ``poll`` (throttled, never raising, sim-thread safe),
so hot-plug and mid-session reconnects just work.  The pad can also be toggled
off/on at runtime (``toggle_gamepad``, bound to a key by the UI layer) without
closing the device, so re-enabling is instant.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Command = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Drive envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriveEnvelope:
    """Velocity envelope for interactive driving (m/s, m/s, rad/s).

    These are the values the Viser keyboard control has always used
    (``viser_viewer.py`` ``_max_vel``); this dataclass is now the single source
    of truth so keyboard, gamepad and Viser cannot drift apart.
    """

    forward: float = 1.5
    backward: float = 1.0
    lateral: float = 0.7
    yaw: float = 1.5
    # Playback-only high-speed envelope.  It is selected by a held gamepad
    # trigger, never written to the environment cfg, so training/resume command
    # ranges keep their original contract.
    turbo_forward: float = 2.0
    turbo_backward: float = 2.0

    def clamp(self, vx: float, vy: float, yaw: float) -> Command:
        """Clamp a command into the envelope (forward/backward are asymmetric)."""
        vx = min(max(float(vx), -abs(self.backward)), abs(self.forward))
        vy = min(max(float(vy), -abs(self.lateral)), abs(self.lateral))
        yaw = min(max(float(yaw), -abs(self.yaw)), abs(self.yaw))
        return (vx, vy, yaw)

    def scale(self, ax: float, ay: float, ayaw: float, *, turbo: bool = False) -> Command:
        """Map unit-square axes in [-1, 1] onto the envelope, then clamp."""
        ax = float(ax)
        forward = self.turbo_forward if turbo else self.forward
        backward = self.turbo_backward if turbo else self.backward
        vx = ax * (forward if ax >= 0.0 else backward)
        # ``clamp`` deliberately retains the normal limits, so apply the
        # turbo vx limit here while all lateral/yaw limits remain unchanged.
        vx = min(max(vx, -abs(backward)), abs(forward))
        vy = min(max(float(ay) * self.lateral, -abs(self.lateral)), abs(self.lateral))
        yaw = min(max(float(ayaw) * self.yaw, -abs(self.yaw)), abs(self.yaw))
        return (vx, vy, yaw)

    def as_dict(self) -> Dict[str, float]:
        return {
            "fwd": self.forward,
            "back": self.backward,
            "lat": self.lateral,
            "yaw": self.yaw,
            "turbo_fwd": self.turbo_forward,
            "turbo_back": self.turbo_backward,
        }


DEFAULT_ENVELOPE = DriveEnvelope()

# Ramp time constants, carried over from the Viser keyboard control: accelerate
# quickly, coast down slowly, so releasing a key does not snap the command.
DEFAULT_TAU_ACCEL = 0.10
DEFAULT_TAU_DECEL = 0.30

# F310 sticks drift a few percent when centred; without a deadzone the robot
# walks on its own with nobody touching the pad.
DEFAULT_DEADZONE = 0.12
DEFAULT_EXPO = 0.35


# ---------------------------------------------------------------------------
# Pure response-curve helpers
# ---------------------------------------------------------------------------

def apply_deadzone(value: float, deadzone: float) -> float:
    """Radial deadzone with re-normalisation.

    Values inside ``deadzone`` return exactly 0.0; outside, the remaining range
    is stretched back to full scale so the first usable stick position is still
    a *small* command (no jump from 0 to ``deadzone``).
    """
    v = float(value)
    # Guard against a pathological deadzone that would kill the whole range.
    dz = min(max(float(deadzone), 0.0), 0.95)
    if dz <= 0.0:
        return min(max(v, -1.0), 1.0)
    mag = abs(v)
    if mag <= dz:
        return 0.0
    scaled = (mag - dz) / (1.0 - dz)
    return math.copysign(min(scaled, 1.0), v)


def apply_expo(value: float, expo: float) -> float:
    """Blend linear and cubic response: ``(1-e)*v + e*v^3``.

    ``expo=0`` is linear, ``expo=1`` fully cubic.  Endpoints (+-1) are fixed, so
    the top speed is unchanged; only the fine control around centre softens.
    """
    v = min(max(float(value), -1.0), 1.0)
    e = min(max(float(expo), 0.0), 1.0)
    return (1.0 - e) * v + e * (v * v * v)


def ramp_towards(
    current: float,
    target: float,
    dt: float,
    tau_accel: float = DEFAULT_TAU_ACCEL,
    tau_decel: float = DEFAULT_TAU_DECEL,
) -> float:
    """First-order ramp of ``current`` toward ``target`` over ``dt`` seconds."""
    dt = float(dt)
    if dt <= 0.0:
        return float(current)
    tau = tau_accel if abs(target) > abs(current) else tau_decel
    if tau <= 1e-6:
        return float(target)
    alpha = 1.0 - math.exp(-dt / tau)
    return float(current) + (float(target) - float(current)) * alpha


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------

#: Logical drive actions.  The *keys* that map onto them belong to the UI layer
#: (``legged_gym/utils/native_ui.py``), not here.
DRIVE_ACTIONS = (
    "forward",
    "backward",
    "strafe_left",
    "strafe_right",
    "yaw_left",
    "yaw_right",
)


class KeyboardSource:
    """Ramped velocity command driven by real key press / release events.

    ``press`` / ``release`` are called from the viewer thread (Genesis runs the
    pyglet viewer in a background thread on Linux); ``poll`` is called from the
    sim thread.  A lock guards the held-key set.
    """

    def __init__(
        self,
        envelope: Optional[DriveEnvelope] = None,
        tau_accel: float = DEFAULT_TAU_ACCEL,
        tau_decel: float = DEFAULT_TAU_DECEL,
    ) -> None:
        self.envelope = envelope or DEFAULT_ENVELOPE
        self.tau_accel = float(tau_accel)
        self.tau_decel = float(tau_decel)
        self._lock = threading.Lock()
        self._held: Dict[str, bool] = {a: False for a in DRIVE_ACTIONS}
        self._vel: List[float] = [0.0, 0.0, 0.0]

    # -- event ingress (viewer thread) --------------------------------------
    def press(self, action: str) -> None:
        if action not in self._held:
            return
        with self._lock:
            self._held[action] = True

    def release(self, action: str) -> None:
        if action not in self._held:
            return
        with self._lock:
            self._held[action] = False

    def clear(self) -> None:
        """Hard stop: forget every held key and zero the ramped command."""
        with self._lock:
            for a in self._held:
                self._held[a] = False
            self._vel = [0.0, 0.0, 0.0]

    # -- query ---------------------------------------------------------------
    def held(self) -> Dict[str, bool]:
        with self._lock:
            return dict(self._held)

    def target(self) -> Command:
        """Un-ramped command implied by the currently held keys."""
        h = self.held()
        m = self.envelope
        vx = (m.forward if h["forward"] else 0.0) - (m.backward if h["backward"] else 0.0)
        vy = (m.lateral if h["strafe_left"] else 0.0) - (m.lateral if h["strafe_right"] else 0.0)
        yaw = (m.yaw if h["yaw_left"] else 0.0) - (m.yaw if h["yaw_right"] else 0.0)
        return m.clamp(vx, vy, yaw)

    @property
    def active(self) -> bool:
        """True while a key is held or the command has not yet coasted to zero."""
        with self._lock:
            if any(self._held.values()):
                return True
            return any(abs(v) > 1e-4 for v in self._vel)

    # -- poll (sim thread) ---------------------------------------------------
    def poll(self, dt: float) -> Command:
        tgt = self.target()
        with self._lock:
            for i in range(3):
                self._vel[i] = ramp_towards(
                    self._vel[i], tgt[i], dt, self.tau_accel, self.tau_decel
                )
            return (self._vel[0], self._vel[1], self._vel[2])


# ---------------------------------------------------------------------------
# Gamepad
# ---------------------------------------------------------------------------

#: Gamepad button -> arena event.  Consumed by ``NativeArenaUI`` / play.py, so
#: the pad can switch tabs and respawn without touching the keyboard.
#:
#: LB/RB are deliberately NOT in this table: they are the yaw axis now (see
#: ``YAW_BUTTONS`` and ``GamepadSource.poll``), which frees the right stick for
#: camera look.  Tabs and levels therefore moved onto the d-pad, which pyglet's
#: ``Controller`` exposes as the boolean ``dpup``/``dpdown``/``dpleft``/
#: ``dpright`` attributes -- same shape as any other button, so ``_scan_buttons``
#: needs no special case.  X/B are left unbound on purpose: the d-pad owns
#: levels now and a second, redundant binding only makes the legend lie.
DEFAULT_BUTTON_EVENTS: Dict[str, str] = {
    "dpleft": "tab_prev",
    "dpright": "tab_next",
    "dpdown": "level_prev",
    "dpup": "level_next",
    "a": "respawn",
    "y": "camera_next",
    "rightstick": "camera_snap",
    "start": "stop",
    "back": "legend",
}

#: (yaw-left button, yaw-right button).  Read as an axis, never as an event.
YAW_BUTTONS: Tuple[str, str] = ("leftshoulder", "rightshoulder")

# Pyglet's standard controller mapping calls this ``lefttrigger``.  The two
# fallbacks cover common Linux mappings without making a controller-specific
# layout part of play.py.
TURBO_TRIGGER_NAMES: Tuple[str, ...] = ("lefttrigger", "left_trigger", "l2")
TURBO_TRIGGER_THRESHOLD = 0.5


class GamepadSource:
    """Analog velocity command from a game controller, via pyglet.

    pyglet (not pygame) on purpose: ``pygame.init()`` also brings up the video
    subsystem, which can fight the viewer's own pyglet window, and pyglet is
    already a hard dependency of the Genesis viewer.

    Layout ("GTA-style", since the right stick became a camera):

    * left stick   -> vx / vy, unchanged: ``vx = -lefty``, ``vy = -leftx``,
      exactly the signs play.py used with the old pygame ``Joystick``, so
      swapping the backend cannot silently flip the drive controls.
      The command stays in the ROBOT BODY frame; it is deliberately *not*
      rotated into the camera frame, so what the left stick means does not
      depend on where the camera happens to be pointing.
    * LB / RB      -> yaw left / right.  These are digital, so a raw press
      would be a step input into a velocity-tracking policy; they are run
      through the same first-order ramp the keyboard uses (``ramp_towards``
      with ``tau_accel`` / ``tau_decel``) and shaped in *axis* space so the
      envelope clamp stays the single source of truth for the top yaw rate.
    * right stick  -> camera look (``look``), NOT part of the command.  It is
      exposed as a separate value on purpose: ``poll`` must keep returning the
      3-tuple ``Command`` that the Viser path and every caller expect, so the
      camera cannot be smuggled through the velocity contract.
    """

    def __init__(
        self,
        envelope: Optional[DriveEnvelope] = None,
        deadzone: float = DEFAULT_DEADZONE,
        expo: float = DEFAULT_EXPO,
        controller=None,
        button_events: Optional[Dict[str, str]] = None,
        auto_open: bool = True,
        tau_accel: float = DEFAULT_TAU_ACCEL,
        tau_decel: float = DEFAULT_TAU_DECEL,
    ) -> None:
        self.envelope = envelope or DEFAULT_ENVELOPE
        self.deadzone = float(deadzone)
        self.expo = float(expo)
        self.tau_accel = float(tau_accel)
        self.tau_decel = float(tau_decel)
        self.button_events = dict(
            DEFAULT_BUTTON_EVENTS if button_events is None else button_events
        )
        self.controller = controller
        self.name: str = ""
        self.stick_active = False
        #: Ramped yaw in unit-axis space ([-1, 1]); scaled by the envelope in
        #: ``poll``.  Ramping the axis rather than the m/s value keeps the
        #: envelope the only place the top rate is defined.
        self._yaw_axis = 0.0
        #: Last shaped right-stick position, (x, y), read by the camera rig.
        self._look: Tuple[float, float] = (0.0, 0.0)
        self._prev_buttons: Dict[str, bool] = {}
        self._events: List[str] = []
        self._owns_pump = False
        if controller is None and auto_open:
            self.open()
        elif controller is not None:
            self.name = str(getattr(controller, "name", "controller"))

    # -- lifecycle -----------------------------------------------------------
    def open(self) -> bool:
        """Try to grab the first connected controller.  Never raises."""
        return self._open_first_controller(quiet=False)

    def rescan(self) -> bool:
        """Retry enumeration + open after a boot-time miss or a mid-session drop.

        This is the hot-plug entry point used by ``MergedSource``: a pad that
        was absent at startup, or one that ``pump()`` dropped after a
        disconnect (``self.controller`` set to None), is picked up without
        restarting play.py.  Returns True immediately if a controller is
        already held.  Never raises; failure is silent because the caller
        retries on a throttle, while success prints the same
        ``[input] gamepad: <name>`` line as ``open()``.
        """
        if self.available:
            return True
        try:
            return self._open_first_controller(quiet=True)
        except Exception:  # belt and braces: the sim thread must survive this
            return False

    def _open_first_controller(self, quiet: bool) -> bool:
        """Enumerate and open the first controller; ``quiet`` mutes failures.

        The failure prints are only wanted for the explicit boot-time attempt:
        the throttled hot-plug retry would otherwise spam "no gamepad
        detected" every couple of seconds for the whole session.
        """
        try:
            import pyglet  # noqa: F401
            from pyglet.input import get_controllers
        except Exception as exc:  # pragma: no cover - depends on install
            if not quiet:
                print(f"[input] gamepad disabled: pyglet unavailable ({exc})")
            return False
        try:
            controllers = get_controllers()
        except Exception as exc:  # pragma: no cover - depends on hardware
            if not quiet:
                print(f"[input] gamepad disabled: enumeration failed ({exc})")
            return False
        if not controllers:
            if not quiet:
                print("[input] no gamepad detected; keyboard only")
            return False
        controller = controllers[0]
        try:
            controller.open()
        except Exception as exc:  # pragma: no cover - depends on hardware
            if not quiet:
                print(f"[input] gamepad disabled: open failed ({exc})")
            return False
        self.controller = controller
        self.name = str(getattr(controller, "name", "controller"))
        self._claim_pump(controller)
        print(f"[input] gamepad: {self.name} (deadzone={self.deadzone:.2f}, "
              f"expo={self.expo:.2f})")
        return True

    def _claim_pump(self, controller) -> None:
        """Take the device off pyglet's app event loop and pump it ourselves.

        Genesis runs the viewer in a background thread that calls
        ``pyglet.app.platform_event_loop.step(0.0)``; that loop reads every
        registered evdev device.  If we also read the same fd from the sim
        thread the two races on the device's internal event queue.  Removing the
        device from ``select_devices`` makes this source the single reader, and
        it is also what makes the pad work at all in headless / Viser mode where
        no pyglet loop is running.
        """
        device = getattr(controller, "device", None)
        if device is None or not hasattr(device, "poll") or not hasattr(device, "select"):
            return
        try:
            import pyglet.app

            loop = getattr(pyglet.app, "platform_event_loop", None)
            devices = getattr(loop, "select_devices", None)
            if devices is not None and device in devices:
                devices.discard(device)
                self._owns_pump = True
        except Exception:  # pragma: no cover - platform specific
            self._owns_pump = False

    def close(self) -> None:
        controller = self.controller
        self.controller = None
        if controller is None:
            return
        device = getattr(controller, "device", None)
        if self._owns_pump and device is not None:
            # ``EvdevDevice.close`` does ``select_devices.remove(self)``; put it
            # back first or that raises KeyError.
            try:
                import pyglet.app

                loop = getattr(pyglet.app, "platform_event_loop", None)
                devices = getattr(loop, "select_devices", None)
                if devices is not None:
                    devices.add(device)
            except Exception:  # pragma: no cover
                pass
        try:
            controller.close()
        except Exception:  # pragma: no cover
            pass

    @property
    def available(self) -> bool:
        return self.controller is not None

    # -- pumping -------------------------------------------------------------
    def pump(self) -> None:
        """Drain pending evdev events into the controller's pollable state."""
        controller = self.controller
        if controller is None:
            return
        device = getattr(controller, "device", None)
        if device is None or not hasattr(device, "poll"):
            return
        try:
            # Bounded so a flood of events cannot stall the sim loop.
            for _ in range(64):
                if not device.poll():
                    break
                device.select()
        except Exception as exc:
            print(f"[input] gamepad disconnected ({exc}); falling back to keyboard")
            self.controller = None

    # -- poll ----------------------------------------------------------------
    def _shape(self, raw: float) -> float:
        return apply_expo(apply_deadzone(raw, self.deadzone), self.expo)

    @staticmethod
    def _turbo_held(controller) -> bool:
        """Whether LT is held enough to select the ±2 m/s playback envelope.

        Mapped trigger values are normally in ``[0, 1]``.  Bool mappings are
        accepted too, which is useful for controllers whose Linux driver
        exposes LT as a button rather than an axis.
        """
        for name in TURBO_TRIGGER_NAMES:
            try:
                value = getattr(controller, name, None)
            except Exception:  # pragma: no cover - device vanished mid-read
                continue
            if isinstance(value, bool):
                if value:
                    return True
                continue
            try:
                if float(value) >= TURBO_TRIGGER_THRESHOLD:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @property
    def look(self) -> Tuple[float, float]:
        """Shaped right-stick position ``(x, y)`` for the camera rig.

        Deliberately outside the ``Command`` tuple: the velocity contract is
        3 numbers and several consumers (Viser, the unit tests, play.py's
        ``env.commands[:, 0:3]`` write) depend on that.  Sign convention is the
        raw pad convention -- ``x > 0`` is stick right, ``y < 0`` is stick up
        (an SDL-style pad reports -1 at the top, the same reason ``vx = -lefty``)
        -- and the camera layer, not this module, decides what "up" should do.
        Zero whenever no pad is connected, so a caller never has to check.
        """
        return self._look

    def poll(self, dt: float = 0.0) -> Command:
        """Return the stick/shoulder command and latch button edges as events.

        ``dt`` is no longer unused: the yaw axis now comes from the digital
        LB/RB buttons and needs the same first-order ramp the keyboard uses,
        or every turn would be a step command into the policy.
        """
        controller = self.controller
        if controller is None:
            self._idle(dt)
            return self.envelope.scale(0.0, 0.0, self._yaw_axis)
        self.pump()
        controller = self.controller
        if controller is None:
            self._idle(dt)
            return self.envelope.scale(0.0, 0.0, self._yaw_axis)

        try:
            lx = float(getattr(controller, "leftx", 0.0) or 0.0)
            ly = float(getattr(controller, "lefty", 0.0) or 0.0)
            rx = float(getattr(controller, "rightx", 0.0) or 0.0)
            ry = float(getattr(controller, "righty", 0.0) or 0.0)
            yaw_left = bool(getattr(controller, YAW_BUTTONS[0], False))
            yaw_right = bool(getattr(controller, YAW_BUTTONS[1], False))
        except Exception:  # pragma: no cover - disconnect mid-read
            self._idle(dt)
            return self.envelope.scale(0.0, 0.0, self._yaw_axis)

        # Same signs as the old pygame mapping in play.py.
        ax = self._shape(-ly)
        ay = self._shape(-lx)
        turbo = self._turbo_held(controller)
        # LB/RB are a +-1 axis target, then ramped.  Both held cancel out, like
        # the opposing keyboard keys do.
        yaw_target = (1.0 if yaw_left else 0.0) - (1.0 if yaw_right else 0.0)
        self._yaw_axis = ramp_towards(
            self._yaw_axis, yaw_target, dt, self.tau_accel, self.tau_decel
        )
        self._look = (self._shape(rx), self._shape(ry))
        # Ownership rule (MergedSource.poll): the pad only wins the frame while
        # this is True.  It must therefore include the shoulder yaw -- holding
        # only LB would otherwise hand the command straight back to the
        # keyboard, which is zero, and the robot would not turn.  It also stays
        # True while the released ramp coasts down (same rule as
        # KeyboardSource.active), so letting go of LB decays instead of
        # snapping to whatever the keyboard happens to hold.  The look stick is
        # NOT included: moving the camera must not seize the drive command.
        self.stick_active = (
            abs(ax) > 0.0
            or abs(ay) > 0.0
            or turbo
            or yaw_target != 0.0
            or abs(self._yaw_axis) > 1e-4
        )
        self._scan_buttons(controller)
        return self.envelope.scale(ax, ay, self._yaw_axis, turbo=turbo)

    def _idle(self, dt: float) -> None:
        """No controller this frame: coast the yaw ramp down, zero the look.

        The ramp is still advanced so a pad that disconnects mid-turn decays
        the same way a released button does instead of freezing at its last
        value; ``stick_active`` follows it down and then hands the command back
        to the keyboard.
        """
        self._yaw_axis = ramp_towards(
            self._yaw_axis, 0.0, dt, self.tau_accel, self.tau_decel
        )
        self._look = (0.0, 0.0)
        self.stick_active = abs(self._yaw_axis) > 1e-4

    def _scan_buttons(self, controller) -> None:
        for button, event in self.button_events.items():
            try:
                now = bool(getattr(controller, button, False))
            except Exception:  # pragma: no cover
                continue
            if now and not self._prev_buttons.get(button, False):
                self._events.append(event)
            self._prev_buttons[button] = now

    def drain_events(self) -> List[str]:
        events, self._events = self._events, []
        return events


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class MergedSource:
    """Gamepad and keyboard coexisting.

    Priority is per-frame and value-based, not modal: while the pad is
    *driving* -- left stick outside its deadzone, or LB/RB held (or their ramp
    still coasting down) -- the gamepad owns the command, otherwise the
    keyboard does.  So letting go hands control straight back to the keys with
    no mode switch to remember.  The right stick is explicitly excluded from
    that test: it only moves the camera, and looking around must never take the
    drive command away from the keyboard.

    The pad is enabled by default (``gamepad_enabled``) and hot-pluggable:
    while it is enabled but absent, ``poll`` retries enumeration throttled to
    ``rescan_period_s`` and builds the ``GamepadSource`` lazily from the stored
    construction params when none existed at boot.  ``set_gamepad_enabled`` /
    ``toggle_gamepad`` switch the pad at runtime without closing the device, so
    re-enabling is instant.
    """

    #: Minimum seconds between hot-plug rescans in ``poll``.  Enumeration
    #: (``pyglet.input.get_controllers``) at this 0.5 Hz rate is cheap, and the
    #: throttle keeps a missing pad from adding per-frame cost on the sim
    #: thread.
    rescan_period_s: float = 2.0

    def __init__(
        self,
        keyboard: Optional[KeyboardSource] = None,
        gamepad: Optional[GamepadSource] = None,
        envelope: Optional[DriveEnvelope] = None,
        deadzone: float = DEFAULT_DEADZONE,
        expo: float = DEFAULT_EXPO,
    ) -> None:
        self.envelope = envelope or DEFAULT_ENVELOPE
        self.keyboard = keyboard if keyboard is not None else KeyboardSource(self.envelope)
        self.gamepad = gamepad
        # Construction params for a pad that shows up later: the GamepadSource
        # is built lazily on the first hot-plug rescan, so the values must
        # match what build_input_source() would have used at boot.
        self._gamepad_deadzone = float(deadzone)
        self._gamepad_expo = float(expo)
        self._gamepad_enabled = True
        self._last_rescan_s = float("-inf")
        self._rescan_lock = threading.Lock()
        self._lock = threading.Lock()
        self._events: List[str] = []
        self._last: Command = (0.0, 0.0, 0.0)

    @property
    def last_command(self) -> Command:
        return self._last

    @property
    def look(self) -> Tuple[float, float]:
        """Right-stick camera look, or ``(0.0, 0.0)`` when no pad owns it.

        A pure pass-through so the camera layer has exactly one thing to read
        and never has to know whether a pad exists, is enabled, or was
        hot-plugged.  Disabled-by-J is reported as centred: the J toggle means
        "the pad drives nothing", camera included.
        """
        if not self._gamepad_enabled:
            return (0.0, 0.0)
        gamepad = self.gamepad
        if gamepad is None or not getattr(gamepad, "available", False):
            return (0.0, 0.0)
        return tuple(getattr(gamepad, "look", (0.0, 0.0)))  # type: ignore[return-value]

    @property
    def gamepad_enabled(self) -> bool:
        """Whether the pad may own the command (the UI binds a key to toggle it)."""
        return self._gamepad_enabled

    def set_gamepad_enabled(self, enabled: bool) -> None:
        """Enable/disable the pad at runtime without closing the device.

        Disabling only makes ``poll`` skip the pad branch, so the keyboard owns
        the command; an open controller stays open and re-enabling is instant.
        Enabling while no pad is present runs one immediate rescan so the
        toggle feels responsive; the throttled hot-plug in ``poll`` keeps
        trying afterwards.
        """
        self._gamepad_enabled = bool(enabled)
        if not self._gamepad_enabled:
            if self.gamepad is not None:
                # Honest display state: the pad no longer drives anything.
                self.gamepad.stick_active = False
            return
        self._maybe_rescan_gamepad(force=True)

    def toggle_gamepad(self) -> bool:
        """Flip ``gamepad_enabled`` and return the new state (bound to a key)."""
        self.set_gamepad_enabled(not self._gamepad_enabled)
        return self._gamepad_enabled

    def _maybe_rescan_gamepad(self, force: bool = False) -> None:
        """Hot-plug: retry opening the pad while it is enabled but absent.

        Runs on the sim thread from ``poll`` (and on the viewer thread from
        ``set_gamepad_enabled``); throttled to one attempt per
        ``rescan_period_s`` and must never raise, so USB churn cannot disturb
        the sim loop.  Covers both "no pad at boot" (the GamepadSource is
        created lazily here from the stored params) and "pad dropped
        mid-session" (``pump()`` sets ``controller = None`` on a disconnect;
        ``rescan()`` re-enumerates).
        """
        if not self._gamepad_enabled:
            return
        gamepad = self.gamepad
        if gamepad is not None and gamepad.available:
            return
        now = time.monotonic()
        if not force and (now - self._last_rescan_s) < self.rescan_period_s:
            return
        with self._rescan_lock:
            # Re-check under the lock: a toggle on the viewer thread may have
            # just run a forced rescan, or created the pad already.
            gamepad = self.gamepad
            if gamepad is not None and gamepad.available:
                return
            now = time.monotonic()
            if not force and (now - self._last_rescan_s) < self.rescan_period_s:
                return
            self._last_rescan_s = now
            try:
                if gamepad is None:
                    gamepad = GamepadSource(
                        envelope=self.envelope,
                        deadzone=self._gamepad_deadzone,
                        expo=self._gamepad_expo,
                        auto_open=False,
                    )
                    # Kept even while unavailable: identity stays stable for
                    # readers (the HUD) and later rescans reuse the object.
                    self.gamepad = gamepad
                gamepad.rescan()
            except Exception:  # pragma: no cover - rescan() never raises
                pass

    def push_event(self, event: str) -> None:
        """Queue an arena event (safe to call from the viewer thread)."""
        with self._lock:
            self._events.append(str(event))

    def drain_events(self) -> List[str]:
        """Pop every queued event.  Call from the sim thread only."""
        if self.gamepad is not None:
            for event in self.gamepad.drain_events():
                self.push_event(event)
        with self._lock:
            events, self._events = self._events, []
        return events

    def clear(self) -> None:
        """Hard stop.  Only the keyboard latches state; sticks are live values."""
        self.keyboard.clear()
        self._last = (0.0, 0.0, 0.0)

    def poll(self, dt: float) -> Command:
        pad_cmd: Command = (0.0, 0.0, 0.0)
        pad_active = False
        if self._gamepad_enabled:
            # Cheap when the pad is present (early return); throttled retry
            # when it is missing, which is what makes hot-plug work.
            self._maybe_rescan_gamepad()
            gamepad = self.gamepad
            if gamepad is not None and gamepad.available:
                pad_cmd = gamepad.poll(dt)
                pad_active = gamepad.stick_active
        # The keyboard ramp must advance every frame even when the pad wins, or
        # a held key would jump to full speed the moment the sticks re-centre.
        key_cmd = self.keyboard.poll(dt)
        self._last = pad_cmd if pad_active else key_cmd
        return self._last


def build_input_source(
    use_gamepad: bool = True,
    envelope: Optional[DriveEnvelope] = None,
    deadzone: float = DEFAULT_DEADZONE,
    expo: float = DEFAULT_EXPO,
) -> MergedSource:
    """Convenience factory: keyboard always, gamepad on by default.

    The pad is probed once here; when it is absent at boot the returned
    MergedSource still picks it up later via the throttled hot-plug rescan in
    ``poll``, so callers no longer need to know up front whether a pad session
    is wanted (the old ``--use_joystick`` flag simply became unnecessary).
    """
    envelope = envelope or DEFAULT_ENVELOPE
    gamepad: Optional[GamepadSource] = None
    if use_gamepad:
        candidate = GamepadSource(envelope=envelope, deadzone=deadzone, expo=expo)
        gamepad = candidate if candidate.available else None
    return MergedSource(
        keyboard=KeyboardSource(envelope), gamepad=gamepad, envelope=envelope,
        deadzone=deadzone, expo=expo,
    )
