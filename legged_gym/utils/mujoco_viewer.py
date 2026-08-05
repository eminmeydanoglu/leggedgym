"""Interactive MuJoCo viewer driven synchronously from the simulation thread.

Why not ``mujoco.viewer.launch_passive``: the passive viewer owns its own
window/event loop and only exposes a *key-press* hook (no releases) with no way
to draw text.  The play arena needs both -- its drive keys ramp a velocity
command for as long as a key is *held*, which is a press/release state machine,
and the HUD is the only place the arena tells the user which terrain tab, level
and policy are active.  So this is a plain GLFW window plus MuJoCo's ``mjv``
(scene) / ``mjr`` (render) C API.

The whole thing renders from :meth:`MujocoViewer.sync`, i.e. on the caller's
thread, inside the caller's step loop.  That is the real reason to prefer it
over the Genesis path: there is no viewer thread, so there is nothing to lock.
Key callbacks, camera writes and HUD writes all happen on the one thread that
also steps physics, and ``native_ui``'s careful queue/lock dance
(``legged_gym/utils/native_ui.py``) is simply unnecessary here.

Failure policy: a viewer is a convenience, never a dependency.  Construction on
a machine with no display must not raise and must not stop the simulation --
every failure path prints exactly one line, marks the viewer dead and turns
:meth:`sync` into a no-op.  Callers that want to run headless should therefore
*not* gate their loop on :attr:`is_running` alone.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Tuple

import glfw
import mujoco
import numpy as np

# Deliberately no ``legged_gym`` imports at module scope: importing this file
# must not drag in the package __init__, which requires a configured SIMULATOR.
# ``viewer_cfg`` is duck-typed via getattr for the same reason.


# ---------------------------------------------------------------------------
# Keymap
# ---------------------------------------------------------------------------
#
# The arena keymap (``native_ui.DRIVE_KEYMAP`` / ``ACTION_KEYMAP``) was written
# against Genesis, i.e. against *pyglet* key names.  Rather than make every
# caller carry a GLFW translation table, this module speaks pyglet's names:
# BRACKETLEFT/BRACKETRIGHT rather than GLFW's LEFT_BRACKET/RIGHT_BRACKET, and
# LSHIFT/RSHIFT rather than LEFT_SHIFT/RIGHT_SHIFT.  Anything not in this table
# is never reported -- an unmapped key must not reach the caller as a mystery
# string it has no branch for.
_KEY_NAME_BY_GLFW_CODE = {
    glfw.KEY_UP: "UP",
    glfw.KEY_DOWN: "DOWN",
    glfw.KEY_LEFT: "LEFT",
    glfw.KEY_RIGHT: "RIGHT",
    glfw.KEY_Q: "Q",
    glfw.KEY_E: "E",
    glfw.KEY_T: "T",
    glfw.KEY_M: "M",
    glfw.KEY_N: "N",
    glfw.KEY_J: "J",
    glfw.KEY_G: "G",
    glfw.KEY_TAB: "TAB",
    glfw.KEY_SPACE: "SPACE",
    glfw.KEY_BACKSPACE: "BACKSPACE",
    glfw.KEY_LEFT_BRACKET: "BRACKETLEFT",
    glfw.KEY_RIGHT_BRACKET: "BRACKETRIGHT",
    glfw.KEY_LEFT_SHIFT: "LSHIFT",
    glfw.KEY_RIGHT_SHIFT: "RSHIFT",
    glfw.KEY_ESCAPE: "ESCAPE",
}

# Public, sorted for tests and for anyone who wants to advertise the bindings.
SUPPORTED_KEY_NAMES: Tuple[str, ...] = tuple(sorted(_KEY_NAME_BY_GLFW_CODE.values()))


def glfw_key_name(keycode: int) -> Optional[str]:
    """GLFW keycode -> the uppercase name the arena keymap uses, or None.

    Split out of the class so the translation can be unit-tested without ever
    opening a window (GLFW keycodes are plain module constants; they need no
    ``glfw.init()``).
    """
    return _KEY_NAME_BY_GLFW_CODE.get(keycode)


# ---------------------------------------------------------------------------
# Camera conversion
# ---------------------------------------------------------------------------
#
# MuJoCo's free camera is polar, not a pose: it stores ``lookat`` plus
# ``azimuth``/``elevation`` (degrees) and ``distance``.  ``mjv_updateCamera``
# rebuilds the eye from them as
#
#     forward = (cos(az)cos(el), sin(az)cos(el), sin(el))
#     eye     = lookat - distance * forward
#
# so inverting it is just the spherical coordinates of ``target - eye``.  Sanity
# check against MuJoCo's own default free camera (az=90, el=-45): forward
# becomes (0, +0.707, -0.707), putting the eye south of and above the lookat --
# which is exactly the default view you get from ``simulate``.
_MIN_DISTANCE = 1e-6


def camera_from_eye_target(
    eye,
    target,
    azimuth_fallback: float = 0.0,
    elevation_fallback: float = 0.0,
) -> Tuple[float, float, float, np.ndarray]:
    """World-space eye/target -> ``(azimuth_deg, elevation_deg, distance, lookat)``.

    ``*_fallback`` cover the two degeneracies the caller's orbit rig can hit
    transiently: a straight-down/straight-up view leaves azimuth undefined, and
    ``eye == target`` leaves both angles undefined.  Holding the previous angle
    keeps the view from snapping when the rig passes through the pole.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    lookat = np.asarray(target, dtype=np.float64).reshape(3).copy()

    delta = lookat - eye
    distance = float(np.linalg.norm(delta))
    if distance < _MIN_DISTANCE:
        # Eye sits on the target: nothing about the orientation is recoverable.
        return float(azimuth_fallback), float(elevation_fallback), 0.0, lookat

    forward = delta / distance
    # clip guards against |z| drifting a hair past 1 through rounding.
    elevation = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))

    horizontal = math.hypot(float(forward[0]), float(forward[1]))
    if horizontal < _MIN_DISTANCE:
        azimuth = float(azimuth_fallback)
    else:
        azimuth = math.degrees(math.atan2(float(forward[1]), float(forward[0])))

    return azimuth, elevation, distance, lookat


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

_HUD_SLOTS = 4
# HUD anchor per slot, in the order the caller passes them: top-centre,
# bottom-left upper row, bottom-left lower row, top-right.  The two bottom-left
# rows share one ``mjr_overlay`` call because that anchor grows *upward* from
# the bottom edge -- drawing them separately would stack them both on the floor.
_MAX_SCENE_GEOM = 20000


def _sanitize_hud_lines(lines: Optional[Iterable]) -> list:
    """Coerce whatever the caller passed into exactly ``_HUD_SLOTS`` strings.

    The HUD is cosmetic, so a wrong-length or wrong-typed list must degrade to
    a shorter HUD, never to an exception in the middle of the draw pass.
    """
    if lines is None:
        return [""] * _HUD_SLOTS
    if isinstance(lines, str):  # a bare string is almost certainly a mistake
        lines = [lines]
    out = []
    for item in list(lines)[:_HUD_SLOTS]:
        out.append("" if item is None else str(item))
    out.extend([""] * (_HUD_SLOTS - len(out)))
    return out


class MujocoViewer:
    """GLFW + mjv/mjr window rendered one frame per :meth:`sync` call."""

    def __init__(
        self,
        model,
        data,
        viewer_cfg=None,
        title: str = "LeggedGym-Ex (MuJoCo)",
        width: int = 1920,
        height: int = 1080,
    ):
        self.model = model
        self.data = data

        self._alive = False
        self._death_printed = False
        self._window = None
        self._ctx = None
        self._scene = None
        self._cam = None
        self._opt = None

        self._hud_lines = [""] * _HUD_SLOTS
        self._key_cb: Optional[Callable[[str, bool], None]] = None
        self._scroll_cb: Optional[Callable[[float], None]] = None
        self._drag_cb: Optional[Callable[[float, float], None]] = None

        self._buttons = {"left": False, "right": False, "middle": False}
        self._last_cursor = (0.0, 0.0)

        # Only two knobs are read off the legged_gym viewer config.  Note that
        # ``render_refresh_rate`` is *advisory here*: this viewer is driven by
        # the caller's loop, so it renders exactly one frame per sync() and has
        # no clock of its own to throttle against.  It is kept only so callers
        # can pace their own sync() calls from the same number.
        self.refresh_rate = float(getattr(viewer_cfg, "render_refresh_rate", 0.0) or 0.0)
        camera_fov = getattr(viewer_cfg, "camera_fov", None) if viewer_cfg is not None else None

        try:
            # glfw.init() is idempotent and cheap, but on a headless box it
            # either returns False or raises GLFWError depending on the wrapper
            # version -- both have to be caught here.
            if not glfw.init():
                self._die("no display (glfw.init failed)")
                return
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            # Four samples remove the most visible jagged edges at 1080p while
            # leaving the CPU-bound simulation/policy path untouched.
            glfw.window_hint(glfw.SAMPLES, 4)
            self._window = glfw.create_window(int(width), int(height), str(title), None, None)
            if not self._window:
                self._die("could not create a window (no GL context?)")
                return
            glfw.make_context_current(self._window)
            # Do NOT vsync: swap_buffers would then block the physics loop for
            # up to a frame, silently capping sim rate at the monitor refresh.
            glfw.swap_interval(0)

            if camera_fov is not None:
                # vis.global_.fovy is what mjv_updateScene feeds the free camera.
                self.model.vis.global_.fovy = float(camera_fov)

            # The stock 4096 map is visibly blocky across the multi-tile MoE
            # arena. This is allocated once on the GPU and does not add any
            # CPU/GPU policy synchronization to the 50 Hz control loop.
            self.model.vis.quality.shadowsize = max(
                int(self.model.vis.quality.shadowsize), 8192
            )

            self._scene = mujoco.MjvScene(self.model, maxgeom=_MAX_SCENE_GEOM)
            self._scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            self._opt = mujoco.MjvOption()
            self._cam = mujoco.MjvCamera()
            # Start from MuJoCo's own default free camera (it derives a sane
            # distance from model.stat), so the window is usable by mouse alone
            # until/unless the caller starts pushing poses.
            mujoco.mjv_defaultFreeCamera(self.model, self._cam)
            self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE

            self._ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self._ctx)

            glfw.set_key_callback(self._window, self._on_key)
            glfw.set_scroll_callback(self._window, self._on_scroll)
            glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
            glfw.set_cursor_pos_callback(self._window, self._on_cursor_pos)

            self._alive = True
        except Exception as exc:  # noqa: BLE001 - any GL/driver failure is survivable
            self._die(f"{type(exc).__name__}: {exc}")

    # -- lifecycle ---------------------------------------------------------

    def _die(self, reason: str) -> None:
        """One printed line, then permanently inert.  Never raises."""
        if not self._death_printed:
            self._death_printed = True
            print(f"[mujoco_viewer] disabled - {reason}; simulation continues headless.")
        self._alive = False
        try:
            if self._window is not None:
                glfw.destroy_window(self._window)
        except Exception:  # noqa: BLE001 - teardown must not mask the real failure
            pass
        self._window = None

    @property
    def is_running(self) -> bool:
        return bool(self._alive)

    def close(self) -> None:
        if not self._alive:
            self._window = None
            return
        self._alive = False
        try:
            # Freeing an MjrContext touches GL objects, so its context has to be
            # current at that moment.
            glfw.make_context_current(self._window)
            if self._ctx is not None:
                self._ctx.free()
            if self._scene is not None:
                self._scene.free()
        except Exception:  # noqa: BLE001
            pass
        try:
            glfw.destroy_window(self._window)
        except Exception:  # noqa: BLE001
            pass
        self._window = None
        # Intentionally no glfw.terminate(): the process may own other GLFW
        # windows (e.g. a second eval viewer), and terminate() would kill them.

    # -- per-frame ---------------------------------------------------------

    def sync(self) -> None:
        """Poll input and draw exactly one frame.  Never raises."""
        if not self._alive:
            return
        try:
            if glfw.window_should_close(self._window):
                self.close()
                return
            glfw.make_context_current(self._window)
            self._draw_frame()
            glfw.swap_buffers(self._window)
            # Events are polled *after* the draw so the callbacks the caller
            # will read on its next step see the state of the frame it just saw.
            glfw.poll_events()
        except Exception as exc:  # noqa: BLE001
            self._die(f"render failed ({type(exc).__name__}: {exc})")

    def _draw_frame(self) -> None:
        # Framebuffer size, not window size: they differ on HiDPI/fractional
        # scaling, and a window-sized viewport renders into a quarter of the
        # screen there.
        fb_w, fb_h = glfw.get_framebuffer_size(self._window)
        viewport = mujoco.MjrRect(0, 0, int(fb_w), int(fb_h))
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self._opt,
            None,
            self._cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._ctx)
        self._draw_hud(viewport)

    def _draw_hud(self, viewport) -> None:
        # Snapshot the caller's live list once per frame: it is mutated in place
        # from the sim loop, and re-reading it between the overlay calls below
        # could otherwise paint two different HUD states in one frame.
        top, bottom_upper, bottom_lower, top_right = _sanitize_hud_lines(self._hud_lines)
        font = mujoco.mjtFont.mjFONT_NORMAL
        if top:
            mujoco.mjr_overlay(font, mujoco.mjtGridPos.mjGRID_TOP, viewport, top, "", self._ctx)
        if bottom_upper or bottom_lower:
            # Two rows in one call; the anchor stacks upward from the bottom
            # edge, so line 0 lands above line 1 exactly as the caller expects.
            mujoco.mjr_overlay(
                font,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                f"{bottom_upper}\n{bottom_lower}",
                "",
                self._ctx,
            )
        if top_right:
            mujoco.mjr_overlay(
                font, mujoco.mjtGridPos.mjGRID_TOPRIGHT, viewport, top_right, "", self._ctx
            )

    # -- caller-facing setters --------------------------------------------

    def set_camera(self, eye, target) -> None:
        """Point the free camera at ``target`` from ``eye`` (world space).

        Camera *policy* (the orbit rig, follow smoothing, collision) belongs to
        the caller; this only converts a pose into the polar form mjvCamera
        wants.  Until this is called the default free camera is left untouched,
        so the mouse trackball still works out of the box.
        """
        if not self._alive:
            return
        azimuth, elevation, distance, lookat = camera_from_eye_target(
            eye, target, azimuth_fallback=self._cam.azimuth, elevation_fallback=self._cam.elevation
        )
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._cam.lookat[:] = lookat
        self._cam.distance = distance
        self._cam.azimuth = azimuth
        self._cam.elevation = elevation

    def set_hud_lines(self, lines) -> None:
        """Adopt ``lines`` as the live HUD source -- BY REFERENCE, not by copy.

        Callers hand over one list at startup and then swap strings into it in
        place every frame (native_ui._install_hud does exactly this, and its
        Genesis plugin has the same contract).  Copying here would freeze the
        HUD on whatever text happened to be in the list at install time, so the
        coercion to four strings is deferred to :meth:`_draw_hud` instead, where
        it also acts as the per-frame snapshot that keeps the overlay from
        tearing mid-draw.
        """
        self._hud_lines = lines

    def set_key_callback(self, cb) -> None:
        self._key_cb = cb

    def set_scroll_callback(self, cb) -> None:
        self._scroll_cb = cb

    def set_drag_callback(self, cb) -> None:
        self._drag_cb = cb

    # -- GLFW callbacks ----------------------------------------------------
    #
    # Every one of these is entered from C.  An exception escaping a GLFW
    # callback unwinds through the C stack and is at best swallowed with a
    # warning, at worst a crash -- so each one is wrapped by _guarded().

    def _guarded(self, what: str, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            print(f"[mujoco_viewer] {what} raised: {type(exc).__name__}: {exc}")

    def _on_key(self, _window, keycode, _scancode, action, _mods) -> None:
        # glfw.REPEAT is dropped on purpose.  The caller treats press/release as
        # a held-state latch, and auto-repeat would deliver a stream of presses
        # with no matching releases -- the key would look stuck.
        if action not in (glfw.PRESS, glfw.RELEASE):
            return
        name = glfw_key_name(keycode)
        if name is not None and self._key_cb is not None:
            self._guarded("key callback", self._key_cb, name, action == glfw.PRESS)

    def _on_scroll(self, _window, _dx, dy) -> None:
        if self._scroll_cb is not None:
            self._guarded("scroll callback", self._scroll_cb, float(dy))
            return
        # Wheel notches are already relative, so unlike drags they must not be
        # divided by the window height; 0.05/notch is MuJoCo simulate's value.
        self._move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * float(dy))

    def _on_mouse_button(self, window, button, action, _mods) -> None:
        pressed = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._buttons["left"] = pressed
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._buttons["right"] = pressed
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._buttons["middle"] = pressed
        # Re-seed the cursor origin on every button edge, otherwise the first
        # motion event of a drag reports the distance travelled since the last
        # drag ended and the view jumps.
        self._last_cursor = glfw.get_cursor_pos(window)

    def _on_cursor_pos(self, _window, xpos, ypos) -> None:
        last_x, last_y = self._last_cursor
        dx = float(xpos) - float(last_x)
        dy = float(ypos) - float(last_y)
        self._last_cursor = (float(xpos), float(ypos))
        if dx == 0.0 and dy == 0.0:
            return

        if self._drag_cb is not None:
            # Only the left button is forwarded.  A caller that owns look-around
            # owns the whole camera, so letting the right button pan here would
            # just be undone by its next set_camera().
            #
            # dy is NEGATED on the way out: GLFW's cursor origin is the top-left
            # corner (+y down), pyglet's is the bottom-left (+y up).  Callers
            # share one mouse-look integrator across backends (see
            # native_ui._install_look_handlers_mujoco), so the sign has to be
            # normalised to pyglet's here or dragging pitches the wrong way.
            # The trackball fallback below keeps the raw GLFW sign because
            # mjv_moveCamera wants exactly that.
            if self._buttons["left"]:
                self._guarded("drag callback", self._drag_cb, dx, -dy)
            return

        # Standalone fallback: MuJoCo's usual trackball, so the window is still
        # useful when nobody has installed callbacks.  mjv_moveCamera wants
        # deltas relative to window height, not pixels.
        _, win_h = glfw.get_window_size(self._window)
        scale = float(win_h) if win_h else 1.0
        if self._buttons["left"]:
            self._move_camera(mujoco.mjtMouse.mjMOUSE_ROTATE_V, dx / scale, dy / scale)
        elif self._buttons["right"]:
            self._move_camera(mujoco.mjtMouse.mjMOUSE_MOVE_V, dx / scale, dy / scale)
        elif self._buttons["middle"]:
            self._move_camera(mujoco.mjtMouse.mjMOUSE_ZOOM, dx / scale, dy / scale)

    def _move_camera(self, action, reldx: float, reldy: float) -> None:
        if self._alive:
            self._guarded(
                "camera move",
                mujoco.mjv_moveCamera,
                self.model, action, reldx, reldy, self._scene, self._cam,
            )
