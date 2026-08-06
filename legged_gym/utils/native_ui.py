"""Native (Genesis viewer) play-arena UI: terrain "tabs" + real keyboard drive.

The showcase grid is already laid out as ``rows = difficulty level`` x
``columns = terrain type`` (``terrain.py:add_moe_terrain_to_map``), so a "stairs
tab" is nothing more than the ``stairs_up`` column, and switching tabs is a
teleport plus a camera reframe.  ``scene.build()`` is one-shot in Genesis --
entities and terrain cannot change afterwards -- so a tab is deliberately never
a rebuild.

Backends
--------
Everything below the ingress -- drive, orbit rig, tabs/levels, HUD text, gamepad,
checkpoint hot-swap -- is backend-agnostic and is written once.  Only five entry
points know which window is open (``_find_viewer``, ``_register_keybinds``,
``_install_look_handlers``, ``_install_hud``, ``_install_focus_guard``), and they
dispatch on ``_viewer_backend`` in {"genesis", "mujoco", None}.  Detection is by
attribute provenance, never ``isinstance``: play.py imports this module before it
knows which simulator it will build, so it must import with neither package
installed.  Anything a backend cannot offer prints one line and leaves the
session running, exactly as a missing Genesis feature already does.

Threading contract
------------------
Genesis runs the pyglet viewer in a background thread on Linux.  Keybind
callbacks therefore fire on the *viewer* thread while the sim steps on the main
thread.  Callbacks here only ever mutate thread-safe state:

* drive keys go straight into ``KeyboardSource`` (lock-protected),
* arena actions (tab / level / respawn / ...) are *queued* as events, and
* mouse-look drag/scroll only *accumulates* deltas under ``_look_lock``; the
  orbit camera is integrated and written from ``update_follow_camera`` on the
  sim thread (see ``_install_look_handlers``).

``interaction_loop`` drains the queue with :meth:`NativeArenaUI.drain_and_apply`
on the sim thread, which is where ``env.reset_idx`` / ``teleport_...`` are legal
to call.

The MuJoCo viewer is pumped synchronously from the sim thread, so ITS callbacks
already run where env state may be touched.  They deliberately follow the same
latch-only discipline anyway: the locks stay load-bearing for Genesis, they are
free uncontended, and one discipline keeps the two backends from diverging.

The on-screen HUD follows the same contract from the other direction: it is a
pyrender *plugin* (``_HudPlugin``) whose ``on_draw`` runs on the viewer thread,
after Genesis's own caption pass, and only ever *reads* the pre-formatted
strings in ``NativeArenaUI._hud_lines`` -- those are rebuilt on the sim thread
under an 8 Hz throttle by :meth:`NativeArenaUI.update_hud`.  The plugin never
touches env state and never lets an exception escape into the draw loop.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from legged_gym.scripts.input_source import DriveEnvelope, MergedSource
from legged_gym.utils.terrain import (
    MOE_SHOWCASE_LEVELS,
    moe_showcase_columns,
    moe_showcase_levels_for_column,
    teleport_env_to_taxonomy_tile,
)


# ---------------------------------------------------------------------------
# Keymap
# ---------------------------------------------------------------------------
#
# The arena deliberately takes over W/A/S/D from Genesis's default debug
# controls.  Interactive driving is the primary use of native play, and a
# familiar WASD layout is more valuable here than those viewer toggles.  The
# bindings below are registered with ``overwrite=True`` so the RELEASE actions
# cannot also fire Genesis's old W/A/S/D callbacks.  I remains protected by the
# pyrender help overlay and is intentionally not used.
#
# name -> (Key attribute name, drive action)
DRIVE_KEYMAP: Dict[str, Tuple[str, str]] = {
    "arena_forward": ("W", "forward"),
    "arena_backward": ("S", "backward"),
    "arena_strafe_left": ("A", "strafe_left"),
    "arena_strafe_right": ("D", "strafe_right"),
    "arena_yaw_left": ("Q", "yaw_left"),
    "arena_yaw_right": ("E", "yaw_right"),
}

# name -> (Key attribute name, event) fired once per press.
ACTION_KEYMAP: Dict[str, Tuple[str, str]] = {
    "arena_tab": ("TAB", "tab_next"),          # Shift+Tab -> tab_prev (see below)
    "arena_level_prev": ("BRACKETLEFT", "level_prev"),
    "arena_level_next": ("BRACKETRIGHT", "level_next"),
    "arena_stop": ("SPACE", "stop"),
    "arena_respawn": ("BACKSPACE", "respawn"),
    "arena_follow": ("T", "camera_next"),
    "arena_legend": ("M", "legend"),
    "arena_model_next": ("N", "model_next"),
    # J and G are free: the Genesis-owned set above stops at F11/I and no
    # default plugin binds either of them.
    "arena_gamepad_toggle": ("J", "gamepad_toggle"),
    "arena_camera_snap": ("G", "camera_snap"),
}

ARENA_EVENTS = (
    "tab_next",
    "tab_prev",
    "level_next",
    "level_prev",
    "stop",
    "respawn",
    "camera_next",
    "camera_snap",
    "legend",
    "model_next",
    "model_prev",
    "gamepad_toggle",
)


# ---------------------------------------------------------------------------
# GTA-style orbit camera
# ---------------------------------------------------------------------------
#
# ``gta`` is a spherical rig -- (azimuth, pitch, distance) -- around the SAME
# smoothed body anchor the rear/front modes use, so it inherits the anti-bob
# filtering (fixed world Z, rate-limited XY) for free.
#
# The one thing that is deliberately NOT inherited is heading: the azimuth is an
# ABSOLUTE WORLD ANGLE and never tracks the robot's yaw.  Auto-recentring behind
# the robot is exactly what makes it impossible to look at the robot's front,
# which is the whole reason this mode exists.  Recentring happens only on
# explicit request (R3 / G / a teleport).
#
# Convention: ``azimuth`` is the world-frame bearing FROM the robot TO the eye,
# so "behind the robot" is ``base_yaw + pi``; ``pitch`` is the eye's elevation
# above the anchor (positive = camera above, looking down).
CAMERA_MODES = ("gta", "rear", "front", "free")
#: Modes that write a robot-relative camera pose every control frame.
FOLLOW_CAMERA_MODES = frozenset({"gta", "rear", "front"})

GTA_DISTANCE = 4.5                     # m, default orbit radius
GTA_DISTANCE_LIMITS = (1.5, 12.0)      # m, scroll-wheel clamp
GTA_PITCH = math.radians(16.0)         # default elevation: slightly above
GTA_PITCH_LIMITS = (math.radians(-25.0), math.radians(70.0))
GTA_AZIMUTH_RATE = 2.5                 # rad/s at full right-stick deflection
GTA_PITCH_RATE = 1.5                   # rad/s at full right-stick deflection
GTA_MOUSE_AZIMUTH_PER_PX = 0.006       # rad per pixel of drag
GTA_MOUSE_PITCH_PER_PX = 0.005         # rad per pixel of drag
GTA_SCROLL_STEP = 0.12                 # fraction of distance per wheel notch
GTA_LOOKAT_HEIGHT = 0.35               # m above the anchor the camera aims at
#: Upper bound on the dt used to integrate look rates.  ``update_follow_camera``
#: measures its own dt from the wall clock, and a stall (checkpoint hot-swap, a
#: long teleport) would otherwise turn one frame into a huge camera whip.
GTA_MAX_LOOK_DT = 0.1


def default_tabs_for_cfg(terrain_cfg) -> List[str]:
    """Tab (= terrain-type column) names for a play terrain config.

    Derived, never hard-coded: the moe exhibit's columns come from
    ``moe_showcase_columns(terrain_proportions)``, so changing the proportions
    changes the tabs automatically (types with zero weight simply have no
    column and therefore no tab).
    """
    if getattr(terrain_cfg, "moe_showcase", False):
        proportions = getattr(terrain_cfg, "terrain_proportions", None)
        if proportions:
            return [name for name, _ in moe_showcase_columns(proportions)]
    labels = getattr(terrain_cfg, "taxonomy_labels", None)
    if labels:
        return [str(x) for x in labels]
    return [f"col {j}" for j in range(int(getattr(terrain_cfg, "num_cols", 1)))]


def default_levels_for_cfg(terrain_cfg) -> List[int]:
    """Difficulty label per grid row (row index -> training level number)."""
    if getattr(terrain_cfg, "moe_showcase", False):
        levels = getattr(terrain_cfg, "moe_showcase_levels", MOE_SHOWCASE_LEVELS)
        return [int(x) for x in levels]
    return list(range(int(getattr(terrain_cfg, "num_rows", 1))))


# ---------------------------------------------------------------------------
# On-screen HUD
# ---------------------------------------------------------------------------
#
# The HUD is painted by ``_HudPlugin``, appended to the pyrender viewer's
# ``plugins`` list -- NOT through ``viewer_flags["caption"]``.  Why: the
# vendored viewer's ``on_draw`` (``genesis/ext/pyrender/viewer.py``) runs its
# caption loop BEFORE ``plugin.on_draw()``, so anything a plugin draws lands
# on top of caption text.  Amber-on-frame is illegible when the camera faces
# the white terrain, so every line needs a black backplate *behind* it -- which
# is only possible if the plugin draws BOTH the rect and the text.  The caption
# slot is therefore left completely untouched (whoever owned it keeps it).
#
# Two hard limits survive from the caption path because the text still goes
# through ``renderer.render_text``:
#
# * ONE LINE per string, ASCII ONLY.  ``font.py`` builds its glyph map from
#   ``chr(0..127)`` and ``render_string`` walks the string character by
#   character, so a ``\n`` or any non-ASCII glyph raises ``KeyError`` -- and it
#   would raise on the *viewer* thread, taking the render loop down with it.
# * Placement is one of the nine fixed ``TextAlign`` anchors; there is no free
#   x/y, so stacked lines mean separate anchors.
#
# Genesis already owns TOP_LEFT (its key-instruction list) and BOTTOM_RIGHT
# (transient messages) in ``_render_help_text``.  The HUD takes TOP_CENTER
# (status), BOTTOM_LEFT (telemetry) and TOP_RIGHT (a SHORT hint) and leaves
# both of those alone.  BOTTOM_CENTER is deliberately unused: the long hint
# that used to live there shared the bottom band with the telemetry and the
# two lines rendered on top of each other into unreadable mush.
HUD_REFRESH_HZ = 8.0          # rebuilding formatted strings at 60 Hz is waste
HUD_FONT_NAME = "UbuntuMono-Regular"   # monospace: telemetry columns stay put
# 26 pt is Genesis's own FONT_SIZE and is glyph-safe (see hud_font_is_safe --
# UbuntuMono at 20 pt is NOT).  Fallbacks are tried in order.
HUD_FONT_CANDIDATES = (
    ("UbuntuMono-Regular", 26),
    ("UbuntuMono-Regular", 40),
    ("OpenSans-Regular", 26),
)
HUD_FONT_PT = 26
HUD_COLOR = (1.0, 0.85, 0.15, 1.0)     # amber reads well on the black backplate
# Backplate: solid black at ~2/3 opacity, padded a few px past the glyph ink.
# Color/opacity are pyglet's 0-255 ints (pyglet.shapes.Rectangle).
HUD_BG_COLOR = (0, 0, 0)
HUD_BG_OPACITY = 160
HUD_PAD_X = 6.0    # horizontal padding between glyph ink and backplate edge
HUD_PAD_Y = 4.0    # vertical padding


def hud_font_is_safe(font_name: str, font_pt: int) -> bool:
    """True when drawing this (font, size) pair cannot crash the viewer thread.

    ``genesis/ext/pyrender/font.py`` builds a GL texture for EVERY glyph
    ``chr(0..127)`` the first time a (font, size) pair is drawn, regardless of
    which characters are actually in the string.  A glyph that rasterises
    exactly one pixel tall -- ``_`` in UbuntuMono at 20 pt, for instance -- is
    squeezed to a 1-D array by ``format_texture_source``, and
    ``Texture._add_to_context`` then raises ``IndexError`` on
    ``source.shape[1]``.  That happens *inside the draw loop*, on the viewer
    thread, and takes rendering down for the rest of the session.

    So probe the pair once, up front, on the sim thread, where an exception is
    survivable.  Building the ``Font`` rasterises the glyphs but touches no GL
    context (only ``_add_to_context`` does), so this is safe to call anywhere.
    """
    try:
        import os

        from genesis.ext.pyrender import font as pyrender_font

        # FontCache.get_font() keys its cache on the live GL context, so it
        # cannot be called from the sim thread.  Resolve the .ttf the same way
        # it does and build the Font directly instead -- rasterising glyphs
        # needs no GL.
        path = font_name
        if not os.path.isfile(path):
            fonts_dir = os.path.join(
                os.path.dirname(os.path.realpath(pyrender_font.__file__)), "fonts")
            path = os.path.join(fonts_dir, f"{font_name}.ttf")
        if not os.path.isfile(path):
            return False
        font = pyrender_font.Font(path, int(font_pt))
        for character in font._character_map.values():
            source = getattr(character.texture, "source", None)
            if source is not None and getattr(source, "ndim", 2) < 2:
                return False
        return True
    except Exception:
        return False


def _ascii_only(text: str) -> str:
    """Drop anything the pyrender glyph map cannot render (chr(0..127) only).

    A stray non-ASCII character would raise ``KeyError`` inside the viewer
    thread's draw call, so this is a hard safety net rather than cosmetics.
    """
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in str(text))


def format_hud_lines(
    tabs: Sequence[str],
    levels: Sequence[int],
    tab_index: int,
    level_index: int,
    command: Optional[Sequence[float]] = None,
    achieved: Optional[Sequence[float]] = None,
    fps: Optional[float] = None,
    realtime: Optional[float] = None,
    device: str = "keyboard",
    following: bool = False,
    camera_mode: Optional[str] = None,
    camera_detail: str = "",
) -> List[str]:
    """The four single-line HUD strings: status, commanded, achieved, hint.

    Line 3 is deliberately just the two keys worth advertising (T cycles the
    camera, M prints the legend).  The full key list already lives in the
    terminal legend (M) and in Genesis's I overlay, and the long hint
    that used to sit at BOTTOM_CENTER overlapped the BOTTOM_LEFT telemetry
    into unreadable mush.  TOP_RIGHT only stays collision-free BECAUSE the
    hint is a handful of characters: a centered status line (~45 chars) and a
    ~7-char right-anchored hint cannot touch at window widths >= ~1024 px.
    """
    n_tabs = max(len(tabs), 1)
    n_lvls = max(len(levels), 1)
    ti = min(max(int(tab_index), 0), n_tabs - 1)
    li = min(max(int(level_index), 0), n_lvls - 1)
    name = tabs[ti] if tabs else "col 0"
    level = levels[li] if levels else 0

    if camera_mode is None:
        camera_mode = "front" if following else "free"
    # ``camera_detail`` is the gta rig's live orbit state (``d4.5``).  Kept to a
    # handful of characters on purpose: the status line is TOP_CENTER and the
    # TOP_RIGHT hint only stays clear of it because the status stays short (see
    # the docstring above and _install_hud).
    cam = f"cam {camera_mode}"
    if camera_detail:
        cam += f" {camera_detail}"
    status = (f"tab {ti + 1}/{n_tabs} {name}   "
              f"level L{level} (row {li + 1}/{n_lvls})   "
              f"{cam}")

    vx, vy, yaw = (list(command) + [0.0, 0.0, 0.0])[:3] if command is not None \
        else (0.0, 0.0, 0.0)
    commanded = (f"commanded vx: {float(vx):+.2f}  vy: {float(vy):+.2f}  "
                 f"yaw: {float(yaw):+.2f}   in: {device}")
    avx, avy, ayaw = (list(achieved) + [0.0, 0.0, 0.0])[:3] \
        if achieved is not None else (0.0, 0.0, 0.0)
    achieved_line = (f"achieved  vx: {float(avx):+.2f}  vy: {float(avy):+.2f}  "
                     f"yaw: {float(ayaw):+.2f}")
    if fps is not None:
        achieved_line += f"   {float(fps):5.1f} FPS"
    if realtime is not None:
        achieved_line += f"   {float(realtime):.2f}x RT"

    hint = "T: cam  M: keys"

    return [_ascii_only(status), _ascii_only(commanded),
            _ascii_only(achieved_line), _ascii_only(hint)]


def format_arena_legend(
    tabs: Sequence[str],
    levels: Sequence[int],
    tab_index: int = 0,
    level_index: int = 0,
    gamepad: str = "",
    backend: str = "genesis",
) -> str:
    """Console legend: tabs, levels and every keybinding."""
    backend_name = "MuJoCo" if backend == "mujoco" else "Genesis"
    head = f"Play arena — native {backend_name} viewer"
    lines = [head, "=" * len(head)]
    lines.append("  tabs (terrain type, Tab / Shift+Tab):")
    for j, name in enumerate(tabs):
        mark = "*" if j == int(tab_index) else " "
        lines.append(f"    {mark} [{j}] {name}")
    level_str = "  ".join(
        ("*" if i == int(level_index) else " ") + f"[{i}] L{lvl}"
        for i, lvl in enumerate(levels)
    )
    lines.append(f"  levels (difficulty, [ / ]): {level_str}")
    lines.append("  drive (robot body frame -- NOT camera-relative):")
    lines.append("    W / S         forward / backward")
    lines.append("    A / D         strafe left / right")
    lines.append("    Q / E         turn left / right")
    lines.append("    Shift + W/S   turbo forward/backward: 2.0 m/s")
    lines.append("    Space         hard stop")
    lines.append("  camera:")
    lines.append("    T                 mode: gta -> rear -> front -> free")
    other_camera = "MuJoCo free camera" if backend == "mujoco" else "Genesis trackball"
    lines.append("    mouse drag        gta: orbit (azimuth / pitch); other modes: "
                 + other_camera)
    lines.append("    mouse wheel       gta: zoom (distance); other modes: trackball")
    lines.append("    G                 snap the camera behind the robot")
    lines.append("    (gta azimuth is a free world angle: it never auto-recenters, so")
    lines.append("     you can orbit around and look at the robot's front.)")
    lines.append("  arena:")
    lines.append("    Tab / Shift+Tab   next / previous terrain tab")
    lines.append("    [ / ]             easier / harder level in this tab")
    lines.append("    Backspace         respawn on the current tile")
    lines.append("    N                 hot-swap to the next checkpoint of this run")
    lines.append("    J                 toggle gamepad on/off (auto-rescans when on)")
    lines.append("    M                 reprint this legend")
    if backend == "genesis":
        lines.append("  (Genesis reserves I for its own help overlay; W/A/S/D are assigned to drive.)")
    lines.append("  gamepad: auto-detected at startup, no launch flag needed; a")
    lines.append("    hot-plugged pad is picked up within ~2 s; J toggles it on/off")
    lines.append(f"    connected: {gamepad if gamepad else 'none'}")
    lines.append("    left stick   vx / vy (body frame)   LT + stick  turbo vx ±2.0 m/s")
    lines.append("    LB / RB      yaw left / right")
    lines.append("    right stick  camera look            R3       snap camera behind")
    lines.append("    D-pad L/R  tab    D-pad U/D  level    A  respawn")
    lines.append("    Y  next camera mode  START  hard stop  BACK  legend")
    return "\n".join(lines)


class _HudPlugin:
    """pyrender-viewer plugin that paints the HUD: black backplate + text.

    Appended directly to ``window.plugins`` rather than registered through
    ``register_plugin`` -- registration would also ``push_handlers`` and start
    dispatching pyglet window events to an object that has nothing to do with
    them.  The vendored viewer only ever calls ``on_draw`` (every frame),
    ``update_on_sim_step`` and ``on_close`` on entries of that list, and
    ``genesis.vis.viewer``'s rebuild/step polling walks a *different* list
    (``_viewer_plugins``), so the two no-ops below cover the whole surface this
    object can be reached through.  Duck-typed instead of a ``ViewerPlugin``
    subclass so that importing this module needs no Genesis import at all.

    Everything GL happens in ``on_draw``: the viewer thread holds the context
    current there (it is the same context ``renderer.render_text`` uses), which
    is also the only place ``FontCache.get_font`` may be called -- that cache is
    keyed on the LIVE GL context.  Positions are recomputed from
    ``_location_to_x_y`` every frame, so window resizes need no event handling.

    Any failure disables the layer permanently (``_broken``) instead of raising
    into the draw loop: an exception on the viewer thread would take rendering
    down for the rest of the session.
    """

    def __init__(self, window, lines, anchors, font_name, font_pt):
        self._window = window
        # The exact list NativeArenaUI.update_hud swaps strings into on the sim
        # thread; read-only here, and never reallocated by either side.
        self._lines = lines
        self._anchors = anchors
        self._font_name = font_name
        self._font_pt = font_pt
        self._batch = None      # built lazily on first draw (needs a GL context)
        self._rects = None
        self._broken = False

    # -- the rest of the plugin surface the viewer can call -------------------
    def update_on_sim_step(self):
        pass

    def on_close(self):
        pass

    # -- drawing ---------------------------------------------------------------
    def on_draw(self):
        if self._broken:
            return
        try:
            self._draw()
        except Exception as exc:
            # NOTHING may escape into the viewer thread.  One message, then the
            # HUD layer switches itself off rather than spamming once a frame.
            self._broken = True
            print(f"[arena] on-screen HUD disabled after draw failure: {exc}")

    def _draw(self):
        import pyglet  # lazy: importing native_ui must stay display-free
        from pyglet.gl import GL_DEPTH_TEST, glDisable
        from genesis.ext.pyrender.constants import TextAlign

        window = self._window
        renderer = window._renderer
        if renderer is None:
            return  # viewer not fully started yet; try again next frame
        if self._batch is None:
            self._batch = pyglet.graphics.Batch()
            self._rects = [
                pyglet.shapes.Rectangle(
                    0, 0, 0, 0, color=HUD_BG_COLOR, batch=self._batch
                )
                for _ in self._anchors
            ]
            for rect in self._rects:
                rect.opacity = HUD_BG_OPACITY

        # Rects first, then texts, every frame, from the cached strings.
        # Depth test is killed explicitly: the scene pass may have left it on
        # when Genesis's own help text is disabled, and the HUD must win the
        # depth fight against the rendered terrain either way.
        glDisable(GL_DEPTH_TEST)
        placements = []
        bottom_left_seen = 0
        bottom_left_total = sum(anchor == TextAlign.BOTTOM_LEFT
                                for anchor in self._anchors)
        for rect, anchor, text in zip(self._rects, self._anchors, self._lines):
            if not text:
                rect.width = rect.height = 0.0
                continue
            x, y = window._location_to_x_y(anchor)
            if anchor == TextAlign.BOTTOM_LEFT:
                # Multiple telemetry rows share one aligned column.  Earlier
                # rows sit above later rows; each remains a newline-free string
                # because pyrender's glyph renderer cannot safely draw '\n'.
                rows_below = bottom_left_total - bottom_left_seen - 1
                y += rows_below * (self._font_pt + 2 * HUD_PAD_Y + 4)
                bottom_left_seen += 1
            rect.x, rect.y, rect.width, rect.height = self._backplate(
                renderer, x, y, text, anchor)
            placements.append((text, x, y, anchor))
        self._batch.draw()
        for text, x, y, anchor in placements:
            renderer.render_text(
                text, x, y,
                font_name=self._font_name, font_pt=self._font_pt,
                color=HUD_COLOR, scale=1.0, align=anchor,
            )

    def _backplate(self, renderer, x, y, text, anchor):
        """Rect (left, bottom, width, height) covering the drawn glyphs.

        Mirrors ``font.py:render_string``'s placement math: *_LEFT pins the
        left glyph edge to ``x``, *_RIGHT the right edge, *_CENTER the middle;
        TOP_* puts the glyph tops at ``y`` (baseline at ``y - ascender``),
        BOTTOM_* puts the baseline at ``y``, and CENTER_* halves it.  The rect
        is the union box padded by HUD_PAD_X / HUD_PAD_Y.
        """
        from genesis.ext.pyrender.constants import TextAlign

        width, asc, desc = self._measure(renderer, text)
        if anchor in (TextAlign.TOP_LEFT, TextAlign.CENTER_LEFT,
                      TextAlign.BOTTOM_LEFT):
            left = x
        elif anchor in (TextAlign.TOP_RIGHT, TextAlign.CENTER_RIGHT,
                        TextAlign.BOTTOM_RIGHT):
            left = x - width
        else:
            left = x - width / 2.0
        if anchor in (TextAlign.TOP_LEFT, TextAlign.TOP_CENTER,
                      TextAlign.TOP_RIGHT):
            baseline = y - asc
        elif anchor in (TextAlign.CENTER, TextAlign.CENTER_LEFT,
                        TextAlign.CENTER_RIGHT):
            baseline = y - asc / 2.0
        else:
            baseline = y
        return (left - HUD_PAD_X, baseline - desc - HUD_PAD_Y,
                width + 2.0 * HUD_PAD_X, asc + desc + 2.0 * HUD_PAD_Y)

    def _measure(self, renderer, text):
        """(width, ascender, descender) of ``text`` in logical pixels.

        Reads glyph advances from the same ``Font`` object ``render_text``
        draws with (``advance`` is 26.6 fixed-point, hence ``>> 6`` -- the same
        shift ``render_string`` uses).  ``render_text`` internally multiplies
        font_pt by ``renderer.dpscale``; freetype advances scale linearly with
        pt size, so measuring at the UNSCALED pt yields logical-pixel metrics
        that match both ``_location_to_x_y`` and pyglet's window projection.

        Falls back to a monospace estimate when metrics are unavailable (font
        cache miss, a Genesis refactor, ...): the rect may then be a few px
        off, but drawing never fails.
        """
        try:
            font = renderer._font_cache.get_font(self._font_name, self._font_pt)
            width, asc, desc = 0.0, 0.0, 0.0
            cmap = font._character_map
            for ch in text:
                glyph = cmap.get(ch)  # strings are _ascii_only'd upstream
                if glyph is None:
                    continue
                width += glyph.advance >> 6
                asc = max(asc, glyph.bearing[1])
                desc = max(desc, glyph.size[1] - glyph.bearing[1])
            if width > 0.0:
                return width, asc, desc
        except Exception:
            pass
        pt = float(self._font_pt)
        return len(text) * pt * 0.6, pt, pt * 0.3


class NativeArenaUI:
    """Terrain-tab arena bound to the Genesis native viewer.

    Constructing this never fails on a headless run: when ``scene.viewer`` is
    ``None`` no keybinds are registered and the object still works as an event
    sink for the gamepad.
    """

    def __init__(
        self,
        env,
        source: MergedSource,
        robot_index: int = 0,
        envelope: Optional[DriveEnvelope] = None,
        tabs: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[int]] = None,
        model_swap_cb: Optional[Callable[[int], None]] = None,
        quiet: bool = False,
    ) -> None:
        self.env = env
        self.source = source
        self.robot_index = int(robot_index)
        self.envelope = envelope or source.envelope
        self.model_swap_cb = model_swap_cb
        self.quiet = bool(quiet)

        terrain_cfg = getattr(getattr(env, "cfg", None), "terrain", None)
        self.tabs: List[str] = list(
            tabs if tabs is not None else default_tabs_for_cfg(terrain_cfg)
        ) or ["col 0"]
        self.levels: List[int] = list(
            levels if levels is not None else default_levels_for_cfg(terrain_cfg)
        ) or [0]

        self.tab_index = 0
        self.level_index = 0
        # T cycles CAMERA_MODES.  ``free`` deliberately writes no camera pose:
        # Genesis's native trackball then owns the view completely.  ``gta`` is
        # the startup default because it is the only mode that can be aimed.
        self.camera_mode = CAMERA_MODES[0]
        # gta writes a pose from the very first frame, so the arena owns the
        # camera immediately; play.py's --follow_robot must not also write one
        # (two writers per frame = a camera that visibly fights itself).  This
        # used to flip only on a T press, back when the default was ``free``.
        self._camera_override_active = True
        # Smoothed world anchor: fast enough in XY to follow locomotion, but
        # intentionally slow in Z so the camera does not inherit body bounce
        # from individual foot impacts.  Yaw gets its own wrapped filter.
        self._camera_anchor: Optional[np.ndarray] = None
        self._camera_yaw: Optional[float] = None
        self._camera_height: Optional[float] = None
        viewer_cfg = getattr(getattr(env, "cfg", None), "viewer", None)
        self._follow_camera_stride = max(
            int(getattr(viewer_cfg, "follow_camera_stride", 1)), 1
        )
        self._follow_camera_frame = 0
        # gta orbit rig.  ``None`` azimuth means "snap behind the robot on the
        # next update" -- the state used at startup, after a teleport, and after
        # an explicit snap request.  Once set it is a free world angle that
        # nothing but the user changes.
        self._camera_azimuth: Optional[float] = None
        self._camera_pitch = float(GTA_PITCH)
        self._camera_distance = float(GTA_DISTANCE)
        self._camera_snap_pending = False
        # Mouse-look ingress.  The pyglet handlers below run on the VIEWER
        # thread and may only stash numbers here; the rig is integrated on the
        # sim thread in update_follow_camera, exactly like the drive keys are
        # only latched by the keybind callbacks and ramped in poll().  Anything
        # else would call env.set_viewer_camera off-thread.
        self._look_lock = threading.Lock()
        self._mouse_look = [0.0, 0.0]   # accumulated drag pixels (dx, dy)
        self._mouse_scroll = 0.0        # accumulated wheel notches
        self._look_clock: Optional[float] = None
        self._shift_held = False
        self._shift_keys = set()
        self._lock = threading.Lock()
        self._registered: List[str] = []
        self._focus_guard_window = None
        self._look_window = None
        # HUD state (see _install_hud / update_hud)
        self._hud_window = None
        self._hud_plugin: Optional[_HudPlugin] = None
        self._hud_lines: Optional[List[str]] = None
        self._hud_last = 0.0
        # Which viewer backend the ingress points dispatch on; set by
        # _find_viewer.  Pre-seeded to None so that a half-built arena (the unit
        # tests wire a fake viewer in by hand, bypassing _find_viewer) still has
        # the attribute, and so that "unknown" falls through to the Genesis
        # bodies -- those are the ones that existed first and must not move.
        self._viewer_backend: Optional[str] = None

        self.viewer = self._find_viewer()
        if self.viewer is not None:
            self._register_keybinds()

    # -- wiring --------------------------------------------------------------
    def _find_viewer(self):
        """Locate the play window, whichever physics backend is running.

        Genesis hangs its viewer off the built scene (``simulator._scene.viewer``);
        the MuJoCo simulator owns its window directly (``simulator._viewer``,
        None when headless).  Genesis wins when both somehow exist because it is
        the backend every ingress body below was written against.

        Provenance -- which ATTRIBUTE the object came out of -- is what selects
        the backend, never ``isinstance``: this module has to stay importable
        with neither genesis nor mujoco installed (play.py imports it before it
        knows which simulator it will build), so it may not name either class.
        """
        simulator = getattr(self.env, "simulator", None)
        scene = getattr(simulator, "_scene", None)
        viewer = getattr(scene, "viewer", None)
        if viewer is not None:
            self._viewer_backend = "genesis"
            return viewer
        viewer = getattr(simulator, "_viewer", None)
        if viewer is not None:
            self._viewer_backend = "mujoco"
            return viewer
        self._viewer_backend = None
        return None

    def _register_keybinds(self) -> None:
        if self._viewer_backend == "mujoco":
            self._register_keybinds_mujoco()
            return
        try:
            from genesis.vis.keybindings import Key, KeyAction, Keybind
        except Exception as exc:  # pragma: no cover - depends on Genesis build
            print(f"[arena] keybinds unavailable ({exc}); keyboard drive is off")
            self.viewer = None
            return

        keybinds = []
        for name, (key_name, action) in DRIVE_KEYMAP.items():
            key = getattr(Key, key_name)
            keybinds.append(
                Keybind(
                    f"{name}_press",
                    key,
                    key_action=KeyAction.PRESS,
                    callback=self.source.keyboard.press,
                    args=(action,),
                )
            )
            keybinds.append(
                Keybind(
                    f"{name}_release",
                    key,
                    key_action=KeyAction.RELEASE,
                    callback=self.source.keyboard.release,
                    args=(action,),
                )
            )

        for name, (key_name, event) in ACTION_KEYMAP.items():
            keybinds.append(
                Keybind(
                    name,
                    getattr(Key, key_name),
                    key_action=KeyAction.PRESS,
                    callback=self._on_action_key,
                    args=(event,),
                )
            )

        # Shift is tracked by hand rather than through ``key_mods``: pyglet folds
        # NUMLOCK / CAPSLOCK into the modifier bitmask, and Genesis hashes
        # keybinds on the exact bitmask, so a Shift+Tab bind silently stops
        # matching whenever numlock happens to be on.
        for side in ("LSHIFT", "RSHIFT"):
            keybinds.append(
                Keybind(
                    f"arena_{side.lower()}_press",
                    getattr(Key, side),
                    key_action=KeyAction.PRESS,
                    callback=self._set_shift,
                    args=(side, True),
                )
            )
            keybinds.append(
                Keybind(
                    f"arena_{side.lower()}_release",
                    getattr(Key, side),
                    key_action=KeyAction.RELEASE,
                    callback=self._set_shift,
                    args=(side, False),
                )
            )

        try:
            # W/A/S/D replace the native viewer's RELEASE debug bindings; the
            # matching RELEASE drive bindings must replace them too, otherwise
            # steering also toggles viewer state when a key is released.
            self.viewer.register_keybinds(*keybinds, overwrite=True)
        except Exception as exc:
            print(f"[arena] could not register keybinds ({exc}); keyboard drive is off")
            self.viewer = None
            return
        self._registered = [kb.name for kb in keybinds]
        self._install_focus_guard()
        self._install_look_handlers()
        self._install_hud()

    def _register_keybinds_mujoco(self) -> None:
        """One key callback instead of ~30 Keybind objects.

        ``MujocoViewer.set_key_callback(cb)`` delivers ``cb(key_name, pressed)``
        for BOTH edges and suppresses auto-repeat, which is exactly the two
        properties the Genesis path buys with a PRESS Keybind plus a RELEASE
        Keybind per key.  The viewer's key names were deliberately chosen to be
        the very strings in DRIVE_KEYMAP / ACTION_KEYMAP, so there is no
        translation table here and nothing to keep in sync when a key is added.

        THREADING -- this is the one real difference from Genesis.  Genesis runs
        its pyglet viewer on a background thread, so its callbacks land on the
        *viewer* thread and may only latch (see the module docstring).  The
        MuJoCo viewer is pumped synchronously from the sim thread, so this
        callback is already on the thread that is allowed to touch env state.
        It still goes through the same latch-only path on purpose: the locks in
        KeyboardSource and around ``_mouse_look`` must stay correct for Genesis,
        they cost nothing uncontended, and one shared discipline means arena
        behaviour cannot quietly diverge between the two backends.

        Shift is still tracked by hand (see the Genesis body): the reason there
        was pyglet's modifier bitmask, but the property that matters -- Shift+Tab
        resolving inside ``_on_action_key`` rather than through a second bound
        combination -- is what keeps Shift+Tab working identically on a backend
        that reports no modifier state at all.
        """
        # Reverse lookups: key NAME -> what it does.  Built per install rather
        # than at module scope so the keymap tables stay the single source of
        # truth and no new module-level name appears.
        drive_by_key = {key: action for key, action in DRIVE_KEYMAP.values()}
        action_by_key = {key: event for key, event in ACTION_KEYMAP.values()}

        def on_key(key_name: str, pressed: bool) -> None:
            try:
                if key_name in ("LSHIFT", "RSHIFT"):
                    self._set_shift(key_name, pressed)
                    return
                action = drive_by_key.get(key_name)
                if action is not None:
                    if pressed:
                        self.source.keyboard.press(action)
                    else:
                        self.source.keyboard.release(action)
                    return
                if not pressed:
                    return  # arena actions fire once, on the press edge only
                event = action_by_key.get(key_name)
                if event is not None:
                    # Resolves Shift+Tab -> tab_prev from the hand-tracked flag.
                    self._on_action_key(event)
            except Exception as exc:  # pragma: no cover - must never kill input
                print(f"[arena] key {key_name!r} failed: {exc}")

        try:
            self.viewer.set_key_callback(on_key)
        except Exception as exc:
            print(f"[arena] keybinds unavailable ({exc}); keyboard drive is off")
            self.viewer = None
            return
        # ``_registered`` stays empty: it exists to feed viewer.remove_keybind()
        # in unregister(), and this backend has a single callback slot with no
        # per-binding names to remove.
        self._registered = []
        self._install_focus_guard()
        self._install_look_handlers()
        self._install_hud()

    # -- HUD -----------------------------------------------------------------
    def _install_hud(self) -> None:
        """Attach the on-screen HUD plugin.  Silent no-op when unavailable.

        Allocates the shared line list ONCE; the plugin reads it on the viewer
        thread every frame and :meth:`update_hud` only swaps the strings in
        place.  The list must never be reallocated -- a string swap is a benign
        race, replacing the list under an active iteration is not.  The
        viewer's ``viewer_flags["caption"]`` slot is deliberately never
        touched: the caption loop runs BEFORE plugin drawing, so caption text
        could never sit on top of a backplate anyway (see _HudPlugin).

        On MuJoCo the viewer paints the overlay itself, so there is no plugin,
        no font probe and no backplate math -- only the same shared list handed
        over once.
        """
        if self._viewer_backend == "mujoco":
            # Same contract as the plugin, one indirection shorter: the viewer
            # KEEPS THIS EXACT LIST OBJECT and reads it every frame it draws, so
            # update_hud must keep swapping strings into it IN PLACE.  A
            # reallocation here would leave the viewer painting the original
            # (now orphaned) list forever -- the HUD would freeze on its startup
            # text.  That is the same reason _HudPlugin never reallocates.
            self._hud_lines = ["", "", "", ""]
            try:
                self.viewer.set_hud_lines(self._hud_lines)
            except Exception as exc:
                print(f"[arena] on-screen HUD unavailable ({exc}); terminal only")
                self._hud_lines = None
                return
            # No pyrender window and no plugin on this path; _remove_hud's
            # early-out covers exactly that shape, so teardown stays a no-op.
            self.update_hud(force=True)
            return
        window = getattr(self.viewer, "_pyrender_viewer", None)
        if window is None:
            return
        try:
            from genesis.ext.pyrender.constants import TextAlign
        except Exception as exc:  # pragma: no cover - depends on Genesis build
            print(f"[arena] on-screen HUD unavailable ({exc}); terminal only")
            return
        # Refuse to install a (font, size) pair whose glyph atlas would raise
        # inside the draw loop; see hud_font_is_safe.
        font = next(
            ((name, pt) for name, pt in HUD_FONT_CANDIDATES
             if hud_font_is_safe(name, pt)),
            None,
        )
        if font is None:
            print("[arena] on-screen HUD disabled: no usable font/size on this "
                  "Genesis build; use M for the terminal legend")
            return
        font_name, font_pt = font
        # TOP_LEFT / BOTTOM_RIGHT belong to Genesis's own help + message text.
        # Status and hint share the top band but cannot collide: the status is
        # centered (~700 px wide at 26 pt) while the hint is a ~7-char stub at
        # the far right, ~350 px clear of it even at 1024 px window width.
        anchors = (TextAlign.TOP_CENTER, TextAlign.BOTTOM_LEFT,
                   TextAlign.BOTTOM_LEFT, TextAlign.TOP_RIGHT)
        self._hud_lines = ["", "", "", ""]
        plugin = _HudPlugin(window, self._hud_lines, anchors, font_name, font_pt)
        try:
            window.plugins.append(plugin)
        except Exception as exc:  # pragma: no cover - depends on Genesis build
            print(f"[arena] on-screen HUD unavailable ({exc}); terminal only")
            self._hud_lines = None
            return
        self._hud_window = window
        self._hud_plugin = plugin
        self.update_hud(force=True)

    def _remove_hud(self) -> None:
        window = self._hud_window
        plugin = self._hud_plugin
        self._hud_window = None
        self._hud_plugin = None
        self._hud_lines = None
        if window is None or plugin is None:
            return
        # List removal is atomic under the GIL; the worst race is one extra
        # frame drawn from the stale (but still valid) strings.  The caption
        # slot was never touched on install, so there is nothing to restore.
        try:
            window.plugins.remove(plugin)
        except Exception:
            pass

    def hud_due(self) -> bool:
        """True when the next :meth:`update_hud` would actually rebuild text.

        Lets callers skip gathering HUD inputs (reading ``env.commands`` costs a
        GPU->CPU sync) on the ~7 frames out of 8 that would be thrown away.
        """
        if self._hud_lines is None:
            return False
        return (time.monotonic() - self._hud_last) >= (1.0 / HUD_REFRESH_HZ)

    def update_hud(
        self,
        command: Optional[Sequence[float]] = None,
        achieved: Optional[Sequence[float]] = None,
        fps: Optional[float] = None,
        realtime: Optional[float] = None,
        force: bool = False,
    ) -> bool:
        """Refresh the on-screen text.  Sim thread only; throttled internally.

        Returns True when the strings were actually rebuilt.  Safe (and cheap)
        to call every frame: the throttle is here so callers do not have to
        carry their own timer.
        """
        lines = self._hud_lines
        if lines is None:
            return False
        now = time.monotonic()
        if not force and (now - self._hud_last) < (1.0 / HUD_REFRESH_HZ):
            return False
        self._hud_last = now

        try:
            hud_levels = self.levels
            terrain_cfg = getattr(getattr(self.env, "cfg", None), "terrain", None)
            if getattr(terrain_cfg, "moe_showcase", False):
                hud_levels = list(self.levels)
                hud_levels[self.level_index] = self._display_level(
                    self.tab_index, self.level_index
                )
            new_lines = format_hud_lines(
                self.tabs, hud_levels, self.tab_index, self.level_index,
                command=command, achieved=achieved, fps=fps, realtime=realtime,
                device=self._device_label(),
                following=self.following,
                camera_mode=self.camera_mode,
                camera_detail=self.camera_detail(),
            )
        except Exception as exc:
            print(f"[arena] HUD text build failed: {exc}")
            return False
        # Swap in place: the plugin iterates this exact list on the viewer
        # thread, so it must never be reallocated (see _install_hud).
        for i, text in enumerate(new_lines):
            lines[i] = text
        return True

    def _device_label(self) -> str:
        """The ``in: ...`` HUD fragment: who currently owns the command.

        ``source.gamepad_enabled`` (the J toggle) gates the pad even while one
        is connected, so a present-but-disabled pad is called out explicitly
        instead of silently looking like "no pad".  The getattr default keeps
        the pre-toggle behaviour (pad credited whenever available) against an
        input_source that predates the toggle.
        """
        gamepad = getattr(self.source, "gamepad", None)
        if gamepad is None or not getattr(gamepad, "available", False):
            return "keyboard"
        if not getattr(self.source, "gamepad_enabled", True):
            return "keyboard (pad off)"
        if getattr(gamepad, "stick_active", False):
            return f"pad {gamepad.name}"
        return f"keyboard (pad: {gamepad.name})"

    def _install_focus_guard(self) -> None:
        """Drop every held drive key when the viewer window loses focus.

        ``pyrender.Viewer.on_deactivate`` clears Genesis's *own* ``_held_keys``
        but deliberately does not fire the RELEASE callbacks, so a key that was
        down when the user alt-tabbed away would stay latched in
        ``KeyboardSource`` and keep driving the robot with nobody touching the
        keyboard -- the exact stuck-key failure this module exists to remove.

        pyglet resolves event handlers as instance attributes before class
        methods, so assigning one here shadows the class method; it chains to
        the original so Genesis's bookkeeping still runs.  Only the lock-guarded
        pure-Python ``source.clear()`` is touched: this runs on the viewer
        thread and must not write ``env.commands`` (a device tensor).

        No-op on MuJoCo: that viewer's contract exposes no focus/deactivate
        hook to wrap.  Nothing to degrade gracefully *to*, so it stays silent
        rather than printing a warning once per launch; the worst case is the
        pre-existing stuck-key behaviour on alt-tab, which Space (hard stop)
        clears.
        """
        if self._viewer_backend == "mujoco":
            return
        window = getattr(self.viewer, "_pyrender_viewer", None)
        if window is None or "on_deactivate" in vars(window):
            return  # nothing to wrap, or already wrapped
        original = getattr(window, "on_deactivate", None)
        if original is None:
            return

        def on_deactivate():
            try:
                self.source.clear()
                self._shift_held = False
                self._shift_keys.clear()
            except Exception:
                pass
            return original()

        try:
            window.on_deactivate = on_deactivate
        except Exception as exc:  # pragma: no cover - depends on Genesis build
            print(f"[arena] focus guard unavailable ({exc})")
            return
        self._focus_guard_window = window

    def _install_look_handlers(self) -> None:
        """Mouse-look for keyboard users: drag = orbit, wheel = zoom, in gta only.

        Same shadowing trick as ``_install_focus_guard``: pyglet resolves event
        handlers as instance attributes before class methods, so assigning here
        wins over ``pyrender.Viewer.on_mouse_drag`` without patching Genesis.

        Two contracts hold this together:

        * THREADING -- these fire on the viewer thread.  They only accumulate
          plain floats under ``_look_lock``; the rig is integrated and
          ``env.set_viewer_camera`` is called on the SIM thread, from
          ``update_follow_camera``.  Touching the camera (or anything env-side)
          from here would be the same class of bug the drive keys avoid by
          latching instead of writing ``env.commands``.
        * OWNERSHIP -- in every mode except ``gta`` the event is passed straight
          through to the original handler, so Genesis's trackball behaves
          exactly as it always did.  In ``gta`` it is *swallowed* (return True =
          pyglet's EVENT_HANDLED) rather than chained: letting the trackball
          also process the drag would silently accumulate a pose that jumps out
          at the user the moment they cycle to ``free``.
        """
        if self._viewer_backend == "mujoco":
            self._install_look_handlers_mujoco()
            return
        window = getattr(self.viewer, "_pyrender_viewer", None)
        if window is None:
            return
        if "on_mouse_drag" in vars(window) or "on_mouse_scroll" in vars(window):
            return  # already wrapped
        original_drag = getattr(window, "on_mouse_drag", None)
        original_scroll = getattr(window, "on_mouse_scroll", None)
        if original_drag is None or original_scroll is None:
            return

        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if self.camera_mode != "gta":
                return original_drag(x, y, dx, dy, buttons, modifiers)
            try:
                with self._look_lock:
                    self._mouse_look[0] += float(dx)
                    self._mouse_look[1] += float(dy)
            except Exception:  # pragma: no cover - must never break the viewer
                pass
            return True

        def on_mouse_scroll(x, y, dx, dy):
            if self.camera_mode != "gta":
                return original_scroll(x, y, dx, dy)
            try:
                with self._look_lock:
                    self._mouse_scroll += float(dy)
            except Exception:  # pragma: no cover
                pass
            return True

        try:
            window.on_mouse_drag = on_mouse_drag
            window.on_mouse_scroll = on_mouse_scroll
        except Exception as exc:  # pragma: no cover - depends on Genesis build
            print(f"[arena] mouse-look unavailable ({exc}); pad/keys still work")
            return
        self._look_window = window

    def _install_look_handlers_mujoco(self) -> None:
        """Same two accumulators, fed by the MuJoCo viewer's callbacks.

        UNITS are the contract that keeps ``_advance_gta_rig`` untouched: the
        drag callback delivers pixel deltas with pyglet's sign convention (+x
        right, +y up) and the scroll callback wheel notches, which is exactly
        what ``on_mouse_drag`` / ``on_mouse_scroll`` stash above.  So this is a
        pass-through: no scaling, no negation, no per-backend constants.

        OWNERSHIP is the mirror of the Genesis branch.  There the non-gta event
        is chained to pyrender's trackball; here it is simply dropped, because
        the MuJoCo viewer already runs its own camera on the same drag and
        chaining is not ours to do -- the callback contract has no "handled"
        return.  Accumulating outside gta would bank pixels nobody ever drains.

        The lock is kept even though these fire on the sim thread (the viewer is
        pumped synchronously): ``_take_mouse_look`` is shared with the Genesis
        path, where the lock is load-bearing, and an uncontended acquire is free.
        """
        viewer = self.viewer

        def on_drag(dx, dy) -> None:
            if self.camera_mode != "gta":
                return
            try:
                with self._look_lock:
                    self._mouse_look[0] += float(dx)
                    self._mouse_look[1] += float(dy)
            except Exception:  # pragma: no cover - must never break the viewer
                pass

        def on_scroll(notches) -> None:
            if self.camera_mode != "gta":
                return
            try:
                with self._look_lock:
                    self._mouse_scroll += float(notches)
            except Exception:  # pragma: no cover
                pass

        try:
            viewer.set_drag_callback(on_drag)
            viewer.set_scroll_callback(on_scroll)
        except Exception as exc:
            print(f"[arena] mouse-look unavailable ({exc}); pad/keys still work")
            return
        # ``_look_window`` stays None: it exists only so unregister() can delete
        # the shadowing pyglet instance attributes, and there are none here.

    def unregister(self) -> None:
        """Best-effort removal of every keybind this object registered."""
        self._remove_hud()
        if self._viewer_backend == "mujoco":
            # Nothing to unwind.  The MujocoViewer contract has setters but no
            # removers, and handing back None could crash a viewer that invokes
            # its stored callback unconditionally.  Leaving them attached is
            # harmless: after _remove_hud the callbacks only latch into
            # KeyboardSource and the look accumulators, which nothing drains any
            # more, and the viewer is closed with the session anyway.
            self._registered = []
            return
        window = getattr(self, "_focus_guard_window", None)
        if window is not None:
            try:
                del window.on_deactivate
            except Exception:
                pass
            self._focus_guard_window = None
        window = getattr(self, "_look_window", None)
        if window is not None:
            # Deleting the instance attributes un-shadows pyrender's own class
            # methods, so the trackball is exactly as it was before install.
            for name in ("on_mouse_drag", "on_mouse_scroll"):
                try:
                    delattr(window, name)
                except Exception:
                    pass
            self._look_window = None
        viewer = self.viewer
        if viewer is None:
            return
        for name in self._registered:
            try:
                viewer.remove_keybind(name)
            except Exception:
                pass
        self._registered = []

    # -- viewer-thread callbacks --------------------------------------------
    def _set_shift(self, side: str, down: bool) -> None:
        """Track either Shift key and expose its aggregate state to driving."""
        if down:
            self._shift_keys.add(side)
        else:
            self._shift_keys.discard(side)
        self._shift_held = bool(self._shift_keys)
        self.source.keyboard.set_turbo(self._shift_held)

    def _on_action_key(self, event: str) -> None:
        if event == "tab_next" and self._shift_held:
            event = "tab_prev"
        elif event == "model_next" and self._shift_held:
            event = "model_prev"
        self.source.push_event(event)

    # -- sim-thread handling -------------------------------------------------
    def drain_and_apply(self) -> List[str]:
        """Execute every queued arena event.  Sim thread only."""
        events = self.source.drain_events()
        for event in events:
            try:
                self.apply(event)
            except Exception as exc:
                print(f"[arena] event {event!r} failed: {exc}")
        return events

    def apply(self, event: str) -> None:
        if event == "tab_next":
            self._move_tab(+1)
        elif event == "tab_prev":
            self._move_tab(-1)
        elif event == "level_next":
            self._move_level(+1)
        elif event == "level_prev":
            self._move_level(-1)
        elif event == "stop":
            self.source.clear()
            self._zero_command()
        elif event == "respawn":
            self._goto(self.tab_index, self.level_index, announce="respawn")
        elif event == "camera_next":
            self.next_camera_mode()
        elif event == "camera_snap":
            self.snap_camera_behind()
        elif event == "legend":
            self.print_legend()
        elif event in ("model_next", "model_prev"):
            if self.model_swap_cb is not None:
                self.model_swap_cb(+1 if event == "model_next" else -1)
            else:
                print("[arena] model hot-swap unavailable for this run")
        elif event == "gamepad_toggle":
            # toggle_gamepad() -> bool lives on MergedSource (input_source.py).
            # A missing method raises AttributeError, which drain_and_apply
            # catches and reports -- the arena keeps working on the keyboard.
            state = self.source.toggle_gamepad()
            if state:
                gamepad = getattr(self.source, "gamepad", None)
                name = getattr(gamepad, "name", None) \
                    if gamepad is not None and getattr(gamepad, "available", False) \
                    else None
                print(f"[arena] gamepad ON ({name})" if name else
                      "[arena] gamepad ON (no pad detected yet; rescanning)")
            else:
                print("[arena] gamepad OFF (keyboard only)")
            # The device label changes with the toggle; show it immediately
            # instead of up to one HUD refresh period later.
            self.update_hud(force=True)
        else:
            print(f"[arena] unknown event {event!r}")

    def _move_tab(self, delta: int) -> None:
        with self._lock:
            self.tab_index = (self.tab_index + int(delta)) % len(self.tabs)
        self._goto(self.tab_index, self.level_index, announce="tab")

    def _move_level(self, delta: int) -> None:
        with self._lock:
            self.level_index = min(
                max(self.level_index + int(delta), 0), len(self.levels) - 1
            )
        self._goto(self.tab_index, self.level_index, announce="level")

    @property
    def current(self) -> Tuple[str, int]:
        return (self.tabs[self.tab_index], self._display_level(
            self.tab_index, self.level_index))

    def _display_level(self, tab_index: int, level_index: int) -> int:
        """Training-level label for a row in the selected showcase column."""
        row = min(max(int(level_index), 0), len(self.levels) - 1)
        terrain_cfg = getattr(getattr(self.env, "cfg", None), "terrain", None)
        if getattr(terrain_cfg, "moe_showcase", False):
            tab = min(max(int(tab_index), 0), len(self.tabs) - 1)
            return moe_showcase_levels_for_column(
                self.tabs[tab], self.levels
            )[row]
        return self.levels[row]

    # -- world actions -------------------------------------------------------
    def _zero_command(self) -> None:
        commands = getattr(self.env, "commands", None)
        if commands is None:
            return
        try:
            commands[:, :3] = 0.0
        except Exception:
            pass

    def _goto(self, tab_index: int, level_index: int, announce: str = "") -> bool:
        """Teleport onto (level row, type column), reframe, zero the command."""
        ok = teleport_env_to_taxonomy_tile(
            self.env, self.robot_index, int(level_index), int(tab_index)
        )
        # Zero after the reset so a key held through the teleport does not
        # immediately drive the robot off the fresh tile.
        self.source.clear()
        self._zero_command()
        if ok:
            # A teleport changes the intended camera reference discontinuously.
            # Reacquire the stable rig on the new tile instead of easing across
            # the whole map from the prior one.
            self._camera_anchor = None
            self._camera_yaw = None
            self._camera_height = None
            # Landing on a new tile is the one moment where recentring the free
            # gta azimuth is what the user wants: they asked to be put somewhere
            # else and the first thing they need is to see where the robot is
            # facing.  Ordinary driving never touches it.
            self._camera_azimuth = None
            if self.following:
                self.update_follow_camera(force=True)
            self.frame_current_tile()
        # Reflect the switch on screen immediately instead of up to one HUD
        # refresh period later.
        self.update_hud(force=True)
        name = self.tabs[tab_index]
        level = self._display_level(tab_index, level_index)
        if not self.quiet:
            tag = f"[arena/{announce}]" if announce else "[arena]"
            print(f"{tag} tab {tab_index} '{name}'  ·  level L{level} "
                  f"(row {level_index})  ok={ok}")
        return ok

    def tile_origin(self, level_index: int, tab_index: int) -> Optional[np.ndarray]:
        simulator = getattr(self.env, "simulator", None)
        origins = getattr(simulator, "_terrain_origins", None)
        if origins is None:
            return None
        try:
            rows, cols = int(origins.shape[0]), int(origins.shape[1])
            row = min(max(int(level_index), 0), rows - 1)
            col = min(max(int(tab_index), 0), cols - 1)
            origin = origins[row, col]
            if hasattr(origin, "detach"):
                origin = origin.detach().cpu().numpy()
            return np.asarray(origin, dtype=np.float32)
        except Exception:
            return None

    def frame_current_tile(self) -> bool:
        """Point the fixed camera at the active tile (no-op while following)."""
        if self.following:
            return False
        origin = self.tile_origin(self.level_index, self.tab_index)
        if origin is None:
            return False
        terrain_cfg = getattr(getattr(self.env, "cfg", None), "terrain", None)
        length = float(getattr(terrain_cfg, "terrain_length", 8.0))
        width = float(getattr(terrain_cfg, "terrain_width", 8.0))
        extent = max(length, width)
        eye = origin + np.array(
            [-0.85 * length, -0.85 * width, 0.75 * extent], dtype=np.float32
        )
        try:
            self.env.set_viewer_camera(eye, origin)
        except Exception:
            return False
        return True

    @property
    def following(self) -> bool:
        """Whether this arena currently writes a robot-relative camera pose."""
        return self.camera_mode in FOLLOW_CAMERA_MODES

    @property
    def camera_override_active(self) -> bool:
        """Whether the arena, rather than --follow_robot, owns the camera.

        True from construction now that ``gta`` is the startup mode: it writes
        a pose every control frame, and play.py's ``--follow_robot`` branch
        must stand down or the two writers fight over the same camera.  (It
        used to flip only on a T press, back when the default was ``free``.)
        """
        return self._camera_override_active

    def snap_camera_behind(self) -> bool:
        """Recentre the free gta azimuth behind the robot (R3 / G).

        This is the *only* automatic recentring in the mode, and it is opt-in
        by design: a camera that eases back behind the robot on its own can
        never be parked in front of it.

        Only a flag is set here; the actual angle is taken from the base yaw on
        the next ``update_follow_camera``, which is the one place that is
        allowed to read simulator state (sim thread).  Harmless no-op in
        rear/front (already body-relative) and in free (Genesis owns the view).
        """
        if self.camera_mode != "gta":
            return False
        self._camera_snap_pending = True
        return True

    def next_camera_mode(self) -> str:
        """Cycle gta orbit -> rear third-person -> front -> free camera mode."""
        viewer = self.viewer
        if viewer is None:
            return self.camera_mode
        modes = CAMERA_MODES
        previous_mode = self.camera_mode
        self.camera_mode = modes[(modes.index(self.camera_mode) + 1) % len(modes)]
        # A user who has pressed T explicitly selected a camera mode.  In the
        # subsequent free mode, do not let a launch-time --follow_robot flag
        # resume writing poses behind their mouse movements.
        self._camera_override_active = True
        # Never use Genesis Viewer.follow_entity(): it updates on every physics
        # substep and fights the native trackball.  The play loop writes at most
        # one camera pose per full control frame in rear/front modes.
        try:
            viewer._followed_entity = None
        except Exception:
            pass
        if previous_mode == "free" and self.following:
            # Free mode may have lasted arbitrarily long; reacquire without a
            # long camera fly-in when switching back to a robot-relative mode.
            self._camera_anchor = None
            self._camera_yaw = None
            self._camera_height = None
            # The trackball has been moved by hand in the meantime, so the old
            # orbit angle means nothing any more: come back in behind the robot.
            self._camera_azimuth = None
        if self.following:
            # A camera-mode switch must take effect immediately.  Normal
            # tracking can then use the configured, lower-rate update path.
            self.update_follow_camera(force=True)
        self.update_hud(force=True)
        print(f"[arena] camera mode: {self.camera_mode}")
        return self.camera_mode

    def _look_dt(self, dt: Optional[float]) -> float:
        """Seconds since the last accepted camera update, for rate integration.

        ``update_follow_camera`` is called from ``interaction_loop`` without a
        dt (and is stride-skipped), so the rig measures its own instead of
        guessing a frame time.  Clamped to ``GTA_MAX_LOOK_DT`` so a stall does
        not turn one frame's stick deflection into a full camera whip.  Tests
        pass ``dt`` explicitly to stay deterministic.
        """
        now = time.monotonic()
        if dt is None:
            last = self._look_clock
            dt = 0.0 if last is None else (now - last)
        self._look_clock = now
        return min(max(float(dt), 0.0), GTA_MAX_LOOK_DT)

    def _take_mouse_look(self) -> Tuple[float, float, float]:
        """Drain the viewer thread's accumulated drag/scroll.  Sim thread only."""
        with self._look_lock:
            dx, dy = self._mouse_look
            scroll = self._mouse_scroll
            self._mouse_look[0] = 0.0
            self._mouse_look[1] = 0.0
            self._mouse_scroll = 0.0
        return float(dx), float(dy), float(scroll)

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        """Fold an angle into (-pi, pi].

        The azimuth is integrated every frame and never reset by anything but
        an explicit snap, so orbiting one way for a few minutes would otherwise
        walk it into magnitudes where float32 trig starts losing precision.
        """
        return float(math.atan2(math.sin(angle), math.cos(angle)))

    def _advance_gta_rig(self, dt: float, measured_yaw: float) -> None:
        """Integrate one frame of look input into (azimuth, pitch, distance).

        Sim thread only.  Two input paths feed the same rig: the pad's right
        stick is a *rate* (rad/s, hence the dt) and the mouse is a *delta*
        (pixels already accumulated since the last call, hence no dt).

        Sign convention, consistent between the two so muscle memory transfers:
        stick right / drag right orbits the view to the right (the eye bearing
        decreases, because yaw is CCW-positive), and stick up / drag up lifts
        the eye (pitch increases, looking further down at the robot).
        """
        if self._camera_azimuth is None or self._camera_snap_pending:
            # ``azimuth`` is the bearing from the robot TO the eye, so directly
            # behind a robot facing ``measured_yaw`` is ``measured_yaw + pi``,
            # wrapped into (-pi, pi] like every other write to this field.
            self._camera_azimuth = self._wrap_pi(float(measured_yaw) + math.pi)
            self._camera_snap_pending = False
            # Whatever the user had queued up belongs to the old angle.
            self._take_mouse_look()
            return

        look_x, look_y = getattr(self.source, "look", (0.0, 0.0))
        mouse_dx, mouse_dy, scroll = self._take_mouse_look()

        azimuth = self._camera_azimuth
        azimuth -= float(look_x) * GTA_AZIMUTH_RATE * dt
        azimuth -= mouse_dx * GTA_MOUSE_AZIMUTH_PER_PX
        self._camera_azimuth = self._wrap_pi(azimuth)

        # ``look_y`` is negative when the stick is up (SDL convention, same
        # reason ``vx = -lefty``), so negate it to make "up" lift the camera.
        pitch = self._camera_pitch
        pitch += -float(look_y) * GTA_PITCH_RATE * dt
        pitch += mouse_dy * GTA_MOUSE_PITCH_PER_PX
        # Hard clamp, not a wrap: past ~+-90 deg the eye crosses the pole and
        # the up-vector flips, which reads as the world turning upside down.
        self._camera_pitch = min(max(pitch, GTA_PITCH_LIMITS[0]),
                                 GTA_PITCH_LIMITS[1])

        if scroll:
            # Multiplicative so a notch feels the same close in and far out.
            distance = self._camera_distance * (1.0 - GTA_SCROLL_STEP * scroll)
            self._camera_distance = min(max(distance, GTA_DISTANCE_LIMITS[0]),
                                        GTA_DISTANCE_LIMITS[1])

    def camera_detail(self) -> str:
        """Short ASCII HUD fragment describing the live rig (gta only)."""
        if self.camera_mode != "gta":
            return ""
        return f"d{self._camera_distance:.1f}"

    def update_follow_camera(self, force: bool = False,
                             dt: Optional[float] = None) -> bool:
        """Update the body-anchored camera (gta orbit, or rear/front).

        The camera follows a slow world-space XY anchor, keeps a fixed world
        height and only slowly accepts meaningful yaw changes.  In particular,
        it does not inherit the base's gait-scale vertical bob or tiny yaw
        oscillations from foot contacts.

        The gta mode reuses that exact anchor and adds a spherical orbit on top
        of it.  What it does NOT reuse is ``_camera_yaw``: the orbit azimuth is
        an absolute world angle owned by the user, so unlike rear/front it never
        rotates with the robot.  Consuming the look input here, on the sim
        thread, is deliberate -- the mouse callbacks may only stash deltas.
        """
        if not self.following:
            return False
        if not force and self._follow_camera_stride > 1:
            self._follow_camera_frame = (
                self._follow_camera_frame + 1
            ) % self._follow_camera_stride
            if self._follow_camera_frame != 0:
                return False
        simulator = getattr(self.env, "simulator", None)
        base_pos = getattr(simulator, "base_pos", None)
        if base_pos is None:
            return False
        try:
            target = base_pos[self.robot_index]
            if hasattr(target, "detach"):
                target = target.detach().cpu().numpy()
            target = np.asarray(target, dtype=np.float32).reshape(3)
            base_euler = getattr(simulator, "base_euler", None)
            if base_euler is None:
                measured_yaw = 0.0
            else:
                measured_yaw = float(base_euler[self.robot_index, 2].item())
            if self._camera_anchor is None:
                self._camera_anchor = target.copy()
                self._camera_height = float(target[2])
            else:
                xy_delta = target[:2] - self._camera_anchor[:2]
                xy_distance = float(np.linalg.norm(xy_delta))
                if xy_distance > 3.0:
                    # Teleport/respawn: reframe instead of slowly flying over
                    # the whole arena.  Normal locomotion never reaches this
                    # threshold in one control frame.
                    self._camera_anchor[:2] = target[:2]
                    self._camera_height = float(target[2])
                else:
                    # ~0.3 s XY follow time constant, plus a speed cap.  This
                    # looks like a camera rig following the robot rather than
                    # a camera bolted to its pelvis.
                    xy_step = 0.06 * xy_delta
                    step_norm = float(np.linalg.norm(xy_step))
                    if step_norm > 0.06:
                        xy_step *= 0.06 / step_norm
                    self._camera_anchor[:2] += xy_step
                # The visual camera is world-level, not base-level.  Keep its
                # anchor Z fixed until a real teleport/reframe occurs.
                self._camera_anchor[2] = float(self._camera_height)
            if self._camera_yaw is None:
                self._camera_yaw = measured_yaw
            else:
                yaw_delta = np.arctan2(
                    np.sin(measured_yaw - self._camera_yaw),
                    np.cos(measured_yaw - self._camera_yaw),
                )
                # Ignore sub-degree contact jitter.  Intentional turning still
                # rotates the rig smoothly over a few tenths of a second.
                if abs(yaw_delta) > 0.02:
                    self._camera_yaw += 0.05 * yaw_delta
            target = self._camera_anchor
            if self.camera_mode == "gta":
                # Orbit rig.  The anchor (and therefore all of the anti-bob
                # filtering above) is shared with rear/front; only the offset
                # differs, and it is driven by the user instead of the heading.
                self._advance_gta_rig(self._look_dt(dt), measured_yaw)
                azimuth = float(self._camera_azimuth)
                pitch = float(self._camera_pitch)
                radius = float(self._camera_distance)
                horizontal = radius * math.cos(pitch)
                eye_offset = np.array(
                    [
                        horizontal * math.cos(azimuth),
                        horizontal * math.sin(azimuth),
                        radius * math.sin(pitch),
                    ],
                    dtype=np.float32,
                )
                lookat_offset = np.array(
                    [0.0, 0.0, GTA_LOOKAT_HEIGHT], dtype=np.float32
                )
                # Already world-frame: no body_to_world rotation, which is the
                # whole point of an absolute azimuth.
                self.env.set_viewer_camera(target + eye_offset,
                                           target + lookat_offset)
                return True

            yaw = self._camera_yaw
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

            def body_to_world(offset: np.ndarray) -> np.ndarray:
                return np.array(
                    [
                        cos_yaw * offset[0] - sin_yaw * offset[1],
                        sin_yaw * offset[0] + cos_yaw * offset[1],
                        offset[2],
                    ],
                    dtype=np.float32,
                )

            if self.camera_mode == "rear":
                # Third-person: behind and slightly above the robot, looking
                # just ahead of its body so its travel direction is visible.
                eye_offset = np.array([-4.5, 0.0, 1.55], dtype=np.float32)
                lookat_offset = np.array([0.70, 0.0, 0.38], dtype=np.float32)
            elif self.camera_mode == "front":
                # The former T-like viewpoint, now stable and body-relative.
                eye_offset = np.array([3.0, 0.0, 1.15], dtype=np.float32)
                lookat_offset = np.array([0.0, 0.0, 0.28], dtype=np.float32)
            else:
                return False
            self.env.set_viewer_camera(
                target + body_to_world(eye_offset),
                target + body_to_world(lookat_offset),
            )
        except Exception:
            return False
        return True

    # -- reporting -----------------------------------------------------------
    def print_legend(self) -> None:
        gamepad = ""
        if self.source.gamepad is not None and self.source.gamepad.available:
            gamepad = self.source.gamepad.name
        print(format_arena_legend(
            self.tabs, self.levels, self.tab_index, self.level_index, gamepad,
            backend=self._viewer_backend or "genesis",
        ))
        terrain_cfg = getattr(getattr(self.env, "cfg", None), "terrain", None)
        if getattr(terrain_cfg, "moe_showcase", False):
            stairs = moe_showcase_levels_for_column("stairs_up", self.levels)
            print("  MoE row mapping: other terrains "
                  + "/".join(f"L{level}" for level in self.levels)
                  + " | stairs " + "/".join(f"L{level}" for level in stairs))
