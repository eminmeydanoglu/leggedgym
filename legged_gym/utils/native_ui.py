"""Native (Genesis viewer) play-arena UI: terrain "tabs" + real keyboard drive.

The showcase grid is already laid out as ``rows = difficulty level`` x
``columns = terrain type`` (``terrain.py:add_moe_terrain_to_map``), so a "stairs
tab" is nothing more than the ``stairs_up`` column, and switching tabs is a
teleport plus a camera reframe.  ``scene.build()`` is one-shot in Genesis --
entities and terrain cannot change afterwards -- so a tab is deliberately never
a rebuild.

Threading contract
------------------
Genesis runs the pyglet viewer in a background thread on Linux.  Keybind
callbacks therefore fire on the *viewer* thread while the sim steps on the main
thread.  Callbacks here only ever mutate thread-safe state:

* drive keys go straight into ``KeyboardSource`` (lock-protected), and
* arena actions (tab / level / respawn / ...) are *queued* as events.

``interaction_loop`` drains the queue with :meth:`NativeArenaUI.drain_and_apply`
on the sim thread, which is where ``env.reset_idx`` / ``teleport_...`` are legal
to call.

The on-screen HUD follows the same contract from the other direction: it is a
pyrender *plugin* (``_HudPlugin``) whose ``on_draw`` runs on the viewer thread,
after Genesis's own caption pass, and only ever *reads* the pre-formatted
strings in ``NativeArenaUI._hud_lines`` -- those are rebuilt on the sim thread
under an 8 Hz throttle by :meth:`NativeArenaUI.update_hud`.  The plugin never
touches env state and never lets an exception escape into the draw loop.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from legged_gym.scripts.input_source import DriveEnvelope, MergedSource
from legged_gym.utils.terrain import (
    MOE_SHOWCASE_LEVELS,
    moe_showcase_columns,
    teleport_env_to_taxonomy_tile,
)


# ---------------------------------------------------------------------------
# Keymap
# ---------------------------------------------------------------------------
#
# Genesis's ``DefaultControlsPlugin`` already owns R, S, Z, A, H, F, V, W, L, D,
# O, C, P and F11 (all on KeyAction.RELEASE), and the pyrender viewer reserves I
# for the help overlay as a *protected* keybind.  That rules out both WASD and
# the Viser TFGH/RY layout without overwriting the viewer's own debug toggles,
# which this module deliberately does not do -- nothing here is registered with
# ``overwrite=True``.  Arrows + Q/E are collision-free and need no explanation.
#
# name -> (Key attribute name, drive action)
DRIVE_KEYMAP: Dict[str, Tuple[str, str]] = {
    "arena_forward": ("UP", "forward"),
    "arena_backward": ("DOWN", "backward"),
    "arena_yaw_left": ("LEFT", "yaw_left"),
    "arena_yaw_right": ("RIGHT", "yaw_right"),
    "arena_strafe_left": ("Q", "strafe_left"),
    "arena_strafe_right": ("E", "strafe_right"),
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
    # J is free: the Genesis-owned set above stops at F11/I and no default
    # plugin binds it.
    "arena_gamepad_toggle": ("J", "gamepad_toggle"),
}

ARENA_EVENTS = (
    "tab_next",
    "tab_prev",
    "level_next",
    "level_prev",
    "stop",
    "respawn",
    "camera_next",
    "legend",
    "model_next",
    "model_prev",
    "gamepad_toggle",
)


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
) -> List[str]:
    """The four single-line HUD strings: status, commanded, achieved, hint.

    Line 3 is deliberately just ``M: keys``.  The full key list already lives
    in the terminal legend (M) and in Genesis's I overlay, and the long hint
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
    status = (f"tab {ti + 1}/{n_tabs} {name}   "
              f"level L{level} (row {li + 1}/{n_lvls})   "
              f"cam {camera_mode}")

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

    hint = "M: keys"

    return [_ascii_only(status), _ascii_only(commanded),
            _ascii_only(achieved_line), _ascii_only(hint)]


def format_arena_legend(
    tabs: Sequence[str],
    levels: Sequence[int],
    tab_index: int = 0,
    level_index: int = 0,
    gamepad: str = "",
) -> str:
    """Console legend: tabs, levels and every keybinding."""
    head = "Play arena — native Genesis viewer"
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
    lines.append("  drive:")
    lines.append("    Up / Down     forward / backward")
    lines.append("    Left / Right  turn left / right")
    lines.append("    Q / E         strafe left / right")
    lines.append("    Space         hard stop")
    lines.append("  arena:")
    lines.append("    Tab / Shift+Tab   next / previous terrain tab")
    lines.append("    [ / ]             easier / harder level in this tab")
    lines.append("    Backspace         respawn on the current tile")
    lines.append("    T                 camera: rear -> front -> free")
    lines.append("    N                 hot-swap to the next checkpoint of this run")
    lines.append("    J                 toggle gamepad on/off (auto-rescans when on)")
    lines.append("    M                 reprint this legend")
    lines.append("  (Genesis reserves I for its own help overlay; R/S/Z/A/H/F/V/W/L/D/O/C/P "
                 "stay with the viewer's debug toggles.)")
    lines.append("  gamepad: auto-detected at startup, no launch flag needed; a")
    lines.append("    hot-plugged pad is picked up within ~2 s; J toggles it on/off")
    lines.append(f"    connected: {gamepad if gamepad else 'none'}")
    lines.append("    left stick  vx / vy    right stick X  yaw")
    lines.append("    LB / RB  tab      X / B  level      A  respawn")
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
        # T cycles these modes.  ``free`` deliberately writes no camera pose:
        # Genesis's native trackball then owns the view completely.
        self.camera_mode = "free"
        self._camera_override_active = False
        # Smoothed world anchor: fast enough in XY to follow locomotion, but
        # intentionally slow in Z so the camera does not inherit body bounce
        # from individual foot impacts.  Yaw gets its own wrapped filter.
        self._camera_anchor: Optional[np.ndarray] = None
        self._camera_yaw: Optional[float] = None
        self._shift_held = False
        self._lock = threading.Lock()
        self._registered: List[str] = []
        self._focus_guard_window = None
        # HUD state (see _install_hud / update_hud)
        self._hud_window = None
        self._hud_plugin: Optional[_HudPlugin] = None
        self._hud_lines: Optional[List[str]] = None
        self._hud_last = 0.0

        self.viewer = self._find_viewer()
        if self.viewer is not None:
            self._register_keybinds()

    # -- wiring --------------------------------------------------------------
    def _find_viewer(self):
        simulator = getattr(self.env, "simulator", None)
        scene = getattr(simulator, "_scene", None)
        return getattr(scene, "viewer", None)

    def _register_keybinds(self) -> None:
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
                    args=(True,),
                )
            )
            keybinds.append(
                Keybind(
                    f"arena_{side.lower()}_release",
                    getattr(Key, side),
                    key_action=KeyAction.RELEASE,
                    callback=self._set_shift,
                    args=(False,),
                )
            )

        try:
            self.viewer.register_keybinds(*keybinds)
        except Exception as exc:
            print(f"[arena] could not register keybinds ({exc}); keyboard drive is off")
            self.viewer = None
            return
        self._registered = [kb.name for kb in keybinds]
        self._install_focus_guard()
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
        """
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
            new_lines = format_hud_lines(
                self.tabs, self.levels, self.tab_index, self.level_index,
                command=command, achieved=achieved, fps=fps, realtime=realtime,
                device=self._device_label(),
                following=self.following,
                camera_mode=self.camera_mode,
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
        """
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
            except Exception:
                pass
            return original()

        try:
            window.on_deactivate = on_deactivate
        except Exception as exc:  # pragma: no cover - depends on Genesis build
            print(f"[arena] focus guard unavailable ({exc})")
            return
        self._focus_guard_window = window

    def unregister(self) -> None:
        """Best-effort removal of every keybind this object registered."""
        self._remove_hud()
        window = getattr(self, "_focus_guard_window", None)
        if window is not None:
            try:
                del window.on_deactivate
            except Exception:
                pass
            self._focus_guard_window = None
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
    def _set_shift(self, down: bool) -> None:
        self._shift_held = bool(down)

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
        return (self.tabs[self.tab_index], self.levels[self.level_index])

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
            self.frame_current_tile()
        # Reflect the switch on screen immediately instead of up to one HUD
        # refresh period later.
        self.update_hud(force=True)
        name, level = self.tabs[tab_index], self.levels[level_index]
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
        return self.camera_mode in {"rear", "front"}

    @property
    def camera_override_active(self) -> bool:
        """Whether a T press has taken camera ownership from --follow_robot."""
        return self._camera_override_active

    def next_camera_mode(self) -> str:
        """Cycle rear third-person -> front -> free camera mode."""
        viewer = self.viewer
        if viewer is None:
            return self.camera_mode
        modes = ("rear", "front", "free")
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
        if self.following:
            self.update_follow_camera()
        self.update_hud(force=True)
        print(f"[arena] camera mode: {self.camera_mode}")
        return self.camera_mode

    def update_follow_camera(self) -> bool:
        """Update a body-relative rear/front camera once per control frame."""
        if not self.following:
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
            else:
                # Suppress gait-scale vertical shake much more aggressively
                # than horizontal position, where too much smoothing reads as
                # a laggy camera at locomotion speed.
                alpha = np.array([0.16, 0.16, 0.035], dtype=np.float32)
                self._camera_anchor += alpha * (target - self._camera_anchor)
            if self._camera_yaw is None:
                self._camera_yaw = measured_yaw
            else:
                yaw_delta = np.arctan2(
                    np.sin(measured_yaw - self._camera_yaw),
                    np.cos(measured_yaw - self._camera_yaw),
                )
                self._camera_yaw += 0.14 * yaw_delta
            target = self._camera_anchor
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
            self.tabs, self.levels, self.tab_index, self.level_index, gamepad
        ))
