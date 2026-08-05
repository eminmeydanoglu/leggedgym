"""Headless tests for legged_gym/utils/mujoco_viewer.py.

Everything here runs without a display: the graceful-degradation tests fake a
failing GLFW, and the keymap / camera-maths tests exercise module-level pure
functions that were factored out precisely so they need no window.
"""
import importlib.util
import math
import pathlib

import numpy as np
import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "legged_gym" / "utils" / "mujoco_viewer.py"
)


def _load_viewer_module():
    """Load mujoco_viewer without going through ``legged_gym/__init__.py``.

    That package __init__ raises unless ``SIMULATOR`` is set in the environment,
    and CI has no reason to pick a simulator just to test a window wrapper.
    Loading straight off the path is also the sharpest possible check of the
    module's own rule: it must not import anything from ``legged_gym`` at import
    time, so it has to come up standalone.
    """
    spec = importlib.util.spec_from_file_location("_mujoco_viewer_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mv = _load_viewer_module()


def test_module_imports_without_the_legged_gym_package():
    # Re-load from scratch: if a legged_gym import ever creeps into module
    # scope this raises (SIMULATOR is not required to run the test suite).
    assert _load_viewer_module() is not None


# ---------------------------------------------------------------------------
# Graceful degradation with no display
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fake_init",
    [
        pytest.param(lambda: False, id="init_returns_false"),
        pytest.param(
            lambda: (_ for _ in ()).throw(RuntimeError("no DISPLAY")), id="init_raises"
        ),
    ],
)
def test_glfw_init_failure_degrades_gracefully(monkeypatch, capsys, fake_init):
    monkeypatch.setattr(mv.glfw, "init", fake_init)

    viewer = mv.MujocoViewer(model=None, data=None)  # must not raise

    assert viewer.is_running is False
    assert viewer.sync() is None
    assert viewer.close() is None
    assert viewer.is_running is False

    # Exactly one explanatory line, and it must not look like a crash.
    out = capsys.readouterr().out
    assert out.count("[mujoco_viewer] disabled") == 1


def test_window_creation_failure_degrades_gracefully(monkeypatch, capsys):
    monkeypatch.setattr(mv.glfw, "init", lambda: True)
    monkeypatch.setattr(mv.glfw, "window_hint", lambda *a, **k: None)
    monkeypatch.setattr(mv.glfw, "create_window", lambda *a, **k: None)

    viewer = mv.MujocoViewer(model=None, data=None, width=100, height=100)

    assert viewer.is_running is False
    viewer.sync()
    viewer.close()
    assert "[mujoco_viewer] disabled" in capsys.readouterr().out


def test_dead_viewer_setters_are_inert(monkeypatch):
    monkeypatch.setattr(mv.glfw, "init", lambda: False)
    viewer = mv.MujocoViewer(model=None, data=None)

    # None of these may raise on a viewer that never got a window.
    viewer.set_camera(np.zeros(3), np.ones(3))
    viewer.set_key_callback(lambda name, pressed: None)
    viewer.set_scroll_callback(lambda notches: None)
    viewer.set_drag_callback(lambda dx, dy: None)
    for _ in range(3):
        viewer.sync()
    assert viewer.is_running is False


# ---------------------------------------------------------------------------
# Keycode -> name table
# ---------------------------------------------------------------------------

EXPECTED_KEY_NAMES = {
    "UP", "DOWN", "LEFT", "RIGHT", "Q", "E", "T", "M", "N", "J", "G",
    "TAB", "SPACE", "BACKSPACE", "BRACKETLEFT", "BRACKETRIGHT",
    "LSHIFT", "RSHIFT", "ESCAPE",
}


def test_keymap_covers_exactly_the_contract_names():
    assert set(mv.SUPPORTED_KEY_NAMES) == EXPECTED_KEY_NAMES
    assert len(mv.SUPPORTED_KEY_NAMES) == len(EXPECTED_KEY_NAMES)


def test_keymap_is_unambiguous():
    names = list(mv._KEY_NAME_BY_GLFW_CODE.values())
    codes = list(mv._KEY_NAME_BY_GLFW_CODE.keys())
    assert len(set(names)) == len(names), "two keycodes map to the same name"
    assert len(set(codes)) == len(codes), "duplicate keycode in the table"


@pytest.mark.parametrize(
    "glfw_attr,name",
    [
        ("KEY_UP", "UP"),
        ("KEY_DOWN", "DOWN"),
        ("KEY_LEFT", "LEFT"),
        ("KEY_RIGHT", "RIGHT"),
        ("KEY_Q", "Q"),
        ("KEY_E", "E"),
        ("KEY_T", "T"),
        ("KEY_M", "M"),
        ("KEY_N", "N"),
        ("KEY_J", "J"),
        ("KEY_G", "G"),
        ("KEY_TAB", "TAB"),
        ("KEY_SPACE", "SPACE"),
        ("KEY_BACKSPACE", "BACKSPACE"),
        # pyglet spelling on purpose: the arena keymap already uses these.
        ("KEY_LEFT_BRACKET", "BRACKETLEFT"),
        ("KEY_RIGHT_BRACKET", "BRACKETRIGHT"),
        ("KEY_LEFT_SHIFT", "LSHIFT"),
        ("KEY_RIGHT_SHIFT", "RSHIFT"),
        ("KEY_ESCAPE", "ESCAPE"),
    ],
)
def test_glfw_key_name_translates_each_contract_key(glfw_attr, name):
    assert mv.glfw_key_name(getattr(mv.glfw, glfw_attr)) == name


def test_glfw_key_name_returns_none_for_unmapped_keys():
    for attr in ("KEY_W", "KEY_A", "KEY_S", "KEY_D", "KEY_F11", "KEY_ENTER"):
        assert mv.glfw_key_name(getattr(mv.glfw, attr)) is None
    assert mv.glfw_key_name(-1) is None


# ---------------------------------------------------------------------------
# eye/target -> mjvCamera
# ---------------------------------------------------------------------------

def test_camera_eye_due_south_and_level():
    # MuJoCo free camera: forward = (cos az cos el, sin az cos el, sin el).
    # Looking due north (+y) from 5 m away, level, is azimuth 90 / elevation 0.
    az, el, dist, lookat = mv.camera_from_eye_target(np.array([0.0, -5.0, 0.0]), np.zeros(3))
    assert az == pytest.approx(90.0)
    assert el == pytest.approx(0.0)
    assert dist == pytest.approx(5.0)
    assert lookat == pytest.approx(np.zeros(3))


def test_camera_eye_due_east_and_level():
    # Eye at +x looking back along -x -> azimuth 180.
    az, el, dist, _ = mv.camera_from_eye_target([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert abs(az) == pytest.approx(180.0)
    assert el == pytest.approx(0.0)
    assert dist == pytest.approx(2.0)


def test_camera_eye_directly_above_uses_azimuth_fallback():
    az, el, dist, lookat = mv.camera_from_eye_target(
        [1.0, 2.0, 6.0], [1.0, 2.0, 3.0], azimuth_fallback=37.5
    )
    assert el == pytest.approx(-90.0)
    assert dist == pytest.approx(3.0)
    assert az == pytest.approx(37.5)  # undefined at the pole -> hold previous
    assert lookat == pytest.approx(np.array([1.0, 2.0, 3.0]))


def test_camera_forty_five_degree_case_matches_mujoco_default_view():
    # Eye 1 m south and 1 m up: this is exactly MuJoCo's default free camera
    # orientation (azimuth 90, elevation -45).
    az, el, dist, _ = mv.camera_from_eye_target([0.0, -1.0, 1.0], [0.0, 0.0, 0.0])
    assert az == pytest.approx(90.0)
    assert el == pytest.approx(-45.0)
    assert dist == pytest.approx(math.sqrt(2.0))


def test_camera_offset_lookat_and_diagonal_azimuth():
    target = np.array([10.0, -4.0, 0.5])
    eye = target + np.array([-1.0, -1.0, 0.0])  # south-west of the target
    az, el, dist, lookat = mv.camera_from_eye_target(eye, target)
    assert az == pytest.approx(45.0)
    assert el == pytest.approx(0.0)
    assert dist == pytest.approx(math.sqrt(2.0))
    assert lookat == pytest.approx(target)


@pytest.mark.parametrize(
    "eye,target",
    [
        ([3.0, 1.0, 2.0], [0.0, 0.0, 0.0]),
        ([-2.5, 4.0, -1.0], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, -4.0], [0.0, 0.0, 0.0]),
        ([7.0, -3.0, 9.0], [-2.0, 8.0, 0.25]),
    ],
)
def test_camera_conversion_round_trips_through_mujocos_own_formula(eye, target):
    az, el, dist, lookat = mv.camera_from_eye_target(eye, target)
    a, e = math.radians(az), math.radians(el)
    forward = np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
    # mjv_updateCamera rebuilds the eye as lookat - distance * forward.
    assert lookat - dist * forward == pytest.approx(np.asarray(eye, dtype=float), abs=1e-9)


def test_camera_degenerate_eye_equals_target():
    az, el, dist, lookat = mv.camera_from_eye_target(
        [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], azimuth_fallback=12.0, elevation_fallback=-34.0
    )
    assert dist == 0.0
    assert az == pytest.approx(12.0)
    assert el == pytest.approx(-34.0)
    assert lookat == pytest.approx(np.ones(3))


def test_camera_lookat_is_a_copy_not_an_alias():
    target = np.array([1.0, 2.0, 3.0])
    _, _, _, lookat = mv.camera_from_eye_target(np.zeros(3), target)
    lookat[0] = 99.0
    assert target[0] == 1.0


# ---------------------------------------------------------------------------
# HUD line storage
# ---------------------------------------------------------------------------

def test_sanitize_hud_lines_always_returns_four_strings():
    for value in (None, [], ["a"], ["a", "b"], ["a", "b", "c", "d"], ["a", "b", "c", "d", "e"]):
        out = mv._sanitize_hud_lines(value)
        assert len(out) == mv._HUD_SLOTS
        assert all(isinstance(s, str) for s in out)


def test_sanitize_hud_lines_preserves_order_and_pads():
    assert mv._sanitize_hud_lines(["top", "bl_up"]) == ["top", "bl_up", "", ""]
    assert mv._sanitize_hud_lines(["a", "b", "c", "d", "e", "f"]) == ["a", "b", "c", "d"]


def test_sanitize_hud_lines_coerces_non_strings():
    assert mv._sanitize_hud_lines([1, None, 2.5, True]) == ["1", "", "2.5", "True"]


def test_sanitize_hud_lines_treats_bare_string_as_one_line():
    # A bare string would otherwise be iterated character-by-character.
    assert mv._sanitize_hud_lines("hello") == ["hello", "", "", ""]


def test_drag_callback_flips_dy_to_pyglet_convention(monkeypatch):
    """GLFW measures +y downward; every caller of this expects pyglet's +y up.

    ``native_ui`` accumulates the forwarded delta into ONE mouse-look integrator
    that the Genesis (pyglet) path also feeds, so a raw pass-through would pitch
    the GTA camera the opposite way on MuJoCo.  dx is unaffected -- both APIs
    agree on +x right.
    """
    monkeypatch.setattr(mv.glfw, "init", lambda: False)
    viewer = mv.MujocoViewer(model=None, data=None)

    seen = []
    viewer.set_drag_callback(lambda dx, dy: seen.append((dx, dy)))
    viewer._buttons["left"] = True
    viewer._last_cursor = (100.0, 100.0)

    # Cursor moves right and *down* the screen in GLFW terms.
    viewer._on_cursor_pos(None, 110.0, 130.0)
    assert seen == [(10.0, -30.0)]

    # ... and back up: the sign flips with it, no accumulated bias.
    viewer._on_cursor_pos(None, 110.0, 100.0)
    assert seen[-1] == (0.0, 30.0)


def test_set_hud_lines_tracks_the_caller_list_live(monkeypatch):
    """The caller owns the list and mutates it in place; the HUD must follow.

    ``native_ui._install_hud`` hands over one list at startup and then only ever
    swaps strings into it, so a detached copy here would freeze the on-screen
    HUD on its startup text.  Coercion to four strings therefore happens at draw
    time, not on assignment -- which this test pins from both ends.
    """
    monkeypatch.setattr(mv.glfw, "init", lambda: False)
    viewer = mv.MujocoViewer(model=None, data=None)
    assert mv._sanitize_hud_lines(viewer._hud_lines) == ["", "", "", ""]

    caller_list = ["top", "bl_up", "bl_lo", "tr"]
    viewer.set_hud_lines(caller_list)
    assert viewer._hud_lines is caller_list, "HUD must render from the caller's list"

    caller_list[0] = "mutated"
    assert mv._sanitize_hud_lines(viewer._hud_lines)[0] == "mutated"

    # Draw-time coercion still pads/truncates/stringifies whatever it finds.
    viewer.set_hud_lines(["only-top"])
    assert mv._sanitize_hud_lines(viewer._hud_lines) == ["only-top", "", "", ""]
