"""Tests for `legged_gym.simulator.mujoco_scene`.

The load-bearing test here is `test_hfield_orientation_matches_genesis_contract`.
Everything else in the MuJoCo backend can be eyeballed in a viewer; the row/col ->
x/y mapping of a heightfield cannot, because a transposed or flipped terrain still
*looks* like plausible terrain. It only shows up as a policy that mysteriously
underperforms, because the height scan it observes describes different ground than
the one it is standing on. So we prove the mapping by ray-casting against the
compiled model and comparing to `height_field_raw` directly.

The height field is built by hand rather than via `legged_gym.utils.terrain.Terrain`:
the real class is slow and drags in the whole env stack (and Genesis), and all this
module needs from it is three attributes.
"""

import importlib.util
import os
import types

import numpy as np
import pytest

# `legged_gym/__init__.py` refuses to import without this, and these tests are
# specifically about the MuJoCo backend. setdefault so an explicit
# `SIMULATOR=mujoco pytest ...` still wins and nothing else is clobbered.
os.environ.setdefault("SIMULATOR", "mujoco")

import mujoco  # noqa: E402

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402

# Loaded straight off disk instead of `from legged_gym.simulator.mujoco_scene import ...`.
# `legged_gym/simulator/__init__.py` imports the Genesis/IsaacGym backends, which import
# the whole env stack, which imports back into `legged_gym.simulator` -- a pre-existing
# circular import that only resolves when `legged_gym.envs` is imported first. Going
# through it here would drag Genesis and the go2 configs into a test that needs neither
# and would turn a fast unit test into a multi-second env import. `mujoco_scene` itself
# only depends on `mujoco`, `numpy` and the top-level `LEGGED_GYM_ROOT_DIR`, so loading
# it in isolation exercises exactly the code under test.
_MODULE_PATH = os.path.join(
    LEGGED_GYM_ROOT_DIR, "legged_gym", "simulator", "mujoco_scene.py"
)
_spec = importlib.util.spec_from_file_location("_mujoco_scene_under_test", _MODULE_PATH)
mujoco_scene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mujoco_scene)

build_scene = mujoco_scene.build_scene
hfield_world_origin = mujoco_scene.hfield_world_origin

GO2_XML = "{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/go2/go2.xml"

HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE = 0.005
BORDER_SIZE = 2.0
# Deliberately unequal and coprime-ish, so a transpose cannot survive by luck (it
# would not even produce a same-shaped array) and no accidental symmetry hides a flip.
TOT_ROWS = 61
TOT_COLS = 41

# Ray origin height: comfortably above the tallest sample in `make_height_field`.
RAY_Z = 10.0


def make_cfg(mesh_type="heightfield", stair_spec=None):
    """Minimal stand-in for the legged_gym env config.

    A throwaway namespace rather than the real go2 config: importing those pulls in
    the env stack, and `build_scene` only reads the handful of fields below.
    """
    return types.SimpleNamespace(
        asset=types.SimpleNamespace(xml_file=GO2_XML),
        terrain=types.SimpleNamespace(
            mesh_type=mesh_type,
            horizontal_scale=HORIZONTAL_SCALE,
            vertical_scale=VERTICAL_SCALE,
            border_size=BORDER_SIZE,
            static_friction=0.8,
            plane_length=20.0,
            play_stair_collision_spec=stair_spec,
        ),
        sim=types.SimpleNamespace(dt=0.005, gravity=[0.0, 0.0, -9.81]),
    )


def make_height_field():
    """An aggressively asymmetric field: ramp in x, cliff in y, trench at the x border.

    Every feature is chosen to make a wrong mapping loud:
      * the x ramp means an x/y transpose lands on a totally different profile,
      * the y cliff means a y flip moves a 0.75 m wall to the wrong side,
      * the trench near i = 0 breaks the ramp's symmetry so an x flip is detectable.
    """
    i = np.arange(TOT_ROWS)[:, None]
    j = np.arange(TOT_COLS)[None, :]

    field = np.zeros((TOT_ROWS, TOT_COLS), dtype=np.int64)
    field += i * 8                      # ramp along +x: 0.04 m per cell
    field += np.where(j >= 25, 150, 0)  # cliff along +y at j = 25: 0.75 m
    field -= np.where(i < 6, 100, 0)    # trench hugging the x = -border edge
    return field.astype(np.int16)


def make_terrain():
    return types.SimpleNamespace(
        height_field_raw=make_height_field(),
        tot_rows=TOT_ROWS,
        tot_cols=TOT_COLS,
    )


def grid_to_world(i, j):
    """The Genesis contract, spelled out independently of the module under test."""
    return (
        i * HORIZONTAL_SCALE - BORDER_SIZE,
        j * HORIZONTAL_SCALE - BORDER_SIZE,
    )


def surface_height(model, data, x, y):
    """Cast a downward ray and return the ground surface z at (x, y), or None."""
    pnt = np.array([x, y, RAY_Z], dtype=np.float64)
    vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    # Mask to the terrain's render group only. The robot's own geoms live in groups 2
    # (visual) and 3 (collision), and the play-stair colliders in group 3, so without
    # this mask a ray dropped near the origin would hit the Go2 standing there.
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[0] = 1
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, pnt, vec, geomgroup, 1, -1, geomid)
    if geomid[0] < 0:
        return None
    return RAY_Z - dist


@pytest.fixture(scope="module")
def hfield_scene():
    cfg = make_cfg("heightfield")
    terrain = make_terrain()
    model = build_scene(cfg, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, terrain


# --------------------------------------------------------------------------------
# 1. Orientation proof
# --------------------------------------------------------------------------------

def test_hfield_orientation_matches_genesis_contract(hfield_scene):
    """world_z at (i*hs - border, j*hs - border) must equal hfr[i, j] * vertical_scale.

    Probed at exact grid vertices: a heightfield is a triangulated surface that passes
    exactly through its samples, so at a vertex the expected value is exact and the
    tolerance below is only absorbing float32 storage of the normalized elevations.
    """
    model, data, terrain = hfield_scene
    hfr = terrain.height_field_raw

    # Corners, both borders, the cliff on either side, and interior scatter.
    probes = [
        (0, 0), (0, TOT_COLS - 1), (TOT_ROWS - 1, 0), (TOT_ROWS - 1, TOT_COLS - 1),
        (1, 1), (3, 30), (5, 24), (5, 26), (6, 24), (6, 26),
        (12, 7), (20, 24), (20, 25), (23, 40), (30, 12), (37, 33),
        (44, 3), (50, 25), (55, 18), (60, 20), (0, 25), (60, 40),
    ]

    failures = []
    for i, j in probes:
        x, y = grid_to_world(i, j)
        measured = surface_height(model, data, x, y)
        expected = float(hfr[i, j]) * VERTICAL_SCALE
        if measured is None:
            failures.append(f"[{i},{j}] at ({x:.3f},{y:.3f}): ray missed the terrain")
        elif abs(measured - expected) > 1e-3:
            failures.append(
                f"[{i},{j}] at ({x:.3f},{y:.3f}): got {measured:.4f}, want {expected:.4f}"
            )
    assert not failures, "heightfield orientation mismatch:\n  " + "\n  ".join(failures)


def test_orientation_probe_would_catch_a_flip(hfield_scene):
    """Guard on the guard: prove the pattern is asymmetric enough to fail a wrong map.

    Without this, a field that happened to be symmetric would let the test above pass
    against a flipped or transposed terrain, and the real bug would ship.
    """
    model, data, terrain = hfield_scene
    hfr = terrain.height_field_raw

    # Both indices kept below min(TOT_ROWS, TOT_COLS) so the transposed read is in
    # range, and off the i == TOT_ROWS-1-i / j == TOT_COLS-1-j midlines so a flip
    # actually moves the sample.
    for i, j in [(6, 30), (12, 7), (10, 8), (35, 12), (3, 26)]:
        x, y = grid_to_world(i, j)
        measured = surface_height(model, data, x, y)
        correct = float(hfr[i, j]) * VERTICAL_SCALE

        x_flipped = float(hfr[TOT_ROWS - 1 - i, j]) * VERTICAL_SCALE
        y_flipped = float(hfr[i, TOT_COLS - 1 - j]) * VERTICAL_SCALE
        # The transposed reading only exists where the index is in range; both of our
        # probe axes are, for these picks.
        transposed = float(hfr[j, i]) * VERTICAL_SCALE

        assert abs(measured - correct) <= 1e-3
        for name, wrong in (("x-flip", x_flipped), ("y-flip", y_flipped),
                            ("transpose", transposed)):
            assert abs(wrong - correct) > 1e-2, (
                f"probe [{i},{j}] cannot discriminate a {name}: both read {correct:.4f}"
            )
            assert abs(measured - wrong) > 1e-2


def test_hfield_surface_is_bracketed_by_its_cell_corners(hfield_scene):
    """Off-vertex points must land inside the cell they fall in.

    Vertex probes pin the mapping; this checks the field in between is not shifted by
    half a cell (which vertex-only sampling on a linear ramp could partially hide).
    """
    model, data, terrain = hfield_scene
    hfr = terrain.height_field_raw
    rng = np.random.default_rng(0)

    for _ in range(60):
        # Stay one cell inside the far edges so the containing cell is well defined.
        fi = rng.uniform(0.0, TOT_ROWS - 1.001)
        fj = rng.uniform(0.0, TOT_COLS - 1.001)
        x = fi * HORIZONTAL_SCALE - BORDER_SIZE
        y = fj * HORIZONTAL_SCALE - BORDER_SIZE
        i0, j0 = int(np.floor(fi)), int(np.floor(fj))

        corners = hfr[i0:i0 + 2, j0:j0 + 2].astype(np.float64) * VERTICAL_SCALE
        measured = surface_height(model, data, x, y)
        assert measured is not None, f"ray missed at ({x:.3f}, {y:.3f})"
        assert corners.min() - 1e-3 <= measured <= corners.max() + 1e-3, (
            f"({x:.3f},{y:.3f}) -> {measured:.4f} outside cell "
            f"[{i0},{j0}] range [{corners.min():.4f}, {corners.max():.4f}]"
        )


def test_hfield_world_origin_matches_the_contract():
    cfg = make_cfg("heightfield")
    assert hfield_world_origin(cfg) == (-BORDER_SIZE, -BORDER_SIZE)


def test_hfield_geom_spans_the_expected_world_extent(hfield_scene):
    """Field edges must sit at -border and (tot-1)*hs - border, not one cell off."""
    model, _, _ = hfield_scene
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    pos = model.geom_pos[gid]
    rx, ry = model.hfield_size[0][0], model.hfield_size[0][1]

    assert pos[0] - rx == pytest.approx(-BORDER_SIZE)
    assert pos[1] - ry == pytest.approx(-BORDER_SIZE)
    assert pos[0] + rx == pytest.approx((TOT_ROWS - 1) * HORIZONTAL_SCALE - BORDER_SIZE)
    assert pos[1] + ry == pytest.approx((TOT_COLS - 1) * HORIZONTAL_SCALE - BORDER_SIZE)
    # MuJoCo rows run along +y, columns along +x -- the transpose, asserted directly.
    assert model.hfield_nrow[0] == TOT_COLS
    assert model.hfield_ncol[0] == TOT_ROWS


# --------------------------------------------------------------------------------
# 2. The robot survived the surgery
# --------------------------------------------------------------------------------

def test_model_contains_the_robot_and_sim_options(hfield_scene):
    model, _, _ = hfield_scene
    cfg = make_cfg("heightfield")

    assert model.nu == 12, f"expected 12 motor actuators, got {model.nu}"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base") >= 0

    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    foot_bodies = sorted(n for n in names if n and n.endswith("_foot"))
    assert foot_bodies == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"], foot_bodies

    assert model.opt.timestep == cfg.sim.dt
    np.testing.assert_allclose(model.opt.gravity, cfg.sim.gravity)
    # Double-clipping guard: the caller owns torque limiting, MuJoCo must not re-clip.
    assert not model.actuator_ctrllimited.any()
    # The robot MJCF's own contact settings must survive being merged into the scene.
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == pytest.approx(100.0)
    assert model.nlight >= 1


def test_terrain_geom_friction_comes_from_cfg(hfield_scene):
    model, _, _ = hfield_scene
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert model.geom_condim[gid] == 3
    assert model.geom_friction[gid][0] == pytest.approx(0.8)


# --------------------------------------------------------------------------------
# 3. Plane mode and stair colliders
# --------------------------------------------------------------------------------

def test_plane_mode_builds_a_flat_ground_at_z0():
    cfg = make_cfg("plane")
    model = build_scene(cfg, None)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert gid >= 0
    assert model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE
    assert model.geom_size[gid][0] == pytest.approx(cfg.terrain.plane_length / 2)
    assert model.geom_size[gid][1] == pytest.approx(cfg.terrain.plane_length / 2)
    assert model.geom_condim[gid] == 3
    assert model.geom_friction[gid][0] == pytest.approx(cfg.terrain.static_friction)

    for x, y in [(0.0, 0.0), (3.0, -4.0), (-7.5, 7.5)]:
        assert surface_height(model, data, x, y) == pytest.approx(0.0, abs=1e-6)

    assert model.nu == 12
    assert model.opt.timestep == cfg.sim.dt


def test_play_stair_colliders_match_the_genesis_extents():
    stair_spec = {
        "x_start": 1.0,
        "step_width": 0.30,
        "y_start": -1.5,
        "y_stop": 1.5,
        "step_height": 0.12,
        "num_steps": 5,
    }
    cfg = make_cfg("plane", stair_spec=stair_spec)
    model = build_scene(cfg, None)

    for level in range(1, stair_spec["num_steps"] + 1):
        gid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"play_stair_collider_{level}"
        )
        assert gid >= 0, f"missing collider for step {level}"
        # Genesis: lower=(x0, y_start, -0.20), upper=(x1, y_stop, level*step_height).
        x0 = stair_spec["x_start"] + (level - 1) * stair_spec["step_width"]
        x1 = x0 + stair_spec["step_width"]
        z_top = level * stair_spec["step_height"]
        lower = np.array([x0, stair_spec["y_start"], -0.20])
        upper = np.array([x1, stair_spec["y_stop"], z_top])

        np.testing.assert_allclose(model.geom_pos[gid], (lower + upper) / 2, atol=1e-9)
        np.testing.assert_allclose(model.geom_size[gid], (upper - lower) / 2, atol=1e-9)

    # Absent spec must be a no-op, not a crash.
    plain = build_scene(make_cfg("plane"), None)
    assert mujoco.mj_name2id(
        plain, mujoco.mjtObj.mjOBJ_GEOM, "play_stair_collider_1"
    ) == -1


def test_flat_heightfield_does_not_divide_by_zero():
    """Degenerate but reachable: a fully flat patch must compile and sit at its level."""
    cfg = make_cfg("heightfield")
    flat_value = 40  # 40 * 0.005 = 0.20 m
    terrain = types.SimpleNamespace(
        height_field_raw=np.full((TOT_ROWS, TOT_COLS), flat_value, dtype=np.int16),
        tot_rows=TOT_ROWS,
        tot_cols=TOT_COLS,
    )
    model = build_scene(cfg, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for i, j in [(0, 0), (30, 20), (TOT_ROWS - 1, TOT_COLS - 1)]:
        x, y = grid_to_world(i, j)
        assert surface_height(model, data, x, y) == pytest.approx(
            flat_value * VERTICAL_SCALE, abs=1e-4
        )


def test_unknown_mesh_type_is_rejected():
    with pytest.raises(ValueError):
        build_scene(make_cfg("trimesh"), None)
