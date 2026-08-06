"""Compile a MuJoCo `MjModel` for the play world: robot MJCF + plane or terrain heightfield.

WHY THIS MODULE EXISTS
----------------------
The training stack generates terrain procedurally (`legged_gym.utils.terrain.Terrain`)
and the Genesis backend consumes it directly. The MuJoCo backend needs the *same*
geometry, in the *same* world frame, because the policy's observations are computed
from the raw height field while the physics comes from the compiled geom. If those
two disagree by even one cell the robot's height scan describes a world it is not
standing in, and the failure is silent -- the policy just degrades. So the whole
point of this module is the frame contract below.

THE WORLD-FRAME CONTRACT (authoritative)
----------------------------------------
`terrain.height_field_raw` is an int16 array of shape `(tot_rows, tot_cols)`. The
Genesis backend places it with its `[0, 0]` corner at world `(-border_size, -border_size, 0)`:

    world_x = i * horizontal_scale - border_size     # i = row index    -> +x
    world_y = j * horizontal_scale - border_size     # j = column index -> +y
    world_z = height_field_raw[i, j] * vertical_scale

The rest of the codebase *depends* on this. `genesis_simulator.py::_update_surrounding_heights`
(~line 846) inverts exactly this mapping::

    points += border_size
    idx = (points / horizontal_scale).long()
    h = height_samples[idx_x, idx_y] * vertical_scale

so row index is the x axis and column index is the y axis. Anything we compile here
must reproduce that mapping to the cell.

HOW MUJOCO'S HEIGHTFIELD DIFFERS (determined empirically, not from memory)
--------------------------------------------------------------------------
MuJoCo stores `hfield_data` as `nrow x ncol` and -- verified by ray-casting a
deliberately asymmetric field (see `tests/test_mujoco_scene.py`) --

  * the MuJoCo **column** index runs along **+x**, spanning `2 * size[0]` (radius_x),
  * the MuJoCo **row** index runs along **+y**, spanning `2 * size[1]` (radius_y),
  * `hfield_data[0, 0]` sits at the (min x, min y) corner -- so there is **no flip**
    on either axis,
  * samples live at grid **vertices**, not cell centers: the last row/column lands
    exactly on the far edge, so `ncol` vertices span `(ncol - 1)` cells.

Genesis is `[i -> +x][j -> +y]`; MuJoCo is `[r -> +y][c -> +x]`. The two index
orders are swapped and neither axis is reversed, so the required transform is a
**pure transpose and nothing else**:

    hfield_data = height_field_raw.T        # shape (tot_cols, tot_rows)
    nrow = tot_cols                         # y extent
    ncol = tot_rows                         # x extent

Two further MuJoCo quirks that the geometry has to absorb:

  * `geom.pos` is the **center** of the field, not a corner. With vertex sampling
    the field spans `[-border, (tot_rows - 1) * hs - border]` in x, so the center is
    `(tot_rows - 1) * hs / 2 - border`.
  * The compiler **normalizes** the elevation samples to `[0, 1]` as
    `(v - v_min) / (v_max - v_min)`, then renders `z = geom.pos.z + normalized * size[2]`.
    To recover true metric heights we therefore feed pre-normalized data, set
    `size[2] = (v_max - v_min)` and push `v_min` into `geom.pos.z`. Pre-normalizing
    makes the compiler's own normalization a no-op, so the round trip is exact.

Public API (frozen -- the simulator is written against these two signatures):
    build_scene(cfg, terrain=None) -> mujoco.MjModel
    hfield_world_origin(cfg) -> tuple[float, float]
"""

from __future__ import annotations

import numpy as np
import mujoco

from legged_gym import LEGGED_GYM_ROOT_DIR

# Rendering group for the terrain / ground. Groups 2 (visual) and 3 (collision) are
# already claimed by the robot MJCF's `visual`/`collision` default classes, so group 0
# keeps the ground visible by default without colliding with the robot's own scheme.
_TERRAIN_GROUP = 0
# Play-only stair colliders go in group 3, matching the robot's collision geoms: the
# viewer hides that group by default, which reproduces Genesis's `visualization=False`
# while still letting a human toggle them on when debugging a stair failure.
_COLLIDER_GROUP = 3

# Solid skirt hanging below the lowest terrain sample. MuJoCo heightfields are a
# surface, not a volume; without a base the robot can tunnel out through the underside
# near steep walls. One meter is far more than a Go2 can penetrate in a single step.
_HFIELD_BASE_DEPTH = 1.0

# Torsional / rolling friction for ground contacts. MuJoCo wants a 3-vector; the config
# only carries a sliding coefficient, so we pair it with MuJoCo's own defaults.
_TORSIONAL_FRICTION = 0.005
_ROLLING_FRICTION = 0.0001


def hfield_world_origin(cfg) -> tuple[float, float]:
    """(x, y) world coords of ``height_field_raw[0, 0]``.

    This is the single place the `-border_size` offset is spelled out, so callers that
    need to convert between grid indices and world coordinates can share one definition
    with the geometry actually compiled in `build_scene`. It is well defined even in
    'plane' mode (there is no height field, but `border_size` still exists), which keeps
    caller code branch-free.
    """
    border = float(cfg.terrain.border_size)
    return (-border, -border)


def _hfield_geometry(cfg, terrain):
    """Derive every MuJoCo heightfield number from the Genesis frame contract.

    Split out from `build_scene` so the arithmetic that encodes the contract can be
    reasoned about (and unit-tested) without dragging in MJCF compilation.

    Returns a dict with the normalized data plus the geom placement.
    """
    hs = float(cfg.terrain.horizontal_scale)
    vs = float(cfg.terrain.vertical_scale)
    border = float(cfg.terrain.border_size)

    raw = np.asarray(terrain.height_field_raw)
    tot_rows = int(terrain.tot_rows)
    tot_cols = int(terrain.tot_cols)
    if raw.shape != (tot_rows, tot_cols):
        raise ValueError(
            f"height_field_raw shape {raw.shape} disagrees with "
            f"(tot_rows, tot_cols) = ({tot_rows}, {tot_cols})"
        )
    if tot_rows < 2 or tot_cols < 2:
        # A single vertex has no extent; MuJoCo would compile a degenerate geom.
        raise ValueError(f"height field too small to compile: {raw.shape}")

    # Metric heights, float64 for the min/max arithmetic; the cast to float32 happens
    # once, at the end, after normalization (int16 * vertical_scale can be large).
    heights = raw.astype(np.float64) * vs
    z_min = float(heights.min())
    z_max = float(heights.max())
    z_span = z_max - z_min

    # THE TRANSPOSE. Genesis indexes [x, y]; MuJoCo indexes [y, x]. Nothing is flipped
    # -- both put index 0 at the minimum-coordinate edge. Verified by ray casting in
    # tests/test_mujoco_scene.py::test_hfield_orientation_matches_genesis_contract.
    # `.T` is a view; ascontiguousarray materializes it once (vectorized, no per-cell loop).
    data_mj = np.ascontiguousarray(heights.T)  # shape (tot_cols, tot_rows) = (nrow, ncol)

    if z_span > 0.0:
        normalized = (data_mj - z_min) / z_span
        elevation = z_span
    else:
        # Perfectly flat field. The compiler's normalization divides by the range, and
        # `size[2]` must be strictly positive or compilation fails outright ("size
        # parameter is not positive in hfield"). All-zero data times any positive
        # elevation is still zero, so the plane lands exactly at z_min either way.
        normalized = np.zeros_like(data_mj)
        elevation = 1.0

    # Vertex sampling: `n` samples span `n - 1` cells, so the field's full width is
    # (tot_rows - 1) * hs and its half-width (MuJoCo's "radius") is half of that.
    radius_x = (tot_rows - 1) * hs * 0.5
    radius_y = (tot_cols - 1) * hs * 0.5

    # geom.pos is the CENTER of the field. Sample [0, 0] must land at (-border, -border)
    # per the contract, so the center sits one radius further along each axis.
    center_x = radius_x - border
    center_y = radius_y - border

    return {
        "data": normalized.astype(np.float32),
        "nrow": tot_cols,   # MuJoCo rows run along +y  -> Genesis column count
        "ncol": tot_rows,   # MuJoCo cols run along +x  -> Genesis row count
        # size = (radius_x, radius_y, elevation_z, base_z)
        "size": [radius_x, radius_y, elevation, _HFIELD_BASE_DEPTH],
        # pos.z = z_min because the compiler rebases the data to 0 at its minimum.
        "pos": [center_x, center_y, z_min],
    }


# Horizon tone. The skybox fades to it, and distant geometry hazes into it, so the two
# have to be the same colour or the terrain edge cuts a hard line against the sky.
_HORIZON_RGB = (0.62, 0.70, 0.78)

# Half-width, in metres, of the ground area the key light's shadow map covers. MuJoCo
# expresses this as `vis.map.shadowclip`, a *fraction of model extent*, but extent
# swings from ~20 m (plane worlds) to ~42 m and up (multi-tile terrain banks), so a
# fixed fraction means the shadow resolution silently changes with the terrain. We fix
# the metres instead and convert once the model is compiled and extent is known.
#
# Measured on both a 20 m plane world and a 42 m heightfield world: below ~4.5 m the
# shadow disappears entirely rather than degrading, so 6.0 keeps a margin over that
# cliff while still being tight enough that the shadow edge is smooth rather than
# stair-stepped. Only meaningful together with the tracked key light below -- a
# world-fixed light with a frustum this tight would drop the robot's shadow as soon as
# it walked away from the origin.
_SHADOW_RADIUS_M = 6.0


def _value_noise(shape, octaves, rng):
    """Wrapping multi-octave value noise in [0, 1].

    Wrapping matters: these textures are tiled across the whole world by `texrepeat`,
    and a seam at every tile boundary would read as a grid of lines on the ground.
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    amp_total = 0.0
    for octave in range(octaves):
        n = 2 ** (octave + 2)
        amp = 0.5 ** octave

        lattice = rng.random((n, n), dtype=np.float32)
        ys = np.linspace(0, n, h, endpoint=False)
        xs = np.linspace(0, n, w, endpoint=False)
        y0 = np.floor(ys).astype(int) % n
        x0 = np.floor(xs).astype(int) % n
        # The modulo on the *upper* neighbour is what makes the result wrap.
        y1 = (y0 + 1) % n
        x1 = (x0 + 1) % n

        fy = (ys - np.floor(ys)).astype(np.float32)[:, None]
        fx = (xs - np.floor(xs)).astype(np.float32)[None, :]
        fy = fy * fy * (3.0 - 2.0 * fy)   # smoothstep: rounded blobs, not diamonds
        fx = fx * fx * (3.0 - 2.0 * fx)

        top = lattice[np.ix_(y0, x0)] * (1.0 - fx) + lattice[np.ix_(y0, x1)] * fx
        bot = lattice[np.ix_(y1, x0)] * (1.0 - fx) + lattice[np.ix_(y1, x1)] * fx
        out += amp * (top * (1.0 - fy) + bot * fy)
        amp_total += amp

    return out / amp_total


def _terrain_texture_rgb(size=512, seed=7):
    """Earth-toned ground detail for the heightfield.

    Two noise bands rather than one: the low band keeps a large slope from reading as a
    single flat colour, and the high band keeps the surface textured right under the
    robot, which is where the eye actually judges whether it is moving.
    """
    rng = np.random.default_rng(seed)
    macro = _value_noise((size, size), octaves=3, rng=rng)
    micro = _value_noise((size, size), octaves=6, rng=rng)
    grain = rng.random((size, size), dtype=np.float32)

    dark = np.array([0.30, 0.28, 0.24], dtype=np.float32)
    light = np.array([0.55, 0.52, 0.44], dtype=np.float32)

    blend = np.clip(0.5 + 0.7 * (macro - 0.5) + 1.1 * (micro - 0.5), 0.0, 1.0)[..., None]
    rgb = dark + (light - dark) * blend
    rgb *= (0.90 + 0.20 * grain)[..., None]
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _floor_texture_rgb(size=512, seed=3):
    """Checker floor for plane worlds.

    A checker gives the eye a fixed reference for translation; on a featureless plane a
    walking robot is otherwise almost impossible to distinguish from a hovering one. The
    noise on top is deliberately weak -- at grazing angles a strong per-pixel grain
    aliases into crawling speckle across the whole floor.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(size)
    half = size // 2

    check = ((idx[:, None] // half + idx[None, :] // half) % 2).astype(np.float32)
    grain = _value_noise((size, size), octaves=5, rng=rng)

    base = np.array([0.30, 0.32, 0.35], dtype=np.float32)
    alt = np.array([0.37, 0.39, 0.43], dtype=np.float32)
    rgb = base + (alt - base) * check[..., None]
    rgb *= (0.96 + 0.08 * grain)[..., None]

    # Thin darker grout along the tile seams: a strong near-field depth cue.
    line = ((idx % half) < max(1, size // 256)).astype(np.float32)
    seam = np.maximum(line[:, None], line[None, :])[..., None]
    rgb *= 1.0 - 0.35 * seam
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def _add_rgb_texture(spec, name, rgb):
    """Attach an (H, W, 3) uint8 array as a 2D texture."""
    tex = spec.add_texture()
    tex.name = name
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
    tex.height, tex.width = int(rgb.shape[0]), int(rgb.shape[1])
    tex.nchannel = 3
    tex.data = rgb.tobytes()
    return tex


def _set_rgb_texture(material, texture_name):
    # `textures` is a fixed-length vector indexed by mjtTextureRole, not a plain list;
    # copy it out, set the RGB slot, and assign the whole thing back.
    roles = [""] * len(material.textures)
    roles[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name
    material.textures = roles


def _robot_root_body(spec):
    """The robot's floating base, or None if the MJCF has no free-jointed body.

    Used only to host the key light, so a miss is not fatal -- the caller falls back to
    a world-fixed light and merely loses the tight shadow frustum.
    """
    for body in spec.bodies:
        if body.name == "world":
            continue
        for joint in body.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                return body
    return None


def _add_visual_dressing(spec):
    """Skybox, ground materials and lights.

    This world is driven interactively by a human (play scripts, teleop), so a bare grey
    void is a real usability problem -- without a horizon, ground texture or shadows it
    is very hard to judge the robot's heading, its height off the ground, and the
    terrain's slope. None of this affects physics.
    """
    sky = spec.add_texture()
    sky.name = "skybox"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.15, 0.26, 0.43]
    # Bottom of the gradient is the horizon, so it has to match the haze colour.
    sky.rgb2 = list(_HORIZON_RGB)
    sky.width = 512
    sky.height = 3072

    _add_rgb_texture(spec, "groundtex", _floor_texture_rgb())
    _add_rgb_texture(spec, "terraintex", _terrain_texture_rgb())

    ground_mat = spec.add_material()
    ground_mat.name = "groundmat"
    _set_rgb_texture(ground_mat, "groundtex")
    # texuniform makes texrepeat a per-metre rate, so tile size is independent of how
    # large the plane geom happens to be: 1.0 here means one texture (two checkers)
    # per metre.
    ground_mat.texrepeat = [1.0, 1.0]
    ground_mat.texuniform = True
    ground_mat.reflectance = 0.12
    ground_mat.specular = 0.2
    ground_mat.shininess = 0.3

    terrain_mat = spec.add_material()
    terrain_mat.name = "terrainmat"
    _set_rgb_texture(terrain_mat, "terraintex")
    terrain_mat.texrepeat = [0.5, 0.5]
    terrain_mat.texuniform = True
    # White base: the colour comes from the texture, not from rgba tinting it.
    terrain_mat.rgba = [1.0, 1.0, 1.0, 1.0]
    terrain_mat.specular = 0.12
    terrain_mat.shininess = 0.15

    # Directional key light: casts the shadows that make slopes and stair risers legible,
    # and -- more than anything else -- tell the viewer whether the robot's feet are on
    # the ground.
    #
    # Hosted on the robot's floating base with mode=TRACK rather than in the worldbody:
    # TRACK moves the light with the body while holding its orientation fixed in world,
    # so the lighting direction never swims as the robot pitches and rolls, but the
    # shadow frustum stays centred on the robot. That is what lets `_SHADOW_RADIUS_M`
    # be small enough for a smooth shadow edge. The 12 m offset is not cosmetic: the
    # shadow camera has to sit above the geometry it renders, and dropping it to the
    # robot's own height loses the shadow completely.
    host = _robot_root_body(spec) or spec.worldbody
    key = host.add_light()
    key.name = "keylight"
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.pos = [0.0, 0.0, 12.0]
    key.dir = [-0.35, -0.35, -0.87]
    key.diffuse = [0.85, 0.82, 0.75]
    key.specular = [0.35, 0.35, 0.35]
    key.castshadow = True
    if host is not spec.worldbody:
        key.mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACK

    # Fill light from the opposite side, shadowless, so the shadowed faces of steps do
    # not go fully black. Tinted cool against the warm key.
    fill = spec.worldbody.add_light()
    fill.name = "filllight"
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.pos = [0.0, 0.0, 12.0]
    fill.dir = [0.5, 0.5, -0.75]
    fill.diffuse = [0.22, 0.25, 0.30]
    fill.specular = [0.0, 0.0, 0.0]
    fill.castshadow = False

    # Low ambient: the old 0.35 flattened every surface by lifting the shadowed faces
    # almost to the lit ones, which is most of why the arena looked depthless.
    spec.visual.headlight.ambient = [0.18, 0.19, 0.21]
    spec.visual.headlight.diffuse = [0.20, 0.20, 0.20]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]

    spec.visual.rgba.haze = [*_HORIZON_RGB, 1.0]
    spec.visual.map.haze = 0.25
    spec.visual.quality.shadowsize = 8192
    spec.visual.quality.offsamples = 8


def _apply_shadow_radius(model):
    """Convert `_SHADOW_RADIUS_M` into MuJoCo's extent-relative `shadowclip`.

    Post-compile because `stat.extent` is only known once the model exists.
    """
    extent = float(model.stat.extent)
    if extent > 0.0:
        model.vis.map.shadowclip = _SHADOW_RADIUS_M / extent


def _add_play_stair_colliders(spec, cfg):
    """Mirror `genesis_simulator.py::_add_play_stair_colliders` as MuJoCo boxes.

    Same reasoning as the Genesis version: a heightfield's sampled discontinuity is not
    a reliable rigid riser (the robot's toe can clip through the vertical face between
    samples), so the heightmap stays as the *visible* terrain while these boxes provide
    the solid volume underneath each step. Extents are copied exactly from the Genesis
    path so the two backends present identical stairs.
    """
    stair_spec = getattr(cfg.terrain, "play_stair_collision_spec", None)
    if not stair_spec:
        return

    friction = _ground_friction(cfg)
    num_steps = int(stair_spec["num_steps"])
    y_start = float(stair_spec["y_start"])
    y_stop = float(stair_spec["y_stop"])
    step_width = float(stair_spec["step_width"])
    step_height = float(stair_spec["step_height"])
    x_start = float(stair_spec["x_start"])

    # Genesis uses lower=(x0, y_start, -0.20), upper=(x1, y_stop, level*step_height).
    # The -0.20 floor buries each box below the surrounding ground so there is no seam.
    z_bottom = -0.20

    for level in range(1, num_steps + 1):
        x0 = x_start + (level - 1) * step_width
        x1 = x0 + step_width
        z_top = level * step_height

        box = spec.worldbody.add_geom()
        box.name = f"play_stair_collider_{level}"
        box.type = mujoco.mjtGeom.mjGEOM_BOX
        # MuJoCo boxes are center + half-extents; Genesis gave us lower/upper corners.
        box.pos = [0.5 * (x0 + x1), 0.5 * (y_start + y_stop), 0.5 * (z_bottom + z_top)]
        box.size = [
            0.5 * (x1 - x0),
            0.5 * (y_stop - y_start),
            0.5 * (z_top - z_bottom),
        ]
        box.condim = 3
        box.friction = friction
        box.group = _COLLIDER_GROUP
        # Translucent rather than invisible: hidden by default with group 3, but legible
        # the moment someone enables that group to debug a stair failure.
        box.rgba = [0.85, 0.55, 0.20, 0.35]


def _ground_friction(cfg) -> list[float]:
    """Sliding / torsional / rolling friction triple for ground contacts.

    NOTE ON WHY THIS BARELY MATTERS IN PRACTICE: the Go2 MJCF marks its four foot geoms
    `priority="1"`. MuJoCo resolves a contact pair by taking the friction of the geom
    with the higher priority *outright* rather than combining the two, so for any
    foot-vs-ground contact the foot's own friction wins and this value is ignored.
    It still applies to every non-foot contact (calf, thigh, base scraping the ground),
    which is exactly where terrain friction should matter, so we set it faithfully
    from the config -- just do not be surprised when changing
    `cfg.terrain.static_friction` alone does not visibly change gait.
    """
    return [float(cfg.terrain.static_friction), _TORSIONAL_FRICTION, _ROLLING_FRICTION]


def _apply_sim_options(model, cfg):
    """Physics options that must match the training-time simulator."""
    model.opt.timestep = float(cfg.sim.dt)
    model.opt.gravity[:] = np.asarray(cfg.sim.gravity, dtype=np.float64)

    # Newton + elliptic cone is the accurate combination for contact-rich locomotion.
    # `cone="elliptic"` and `impratio="100"` come from the robot MJCF's own <option> and
    # are deliberately left alone: a high impratio is what stops the feet from sliding
    # under load, and the elliptic cone is what makes that impratio meaningful.
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # implicitfast integrates joint damping and actuator forces implicitly. With the
    # stiff PD gains this stack uses, explicit Euler needs a much smaller dt to stay
    # stable; implicitfast costs almost nothing extra per step and removes that limit.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Newton converges quadratically, so 50 major iterations is generous for a 12-DoF
    # quadruped while still bounding worst-case step time for an interactive viewer.
    model.opt.iterations = 50
    model.opt.ls_iterations = 20

    # The caller applies its own torque clipping (config `torque_limits` / action scale)
    # before writing `data.ctrl`. The MJCF's `<motor ctrlrange="...">` would clip a
    # second time on top of that, which silently changes the effective torque limit and
    # makes the MuJoCo backend disagree with Genesis. Clearing the limit flag leaves the
    # ranges in the model for reference but stops MuJoCo from enforcing them.
    model.actuator_ctrllimited[:] = 0


def build_scene(cfg, terrain=None) -> "mujoco.MjModel":
    """Compile the robot MJCF plus the play world (plane or terrain heightfield).

    Args:
        cfg: legged_gym env config. Reads `cfg.asset.xml_file`, the `cfg.terrain.*`
            fields named in the module docstring, and `cfg.sim.dt` / `cfg.sim.gravity`.
        terrain: a `legged_gym.utils.terrain.Terrain` (or anything exposing
            `height_field_raw`, `tot_rows`, `tot_cols`). Required when
            `cfg.terrain.mesh_type == 'heightfield'`, ignored for 'plane'.

    Returns:
        A compiled `mujoco.MjModel` whose ground geometry satisfies the world-frame
        contract documented at the top of this module.
    """
    xml_path = cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

    # MjSpec rather than string-concatenating XML: `from_file` keeps the MJCF's own
    # <compiler meshdir="meshes"> resolution anchored to the robot XML's directory, so
    # the STL assets load without us having to rewrite any paths. Everything below is
    # then added programmatically to the same spec and compiled in one pass.
    spec = mujoco.MjSpec.from_file(xml_path)

    _add_visual_dressing(spec)

    mesh_type = cfg.terrain.mesh_type
    friction = _ground_friction(cfg)

    if mesh_type == "plane":
        half = float(cfg.terrain.plane_length) * 0.5
        ground = spec.worldbody.add_geom()
        ground.name = "terrain"  # same name in both modes: callers look up one geom
        ground.type = mujoco.mjtGeom.mjGEOM_PLANE
        # MuJoCo plane size = (x half-extent, y half-extent, render grid spacing).
        ground.size = [half, half, 0.5]
        ground.pos = [0.0, 0.0, 0.0]
        ground.condim = 3
        ground.friction = friction
        ground.group = _TERRAIN_GROUP
        ground.material = "groundmat"
    elif mesh_type == "heightfield":
        if terrain is None:
            raise ValueError("mesh_type == 'heightfield' requires a terrain object")
        geo = _hfield_geometry(cfg, terrain)

        hfield = spec.add_hfield()
        hfield.name = "terrain_hfield"
        hfield.nrow = geo["nrow"]
        hfield.ncol = geo["ncol"]
        hfield.size = geo["size"]
        # Flattened row-major, matching MuJoCo's (nrow, ncol) storage order.
        hfield.userdata = geo["data"].ravel()

        ground = spec.worldbody.add_geom()
        ground.name = "terrain"
        ground.type = mujoco.mjtGeom.mjGEOM_HFIELD
        ground.hfieldname = "terrain_hfield"
        ground.pos = geo["pos"]
        ground.condim = 3
        ground.friction = friction
        ground.group = _TERRAIN_GROUP
        ground.material = "terrainmat"
    else:
        raise ValueError(
            f"mujoco_scene supports mesh_type 'plane' and 'heightfield', got {mesh_type!r}"
        )

    _add_play_stair_colliders(spec, cfg)

    model = spec.compile()
    _apply_sim_options(model, cfg)
    _apply_shadow_radius(model)
    return model
