# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from typing import Sequence

import numpy as np
import trimesh

from . import terrain_utils
# Avoid importing legged_gym.envs here (that package pulls simulators which import
# Terrain) — keeps pure heightfield unit tests importable without a cycle.

# ---------------------------------------------------------------------------
# LP-ACRL taxonomy showcase (6 types x 4 levels) — pure geometry tables
# Source of truth: lpacr/solun_plani.md §4 / paper Appendix C.
# ---------------------------------------------------------------------------
TAXONOMY_NUM_LEVELS = 4
TAXONOMY_NUM_TYPES = 6
TAXONOMY_STEP_WIDTH = 0.4  # m, fixed for all stair levels
TAXONOMY_STEP_HEIGHTS = (0.05, 0.10, 0.15, 0.20)  # m, L0..L3
TAXONOMY_SLOPE_GRADIENTS = (0.00, 0.13, 0.27, 0.40)  # |gradient|, L0..L3
TAXONOMY_ROUGH_AMPLITUDES = (0.02, 0.045, 0.07, 0.10)  # m ±amplitude, L0..L3
# English paper-style labels (LP-ACRL taxonomy types).
TAXONOMY_TYPE_NAMES = (
    "Ascending stairs",
    "Descending stairs",
    "Upslope",
    "Downslope",
    "Random roughness",
    "Flat",
)
# Per-type RGBA for debug spheres / console legend when in-world text is unavailable.
TAXONOMY_TYPE_COLORS = (
    (1.0, 0.25, 0.2, 1.0),   # ascending stairs — red
    (0.2, 0.45, 1.0, 1.0),   # descending stairs — blue
    (1.0, 0.65, 0.1, 1.0),   # upslope — orange
    (0.15, 0.85, 0.35, 1.0), # downslope — green
    (0.75, 0.25, 0.85, 1.0), # random roughness — purple
    (0.7, 0.7, 0.7, 1.0),    # flat — gray
)
TAXONOMY_LABEL_Z_OFFSET = 1.5  # m above tile origin


def taxonomy_tile_label(level: int, type_idx: int) -> str:
    """Return the showcase label for grid cell (level row, type column)."""
    if not 0 <= level < TAXONOMY_NUM_LEVELS:
        raise ValueError(f"level must be in 0..{TAXONOMY_NUM_LEVELS - 1}, got {level}")
    if not 0 <= type_idx < TAXONOMY_NUM_TYPES:
        raise ValueError(f"type_idx must be in 0..{TAXONOMY_NUM_TYPES - 1}, got {type_idx}")
    return f"{TAXONOMY_TYPE_NAMES[type_idx]} L{level}"


def build_taxonomy_label_map(env_origins, z_offset: float = TAXONOMY_LABEL_Z_OFFSET):
    """Build per-tile label metadata from env_origins shaped (num_rows, num_cols, 3).

    Returns a list of dicts: row, col, label, color, position (x,y,z with z_offset).
    """
    origins = np.asarray(env_origins, dtype=np.float64)
    if origins.ndim != 3 or origins.shape[2] != 3:
        raise ValueError(f"env_origins must be (R, C, 3), got {origins.shape}")
    num_rows, num_cols = origins.shape[0], origins.shape[1]
    labels = []
    for i in range(num_rows):
        for j in range(num_cols):
            # Clamp type name index when a non-showcase grid is passed in.
            type_idx = min(j, TAXONOMY_NUM_TYPES - 1)
            level = min(i, TAXONOMY_NUM_LEVELS - 1)
            ox, oy, oz = origins[i, j]
            labels.append({
                "row": i,
                "col": j,
                "level": level,
                "type_idx": type_idx,
                "label": taxonomy_tile_label(level, type_idx) if (
                    num_rows == TAXONOMY_NUM_LEVELS and num_cols == TAXONOMY_NUM_TYPES
                ) else f"tile r{i}c{j}",
                "color": TAXONOMY_TYPE_COLORS[type_idx],
                "position": np.array([ox, oy, oz + z_offset], dtype=np.float64),
            })
    return labels


def format_taxonomy_console_map(label_map) -> str:
    """Human-readable 6×4 (cols×rows printed top=L3) grid of labels + positions."""
    by_rc = {(e["row"], e["col"]): e for e in label_map}
    if not by_rc:
        return "(empty taxonomy map)"
    max_r = max(r for r, _ in by_rc)
    max_c = max(c for _, c in by_rc)
    lines = ["Taxonomy showcase grid (row=L0.. bottom→top, col=type):"]
    # Print hardest level first so top-of-console matches top-of-grid (higher X).
    for i in range(max_r, -1, -1):
        cells = []
        for j in range(max_c + 1):
            e = by_rc[(i, j)]
            p = e["position"]
            cells.append(f"{e['label']} @({p[0]:.1f},{p[1]:.1f})")
        lines.append(f"  L{i}: " + " | ".join(cells))
    lines.append("Legend colors (type index): " + ", ".join(
        f"{k}:{TAXONOMY_TYPE_NAMES[k]}" for k in range(TAXONOMY_NUM_TYPES)
    ))
    return "\n".join(lines)


def is_taxonomy_terrain_cfg(cfg) -> bool:
    """True when cfg selects the LP-ACRL taxonomy showcase builder."""
    mode = getattr(cfg, "mode", None)
    if mode is not None and str(mode).lower() in ("taxonomy", "showcase"):
        return True
    return bool(getattr(cfg, "taxonomy_showcase", False))


def is_ued_training_terrain_cfg(cfg) -> bool:
    """True only for the static terrain grid used by the moving UED support."""
    return bool(getattr(cfg, "ued_training_grid", False))


# ---------------------------------------------------------------------------
# V6 frontier showcase — subsampled view of the real V4/ETH curriculum bank
# used by go2_v6_frontier.  Geometry comes from make_terrain(choice, difficulty)
# with training indices (level / 10, column / 12 + 0.001), NOT the V5 taxonomy
# tables.
# ---------------------------------------------------------------------------
V6_TRAINING_NUM_ROWS = 10          # v6 difficulty levels
V6_TRAINING_NUM_COLS = 10          # native V4 terrain columns
# Display names in FAMILY_COLUMNS order.  Signs match *experienced* direction
# (robot spawns on the center platform and walks outward): negative generator
# slope/step_height → ascending, positive → descending.  Same mapping as
# V4FrontierTaskSpace.TERRAIN_FAMILIES / FAMILY_COLUMNS.
V6_FAMILY_NAMES = (
    "Upslope",
    "Downslope",
    "Random roughness",
    "Ascending stairs",
    "Descending stairs",
    "Discrete obstacles",
)
V6_FAMILY_COLUMNS = ((0,), (1,), (2,), (3, 4, 5), (6, 7), (8, 9))
# Reuse the taxonomy palette style (one color per semantic family).
V6_FAMILY_COLORS = (
    (1.0, 0.65, 0.1, 1.0),   # upslope — orange
    (0.15, 0.85, 0.35, 1.0), # downslope — green
    (0.75, 0.25, 0.85, 1.0), # random roughness — purple
    (1.0, 0.25, 0.2, 1.0),   # ascending stairs — red
    (0.2, 0.45, 1.0, 1.0),   # descending stairs — blue
    (0.85, 0.75, 0.2, 1.0),  # discrete obstacles — gold
)
V6_SHOWCASE_LEVELS = (0, 2, 4, 6, 9)           # default subsampled rows
V6_SHOWCASE_COLUMNS = (0, 1, 2, 3, 6, 8)        # default: one column per family
V6_LABEL_Z_OFFSET = 1.5  # m above tile origin
# Play-default tile envelope matches the live V4-style training bank.
V6_PLAY_TILE_LENGTH = 8.0
V6_PLAY_TILE_WIDTH = 8.0
V6_PLAY_PLATFORM_SIZE = 4.0
# Default difficulty expressions (V4 base + v6 rough_height).  Used only when
# a label map is built without a cfg; the Terrain builder always reads the cfg.
V6_DEFAULT_DIFFICULTY = {
    "slope": "difficulty * 0.4",
    "step_height": "0.05 + 0.2 * difficulty",
    "discrete_height": "0.05 + 0.2 * difficulty",
    "rough_height": "0.01 + 0.10 * difficulty",
}


def is_v6_showcase_terrain_cfg(cfg) -> bool:
    """True when cfg selects the V6 frontier showcase builder.

    Takes precedence over ``ued_training_grid`` / taxonomy when dispatching in
    ``Terrain.__init__`` (checked first after ``height_field_raw_override``).
    """
    mode = getattr(cfg, "mode", None)
    if mode is not None and str(mode).lower() in (
        "v6", "v6_frontier", "frontier", "v6_full",
    ):
        return True
    return bool(getattr(cfg, "v6_frontier_showcase", False))


def v6_family_index_for_column(column: int) -> int:
    """Map a training column index to its semantic family (0..5)."""
    column = int(column)
    for family_idx, cols in enumerate(V6_FAMILY_COLUMNS):
        if column in cols:
            return family_idx
    raise ValueError(
        f"column {column} is outside the V6 bank (0..{V6_TRAINING_NUM_COLS - 1})"
    )


def v6_training_choice(column: int) -> float:
    """``choice`` fed to ``make_terrain`` for a training column index."""
    return int(column) / V6_TRAINING_NUM_COLS + 0.001


def v6_training_difficulty(level: int) -> float:
    """``difficulty`` fed to ``make_terrain`` for a training level index."""
    return int(level) / V6_TRAINING_NUM_ROWS


def _eval_terrain_difficulty_expr(expr: str, difficulty: float):
    """Eval a terrain_curriculum_difficulty expression (same scope as make_terrain)."""
    return eval(expr, {"np": np, "difficulty": float(difficulty)})


def v6_severity_label(family_idx: int, level: int, difficulty_cfg) -> str:
    """Human-readable label with evaluated severity for one showcase cell."""
    family_idx = int(family_idx)
    level = int(level)
    if not 0 <= family_idx < len(V6_FAMILY_NAMES):
        raise ValueError(f"family_idx out of range: {family_idx}")
    if not 0 <= level < V6_TRAINING_NUM_ROWS:
        raise ValueError(f"level out of range: {level}")
    difficulty = v6_training_difficulty(level)
    name = V6_FAMILY_NAMES[family_idx]
    cfg = difficulty_cfg or V6_DEFAULT_DIFFICULTY
    if family_idx in (0, 1):  # slopes
        slope = abs(float(_eval_terrain_difficulty_expr(cfg["slope"], difficulty)))
        deg = float(np.degrees(slope))
        return f"{name} L{level}  {slope:.2f} rad ({deg:.1f} deg)"
    if family_idx == 2:  # rough
        rough_expr = cfg.get("rough_height") if isinstance(cfg, dict) else None
        amp = (
            0.05
            if rough_expr is None
            else abs(float(_eval_terrain_difficulty_expr(rough_expr, difficulty)))
        )
        return f"{name} L{level}  +/-{amp * 100.0:.1f}cm"
    if family_idx in (3, 4):  # stairs
        h = abs(float(_eval_terrain_difficulty_expr(cfg["step_height"], difficulty)))
        return f"{name} L{level}  h={int(round(h * 100.0))}cm"
    # discrete obstacles
    h = abs(float(_eval_terrain_difficulty_expr(cfg["discrete_height"], difficulty)))
    return f"{name} L{level}  h={int(round(h * 100.0))}cm"


def build_v6_showcase_label_map(
    env_origins,
    levels=None,
    columns=None,
    difficulty_cfg=None,
    z_offset: float = V6_LABEL_Z_OFFSET,
):
    """Build per-tile label metadata for a V6 frontier showcase grid.

    Schema matches the taxonomy / Viser consumer: row, col, level, type_idx,
    label, color, position (x,y,z with z_offset).
    """
    origins = np.asarray(env_origins, dtype=np.float64)
    if origins.ndim != 3 or origins.shape[2] != 3:
        raise ValueError(f"env_origins must be (R, C, 3), got {origins.shape}")
    levels = tuple(int(x) for x in (V6_SHOWCASE_LEVELS if levels is None else levels))
    columns = tuple(int(x) for x in (V6_SHOWCASE_COLUMNS if columns is None else columns))
    num_rows, num_cols = origins.shape[0], origins.shape[1]
    if len(levels) != num_rows or len(columns) != num_cols:
        # Still label what we can; clamp display indices.
        pass
    labels = []
    for i in range(num_rows):
        for j in range(num_cols):
            level = levels[i] if i < len(levels) else i
            column = columns[j] if j < len(columns) else j
            family_idx = v6_family_index_for_column(column)
            ox, oy, oz = origins[i, j]
            labels.append({
                "row": i,
                "col": j,
                "level": int(level),
                "type_idx": int(family_idx),
                "label": v6_severity_label(family_idx, level, difficulty_cfg),
                "color": V6_FAMILY_COLORS[family_idx],
                "position": np.array([ox, oy, oz + z_offset], dtype=np.float64),
            })
    return labels


def format_v6_showcase_console_map(label_map) -> str:
    """Human-readable V6 showcase grid (hardest level printed first)."""
    by_rc = {(e["row"], e["col"]): e for e in label_map}
    if not by_rc:
        return "(empty v6 showcase map)"
    max_r = max(r for r, _ in by_rc)
    max_c = max(c for _, c in by_rc)
    lines = [
        "V6 frontier showcase grid "
        "(row=training level bottom→top, col=one column per family):"
    ]
    for i in range(max_r, -1, -1):
        cells = []
        for j in range(max_c + 1):
            e = by_rc[(i, j)]
            p = e["position"]
            cells.append(f"{e['label']} @({p[0]:.1f},{p[1]:.1f})")
        level_tag = by_rc[(i, 0)]["level"]
        lines.append(f"  L{level_tag}: " + " | ".join(cells))
    lines.append(
        "Legend (family index): "
        + ", ".join(f"{k}:{V6_FAMILY_NAMES[k]}" for k in range(len(V6_FAMILY_NAMES)))
    )
    return "\n".join(lines)


def ued_training_builder_parameters(cfg) -> dict:
    """Stable JSON-ready geometry description for ``TaskSpace.fingerprint``."""
    return {
        "builder": "ued_training_grid_v1",
        "terrain_type_names": ("stairs_up", "stairs_down", "slope_up", "slope_down", "rough", "flat"),
        "levels": TAXONOMY_NUM_LEVELS,
        "step_width": TAXONOMY_STEP_WIDTH,
        "step_heights": TAXONOMY_STEP_HEIGHTS,
        "slope_gradients": TAXONOMY_SLOPE_GRADIENTS,
        "rough_amplitudes": TAXONOMY_ROUGH_AMPLITUDES,
        "seed": int(getattr(cfg, "ued_training_seed", 0)),
        "horizontal_scale": float(cfg.horizontal_scale),
        "vertical_scale": float(cfg.vertical_scale),
        "terrain_length": float(cfg.terrain_length),
        "terrain_width": float(cfg.terrain_width),
    }


# ---------------------------------------------------------------------------
# Runtime geometry verification (solun_plani.md §7; reviewer item 7).
#
# The frozen UED validation/held-out banks pin one SHA-256 per (terrain_type,
# level) tile.  These hashes are computed over the ACTUAL int16 heightfield
# bytes of the built tile (plus the scale metadata that turns those integers
# into metres), NOT over a builder description.  Both the offline pin generator
# (`build_taxonomy_geometry_hashes`, headless/CPU) and the online rollout
# (`ued_rollout.py`, on the real Genesis scene) call the SAME `hash_terrain_tile`
# so a mismatch means the policy really ran on the wrong / corrupted geometry.
# ---------------------------------------------------------------------------
GEOMETRY_HASH_VERSION = "v5_ued_geometry_v2_hfbytes"

# Frozen terrain-builder parameters for the UED grid, mirrored from
# Go2BenchmarkV4TerrainCfg.terrain / Go2V5*Cfg.terrain (common_cfgs.py:514-528,
# go2_v5_config.py:164-171).  Kept here so the geometry pins can be regenerated
# headless (no Genesis/GPU); a runtime mismatch against the real scene is caught
# by the rollout's per-tile hash, and drift from the cfg is caught by
# tests/test_ued_checkpoint_selection.py.
UED_TRAINING_TERRAIN_PARAMS = {
    "mesh_type": "heightfield",
    "simplify_mesh": True,
    "horizontal_scale": 0.1,
    "vertical_scale": 0.005,
    "border_size": 20.0,
    "terrain_length": 8.0,
    "terrain_width": 8.0,
    "platform_size": 4.0,
    "num_rows": TAXONOMY_NUM_LEVELS,
    "num_cols": TAXONOMY_NUM_TYPES,
    "terrain_proportions": [0.2, 0.1, 0.25, 0.25, 0.2],
    "curriculum": False,
    "selected": False,
    "ued_training_grid": True,
    "ued_training_seed": 0,
    "slope_treshold": 0.75,
}

# Bank terrain-type name -> physical column index in the 4x6 UED grid, matching
# `_make_taxonomy_subterrain`'s type_idx branches and TaskSpace.TERRAIN_TYPE_NAMES.
UED_TERRAIN_TYPE_INDEX = {
    "stairs_up": 0, "stairs_down": 1, "slope_up": 2, "slope_down": 3, "rough": 4, "flat": 5,
}


def extract_terrain_tile(terrain, level: int, type_idx: int) -> np.ndarray:
    """Return the (length_per_env x width_per_env) int16 heightfield sub-block
    for grid cell (``level`` row, ``type_idx`` column) of a built ``Terrain``."""
    sx = terrain.border + int(level) * terrain.length_per_env_pixels
    ex = terrain.border + (int(level) + 1) * terrain.length_per_env_pixels
    sy = terrain.border + int(type_idx) * terrain.width_per_env_pixels
    ey = terrain.border + (int(type_idx) + 1) * terrain.width_per_env_pixels
    return terrain.height_field_raw[sx:ex, sy:ey]


def hash_height_field_tile(block, horizontal_scale: float, vertical_scale: float) -> str:
    """Canonical SHA-256 over one tile's raw int16 heightfield bytes + scales.

    The little-endian int16 byte layout plus the (horizontal, vertical) scales
    fully determine the physical tile geometry, so this hash changes iff the
    realized heightfield changes.  Used identically offline and at runtime."""
    import hashlib
    import json

    arr = np.ascontiguousarray(np.asarray(block, dtype="<i2"))
    header = {
        "v": GEOMETRY_HASH_VERSION,
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "horizontal_scale": float(horizontal_scale),
        "vertical_scale": float(vertical_scale),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def taxonomy_tile_geometry_hash(terrain, level: int, type_idx: int) -> str:
    """Runtime geometry hash for one built taxonomy tile (rollout entry point)."""
    block = extract_terrain_tile(terrain, level, type_idx)
    return hash_height_field_tile(block, terrain.cfg.horizontal_scale, terrain.cfg.vertical_scale)


def build_ued_training_terrain(params: dict | None = None):
    """Build the frozen UED 4x6 grid headless (pure numpy, no Genesis)."""
    from types import SimpleNamespace

    cfg = SimpleNamespace(**(params or UED_TRAINING_TERRAIN_PARAMS))
    return Terrain(cfg)


def build_taxonomy_geometry_hashes(
    terrain_types: Sequence[str] = ("stairs_up", "stairs_down", "slope_up", "slope_down", "rough"),
    *,
    flat_terrain_type: str = "flat",
    flat_terrain_level: int = 0,
    params: dict | None = None,
) -> dict[str, dict[int, str]]:
    """Regenerate the per-(type, level) heightfield geometry pins headless.

    Returns ``{terrain_type: {level: sha256}}`` for every moving type across all
    four levels plus the single flat cell.  This is the source of truth for the
    ``validation_bank.geometry_hashes`` pins in ``configs/eval/v5_ued.yaml``;
    the online rollout must reproduce the matching value per replica."""
    terrain = build_ued_training_terrain(params)
    hashes: dict[str, dict[int, str]] = {}
    for name in terrain_types:
        type_idx = UED_TERRAIN_TYPE_INDEX[name]
        hashes[name] = {
            level: taxonomy_tile_geometry_hash(terrain, level, type_idx)
            for level in range(TAXONOMY_NUM_LEVELS)
        }
    flat_idx = UED_TERRAIN_TYPE_INDEX[flat_terrain_type]
    hashes[flat_terrain_type] = {
        int(flat_terrain_level): taxonomy_tile_geometry_hash(terrain, flat_terrain_level, flat_idx)
    }
    return hashes


def clamp_taxonomy_spawn(level: int, type_idx: int, num_rows: int = TAXONOMY_NUM_LEVELS,
                         num_cols: int = TAXONOMY_NUM_TYPES):
    """Clamp (level, type) into a valid grid cell."""
    level = int(max(0, min(int(level), int(num_rows) - 1)))
    type_idx = int(max(0, min(int(type_idx), int(num_cols) - 1)))
    return level, type_idx


def apply_taxonomy_tile_assignment(
    terrain_levels,
    terrain_types,
    env_origins,
    terrain_origins,
    env_ids,
    level: int,
    type_idx: int,
):
    """Assign envs to a taxonomy tile origin (in-place).

    Works with numpy arrays or torch tensors.  ``terrain_origins`` is shaped
    (num_rows, num_cols, 3).  Returns the origin applied (length-3 vector /
    tensor row).
    """
    num_rows = int(terrain_origins.shape[0])
    num_cols = int(terrain_origins.shape[1])
    level, type_idx = clamp_taxonomy_spawn(level, type_idx, num_rows, num_cols)
    origin = terrain_origins[level, type_idx]
    # Support list/tuple/np/torch env_ids.
    try:
        import torch
        if isinstance(env_ids, torch.Tensor):
            ids_iter = env_ids.detach().cpu().tolist()
        else:
            ids_iter = list(env_ids)
    except Exception:
        ids_iter = list(env_ids)
    for i in ids_iter:
        ii = int(i)
        terrain_levels[ii] = level
        terrain_types[ii] = type_idx
        env_origins[ii] = origin
    return origin


def teleport_env_to_taxonomy_tile(env, env_index: int, level: int, type_idx: int) -> bool:
    """Teleport one play env onto a taxonomy tile and reset it there.

    Requires a heightfield/trimesh sim with ``_terrain_origins``.  Returns True
    on success.
    """
    import torch
    sim = env.simulator
    if not getattr(sim, "custom_origins", False):
        return False
    if not hasattr(sim, "_terrain_origins"):
        return False
    env_ids = torch.tensor([int(env_index)], device=env.device, dtype=torch.long)
    apply_taxonomy_tile_assignment(
        sim.terrain_levels,
        sim.terrain_types,
        sim.env_origins,
        sim._terrain_origins,
        env_ids,
        level,
        type_idx,
    )
    env.reset_idx(env_ids)
    return True


class Terrain:
    def __init__(self, cfg) -> None:

        self.cfg = cfg
        self.type = cfg.mesh_type
        self.simplify_mesh = cfg.simplify_mesh
        self.taxonomy_labels = []
        self.v6_showcase_labels = []
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.platform_size = cfg.platform_size
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        
        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # row - length, X
        # col - width,  Y
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
    
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        # edge mask to indicate the edge points of the terrain, for use in rewards
        self.edge_mask = np.zeros((self.tot_rows, self.tot_cols), dtype=bool)
        self.terrain_meshes = []
        # Eval V2 can supply a deterministic height map.  This avoids baking an
        # evaluator-specific terrain generator into the simulator and guarantees
        # that severity levels differ only by amplitude, not by a fresh RNG draw.
        # It is intentionally heightfield-only; trimesh follows a different
        # conversion path and is not supported by Genesis in this repository.
        override = getattr(cfg, "height_field_raw_override", None)
        if override is not None:
            if self.type != "heightfield":
                raise ValueError("height_field_raw_override requires mesh_type='heightfield'")
            raw = np.asarray(override, dtype=np.int16)
            expected = (self.tot_rows, self.tot_cols)
            if raw.shape != expected:
                raise ValueError(
                    f"height_field_raw_override shape {raw.shape} != terrain shape {expected}"
                )
            self.height_field_raw = raw.copy()
            self.heightsamples = self.height_field_raw
            return
        # V6 frontier showcase is checked before UED/taxonomy: those flags can
        # stay set on a V5/V6 task cfg, and play clears them, but an explicit
        # v6 mode must win if both are true (see is_v6_showcase_terrain_cfg).
        if is_v6_showcase_terrain_cfg(cfg):
            print(
                "Generating V6 frontier showcase terrain "
                f"({cfg.num_rows} levels x {cfg.num_cols} columns)..."
            )
            self.v6_frontier_showcase()
        elif is_ued_training_terrain_cfg(cfg):
            print("Generating deterministic UED training terrain (21 configs / 84 tasks)...")
            self.ued_training_grid()
        elif is_taxonomy_terrain_cfg(cfg):
            print("Generating taxonomy showcase terrain (6 types x 4 levels)...")
            self.taxonomy_showcase()
        elif cfg.curriculum and cfg.selected:
            raise ValueError("Curriculum and selected terrain cannot be both True.")
        elif cfg.curriculum:
            print("Generating curriculum terrain...")
            self.terrain_curriculum_difficulty = cfg.terrain_curriculum_difficulty
            self.curiculum()
        elif cfg.selected:
            print("Generating selected terrain...")
            self.selected_terrain()
        else:
            print("Generating randomized terrain...")
            self.randomized_terrain()
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self._add_terrain_border()
            self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)
            
            # self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
            #                                                                                 self.cfg.horizontal_scale,
            #                                                                                 self.cfg.vertical_scale,
            #                                                                                 self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):     # Y
            for i in range(self.cfg.num_rows): # X
                difficulty = i / self.cfg.num_rows      # add difficulty along X axis, row
                choice = j / self.cfg.num_cols + 0.001 # change terrain type along Y axis, col

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)
                
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs, terrain_type=self.type)
            self.add_terrain_to_map(terrain, i, j)

    def taxonomy_showcase(self):
        """Build the LP-ACRL 6-type × 4-level taxonomy grid as a fixed exhibit.

        Rows (X) are difficulty L0..L3 bottom→top; columns (Y) are:
          0 ascending stairs, 1 descending stairs, 2 upslope, 3 downslope,
          4 random roughness, 5 flat (degenerate geometry, still 4 cells).

        Curriculum promote/demote is intentionally not used here — this is a
        static showcase for visualization / play.
        """
        if self.cfg.num_rows != TAXONOMY_NUM_LEVELS or self.cfg.num_cols != TAXONOMY_NUM_TYPES:
            # Still build what was requested, but warn: geometry tables index by level.
            print(
                f"[taxonomy] warning: expected {TAXONOMY_NUM_LEVELS}x{TAXONOMY_NUM_TYPES} "
                f"grid, got {self.cfg.num_rows}x{self.cfg.num_cols}"
            )
        seed = int(getattr(self.cfg, "taxonomy_seed", 0))
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            for i in range(self.cfg.num_rows):
                for j in range(self.cfg.num_cols):
                    level = min(i, TAXONOMY_NUM_LEVELS - 1)
                    type_idx = min(j, TAXONOMY_NUM_TYPES - 1)
                    terrain = self._make_taxonomy_subterrain(level, type_idx)
                    self.add_terrain_to_map(terrain, i, j)
        finally:
            np.random.set_state(rng_state)

        self.taxonomy_labels = build_taxonomy_label_map(
            self.env_origins, z_offset=TAXONOMY_LABEL_Z_OFFSET
        )

    def v6_frontier_showcase(self):
        """Build a (sub)sampled V6 frontier task-space exhibit via make_terrain.

        Display grid size is ``cfg.num_rows x cfg.num_cols``.  Each display cell
        (r, c) is the *training* tile at
        ``(v6_showcase_levels[r], v6_showcase_columns[c])``, generated with

            difficulty = training_level / V6_TRAINING_NUM_ROWS   # always /10
            choice     = training_column / V6_TRAINING_NUM_COLS + 0.001  # /12

        i.e. the same arguments ``curiculum()`` would pass for that cell — not
        ``r / num_display_rows``.  Labels are also assigned to
        ``self.taxonomy_labels`` so existing Viser/console plumbing works.
        """
        levels = tuple(
            int(x) for x in getattr(self.cfg, "v6_showcase_levels", V6_SHOWCASE_LEVELS)
        )
        columns = tuple(
            int(x) for x in getattr(self.cfg, "v6_showcase_columns", V6_SHOWCASE_COLUMNS)
        )
        if len(levels) != self.cfg.num_rows or len(columns) != self.cfg.num_cols:
            print(
                f"[v6_showcase] warning: display grid is {self.cfg.num_rows}x"
                f"{self.cfg.num_cols} but levels/columns lists are "
                f"{len(levels)}x{len(columns)}; clamping indices"
            )
        # Must set before make_terrain (curriculum branch normally does this).
        self.terrain_curriculum_difficulty = self.cfg.terrain_curriculum_difficulty

        level_to_row = {int(lv): r for r, lv in enumerate(levels)}
        col_to_display = {int(col): c for c, col in enumerate(columns)}

        seed = int(getattr(self.cfg, "v6_showcase_seed", 0))
        rng_state = np.random.get_state()
        np.random.seed(seed)
        try:
            # Same outer/inner order as curiculum() (cols then rows) so that a
            # full 10x10 exhibit is byte-identical to a curriculum build with
            # the same seed.  When subsampled we still walk the full training
            # bank and only *place* the selected tiles — this keeps rough /
            # discrete RNG streams aligned with a full curriculum build at the
            # same seed for every (level, column) that appears in the exhibit.
            for j in range(V6_TRAINING_NUM_COLS):
                for i in range(V6_TRAINING_NUM_ROWS):
                    difficulty = v6_training_difficulty(i)
                    choice = v6_training_choice(j)
                    terrain = self.make_terrain(choice, difficulty)
                    if i in level_to_row and j in col_to_display:
                        self.add_terrain_to_map(
                            terrain, level_to_row[i], col_to_display[j]
                        )
        finally:
            np.random.set_state(rng_state)

        self.v6_showcase_labels = build_v6_showcase_label_map(
            self.env_origins,
            levels=levels[: self.cfg.num_rows],
            columns=columns[: self.cfg.num_cols],
            difficulty_cfg=self.terrain_curriculum_difficulty,
            z_offset=V6_LABEL_Z_OFFSET,
        )
        # Reuse taxonomy Viser/console plumbing without duplicating it.
        self.taxonomy_labels = self.v6_showcase_labels

    def ued_training_grid(self):
        """Build the static teleport grid, intentionally separate from exhibit mode.

        The physical layout is 4x6 so simulator origins remain addressable as
        ``[terrain_level, terrain_type]``.  The flat column is geometrically
        repeated but only its level-zero origin is part of `TaskSpace`, yielding
        five four-level terrain families plus one flat configuration (21 x four
        velocity bins = 84 task identities).
        """
        if self.cfg.num_rows != TAXONOMY_NUM_LEVELS or self.cfg.num_cols != TAXONOMY_NUM_TYPES:
            raise ValueError("ued_training_grid requires a 4-row by 6-column terrain grid")
        rng_state = np.random.get_state()
        np.random.seed(int(getattr(self.cfg, "ued_training_seed", 0)))
        try:
            for level in range(TAXONOMY_NUM_LEVELS):
                for type_idx in range(TAXONOMY_NUM_TYPES):
                    self.add_terrain_to_map(self._make_taxonomy_subterrain(level, type_idx), level, type_idx)
        finally:
            np.random.set_state(rng_state)

    def _make_taxonomy_subterrain(self, level: int, type_idx: int):
        """Create one SubTerrain cell for taxonomy type/level using shared generators."""
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        # Slightly smaller center platform than training defaults so stair/slope
        # rings remain visible on a 6 m tile.
        platform = min(float(self.platform_size), 2.0)

        # pyramid_stairs_terrain / pyramid_sloped_terrain always place the
        # *highest* point at the tile's center platform for a positive
        # parameter (rings/slope descend outward toward the tile edges) —
        # and the robot always spawns on that center platform (env_origins
        # = tile center, see add_terrain_to_map) with commands sampled
        # symmetrically in every direction. So an agent's actual traversal
        # is always "spawn on platform, walk outward": a positive parameter
        # is experienced as descending, negative as ascending. Signs below
        # are chosen so the *_up / *_down type names match that experienced
        # direction, not the raw generator-parameter sign.
        if type_idx == 0:  # ascending stairs: robot climbs walking away from spawn
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=TAXONOMY_STEP_WIDTH,
                step_height=-TAXONOMY_STEP_HEIGHTS[level],
                platform_size=platform,
                terrain_type=self.type,
                simplify_mesh=self.simplify_mesh,
            )
        elif type_idx == 1:  # descending stairs: robot descends walking away from spawn
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=TAXONOMY_STEP_WIDTH,
                step_height=TAXONOMY_STEP_HEIGHTS[level],
                platform_size=platform,
                terrain_type=self.type,
                simplify_mesh=self.simplify_mesh,
            )
        elif type_idx == 2:  # upslope: robot climbs walking away from spawn
            terrain_utils.pyramid_sloped_terrain(
                terrain,
                slope=-TAXONOMY_SLOPE_GRADIENTS[level],
                platform_size=platform,
                terrain_type=self.type,
            )
        elif type_idx == 3:  # downslope: robot descends walking away from spawn
            terrain_utils.pyramid_sloped_terrain(
                terrain,
                slope=TAXONOMY_SLOPE_GRADIENTS[level],
                platform_size=platform,
                terrain_type=self.type,
            )
        elif type_idx == 4:  # random roughness
            amp = TAXONOMY_ROUGH_AMPLITUDES[level]
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-amp,
                max_height=amp,
                step=0.005,
                downsampled_scale=0.2,
                terrain_type=self.type,
            )
        elif type_idx == 5:
            # Flat / platform only — leave zeros (degenerate geometry).
            pass
        else:
            raise ValueError(f"Unknown taxonomy type_idx={type_idx}")
        return terrain
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = eval(self.terrain_curriculum_difficulty["slope"])
        step_height = eval(self.terrain_curriculum_difficulty["step_height"])
        discrete_obstacles_height = eval(self.terrain_curriculum_difficulty["discrete_height"])
        # Optional per-tile severity jitter for geometry-bank curricula.  The
        # legacy builder has no such field, so existing V4 maps consume no extra
        # RNG draw and remain unchanged.  With the field enabled, replicas in
        # one semantic level are nearby samples rather than byte-identical
        # slope/stair tiles.
        replica_variation = float(getattr(self.cfg, "terrain_replica_variation", 0.0))
        replica_scale = (
            np.random.uniform(1.0 - replica_variation, 1.0 + replica_variation)
            if replica_variation > 0.0
            else 1.0
        )
        slope *= replica_scale
        step_height *= replica_scale
        discrete_obstacles_height *= replica_scale
        stepping_stones_params = self.terrain_curriculum_difficulty["stepping_stones_params"]
        gap_size = eval(self.terrain_curriculum_difficulty["gap_size"])
        pit_depth = eval(self.terrain_curriculum_difficulty["pit_depth"])
        # get params if exist
        high_platform_params = self.terrain_curriculum_difficulty.get("high_platform_params", None)
        high_platform_gaps_params = self.terrain_curriculum_difficulty.get("high_platform_gaps_params", None)
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2: # slope
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, 
                                                 slope=slope, 
                                                 platform_size=self.platform_size,
                                                 terrain_type=self.type)
        elif choice < self.proportions[1]: # random uniform
            # Legacy V4 keeps roughness fixed at ±5 cm.  A task-space
            # curriculum may opt into a level-dependent amplitude while using
            # the same generator; absence of the key preserves every existing
            # terrain byte-for-byte.
            rough_height_expr = self.terrain_curriculum_difficulty.get("rough_height")
            rough_height = (
                0.05 if rough_height_expr is None else eval(rough_height_expr)
            ) * replica_scale
            terrain_utils.random_uniform_terrain(terrain, 
                                                 min_height=-rough_height,
                                                 max_height=rough_height,
                                                 step=0.005, 
                                                 downsampled_scale=0.2, 
                                                 terrain_type=self.type)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]: # stairs
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, 
                                                 step_width=0.4, 
                                                 step_height=step_height, 
                                                 platform_size=self.platform_size,
                                                 terrain_type=self.type,
                                                 simplify_mesh=self.simplify_mesh)
        elif choice < self.proportions[4]: # discrete obstacles
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, 
                                                     discrete_obstacles_height, 
                                                     rectangle_min_size, 
                                                     rectangle_max_size, 
                                                     num_rectangles, 
                                                     platform_size=self.platform_size,
                                                     terrain_type=self.type,
                                                     simplify_mesh=self.simplify_mesh)
        elif choice < self.proportions[5]: # stepping stones
            terrain_utils.stepping_stones_terrain(terrain, 
                                                  stone_length=eval(stepping_stones_params["stone_length"]), 
                                                  stone_width=eval(stepping_stones_params["stone_width"]),
                                                  stone_distance_x=eval(stepping_stones_params["stone_distance_x"]),
                                                  stone_distance_y=eval(stepping_stones_params["stone_distance_y"]), 
                                                  max_height=eval(stepping_stones_params["max_height"]), 
                                                  platform_size=self.platform_size,
                                                  terrain_type=self.type,
                                                  simplify_mesh=self.simplify_mesh)
        elif choice < self.proportions[6]: # gap
            terrain_utils.gap_terrain(terrain, 
                                      gap_size=gap_size, 
                                      platform_size=self.platform_size,
                                      terrain_type=self.type,
                                      simplify_mesh=self.simplify_mesh)
        elif choice < self.proportions[7]: # pit
            terrain_utils.pit_terrain(terrain, 
                                      depth=pit_depth, 
                                      platform_size=self.platform_size,
                                      terrain_type=self.type,
                                      simplify_mesh=self.simplify_mesh)
        elif choice < self.proportions[8]: # multiple high platforms
            if high_platform_params is None:
                raise ValueError("high_platform_params is required for multiple high platforms terrain.")
            terrain_utils.multiple_high_platforms_terrain(terrain, 
                                                        high_platform_height=eval(high_platform_params["high_platform_height"]), 
                                                        high_platform_length=eval(high_platform_params["high_platform_length"]), 
                                                        high_platform_width=eval(high_platform_params["high_platform_width"]), 
                                                        high_platform_interval=eval(high_platform_params["high_platform_interval"]), 
                                                        platform_size=self.platform_size,
                                                        terrain_type=self.type,
                                                        simplify_mesh=self.simplify_mesh)
        elif choice < self.proportions[9]: # high platform gaps
            if high_platform_gaps_params is None:
                raise ValueError("high_platform_gaps_params is required for high platform gaps terrain.")
            terrain_utils.high_platform_gaps_terrain(terrain, 
                                                        high_platform_height=eval(high_platform_gaps_params["high_platform_height"]), 
                                                        high_platform_length=eval(high_platform_gaps_params["high_platform_length"]), 
                                                        high_platform_width=eval(high_platform_gaps_params["high_platform_width"]), 
                                                        high_platform_distance_y=eval(high_platform_gaps_params["high_platform_distance_y"]), 
                                                        gap_size=eval(high_platform_gaps_params["gap_size"]),
                                                        platform_size=self.platform_size,
                                                        terrain_type=self.type,
                                                        simplify_mesh=self.simplify_mesh)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw
        
        # add edge mask for the terrain, to indicate the edge points of the terrain, for use in rewards
        self.edge_mask[start_x: end_x, start_y:end_y] = terrain.edge_mask

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        # use the origin height as the max height of a 2mx2m square
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        
        if self.type == "trimesh":
            # apply translation to the trimesh, align with the env origin
            translation = np.array([
                start_x * terrain.horizontal_scale,
                start_y * terrain.horizontal_scale,
                0
            ])
            terrain.terrain_mesh.apply_translation(translation)
            self.terrain_meshes.append(terrain.terrain_mesh)
    
    #---------- Protected Methods ----------#
    
    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        # border parameters
        border_size = (
            self.cfg.num_rows * self.cfg.terrain_length + 2 * self.cfg.border_size,
            self.cfg.num_cols * self.cfg.terrain_width + 2 * self.cfg.border_size,
        )
        inner_size = (
            self.cfg.num_rows * self.cfg.terrain_length - self.cfg.horizontal_scale, # a small offset to align the subterrain with border
            self.cfg.num_cols * self.cfg.terrain_width - self.cfg.horizontal_scale
        )
        border_center = (
            self.cfg.num_rows * self.cfg.terrain_length / 2 + self.cfg.border_size,
            self.cfg.num_cols * self.cfg.terrain_width / 2 + self.cfg.border_size,
            -self.cfg.border_height / 2,
        )
        # border mesh
        border_meshes = terrain_utils.make_border(border_size, 
                                                  inner_size, 
                                                  height=abs(self.cfg.border_height), 
                                                  position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)
