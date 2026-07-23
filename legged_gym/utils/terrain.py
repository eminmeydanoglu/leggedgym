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
        if is_ued_training_terrain_cfg(cfg):
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

        if type_idx == 0:  # ascending stairs
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=TAXONOMY_STEP_WIDTH,
                step_height=TAXONOMY_STEP_HEIGHTS[level],
                platform_size=platform,
                terrain_type=self.type,
                simplify_mesh=self.simplify_mesh,
            )
        elif type_idx == 1:  # descending stairs
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=TAXONOMY_STEP_WIDTH,
                step_height=-TAXONOMY_STEP_HEIGHTS[level],
                platform_size=platform,
                terrain_type=self.type,
                simplify_mesh=self.simplify_mesh,
            )
        elif type_idx == 2:  # upslope
            terrain_utils.pyramid_sloped_terrain(
                terrain,
                slope=TAXONOMY_SLOPE_GRADIENTS[level],
                platform_size=platform,
                terrain_type=self.type,
            )
        elif type_idx == 3:  # downslope
            terrain_utils.pyramid_sloped_terrain(
                terrain,
                slope=-TAXONOMY_SLOPE_GRADIENTS[level],
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
            terrain_utils.random_uniform_terrain(terrain, 
                                                 min_height=-0.05, 
                                                 max_height=0.05, 
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
