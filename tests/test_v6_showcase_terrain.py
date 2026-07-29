"""Unit tests for the V6 frontier showcase terrain exhibit.

Pure NumPy heightfield tests — does not import legged_gym.envs (avoids the
simulator import cycle documented at the top of legged_gym.utils.terrain).
"""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("SIMULATOR", "genesis")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from legged_gym.utils.terrain import (  # noqa: E402
    V6_DEFAULT_DIFFICULTY,
    V6_FAMILY_COLUMNS,
    V6_FAMILY_NAMES,
    V6_SHOWCASE_COLUMNS,
    V6_SHOWCASE_LEVELS,
    V6_TRAINING_NUM_COLS,
    V6_TRAINING_NUM_ROWS,
    Terrain,
    build_v6_showcase_label_map,
    extract_terrain_tile,
    format_v6_showcase_console_map,
    is_v6_showcase_terrain_cfg,
    v6_family_index_for_column,
    v6_severity_label,
    v6_training_choice,
    v6_training_difficulty,
)


# V4 base difficulty without the v6-only rough_height key.
_V4_DIFFICULTY = {
    "slope": "difficulty * 0.4",
    "step_height": "0.05 + 0.2 * difficulty",
    "discrete_height": "0.05 + 0.2 * difficulty",
    "stepping_stones_params": {
        "stone_length": "1.6",
        "stone_width": "1.0",
        "stone_distance_x": "0.1",
        "stone_distance_y": "0.3",
        "max_height": "0",
    },
    "gap_size": "0.1",
    "pit_depth": "0.1",
}

_V6_DIFFICULTY = {
    **_V4_DIFFICULTY,
    "rough_height": "0.01 + 0.10 * difficulty",
}

_V6_PROPORTIONS = [0.2, 0.1, 0.25, 0.25, 0.2]


def _base_cfg(**overrides):
    cfg = SimpleNamespace(
        mesh_type="heightfield",
        simplify_mesh=False,
        terrain_length=8.0,
        terrain_width=8.0,
        platform_size=4.0,
        terrain_proportions=list(_V6_PROPORTIONS),
        num_rows=V6_TRAINING_NUM_ROWS,
        num_cols=V6_TRAINING_NUM_COLS,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=2.0,
        curriculum=False,
        selected=False,
        mode=None,
        taxonomy_showcase=False,
        v6_frontier_showcase=False,
        ued_training_grid=False,
        terrain_curriculum_difficulty=dict(_V6_DIFFICULTY),
        terrain_replica_variation=0.0,
        v6_showcase_seed=0,
        v6_showcase_levels=tuple(V6_SHOWCASE_LEVELS),
        v6_showcase_columns=tuple(V6_SHOWCASE_COLUMNS),
        height_field_raw_override=None,
        terrain_kwargs=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _training_curriculum_cfg(**overrides):
    defaults = dict(
        curriculum=True,
        num_rows=V6_TRAINING_NUM_ROWS,
        num_cols=V6_TRAINING_NUM_COLS,
    )
    defaults.update(overrides)
    return _base_cfg(**defaults)


def _showcase_cfg(levels=None, columns=None, **overrides):
    levels = tuple(V6_SHOWCASE_LEVELS if levels is None else levels)
    columns = tuple(V6_SHOWCASE_COLUMNS if columns is None else columns)
    defaults = dict(
        mode="v6",
        v6_frontier_showcase=True,
        curriculum=False,
        num_rows=len(levels),
        num_cols=len(columns),
        v6_showcase_levels=levels,
        v6_showcase_columns=columns,
    )
    defaults.update(overrides)
    return _base_cfg(**defaults)


def _tile_max_abs_m(tile: np.ndarray, vertical_scale: float) -> float:
    return float(np.max(np.abs(tile.astype(np.float64))) * vertical_scale)


def _is_piecewise_constant_along_x(tile: np.ndarray, min_plateaus: int = 3) -> bool:
    """Stairs / discrete tiles: many constant bands along the row (X) midline."""
    mid_y = tile.shape[1] // 2
    profile = tile[:, mid_y]
    # Count runs of equal height.
    runs = 1 + int(np.sum(profile[1:] != profile[:-1]))
    return runs >= min_plateaus


def _slope_is_monotone_outward(tile: np.ndarray, ascending: bool) -> bool:
    """Pyramid slopes: center platform is extreme; height falls/rises outward in X.

    ``ascending`` (experienced): platform is the *low* point (negative slope param).
    ``descending``: platform is the *high* point (positive slope param).
    """
    mid_x = tile.shape[0] // 2
    mid_y = tile.shape[1] // 2
    # Sample from platform outward along +X (away from center).
    platform = float(tile[mid_x, mid_y])
    outer = float(tile[tile.shape[0] - 2, mid_y])
    if ascending:
        return outer > platform + 1  # quantized units
    return outer < platform - 1


class TestV6GeometryEquivalence(unittest.TestCase):
    """Critical: showcase tiles equal curriculum tiles at the same (level, col)."""

    @classmethod
    def setUpClass(cls):
        seed = 0
        np.random.seed(seed)
        cls.train = Terrain(_training_curriculum_cfg())
        # Showcase must start from the same seed; Terrain reseeds internally.
        cls.show = Terrain(
            _showcase_cfg(
                levels=V6_SHOWCASE_LEVELS,
                columns=V6_SHOWCASE_COLUMNS,
                v6_showcase_seed=seed,
            )
        )
        cls.vs = 0.005

    def test_all_displayed_tiles_match_training(self):
        """Exact heightfield equality for every (level, column) in the exhibit.

        The showcase walks the full 10x10 bank in ``curiculum()`` order and only
        places selected cells, so rough/discrete RNG streams align with a full
        curriculum build at the same seed.  With terrain_replica_variation=0,
        slopes/stairs are deterministic and also match.
        """
        for r, level in enumerate(V6_SHOWCASE_LEVELS):
            for c, column in enumerate(V6_SHOWCASE_COLUMNS):
                train_tile = extract_terrain_tile(self.train, level, column)
                show_tile = extract_terrain_tile(self.show, r, c)
                self.assertTrue(
                    np.array_equal(train_tile, show_tile),
                    msg=(
                        f"tile mismatch training (L{level}, c{column}) vs "
                        f"showcase display (r{r}, c{c})"
                    ),
                )

    def test_full_showcase_matches_full_curriculum_heightfield(self):
        """v6_full (10x10) is byte-identical to curriculum under the same seed."""
        seed = 7
        np.random.seed(seed)
        train = Terrain(_training_curriculum_cfg())
        show = Terrain(
            _showcase_cfg(
                levels=tuple(range(V6_TRAINING_NUM_ROWS)),
                columns=tuple(range(V6_TRAINING_NUM_COLS)),
                v6_showcase_seed=seed,
            )
        )
        self.assertTrue(np.array_equal(train.height_field_raw, show.height_field_raw))


class TestV6LevelSubsampling(unittest.TestCase):
    def test_display_row_uses_training_level_not_display_fraction(self):
        """A 5-row exhibit of levels (0,2,4,6,9) must NOT equal difficulty=r/5.

        Display row r=4 is training L9 → difficulty 0.9.  Naive r/5 → 0.8.
        Stair geometry at those difficulties differs.
        """
        levels = (0, 2, 4, 6, 9)
        columns = (6,)  # ascending stairs family
        show = Terrain(_showcase_cfg(levels=levels, columns=columns, v6_showcase_seed=0))
        # Training reference at L9 stairs_up.
        np.random.seed(0)
        train = Terrain(_training_curriculum_cfg())
        train_l9 = extract_terrain_tile(train, 9, 6)
        show_r4 = extract_terrain_tile(show, 4, 0)
        self.assertTrue(np.array_equal(train_l9, show_r4))

        # Build a single-cell "naive" tile: difficulty = display_row / 5.
        naive_cfg = _base_cfg(
            curriculum=True,
            num_rows=5,  # so difficulty for row 4 is 4/5 = 0.8
            num_cols=1,
            terrain_proportions=[0.0, 0.0, 1.0, 0.0, 0.0],  # force stairs branch
        )
        # Force stairs_up via proportions: choice = 0/1+0.001 = 0.001, need
        # proportions such that we hit stairs with negative height.
        # Easier: call make_terrain directly on a temporary Terrain.
        tmp = Terrain(
            _base_cfg(
                curriculum=False,
                selected=False,
                mode="v6",
                v6_frontier_showcase=True,
                num_rows=1,
                num_cols=1,
                v6_showcase_levels=(0,),
                v6_showcase_columns=(6,),
                v6_showcase_seed=0,
            )
        )
        # Manually build a tile with wrong difficulty 0.8 at choice for col 6.
        tmp.terrain_curriculum_difficulty = dict(_V6_DIFFICULTY)
        wrong = tmp.make_terrain(v6_training_choice(6), 4.0 / 5.0)
        self.assertFalse(
            np.array_equal(show_r4, wrong.height_field_raw),
            msg="L9 stairs should differ from naive difficulty=0.8 stairs",
        )


class TestV6ColumnSubsampling(unittest.TestCase):
    def test_display_col_maps_to_training_column_and_family(self):
        show = Terrain(_showcase_cfg(v6_showcase_seed=0))
        for c, column in enumerate(V6_SHOWCASE_COLUMNS):
            family = v6_family_index_for_column(column)
            self.assertEqual(family, c)  # default columns pick one per family
            self.assertEqual(V6_FAMILY_NAMES[family], V6_FAMILY_NAMES[c])
        # Labels carry the expected family name at each display col.
        for entry in show.v6_showcase_labels:
            expected_family = v6_family_index_for_column(
                V6_SHOWCASE_COLUMNS[entry["col"]]
            )
            self.assertEqual(entry["type_idx"], expected_family)
            self.assertIn(V6_FAMILY_NAMES[expected_family], entry["label"])


class TestV6FamilyDispatch(unittest.TestCase):
    """Structural checks — not just re-running the choice arithmetic."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.train = Terrain(_training_curriculum_cfg())
        cls.vs = 0.005
        cls.level = 6  # difficulty 0.6 — clear geometric signal
        cls.difficulty = v6_training_difficulty(cls.level)

    def test_all_ten_columns_match_family_columns(self):
        for column in range(V6_TRAINING_NUM_COLS):
            family = v6_family_index_for_column(column)
            tile = extract_terrain_tile(self.train, self.level, column)
            if family == 0:  # slope_up / ascending experience
                self.assertTrue(
                    _slope_is_monotone_outward(tile, ascending=True),
                    msg=f"col {column} should be ascending slope",
                )
            elif family == 1:  # slope_down
                self.assertTrue(
                    _slope_is_monotone_outward(tile, ascending=False),
                    msg=f"col {column} should be descending slope",
                )
            elif family == 2:  # rough
                amp = 0.01 + 0.10 * self.difficulty
                max_abs = _tile_max_abs_m(tile, self.vs)
                # Quantization + random samples: max abs within amp + one vertical step.
                self.assertLessEqual(max_abs, amp + self.vs + 1e-9)
                self.assertGreater(max_abs, 0.0)
            elif family == 3:  # stairs_up
                self.assertTrue(_is_piecewise_constant_along_x(tile))
                self.assertLess(int(tile.min()), 0, msg="ascending stairs: platform low")
            elif family == 4:  # stairs_down
                self.assertTrue(_is_piecewise_constant_along_x(tile))
                self.assertGreater(int(tile.max()), 0, msg="descending stairs: platform high")
            elif family == 5:  # discrete
                expected_h = 0.05 + 0.2 * self.difficulty
                max_abs = _tile_max_abs_m(tile, self.vs)
                # Rectangle heights are sampled inside the configured bound.
                self.assertGreater(max_abs, 0.0)
                self.assertLessEqual(max_abs, expected_h + self.vs + 1e-9)
            # FAMILY_COLUMNS ground truth
            self.assertIn(column, V6_FAMILY_COLUMNS[family])

    def test_family_columns_partition(self):
        covered = sorted(c for pair in V6_FAMILY_COLUMNS for c in pair)
        self.assertEqual(covered, list(range(V6_TRAINING_NUM_COLS)))
        self.assertEqual(len(V6_FAMILY_NAMES), len(V6_FAMILY_COLUMNS))


class TestV6RoughHeightScaling(unittest.TestCase):
    def test_v6_rough_scales_l0_to_l9(self):
        np.random.seed(0)
        train = Terrain(_training_curriculum_cfg())
        vs = 0.005
        # Native V4 column 2 = rough family.
        l0 = extract_terrain_tile(train, 0, 2)
        l9 = extract_terrain_tile(train, 9, 2)
        amp0 = 0.01 + 0.10 * 0.0  # 1 cm
        amp9 = 0.01 + 0.10 * 0.9  # 10 cm
        self.assertLessEqual(_tile_max_abs_m(l0, vs), amp0 + vs + 1e-9)
        self.assertGreater(_tile_max_abs_m(l0, vs), 0.0)
        # L9 should reach near 10 cm (stochastic; allow missing the absolute peak).
        self.assertGreater(_tile_max_abs_m(l9, vs), 0.05)
        self.assertLessEqual(_tile_max_abs_m(l9, vs), amp9 + vs + 1e-9)

    def test_v4_without_rough_height_stays_fixed_5cm(self):
        """Regression: omitting rough_height must keep legacy ±5 cm roughness."""
        np.random.seed(1)
        cfg = _training_curriculum_cfg(
            terrain_curriculum_difficulty=dict(_V4_DIFFICULTY),
            # V4 proportions still include a rough band.
            terrain_proportions=[0.2, 0.1, 0.25, 0.25, 0.2],
            num_cols=10,
        )
        train = Terrain(cfg)
        vs = 0.005
        # With V4 proportions [0.2,0.1,...] and 10 cols, rough is columns where
        # choice in [0.2, 0.3): col 2 → 2/10+0.001=0.201.
        tile = extract_terrain_tile(train, 9, 2)
        max_abs = _tile_max_abs_m(tile, vs)
        self.assertLessEqual(max_abs, 0.05 + vs + 1e-9)
        self.assertGreater(max_abs, 0.0)
        # Must NOT scale to v6's L9 ±10 cm.
        self.assertLess(max_abs, 0.08)


class TestV6LabelMap(unittest.TestCase):
    def test_schema_and_spot_severities(self):
        show = Terrain(_showcase_cfg(v6_showcase_seed=0))
        labels = show.v6_showcase_labels
        self.assertEqual(len(labels), len(V6_SHOWCASE_LEVELS) * len(V6_SHOWCASE_COLUMNS))
        required = {"row", "col", "level", "type_idx", "label", "color", "position"}
        for e in labels:
            self.assertTrue(required.issubset(e.keys()))
            self.assertEqual(len(e["color"]), 4)
            self.assertEqual(np.asarray(e["position"]).shape, (3,))

        # Spot-check severity strings against eval of the same expressions.
        by = {(e["row"], e["col"]): e for e in labels}
        # Ascending stairs at display row for L6, representative column 3.
        r_l6 = V6_SHOWCASE_LEVELS.index(6)
        c_stairs = V6_SHOWCASE_COLUMNS.index(3)
        lab = by[(r_l6, c_stairs)]["label"]
        self.assertIn("Ascending stairs L6", lab)
        self.assertIn("h=17cm", lab)  # 0.05 + 0.2*0.6 = 0.17

        r_l4 = V6_SHOWCASE_LEVELS.index(4)
        c_up = V6_SHOWCASE_COLUMNS.index(0)
        lab = by[(r_l4, c_up)]["label"]
        self.assertIn("Upslope L4", lab)
        self.assertIn("0.16 rad", lab)  # 0.4 * 0.4
        self.assertIn("deg", lab)

        r_l9 = V6_SHOWCASE_LEVELS.index(9)
        c_rough = V6_SHOWCASE_COLUMNS.index(2)
        lab = by[(r_l9, c_rough)]["label"]
        self.assertIn("Random roughness L9", lab)
        self.assertIn("+/-10.0cm", lab)

        r_l2 = V6_SHOWCASE_LEVELS.index(2)
        c_disc = V6_SHOWCASE_COLUMNS.index(8)
        lab = by[(r_l2, c_disc)]["label"]
        self.assertIn("Discrete obstacles L2", lab)
        self.assertIn("h=9cm", lab)  # 0.05 + 0.2*0.2 = 0.09

    def test_console_map_hardest_first(self):
        origins = np.zeros((len(V6_SHOWCASE_LEVELS), len(V6_SHOWCASE_COLUMNS), 3))
        for i, lv in enumerate(V6_SHOWCASE_LEVELS):
            for j, col in enumerate(V6_SHOWCASE_COLUMNS):
                origins[i, j] = [(i + 0.5) * 8.0, (j + 0.5) * 8.0, 0.0]
        label_map = build_v6_showcase_label_map(
            origins,
            levels=V6_SHOWCASE_LEVELS,
            columns=V6_SHOWCASE_COLUMNS,
            difficulty_cfg=_V6_DIFFICULTY,
        )
        text = format_v6_showcase_console_map(label_map)
        # Hardest training level appears before easiest in the printed text.
        self.assertLess(text.index("L9:"), text.index("L0:"))
        for name in V6_FAMILY_NAMES:
            self.assertIn(name, text)

    def test_severity_helper_matches_default_formulas(self):
        self.assertEqual(
            v6_severity_label(3, 6, V6_DEFAULT_DIFFICULTY),
            "Ascending stairs L6  h=17cm",
        )


class TestV6Priority(unittest.TestCase):
    def test_v6_wins_over_ued_training_grid(self):
        """Documented precedence: v6 showcase before UED grid in Terrain.__init__."""
        cfg = _showcase_cfg(
            levels=(0, 2),
            columns=(0, 2),
            ued_training_grid=True,  # would otherwise build 4x6 taxonomy grid
            v6_frontier_showcase=True,
            mode="v6",
        )
        self.assertTrue(is_v6_showcase_terrain_cfg(cfg))
        terrain = Terrain(cfg)
        # Display grid is 2x2, not the UED 4x6.
        self.assertEqual(terrain.env_origins.shape, (2, 2, 3))
        self.assertTrue(len(terrain.v6_showcase_labels) == 4)

    def test_is_v6_showcase_cfg_modes(self):
        self.assertTrue(is_v6_showcase_terrain_cfg(SimpleNamespace(mode="v6")))
        self.assertTrue(is_v6_showcase_terrain_cfg(SimpleNamespace(mode="v6_frontier")))
        self.assertTrue(is_v6_showcase_terrain_cfg(SimpleNamespace(mode="frontier")))
        self.assertTrue(is_v6_showcase_terrain_cfg(SimpleNamespace(mode="v6_full")))
        self.assertTrue(
            is_v6_showcase_terrain_cfg(
                SimpleNamespace(mode=None, v6_frontier_showcase=True)
            )
        )
        self.assertFalse(
            is_v6_showcase_terrain_cfg(
                SimpleNamespace(mode=None, v6_frontier_showcase=False)
            )
        )


class TestV6PlayConfig(unittest.TestCase):
    def test_configure_play_terrain_v6(self):
        from legged_gym.scripts import play as play_mod

        terrain = SimpleNamespace(
            mesh_type="plane",
            curriculum=True,
            selected=False,
            terrain_kwargs={"type": "x"},
            num_rows=10,
            num_cols=10,
            border_size=20.0,
            max_init_terrain_level=5,
            fixed_terrain_level=None,
            terrain_length=6.0,
            terrain_width=6.0,
            platform_size=2.0,
            horizontal_scale=0.1,
            simplify_mesh=True,
            mode=None,
            taxonomy_showcase=False,
            ued_training_grid=True,
            terrain_proportions=[0.2, 0.1, 0.25, 0.25, 0.2],
            terrain_curriculum_difficulty={},
        )
        env_cfg = SimpleNamespace(terrain=terrain)
        play_mod.configure_play_terrain(env_cfg, "v6")
        self.assertTrue(env_cfg.terrain.v6_frontier_showcase)
        self.assertFalse(env_cfg.terrain.curriculum)
        self.assertFalse(env_cfg.terrain.selected)
        self.assertFalse(env_cfg.terrain.ued_training_grid)
        self.assertFalse(env_cfg.terrain.taxonomy_showcase)
        self.assertEqual(env_cfg.terrain.num_rows, len(V6_SHOWCASE_LEVELS))
        self.assertEqual(env_cfg.terrain.num_cols, len(V6_SHOWCASE_COLUMNS))
        self.assertEqual(env_cfg.terrain.border_size, 2.0)
        self.assertEqual(env_cfg.terrain.max_init_terrain_level, 0)
        # Default play tile envelope is the legacy 8 m showcase size.
        self.assertEqual(env_cfg.terrain.v6_tile_size, "play")
        self.assertAlmostEqual(env_cfg.terrain.terrain_length, 8.0)
        self.assertAlmostEqual(env_cfg.terrain.terrain_width, 8.0)
        self.assertAlmostEqual(env_cfg.terrain.platform_size, 4.0)
        self.assertIn("rough_height", env_cfg.terrain.terrain_curriculum_difficulty)
        self.assertEqual(env_cfg.terrain.terrain_replica_variation, 0.10)
        self.assertFalse(env_cfg.terrain.simplify_mesh)

        play_mod.configure_play_terrain(env_cfg, "v6", v6_tile_size="train")
        self.assertEqual(env_cfg.terrain.v6_tile_size, "train")
        self.assertAlmostEqual(env_cfg.terrain.terrain_length, 8.0)
        self.assertAlmostEqual(env_cfg.terrain.terrain_width, 8.0)
        self.assertAlmostEqual(env_cfg.terrain.platform_size, 4.0)
        # Severity contract still comes from training either way.
        self.assertIn("rough_height", env_cfg.terrain.terrain_curriculum_difficulty)

    def test_configure_play_terrain_v6_full(self):
        from legged_gym.scripts import play as play_mod

        terrain = SimpleNamespace(
            mesh_type="plane",
            curriculum=True,
            selected=False,
            terrain_kwargs=None,
            num_rows=4,
            num_cols=6,
            border_size=20.0,
            max_init_terrain_level=1,
            fixed_terrain_level=None,
            terrain_length=6.0,
            terrain_width=6.0,
            platform_size=2.0,
            horizontal_scale=0.1,
            simplify_mesh=True,
            mode=None,
            taxonomy_showcase=False,
            ued_training_grid=False,
            terrain_proportions=[0.2, 0.1, 0.25, 0.25, 0.2],
            terrain_curriculum_difficulty={},
        )
        env_cfg = SimpleNamespace(terrain=terrain)
        play_mod.configure_play_terrain(env_cfg, "v6_full")
        self.assertEqual(env_cfg.terrain.num_rows, V6_TRAINING_NUM_ROWS)
        self.assertEqual(env_cfg.terrain.num_cols, V6_TRAINING_NUM_COLS)
        self.assertEqual(
            tuple(env_cfg.terrain.v6_showcase_levels),
            tuple(range(V6_TRAINING_NUM_ROWS)),
        )

    def test_cli_accepts_v6(self):
        from legged_gym.utils.helpers import get_args

        old = sys.argv
        try:
            sys.argv = ["play.py", "--task", "go2", "--terrain", "v6", "--headless"]
            try:
                args = get_args()
            except SystemExit as e:
                self.fail(f"get_args rejected v6: {e}")
            self.assertEqual(args.terrain, "v6")
            self.assertEqual(args.v6_tile_size, "play")
        finally:
            sys.argv = old

    def test_cli_v6_tile_size_train(self):
        from legged_gym.utils.helpers import get_args

        old = sys.argv
        try:
            sys.argv = [
                "play.py",
                "--task",
                "go2",
                "--terrain",
                "v6",
                "--v6_tile_size",
                "train",
                "--headless",
            ]
            try:
                args = get_args()
            except SystemExit as e:
                self.fail(f"get_args rejected v6_tile_size: {e}")
            self.assertEqual(args.v6_tile_size, "train")
        finally:
            sys.argv = old


if __name__ == "__main__":
    unittest.main()
