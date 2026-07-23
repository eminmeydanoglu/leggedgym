"""Unit tests for the LP-ACRL taxonomy showcase terrain builder and play config.

Drives the shipped Terrain builder and configure_play_terrain path (no reimplementation).
"""
from __future__ import annotations

import argparse
import os
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

# Repo root on path so `legged_gym` imports resolve without install.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from legged_gym.utils.terrain import (
    TAXONOMY_NUM_LEVELS,
    TAXONOMY_NUM_TYPES,
    TAXONOMY_ROUGH_AMPLITUDES,
    TAXONOMY_SLOPE_GRADIENTS,
    TAXONOMY_STEP_HEIGHTS,
    TAXONOMY_STEP_WIDTH,
    TAXONOMY_TYPE_NAMES,
    Terrain,
    build_taxonomy_label_map,
    format_taxonomy_console_map,
    is_taxonomy_terrain_cfg,
    taxonomy_tile_label,
)


def _minimal_taxonomy_cfg(**overrides):
    """Minimal cfg object sufficient for Terrain(taxonomy) construction."""
    cfg = SimpleNamespace(
        mesh_type="heightfield",
        simplify_mesh=False,
        terrain_length=6.0,
        terrain_width=6.0,
        platform_size=2.0,
        terrain_proportions=[0.2, 0.1, 0.25, 0.25, 0.2],
        num_rows=TAXONOMY_NUM_LEVELS,
        num_cols=TAXONOMY_NUM_TYPES,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=2.0,
        curriculum=False,
        selected=False,
        mode="taxonomy",
        taxonomy_showcase=True,
        taxonomy_seed=0,
        height_field_raw_override=None,
        terrain_kwargs=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _tile_slice(terrain: Terrain, row: int, col: int):
    """Return the heightfield block for one (row, col) sub-terrain."""
    start_x = terrain.border + row * terrain.length_per_env_pixels
    end_x = terrain.border + (row + 1) * terrain.length_per_env_pixels
    start_y = terrain.border + col * terrain.width_per_env_pixels
    end_y = terrain.border + (col + 1) * terrain.width_per_env_pixels
    return terrain.height_field_raw[start_x:end_x, start_y:end_y]


def _stair_step_height_m(tile: np.ndarray, vertical_scale: float) -> float:
    """Estimate discrete step height from unique quantized heights (meters)."""
    unique = np.unique(tile.astype(np.int32))
    if unique.size < 2:
        return 0.0
    diffs = np.diff(np.sort(unique))
    # Stairs use a constant step in quantized units; take the modal positive diff.
    pos = diffs[diffs > 0]
    if pos.size == 0:
        return 0.0
    # Mode of positive diffs
    vals, counts = np.unique(pos, return_counts=True)
    step_q = int(vals[np.argmax(counts)])
    return step_q * vertical_scale


class TestTaxonomyGeometry(unittest.TestCase):
    def test_builder_grid_and_origins(self):
        cfg = _minimal_taxonomy_cfg()
        terrain = Terrain(cfg)
        self.assertEqual(terrain.env_origins.shape, (4, 6, 3))
        self.assertEqual(len(terrain.taxonomy_labels), 24)
        # Origins should land at tile centers: (i+0.5)*length, (j+0.5)*width
        for i in range(4):
            for j in range(6):
                ox, oy, oz = terrain.env_origins[i, j]
                self.assertAlmostEqual(ox, (i + 0.5) * 6.0, places=5)
                self.assertAlmostEqual(oy, (j + 0.5) * 6.0, places=5)
                self.assertGreaterEqual(oz, -1.0)  # can be negative for descending

    def test_l3_ascending_stairs_step_height(self):
        cfg = _minimal_taxonomy_cfg()
        terrain = Terrain(cfg)
        # Col 0 = ascending stairs, row 3 = L3 → 0.20 m
        tile = _tile_slice(terrain, row=3, col=0)
        step_m = _stair_step_height_m(tile, cfg.vertical_scale)
        expected = TAXONOMY_STEP_HEIGHTS[3]
        # Quantization-aligned tolerance (vertical_scale = 0.005).
        self.assertAlmostEqual(step_m, expected, delta=cfg.vertical_scale + 1e-9)
        # Positive heights for ascending stairs
        self.assertGreater(tile.max(), 0)

    def test_l3_descending_stairs_negative(self):
        cfg = _minimal_taxonomy_cfg()
        terrain = Terrain(cfg)
        tile = _tile_slice(terrain, row=3, col=1)
        step_m = _stair_step_height_m(tile, cfg.vertical_scale)
        self.assertAlmostEqual(step_m, TAXONOMY_STEP_HEIGHTS[3], delta=cfg.vertical_scale + 1e-9)
        self.assertLess(tile.min(), 0)

    def test_slope_amplitude_ordering(self):
        cfg = _minimal_taxonomy_cfg()
        terrain = Terrain(cfg)
        # Col 2 = upslope; peak height should increase L0→L3 (L0 gradient=0 → flat).
        peaks = []
        for level in range(4):
            tile = _tile_slice(terrain, row=level, col=2)
            peaks.append(float(tile.max()) * cfg.vertical_scale)
        self.assertAlmostEqual(peaks[0], 0.0, delta=cfg.vertical_scale)
        self.assertLess(peaks[1], peaks[2])
        self.assertLess(peaks[2], peaks[3])
        # L3 should be substantially taller than L1
        self.assertGreater(peaks[3], peaks[1] * 1.5)

    def test_rough_amplitude_ordering(self):
        cfg = _minimal_taxonomy_cfg(taxonomy_seed=0)
        terrain = Terrain(cfg)
        amps = []
        for level in range(4):
            tile = _tile_slice(terrain, row=level, col=4)
            amp = float(np.max(np.abs(tile))) * cfg.vertical_scale
            amps.append(amp)
        # Each level's max |h| should not exceed table amplitude (+ a little
        # interpolation overshoot is rare; stay within 1.5× for safety).
        for level, amp in enumerate(amps):
            self.assertLessEqual(amp, TAXONOMY_ROUGH_AMPLITUDES[level] * 1.5 + cfg.vertical_scale)
        # L3 should be rougher than L0
        self.assertGreater(amps[3], amps[0])

    def test_flat_column_near_zero(self):
        cfg = _minimal_taxonomy_cfg()
        terrain = Terrain(cfg)
        for level in range(4):
            tile = _tile_slice(terrain, row=level, col=5)
            self.assertEqual(int(tile.max()), 0)
            self.assertEqual(int(tile.min()), 0)

    def test_labels_cover_all_24(self):
        cfg = _minimal_taxonomy_cfg()
        terrain = Terrain(cfg)
        labels = terrain.taxonomy_labels
        self.assertEqual(len(labels), 24)
        names = {e["label"] for e in labels}
        for level in range(4):
            for t in range(6):
                self.assertIn(taxonomy_tile_label(level, t), names)
        # Positions ~1.5 m above origin z
        for e in labels:
            i, j = e["row"], e["col"]
            ox, oy, oz = terrain.env_origins[i, j]
            pos = e["position"]
            self.assertAlmostEqual(pos[0], ox, places=5)
            self.assertAlmostEqual(pos[1], oy, places=5)
            self.assertAlmostEqual(pos[2], oz + 1.5, places=5)


class TestTaxonomyPlayConfig(unittest.TestCase):
    def _make_env_cfg(self):
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
            horizontal_scale=0.1,
            mode=None,
            taxonomy_showcase=False,
        )
        env = SimpleNamespace(num_envs=4096, debug=False, auto_reset=True)
        viewer = SimpleNamespace(pos=[2.0, 2.0, 1.0], lookat=[0.0, 0.0, 0.0], rendered_envs_idx=[])
        domain_rand = SimpleNamespace(push_robots=True)
        commands = SimpleNamespace(
            zero_cmd_prob=0.4,
            resampling_time=10.0,
            heading_command=True,
            ranges=SimpleNamespace(
                lin_vel_x=[-1, 1],
                lin_vel_y=[-1, 1],
                ang_vel_yaw=[-1, 1],
                heading=[-3, 3],
            ),
        )
        return SimpleNamespace(
            terrain=terrain,
            env=env,
            viewer=viewer,
            domain_rand=domain_rand,
            commands=commands,
        )

    def test_configure_play_terrain_taxonomy(self):
        # Import play after path setup; patch SIMULATOR if needed.
        from legged_gym.scripts import play as play_mod

        env_cfg = self._make_env_cfg()
        play_mod.configure_play_terrain(env_cfg, "taxonomy")
        self.assertFalse(env_cfg.terrain.curriculum)
        self.assertFalse(env_cfg.terrain.selected)
        self.assertEqual(env_cfg.terrain.num_rows, 4)
        self.assertEqual(env_cfg.terrain.num_cols, 6)
        self.assertEqual(env_cfg.terrain.mode, "taxonomy")
        self.assertTrue(env_cfg.terrain.taxonomy_showcase)
        self.assertEqual(env_cfg.terrain.border_size, 2.0)
        self.assertIn(env_cfg.terrain.mesh_type, ("heightfield", "trimesh"))

    def test_cli_accepts_taxonomy(self):
        from legged_gym.utils.helpers import get_args
        # get_args parses sys.argv; inject taxonomy choice
        old = sys.argv
        try:
            sys.argv = ["play.py", "--task", "go2", "--terrain", "taxonomy", "--headless"]
            # get_args may call parse_known_args or full parse; if it needs extra,
            # wrap.
            try:
                args = get_args()
            except SystemExit as e:
                self.fail(f"get_args rejected taxonomy: {e}")
            self.assertEqual(args.terrain, "taxonomy")
        finally:
            sys.argv = old

    def test_is_taxonomy_cfg(self):
        self.assertTrue(is_taxonomy_terrain_cfg(SimpleNamespace(mode="taxonomy")))
        self.assertTrue(is_taxonomy_terrain_cfg(SimpleNamespace(mode="showcase")))
        self.assertTrue(is_taxonomy_terrain_cfg(SimpleNamespace(taxonomy_showcase=True, mode=None)))
        self.assertFalse(is_taxonomy_terrain_cfg(SimpleNamespace(mode=None, taxonomy_showcase=False)))

    def test_console_map_mentions_all_types(self):
        origins = np.zeros((4, 6, 3), dtype=np.float64)
        for i in range(4):
            for j in range(6):
                origins[i, j] = [(i + 0.5) * 6, (j + 0.5) * 6, 0.0]
        label_map = build_taxonomy_label_map(origins)
        text = format_taxonomy_console_map(label_map)
        for name in TAXONOMY_TYPE_NAMES:
            self.assertIn(name, text)
        self.assertIn("L3", text)
        self.assertIn("L0", text)


class TestGenesisTextApiProbe(unittest.TestCase):
    def test_probe_or_skip(self):
        """Record whether Scene exposes a text/billboard draw API."""
        try:
            from genesis.engine.scene import Scene
        except Exception as e:
            # Environment without genesis still documents the probe result via skip.
            self.skipTest(f"genesis not importable: {e}")
        draw_debug = [m for m in dir(Scene) if m.startswith("draw_debug")]
        textish = [
            m for m in dir(Scene)
            if any(k in m.lower() for k in ("text", "billboard", "label"))
        ]
        # Document expectation used by genesis_simulator fallback wiring.
        self.assertTrue(any(m.startswith("draw_debug") for m in draw_debug))
        self.assertIn("draw_debug_spheres", draw_debug)
        # Current genesis 1.0.0 has no text draw — absence means console+sphere fallback.
        if "draw_debug_text" in draw_debug:
            self.assertTrue(callable(getattr(Scene, "draw_debug_text")))
        else:
            self.assertNotIn("draw_debug_text", draw_debug)
        # Simulator source wires the fallback call site.
        sim_path = os.path.join(
            _ROOT, "legged_gym", "simulator", "genesis_simulator.py"
        )
        with open(sim_path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("_draw_taxonomy_labels", src)
        self.assertIn("draw_debug_spheres", src)
        # Keep probe result accessible for the verification log.
        self._probe_summary = {
            "draw_debug": sorted(draw_debug),
            "textish": sorted(textish),
            "has_draw_debug_text": "draw_debug_text" in draw_debug,
        }
        print("GENESIS_TEXT_API_PROBE:", self._probe_summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
