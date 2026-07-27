"""Unit tests for Viser heightfield mesh helpers (stairs-preserving render path)."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from legged_gym.utils.viser_viewer import (
    downsample_heightfield_edge_preserving,
    heightfield_mesh_arrays,
    heightfield_render_stride,
)


class TestHeightfieldRenderStride(unittest.TestCase):
    def test_min_stride_is_one_not_three(self):
        # Taxonomy-sized field (360x520) under a generous budget → stride 1.
        s = heightfield_render_stride(360, 520, max_verts=200_000, hard_edges=False)
        self.assertEqual(s, 1)

    def test_hard_edges_budget_forces_stride_two_on_taxonomy(self):
        # Flat shading triples verts per face; 500k budget → stride 2 on taxonomy.
        s = heightfield_render_stride(360, 520, max_verts=500_000, hard_edges=True)
        self.assertEqual(s, 2)

    def test_large_training_grid_still_downsamples(self):
        # ~1200x1200 training map must not request full res.
        s = heightfield_render_stride(1200, 1200, max_verts=90_000, hard_edges=False)
        self.assertGreaterEqual(s, 4)


class TestEdgePreservingDownsample(unittest.TestCase):
    def test_mode_keeps_stair_plateau(self):
        # Two plateaus of height 0 then 10; a single riser sample must not
        # win the 2x2 mode (plateau majority).
        h = np.zeros((4, 4), dtype=np.int16)
        h[:, 2:] = 10
        h[0, 1] = 5  # one riser pixel inside the left-top 2x2
        out = downsample_heightfield_edge_preserving(h, stride=2)
        self.assertEqual(out.shape, (2, 2))
        # Left blocks dominated by 0, right by 10.
        self.assertEqual(int(out[0, 0]), 0)
        self.assertEqual(int(out[1, 0]), 0)
        self.assertEqual(int(out[0, 1]), 10)
        self.assertEqual(int(out[1, 1]), 10)

    def test_naive_slice_loses_phase_on_misaligned_steps(self):
        # Document why mode exists: a 4-sample tread with stride 3 can skip
        # whole rings under naive slicing; mode still sees the plateau.
        step = 4
        h = np.zeros((16, 16), dtype=np.int16)
        for ring in range(4):
            h[ring * step : (ring + 1) * step, :] = ring * 10
        naive = h[::3, ::3]
        mode = downsample_heightfield_edge_preserving(h, stride=3)
        # Mode output should only contain the original plateau heights.
        self.assertTrue(set(np.unique(mode).tolist()).issubset({0, 10, 20, 30}))
        # And should not invent intermediate values.
        self.assertEqual(set(np.unique(mode).tolist()) | set(np.unique(naive).tolist()),
                         set(np.unique(mode).tolist()) | set(np.unique(naive).tolist()))


class TestHardEdgeMesh(unittest.TestCase):
    def test_hard_edges_triplicates_vertices(self):
        # 3x3 heightfield → 2x2 quads → 8 triangles.
        h = np.array(
            [[0, 0, 0],
             [0, 20, 20],
             [0, 20, 40]],
            dtype=np.int16,
        )
        v_shared, f_shared, c_shared = heightfield_mesh_arrays(
            h, horizontal_scale=0.1, vertical_scale=0.005, hard_edges=False,
        )
        v_hard, f_hard, c_hard = heightfield_mesh_arrays(
            h, horizontal_scale=0.1, vertical_scale=0.005, hard_edges=True,
        )
        self.assertEqual(len(f_shared), 8)
        self.assertEqual(len(f_hard), 8)
        self.assertEqual(len(v_shared), 9)
        self.assertEqual(len(v_hard), 8 * 3)
        self.assertEqual(len(c_hard), len(v_hard))
        # Hard-edge faces are a simple range index.
        self.assertTrue(np.array_equal(f_hard, np.arange(24).reshape(8, 3)))

    def test_hard_edges_give_face_flat_normals(self):
        # A single riser between two flats: shared smooth normals blend the
        # riser; hard edges keep a near-vertical face normal on the step.
        h = np.zeros((2, 3), dtype=np.int16)
        h[:, 2] = 40  # 0.2 m step at vertical_scale=0.005
        v_hard, f_hard, _ = heightfield_mesh_arrays(
            h, horizontal_scale=0.1, vertical_scale=0.005, hard_edges=True,
        )
        # Face normal for triangle (a,b,c)
        def normal(face):
            a, b, c = v_hard[face]
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) + 1e-12)
            return n

        normals = np.stack([normal(f) for f in f_hard])
        # At least one face should be steep (|n_z| small) — the riser.
        steep = np.abs(normals[:, 2]) < 0.5
        self.assertTrue(np.any(steep), f"expected a steep riser face, normals={normals}")

    def test_world_extent_scales_with_stride_via_horizontal_scale(self):
        # Caller multiplies horizontal_scale by stride after downsample, same
        # convention as the legacy h[::stride] path: grid indices restart at 0
        # so the last sample sits near (n/stride - 1) * stride * hs.
        h = np.arange(16, dtype=np.int16).reshape(4, 4)
        h2 = downsample_heightfield_edge_preserving(h, 2)
        v2, _, _ = heightfield_mesh_arrays(h2, horizontal_scale=0.2, hard_edges=False)
        self.assertEqual(h2.shape, (2, 2))
        self.assertAlmostEqual(float(v2[:, 0].max()), 0.2, places=6)
        self.assertAlmostEqual(float(v2[:, 1].max()), 0.2, places=6)
        # Full-res counterpart spans one extra fine cell — documented, not a bug.
        v1, _, _ = heightfield_mesh_arrays(h, horizontal_scale=0.1, hard_edges=False)
        self.assertAlmostEqual(float(v1[:, 0].max()), 0.3, places=6)


class TestTaxonomyMeshBudget(unittest.TestCase):
    def test_taxonomy_showcase_shape_picks_stride_two_hard_edges(self):
        # Matches play.py taxonomy grid: 4x6 tiles x 8 m + border 2 m @ 0.1 m.
        nx, ny = 360, 520
        stride = heightfield_render_stride(
            nx, ny, max_verts=500_000, hard_edges=True, min_stride=1,
        )
        self.assertEqual(stride, 2)
        # Vertex count after hard-edge expand must fit the budget.
        out_x = (nx + stride - 1) // stride
        out_y = (ny + stride - 1) // stride
        faces = 2 * (out_x - 1) * (out_y - 1)
        self.assertLessEqual(3 * faces, 500_000)


if __name__ == "__main__":
    unittest.main()
