"""Pure-math unit tests for the MoE gate diagnostic probe
(legged_gym/scripts/eval/probe_moe_gate.py, go2_moects task).

The probe module is loaded by FILE PATH via importlib so the legged_gym /
genesis import chain never runs: CPU-only, no simulator. All checks use
synthetic tensors with known answers (fixed seeds, explicit tolerances).

Covered (module top-level functions, shared API contract):
  effective_weights, row_entropy, effective_experts, pairwise_cosine_mean,
  linear_cka, responsibilities, normalized_mi, chi2_stat, ablation_weights,
  mix_latent -- plus the "top-level imports are stdlib + numpy + torch only"
  packaging contract (AST scan).

If the probe module does not exist yet, every test class skips cleanly.

Run:
  .venv/bin/python -m unittest tests/test_moe_gate_probe.py -v
(or:  .venv/bin/python -m pytest tests/test_moe_gate_probe.py -q)
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import ast
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "legged_gym" / "scripts" / "eval" / "probe_moe_gate.py"

N, K, D = 16, 8, 32     # samples, experts, latent dim for the shared fixtures


def _load_probe():
    """Load the probe module by file path (no legged_gym package import)."""
    if not PROBE_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("probe_moe_gate", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROBE = _load_probe()
_NEEDS_PROBE = unittest.skipUnless(
    PROBE is not None, f"probe module not found at {PROBE_PATH}")


def _one_hot(idx, num_classes):
    return Fn.one_hot(idx, num_classes).to(torch.float32)


@unittest.skipUnless(PROBE_PATH.exists(), f"probe module not found at {PROBE_PATH}")
class TestModuleImportContract(unittest.TestCase):
    """Top-level imports must be stdlib + numpy + torch only, so the module
    loads CPU-only without the legged_gym/genesis chain. Only direct module
    -body statements are scanned: imports inside functions, ``if
    TYPE_CHECKING:`` blocks or guarded try/except are allowed by design."""

    def test_top_level_imports_are_stdlib_numpy_torch(self):
        tree = ast.parse(PROBE_PATH.read_text())
        allowed = set(sys.stdlib_module_names) | {"numpy", "torch"}
        bad = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in allowed:
                        bad.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0:
                    bad.append("." * node.level + (node.module or ""))
                elif (node.module or "").split(".")[0] not in allowed:
                    bad.append(node.module)
        self.assertEqual(bad, [])


@_NEEDS_PROBE
class TestEffectiveWeights(unittest.TestCase):
    """effective_weights(g, E): w = g*||E||, row-normalized."""

    def test_rows_sum_to_one(self):
        torch.manual_seed(0)
        g = torch.softmax(torch.randn(N, K), dim=1)
        E = torch.randn(N, K, D)
        w = PROBE.effective_weights(g, E)
        self.assertEqual(tuple(w.shape), (N, K))
        self.assertTrue(torch.allclose(w.sum(dim=1), torch.ones(N), atol=1e-6))

    def test_known_values(self):
        # norms [5, 1], g [0.5, 0.5] -> unnorm [2.5, 0.5] -> w [5/6, 1/6]
        g = torch.tensor([[0.5, 0.5]])
        E = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]])
        w = PROBE.effective_weights(g, E)
        self.assertTrue(torch.allclose(w, torch.tensor([[5.0 / 6.0, 1.0 / 6.0]]),
                                       atol=1e-6))

    def test_single_nonzero_expert_gives_one_hot(self):
        torch.manual_seed(0)
        g = torch.full((N, K), 1.0 / K)
        E = torch.zeros(N, K, D)
        E[:, 0, :] = torch.randn(N, D)
        w = PROBE.effective_weights(g, E)
        self.assertTrue(torch.allclose(w[:, 0], torch.ones(N), atol=1e-6))
        self.assertTrue(torch.allclose(w[:, 1:], torch.zeros(N, K - 1), atol=1e-6))

    def test_equal_expert_norms_recover_gate(self):
        # every expert norm exactly 1 (1/32 entries are exact in float32)
        torch.manual_seed(0)
        g = torch.softmax(torch.randn(N, K), dim=1)
        E = torch.full((N, K, D), 1.0 / math.sqrt(D))
        w = PROBE.effective_weights(g, E)
        self.assertTrue(torch.allclose(w, g, atol=1e-6))

    def test_zero_denominator_guard_stays_finite(self):
        torch.manual_seed(0)
        g = torch.softmax(torch.randn(N, K), dim=1)
        w = PROBE.effective_weights(g, torch.zeros(N, K, D))
        self.assertTrue(torch.isfinite(w).all().item())


@_NEEDS_PROBE
class TestEntropyHelpers(unittest.TestCase):
    """row_entropy / effective_experts on known distributions."""

    def test_uniform_over_eight(self):
        p = torch.full((4, K), 1.0 / K)
        self.assertTrue(torch.allclose(
            PROBE.row_entropy(p), torch.full((4,), math.log(K)), atol=1e-5))
        self.assertTrue(torch.allclose(
            PROBE.effective_experts(p), torch.full((4,), float(K)), atol=1e-4))

    def test_one_hot(self):
        p = _one_hot(torch.tensor([0, 3, 7]), K)
        self.assertTrue(torch.allclose(
            PROBE.row_entropy(p), torch.zeros(3), atol=1e-5))
        self.assertTrue(torch.allclose(
            PROBE.effective_experts(p), torch.ones(3), atol=1e-5))

    def test_two_point_distribution(self):
        p = torch.zeros(2, K)
        p[:, :2] = 0.5
        self.assertTrue(torch.allclose(
            PROBE.row_entropy(p), torch.full((2,), math.log(2.0)), atol=1e-5))
        self.assertTrue(torch.allclose(
            PROBE.effective_experts(p), torch.full((2,), 2.0), atol=1e-4))


@_NEEDS_PROBE
class TestPairwiseCosineMean(unittest.TestCase):
    """pairwise_cosine_mean(F): (K,K) mean sample cosine between experts."""

    def test_identical_experts_give_all_ones(self):
        torch.manual_seed(0)
        base = torch.randn(D)
        F = base.view(1, 1, D).repeat(4, K, 1)
        C = PROBE.pairwise_cosine_mean(F)
        self.assertEqual(tuple(C.shape), (K, K))
        self.assertTrue(torch.allclose(C, torch.ones(K, K), atol=1e-6))

    def test_crafted_parallel_and_orthogonal(self):
        # experts 0,1 parallel per sample, expert 2 orthogonal to both
        F = torch.zeros(3, 3, 3)
        e1 = torch.tensor([1.0, 0.0, 0.0])
        e2 = torch.tensor([0.0, 1.0, 0.0])
        for n in range(3):
            F[n, 0] = (n + 1) * e1
            F[n, 1] = (n + 2) * e1
            F[n, 2] = (n + 1) * e2
        expected = torch.tensor([[1.0, 1.0, 0.0],
                                 [1.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])
        C = PROBE.pairwise_cosine_mean(F)
        self.assertTrue(torch.allclose(C, expected, atol=1e-6))


@_NEEDS_PROBE
class TestLinearCka(unittest.TestCase):
    """linear_cka(X, Y): centered linear CKA, plain float return."""

    def test_self_similarity_is_one(self):
        torch.manual_seed(0)
        X = torch.randn(256, D)
        val = PROBE.linear_cka(X, X)
        self.assertIsInstance(val, (float, np.floating))
        self.assertAlmostEqual(float(val), 1.0, places=5)

    def test_orthogonal_transform_invariance(self):
        # CKA is exactly invariant under orthogonal (hence invertible) maps.
        # NOTE: a *general* invertible A does NOT preserve CKA (verified:
        # cka(X, X@A) ~ 0.74 for a random well-conditioned A), so the
        # "invertible linear transform" case is pinned with an orthogonal one.
        torch.manual_seed(0)
        X = torch.randn(256, D)
        Q, _ = torch.linalg.qr(torch.randn(D, D))
        val = PROBE.linear_cka(X, X @ Q)
        self.assertAlmostEqual(float(val), 1.0, places=4)

    def test_independent_random_is_near_zero(self):
        # N=1024 seed 0: reference formula gives 0.0345; bound 0.1 is loose.
        torch.manual_seed(0)
        X = torch.randn(1024, D)
        Y = torch.randn(1024, D)
        val = PROBE.linear_cka(X, Y)
        self.assertGreaterEqual(float(val), 0.0)
        self.assertLess(float(val), 0.1)


@_NEEDS_PROBE
class TestResponsibilities(unittest.TestCase):
    """responsibilities(d, tau) = softmax(-d/tau, dim=1)."""

    def test_rows_sum_to_one(self):
        torch.manual_seed(0)
        r = PROBE.responsibilities(torch.randn(N, K), 1.0)
        self.assertEqual(tuple(r.shape), (N, K))
        self.assertTrue(torch.allclose(r.sum(dim=1), torch.ones(N), atol=1e-6))

    def test_huge_tau_is_uniform(self):
        torch.manual_seed(0)
        r = PROBE.responsibilities(torch.randn(N, K), 1e12)
        self.assertTrue(torch.allclose(r, torch.full((N, K), 1.0 / K), atol=1e-6))

    def test_tiny_tau_is_one_hot_on_argmin(self):
        d = torch.tensor([[1.0, 0.5, 2.0, 3.0],
                          [0.0, 1.0, -1.0, 2.0]])
        r = PROBE.responsibilities(d, 1e-12)
        self.assertTrue(torch.equal(r.argmax(dim=1), d.argmin(dim=1)))
        self.assertTrue(torch.allclose(
            r, _one_hot(d.argmin(dim=1), 4), atol=1e-6))


@_NEEDS_PROBE
class TestNormalizedMi(unittest.TestCase):
    """normalized_mi(a, b) = MI(a;b) / min(H(a), H(b)), natural log."""

    def test_identical_labels_give_one(self):
        torch.manual_seed(0)
        a = torch.randint(0, 9, (8192,))
        val = PROBE.normalized_mi(a, a)
        self.assertAlmostEqual(float(val), 1.0, places=5)

    def test_independent_labels_are_near_zero(self):
        # seed 0: reference estimator gives 0.0019; bound 0.05 is loose.
        torch.manual_seed(0)
        a = torch.randint(0, 9, (8192,))
        b = torch.randint(0, 9, (8192,))
        val = PROBE.normalized_mi(a, b)
        self.assertGreaterEqual(float(val), 0.0)
        self.assertLess(float(val), 0.05)

    def test_constant_labels_give_zero(self):
        torch.manual_seed(0)
        b = torch.randint(0, 9, (100,))
        a_const = torch.zeros(100, dtype=torch.long)
        self.assertEqual(float(PROBE.normalized_mi(a_const, b)), 0.0)
        self.assertEqual(float(PROBE.normalized_mi(a_const, a_const)), 0.0)


@_NEEDS_PROBE
class TestChi2Stat(unittest.TestCase):
    """chi2_stat(a, b): contingency chi2 with hand-computed values."""

    def test_balanced_2x2_mixed_table(self):
        # a=[0,0,0,1,1,1], b=[0,0,1,0,1,1] -> table [[2,1],[1,2]], n=6,
        # row/col sums all 3 -> E = 1.5 everywhere ->
        # chi2 = 4 * (0.5^2 / 1.5) = 2/3, dof = (2-1)(2-1) = 1
        a = torch.tensor([0, 0, 0, 1, 1, 1])
        b = torch.tensor([0, 0, 1, 0, 1, 1])
        chi2, dof = PROBE.chi2_stat(a, b)
        self.assertIsInstance(chi2, (float, np.floating))
        self.assertIsInstance(dof, (int, np.integer))
        self.assertAlmostEqual(float(chi2), 2.0 / 3.0, places=6)
        self.assertEqual(int(dof), 1)

    def test_perfectly_dependent_2x2(self):
        # a == b -> table [[3,0],[0,3]], E = 1.5 ->
        # chi2 = 4 * (1.5^2 / 1.5) = 6.0, dof = 1
        a = torch.tensor([0, 0, 0, 1, 1, 1])
        chi2, dof = PROBE.chi2_stat(a, a.clone())
        self.assertAlmostEqual(float(chi2), 6.0, places=6)
        self.assertEqual(int(dof), 1)

    def test_dof_for_3x2_independent_table(self):
        # table all ones (3x2), n=6 -> E = 1 everywhere -> chi2 = 0,
        # dof = (3-1)(2-1) = 2
        a = torch.tensor([0, 0, 1, 1, 2, 2])
        b = torch.tensor([0, 1, 0, 1, 0, 1])
        chi2, dof = PROBE.chi2_stat(a, b)
        self.assertAlmostEqual(float(chi2), 0.0, places=6)
        self.assertEqual(int(dof), 2)


@_NEEDS_PROBE
class TestAblationWeights(unittest.TestCase):
    """ablation_weights(g, d, variant, generator) for all five variants."""

    def setUp(self):
        torch.manual_seed(0)   # pinned: every row argmax/argmin is unique
        self.g = torch.softmax(torch.randn(N, K), dim=1)
        self.d = torch.randn(N, K)

    def test_all_variants_are_row_stochastic(self):
        for variant in ("learned", "uniform", "shuffled", "top1", "oracle"):
            with self.subTest(variant=variant):
                w = PROBE.ablation_weights(self.g, self.d, variant)
                self.assertEqual(tuple(w.shape), (N, K))
                self.assertTrue(
                    torch.allclose(w.sum(dim=1), torch.ones(N), atol=1e-6))

    def test_learned_returns_gate(self):
        w = PROBE.ablation_weights(self.g, self.d, "learned")
        self.assertTrue(torch.allclose(w, self.g, atol=1e-6))

    def test_uniform_is_flat(self):
        w = PROBE.ablation_weights(self.g, self.d, "uniform")
        self.assertTrue(torch.allclose(w, torch.full((N, K), 1.0 / K), atol=1e-6))

    def test_top1_is_one_hot_on_argmax_gate(self):
        w = PROBE.ablation_weights(self.g, self.d, "top1")
        expected = _one_hot(self.g.argmax(dim=1), K)
        self.assertTrue(torch.allclose(w, expected, atol=1e-6))

    def test_oracle_is_one_hot_on_argmin_distance(self):
        w = PROBE.ablation_weights(self.g, self.d, "oracle")
        expected = _one_hot(self.d.argmin(dim=1), K)
        self.assertTrue(torch.allclose(w, expected, atol=1e-6))

    def test_shuffled_is_a_row_permutation_of_gate(self):
        gen = torch.Generator().manual_seed(123)
        w = PROBE.ablation_weights(self.g, self.d, "shuffled", generator=gen)
        # multiset of rows must be preserved (a permutation copies values
        # exactly, so exact tuple comparison is valid)
        row_multiset = lambda t: sorted(tuple(row) for row in t.tolist())
        self.assertEqual(row_multiset(w), row_multiset(self.g))
        # and the shuffle must actually reorder for this seed (anti no-op)
        self.assertFalse(torch.equal(w, self.g))

    def test_shuffled_deterministic_for_fixed_generator_seed(self):
        w1 = PROBE.ablation_weights(self.g, self.d, "shuffled",
                                    generator=torch.Generator().manual_seed(123))
        w2 = PROBE.ablation_weights(self.g, self.d, "shuffled",
                                    generator=torch.Generator().manual_seed(123))
        self.assertTrue(torch.equal(w1, w2))


@_NEEDS_PROBE
class TestMixLatent(unittest.TestCase):
    """mix_latent(E, w) = normalize((w.unsqueeze(-1) * E).sum(1))."""

    def test_output_rows_have_unit_norm(self):
        torch.manual_seed(0)
        E = torch.randn(N, K, D)
        w = torch.softmax(torch.randn(N, K), dim=1)
        out = PROBE.mix_latent(E, w)
        self.assertEqual(tuple(out.shape), (N, D))
        self.assertTrue(torch.allclose(out.norm(dim=-1), torch.ones(N), atol=1e-5))

    def test_one_hot_weight_selects_normalized_expert(self):
        torch.manual_seed(0)
        E = torch.randn(N, K, D)
        expert_idx = 3
        w = _one_hot(torch.full((N,), expert_idx), K)
        out = PROBE.mix_latent(E, w)
        expected = Fn.normalize(E[:, expert_idx, :], dim=-1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
