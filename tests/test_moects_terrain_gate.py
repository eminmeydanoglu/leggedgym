"""Tests for terrain-conditioned MoE gating telemetry (go2_moects).

Group 1 (pure math, rsl_rl/utils/moe_terrain_gate.py):
compute_terrain_gate_stats() on synthetic (weights, terrain_ids) -- no
simulator, no GPU. Checks per-terrain means, empty-bucket skipping, and the
terrain_specialization summary scalar's two extremes (terrain-independent
gating ~ 0, perfectly terrain-routed gating clearly > 0).
Group 2 (runner wiring, rsl_rl/runners/moe_cts_runner.py):
MoECTSRunner._log_terrain_gate_stats -- interval gating, the degrade-to-noop
paths (missing terrain ids / student idxs / obs_history / encoder), and the
TensorBoard tag names on a bare runner instance (no simulator, no policy
forward -- a stub encoder stands in for StudentMoEEncoder).

CPU-only. Run:
  .venv/bin/python -m pytest tests/test_moects_terrain_gate.py -q
"""

import os
os.environ.setdefault("SIMULATOR", "genesis")

import unittest
from types import SimpleNamespace

import torch

from rsl_rl.utils.moe_terrain_gate import TERRAIN_NAMES, compute_terrain_gate_stats


class TestComputeTerrainGateStats(unittest.TestCase):

    def test_per_terrain_means_are_correct(self):
        # 2 terrains (id 0, id 2), 3 experts. Hand-computable means.
        weights = torch.tensor([
            [1.0, 0.0, 0.0],   # terrain 0
            [0.5, 0.5, 0.0],   # terrain 0
            [0.0, 0.0, 1.0],   # terrain 2
        ])
        terrain_ids = torch.tensor([0, 0, 2])
        names = ("a", "b", "c")
        stats = compute_terrain_gate_stats(weights, terrain_ids, names)
        self.assertIsNotNone(stats)
        self.assertEqual(set(stats["per_terrain"].keys()), {"a", "c"})
        torch.testing.assert_close(
            stats["per_terrain"]["a"]["mean_gate"], torch.tensor([0.75, 0.25, 0.0]))
        torch.testing.assert_close(
            stats["per_terrain"]["c"]["mean_gate"], torch.tensor([0.0, 0.0, 1.0]))
        self.assertEqual(stats["per_terrain"]["a"]["count"], 2)
        self.assertEqual(stats["per_terrain"]["c"]["count"], 1)
        torch.testing.assert_close(
            stats["global_mean_gate"], weights.mean(dim=0))
        # terrain "a" row [1,0,0] has entropy 0, row [.5,.5,0] has entropy
        # log(2); mean = log(2)/2.
        import math
        self.assertAlmostEqual(
            stats["per_terrain"]["a"]["entropy"], math.log(2) / 2, places=6)
        # terrain "c" single row is a delta -> entropy 0, max weight 1.0.
        self.assertAlmostEqual(stats["per_terrain"]["c"]["entropy"], 0.0, places=6)
        self.assertAlmostEqual(stats["per_terrain"]["c"]["max_weight"], 1.0, places=6)

    def test_empty_buckets_are_skipped(self):
        # 9-way terrain space (real TERRAIN_NAMES), only ids 0 and 8 populated.
        weights = torch.softmax(torch.randn(20, 8), dim=-1)
        terrain_ids = torch.cat([torch.zeros(10, dtype=torch.long),
                                  torch.full((10,), 8, dtype=torch.long)])
        stats = compute_terrain_gate_stats(weights, terrain_ids, TERRAIN_NAMES)
        self.assertEqual(set(stats["per_terrain"].keys()), {"wave", "flat"})
        for name in ("slope", "rough_slope", "stairs_up", "stairs_down",
                     "obstacles", "stepping_stones", "gap"):
            self.assertNotIn(name, stats["per_terrain"])

    def test_out_of_range_ids_are_dropped_not_raised(self):
        weights = torch.softmax(torch.randn(4, 3), dim=-1)
        terrain_ids = torch.tensor([0, 1, 99, -1])
        stats = compute_terrain_gate_stats(weights, terrain_ids, ("a", "b", "c"))
        self.assertIsNotNone(stats)
        self.assertEqual(set(stats["per_terrain"].keys()), {"a", "b"})

    def test_empty_input_returns_none(self):
        self.assertIsNone(compute_terrain_gate_stats(
            torch.zeros(0, 8), torch.zeros(0, dtype=torch.long), TERRAIN_NAMES))
        self.assertIsNone(compute_terrain_gate_stats(None, None, TERRAIN_NAMES))

    def test_specialization_near_zero_for_terrain_independent_gating(self):
        torch.manual_seed(0)
        n = 4000
        num_experts = 8
        num_terrains = 9
        # Gating weights drawn independently of terrain assignment.
        logits = torch.randn(n, num_experts)
        weights = torch.softmax(logits, dim=-1)
        terrain_ids = torch.randint(0, num_terrains, (n,))
        stats = compute_terrain_gate_stats(weights, terrain_ids, TERRAIN_NAMES)
        self.assertLess(stats["specialization"], 0.01)

    def test_specialization_clearly_positive_for_terrain_routed_gating(self):
        n = 900  # 9 terrains x 100 samples
        num_experts = 8
        terrain_ids = torch.arange(9).repeat_interleave(100)
        # Each terrain routes hard (one-hot) to expert (terrain_id % 8), so
        # two terrains share expert 0 and the rest are 1:1: distinct enough
        # per-terrain profiles for a strong divergence from the global mean.
        weights = torch.zeros(n, num_experts)
        weights[torch.arange(n), terrain_ids % num_experts] = 1.0
        stats = compute_terrain_gate_stats(weights, terrain_ids, TERRAIN_NAMES)
        # Clearly separated from the ~0.01 near-zero baseline of the
        # terrain-independent case above (two orders of magnitude larger).
        self.assertGreater(stats["specialization"], 0.4)

    def test_specialization_zero_with_single_populated_terrain(self):
        weights = torch.softmax(torch.randn(50, 8), dim=-1)
        terrain_ids = torch.zeros(50, dtype=torch.long)
        stats = compute_terrain_gate_stats(weights, terrain_ids, TERRAIN_NAMES)
        self.assertAlmostEqual(stats["specialization"], 0.0, places=6)


class _StubEncoder:
    """Deterministic stand-in for StudentMoEEncoder.forward_with_weights.

    Routes to expert (int(history[:, 0]) % num_experts) with an otherwise
    uniform-ish softmax profile, so a terrain-correlated first history
    column produces terrain-correlated routing without a real MoE module.
    """

    def __init__(self, num_experts=4):
        self.num_experts = num_experts
        self.calls = []

    def forward_with_weights(self, history):
        self.calls.append(history)
        idx = history[:, 0].long().clamp(0, self.num_experts - 1)
        logits = torch.full((history.shape[0], self.num_experts), -2.0)
        logits[torch.arange(history.shape[0]), idx] = 5.0
        weights = torch.softmax(logits, dim=-1)
        return torch.zeros(history.shape[0], 3), weights


class TestRunnerTerrainGateLogging(unittest.TestCase):
    """MoECTSRunner._log_terrain_gate_stats: gating, no-op paths, tag names."""

    def _make_runner(self, interval=1, num_envs=6, num_student=4, num_experts=4):
        from rsl_rl.runners.moe_cts_runner import MoECTSRunner
        written = {}

        def add_scalar(tag, val, it):
            written[tag] = val

        writer = SimpleNamespace(add_scalar=add_scalar)
        runner = MoECTSRunner.__new__(MoECTSRunner)
        runner.writer = writer
        runner.cfg = {"terrain_gate_log_interval": interval}

        encoder = _StubEncoder(num_experts=num_experts)
        runner.alg = SimpleNamespace(
            actor_critic=SimpleNamespace(history_encoder=encoder),
            student_env_idxs=torch.arange(num_envs - num_student, num_envs),
        )
        # terrain ids: teacher envs (not in student_env_idxs) get id 8
        # (flat), student envs get ids 0..num_student-1 (mod TERRAIN_NAMES).
        terrain_ids = torch.full((num_envs,), 8, dtype=torch.long)
        terrain_ids[num_envs - num_student:] = torch.arange(num_student) % len(TERRAIN_NAMES)
        runner.env = SimpleNamespace(wty_terrain_ids=terrain_ids)

        history = torch.zeros(num_envs, 5)
        # first column encodes which expert this env's history should route
        # to; student envs route to expert == (row index within student block)
        history[num_envs - num_student:, 0] = torch.arange(num_student).float()
        locs = {"it": 3, "completed_iteration": 30, "obs_history": history}
        return runner, written, locs, encoder

    def test_disabled_when_interval_zero(self):
        runner, written, locs, encoder = self._make_runner(interval=0)
        runner._log_terrain_gate_stats(locs)
        self.assertEqual(written, {})
        self.assertEqual(encoder.calls, [])

    def test_skips_iterations_off_interval(self):
        runner, written, locs, encoder = self._make_runner(interval=10)
        locs["completed_iteration"] = 7
        runner._log_terrain_gate_stats(locs)
        self.assertEqual(written, {})

    def test_fires_on_interval_boundary_and_writes_expected_tags(self):
        runner, written, locs, encoder = self._make_runner(
            interval=10, num_envs=6, num_student=4, num_experts=4)
        locs["completed_iteration"] = 10
        runner._log_terrain_gate_stats(locs)
        self.assertEqual(len(encoder.calls), 1)
        # student envs got terrain ids wave(0), slope(1), rough_slope(2),
        # stairs_up(3) and route (by _StubEncoder) to experts 0,1,2,3 resp.
        for i, name in enumerate(("wave", "slope", "rough_slope", "stairs_up")):
            for e in range(4):
                tag = f"MoE/terrain_gate/{name}/expert_{e}"
                self.assertIn(tag, written)
            self.assertIn(f"MoE/terrain_entropy/{name}", written)
            self.assertIn(f"MoE/terrain_max_weight/{name}", written)
            # near-one-hot routing to expert i -> that expert's weight near 1
            self.assertGreater(written[f"MoE/terrain_gate/{name}/expert_{i}"], 0.9)
        self.assertIn("MoE/terrain_specialization", written)
        # flat (teacher-only terrain id) never appears: no student env has it.
        self.assertNotIn("MoE/terrain_entropy/flat", written)

    def test_noop_when_terrain_ids_missing(self):
        runner, written, locs, encoder = self._make_runner(interval=1)
        runner.env = SimpleNamespace()  # no wty_terrain_ids attribute
        runner._log_terrain_gate_stats(locs)
        self.assertEqual(written, {})
        self.assertEqual(encoder.calls, [])

    def test_noop_when_student_idxs_missing(self):
        runner, written, locs, encoder = self._make_runner(interval=1)
        runner.alg.student_env_idxs = None
        runner._log_terrain_gate_stats(locs)
        self.assertEqual(written, {})

    def test_noop_when_obs_history_missing(self):
        runner, written, locs, encoder = self._make_runner(interval=1)
        del locs["obs_history"]
        runner._log_terrain_gate_stats(locs)
        self.assertEqual(written, {})

    def test_noop_when_history_encoder_missing(self):
        runner, written, locs, encoder = self._make_runner(interval=1)
        runner.alg.actor_critic = SimpleNamespace()  # no history_encoder
        runner._log_terrain_gate_stats(locs)
        self.assertEqual(written, {})

    def test_never_raises_on_unexpected_error(self):
        runner, written, locs, encoder = self._make_runner(interval=1)

        def boom(history):
            raise RuntimeError("simulated wiring mismatch")

        runner.alg.actor_critic.history_encoder.forward_with_weights = boom
        try:
            runner._log_terrain_gate_stats(locs)
        except Exception as exc:  # pragma: no cover - failure path under test
            self.fail(f"_log_terrain_gate_stats raised: {exc!r}")
        self.assertEqual(written, {})


if __name__ == "__main__":
    unittest.main()
