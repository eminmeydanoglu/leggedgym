"""Unit tests for the stateless (Eq. 7) LP-ACRL / ALP episode curriculum.

Pins the anti-absorbing-state contract of ``_FiniteEpisodeCurriculum.advance``:
the task-sampling distribution is rebuilt every stage as a full-space softmax of
learning progress (unobserved cells imputed LP=0), so probability is never
carried across stages and no cell can be permanently starved.  See the plan at
plans/ne-gerekiyorsa-oku-ve-snug-quail.md and go2_v5_config.py:53 (beta freeze).
"""
import unittest

import numpy as np

from legged_gym.utils.ued import (
    ALPEpisodeCurriculum,
    EpisodeOutcomeBatch,
    LPACRLEpisodeCurriculum,
    TaskSpace,
)

BETA = 5.0  # frozen V5_BETA


def _observe(cur, cells, returns, *, revision, length=100):
    """Feed one same-revision outcome per (cell, return) into ``cur``."""
    cells = np.asarray(cells, dtype=np.int64)
    returns = np.asarray(returns, dtype=np.float64)
    cur.observe(EpisodeOutcomeBatch(
        task_ids=cells,
        assigned_revision=np.full(cells.shape, revision, dtype=np.int64),
        completion_revision=revision,
        episodic_returns=returns,
        episode_lengths=np.full(cells.shape, length, dtype=np.int64),
        terminal_reasons=np.full(cells.shape, "timeout", dtype="U10"),
    ))


class _Driver:
    """Drives a curriculum stage-by-stage over the full task space."""

    def __init__(self, cur):
        self.cur = cur
        self.n = cur._n
        self.step = 0

    def stage(self, returns, *, skip=()):
        """Observe every cell (except ``skip``) with ``returns`` then advance."""
        returns = np.asarray(returns, dtype=np.float64)
        cells = [c for c in range(self.n) if c not in set(skip)]
        _observe(self.cur, cells, returns[cells], revision=self.cur.sampler_revision)
        self.step += self.cur.stage_length_control_steps
        return self.cur.advance(self.step)


class TestStatelessCurriculum(unittest.TestCase):
    def setUp(self):
        self.space = TaskSpace()
        self.n = self.space.size

    def _lpacrl(self, **kw):
        return LPACRLEpisodeCurriculum(self.space, stage_length_control_steps=1, beta=BETA, seed=0, **kw)

    # ------------------------------------------------------------------ core
    def test_cold_start_is_uniform(self):
        cur = self._lpacrl()
        d = _Driver(cur)
        # First advance has no prior-stage observation -> progress_mask empty ->
        # softmax(zeros) == uniform, matching the bootstrap distribution.
        d.stage(np.full(self.n, 3.0))
        p = cur.probabilities()
        self.assertTrue(np.allclose(p, 1.0 / self.n))

    def test_no_absorbing_state_after_adversarial_spike(self):
        cur = self._lpacrl()
        d = _Driver(cur)
        d.stage(np.zeros(self.n))          # establish baseline (uniform)
        spike = np.zeros(self.n)
        spike[0] = 20.0                    # one cell shows LP=+20
        d.stage(spike)
        p = cur.probabilities()
        # Every cell keeps sampleable mass -- the old freeze update drove the
        # cold cells to ~2e-9 (below the 1/8192 coverage floor); beta=5 keeps
        # them at ~7e-3 (see verify_coverage_regime.py).
        self.assertGreater(p.min(), 1e-3)
        self.assertAlmostEqual(p.sum(), 1.0, places=12)
        self.assertLess(p[0], 1.0)         # hot cell concentrated but not absorbing

    def test_starved_cell_recovers(self):
        cur = self._lpacrl()
        d = _Driver(cur)
        d.stage(np.zeros(self.n))
        spike = np.zeros(self.n)
        spike[0] = 20.0
        d.stage(spike)                     # cell 0 dominates
        peak = cur.diagnostics()
        # Cell 0 plateaus (return falls back to 0 -> LP goes negative), so the
        # rebuild broadens again -- automatic recovery the freeze update could
        # never do.
        d.stage(np.zeros(self.n))
        recovered = cur.diagnostics()
        self.assertGreater(recovered["effective_sample_size"], peak["effective_sample_size"])
        self.assertLess(recovered["max_cell_probability"], peak["max_cell_probability"])
        self.assertLess(cur.probabilities()[0], peak["max_cell_probability"])

    def test_full_rebuild_ignores_prior_probability(self):
        # Two curricula with identical LP inputs but different starting
        # probabilities must land on the same post-advance distribution: Eq. 7
        # does not read c_j on the RHS.
        spike = np.zeros(self.n)
        spike[3] = 12.0

        ref = self._lpacrl()
        dref = _Driver(ref)
        dref.stage(np.zeros(self.n))
        dref.stage(spike)
        expected = ref.probabilities()

        other = self._lpacrl()
        dother = _Driver(other)
        dother.stage(np.zeros(self.n))
        # Corrupt the carried distribution right before the second advance.
        rng = np.random.default_rng(7)
        skew = rng.random(self.n)
        other._probabilities = skew / skew.sum()
        dother.stage(spike)
        self.assertTrue(np.allclose(other.probabilities(), expected))

    def test_unobserved_cell_is_neutral_not_frozen(self):
        cur = self._lpacrl()
        d = _Driver(cur)
        d.stage(np.zeros(self.n))
        spike = np.zeros(self.n)
        spike[0] = 20.0
        # Cell 7 is not observed this stage: it must be imputed LP=0 (baseline
        # weight e^0), NOT keep its prior probability.  It should match another
        # cell whose measured LP is exactly 0.
        d.stage(spike, skip=(7,))
        p = cur.probabilities()
        self.assertAlmostEqual(p[7], p[50], places=12)   # cell 50 measured LP=0
        self.assertGreater(p[7], 1e-3)

    # ------------------------------------------------------------------ epsilon
    def test_epsilon_floor_bounds(self):
        eps = 0.05
        cur = self._lpacrl(epsilon=eps)
        d = _Driver(cur)
        d.stage(np.zeros(self.n))
        spike = np.zeros(self.n)
        spike[0] = 40.0                    # extreme spike
        d.stage(spike)
        p = cur.probabilities()
        self.assertGreaterEqual(p.min(), eps / self.n * (1 - 1e-9))
        self.assertLessEqual(p.max(), (1 - eps) + eps / self.n + 1e-9)

    def test_epsilon_zero_leaves_softmax_untouched(self):
        floor = self._lpacrl(epsilon=0.0)
        d = _Driver(floor)
        d.stage(np.zeros(self.n))
        spike = np.zeros(self.n)
        spike[0] = 40.0
        d.stage(spike)
        # Without the floor the coldest cell sits far below eps/n.
        self.assertLess(floor.probabilities().min(), 0.05 / self.n)

    # ------------------------------------------------------------------ ALP
    def test_alp_uses_abs_lp(self):
        cur = ALPEpisodeCurriculum(self.space, stage_length_control_steps=1, beta=BETA, seed=0)
        d = _Driver(cur)
        base = np.full(self.n, 10.0)
        d.stage(base)                      # baseline returns = 10 everywhere
        nxt = np.full(self.n, 10.0)
        nxt[2] = -5.0                       # cell 2 REGRESSES (LP = -15)
        d.stage(nxt)
        p = cur.probabilities()
        # ALP scores |LP|, so a regression raises sampling just like progress.
        self.assertGreater(p[2], p[50])     # cell 50 has LP=0
        self.assertGreater(p.min(), 1e-4)

    # ------------------------------------------------------------------ checkpoint
    def test_checkpoint_roundtrip_with_epsilon(self):
        cur = self._lpacrl(epsilon=0.03)
        d = _Driver(cur)
        d.stage(np.zeros(self.n))
        spike = np.zeros(self.n)
        spike[1] = 8.0
        d.stage(spike)
        state = cur.state_dict()

        same = self._lpacrl(epsilon=0.03)
        same.load_state_dict(state)
        self.assertTrue(np.allclose(same.probabilities(), cur.probabilities()))

        different = self._lpacrl(epsilon=0.05)
        with self.assertRaises(ValueError):
            different.load_state_dict(state)   # config fingerprint mismatch


if __name__ == "__main__":
    unittest.main()
