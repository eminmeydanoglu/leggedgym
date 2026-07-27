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

BETA = 5.0  # test temperature (the shipped freeze is V5_BETA = 1.0)


def _observe_multi(cur, per_cell_returns, *, revision, length=100):
    """Feed several same-revision outcomes per cell.

    ``per_cell_returns`` maps a cell id to an iterable of episodic returns; each
    return becomes one completed episode, so cells can clear the
    ``min_stage_episodes_for_lp`` eligibility gate and get a finite SEM.
    """
    cells = []
    returns = []
    for cell, cell_returns in per_cell_returns.items():
        for r in cell_returns:
            cells.append(cell)
            returns.append(r)
    _observe(cur, cells, returns, revision=revision, length=length)


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
        """Observe every cell (except ``skip``) with ``returns`` then advance.

        Each cell gets ``min_stage_episodes_for_lp`` copies of its return so the
        episode gate admits the LP; a single episode per cell is exactly what
        the gate is there to reject.
        """
        returns = np.asarray(returns, dtype=np.float64)
        cells = [c for c in range(self.n) if c not in set(skip)]
        reps = self.cur.min_stage_episodes_for_lp
        _observe(
            self.cur,
            [c for c in cells for _ in range(reps)],
            np.repeat(returns[cells], reps),
            revision=self.cur.sampler_revision,
        )
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
        # max_cell_probability=1.0 isolates the epsilon floor: the shipped cap
        # would redistribute the spike's mass and lift the coldest cell back
        # above the threshold this test is asserting on.
        floor = self._lpacrl(epsilon=0.0, max_cell_probability=1.0)
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

    # ------------------------------------------------ curriculum diagnostics
    def _lpacrl_eligible(self, **kw):
        # min_stage_episodes_for_lp=2 so a couple of episodes per cell clears
        # the LP eligibility gate in these small synthetic sequences.
        return LPACRLEpisodeCurriculum(
            self.space, stage_length_control_steps=1, beta=BETA, seed=0,
            min_stage_episodes_for_lp=2, **kw,
        )

    def test_top10_overlap_is_a_known_fraction(self):
        cur = self._lpacrl()
        current = np.zeros(self.n)
        current[np.arange(10)] = np.arange(10, 0, -1)      # top-10 == {0..9}
        previous = np.zeros(self.n)
        previous[np.arange(5, 15)] = np.arange(10, 0, -1)  # top-10 == {5..14}
        # {0..9} ∩ {5..14} == {5,6,7,8,9} -> 5/10.
        self.assertAlmostEqual(cur._top10_overlap(current, previous), 0.5)
        self.assertAlmostEqual(cur._top10_overlap(current, current), 1.0)

    def test_tv_distance_uniform_unit(self):
        cur = self._lpacrl()
        # Uniform distribution -> zero total variation from uniform.
        self.assertAlmostEqual(cur.diagnostics()["tv_distance_uniform"], 0.0, places=12)
        skew = np.zeros(self.n)
        skew[0] = 0.5
        skew[1:] = 0.5 / (self.n - 1)
        cur._probabilities = skew
        expected = 0.5 * np.sum(np.abs(skew - 1.0 / self.n))
        self.assertAlmostEqual(cur.diagnostics()["tv_distance_uniform"], float(expected), places=12)

    def test_first_stage_diagnostics_are_null(self):
        cur = self._lpacrl_eligible()
        _observe_multi(cur, {c: (0.0, 0.0) for c in range(self.n)}, revision=0)
        cur.advance(cur.stage_length_control_steps)
        diag = cur.diagnostics()
        # No prior stage / no cross-stage LP yet -> the three cross-stage
        # metrics are NaN; TV is still defined (distribution is uniform -> 0).
        self.assertTrue(np.isnan(diag["top10_overlap_prev"]))
        self.assertTrue(np.isnan(diag["lp_reliability_median"]))
        self.assertTrue(np.isnan(diag["sampled_lp_mass"]))
        self.assertAlmostEqual(diag["tv_distance_uniform"], 0.0, places=12)

    def test_reliability_and_mass_on_two_stage_sequence(self):
        cur = self._lpacrl_eligible()
        # Stage 1 baseline: every cell measured twice at return 0 (finite SEM=0).
        _observe_multi(cur, {c: (0.0, 0.0) for c in range(self.n)}, revision=cur.sampler_revision)
        cur.advance(cur.stage_length_control_steps)
        # Stage 2: every cell jumps to a noiseless return 10 -> LP=10, SEM=0,
        # so reliability |LP|/(|LP|+lp_sem) == 1 for every eligible cell.
        _observe_multi(cur, {c: (10.0, 10.0) for c in range(self.n)}, revision=cur.sampler_revision)
        cur.advance(2 * cur.stage_length_control_steps)
        diag = cur.diagnostics()
        self.assertAlmostEqual(diag["lp_reliability_median"], 1.0, places=9)
        # Uniform LP over all cells -> uniform distribution -> the sampler
        # targets exactly its fair share 1/n of the positive-LP mass.
        self.assertAlmostEqual(diag["sampled_lp_mass"], 1.0 / self.n, places=9)
        # A real prior stage now exists -> overlap is defined and in [0, 1].
        self.assertFalse(np.isnan(diag["top10_overlap_prev"]))
        self.assertGreaterEqual(diag["top10_overlap_prev"], 0.0)
        self.assertLessEqual(diag["top10_overlap_prev"], 1.0)

    def test_sampled_lp_mass_rises_when_distribution_targets_positive_lp(self):
        cur = self._lpacrl_eligible()
        _observe_multi(cur, {c: (0.0, 0.0) for c in range(self.n)}, revision=cur.sampler_revision)
        cur.advance(cur.stage_length_control_steps)
        # One cell shows large LP, the rest a tiny positive LP: the softmax
        # concentrates on the hot cell, so the sampled positive-LP mass exceeds
        # the uniform fair share 1/n.
        nxt = {c: (0.5, 0.5) for c in range(self.n)}
        nxt[0] = (40.0, 40.0)
        _observe_multi(cur, nxt, revision=cur.sampler_revision)
        cur.advance(2 * cur.stage_length_control_steps)
        diag = cur.diagnostics()
        self.assertGreater(diag["sampled_lp_mass"], 1.0 / self.n)

    def test_curriculum_diagnostics_reset_to_null_after_resume(self):
        cur = self._lpacrl_eligible()
        _observe_multi(cur, {c: (0.0, 0.0) for c in range(self.n)}, revision=cur.sampler_revision)
        cur.advance(cur.stage_length_control_steps)
        _observe_multi(cur, {c: (10.0, 10.0) for c in range(self.n)}, revision=cur.sampler_revision)
        cur.advance(2 * cur.stage_length_control_steps)
        state = cur.state_dict()
        # State does not carry the previous-stage distribution, so a resumed
        # curriculum degrades top10_overlap_prev to NaN on its first stage.
        resumed = self._lpacrl_eligible()
        resumed.load_state_dict(state)
        _observe_multi(resumed, {c: (20.0, 20.0) for c in range(self.n)}, revision=resumed.sampler_revision)
        step = (resumed.stage_index + 1) * resumed.stage_length_control_steps
        resumed.advance(step)
        self.assertTrue(np.isnan(resumed.diagnostics()["top10_overlap_prev"]))

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
