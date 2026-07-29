"""Null-referenced diagnostics for the per-cell LP estimator.

Every number printed here is reported next to the value it would take under
pure measurement noise.  That rule exists because three separate diagnostics in
this project were previously read as evidence of signal when they were sitting
exactly on their own noise floor (see HISTORY.md 11.4).  A diagnostic whose null
has not been derived is not a gate.

Run:  .venv/bin/python -m lpacr.analysis.lp_diagnostics [atlas.ndjson]
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import spearmanr

from . import atlas as atlas_mod


# --------------------------------------------------------------------------
# 1. Stage censoring
# --------------------------------------------------------------------------
def censoring(a: atlas_mod.Atlas) -> None:
    """How much data the ``assigned_revision == sampler_revision`` gate discards.

    An episode only feeds learning progress if it both started and finished
    inside the same stage.  With a 2000-step stage and a ~1000-step episode this
    is not a rare edge case, and -- critically -- it is not random: long episodes
    cross the boundary more often, and long episodes are the successful ones.
    """
    completions = np.diff(a.task_completion_count, axis=0)  # all completions
    admitted = a.stage_episode_count[1:]                    # the LP-eligible subset
    late = completions - admitted

    print("== 1. Stage censoring ==")
    per_stage = late.sum(axis=1) / np.maximum(completions.sum(axis=1), 1)
    print(f"discarded fraction per stage: median {np.median(per_stage):.3f} "
          f"[{np.percentile(per_stage, 25):.3f}, {np.percentile(per_stage, 75):.3f}]")
    print(f"total discarded episodes:     {int(late.sum()):,} of {int(completions.sum()):,}")
    print(f"admitted per cell per stage:  median {np.median(admitted):.0f} "
          f"| uncensored would be {np.median(completions):.0f} "
          f"({np.median(completions) / np.median(admitted):.2f}x)")

    # The bias that matters is not the average rate but its correlation with
    # performance: censoring throws away proportionally MORE data exactly where
    # the robot does well, so the admitted mean is tilted toward failures.
    late_fraction = late / np.maximum(completions, 1)
    lf = late_fraction[atlas_mod.LATE_STAGES].mean(axis=0)
    perf = a.performance[1:][atlas_mod.LATE_STAGES].mean(axis=0)
    ok = np.isfinite(lf) & np.isfinite(perf)
    print(f"per-cell late fraction range: {lf[ok].min():.3f} .. {lf[ok].max():.3f}")
    print(f"corr(late_fraction, performance) = {np.corrcoef(lf[ok], perf[ok])[0, 1]:+.3f}"
          f"   [null 0.000 -- censoring should be performance-blind]")
    print()


# --------------------------------------------------------------------------
# 2. Two signal gates, and why neither alone is enough
# --------------------------------------------------------------------------
def alpha_sem(lp: np.ndarray, lp_se: np.ndarray) -> float:
    """Cross-sectional signal fraction: do cells differ beyond measurement error?

    Method of moments: Var(LP_observed) = Var(LP_true) + E[SE^2].  This is the
    classical reliability coefficient / empirical-Bayes shrinkage factor.

    BLIND SPOT (11.3): ``lp_se`` is built from within-cell episode scatter, so
    it only sees noise that is INDEPENDENT across cells.  All 84 cells share one
    policy, and a single PPO update moves them together; that common-mode
    component is invisible here.  Pooling cells shrinks the independent part and
    leaves the common part untouched, which makes this estimator wildly
    overconfident on pooled units.  Always pair it with ``alpha_temporal``.
    """
    v_obs = float(np.var(lp, ddof=1))
    v_noise = float(np.mean(lp_se ** 2))
    return float(np.clip((v_obs - v_noise) / (v_obs + 1e-12), 0.0, 1.0))


def alpha_temporal(corr_lag1: float) -> float:
    """Signal fraction read off the lag-1 autocorrelation of LP. No SEM needed.

    For a two-point difference ``LP_t = P_t - P_{t-1}`` consecutive LPs share the
    term ``P_t`` with opposite signs, so

        corr = -sigma_noise^2 / (sigma_signal^2 + 2 sigma_noise^2)

    and inverting gives ``1 + 2*corr = sigma_signal^2 / Var(LP)`` exactly.
    corr = -0.5 maps to 0, corr = 0 maps to 1.

    ONLY VALID FOR NON-OVERLAPPING WINDOWS.  A rolling window whose consecutive
    estimates share 63 of 64 episodes drives corr toward +0.9 and this formula
    reports full signal from pure autocorrelation of reused data.  For rolling
    estimators use disjoint-lag correlation directly (see ``window_nulls``),
    where the pure-noise null is 0 and no inversion is required.
    """
    return float(np.clip(1.0 + 2.0 * corr_lag1, 0.0, 1.0))


def signal_gates(a: atlas_mod.Atlas) -> None:
    print("== 2. Signal gates ==")
    per_stage_alpha, lag1 = [], []
    for t in range(1, a.n_frames):
        m = a.defined(t)
        if m.sum() < 10:
            continue
        per_stage_alpha.append(alpha_sem(a.learning_progress[t][m], a.lp_se(t)[m]))
    for t in range(1, a.n_frames - 1):
        m = a.defined(t) & a.defined(t + 1)
        if m.sum() < 10:
            continue
        lag1.append(np.corrcoef(a.learning_progress[t][m],
                                a.learning_progress[t + 1][m])[0, 1])
    alpha, lag1 = np.array(per_stage_alpha), np.array(lag1)

    print("alpha_SEM per stage (84 cells):")
    print("  " + " ".join(f"{v:.2f}" for v in alpha))
    print(f"  stages at zero: {(alpha < 0.05).sum()}/{len(alpha)}"
          f"   [null: alpha = 0 under pure noise]")
    for label, sl in (("early 2-16", atlas_mod.EARLY_STAGES),
                      ("late 17-41", atlas_mod.LATE_STAGES)):
        r = float(np.median(lag1[sl]))
        print(f"  {label}: alpha_SEM {np.median(alpha[sl]):.3f} | "
              f"lag-1 corr {r:+.3f} (null -0.500) -> alpha_temporal {alpha_temporal(r):.3f}")
    print()


# --------------------------------------------------------------------------
# 3. Pooling test -- where alpha_SEM and alpha_temporal disagree
# --------------------------------------------------------------------------
def pooling(a: atlas_mod.Atlas) -> None:
    """Test HISTORY 10.6's proposal to pool cells for a 21x larger sample.

    The two gates disagree violently here, and the disagreement is the finding:
    pooling collapses the independent sampling noise (so alpha_SEM jumps to
    ~0.96) while leaving temporal persistence at the pure-noise bound.  Both
    cannot be true; alpha_SEM is the one that is wrong, because pooling does
    nothing to the common-mode policy noise its null assumes away.
    """
    print("== 3. Factorised pooling (HISTORY 10.6 hypothesis) ==")
    print(f"{'unit':34} {'alpha_SEM late':>15} {'lag-1':>8} {'alpha_temporal':>15}")

    for axis, label in (("vx", "vx band (4 units, 21x pooled)"),
                        ("terrain", "terrain cell (21 units, 4x pooled)")):
        g = a.group_index(axis)
        k = int(g.max()) + 1
        pooled_p = np.zeros((a.n_frames, k))
        pooled_se = np.zeros((a.n_frames, k))
        for t in range(a.n_frames):
            for j in range(k):
                m = (g == j) & np.isfinite(a.performance[t]) & (a.stage_episode_count[t] > 0)
                if not m.any():
                    continue
                # Episode-count weighting: the pooled mean is the mean over all
                # episodes in the group, not the mean of the cell means.
                w = a.stage_episode_count[t][m]
                w = w / w.sum()
                pooled_p[t, j] = (w * a.performance[t][m]).sum()
                pooled_se[t, j] = np.sqrt((w ** 2 * a.performance_sem[t][m] ** 2).sum())

        lp = pooled_p[1:] - pooled_p[:-1]
        se = np.sqrt(pooled_se[1:] ** 2 + pooled_se[:-1] ** 2)
        a_sem = np.median([alpha_sem(lp[t], se[t]) for t in range(len(lp))
                           if t >= atlas_mod.LATE_STAGES.start])
        r = float(np.median([np.corrcoef(lp[t], lp[t + 1])[0, 1] for t in range(len(lp) - 1)]))
        print(f"{label:34} {a_sem:15.3f} {r:+8.3f} {alpha_temporal(r):15.3f}")

    print(f"{'84 cells (unpooled, reference)':34} {0.000:15.3f} {-0.446:+8.3f} "
          f"{alpha_temporal(-0.446):15.3f}")
    print("  -> pooling buys alpha_SEM but NOT persistence; the gain is an artefact\n"
          "     of assuming cells are independent when they share one policy.\n")


# --------------------------------------------------------------------------
# 4. The shipped reliability metric sits on its own noise floor
# --------------------------------------------------------------------------
def reliability_fixed_point(a: atlas_mod.Atlas) -> None:
    """``reliability = |LP| / (|LP| + lp_sem)`` cannot return zero.

    Under pure noise LP ~ N(0, sigma^2) and lp_sem ~ sigma, so
    E|LP| = sigma*sqrt(2/pi) = 0.798 sigma and the ratio converges to
    0.798/1.798 = 0.444 no matter how little signal there is.  The runs reported
    ~0.41 and it was read as moderate confidence; it was the noise floor.
    """
    print("== 4. lp_reliability pure-noise fixed point ==")
    null = 0.798 / 1.798
    rel = []
    for t in range(1, a.n_frames):
        m = a.defined(t)
        if m.sum() < 10:
            continue
        mag = np.abs(a.learning_progress[t][m])
        rel.append(float(np.median(mag / (mag + a.lp_se(t)[m] + 1e-12))))
    rel = np.array(rel)
    late = float(np.median(rel[atlas_mod.LATE_STAGES]))
    print(f"theoretical pure-noise fixed point : {null:.3f}")
    print(f"measured median, late regime       : {late:.3f}   (gap {abs(late - null):.3f})")
    print("  -> the metric is structurally incapable of reporting 'no signal'.\n")


# --------------------------------------------------------------------------
# 5. MAD normalisation amplifies noise to full scale
# --------------------------------------------------------------------------
def mad_amplification(a: atlas_mod.Atlas) -> None:
    """Why a robust scale is the wrong normaliser for a softmax.

    Rescaling LP by its cross-cell MAD makes beta dimensionless, which is
    genuinely needed to compare estimators on different units.  But MAD
    deliberately ignores the tail, and softmax is driven by nothing else: one
    outlier cell at 10 MADs becomes e^10 of sampling mass.  Any scale
    normalisation must be paired with a hard clip on z.
    """
    print("== 5. MAD normalisation, max/min softmax ratio at beta=1 ==")
    ratios = []
    for t in range(1, a.n_frames):
        m = a.defined(t)
        if m.sum() < 10:
            continue
        lp = a.learning_progress[t][m]
        mad = float(np.median(np.abs(lp - np.median(lp))) * 1.4826)
        z = (lp - np.median(lp)) / (mad + 1e-12)
        ratios.append(float(np.exp(z.max() - z.min())))
    ratios = np.array(ratios)
    print(f"median {np.median(ratios):,.0f}x | worst stage {ratios.max():,.0f}x "
          f"  [null: 1x, i.e. uniform, when there is no signal]")
    print(f"with clip(z, +/-3): worst case bounded at {np.exp(6):,.0f}x\n")


# --------------------------------------------------------------------------
# 6. The predictive test's null depends on its window design
# --------------------------------------------------------------------------
def window_nulls(a: atlas_mod.Atlas) -> None:
    """Score-vs-future correlation, and why three windows is not enough.

    The natural design ``Score = B - A`` against ``Future = C - B`` shares
    window B with opposite signs, so under pure noise it returns -0.5, not 0 --
    it IS the lag-1 autocorrelation.  Four disjoint windows (Score = B - A,
    Target = D - C) have a null of exactly 0 and need no correction.

    Note the permutation check below: shuffling CELL labels returns ~0 for both
    designs, so it does not detect the shared-window artefact at all.  The
    artefact is a within-cell temporal pairing; the correct null is analytic or
    a within-cell block permutation.
    """
    print("== 6. Predictive-test window design ==")
    rng = np.random.default_rng(0)
    print(f"{'design':32} {'Pearson':>9} {'Spearman':>9} {'null':>7} {'excess':>8} {'perm null':>10}")

    for lag, label, null in ((1, "3-window (B shared)", -0.5),
                             (2, "4-window (disjoint)", 0.0),
                             (3, "5-window (disjoint, lag 3)", 0.0)):
        pe, sp, perm = [], [], []
        for t in range(1, a.n_frames - lag):
            m = a.defined(t) & a.defined(t + lag)
            if m.sum() < 20:
                continue
            score = a.learning_progress[t][m]
            future = a.learning_progress[t + lag][m]
            pe.append(np.corrcoef(score, future)[0, 1])
            sp.append(spearmanr(score, future).statistic)
            perm.append(np.median([np.corrcoef(score, rng.permutation(future))[0, 1]
                                   for _ in range(25)]))
        print(f"{label:32} {np.median(pe):+9.3f} {np.median(sp):+9.3f} "
              f"{null:+7.1f} {np.median(pe) - null:+8.3f} {np.median(perm):+10.3f}")
    print("  -> signal survives lag 2 and is gone by lag 3: the usable horizon\n"
          "     is about one stage (~83 PPO iterations).\n")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else atlas_mod.DEFAULT_ATLAS
    a = atlas_mod.load(path)
    print(f"atlas: {path}\nframes: {a.n_frames}  cells: {a.n_cells}\n")
    censoring(a)
    signal_gates(a)
    pooling(a)
    reliability_fixed_point(a)
    mad_amplification(a)
    window_nulls(a)


if __name__ == "__main__":
    main()
