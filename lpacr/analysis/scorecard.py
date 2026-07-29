"""LP quality scorecard — reusable metrics with pure-noise nulls (§11.0).

Each metric returns a dict with at least:
  value, null, pass (bool|None), ci (optional), unavailable/reason when blocked.

Metrics:
  A1 reliability gates α_SEM / α_temporal / α=min
  A3 temporal ACF lag 1..5 + half-life
  A5 top-k ranking stability vs hypergeometric null
  A4 criterion validity (validation bank)
  A7 saturation / headroom
  C2 cross-arm split-half reliability
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import hypergeom, spearmanr

from . import atlas as atlas_mod
from .lp_diagnostics import alpha_sem, alpha_temporal
from .validation_bank import ValidationBank, cell_improvement

RNG = np.random.default_rng(20260729)
N_PERM = 200
N_BOOT = 200


def _median_ci(x: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    med = float(np.median(x))
    if x.size < 3:
        return (med, med, med)
    boots = [
        float(np.median(RNG.choice(x, size=x.size, replace=True)))
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, float(lo), float(hi)


def _metric(
    value: float | None,
    null: float | None,
    *,
    pass_: bool | None = None,
    ci: tuple[float, float] | None = None,
    extra: dict | None = None,
    unavailable: str | None = None,
) -> dict[str, Any]:
    if unavailable:
        return {
            "value": None,
            "null": null,
            "pass": None,
            "unavailable": unavailable,
            **(extra or {}),
        }
    out: dict[str, Any] = {
        "value": None if value is None or (isinstance(value, float) and not np.isfinite(value)) else float(value),
        "null": null if null is None else float(null),
        "pass": pass_,
    }
    if ci is not None:
        out["ci95"] = [float(ci[0]), float(ci[1])]
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# A1 — reliability
# ---------------------------------------------------------------------------
def _quality_req(a: atlas_mod.Atlas) -> bool:
    """Eligible gate only when the mask actually selects cells (not all-False)."""
    return a.quality_require_eligible()


def compute_alpha_series(a: atlas_mod.Atlas) -> dict[str, Any]:
    """Per-stage α_SEM, lag-1 corr, α_temporal, α=min."""
    if a.n_frames < 3:
        return {"unavailable": "too few frames", "per_stage": []}

    # §11 uses eligible when meaningful; all-False eligible (UNI/crash dumps)
    # falls back to finite LP so observational quality is still measurable.
    req = _quality_req(a)
    mask_note = a.quality_mask_note()

    a_sem_list: list[float] = []
    stage_idx: list[int] = []
    for t in range(1, a.n_frames):
        m = a.defined(t, require_eligible=req)
        if m.sum() < 10:
            a_sem_list.append(float("nan"))
            stage_idx.append(t)
            continue
        if a.has_sem:
            a_sem_list.append(alpha_sem(a.learning_progress[t][m], a.lp_se(t)[m]))
        else:
            a_sem_list.append(float("nan"))
        stage_idx.append(t)

    lag1: list[float] = []
    for t in range(1, a.n_frames - 1):
        m = a.defined(t, require_eligible=req) & a.defined(
            t + 1, require_eligible=req
        )
        if m.sum() < 10:
            lag1.append(float("nan"))
            continue
        c = np.corrcoef(a.learning_progress[t][m], a.learning_progress[t + 1][m])[0, 1]
        lag1.append(float(c))

    a_sem_arr = np.array(a_sem_list, float)
    lag1_arr = np.array(lag1, float)
    # align lag1 to stages 1..n-2 → length n_frames-2; a_sem length n_frames-1
    a_temp_arr = np.array(
        [alpha_temporal(r) if np.isfinite(r) else np.nan for r in lag1_arr]
    )
    # α_min per stage where both available: use lag1 stages
    n = min(len(a_sem_arr) - 1, len(a_temp_arr)) if len(a_sem_arr) else 0
    a_min_list = []
    for i in range(n):
        s, t = a_sem_arr[i], a_temp_arr[i]
        if np.isfinite(s) and np.isfinite(t):
            a_min_list.append(min(s, t))
        elif np.isfinite(t):
            a_min_list.append(t)  # SEM missing: temporal only
        else:
            a_min_list.append(np.nan)
    a_min_arr = np.array(a_min_list, float)

    def _regime_chunk(arr: np.ndarray, sl: slice) -> np.ndarray:
        return np.asarray(arr[sl], float)

    def _regime_summary(arr: np.ndarray, sl: slice) -> dict[str, Any]:
        chunk = _regime_chunk(arr, sl)
        med, lo, hi = _median_ci(chunk)
        return {
            "median": med,
            "ci95": [lo, hi],
            "n_stages": int(np.isfinite(chunk).sum()),
            "frac_near_zero": float(np.mean(chunk[np.isfinite(chunk)] < 0.05))
            if np.isfinite(chunk).any()
            else None,
        }

    # Match §11.2: α_temporal from median(lag1) then invert (not median of α_t).
    def _alpha_temp_regime(sl: slice) -> dict[str, Any]:
        chunk = _regime_chunk(lag1_arr, sl)
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size == 0:
            return _metric(None, 0.0, unavailable="insufficient stages")
        r_med, r_lo, r_hi = _median_ci(chunk)
        # CI on α via transform of corr CI (monotone in r for r in [-0.5, 0.5])
        a_med = alpha_temporal(r_med)
        a_lo = alpha_temporal(r_lo)
        a_hi = alpha_temporal(r_hi)
        # ensure lo<=hi after transform
        if a_lo > a_hi:
            a_lo, a_hi = a_hi, a_lo
        return _metric(
            a_med,
            0.0,
            pass_=bool(a_med > 0.05),
            ci=(a_lo, a_hi),
            extra={
                "lag1_median": r_med,
                "lag1_null": -0.5,
                "n_stages": int(chunk.size),
            },
        )

    early_sl = atlas_mod.EARLY_STAGES
    late_sl = atlas_mod.LATE_STAGES
    sem_unavailable = (
        None if a.has_sem else "performance_sem missing (run #1 partial schema)"
    )

    def _gate_sem(sl, pass_rule):
        if not a.has_sem:
            return _metric(None, 0.0, unavailable=sem_unavailable)
        s = _regime_summary(a_sem_arr, sl)
        val = s["median"]
        if not np.isfinite(val):
            return _metric(None, 0.0, unavailable="insufficient stages")
        return _metric(
            val,
            0.0,
            pass_=pass_rule(val),
            ci=(s["ci95"][0], s["ci95"][1]),
            extra={
                "n_stages": s["n_stages"],
                "frac_near_zero": s["frac_near_zero"],
            },
        )

    def pass_alpha(v):
        return bool(v > 0.05)

    # α_min: min of regime medians (same as pairing gates)
    def _alpha_min_regime(sl):
        s = _gate_sem(sl, pass_alpha)
        t = _alpha_temp_regime(sl)
        if s.get("unavailable") and t.get("value") is not None:
            return _metric(
                t["value"],
                0.0,
                pass_=t.get("pass"),
                ci=tuple(t["ci95"]) if t.get("ci95") else None,
                extra={"note": "SEM missing; α_min = α_temporal only", **{
                    k: t[k] for k in ("lag1_median", "n_stages") if k in t
                }},
            )
        if s.get("value") is None or t.get("value") is None:
            return _metric(
                None,
                0.0,
                unavailable=s.get("unavailable") or t.get("unavailable"),
            )
        val = min(float(s["value"]), float(t["value"]))
        return _metric(
            val,
            0.0,
            pass_=pass_alpha(val),
            extra={"alpha_sem": s["value"], "alpha_temporal": t["value"]},
        )

    out = {
        "mask_note": mask_note,
        "require_eligible": req,
        "alpha_sem": {
            "null": 0.0,
            "null_meaning": "pure measurement noise: Var(LP)=E[SE^2]",
            "early": _gate_sem(early_sl, pass_alpha),
            "late": _gate_sem(late_sl, pass_alpha),
            "all": _gate_sem(slice(None), pass_alpha),
            "per_stage": [None if not np.isfinite(v) else float(v) for v in a_sem_arr],
            "stages_at_zero": int(np.sum(a_sem_arr[np.isfinite(a_sem_arr)] < 0.05))
            if a.has_sem
            else None,
            "n_stages_total": int(np.isfinite(a_sem_arr).sum()) if a.has_sem else 0,
            "unavailable": sem_unavailable,
        },
        "alpha_temporal": {
            "null": 0.0,
            "null_meaning": "corr_lag1 = -0.5 under pure noise for non-overlapping ΔP",
            "early": _alpha_temp_regime(early_sl),
            "late": _alpha_temp_regime(late_sl),
            "all": _alpha_temp_regime(slice(None)),
            "lag1_per_stage": [None if not np.isfinite(v) else float(v) for v in lag1_arr],
            "lag1_null": -0.5,
        },
        "alpha_min": {
            "null": 0.0,
            "null_meaning": "min(α_SEM, α_temporal); either gate alone is blind",
            "early": _alpha_min_regime(early_sl),
            "late": _alpha_min_regime(late_sl),
            "all": _alpha_min_regime(slice(None)),
            "per_stage": [None if not np.isfinite(v) else float(v) for v in a_min_arr],
            "note": "when SEM missing, α_min falls back to α_temporal only",
        },
    }
    return out


# ---------------------------------------------------------------------------
# A3 — ACF + half-life
# ---------------------------------------------------------------------------
def compute_acf(a: atlas_mod.Atlas, max_lag: int = 5) -> dict[str, Any]:
    """Disjoint-window ACF of LP at lags 1..max_lag with analytic nulls.

    lag 1 (shared window B): analytic null = -0.5
    lag >= 2 (disjoint): analytic null = 0.0
    Also reports excess over null and block-permutation check.
    """
    req = _quality_req(a)
    results = []
    half_life_stage = None
    for lag in range(1, max_lag + 1):
        null = -0.5 if lag == 1 else 0.0
        pe_list, sp_list = [], []
        for t in range(1, a.n_frames - lag):
            m = a.defined(t, require_eligible=req) & a.defined(
                t + lag, require_eligible=req
            )
            if m.sum() < 20:
                continue
            s = a.learning_progress[t][m]
            f = a.learning_progress[t + lag][m]
            pe_list.append(float(np.corrcoef(s, f)[0, 1]))
            sp_list.append(float(spearmanr(s, f).statistic))
        pe = np.array(pe_list, float)
        sp = np.array(sp_list, float)
        if pe.size == 0:
            results.append(
                _metric(None, null, unavailable="insufficient stage pairs", extra={"lag": lag})
            )
            continue
        med, lo, hi = _median_ci(pe)
        excess = med - null
        # block perm: shuffle future within each stage pair (cell labels)
        # correct check is within-cell temporal block — approximate by
        # permuting future vector (cell shuffle) which should NOT kill shared-window
        # artefact; report analytic null as primary.
        perm_vals = []
        rng = np.random.default_rng(lag * 17 + 3)
        for t in range(1, a.n_frames - lag):
            m = a.defined(t, require_eligible=req) & a.defined(
                t + lag, require_eligible=req
            )
            if m.sum() < 20:
                continue
            s = a.learning_progress[t][m]
            f = a.learning_progress[t + lag][m]
            for _ in range(5):
                perm_vals.append(float(np.corrcoef(s, rng.permutation(f))[0, 1]))
        perm_med = float(np.median(perm_vals)) if perm_vals else float("nan")
        # pass if excess CI lower bound > 0
        passes = bool(lo - null > 0) if np.isfinite(lo) else False
        results.append(
            _metric(
                med,
                null,
                pass_=passes,
                ci=(lo, hi),
                extra={
                    "lag": lag,
                    "excess_over_null": float(excess),
                    "spearman_median": float(np.median(sp)) if sp.size else None,
                    "cell_label_perm_median": perm_med,
                    "cell_label_perm_note": (
                        "cell-label permutation returns ~0 for both designs and "
                        "does NOT detect shared-window artefact (§11.6)"
                    ),
                    "n_stage_pairs": int(pe.size),
                },
            )
        )
        if half_life_stage is None and lag >= 2 and abs(excess) < 0.02:
            half_life_stage = lag - 1  # signal gone by this lag

    # half-life from lag-1 excess decay
    excesses = [
        r.get("excess_over_null")
        for r in results
        if r.get("excess_over_null") is not None
    ]
    half_life = None
    if excesses and excesses[0] and excesses[0] > 0:
        e0 = excesses[0]
        half_life = None
        for i, e in enumerate(excesses):
            if e is not None and e <= e0 / 2:
                half_life = i + 1
                break
        if half_life is None:
            # interpolate / mark short
            half_life = 1 if e0 > 0 else 0

    return {
        "lags": results,
        "signal_half_life_stages": half_life if half_life is not None else half_life_stage,
        "signal_half_life_iters": (
            (half_life if half_life is not None else half_life_stage or 1) * 83
        ),
        "null_lag1": -0.5,
        "null_lag_ge2": 0.0,
        "mask_note": a.quality_mask_note(),
        "require_eligible": req,
    }


# ---------------------------------------------------------------------------
# A5 — ranking stability
# ---------------------------------------------------------------------------
def hypergeom_overlap_null(k: int, n: int = 84) -> float:
    """Expected fraction overlap of two independent top-k sets of size n.

    E[|A∩B|]/k = k/n  (since E[|A∩B|] = k * (k/n) for sampling without replacement
    of B given A fixed... actually E[|A∩B|] = k * k / n, so E[overlap frac] = k/n.
    For k=10, n=84: 10/84 ≈ 0.119.
    """
    return k / n


def topk_overlap(a: atlas_mod.Atlas, ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, Any]:
    """Adjacent-stage top-k overlap on LP ranking."""
    out: dict[str, Any] = {}
    n = a.n_cells
    for k in ks:
        null = hypergeom_overlap_null(k, n)
        overlaps = []
        req = _quality_req(a)
        for t in range(1, a.n_frames - 1):
            m = a.defined(t, require_eligible=req) & a.defined(
                t + 1, require_eligible=req
            )
            if m.sum() < k * 2:
                continue
            # rank among defined cells only, map back
            ids = np.where(m)[0]
            lp0 = a.learning_progress[t][ids]
            lp1 = a.learning_progress[t + 1][ids]
            top0 = set(ids[np.argsort(-lp0)[:k]])
            top1 = set(ids[np.argsort(-lp1)[:k]])
            overlaps.append(len(top0 & top1) / k)
        if not overlaps:
            out[f"k{k}"] = _metric(None, null, unavailable="insufficient stages")
            continue
        arr = np.array(overlaps, float)
        med, lo, hi = _median_ci(arr)
        # pass if significantly above hypergeom null
        passes = bool(lo > null) if np.isfinite(lo) else False
        # pre-registered campaign criterion >0.3 re-interpreted vs null
        out[f"k{k}"] = _metric(
            med,
            null,
            pass_=passes,
            ci=(lo, hi),
            extra={
                "pre_registered_threshold": 0.3,
                "pre_reg_note": (
                    f"campaign pre-reg >0.3 is {(0.3/null):.1f}× hypergeom null "
                    f"{null:.3f}; still a high bar for stability"
                ),
                "frac_above_prereg": float(np.mean(arr > 0.3)),
                "n_pairs": int(arr.size),
            },
        )
    return out


# ---------------------------------------------------------------------------
# A4 — criterion validity against validation bank
# ---------------------------------------------------------------------------
def atlas_lp_window_mean(
    a: atlas_mod.Atlas, step_lo: int, step_hi: int
) -> np.ndarray:
    """Mean per-cell LP over frames whose step is in [step_lo, step_hi]."""
    mask = (a.step >= step_lo) & (a.step <= step_hi)
    if not mask.any():
        return np.full(a.n_cells, np.nan)
    lp = a.learning_progress[mask]
    return np.nanmean(lp, axis=0)


def criterion_validity(
    a: atlas_mod.Atlas,
    bank: ValidationBank,
    *,
    n_perm: int = N_PERM,
) -> dict[str, Any]:
    """Spearman of atlas LP vs held-out spnte_lin improvement (concurrent + forward).

    spnte lower is better: improvement = spnte[t] - spnte[t+1] (positive = better).
    """
    iters = list(bank.iterations)
    if len(iters) < 2:
        return {"unavailable": "bank has <2 checkpoints"}

    # map checkpoint windows: LP during (it_i, it_{i+1}] predicts improvement it_i→it_{i+1}
    # stage steps: 1 iter = 24 control steps; atlas step is control step
    # iter 1000 ≈ step 24000, etc.
    concurrent = []
    forward = []
    for i in range(len(iters) - 1):
        it0, it1 = iters[i], iters[i + 1]
        step0, step1 = it0 * 24, it1 * 24
        lp = atlas_lp_window_mean(a, step0, step1)
        imp = cell_improvement(bank, i, i + 1, "spnte_lin")
        ok = np.isfinite(lp) & np.isfinite(imp)
        if ok.sum() < 20:
            continue
        sp = float(spearmanr(lp[ok], imp[ok]).statistic)
        concurrent.append({"window": f"{it0}-{it1}", "spearman": sp, "n": int(ok.sum())})

        # forward: LP in window i predicts improvement in window i+1
        if i + 2 <= len(iters) - 1:
            imp_f = cell_improvement(bank, i + 1, i + 2, "spnte_lin")
            okf = np.isfinite(lp) & np.isfinite(imp_f)
            if okf.sum() >= 20:
                spf = float(spearmanr(lp[okf], imp_f[okf]).statistic)
                forward.append(
                    {
                        "lp_window": f"{it0}-{it1}",
                        "future_window": f"{iters[i+1]}-{iters[i+2]}",
                        "spearman": spf,
                        "n": int(okf.sum()),
                    }
                )

    def _agg(rows: list[dict]) -> dict[str, Any]:
        if not rows:
            return _metric(None, 0.0, unavailable="no valid windows")
        vals = np.array([r["spearman"] for r in rows], float)
        med, lo, hi = _median_ci(vals)
        # permutation null for pooled windows
        # use last window for perm detail
        perm = []
        for r in rows:
            # re-derive not stored; approximate with shuffle of reported
            pass
        # overall perm: shuffle improvements for each window and median
        perm_meds = []
        for _ in range(n_perm):
            # null of median spearman under cell shuffle
            perm_meds.append(float(RNG.normal(0, 1 / np.sqrt(84))))  # analytic approx
        # better: recompute one combined
        all_sp = []
        for i in range(len(iters) - 1):
            it0, it1 = iters[i], iters[i + 1]
            lp = atlas_lp_window_mean(a, it0 * 24, it1 * 24)
            imp = cell_improvement(bank, i, i + 1, "spnte_lin")
            ok = np.isfinite(lp) & np.isfinite(imp)
            if ok.sum() < 20:
                continue
            for _ in range(max(1, n_perm // max(len(iters) - 1, 1))):
                all_sp.append(
                    float(spearmanr(lp[ok], RNG.permutation(imp[ok])).statistic)
                )
        perm_null = float(np.median(all_sp)) if all_sp else 0.0
        perm_hi = float(np.percentile(all_sp, 97.5)) if all_sp else 0.0
        passes = bool(lo > perm_hi) if np.isfinite(lo) else False
        return _metric(
            med,
            0.0,
            pass_=passes,
            ci=(lo, hi),
            extra={
                "per_window": rows,
                "perm_null_median": perm_null,
                "perm_null_p975": perm_hi,
                "n_windows": len(rows),
                "exploratory": True,
            },
        )

    return {
        "concurrent": _agg(concurrent),
        "forward": _agg(forward),
        "note": (
            "forward-shifted is the primary test; concurrent is descriptive. "
            "spnte_lin improvement = earlier - later (lower error is better)."
        ),
        "single_seed_warning": True,
    }


# ---------------------------------------------------------------------------
# A7 — saturation / headroom
# ---------------------------------------------------------------------------
def saturation_headroom(bank: ValidationBank) -> dict[str, Any]:
    """Fit per-cell spnte curves; report remaining headroom dispersion."""
    y = bank.cell_spnte_lin  # (T, C)
    T, C = y.shape
    x = np.arange(T, dtype=float)
    slopes = []
    final = y[-1]
    initial = y[0]
    improvement = initial - final  # positive = better
    for c in range(C):
        yc = y[:, c]
        if not np.all(np.isfinite(yc)):
            slopes.append(np.nan)
            continue
        # linear slope of spnte over checkpoints (negative slope = improving)
        b = np.polyfit(x, yc, 1)[0]
        slopes.append(float(b))
    slopes = np.array(slopes, float)
    headroom = final - np.nanmin(final)  # relative to best cell at end
    # dispersion of final performance across cells
    final_std = float(np.nanstd(final, ddof=1))
    final_iqr = float(np.nanpercentile(final, 75) - np.nanpercentile(final, 25))
    # if late checkpoints flat: mean |slope| last half
    late_slopes = []
    if T >= 4:
        for c in range(C):
            yc = y[T // 2 :, c]
            if np.all(np.isfinite(yc)):
                late_slopes.append(float(np.polyfit(np.arange(len(yc)), yc, 1)[0]))
    late_slopes = np.array(late_slopes, float) if late_slopes else np.array([np.nan])

    # null: if no cell differences, final_std ≈ replica SEM
    # approximate from within-cell replica noise at last ckpt
    last = -1
    within_sds = []
    for c in range(C):
        m = bank.cell_id[last] == c
        if m.sum() >= 2:
            within_sds.append(float(np.std(bank.spnte_lin[last][m], ddof=1)))
    noise_sd = float(np.median(within_sds)) / np.sqrt(12) if within_sds else np.nan
    # between-cell signal fraction
    if np.isfinite(noise_sd) and final_std > 0:
        alpha_cells = float(np.clip(1 - noise_sd ** 2 / final_std ** 2, 0, 1))
    else:
        alpha_cells = None

    dispersion_near_zero = bool(final_std < 2 * noise_sd) if np.isfinite(noise_sd) else None

    return {
        "macro_spnte_curve": [float(v) for v in bank.macro_spnte_lin],
        "iterations": list(bank.iterations),
        "final_cell_std": final_std,
        "final_cell_iqr": final_iqr,
        "cell_sem_proxy": noise_sd,
        "between_cell_alpha": alpha_cells,
        "mean_improvement_initial_to_final": float(np.nanmean(improvement)),
        "std_improvement": float(np.nanstd(improvement, ddof=1)),
        "median_late_slope": float(np.nanmedian(late_slopes)),
        "dispersion_near_noise": dispersion_near_zero,
        "null": {
            "between_cell_alpha": 0.0,
            "meaning": "if all cells equally learnable, residual std ≈ cell SEM",
        },
        "pass_heterogeneous_learnability": bool(alpha_cells is not None and alpha_cells > 0.1),
        "interpretation": (
            "dispersion≈noise ⇒ uniform sampling is the correct answer (measured), "
            "not a campaign failure"
            if dispersion_near_zero
            else "cells still differ at final checkpoint beyond replica noise"
        ),
    }


# ---------------------------------------------------------------------------
# C2 — cross-arm reliability
# ---------------------------------------------------------------------------
def cross_arm_reliability(
    a_lp: atlas_mod.Atlas, a_uni: atlas_mod.Atlas
) -> dict[str, Any]:
    """corr(LP_arm(t), UNI_arm(t)) at matched steps — lower bound on reliability."""
    # match by step
    steps_lp = {int(s): i for i, s in enumerate(a_lp.step)}
    steps_uni = {int(s): i for i, s in enumerate(a_uni.step)}
    common = sorted(set(steps_lp) & set(steps_uni))
    # skip first if present (bootstrap already dropped)
    lp_corrs, perf_corrs, steps_out = [], [], []
    for s in common:
        i, j = steps_lp[s], steps_uni[s]
        m = np.isfinite(a_lp.learning_progress[i]) & np.isfinite(
            a_uni.learning_progress[j]
        )
        if m.sum() < 20:
            continue
        lp_corrs.append(
            float(
                np.corrcoef(
                    a_lp.learning_progress[i][m], a_uni.learning_progress[j][m]
                )[0, 1]
            )
        )
        mp = np.isfinite(a_lp.performance[i]) & np.isfinite(a_uni.performance[j])
        if mp.sum() >= 20:
            perf_corrs.append(
                float(
                    np.corrcoef(a_lp.performance[i][mp], a_uni.performance[j][mp])[
                        0, 1
                    ]
                )
            )
        steps_out.append(s)

    if not lp_corrs:
        return {"unavailable": "no matched steps with finite LP"}

    lp_arr = np.array(lp_corrs)
    perf_arr = np.array(perf_corrs) if perf_corrs else np.array([np.nan])
    med, lo, hi = _median_ci(lp_arr)
    # null: shifted stage matching within same run
    shift_corrs = []
    for s in common:
        i = steps_lp[s]
        # match uni at next common step if any
        idx = common.index(s)
        if idx + 1 >= len(common):
            continue
        j = steps_uni[common[idx + 1]]
        m = np.isfinite(a_lp.learning_progress[i]) & np.isfinite(
            a_uni.learning_progress[j]
        )
        if m.sum() < 20:
            continue
        shift_corrs.append(
            float(
                np.corrcoef(
                    a_lp.learning_progress[i][m], a_uni.learning_progress[j][m]
                )[0, 1]
            )
        )
    null_shift = float(np.median(shift_corrs)) if shift_corrs else 0.0

    # early vs late by step
    mid = np.median(steps_out) if steps_out else 0
    early = [c for c, s in zip(lp_corrs, steps_out) if s <= mid]
    late = [c for c, s in zip(lp_corrs, steps_out) if s > mid]

    return {
        "lp_cross_arm": _metric(
            med,
            null_shift,
            pass_=bool(lo > null_shift) if np.isfinite(lo) else False,
            ci=(lo, hi),
            extra={
                "interpretation": "lower bound: policies diverge over time",
                "n_matched_stages": len(lp_corrs),
                "per_step": [
                    {"step": int(s), "corr": float(c)}
                    for s, c in zip(steps_out, lp_corrs)
                ],
                "early_median": float(np.median(early)) if early else None,
                "late_median": float(np.median(late)) if late else None,
            },
        ),
        "performance_cross_arm": _metric(
            float(np.median(perf_arr)),
            0.0,
            pass_=bool(np.median(perf_arr) > 0.5) if np.isfinite(perf_arr).any() else None,
            ci=_median_ci(perf_arr)[1:],
            extra={"note": "performance field should correlate higher than LP"},
        ),
        "null": "shifted-stage cross-arm matching",
    }


# ---------------------------------------------------------------------------
# Full scorecard for one run
# ---------------------------------------------------------------------------
def scorecard_for_run(
    a: atlas_mod.Atlas,
    *,
    bank: ValidationBank | None = None,
    pair_atlas: atlas_mod.Atlas | None = None,
) -> dict[str, Any]:
    sc: dict[str, Any] = {
        "run_id": a.run_id,
        "inventory": a.to_inventory(),
        "A1_reliability": compute_alpha_series(a),
        "A3_acf": compute_acf(a),
        "A5_topk": topk_overlap(a),
    }
    if bank is not None:
        sc["A4_criterion_validity"] = criterion_validity(a, bank)
        sc["A7_saturation"] = saturation_headroom(bank)
    else:
        sc["A4_criterion_validity"] = {
            "unavailable": "validation bank not on disk for this run"
        }
        sc["A7_saturation"] = {
            "unavailable": "validation bank not on disk for this run"
        }
    if pair_atlas is not None:
        sc["C2_cross_arm"] = cross_arm_reliability(a, pair_atlas)
    return sc
