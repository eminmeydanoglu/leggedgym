"""Decompose Var(LP_observed) into measurable components (B1–B5).

Components that cannot be fully identified get honest upper/lower bounds.
Never invent shares that sum exactly to 100% when residual is unallocated.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from . import atlas as atlas_mod
from .lp_diagnostics import alpha_sem, alpha_temporal
from .validation_bank import ValidationBank, within_cell_command_r2


def _safe_var(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.var(x, ddof=1))


def panel_variance_decomposition(a: atlas_mod.Atlas) -> dict[str, Any]:
    """Two-way FE variance shares on performance[stage, cell] (B2).

    performance = stage_FE + cell_FE + residual
    residual includes interaction + pure noise.

    Pure stage FE does not change cross-sectional ranking; heterogeneous
    common-mode lives in residual after cell FE that co-moves across cells.
    """
    P = a.performance.copy()
    # drop rows/cols with all nan
    ok_t = np.isfinite(P).sum(axis=1) >= 10
    P = P[ok_t]
    if P.shape[0] < 3:
        return {"unavailable": "too few stages"}
    # fill remaining nan with cell mean
    for c in range(P.shape[1]):
        col = P[:, c]
        m = np.isfinite(col)
        if m.any():
            col = col.copy()
            col[~m] = np.nanmean(col)
            P[:, c] = col
    if not np.isfinite(P).all():
        P = np.nan_to_num(P, nan=float(np.nanmean(P)))

    grand = float(P.mean())
    stage_means = P.mean(axis=1, keepdims=True)
    cell_means = P.mean(axis=0, keepdims=True)
    stage_fe = stage_means - grand
    cell_fe = cell_means - grand
    fitted = grand + stage_fe + cell_fe
    resid = P - fitted
    # interaction approximation: residual after additive FE
    ss_total = float(np.sum((P - grand) ** 2))
    ss_stage = float(np.sum(stage_fe ** 2) * P.shape[1])
    ss_cell = float(np.sum(cell_fe ** 2) * P.shape[0])
    ss_resid = float(np.sum(resid ** 2))
    # normalize
    shares = {
        "stage_fe": ss_stage / ss_total if ss_total else np.nan,
        "cell_fe": ss_cell / ss_total if ss_total else np.nan,
        "residual_interaction_noise": ss_resid / ss_total if ss_total else np.nan,
    }
    # LP is ΔP; stage FE cancels in difference of consecutive stages for ranking
    # but stage-to-stage residual co-movement is common-mode noise for LP
    dP = np.diff(P, axis=0)
    # common-mode: mean ΔP across cells per stage-pair
    cm = dP.mean(axis=1, keepdims=True)
    demeaned = dP - cm
    var_lp = float(np.var(dP, ddof=1))
    var_cm = float(np.var(cm * np.ones_like(dP), ddof=1)) if dP.size else np.nan
    # better: share of cross-sectional variance that is common (0 by demeaning)
    # Fraction of total LP variance attributable to common shift:
    var_common_shift = float(np.mean(cm ** 2))
    var_idiosyncratic = float(np.mean(demeaned ** 2))
    total = var_common_shift + var_idiosyncratic
    return {
        "performance_panel": {
            "ss_shares": {k: float(v) for k, v in shares.items()},
            "ss_total": ss_total,
            "n_stages": int(P.shape[0]),
            "n_cells": int(P.shape[1]),
        },
        "lp_common_mode": {
            "var_common_shift": var_common_shift,
            "var_idiosyncratic": var_idiosyncratic,
            "share_common": float(var_common_shift / total) if total > 0 else None,
            "share_idiosyncratic": float(var_idiosyncratic / total) if total > 0 else None,
            "note": (
                "common shift does not affect cross-sectional ranking; "
                "heterogeneous policy noise sits in idiosyncratic residual"
            ),
        },
    }


def episode_sampling_share(a: atlas_mod.Atlas) -> dict[str, Any]:
    """B1: E[LP_SE^2] / Var(LP) — independent sampling noise fraction."""
    if not a.has_sem:
        return {
            "unavailable": "performance_sem missing",
            "component": "B1_episode_sampling",
        }
    shares = []
    for t in range(1, a.n_frames):
        m = a.defined(t, require_eligible=a.has_eligible)
        if m.sum() < 10:
            continue
        lp = a.learning_progress[t][m]
        se = a.lp_se(t)[m]
        v = _safe_var(lp)
        if not np.isfinite(v) or v <= 0:
            continue
        shares.append(float(np.mean(se ** 2) / v))
    if not shares:
        return {"unavailable": "no valid stages", "component": "B1"}
    arr = np.array(shares)
    # clip to [0,1] for interpretation; >1 means SEM overstates (known anomaly)
    return {
        "component": "B1_episode_sampling",
        "mean_E_se2_over_var": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "frac_stages_se2_exceeds_var": float(np.mean(arr > 1.0)),
        "note": (
            "share >1 implies performance_sem overstates LP noise "
            "(command heterogeneity double-counted in LP_SE; §11.7 anomaly)"
        ),
        "upper_bound_on_independent_noise": float(np.median(np.clip(arr, 0, 1))),
    }


def censoring_noise(a: atlas_mod.Atlas) -> dict[str, Any]:
    """B4: stage censoring rate and its correlation with performance (§11.1)."""
    if a.n_frames < 3:
        return {"unavailable": "too few frames"}
    # completions cumulative → per-stage
    completions = np.diff(a.task_completion_count, axis=0)
    admitted = a.stage_episode_count[1:]
    # if stage_episode_count is absolute per stage (not cumulative), use as is
    # check: admitted should be positive and smaller than completions typically
    late = completions - admitted
    # guard negative late (schema where counts are per-stage not cumulative)
    if np.nanmedian(late) < 0:
        # task_completion_count may also be per-stage
        completions = a.task_completion_count[1:]
        late = completions - admitted
        late = np.clip(late, 0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_stage_frac = late.sum(axis=1) / np.maximum(completions.sum(axis=1), 1)
    per_stage_frac = per_stage_frac[np.isfinite(per_stage_frac)]
    late_fraction = late / np.maximum(completions, 1)
    # correlate cell mean late frac with performance
    lf = np.nanmean(late_fraction, axis=0)
    perf = np.nanmean(a.performance[1:], axis=0)
    ok = np.isfinite(lf) & np.isfinite(perf)
    corr = float(np.corrcoef(lf[ok], perf[ok])[0, 1]) if ok.sum() > 5 else float("nan")

    # variance contribution: stage-to-stage change in late_frac injects noise
    # into Δ performance composition. Bound: var(Δ late_frac) * scale
    if late_fraction.shape[0] >= 2:
        d_lf = np.diff(np.nanmean(late_fraction, axis=1))
        var_dlf = float(np.var(d_lf, ddof=1)) if d_lf.size > 1 else float("nan")
    else:
        var_dlf = float("nan")

    return {
        "component": "B4_censoring",
        "discarded_fraction_median": float(np.median(per_stage_frac))
        if per_stage_frac.size
        else None,
        "discarded_fraction_iqr": [
            float(np.percentile(per_stage_frac, 25)),
            float(np.percentile(per_stage_frac, 75)),
        ]
        if per_stage_frac.size
        else None,
        "corr_late_fraction_performance": corr,
        "null_corr": 0.0,
        "pass_performance_blind": bool(abs(corr) < 0.1) if np.isfinite(corr) else None,
        "var_delta_stage_late_frac": var_dlf,
        "note": "bias magnitude (admitted vs all-completions mean) not identifiable from atlas",
    }


def feedback_noise_contrast(
    a_lp: atlas_mod.Atlas | None, a_uni: atlas_mod.Atlas | None
) -> dict[str, Any]:
    """B5: feedback-driven noise only on LP arms — contrast LP vs UNI budgets."""
    if a_lp is None or a_uni is None:
        return {"unavailable": "need both LP and UNI arms"}
    # compare median α_temporal (available without SEM) and ESS
    def ess_series(a):
        vals = []
        for d in a.diagnostics:
            e = d.get("effective_sample_size")
            if e is not None:
                vals.append(float(e))
        return np.array(vals, float) if vals else np.array([np.nan])

    def lag1_med(a):
        corrs = []
        for t in range(1, a.n_frames - 1):
            m = np.isfinite(a.learning_progress[t]) & np.isfinite(
                a.learning_progress[t + 1]
            )
            if m.sum() < 10:
                continue
            corrs.append(
                float(
                    np.corrcoef(
                        a.learning_progress[t][m], a.learning_progress[t + 1][m]
                    )[0, 1]
                )
            )
        return float(np.median(corrs)) if corrs else float("nan")

    ess_lp, ess_uni = ess_series(a_lp), ess_series(a_uni)
    # N heterogeneity: CV of stage_episode_count
    def n_cv(a):
        cvs = []
        for t in range(a.n_frames):
            n = a.stage_episode_count[t]
            n = n[np.isfinite(n) & (n > 0)]
            if n.size < 10:
                continue
            cvs.append(float(np.std(n) / (np.mean(n) + 1e-12)))
        return float(np.median(cvs)) if cvs else float("nan")

    return {
        "component": "B5_feedback",
        "ess_median_lp": float(np.nanmedian(ess_lp)),
        "ess_median_uni": float(np.nanmedian(ess_uni)),
        "n_cv_lp": n_cv(a_lp),
        "n_cv_uni": n_cv(a_uni),
        "lag1_lp": lag1_med(a_lp),
        "lag1_uni": lag1_med(a_uni),
        "alpha_temporal_lp": float(alpha_temporal(lag1_med(a_lp)))
        if np.isfinite(lag1_med(a_lp))
        else None,
        "alpha_temporal_uni": float(alpha_temporal(lag1_med(a_uni)))
        if np.isfinite(lag1_med(a_uni))
        else None,
        "note": (
            "If α_temporal is similar on UNI (no feedback) and LP, "
            "feedback is not the dominant noise source; task nature is."
        ),
    }


def command_heterogeneity_budget(
    bank: ValidationBank | None, a: atlas_mod.Atlas | None = None
) -> dict[str, Any]:
    """B3: direct (validation bank) + indirect (vx_bin width natural experiment)."""
    out: dict[str, Any] = {"component": "B3_command_heterogeneity"}
    if bank is not None:
        # all checkpoints linear + last nonlinear
        linear = []
        for i in range(len(bank.iterations)):
            linear.append(within_cell_command_r2(bank, i, nonlinear=False))
        nonlinear_last = within_cell_command_r2(bank, -1, nonlinear=True)
        # also early checkpoint
        nonlinear_early = within_cell_command_r2(bank, 0, nonlinear=True)
        r2s = [r["r2_total"] for r in linear]
        out["direct_validation"] = {
            "r2_linear_per_ckpt": linear,
            "r2_linear_mean": float(np.mean(r2s)),
            "r2_linear_ci95": [
                float(np.percentile(r2s, 2.5)),
                float(np.percentile(r2s, 97.5)),
            ]
            if len(r2s) >= 2
            else [r2s[0], r2s[0]],
            "r2_nonlinear_last": nonlinear_last,
            "r2_nonlinear_early": nonlinear_early,
            "null_r2": 0.0,
            "verdict": (
                "vy/yaw fixation would remove only ~"
                f"{100*float(np.mean(r2s)):.1f}% of within-cell variance (linear mean); "
                "hypothesis 'fix vy/yaw solves noise' is weak on this bank"
                if np.mean(r2s) < 0.15
                else "commands explain substantial within-cell variance"
            ),
        }
    else:
        out["direct_validation"] = {
            "unavailable": "validation bank not on disk"
        }

    # indirect: vx_bin 0 is 0.3 m/s wide, others 0.5 m/s
    if a is not None and a.has_sem and a.n_cells == 84:
        n_terrain = a.n_terrain
        vx = np.arange(a.n_cells) // n_terrain
        # compare mean performance_sem of vx0 vs others, N-controlled
        rows = []
        for t in range(a.n_frames):
            for c in range(a.n_cells):
                n = a.stage_episode_count[t, c]
                sem = a.performance_sem[t, c]
                if not (np.isfinite(n) and np.isfinite(sem) and n > 5):
                    continue
                rows.append((vx[c], n, sem, c % n_terrain))
        if len(rows) > 50:
            arr = np.array(rows, float)
            # residualize SEM on 1/sqrt(N) then compare vx0 vs rest
            inv_sqrt_n = 1.0 / np.sqrt(arr[:, 1])
            # simple residual
            X = np.column_stack([np.ones(len(arr)), inv_sqrt_n])
            beta, *_ = np.linalg.lstsq(X, arr[:, 2], rcond=None)
            resid = arr[:, 2] - X @ beta
            m0 = arr[:, 0] == 0
            mean0 = float(np.mean(resid[m0]))
            mean1 = float(np.mean(resid[~m0]))
            # bootstrap CI on difference
            rng = np.random.default_rng(0)
            diffs = []
            idx0 = np.where(m0)[0]
            idx1 = np.where(~m0)[0]
            for _ in range(200):
                s0 = rng.choice(resid[idx0], size=len(idx0), replace=True)
                s1 = rng.choice(resid[idx1], size=len(idx1), replace=True)
                diffs.append(float(np.mean(s0) - np.mean(s1)))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out["indirect_vx_band_width"] = {
                "resid_sem_vx0_minus_others": mean0 - mean1,
                "ci95": [float(lo), float(hi)],
                "null": 0.0,
                "pass_narrower_band_lower_sem": bool(hi < 0),
                "note": (
                    "if command heterogeneity inflates SEM, vx_bin0 (narrower) "
                    "should have lower residual SEM"
                ),
            }
        else:
            out["indirect_vx_band_width"] = {"unavailable": "insufficient rows"}
    else:
        out["indirect_vx_band_width"] = {
            "unavailable": "need full-schema atlas with SEM and 84 cells"
        }
    return out


def noise_budget_for_run(
    a: atlas_mod.Atlas,
    *,
    bank: ValidationBank | None = None,
    pair_uni: atlas_mod.Atlas | None = None,
) -> dict[str, Any]:
    """Assemble component estimates / bounds for one run."""
    b1 = episode_sampling_share(a)
    b2 = panel_variance_decomposition(a)
    b3 = command_heterogeneity_budget(bank, a)
    b4 = censoring_noise(a)
    b5 = feedback_noise_contrast(a, pair_uni) if pair_uni is not None else {
        "unavailable": "no UNI pair for this run",
        "component": "B5",
    }

    # stacked summary with bounds (not forced to sum to 1)
    summary_shares: dict[str, Any] = {}
    if "upper_bound_on_independent_noise" in b1:
        summary_shares["B1_independent_sampling_ub"] = b1[
            "upper_bound_on_independent_noise"
        ]
        summary_shares["B1_raw_mean_se2_var"] = b1["mean_E_se2_over_var"]
    if "lp_common_mode" in b2 and b2["lp_common_mode"].get("share_common") is not None:
        summary_shares["B2_common_mode_share"] = b2["lp_common_mode"]["share_common"]
        summary_shares["B2_idiosyncratic_share"] = b2["lp_common_mode"][
            "share_idiosyncratic"
        ]
    if "direct_validation" in b3 and "r2_linear_mean" in b3.get("direct_validation", {}):
        summary_shares["B3_command_r2_within_cell"] = b3["direct_validation"][
            "r2_linear_mean"
        ]
    if "discarded_fraction_median" in b4:
        summary_shares["B4_censor_fraction"] = b4["discarded_fraction_median"]

    return {
        "run_id": a.run_id,
        "B1": b1,
        "B2": b2,
        "B3": b3,
        "B4": b4,
        "B5": b5,
        "summary_bounds": summary_shares,
        "caveat": (
            "Components are not a complete partition; B1 upper-bounds independent "
            "noise (SEM double-counts command heterogeneity); B3 is within-cell "
            "R² not LP variance share; residual policy noise may dominate."
        ),
    }
