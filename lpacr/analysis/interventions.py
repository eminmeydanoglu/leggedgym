"""Rank interventions by measured expected gain from offline data (D1–D7).

Each intervention returns measured gain on α / horizon / decision-relevant
metric, or ``not_measurable`` with reason.  No speculation presented as number.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from . import atlas as atlas_mod
from .lp_diagnostics import alpha_sem, alpha_temporal
from .scorecard import compute_acf, compute_alpha_series, _median_ci
from .validation_bank import ValidationBank, within_cell_command_r2


def _alpha_temporal_from_atlas(a: atlas_mod.Atlas) -> dict[str, float]:
    sc = compute_alpha_series(a)
    at = sc["alpha_temporal"]
    return {
        "early": at["early"].get("value"),
        "late": at["late"].get("value"),
        "all": at["all"].get("value"),
    }


def _merge_stages(a: atlas_mod.Atlas, k: int) -> atlas_mod.Atlas:
    """Merge every k consecutive stages (N-weighted performance, combined SEM).

    Offline synthetic longer stages (D2).  Bootstrap already dropped.
    """
    if k <= 1:
        return a
    n = a.n_frames
    n_out = n // k
    if n_out < 2:
        # return empty-ish: keep original with note via same object
        return a

    def merge_metric(M, N, combine="mean"):
        out = []
        for i in range(n_out):
            sl = slice(i * k, (i + 1) * k)
            block = M[sl]
            w = N[sl]
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            wsum = w.sum(axis=0, keepdims=True)
            wsum = np.where(wsum > 0, wsum, 1.0)
            # weighted mean per cell
            num = np.nansum(np.where(np.isfinite(block), block * w, 0.0), axis=0)
            out.append(num / wsum.ravel())
        return np.array(out)

    N = a.stage_episode_count
    perf = merge_metric(a.performance, N)
    # SEM of weighted mean: sqrt(sum w_i^2 sem_i^2) / sum w
    sem_out = []
    for i in range(n_out):
        sl = slice(i * k, (i + 1) * k)
        w = np.where(np.isfinite(N[sl]) & (N[sl] > 0), N[sl], 0.0)
        wsum = w.sum(axis=0)
        wsum = np.where(wsum > 0, wsum, 1.0)
        if a.has_sem:
            se = a.performance_sem[sl]
            se = np.where(np.isfinite(se), se, 0.0)
            merged_se = np.sqrt(np.sum((w * se) ** 2, axis=0)) / wsum
        else:
            merged_se = np.full(a.n_cells, np.nan)
        sem_out.append(merged_se)
    sem = np.array(sem_out)
    n_merged = []
    for i in range(n_out):
        sl = slice(i * k, (i + 1) * k)
        n_merged.append(np.nansum(N[sl], axis=0))
    n_merged = np.array(n_merged)
    # recompute LP
    lp = np.full_like(perf, np.nan)
    lp[0] = np.nan
    if n_out > 1:
        lp[1:] = perf[1:] - perf[:-1]
    sp = merge_metric(a.sampling_probability, np.ones_like(N))
    # diagnostics: subsample every k-th
    diag = [a.diagnostics[i * k + k - 1] for i in range(n_out)]
    step = np.array([int(a.step[i * k + k - 1]) for i in range(n_out)], dtype=np.int64)

    return atlas_mod.Atlas(
        step=step,
        performance=perf,
        performance_sem=sem,
        learning_progress=lp,
        eligible=np.ones_like(perf, dtype=bool)
        if not a.has_eligible
        else np.ones_like(perf, dtype=bool),
        stage_episode_count=n_merged,
        previous_stage_episode_count=np.vstack(
            [np.full(a.n_cells, np.nan), n_merged[:-1]]
        )
        if n_out > 1
        else n_merged,
        sampling_probability=sp,
        task_assignment_count=merge_metric(a.task_assignment_count, np.ones_like(N)),
        task_completion_count=merge_metric(a.task_completion_count, np.ones_like(N)),
        diagnostics=diag,
        vx_labels=a.vx_labels,
        terrain_labels=a.terrain_labels,
        run_id=f"{a.run_id}_merge{k}x",
        available_fields=a.available_fields,
        missing_fields=a.missing_fields,
        unavailable_analyses=a.unavailable_analyses,
        raw_frame_count_including_bootstrap=n_out,
        source_paths=a.source_paths,
    )


def intervention_more_episodes(a: atlas_mod.Atlas) -> dict[str, Any]:
    """D1: extrapolate α / excess z>2 under k× episode budget (uses SEM if present)."""
    if not a.has_sem:
        return {
            "id": "D1_more_episodes",
            "measured_gain": None,
            "not_measurable": "performance_sem missing — cannot scale LP_SE with N",
            "cost": "linear wall-clock / sample cost",
        }
    # late regime: sigma_signal^2 = mean(LP^2) - mean(LP_SE^2)
    lps, ses, ns = [], [], []
    for t in range(1, a.n_frames):
        # late: stage pair index >= 15
        if t - 1 < atlas_mod.EARLY_STAGES.stop:
            continue
        m = a.defined(t)
        if m.sum() < 10:
            continue
        lps.append(a.learning_progress[t][m])
        ses.append(a.lp_se(t)[m])
        ns.append(a.stage_episode_count[t][m])
    if not lps:
        return {
            "id": "D1_more_episodes",
            "measured_gain": None,
            "not_measurable": "no late-regime stages",
        }
    LP = np.concatenate(lps)
    SE = np.concatenate(ses)
    N = np.concatenate(ns)
    sig2 = float(np.mean(LP ** 2) - np.mean(SE ** 2))
    # k× budget: SE' = SE / sqrt(k), z' = |LP|/(SE/sqrt(k)) if LP fixed —
    # but under pure noise LP also shrinks; under no signal, excess z>2 stays ~0
    def excess_z2(k: float) -> float:
        # Pure noise: LP and SE scale together ⇒ z invariant in k.
        # Optimistic upper bound (signal fixed): shrink SE only.
        if sig2 <= 0:
            z = np.abs(LP) / (SE + 1e-12)
            return float(np.mean(z > 2) - 0.0455)
        se_k = SE / np.sqrt(k)
        z = np.abs(LP) / (se_k + 1e-12)
        return float(np.mean(z > 2) - 0.0455)

    table = {
        f"{k}x": {
            "excess_P_z_gt_2": excess_z2(k),
            "model": "invariant_z" if sig2 <= 0 else "optimistic_fixed_LP",
        }
        for k in (1, 2, 4, 8)
    }
    # cells gained at 8x vs 1x
    gain_8x_cells = (
        table["8x"]["excess_P_z_gt_2"] - table["1x"]["excess_P_z_gt_2"]
    ) * 84
    return {
        "id": "D1_more_episodes",
        "sigma_signal2_late": sig2,
        "table_excess_z2": table,
        "measured_gain": {
            "metric": "delta_excess_cells_z_gt_2_per_stage_1x_to_8x",
            "value": float(gain_8x_cells),
            "null": 0.0,
            "note": (
                "if sig2<=0, z is N-invariant so gain≈0; "
                "if sig2>0, optimistic bound keeps |LP| fixed while SE shrinks (§11.7)"
            ),
        },
        "cost": "8× episodes per stage ≈ 8× stage wall time or 8× parallel envs",
        "decision": (
            "do_not_prioritize"
            if sig2 <= 0.02
            else "consider_if_early_regime"
        ),
    }


def intervention_longer_stage(a: atlas_mod.Atlas) -> dict[str, Any]:
    """D2: offline merge 2×/3×/4× stages; recompute α_temporal and ACF horizon."""
    base_at = _alpha_temporal_from_atlas(a)
    base_acf = compute_acf(a, max_lag=3)
    base_excess = None
    if base_acf["lags"] and base_acf["lags"][0].get("excess_over_null") is not None:
        base_excess = base_acf["lags"][0]["excess_over_null"]

    curve = []
    for k in (1, 2, 3, 4):
        ak = _merge_stages(a, k) if k > 1 else a
        if ak.n_frames < 4:
            curve.append(
                {
                    "merge_k": k,
                    "stage_steps": 2000 * k,
                    "not_measurable": f"only {ak.n_frames} merged stages",
                }
            )
            continue
        at = _alpha_temporal_from_atlas(ak)
        acf = compute_acf(ak, max_lag=3)
        excess1 = (
            acf["lags"][0].get("excess_over_null")
            if acf["lags"]
            else None
        )
        # response delay also grows with k: actionable = α * ACF(delay)
        # delay in original stages ≈ 1 merged stage = k original stages
        # usable signal ≈ α_temporal * excess_at_lag_corresponding_to_delay
        # For merged, lag-1 is already one long stage
        actionable = None
        if at.get("all") is not None and excess1 is not None:
            actionable = float(at["all"]) * max(float(excess1), 0.0)
        # vs baseline: baseline actionable ≈ α * excess_lag1 but delay is 1 short stage
        curve.append(
            {
                "merge_k": k,
                "stage_control_steps": 2000 * k,
                "approx_iters": 83 * k,
                "alpha_temporal_all": at.get("all"),
                "alpha_temporal_early": at.get("early"),
                "alpha_temporal_late": at.get("late"),
                "acf_lag1_excess": excess1,
                "actionable_proxy_alpha_x_excess": actionable,
                "n_merged_stages": ak.n_frames,
            }
        )

    # best k by actionable proxy among measurable
    measurable = [
        c for c in curve if c.get("actionable_proxy_alpha_x_excess") is not None
    ]
    best = None
    if measurable:
        best = max(measurable, key=lambda c: c["actionable_proxy_alpha_x_excess"])

    return {
        "id": "D2_longer_stage",
        "baseline_alpha_temporal": base_at,
        "baseline_lag1_excess": base_excess,
        "curve": curve,
        "measured_gain": {
            "metric": "actionable_proxy_alpha_x_excess",
            "value": best["actionable_proxy_alpha_x_excess"] if best else None,
            "best_k": best["merge_k"] if best else None,
            "best_value": best["actionable_proxy_alpha_x_excess"] if best else None,
            "baseline_value": (
                (base_at.get("all") or 0) * max(base_excess or 0, 0)
                if base_at.get("all") is not None
                else None
            ),
            "null": 0.0,
        },
        "cost": "longer stage ⇒ slower curriculum updates (delay scales with k)",
        "decision": "measure_on_curve_optimum_may_be_interior",
    }


def intervention_fix_vy_yaw(bank: ValidationBank | None) -> dict[str, Any]:
    """D3: from B3 — expected SEM reduction if commands residualized / fixed."""
    if bank is None:
        return {
            "id": "D3_fix_vy_yaw",
            "measured_gain": None,
            "not_measurable": "validation bank only on run #1; not on disk for other runs",
            "cost": "policy no longer trained on full vy/yaw distribution (distribution shift on holdout)",
        }
    r = within_cell_command_r2(bank, -1, nonlinear=True)
    r2 = r["r2_total"]
    sem_factor = r.get("sem_reduction_factor")
    return {
        "id": "D3_fix_vy_yaw",
        "measured_gain": {
            "metric": "within_cell_spnte_var_fraction_explained_by_commands",
            "value": float(r2),
            "null": 0.0,
            "sem_reduction_factor_if_residualized": sem_factor,
            "ci_note": "point estimate at final checkpoint; see B3 for all ckpts",
        },
        "cost": (
            "fixing vy/yaw shrinks training support; validation bank still "
            "samples full range ⇒ train/test command shift"
        ),
        "decision": "low_priority_if_r2_below_0.15",
        "detail": r,
    }


def intervention_residualization(bank: ValidationBank | None) -> dict[str, Any]:
    """D4: residualize return on commands — ceiling from validation bank."""
    if bank is None:
        return {
            "id": "D4_covariate_residualization",
            "measured_gain": None,
            "not_measurable": (
                "atlas has no per-episode commands; ceiling simulated on "
                "validation bank only (run #1)"
            ),
        }
    r = within_cell_command_r2(bank, -1, nonlinear=True)
    # LP_SE reduction if independent noise shrinks by (1-R2)
    factor = float(np.sqrt(max(1.0 - r["r2_total"], 0.0)))
    return {
        "id": "D4_covariate_residualization",
        "measured_gain": {
            "metric": "lp_se_multiplicative_factor_ceiling",
            "value": factor,
            "null": 1.0,
            "r2_commands": float(r["r2_total"]),
            "note": (
                "ceiling from held-out spnte; training-return residualization "
                "may differ. factor=1 means no reduction"
            ),
        },
        "cost": "instrumentation only (log per-episode commands) — already partially added",
        "decision": "cheap_to_try_but_ceiling_low_if_r2_small",
    }


def intervention_pooling(a: atlas_mod.Atlas) -> dict[str, Any]:
    """D5: factorised pooling — α_SEM vs α_temporal on vx / terrain margins."""
    rows = []
    for axis, label in (("vx", "vx_band"), ("terrain", "terrain_cell")):
        g = a.group_index(axis)
        k = int(g.max()) + 1
        pooled_p = np.zeros((a.n_frames, k))
        pooled_se = np.zeros((a.n_frames, k))
        for t in range(a.n_frames):
            for j in range(k):
                m = (g == j) & np.isfinite(a.performance[t]) & (
                    a.stage_episode_count[t] > 0
                )
                if not m.any():
                    continue
                w = a.stage_episode_count[t][m]
                w = w / (w.sum() + 1e-12)
                pooled_p[t, j] = (w * a.performance[t][m]).sum()
                if a.has_sem:
                    pooled_se[t, j] = np.sqrt(
                        (w ** 2 * a.performance_sem[t][m] ** 2).sum()
                    )
        lp = pooled_p[1:] - pooled_p[:-1]
        lag1 = []
        for t in range(len(lp) - 1):
            if np.isfinite(lp[t]).sum() < max(3, k // 2):
                continue
            lag1.append(float(np.corrcoef(lp[t], lp[t + 1])[0, 1]))
        r = float(np.median(lag1)) if lag1 else float("nan")
        a_temp = alpha_temporal(r) if np.isfinite(r) else float("nan")
        a_sem = float("nan")
        if a.has_sem:
            se = np.sqrt(pooled_se[1:] ** 2 + pooled_se[:-1] ** 2)
            late_start = atlas_mod.EARLY_STAGES.stop
            a_sems = []
            for t in range(len(lp)):
                if t < late_start:
                    continue
                if np.isfinite(lp[t]).sum() < 3:
                    continue
                a_sems.append(alpha_sem(lp[t], se[t]))
            a_sem = float(np.median(a_sems)) if a_sems else float("nan")
        rows.append(
            {
                "unit": label,
                "n_units": k,
                "alpha_sem_late": a_sem,
                "lag1": r,
                "alpha_temporal": a_temp,
            }
        )

    # cell-level reference
    cell_at = _alpha_temporal_from_atlas(a)
    return {
        "id": "D5_pooling_factorization",
        "pooled": rows,
        "cell_alpha_temporal_late": cell_at.get("late"),
        "measured_gain": {
            "metric": "alpha_temporal_change_from_pooling",
            "value": (
                float(rows[0]["alpha_temporal"] - (cell_at.get("late") or 0))
                if rows and cell_at.get("late") is not None
                and np.isfinite(rows[0]["alpha_temporal"])
                else None
            ),
            "null": 0.0,
            "note": (
                "α_SEM may jump; α_temporal is the valid gain metric (§11.3). "
                "Positive gain required to justify pooling."
            ),
        },
        "cost": "loses cell-level targeting; may be correct if interaction≈0",
        "decision": "do_not_pool_if_alpha_temporal_flat",
    }


def intervention_metric_swap(bank: ValidationBank | None) -> dict[str, Any]:
    """D6: which validation metric has longer temporal horizon as LP proxy."""
    if bank is None:
        return {
            "id": "D6_metric_swap",
            "measured_gain": None,
            "not_measurable": (
                "per-cell episode length not in V5 atlas (reward-per-step blocked); "
                "validation bank only on run #1 for metric comparison"
            ),
        }
    metrics = {
        "spnte_lin": bank.cell_spnte_lin,
        "fall_rate": bank.cell_fall_rate,
    }
    # also survival if available at cell level — approximate from measurements
    # build cell survival mean
    T = len(bank.iterations)
    C = bank.n_cells
    surv = np.full((T, C), np.nan)
    for t in range(T):
        for c in range(C):
            m = bank.cell_id[t] == c
            if m.any():
                surv[t, c] = np.nanmean(bank.survival_steps[t][m])
    metrics["survival_steps"] = surv

    rows = []
    for name, Y in metrics.items():
        # LP proxy = -Δ for spnte/fall (lower better), +Δ for survival
        sign = -1.0 if name in ("spnte_lin", "fall_rate") else 1.0
        lp = sign * np.diff(Y, axis=0)  # (T-1, C)
        # lag-1 corr of this LP across checkpoints
        corrs = []
        for t in range(lp.shape[0] - 1):
            m = np.isfinite(lp[t]) & np.isfinite(lp[t + 1])
            if m.sum() < 20:
                continue
            corrs.append(float(np.corrcoef(lp[t][m], lp[t + 1][m])[0, 1]))
        med = float(np.median(corrs)) if corrs else float("nan")
        # analytic null for adjacent Δ sharing window ≈ -0.5
        excess = med - (-0.5) if np.isfinite(med) else float("nan")
        rows.append(
            {
                "metric": name,
                "lag1_median": med,
                "null": -0.5,
                "excess": excess,
                "n_pairs": len(corrs),
            }
        )
    best = max(
        (r for r in rows if np.isfinite(r["excess"])),
        key=lambda r: r["excess"],
        default=None,
    )
    return {
        "id": "D6_metric_swap",
        "candidates": rows,
        "measured_gain": {
            "metric": "best_excess_lag1_over_minus_half",
            "best_metric": best["metric"] if best else None,
            "value": best["excess"] if best else None,
            "null": 0.0,
        },
        "cost": "estimator / reward engineering",
        "atlas_gap": "reward-per-step not reconstructible without per-cell episode length",
        "decision": "prefer_metric_with_longer_horizon",
    }


def intervention_estimator_swap(a: atlas_mod.Atlas) -> dict[str, Any]:
    """D7: offline EWMA / multi-stage regression LP vs stage Δ; race on horizon."""
    # stage LP is already in atlas; build EWMA performance then Δ
    if a.n_frames < 6:
        return {
            "id": "D7_estimator_swap",
            "measured_gain": None,
            "not_measurable": "need >=6 stages",
        }

    def horizon_excess(lp_mat: np.ndarray) -> dict[str, float]:
        """lp_mat shape (frames, cells); row 0 may be nan."""
        out = {}
        for lag, null in ((1, -0.5), (2, 0.0), (3, 0.0)):
            pe = []
            for t in range(1, lp_mat.shape[0] - lag):
                m = np.isfinite(lp_mat[t]) & np.isfinite(lp_mat[t + lag])
                if m.sum() < 20:
                    continue
                pe.append(float(np.corrcoef(lp_mat[t][m], lp_mat[t + lag][m])[0, 1]))
            med = float(np.median(pe)) if pe else float("nan")
            out[f"lag{lag}_excess"] = med - null if np.isfinite(med) else float("nan")
        return out

    # stage estimator
    stage = horizon_excess(a.learning_progress)

    # EWMA of performance (alpha=0.5) then difference
    ewma = np.full_like(a.performance, np.nan)
    ewma[0] = a.performance[0]
    alpha = 0.5
    for t in range(1, a.n_frames):
        ewma[t] = alpha * a.performance[t] + (1 - alpha) * ewma[t - 1]
    ewma_lp = np.full_like(ewma, np.nan)
    ewma_lp[1:] = ewma[1:] - ewma[:-1]
    ewma_h = horizon_excess(ewma_lp)

    # regression slope over last 3 stages as LP
    reg_lp = np.full_like(a.performance, np.nan)
    for t in range(2, a.n_frames):
        # slope of performance over t-2,t-1,t
        xs = np.array([0.0, 1.0, 2.0])
        for c in range(a.n_cells):
            ys = a.performance[t - 2 : t + 1, c]
            if np.all(np.isfinite(ys)):
                reg_lp[t, c] = float(np.polyfit(xs, ys, 1)[0])
    reg_h = horizon_excess(reg_lp)

    candidates = {
        "stage_delta": stage,
        "ewma_delta": ewma_h,
        "regression_slope_3": reg_h,
    }
    # win by lag2 excess (horizon beyond one stage)
    best_name, best_val = None, -1e9
    for name, h in candidates.items():
        v = h.get("lag2_excess")
        if v is not None and np.isfinite(v) and v > best_val:
            best_val = v
            best_name = name

    return {
        "id": "D7_estimator_swap",
        "candidates": candidates,
        "measured_gain": {
            "metric": "lag2_excess_over_null0",
            "best_estimator": best_name,
            "value": float(best_val) if best_name else None,
            "null": 0.0,
            "win_criterion": "horizon (lag2 excess), not lag1 correlation (§11.6)",
        },
        "cost": "code path already supports rolling_completion; offline proxy only",
        "decision": "switch_if_lag2_excess_improves",
    }


def rank_interventions(
    a: atlas_mod.Atlas,
    *,
    bank: ValidationBank | None = None,
) -> list[dict[str, Any]]:
    """Run all intervention measurements and return ranked list."""
    items = [
        intervention_more_episodes(a),
        intervention_longer_stage(a),
        intervention_fix_vy_yaw(bank),
        intervention_residualization(bank),
        intervention_pooling(a),
        intervention_metric_swap(bank),
        intervention_estimator_swap(a),
    ]
    # rank by whether measured and by absolute gain magnitude when comparable
    def sort_key(it):
        mg = it.get("measured_gain")
        if not mg or mg.get("value") is None:
            return (-1, 0.0)  # unmeasurable last among zeros... put at end
        return (1, abs(float(mg["value"])))

    # separate measurable
    meas = [i for i in items if i.get("measured_gain") and i["measured_gain"].get("value") is not None]
    unmeas = [i for i in items if i not in meas]
    meas.sort(key=lambda i: abs(float(i["measured_gain"]["value"])), reverse=True)
    for rank, it in enumerate(meas, 1):
        it["rank"] = rank
    for it in unmeas:
        it["rank"] = None
    return meas + unmeas
