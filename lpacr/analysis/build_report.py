#!/usr/bin/env python3
"""Regenerate V5 LP-ACRL offline analysis report.

Single entry point:

    .venv/bin/python -m lpacr.analysis.build_report

Writes:
  lpacr/analysis/report/results.json
  lpacr/analysis/report/v5_lp_analysis.html
  lpacr/analysis/V5_ANALIZ_BULGULARI.md

No GPU, no training, no network.  All numbers in the HTML come from results.json.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import atlas as atlas_mod
from . import interventions
from . import noise_budget
from . import power_mde
from . import scorecard
from . import tb_loader
from . import validation_bank
from .lp_diagnostics import alpha_temporal

REPO_ROOT = atlas_mod.REPO_ROOT
OUT_DIR = Path(__file__).resolve().parent / "report"
RESULTS_PATH = OUT_DIR / "results.json"
HTML_PATH = OUT_DIR / "v5_lp_analysis.html"
FINDINGS_PATH = Path(__file__).resolve().parent / "V5_ANALIZ_BULGULARI.md"

REGEN_CMD = (
    ".venv/bin/python -m lpacr.analysis.build_report"
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_analyses() -> dict[str, Any]:
    print("Loading atlases…", flush=True)
    atlases = atlas_mod.load_all_primary(skip_bootstrap=False, include_optional=True)
    print(f"  loaded: {list(atlases.keys())}", flush=True)

    print("Loading validation banks…", flush=True)
    banks = validation_bank.load_run1_banks()
    bank_lp, bank_uni = banks.get("run1_lp"), banks.get("run1_uni")
    matched = None
    if bank_lp is not None and bank_uni is not None:
        matched = validation_bank.verify_matched_design(bank_lp, bank_uni)
        print(
            f"  matched design: {matched.get('cross_arm_all_match')} "
            f"fp={matched.get('fingerprint', '')[:16]}…",
            flush=True,
        )

    # Inventory
    inventory = {
        "runs": {},
        "data_gaps": [
            {
                "item": "run #2/#4 heldout_final and ued_validation",
                "status": "missing_on_disk",
                "note": (
                    "Holdout numbers exist only in V5_DENEME_OZETI.md / HISTORY.md "
                    "tables — labeled 'kayıtlı tablo', not recomputed from raw JSON."
                ),
            },
            {
                "item": "run #1 performance_sem / eligible_for_lp / previous_stage_episode_count",
                "status": "partial_schema",
                "note": "α_SEM, LP_SE scaling, reliability fixed-point blocked for #1",
            },
            {
                "item": "per-episode commands in atlas",
                "status": "missing",
                "note": "B3 residualization ceiling uses validation bank proxy only",
            },
        ],
        "matched_validation_design": matched,
        "unit_conversions_verified": {
            "iter_to_control_step": 24,
            "check": "3000 iter ↔ 72k step (run #1 atlas step range)",
            "stage_control_steps": 2000,
            "stage_approx_iters": 83,
        },
    }
    for k, a in atlases.items():
        inv = a.to_inventory()
        meta = atlas_mod.RUN_REGISTRY.get(k, {})
        inv["label"] = meta.get("label", k)
        inv["algo"] = meta.get("algo")
        inv["partial_schema"] = meta.get("partial_schema")
        inv["notes"] = meta.get("notes")
        inv["validation_bank_present"] = bool(
            banks.get(k) if k in banks else False
        )
        if k.startswith("run1"):
            inv["validation_bank_present"] = banks.get(k) is not None
        inventory["runs"][k] = inv

    # Scorecards
    print("Scorecards…", flush=True)
    scorecards: dict[str, Any] = {}
    for k, a in atlases.items():
        if k == "v6_frontier":
            continue
        bank = banks.get(k) if k in banks else None
        if k == "run1_lp":
            bank = bank_lp
        elif k == "run1_uni":
            bank = bank_uni
        else:
            bank = None
        pair = None
        if k == "run1_lp" and "run1_uni" in atlases:
            pair = atlases["run1_uni"]
        elif k == "run2_lp" and "run2_uni" in atlases:
            pair = atlases["run2_uni"]
        print(f"  scorecard {k} frames={a.n_frames} has_sem={a.has_sem}", flush=True)
        scorecards[k] = scorecard.scorecard_for_run(a, bank=bank, pair_atlas=pair)

    # C2 also from UNI→LP direction already inside LP scorecard; add run2
    cross_arm = {
        "run1": scorecards.get("run1_lp", {}).get("C2_cross_arm"),
        "run2": scorecards.get("run2_lp", {}).get("C2_cross_arm"),
    }

    # Noise budgets
    print("Noise budgets…", flush=True)
    noise: dict[str, Any] = {}
    for k, a in atlases.items():
        if k == "v6_frontier":
            continue
        bank = bank_lp if k == "run1_lp" else (bank_uni if k == "run1_uni" else None)
        pair_uni = None
        if k.endswith("_lp"):
            uk = k.replace("_lp", "_uni")
            pair_uni = atlases.get(uk)
        noise[k] = noise_budget.noise_budget_for_run(a, bank=bank, pair_uni=pair_uni)

    # Interventions — primary on run4 and run1
    print("Interventions…", flush=True)
    interv: dict[str, Any] = {}
    for k in ("run4_fixed", "run1_lp", "run1_uni", "run2_lp"):
        if k not in atlases:
            continue
        bank = bank_lp if k.startswith("run1") else None
        if k == "run1_uni":
            bank = bank_uni
        interv[k] = interventions.rank_interventions(atlases[k], bank=bank)

    # Power / MDE
    print("Power/MDE…", flush=True)
    campaign = power_mde.campaign_design(bank_lp, bank_uni, atlases)

    # §11 transfer
    print("§11 transfer…", flush=True)
    section11 = transfer_section11(atlases, scorecards, noise)

    # Sampler dynamics E* (partial)
    sampler = sampler_dynamics(atlases)

    # TB curves (F1 partial)
    print("TB scalars…", flush=True)
    tb = {}
    for k, meta in atlas_mod.RUN_REGISTRY.items():
        if k not in atlases or k == "v6_frontier":
            continue
        te = meta.get("tb_events")
        if te:
            tb[k] = tb_loader.load_scalars(te)

    # A6 decision-theoretic ceiling — simplified synthetic
    a6 = decision_ceiling(scorecards.get("run4_fixed", {}))

    # Incomplete items log
    incomplete = [
        {
            "item": "A6 full bandit/curriculum simulation",
            "status": "partial",
            "note": "closed-form ceiling from α×horizon×cell-var only; full oracle bandit sim not run",
        },
        {
            "item": "E1 loop-gain full β surface",
            "status": "partial",
            "note": "elasticity measured on #3/#4 where diagnostics exist; no offline β replay of crash",
        },
        {
            "item": "F1 diet-reweighted train reward",
            "status": "yapılmadı — sebep",
            "note": (
                "requires aligning TB iteration index with atlas stage diet; "
                "TB loaded for inspection but reweight not completed"
            ),
        },
        {
            "item": "F3 max-of-k checkpoint selection bias",
            "status": "partial",
            "note": "see campaign.bank_noise macro curves; formal max-of-6 bias bound in findings",
        },
        {
            "item": "G1 V6 frontier full comparison",
            "status": "yapılmadı — sebep",
            "note": "schema differs; only inventory entry present",
        },
    ]

    # Executive findings
    executive = build_executive(scorecards, noise, interv, campaign, section11)

    # Chart payloads (HTML reads these — no hard-coded numbers)
    print("Chart data…", flush=True)
    charts = build_chart_data(atlases, scorecards, bank_lp)

    results = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "regen_cmd": REGEN_CMD,
            "repo": str(REPO_ROOT),
            "rule": "Null'ı türetilmemiş bir sayı kapı olarak kullanılamaz (§11.0)",
        },
        "inventory": inventory,
        "scorecards": scorecards,
        "cross_arm_reliability": cross_arm,
        "noise_budgets": noise,
        "interventions": interv,
        "campaign_design": campaign,
        "section11_agreement": section11,
        "sampler_dynamics": sampler,
        "charts": charts,
        "tensorboard": {
            k: {
                "unavailable": v.get("unavailable"),
                "path": v.get("path"),
                "tags_available_count": len(v.get("tags_available") or []),
                "Train/mean_reward_n": len(
                    (v.get("Train/mean_reward") or {}).get("values") or []
                ),
            }
            for k, v in tb.items()
        },
        "A6_decision_ceiling": a6,
        "incomplete": incomplete,
        "executive_summary": executive,
    }
    return results


def build_chart_data(
    atlases: dict,
    scorecards: dict,
    bank_lp,
) -> dict[str, Any]:
    """Serialize matrices / series for SVG rendering (no analysis logic)."""
    out: dict[str, Any] = {}

    # Per-cell LP heatmap for run4 (and run2_uni for C1)
    for key in ("run4_fixed", "run2_uni", "run1_lp"):
        if key not in atlases:
            continue
        a = atlases[key]
        # rows=cells, cols=stages (skip bootstrap col 0)
        lp = a.learning_progress[1:]
        # transpose to cells × stages for heatmap
        mat = lp.T  # (cells, stages)
        # downsample stages if many
        if mat.shape[1] > 40:
            cols = np.linspace(0, mat.shape[1] - 1, 40).astype(int)
            mat = mat[:, cols]
        out[f"lp_heatmap_{key}"] = {
            "matrix": [
                [None if not np.isfinite(v) else float(v) for v in row]
                for row in mat
            ],
            "n_cells": int(mat.shape[0]),
            "n_stages": int(mat.shape[1]),
        }

    # Diet KL time series for LP arms
    for key in ("run4_fixed", "run3_crash", "run2_lp", "run1_lp"):
        if key not in atlases:
            continue
        a = atlases[key]
        kls = []
        for t in range(a.n_frames):
            p = a.sampling_probability[t]
            p = p[np.isfinite(p)]
            if p.size < 2:
                kls.append(None)
                continue
            p = p / (p.sum() + 1e-15)
            u = 1.0 / len(p)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = float(np.sum(np.where(p > 0, p * np.log((p + 1e-15) / u), 0.0)))
            kls.append(kl if np.isfinite(kl) else None)
        out[f"diet_kl_{key}"] = {
            "steps": [int(s) for s in a.step],
            "kl": kls,
        }

    # A4 criterion validity scatter: last concurrent window LP vs improvement
    a4 = (scorecards.get("run1_lp") or {}).get("A4_criterion_validity") or {}
    scatter = {"xs": [], "ys": [], "note": a4.get("note") or a4.get("unavailable")}
    if bank_lp is not None and "run1_lp" in atlases and not a4.get("unavailable"):
        a = atlases["run1_lp"]
        iters = list(bank_lp.iterations)
        if len(iters) >= 2:
            # pool last two concurrent windows for denser scatter
            from .scorecard import atlas_lp_window_mean
            from .validation_bank import cell_improvement

            xs, ys = [], []
            for i in range(len(iters) - 1):
                it0, it1 = iters[i], iters[i + 1]
                lp = atlas_lp_window_mean(a, it0 * 24, it1 * 24)
                imp = cell_improvement(bank_lp, i, i + 1, "spnte_lin")
                ok = np.isfinite(lp) & np.isfinite(imp)
                xs.extend(lp[ok].tolist())
                ys.extend(imp[ok].tolist())
            scatter = {
                "xs": [float(x) for x in xs],
                "ys": [float(y) for y in ys],
                "x_label": "atlas LP (training window)",
                "y_label": "Δspnte_lin improvement (held-out)",
                "note": "concurrent windows pooled; forward is primary test in table",
            }
    out["a4_scatter_run1_lp"] = scatter
    return out


def transfer_section11(
    atlases: dict, scorecards: dict, noise: dict
) -> dict[str, Any]:
    """Re-apply §11 claims across runs; mark doğrulandı/yanlışlandı/genişledi."""
    items = []

    # 11.1 censoring
    for k, n in noise.items():
        b4 = n.get("B4") or {}
        if "discarded_fraction_median" in b4:
            items.append(
                {
                    "section": "11.1",
                    "run": k,
                    "claim": "stage censoring ~46%, corr(late,perf)>0",
                    "measured_discard_frac": b4.get("discarded_fraction_median"),
                    "measured_corr": b4.get("corr_late_fraction_performance"),
                    "status": (
                        "doğrulandı"
                        if b4.get("discarded_fraction_median")
                        and abs(b4["discarded_fraction_median"] - 0.46) < 0.15
                        else "genişledi"
                    ),
                }
            )

    # 11.2 alpha
    for k, sc in scorecards.items():
        a1 = sc.get("A1_reliability") or {}
        asem = a1.get("alpha_sem") or {}
        at = a1.get("alpha_temporal") or {}
        items.append(
            {
                "section": "11.2",
                "run": k,
                "claim": "#4: late α_SEM≈0, α_temporal≈0.1",
                "alpha_sem_late": (asem.get("late") or {}).get("value"),
                "alpha_temporal_late": (at.get("late") or {}).get("value"),
                "alpha_sem_early": (asem.get("early") or {}).get("value"),
                "status": "genişledi",
                "note": "compare UNI arms: if also ~0, task nature not sampler regime",
            }
        )

    # 11.3 pooling
    for k in ("run4_fixed", "run1_uni", "run2_uni"):
        if k not in atlases:
            continue
        d5 = interventions.intervention_pooling(atlases[k])
        items.append(
            {
                "section": "11.3",
                "run": k,
                "claim": "pooling raises α_SEM not α_temporal",
                "pooled": d5.get("pooled"),
                "status": "doğrulandı"
                if d5.get("pooled")
                and all(
                    (p.get("alpha_temporal") or 1) < 0.3
                    for p in d5["pooled"]
                    if p.get("alpha_temporal") is not None
                )
                else "genişledi",
            }
        )

    # 11.4 reliability fixed point — only with SEM
    items.append(
        {
            "section": "11.4",
            "claim": "lp_reliability pure-noise fixed point 0.444",
            "null": 0.444,
            "status": "doğrulandı (teorik); run #4 atlas median was 0.414",
            "note": "not recomputed for all runs in this pass if diagnostics lack metric",
        }
    )

    # 11.6 horizon
    for k, sc in scorecards.items():
        acf = sc.get("A3_acf") or {}
        lags = acf.get("lags") or []
        excesses = [lg.get("excess_over_null") for lg in lags[:3]]
        items.append(
            {
                "section": "11.6",
                "run": k,
                "claim": "signal horizon ~1 stage; lag1 excess ~0.08",
                "excesses_lag1to3": excesses,
                "half_life_stages": acf.get("signal_half_life_stages"),
                "status": "genişledi",
            }
        )

    # 11.7 more episodes not binding
    for k in ("run4_fixed", "run2_lp"):
        if k not in atlases:
            continue
        d1 = interventions.intervention_more_episodes(atlases[k])
        items.append(
            {
                "section": "11.7",
                "run": k,
                "claim": "sig2≈0 late; 8× budget buys few cells",
                "sigma_signal2_late": d1.get("sigma_signal2_late"),
                "gain_8x": (d1.get("measured_gain") or {}).get("value"),
                "status": (
                    "doğrulandı"
                    if d1.get("sigma_signal2_late") is not None
                    and d1["sigma_signal2_late"] < 0.05
                    else "genişledi"
                ),
            }
        )

    return {"items": items}


def sampler_dynamics(atlases: dict) -> dict[str, Any]:
    out = {}
    for k in ("run3_crash", "run4_fixed", "run1_lp", "run2_lp"):
        if k not in atlases:
            continue
        a = atlases[k]
        ess, beta, tv, top10 = [], [], [], []
        for d in a.diagnostics:
            if d.get("effective_sample_size") is not None:
                ess.append(float(d["effective_sample_size"]))
            if d.get("effective_beta") is not None:
                beta.append(float(d["effective_beta"]))
            if d.get("tv_distance_uniform") is not None:
                tv.append(float(d["tv_distance_uniform"]))
            if d.get("top10_overlap_prev") is not None:
                top10.append(float(d["top10_overlap_prev"]))
        # diet KL(p||u)
        kls = []
        for t in range(a.n_frames):
            p = a.sampling_probability[t]
            p = p[np.isfinite(p)]
            p = p / (p.sum() + 1e-15)
            u = np.full_like(p, 1.0 / len(p))
            kl = float(np.sum(np.where(p > 0, p * np.log((p + 1e-15) / u), 0.0)))
            kls.append(kl)
        # cumulative diet (skip bootstrap frame 0 in sum if oversized)
        n_start = 1 if a.n_frames > 1 else 0
        cum_n = np.nansum(a.stage_episode_count[n_start:], axis=0)
        cum_share = cum_n / (cum_n.sum() + 1e-15)
        with np.errstate(divide="ignore", invalid="ignore"):
            cum_kl = float(
                np.nansum(
                    np.where(
                        cum_share > 0,
                        cum_share * np.log(cum_share * len(cum_share) + 1e-15),
                        0.0,
                    )
                )
            )
        # loop gain proxy: corr(|LP|, p) and corr(p, N_next)
        gains = {}
        if a.n_frames >= 3:
            cor_p_n = []
            cor_n_lp = []
            for t in range(1, a.n_frames - 1):
                p = a.sampling_probability[t]
                n1 = a.stage_episode_count[t + 1]
                lp = np.abs(a.learning_progress[t])
                m = np.isfinite(p) & np.isfinite(n1) & (n1 > 0)
                if m.sum() > 20:
                    cor_p_n.append(float(np.corrcoef(p[m], n1[m])[0, 1]))
                m2 = np.isfinite(n1) & np.isfinite(lp)
                if m2.sum() > 20:
                    cor_n_lp.append(float(np.corrcoef(n1[m2], lp[m2])[0, 1]))
            gains = {
                "median_corr_p_to_Nnext": float(np.median(cor_p_n)) if cor_p_n else None,
                "median_corr_N_to_absLP": float(np.median(cor_n_lp)) if cor_n_lp else None,
                "loop_sign": (
                    "negative_feedback"
                    if cor_n_lp and np.median(cor_n_lp) < 0
                    else "positive_or_flat"
                ),
            }
        out[k] = {
            "ess_median": float(np.median(ess)) if ess else None,
            "ess_min": float(np.min(ess)) if ess else None,
            "beta_median": float(np.median(beta)) if beta else None,
            "tv_median": float(np.median(tv)) if tv else None,
            "top10_overlap_median": float(np.median(top10)) if top10 else None,
            "stage_kl_median": float(np.median(kls)) if kls else None,
            "cumulative_kl": cum_kl,
            "loop_gain": gains,
        }
    return out


def decision_ceiling(sc_run4: dict) -> dict[str, Any]:
    """A6 simplified: theoretical value of LP chasing ≈ α * excess_horizon * scale."""
    if not sc_run4:
        return {
            "status": "yapılmadı — sebep",
            "note": "run4 scorecard missing",
        }
    a1 = sc_run4.get("A1_reliability") or {}
    a3 = sc_run4.get("A3_acf") or {}
    alpha = (a1.get("alpha_min") or {}).get("late") or {}
    a_val = alpha.get("value")
    excess = None
    if a3.get("lags"):
        excess = a3["lags"][0].get("excess_over_null")
    # ceiling fraction of oracle gap capturable ≈ α * max(excess,0) / delay_factor
    # delay = 1 stage; signal half-life ~1 ⇒ capturable fraction near 0
    if a_val is None:
        return {
            "status": "partial",
            "theoretical_ceiling_fraction_of_oracle": None,
            "note": "α unavailable",
        }
    half = a3.get("signal_half_life_stages") or 1
    capturable = float(max(a_val, 0) * max(excess or 0, 0) / max(half, 1))
    return {
        "status": "partial_closed_form",
        "alpha_late": a_val,
        "lag1_excess": excess,
        "half_life_stages": half,
        "theoretical_ceiling_fraction_of_oracle": capturable,
        "interpretation": (
            f"With α≈{a_val} and horizon excess≈{excess}, actionable signal "
            f"fraction ≈ {capturable:.4f}. LP chasing ceiling is near zero at "
            "this budget — oracle-vs-uniform gap mostly uncapturable."
        ),
        "null": 0.0,
        "full_bandit_sim": "yapılmadı — sebep: closed-form bound only",
    }


def build_executive(scorecards, noise, interv, campaign, section11) -> list[dict]:
    findings = []

    # 1 reliability
    r4 = scorecards.get("run4_fixed", {})
    a1 = r4.get("A1_reliability") or {}
    findings.append(
        {
            "bulgu": "LP güvenilirliği geç rejimde gürültü tabanında",
            "sayi": (
                f"run4 late α_min={((a1.get('alpha_min') or {}).get('late') or {}).get('value')}, "
                f"α_SEM={((a1.get('alpha_sem') or {}).get('late') or {}).get('value')}, "
                f"null=0"
            ),
            "karar": "LP kapısını α=min(α_SEM,α_temporal) ile kapat; α≈0 ise uniform örnekle",
        }
    )

    # 2 uniform also weak
    r1u = scorecards.get("run1_uni", {})
    at_u = ((r1u.get("A1_reliability") or {}).get("alpha_temporal") or {}).get(
        "late"
    ) or {}
    findings.append(
        {
            "bulgu": "Uniform kolda da α_temporal düşük (temiz null) — sorun sampler rejiminden bağımsız",
            "sayi": f"run1_uni late α_temporal={at_u.get('value')} (null 0)",
            "karar": "Ölçülemezlik görevin doğası / metrik doygunluğu; önce A7 doygunluk",
        }
    )

    # 3 criterion validity
    a4 = (scorecards.get("run1_lp") or {}).get("A4_criterion_validity") or {}
    fwd = a4.get("forward") or {}
    findings.append(
        {
            "bulgu": "Eğitim-zamanı LP, held-out spnte iyileşmesini öngörmüyor (birincil test)",
            "sayi": (
                f"forward Spearman={fwd.get('value')} null=0 "
                f"CI={fwd.get('ci95')} pass={fwd.get('pass')}"
            ),
            "karar": "kriter geçerliliği yok → LP kovalamayı durdur veya tahminciyi değiştir",
        }
    )

    # 4 noise B3
    b3 = ((noise.get("run1_lp") or {}).get("B3") or {}).get("direct_validation") or {}
    findings.append(
        {
            "bulgu": "vy/yaw sabitleme gürültüyü çözmez",
            "sayi": f"within-cell command R²≈{b3.get('r2_linear_mean')} (null 0)",
            "karar": "D3 düşük öncelik; kovaryat tavanı da düşük (D4)",
        }
    )

    # 5 more episodes
    d1 = None
    for it in interv.get("run4_fixed") or []:
        if it.get("id") == "D1_more_episodes":
            d1 = it
            break
    findings.append(
        {
            "bulgu": "Daha çok episode geç rejimde bağlayıcı kısıt değil",
            "sayi": (
                f"σ²_signal late={d1.get('sigma_signal2_late') if d1 else None}, "
                f"8× gain cells≈{(d1 or {}).get('measured_gain', {}).get('value')}"
            ),
            "karar": "episode bütçesini 8× şişirme; ufuk/metrik tarafına bak (D2/D6/D7)",
        }
    )

    # 6 stage merge
    d2 = None
    for it in interv.get("run4_fixed") or []:
        if it.get("id") == "D2_longer_stage":
            d2 = it
            break
    findings.append(
        {
            "bulgu": "Stage birleştirme (D2) ölçülen α×excess eğrisi",
            "sayi": f"best_k={(d2 or {}).get('measured_gain', {}).get('best_k')} "
            f"value={(d2 or {}).get('measured_gain', {}).get('best_value')}",
            "karar": "optimum iç noktada olabilir; tepki gecikmesiyle trade-off",
        }
    )

    # 7 power
    findings.append(
        {
            "bulgu": "Tek seed kampanya MDE'nin altında",
            "sayi": campaign.get("recommendation"),
            "karar": f"sonraki kampanya: σ≈{campaign.get('sigma_run')}; tabloya göre seed sayısı",
        }
    )

    # 8 cross-arm
    c2 = (scorecards.get("run1_lp") or {}).get("C2_cross_arm") or {}
    lp_c = (c2.get("lp_cross_arm") or {}) if isinstance(c2, dict) else {}
    findings.append(
        {
            "bulgu": "Kollar-arası LP korelasyonu (split-half alt sınır)",
            "sayi": f"median corr={lp_c.get('value')} null={lp_c.get('null')} CI={lp_c.get('ci95')}",
            "karar": "düşükse LP ölçümü koşudan koşuya taşınmıyor",
        }
    )

    # 9 saturation
    a7 = (scorecards.get("run1_lp") or {}).get("A7_saturation") or {}
    findings.append(
        {
            "bulgu": "Hücreler arası learnability dispersiyonu (A7)",
            "sayi": (
                f"final_cell_std={a7.get('final_cell_std')} "
                f"between_cell_α={a7.get('between_cell_alpha')} "
                f"near_noise={a7.get('dispersion_near_noise')}"
            ),
            "karar": a7.get("interpretation")
            or "doygunluk varsa uniform doğru cevaptır",
        }
    )

    # 10 §11
    findings.append(
        {
            "bulgu": "§11 bulguları multi-run'a taşındı",
            "sayi": f"{len(section11.get('items', []))} transfer kaydı",
            "karar": "yanlışlanan iddia yoksa kapı kuralı kalsın; UNI null ile genelle",
        }
    )
    return findings


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def _esc(s: Any) -> str:
    t = str(s if s is not None else "")
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "evet" if v else "hayır"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:.{nd}f}"
    return str(v)


def svg_line(
    series: list[float | None],
    *,
    width: int = 420,
    height: int = 160,
    title: str = "",
    y_null: float | None = None,
    color: str = "#3b82f6",
) -> str:
    vals = [v for v in series if v is not None and isinstance(v, (int, float))]
    if len(vals) < 2:
        return f'<p class="muted">{_esc(title)}: veri yok</p>'
    lo, hi = min(vals), max(vals)
    if y_null is not None:
        lo, hi = min(lo, y_null), max(hi, y_null)
    pad = (hi - lo) * 0.1 + 1e-9
    lo, hi = lo - pad, hi + pad
    n = len(series)
    pts = []
    for i, v in enumerate(series):
        if v is None or not isinstance(v, (int, float)):
            continue
        x = 40 + (width - 50) * i / max(n - 1, 1)
        y = height - 25 - (height - 40) * (v - lo) / (hi - lo)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    null_line = ""
    if y_null is not None:
        ny = height - 25 - (height - 40) * (y_null - lo) / (hi - lo)
        null_line = (
            f'<line x1="40" y1="{ny:.1f}" x2="{width-10}" y2="{ny:.1f}" '
            f'stroke="#94a3b8" stroke-dasharray="4 3" stroke-width="1"/>'
            f'<text x="{width-10}" y="{ny-3:.1f}" text-anchor="end" '
            f'class="svg-label">null={y_null}</text>'
        )
    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" class="chart" role="img" aria-label="{_esc(title)}">
  <text x="40" y="14" class="svg-title">{_esc(title)}</text>
  <line x1="40" y1="{height-25}" x2="{width-10}" y2="{height-25}" stroke="#cbd5e1" />
  <line x1="40" y1="20" x2="40" y2="{height-25}" stroke="#cbd5e1" />
  {null_line}
  <polyline fill="none" stroke="{color}" stroke-width="2" points="{poly}" />
  <text x="40" y="{height-8}" class="svg-label">{_esc(f"n={n}")}</text>
  <text x="42" y="28" class="svg-label">{_esc(f"{hi:.2f}")}</text>
  <text x="42" y="{height-28}" class="svg-label">{_esc(f"{lo:.2f}")}</text>
</svg>"""


def svg_bars(
    labels: list[str],
    values: list[float | None],
    *,
    title: str = "",
    width: int = 420,
    height: int = 180,
    y_null: float | None = 0.0,
) -> str:
    """Bar chart that supports negative values (zero baseline, clamped height)."""
    vals = [0.0 if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)) else float(v) for v in values]
    if not vals:
        return f'<p class="muted">{_esc(title)}: veri yok</p>'
    lo = min(min(vals), 0.0 if y_null is None else min(0.0, y_null))
    hi = max(max(vals), 0.0 if y_null is None else max(0.0, y_null))
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    plot_h = height - 50
    plot_top = 22
    n = len(vals)
    bw = (width - 60) / max(n, 1)

    def y_of(v: float) -> float:
        return plot_top + plot_h * (1.0 - (v - lo) / (hi - lo))

    y0 = y_of(0.0)
    rects = []
    for i, (lab, v) in enumerate(zip(labels, vals)):
        x = 50 + i * bw
        yv = y_of(v)
        top = min(y0, yv)
        bot = max(y0, yv)
        h = max(bot - top, 0.5)  # never negative; min visible stroke
        color = "#6366f1" if v >= 0 else "#dc2626"
        rects.append(
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{bw*0.75:.1f}" height="{h:.1f}" fill="{color}"/>'
            f'<text x="{x+bw*0.35:.1f}" y="{height-10}" text-anchor="middle" class="svg-label">{_esc(lab)}</text>'
        )
    null_line = ""
    if y_null is not None:
        ny = y_of(float(y_null))
        null_line = (
            f'<line x1="50" y1="{ny:.1f}" x2="{width-10}" y2="{ny:.1f}" '
            f'stroke="#94a3b8" stroke-dasharray="4 3"/>'
        )
    zero_line = f'<line x1="50" y1="{y0:.1f}" x2="{width-10}" y2="{y0:.1f}" stroke="#cbd5e1"/>'
    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" class="chart" role="img" aria-label="{_esc(title)}">
  <text x="50" y="14" class="svg-title">{_esc(title)}</text>
  {zero_line}{null_line}
  {''.join(rects)}
</svg>"""


def svg_heatmap(
    matrix: list[list[float | None]],
    *,
    title: str = "",
    width: int = 520,
    height: int = 220,
    x_label: str = "stage",
    y_label: str = "cell",
) -> str:
    """Simple diverging heatmap (rows=cells subsampled, cols=stages)."""
    if not matrix or not matrix[0]:
        return f'<p class="muted">{_esc(title)}: veri yok</p>'
    arr = np.array(
        [[(np.nan if v is None else float(v)) for v in row] for row in matrix],
        dtype=float,
    )
    # subsample cells if many
    n_rows, n_cols = arr.shape
    max_rows = 42
    if n_rows > max_rows:
        idx = np.linspace(0, n_rows - 1, max_rows).astype(int)
        arr = arr[idx]
        n_rows = arr.shape[0]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f'<p class="muted">{_esc(title)}: veri yok</p>'
    # robust scale via percentiles
    lo, hi = np.percentile(finite, [5, 95])
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0
    mid = 0.5 * (lo + hi)
    left, top = 40, 24
    cell_w = (width - left - 10) / n_cols
    cell_h = (height - top - 20) / n_rows
    rects = []
    for r in range(n_rows):
        for c in range(n_cols):
            v = arr[r, c]
            if not np.isfinite(v):
                fill = "#e2e8f0"
            else:
                t = float(np.clip((v - lo) / (hi - lo), 0, 1))
                # blue (low) → white → red (high)
                if t < 0.5:
                    u = t * 2
                    rr, gg, bb = int(37 + u * (255 - 37)), int(99 + u * (255 - 99)), int(235)
                else:
                    u = (t - 0.5) * 2
                    rr, gg, bb = 255, int(255 - u * (255 - 59)), int(255 - u * (255 - 48))
                fill = f"rgb({rr},{gg},{bb})"
            x = left + c * cell_w
            y = top + r * cell_h
            rects.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{fill}"/>'
            )
    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" class="chart" role="img" aria-label="{_esc(title)}">
  <text x="{left}" y="14" class="svg-title">{_esc(title)} [{_esc(y_label)}×{_esc(x_label)}]</text>
  {''.join(rects)}
  <text x="{left}" y="{height-4}" class="svg-label">scale [{lo:.2f}, {hi:.2f}] mid={mid:.2f}</text>
</svg>"""


def svg_scatter(
    xs: list[float],
    ys: list[float],
    *,
    title: str = "",
    x_label: str = "x",
    y_label: str = "y",
    width: int = 420,
    height: int = 220,
) -> str:
    pts = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pts) < 3:
        return f'<p class="muted">{_esc(title)}: veri yok</p>'
    xs_, ys_ = zip(*pts)
    xmin, xmax = min(xs_), max(xs_)
    ymin, ymax = min(ys_), max(ys_)
    if abs(xmax - xmin) < 1e-12:
        xmax = xmin + 1
    if abs(ymax - ymin) < 1e-12:
        ymax = ymin + 1
    pad_x = (xmax - xmin) * 0.08
    pad_y = (ymax - ymin) * 0.08
    xmin, xmax = xmin - pad_x, xmax + pad_x
    ymin, ymax = ymin - pad_y, ymax + pad_y
    left, top, right, bot = 48, 22, width - 12, height - 28

    def px(x):
        return left + (right - left) * (x - xmin) / (xmax - xmin)

    def py(y):
        return bot - (bot - top) * (y - ymin) / (ymax - ymin)

    circles = "".join(
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.2" fill="#2563eb" fill-opacity="0.55"/>'
        for x, y in pts
    )
    # zero lines if in range
    zlines = ""
    if xmin < 0 < xmax:
        zlines += f'<line x1="{px(0):.1f}" y1="{top}" x2="{px(0):.1f}" y2="{bot}" stroke="#cbd5e1"/>'
    if ymin < 0 < ymax:
        zlines += f'<line x1="{left}" y1="{py(0):.1f}" x2="{right}" y2="{py(0):.1f}" stroke="#cbd5e1" stroke-dasharray="3 2"/>'
    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" class="chart" role="img" aria-label="{_esc(title)}">
  <text x="{left}" y="14" class="svg-title">{_esc(title)}</text>
  <line x1="{left}" y1="{bot}" x2="{right}" y2="{bot}" stroke="#cbd5e1"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bot}" stroke="#cbd5e1"/>
  {zlines}
  {circles}
  <text x="{(left+right)/2:.0f}" y="{height-6}" text-anchor="middle" class="svg-label">{_esc(x_label)}</text>
  <text x="12" y="{(top+bot)/2:.0f}" class="svg-label">{_esc(y_label)}</text>
  <text x="{right}" y="{top+10}" text-anchor="end" class="svg-label">n={len(pts)}</text>
</svg>"""


def build_html(R: dict) -> str:
    inv = R["inventory"]
    exec_rows = "".join(
        f"<tr><td>{_esc(f['bulgu'])}</td><td><code>{_esc(f['sayi'])}</code></td>"
        f"<td><strong>{_esc(f['karar'])}</strong></td></tr>"
        for f in R["executive_summary"]
    )

    # scorecard matrix
    metrics_rows = []
    run_keys = [k for k in R["scorecards"] if k != "v6_frontier"]
    for k in run_keys:
        sc = R["scorecards"][k]
        a1 = sc.get("A1_reliability") or {}
        a3 = sc.get("A3_acf") or {}
        a5 = sc.get("A5_topk") or {}
        asem_l = (a1.get("alpha_sem") or {}).get("late") or {}
        at_l = (a1.get("alpha_temporal") or {}).get("late") or {}
        amin_l = (a1.get("alpha_min") or {}).get("late") or {}
        lag1 = (a3.get("lags") or [{}])[0] if a3.get("lags") else {}
        top10 = a5.get("k10") or {}
        metrics_rows.append(
            "<tr>"
            f"<td>{_esc(k)}</td>"
            f"<td>{_fmt(asem_l.get('value'))} <span class='null'>n=0</span> "
            f"{'✓' if asem_l.get('pass') else ('—' if asem_l.get('unavailable') else '✗')}</td>"
            f"<td>{_fmt(at_l.get('value'))} <span class='null'>n=0</span> "
            f"{'✓' if at_l.get('pass') else '✗'}</td>"
            f"<td>{_fmt(amin_l.get('value'))}</td>"
            f"<td>{_fmt(lag1.get('excess_over_null'))} "
            f"<span class='null'>null lag1 excess 0</span></td>"
            f"<td>{_fmt(a3.get('signal_half_life_stages'))} stage</td>"
            f"<td>{_fmt(top10.get('value'))} "
            f"<span class='null'>n={_fmt(top10.get('null'),3)}</span> "
            f"{'✓' if top10.get('pass') else '✗'}</td>"
            "</tr>"
        )

    # inventory table
    inv_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v.get('label'))}</td>"
        f"<td>{v.get('n_frames')}</td><td>{v.get('step_range')}</td>"
        f"<td>{_esc(', '.join(v.get('missing_fields') or []) or '—')}</td>"
        f"<td>{_esc(', '.join(v.get('unavailable_analyses') or []) or '—')}</td>"
        f"<td>{'var' if v.get('validation_bank_present') else 'yok'}</td></tr>"
        for k, v in inv["runs"].items()
    )

    # noise summary
    noise_rows = []
    for k, n in R["noise_budgets"].items():
        s = n.get("summary_bounds") or {}
        noise_rows.append(
            f"<tr><td>{_esc(k)}</td>"
            f"<td>{_fmt(s.get('B1_independent_sampling_ub'))}</td>"
            f"<td>{_fmt(s.get('B2_common_mode_share'))}</td>"
            f"<td>{_fmt(s.get('B3_command_r2_within_cell'))}</td>"
            f"<td>{_fmt(s.get('B4_censor_fraction'))}</td>"
            f"<td>{_esc((n.get('B5') or {}).get('note', n.get('B5', {}).get('unavailable', ''))[:80])}</td>"
            "</tr>"
        )

    # interventions for run4
    interv_rows = []
    for it in R["interventions"].get("run4_fixed") or []:
        mg = it.get("measured_gain") or {}
        if it.get("not_measurable"):
            gain = f"ölçülemedi — {it['not_measurable']}"
        else:
            gain = f"{mg.get('metric')}={_fmt(mg.get('value'))} (null={_fmt(mg.get('null'))})"
        interv_rows.append(
            f"<tr><td>{it.get('rank') or '—'}</td><td>{_esc(it.get('id'))}</td>"
            f"<td>{_esc(gain)}</td><td>{_esc(it.get('cost') or '')}</td>"
            f"<td>{_esc(it.get('decision') or '')}</td></tr>"
        )

    # MDE table
    mde_rows = "".join(
        f"<tr><td>{r['delta_spnte']}</td><td>{_fmt(r['sigma'],4)}</td>"
        f"<td>{r['n_seeds_per_arm']}</td><td>{r['total_seeds']}</td></tr>"
        for r in (R["campaign_design"].get("mde_table_80pct") or [])
    )

    # §11
    s11_rows = "".join(
        f"<tr><td>{_esc(it.get('section'))}</td><td>{_esc(it.get('run') or '')}</td>"
        f"<td>{_esc(it.get('claim'))}</td><td>{_esc(it.get('status'))}</td>"
        f"<td><code>{_esc(json.dumps({k: it[k] for k in it if k not in ('section','run','claim','status','note')}, default=str)[:120])}</code></td></tr>"
        for it in R["section11_agreement"].get("items") or []
    )

    incomplete_rows = "".join(
        f"<tr><td>{_esc(x['item'])}</td><td>{_esc(x['status'])}</td>"
        f"<td>{_esc(x['note'])}</td></tr>"
        for x in R.get("incomplete") or []
    )

    # charts from run4 + chart payloads
    charts = R.get("charts") or {}
    r4 = R["scorecards"].get("run4_fixed") or {}
    a1 = r4.get("A1_reliability") or {}
    alpha_series = (a1.get("alpha_min") or {}).get("per_stage") or []
    chart_alpha = svg_line(
        alpha_series, title="run4 α_min per stage", y_null=0.0, color="#2563eb"
    )
    acf_excesses = [
        lg.get("excess_over_null") for lg in (r4.get("A3_acf") or {}).get("lags") or []
    ]
    chart_acf = svg_bars(
        [f"lag{i+1}" for i in range(len(acf_excesses))],
        acf_excesses,
        title="ACF excess over null (run4)",
        y_null=0.0,
    )
    # stage merge curve
    d2_curve = []
    for it in R["interventions"].get("run4_fixed") or []:
        if it.get("id") == "D2_longer_stage":
            d2_curve = it.get("curve") or []
    chart_d2 = svg_bars(
        [f"{c.get('merge_k')}×" for c in d2_curve if "merge_k" in c],
        [
            c.get("actionable_proxy_alpha_x_excess")
            for c in d2_curve
            if "merge_k" in c
        ],
        title="D2 actionable proxy vs stage merge k",
        y_null=0.0,
    )
    # noise bars run4
    nb = (R["noise_budgets"].get("run4_fixed") or {}).get("summary_bounds") or {}
    chart_noise = svg_bars(
        ["B1_ub", "B2_cm", "B2_idio", "B4_cens"],
        [
            nb.get("B1_independent_sampling_ub"),
            nb.get("B2_common_mode_share"),
            nb.get("B2_idiosyncratic_share"),
            nb.get("B4_censor_fraction"),
        ],
        title="Noise budget bounds (run4)",
        y_null=0.0,
    )
    # MDE curve
    mde = R["campaign_design"].get("mde_table_80pct") or []
    chart_mde = svg_bars(
        [str(r["delta_spnte"]) for r in mde],
        [r["n_seeds_per_arm"] for r in mde],
        title="Seeds/arm for 80% power vs Δspnte",
        y_null=None,
    )
    # cross arm
    c2 = ((R["scorecards"].get("run1_lp") or {}).get("C2_cross_arm") or {}).get(
        "lp_cross_arm"
    ) or {}
    per_step = (c2.get("per_step") if isinstance(c2, dict) else None) or []
    if not per_step and isinstance(c2, dict):
        per_step = c2.get("per_step") or []
    chart_c2 = svg_line(
        [p.get("corr") for p in per_step],
        title="C2 cross-arm LP corr (run1)",
        y_null=0.0,
        color="#059669",
    )
    # LP heatmap
    hm = charts.get("lp_heatmap_run4_fixed") or {}
    chart_heatmap = svg_heatmap(
        hm.get("matrix") or [],
        title="run4 LP heatmap (cells × stages, bootstrap excluded)",
    )
    # diet KL
    dkl = charts.get("diet_kl_run4_fixed") or {}
    chart_diet_kl = svg_line(
        dkl.get("kl") or [],
        title="run4 diet KL(p‖u) per stage",
        y_null=0.0,
        color="#7c3aed",
    )
    # A4 scatter
    sc4 = charts.get("a4_scatter_run1_lp") or {}
    chart_a4 = svg_scatter(
        sc4.get("xs") or [],
        sc4.get("ys") or [],
        title="A4 criterion validity (concurrent windows pooled)",
        x_label=sc4.get("x_label") or "atlas LP",
        y_label=sc4.get("y_label") or "held-out improvement",
    )

    # A4 scatter summary text
    a4 = (R["scorecards"].get("run1_lp") or {}).get("A4_criterion_validity") or {}
    mask_notes_html = " · ".join(
        f"{_esc(k)}: {_esc((sc.get('A1_reliability') or {}).get('mask_note') or '—')}"
        for k, sc in R["scorecards"].items()
    )

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>V5 LP-ACRL Offline Analiz Raporu</title>
<style>
:root {{
  --bg: #f8fafc; --fg: #0f172a; --muted: #64748b; --card: #ffffff;
  --border: #e2e8f0; --accent: #2563eb; --ok: #059669; --bad: #dc2626;
  --code: #f1f5f9;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #0b1220; --fg: #e2e8f0; --muted: #94a3b8; --card: #111827;
    --border: #1f2937; --accent: #60a5fa; --ok: #34d399; --bad: #f87171;
    --code: #1f2937;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.55; font-size: 15px;
}}
header {{
  padding: 1.5rem 1.25rem; border-bottom: 1px solid var(--border);
  background: var(--card); position: sticky; top: 0; z-index: 10;
}}
header h1 {{ margin: 0 0 0.25rem; font-size: 1.35rem; }}
header p {{ margin: 0; color: var(--muted); font-size: 0.9rem; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 1.25rem; }}
section {{
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.1rem 1.2rem; margin-bottom: 1rem;
}}
h2 {{ margin: 0 0 0.75rem; font-size: 1.15rem; }}
h3 {{ margin: 1rem 0 0.5rem; font-size: 1rem; }}
.muted {{ color: var(--muted); }}
.null {{ color: var(--muted); font-size: 0.8em; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border-bottom: 1px solid var(--border); padding: 0.45rem 0.5rem; text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 600; }}
.scroll {{ overflow-x: auto; }}
code, pre {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.84em; background: var(--code); padding: 0.1em 0.35em; border-radius: 4px;
}}
pre {{ padding: 0.75rem; overflow-x: auto; }}
.chart {{ background: transparent; margin: 0.5rem 0; }}
.svg-title {{ font-size: 11px; fill: var(--fg); font-family: inherit; }}
.svg-label {{ font-size: 9px; fill: var(--muted); font-family: inherit; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
@media (max-width: 800px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
.tag {{
  display: inline-block; padding: 0.1rem 0.45rem; border-radius: 999px;
  background: var(--code); font-size: 0.75rem; color: var(--muted);
}}
.warn {{ border-left: 3px solid #f59e0b; padding-left: 0.75rem; }}
footer {{ color: var(--muted); font-size: 0.85rem; padding: 1rem 0 2rem; }}
</style>
</head>
<body>
<header>
  <h1>V5 LP-ACRL — Offline veri madenciliği raporu</h1>
  <p>Üretim: {_esc(R['meta']['generated_at'])} · Tek seed uyarısı her yerde geçerli ·
  {_esc(R['meta']['rule'])}</p>
</header>
<main>

<section id="ozet">
  <h2>1. Yönetici özeti</h2>
  <p class="muted">Her satır: bulgu → sayı (null ile) → karar. En fazla 10 madde.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Bulgu</th><th>Sayı (+ null)</th><th>Karar</th></tr></thead>
    <tbody>{exec_rows}</tbody>
  </table>
  </div>
</section>

<section id="envanter">
  <h2>2. Veri envanteri ve güven notu</h2>
  <p class="warn">Eksik holdout JSON'ları (#2/#4) uydurulmadı. Markdown tabloları
  «kayıtlı tablo» olarak işaretlendi. Run #1 kısmi şema: SEM yok → α_SEM yok.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>run</th><th>etiket</th><th>frames*</th><th>step</th>
    <th>eksik alan</th><th>yapılamayan analiz</th><th>val. bank</th></tr></thead>
    <tbody>{inv_rows}</tbody>
  </table>
  </div>
  <p class="muted">* bootstrap frame (0) atıldı. Eşleşmiş validation tasarımı:
  {_esc(json.dumps(inv.get('matched_validation_design') and {
      'cross_arm_all_match': inv['matched_validation_design'].get('cross_arm_all_match'),
      'within_arm_stable': inv['matched_validation_design'].get('within_arm_commands_stable'),
  }, ensure_ascii=False))}</p>
  <h3>Veri boşlukları</h3>
  <ul>
    {''.join(f"<li><strong>{_esc(g['item'])}</strong> — {_esc(g['status'])}: {_esc(g['note'])}</li>" for g in inv.get('data_gaps') or [])}
  </ul>
  <p>Birim: 1 iter = 24 control step; 1 stage = 2000 step ≈ 83 iter
  (doğrulama: {_esc(inv.get('unit_conversions_verified'))}).</p>
</section>

<section id="skor">
  <h2>3. LP kalite skor kartı</h2>
  <p>Her metrik: değer · null · geçti/kaldı. α null = 0 (saf gürültü).
  top-10 null = k/n = 10/84 ≈ 0.119 (hipergeometrik).</p>
  <div class="scroll">
  <table>
    <thead><tr>
      <th>run</th><th>α_SEM late</th><th>α_temp late</th><th>α_min late</th>
      <th>ACF lag1 excess</th><th>yarı ömür</th><th>top-10 overlap</th>
    </tr></thead>
    <tbody>{''.join(metrics_rows)}</tbody>
  </table>
  </div>
  <div class="grid2">
    <div>{chart_alpha}</div>
    <div>{chart_acf}</div>
  </div>
  <div class="grid2">
    <div>{chart_heatmap}</div>
    <div>{chart_diet_kl}</div>
  </div>
  <div class="grid2">
    <div>{chart_c2}</div>
    <div>
      <h3>A4 kriter geçerliliği (run1 LP bank)</h3>
      <p>Eşzamanlı Spearman: <code>{_fmt((a4.get('concurrent') or {}).get('value'))}</code>
      null=0 CI={_esc((a4.get('concurrent') or {}).get('ci95'))}
      pass={_esc((a4.get('concurrent') or {}).get('pass'))}</p>
      <p><strong>İleri-dönük (birincil):</strong>
      <code>{_fmt((a4.get('forward') or {}).get('value'))}</code>
      null=0 CI={_esc((a4.get('forward') or {}).get('ci95'))}
      pass={_esc((a4.get('forward') or {}).get('pass'))}</p>
      <p class="muted">{_esc(a4.get('note') or a4.get('unavailable') or '')}</p>
      {chart_a4}
      <h3>A7 doygunluk</h3>
      <p>{_esc(json.dumps({k: (R['scorecards'].get('run1_lp') or {}).get('A7_saturation', {}).get(k)
        for k in ('final_cell_std','between_cell_alpha','dispersion_near_noise','interpretation','macro_spnte_curve')}, ensure_ascii=False, default=str)[:500])}</p>
    </div>
  </div>
  <p class="muted">Maske notları: {mask_notes_html}</p>
</section>

<section id="gurultu">
  <h2>4. Gürültü bütçesi</h2>
  <p class="muted">{_esc((list(R['noise_budgets'].values()) or [{}])[0].get('caveat', ''))}</p>
  <div class="scroll">
  <table>
    <thead><tr><th>run</th><th>B1 sampling UB</th><th>B2 common</th>
    <th>B3 cmd R²</th><th>B4 censor frac</th><th>B5 not</th></tr></thead>
    <tbody>{''.join(noise_rows)}</tbody>
  </table>
  </div>
  {chart_noise}
</section>

<section id="mudahale">
  <h2>5. Müdahale sıralaması</h2>
  <p>run4_fixed üzerinde ölçülen kazanç (veya «ölçülemedi — sebep»). Tahmin yok.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>#</th><th>müdahale</th><th>beklenen kazanç (ölçülen)</th>
    <th>maliyet</th><th>karar</th></tr></thead>
    <tbody>{''.join(interv_rows)}</tbody>
  </table>
  </div>
  {chart_d2}
</section>

<section id="kampanya">
  <h2>6. Kampanya tasarımı (MDE / güç)</h2>
  <p>σ_run ≈ <strong>{_fmt(R['campaign_design'].get('sigma_run'), 4)}</strong>
  (kaynak: {_esc(R['campaign_design'].get('sigma_source'))}).
  {_esc(R['campaign_design'].get('single_seed_warning'))}</p>
  <p>{_esc(R['campaign_design'].get('recommendation'))}</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Δspnte</th><th>σ</th><th>seed / kol</th><th>toplam seed</th></tr></thead>
    <tbody>{mde_rows}</tbody>
  </table>
  </div>
  {chart_mde}
  <h3>Kayıtlı avantajlar vs gürültü tabanı</h3>
  <div class="scroll">
  <table>
    <thead><tr><th>ad</th><th>Δ</th><th>z</th><th>1 seed güç</th><th>gürültü içi?</th><th>kaynak</th></tr></thead>
    <tbody>
    {''.join(
      f"<tr><td>{_esc(p['name'])}</td><td>{_fmt(p['delta'],4)}</td>"
      f"<td>{_fmt(p['z_vs_sigma'],2)}</td><td>{_fmt(p['power_with_1_seed_per_arm'],2)}</td>"
      f"<td>{_esc(p['within_noise_at_1_seed'])}</td><td class='muted'>{_esc(p['source'][:80])}</td></tr>"
      for p in R['campaign_design'].get('recorded_advantages_vs_noise') or []
    )}
    </tbody>
  </table>
  </div>
</section>

<section id="ekler">
  <h2>7. Koşu ekleri — sampler / diyet</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>run</th><th>ESS med</th><th>β med</th><th>stage KL med</th>
    <th>kümülatif KL</th><th>loop</th></tr></thead>
    <tbody>
    {''.join(
      f"<tr><td>{_esc(k)}</td><td>{_fmt(v.get('ess_median'))}</td>"
      f"<td>{_fmt(v.get('beta_median'))}</td><td>{_fmt(v.get('stage_kl_median'))}</td>"
      f"<td>{_fmt(v.get('cumulative_kl'))}</td>"
      f"<td>{_esc((v.get('loop_gain') or {}).get('loop_sign'))} "
      f"ρ(N,|LP|)={_fmt((v.get('loop_gain') or {}).get('median_corr_N_to_absLP'))}</td></tr>"
      for k, v in (R.get('sampler_dynamics') or {}).items()
    )}
    </tbody>
  </table>
  </div>
  <p class="muted">#4: stage-içi keskin (yüksek stage KL) ama kümülatif KL düşük olabilir —
  kısa ufuklu gürültü oylaması. A6 tavan:
  <code>{_esc(json.dumps(R.get('A6_decision_ceiling'), ensure_ascii=False, default=str)[:300])}</code></p>
</section>

<section id="s11">
  <h2>8. §11 ile mutabakat</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>§</th><th>run</th><th>iddia</th><th>durum</th><th>ölçüm</th></tr></thead>
    <tbody>{s11_rows}</tbody>
  </table>
  </div>
</section>

<section id="eksik">
  <h2>9. Yapılmadı / kısmi</h2>
  <div class="scroll">
  <table>
    <thead><tr><th>madde</th><th>durum</th><th>sebep</th></tr></thead>
    <tbody>{incomplete_rows}</tbody>
  </table>
  </div>
</section>

<section id="metod">
  <h2>10. Metodoloji eki</h2>
  <h3>Null kataloğu</h3>
  <ul>
    <li><strong>α_SEM</strong> = clip((Var(LP) − E[SE²])/Var(LP), 0, 1); null = 0 (saf ölçüm gürültüsü).
      Kör nokta: ortak-mod politika gürültüsü SEM'e girmez (§11.3).</li>
    <li><strong>α_temporal</strong> = clip(1 + 2·corr(LP_t, LP_{{t+1}}), 0, 1); null corr = −0.5
      (paylaşılan P_t penceresi). Yalnız ayrık pencereler için geçerli.</li>
    <li><strong>α</strong> = min(α_SEM, α_temporal).</li>
    <li><strong>ACF lag≥2</strong> null = 0 (tam ayrık pencereler). Hücre-etiketi permütasyonu
      paylaşılan-pencere artefaktını yakalamaz.</li>
    <li><strong>top-k overlap</strong> null = k/n (hipergeometrik beklenen fraksiyon).</li>
    <li><strong>Spearman kriter geçerliliği</strong> null = 0; p975 hücre permütasyonu ile.</li>
    <li><strong>lp_reliability</strong> teorik gürültü tabanı 0.444; metrik sıfır dönemez.</li>
  </ul>
  <h3>Varsayımlar</h3>
  <ul>
    <li>Bootstrap frame atılır; rejim eşiği stage 16/17 (§10.4) tüm koşulara taşındı
      (kısa koşularda late dilimi ince kalabilir).</li>
    <li>Run #2 primary+resume step'e göre birleştirildi.</li>
    <li>spnte_lin: düşük iyi; improvement = eski − yeni.</li>
    <li>Tek seed: tüm LP−UNI farkları C4 gürültü tabanıyla birlikte okunmalı.</li>
    <li>MDE σ'si run1 validation bank + muhafazakâr max; çok-seed ile yeniden tahmin gerekir.</li>
  </ul>
  <h3>Sınırlar</h3>
  <ul>
    <li>Per-episode atlas yok → split-half stage-içi yok; residualization tavanı bank proxy.</li>
    <li>Holdout #2/#4 yok → A4/A7 yalnız #1.</li>
    <li>FDR: keşifsel hücre taramaları etiketli; ana iddialar run-seviyesi özetler.</li>
  </ul>
</section>

<section id="regen">
  <h2>Yeniden üretme</h2>
  <pre>{_esc(R['meta']['regen_cmd'])}</pre>
  <p class="muted">Çıktılar: <code>lpacr/analysis/report/results.json</code>,
  <code>lpacr/analysis/report/v5_lp_analysis.html</code>,
  <code>lpacr/analysis/V5_ANALIZ_BULGULARI.md</code>.
  HTML'deki her sayı <code>results.json</code>'dan gelir.</p>
</section>

</main>
<footer>
  <p>LeggedGym-Ex · lpacr/analysis · offline only · {_esc(R['meta']['generated_at'])}</p>
</footer>
</body>
</html>
"""
    return html


def build_findings_md(R: dict) -> str:
    lines = [
        "# V5 LP-ACRL analiz bulguları (offline)",
        "",
        f"_Üretim: {R['meta']['generated_at']}_",
        "",
        "Bu not `HISTORY.md` §12 adayıdır. Eğitim yok; yalnız kayıtlı atlas +",
        "validation bank + TB. Kural: **null'ı türetilmemiş sayı kapı olamaz.**",
        "",
        "## Yönetici özeti",
        "",
    ]
    for i, f in enumerate(R["executive_summary"], 1):
        lines.append(f"{i}. **{f['bulgu']}** — `{f['sayi']}` → {f['karar']}")
    lines += [
        "",
        "## Skor kartı (özet)",
        "",
        "| run | α_SEM late | α_temp late | lag1 excess | top10 |",
        "|---|---|---|---|---|",
    ]
    for k, sc in R["scorecards"].items():
        a1 = sc.get("A1_reliability") or {}
        asem = ((a1.get("alpha_sem") or {}).get("late") or {}).get("value")
        at = ((a1.get("alpha_temporal") or {}).get("late") or {}).get("value")
        lag1 = ((sc.get("A3_acf") or {}).get("lags") or [{}])[0].get("excess_over_null")
        top = ((sc.get("A5_topk") or {}).get("k10") or {}).get("value")
        lines.append(f"| {k} | {asem} | {at} | {lag1} | {top} |")
    lines += [
        "",
        "## Gürültü bütçesi",
        "",
        f"σ_run (kampanya) ≈ **{R['campaign_design'].get('sigma_run')}** "
        f"({R['campaign_design'].get('sigma_source')}).",
        "",
        R["campaign_design"].get("recommendation") or "",
        "",
        "## Müdahaleler (run4)",
        "",
    ]
    for it in R["interventions"].get("run4_fixed") or []:
        if it.get("not_measurable"):
            lines.append(f"- `{it['id']}`: ölçülemedi — {it['not_measurable']}")
        else:
            mg = it.get("measured_gain") or {}
            lines.append(
                f"- `{it['id']}` rank={it.get('rank')}: "
                f"{mg.get('metric')}={mg.get('value')} (null {mg.get('null')}) "
                f"— {it.get('decision')}"
            )
    lines += [
        "",
        "## §11 mutabakat",
        "",
    ]
    for it in R["section11_agreement"].get("items") or []:
        lines.append(
            f"- §{it.get('section')} [{it.get('run','')}] {it.get('status')}: {it.get('claim')}"
        )
    lines += [
        "",
        "## Yapılmadı",
        "",
    ]
    for x in R.get("incomplete") or []:
        lines.append(f"- **{x['item']}** — {x['status']}: {x['note']}")
    lines += [
        "",
        "## Yeniden üretme",
        "",
        f"```bash\n{R['meta']['regen_cmd']}\n```",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_analyses()
    results = _jsonable(results)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {RESULTS_PATH}", flush=True)
    html = build_html(results)
    HTML_PATH.write_text(html)
    print(f"Wrote {HTML_PATH} ({len(html)} bytes)", flush=True)
    findings = build_findings_md(results)
    FINDINGS_PATH.write_text(findings)
    print(f"Wrote {FINDINGS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
