"""Run-to-run noise floor and next-campaign MDE / power (C4).

When both arms ate effectively uniform diets, LP vs UNI differences are
run-to-run noise, not mechanism.  Use that floor to size multi-seed campaigns.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from .validation_bank import ValidationBank


# Campaign-reported advantages (labeled as recorded tables — not raw recompute)
# Sources: lpacr/V5_DENEME_OZETI.md / HISTORY.md holdout tables.
RECORDED_ADVANTAGES = {
    "run1_lp_minus_uni_spnte": {
        "delta": 0.04,
        "source": "kayıtlı tablo (V5_DENEME_OZETI / HISTORY) — holdout json local for #1 only; "
        "macro curves show UNI worse early then converges",
        "note": "approximate; verify against bank macros below when available",
    },
    "run2_lp_minus_uni": {
        "delta": 0.001,
        "source": "kayıtlı tablo (HISTORY) — holdout files not on disk",
    },
    "run4_holdout_delta": {
        "delta": -0.030,
        "source": "kayıtlı tablo (HISTORY §10.8) — holdout files not on disk",
    },
}


def _seeds_for_mde(
    sigma: float,
    delta: float,
    power: float = 0.8,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> int:
    """Two-sample equal-n: n per arm for detecting delta at given power.

    Uses normal approximation: n = 2 * (z_{1-α/2} + z_power)^2 * σ^2 / δ^2
    for difference of means with common σ (per-seed outcome noise).
    """
    if delta <= 0 or sigma <= 0:
        return 9999
    z_a = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    n = 2 * (z_a + z_b) ** 2 * (sigma ** 2) / (delta ** 2)
    return int(np.ceil(n))


def noise_from_validation_banks(
    bank_lp: ValidationBank, bank_uni: ValidationBank
) -> dict[str, Any]:
    """Paired macro and cell-level arm differences across checkpoints."""
    # macro curve difference
    n = min(len(bank_lp.iterations), len(bank_uni.iterations))
    macros_lp = bank_lp.macro_spnte_lin[:n]
    macros_uni = bank_uni.macro_spnte_lin[:n]
    # paired per-replica difference at each ckpt
    paired_mean_diff = []
    unpaired_var = []
    for i in range(n):
        # match by measurement index (same design)
        d = bank_lp.spnte_lin[i] - bank_uni.spnte_lin[i]
        paired_mean_diff.append(float(np.mean(d)))
        # unpaired would use var(lp)+var(uni)
        unpaired_var.append(
            float(np.var(bank_lp.spnte_lin[i], ddof=1) + np.var(bank_uni.spnte_lin[i], ddof=1))
            / bank_lp.spnte_lin[i].size
        )
    paired_mean_diff = np.array(paired_mean_diff)
    # SE of paired mean at one ckpt
    paired_se = []
    for i in range(n):
        d = bank_lp.spnte_lin[i] - bank_uni.spnte_lin[i]
        paired_se.append(float(np.std(d, ddof=1) / np.sqrt(d.size)))

    # cell-level mean paired diff
    cell_diff = bank_lp.cell_spnte_lin[:n] - bank_uni.cell_spnte_lin[:n]
    # run-level outcome: final macro difference
    final_delta = float(macros_lp[-1] - macros_uni[-1])
    # across-checkpoint SD of macro differences as proxy for trajectory noise
    sigma_traj = float(np.std(macros_lp - macros_uni, ddof=1)) if n > 2 else float("nan")

    return {
        "iterations": list(bank_lp.iterations[:n]),
        "macro_lp": [float(x) for x in macros_lp],
        "macro_uni": [float(x) for x in macros_uni],
        "macro_diff_lp_minus_uni": [float(x) for x in (macros_lp - macros_uni)],
        "final_macro_diff": final_delta,
        "paired_mean_diff_per_ckpt": [float(x) for x in paired_mean_diff],
        "paired_se_per_ckpt": paired_se,
        "mean_paired_se": float(np.mean(paired_se)),
        "variance_reduction_paired_vs_unpaired": float(
            np.mean(unpaired_var) / (np.mean(np.array(paired_se) ** 2) + 1e-15)
        ),
        "sigma_macro_diff_across_ckpts": sigma_traj,
        "cell_diff_final_std": float(np.std(cell_diff[-1], ddof=1)),
    }


def noise_from_tb_or_atlas(
    atlases: dict[str, Any],
) -> dict[str, Any]:
    """When both arms ~uniform diet, arm differences estimate run noise.

    Uses late-stage mean performance (atlas) difference between pairs.
    """
    pairs = []
    for group in ("run1", "run2"):
        lp_key = f"{group}_lp"
        uni_key = f"{group}_uni"
        if lp_key not in atlases or uni_key not in atlases:
            continue
        a_lp, a_uni = atlases[lp_key], atlases[uni_key]
        # late mean performance over cells
        def late_mean(a):
            if a.n_frames < 2:
                return float("nan")
            start = min(atlas_mod.EARLY_STAGES.stop, a.n_frames - 1)
            block = a.performance[start:]
            return float(np.nanmean(block))

        # ESS check — if both high, diet was near-uniform
        def median_ess(a):
            vals = [
                d.get("effective_sample_size")
                for d in a.diagnostics
                if d.get("effective_sample_size") is not None
            ]
            return float(np.median(vals)) if vals else float("nan")

        pairs.append(
            {
                "group": group,
                "late_mean_perf_lp": late_mean(a_lp),
                "late_mean_perf_uni": late_mean(a_uni),
                "diff": late_mean(a_lp) - late_mean(a_uni),
                "ess_lp": median_ess(a_lp),
                "ess_uni": median_ess(a_uni),
            }
        )
    diffs = [p["diff"] for p in pairs if np.isfinite(p["diff"])]
    sigma = float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else (
        float(np.abs(diffs[0])) if len(diffs) == 1 else float("nan")
    )
    return {
        "pairs": pairs,
        "sigma_run_diff_atlas_perf": sigma,
        "n_pairs": len(diffs),
        "note": (
            "With only 1–2 effectively-uniform pairs, σ is rough. "
            "Prefer validation-bank paired SE when available."
        ),
    }


# fix import
from . import atlas as atlas_mod  # noqa: E402


def power_table(
    sigma: float,
    deltas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03),
    power: float = 0.8,
) -> list[dict[str, Any]]:
    rows = []
    for d in deltas:
        n = _seeds_for_mde(sigma, d, power=power)
        rows.append(
            {
                "delta_spnte": d,
                "sigma": sigma,
                "power": power,
                "alpha": 0.05,
                "n_seeds_per_arm": n,
                "total_seeds": 2 * n,
            }
        )
    return rows


def campaign_design(
    bank_lp: ValidationBank | None,
    bank_uni: ValidationBank | None,
    atlases: dict,
) -> dict[str, Any]:
    """Full C4 block: noise floor + MDE table + place recorded advantages."""
    atlas_noise = noise_from_tb_or_atlas(atlases)
    bank_noise = None
    if bank_lp is not None and bank_uni is not None:
        bank_noise = noise_from_validation_banks(bank_lp, bank_uni)

    # Primary σ: use SD of final-cell-level arm difference / sqrt(n_cells) as
    # macro SE, and also across-ckpt macro diff SD.  For multi-seed planning the
    # relevant σ is seed-level outcome SD.  With one pair we only have a lower
    # bound from within-seed paired SE and an upper bound from treating the
    # single observed |Δ| as a noise draw.
    if bank_noise is not None:
        # Early macro gap is mechanism/learning-speed, not seed noise (both arms
        # were near-uniform diet but policies still differ).  Use LATE checkpoints
        # only for σ, plus |final_delta| as a 1-draw scale, and paired SE×√n_cells
        # as a measurement floor inflated toward seed level.
        diffs = np.array(bank_noise["macro_diff_lp_minus_uni"], float)
        late = diffs[len(diffs) // 2 :] if len(diffs) >= 2 else diffs
        sigma_late = float(np.std(late, ddof=1)) if late.size >= 2 else float(np.abs(late[0]))
        sigma_candidates = [
            sigma_late,
            abs(bank_noise["final_macro_diff"]),
            # inflate per-measurement paired SE to macro: already macro means;
            # seed-level is larger — use 5× paired macro SE as soft floor
            bank_noise["mean_paired_se"] * 5,
        ]
        sigma_candidates = [s for s in sigma_candidates if s is not None and np.isfinite(s) and s > 0]
        # Prefer median of candidates (less hysteria than max of full trajectory)
        sigma = float(np.median(sigma_candidates)) if sigma_candidates else 0.02
        sigma_source = (
            f"validation_bank_run1_late_macro_median_of="
            f"[late_sd={sigma_late:.4f}, |finalΔ|={abs(bank_noise['final_macro_diff']):.4f}, "
            f"5×paired_se={bank_noise['mean_paired_se']*5:.4f}]"
        )
        bank_noise = dict(bank_noise)
        bank_noise["sigma_late_macro_diff"] = sigma_late
    else:
        sigma = atlas_noise.get("sigma_run_diff_atlas_perf") or 0.02
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 0.02
        sigma_source = "atlas_pair_or_default_0.02"

    table = power_table(sigma)
    # also table at half/double sigma for sensitivity
    sensitivity = {
        "half_sigma": power_table(sigma / 2),
        "double_sigma": power_table(sigma * 2),
    }

    # place recorded advantages
    placed = []
    for name, rec in RECORDED_ADVANTAGES.items():
        d = abs(rec["delta"])
        # z-score vs sigma
        z = d / sigma if sigma > 0 else float("nan")
        # detectable with 1 seed? power for n=1
        # power = Φ( |δ|/(σ√2) - z_a ) roughly for n=1 per arm
        se_1 = sigma * np.sqrt(2)
        z_obs = d / se_1 if se_1 > 0 else 0
        power_n1 = float(
            stats.norm.cdf(z_obs - stats.norm.ppf(0.975))
            + stats.norm.cdf(-z_obs - stats.norm.ppf(0.975))
        )
        # actually for alternative, one-sided-ish power:
        power_n1 = float(1 - stats.norm.cdf(stats.norm.ppf(0.975) - d / se_1))
        placed.append(
            {
                "name": name,
                "delta": rec["delta"],
                "source": rec["source"],
                "z_vs_sigma": float(z),
                "power_with_1_seed_per_arm": power_n1,
                "within_noise_at_1_seed": bool(d < 1.96 * se_1),
            }
        )

    return {
        "sigma_run": float(sigma),
        "sigma_source": sigma_source,
        "atlas_noise": atlas_noise,
        "bank_noise": bank_noise,
        "mde_table_80pct": table,
        "sensitivity": sensitivity,
        "recorded_advantages_vs_noise": placed,
        "single_seed_warning": (
            "No multi-seed estimate of σ exists. Treat MDE table as planning "
            "bound; first multi-seed pilot should re-estimate σ."
        ),
        "recommendation": (
            f"With σ≈{sigma:.4f}, detecting Δspnte=0.01 at 80% power needs "
            f"~{table[1]['n_seeds_per_arm']} seeds/arm; Δ=0.02 needs "
            f"~{table[2]['n_seeds_per_arm']}/arm. Single-seed campaign advantages "
            "at 0.001–0.04 are mostly unresolvable."
        ),
    }
