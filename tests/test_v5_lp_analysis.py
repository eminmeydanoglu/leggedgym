"""Tests for offline V5 LP analysis loaders and scorecard (real paths)."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from lpacr.analysis import atlas
from lpacr.analysis import scorecard
from lpacr.analysis import validation_bank
from lpacr.analysis.lp_diagnostics import alpha_sem, alpha_temporal

REPO = Path(__file__).resolve().parents[1]


def test_load_run1_partial_schema_nan_fill():
    a = atlas.load_run("run1_lp", skip_bootstrap=False)
    assert a.n_frames >= 30
    assert a.n_cells == 84
    assert not a.has_sem
    assert "performance_sem" in a.missing_fields
    assert "A1_alpha_sem" in a.unavailable_analyses
    # NaN fill, not zeros
    assert np.isnan(a.performance_sem).all()
    # bootstrap retained
    assert a.raw_frame_count_including_bootstrap == a.n_frames
    # LP defined after bootstrap
    assert np.isfinite(a.learning_progress[1]).sum() > 50


def test_load_run4_full_schema_matches_section11_alpha():
    a = atlas.load_run("run4_fixed", skip_bootstrap=False)
    assert a.has_sem
    assert a.n_frames == 41  # includes bootstrap
    sc = scorecard.compute_alpha_series(a)
    # §11.2: late α_SEM ≈ 0, α_temporal ≈ 0.1
    asem_late = sc["alpha_sem"]["late"]["value"]
    at_late = sc["alpha_temporal"]["late"]["value"]
    assert asem_late is not None and asem_late < 0.05
    assert at_late is not None and 0.05 < at_late < 0.25
    # early α_SEM > late
    asem_early = sc["alpha_sem"]["early"]["value"]
    assert asem_early is not None and asem_early > asem_late
    # stages at zero count near 27/40
    assert sc["alpha_sem"]["stages_at_zero"] >= 20


def test_bootstrap_excluded_from_alpha_series():
    a = atlas.load_run("run4_fixed")
    sc = scorecard.compute_alpha_series(a)
    n = sc["alpha_sem"]["n_stages_total"]
    assert n == a.n_frames - 1  # all non-bootstrap frames with enough cells


def test_run2_uni_eligible_all_false_still_has_scorecard():
    """C1: full-schema UNI with eligible_for_lp all-False must not blank A3."""
    a = atlas.load_run("run2_uni")
    assert a.has_eligible
    assert float(a.eligible.mean()) < 0.05
    assert not a.quality_require_eligible()
    inv = a.to_inventory()
    assert any("eligible_mask" in x for x in inv["unavailable_analyses"])
    assert "near-zero" in inv["quality_mask_note"] or "finite LP" in inv["quality_mask_note"]

    a1 = scorecard.compute_alpha_series(a)
    a3 = scorecard.compute_acf(a)
    # α_temporal must be finite (observational quality)
    at = a1["alpha_temporal"]["all"]["value"]
    assert at is not None and math.isfinite(at)
    # lag1 ACF value present
    lag1 = a3["lags"][0]
    assert lag1.get("value") is not None and math.isfinite(lag1["value"])
    # recoverable: lag1 near -0.5 pure-noise ⇒ α_temporal small but defined
    assert lag1["null"] == -0.5
    assert lag1.get("excess_over_null") is not None


def test_run3_crash_scorecard_not_blank():
    a = atlas.load_run("run3_crash")
    assert float(a.eligible.mean()) < 0.05
    sc = scorecard.compute_alpha_series(a)
    assert sc["alpha_temporal"]["all"]["value"] is not None


def test_validation_bank_matched_design():
    banks = validation_bank.load_run1_banks()
    assert banks["run1_lp"] is not None
    assert banks["run1_uni"] is not None
    m = validation_bank.verify_matched_design(banks["run1_lp"], banks["run1_uni"])
    assert m["cross_arm_all_match"] is True
    assert m["within_arm_commands_stable"] is True
    assert m["fingerprint"]


def test_hypergeom_null_k10():
    assert abs(scorecard.hypergeom_overlap_null(10, 84) - 10 / 84) < 1e-12


def test_alpha_temporal_null():
    assert abs(alpha_temporal(-0.5) - 0.0) < 1e-9
    assert abs(alpha_temporal(0.0) - 1.0) < 1e-9


def test_results_json_if_present():
    """If report was built, structure must satisfy acceptance gates."""
    path = REPO / "lpacr/analysis/report/results.json"
    if not path.is_file():
        pytest.skip("results.json not built yet")
    R = json.loads(path.read_text())
    assert set(R["scorecards"]) >= {
        "run1_lp",
        "run1_uni",
        "run2_lp",
        "run2_uni",
        "run3_crash",
        "run4_fixed",
    }
    for run_id, sc in R["scorecards"].items():
        for key in ("A1_reliability", "A3_acf", "A5_topk"):
            assert key in sc
        # full-schema runs must have finite A3 lag1 (not blank from eligible=0)
        if run_id in ("run2_uni", "run4_fixed", "run2_lp"):
            lags = (sc.get("A3_acf") or {}).get("lags") or []
            assert lags, f"{run_id} missing A3 lags"
            v = lags[0].get("value")
            assert v is not None and math.isfinite(float(v)), (
                f"{run_id} A3 lag1 blank: {lags[0]}"
            )
            at = ((sc.get("A1_reliability") or {}).get("alpha_temporal") or {}).get(
                "all"
            ) or {}
            assert at.get("value") is not None or at.get("unavailable"), run_id
            if run_id == "run2_uni":
                assert at.get("value") is not None, "C1 UNI arm α_temporal must be finite"

    assert R["campaign_design"]["mde_table_80pct"]
    assert len(R["campaign_design"]["mde_table_80pct"]) == 4
    for it in R["interventions"]["run4_fixed"]:
        assert it.get("measured_gain") is not None or it.get("not_measurable")

    # charts required by §6.1
    charts = R.get("charts") or {}
    assert "lp_heatmap_run4_fixed" in charts
    assert charts["lp_heatmap_run4_fixed"].get("matrix")
    assert "diet_kl_run4_fixed" in charts
    assert "a4_scatter_run1_lp" in charts

    html = (REPO / "lpacr/analysis/report/v5_lp_analysis.html").read_text()
    assert "cdn." not in html.lower()
    assert "build_report" in html
    # no negative SVG rect heights
    for m in re.finditer(r'height="(-?[0-9.]+)"', html):
        assert float(m.group(1)) >= 0, m.group(0)
    # required chart titles / sections present
    for s in (
        "Yönetici özeti",
        "Gürültü",
        "Müdahale",
        "Metodoloji",
        "LP heatmap",
        "diet KL",
        "criterion validity",
    ):
        assert s.lower() in html.lower() or s in html, s
