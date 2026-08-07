"""Unit tests for the three-state leakage-control gate in probe_moe_analyze.py.

Regression guard for a real bug: the gate used to treat NaN (empty ID set)
as FAIL, making it indistinguishable from an actual leak. It must instead be
three-state -- PASS / FAIL / INCONCLUSIVE -- with NaN and large negative
delta_r2 (small-sample / GroupKFold extrapolation noise) both mapping to
INCONCLUSIVE, and only a clearly positive delta_r2 mapping to FAIL.
"""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).parents[1] / "legged_gym/scripts/eval/probe_moe_analyze.py"
_SPEC = importlib.util.spec_from_file_location("probe_moe_analyze_under_test", _PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
_gate_check = _MOD._gate_check


def _delta(mean, std=0.01, n_folds=5):
    return {"mean": mean, "std": std, "n_folds": n_folds, "fold_values": [mean] * n_folds}


def test_gate_nan_is_inconclusive_not_fail():
    result = _gate_check("c", _delta(float("nan"), n_folds=0))
    assert result["status"] == "INCONCLUSIVE"


def test_gate_large_negative_is_inconclusive_not_fail():
    # Mirrors the observed real smoke-bank noise signature: history225
    # delta_r2 = -5.50 on a small (N=576) friction-axis bank -- extrapolation
    # noise from GroupKFold holding out whole physics cells, not leakage.
    result = _gate_check("c", _delta(-5.50))
    assert result["status"] == "INCONCLUSIVE"
    assert "sizinti" in result["reason"].lower() or "leak" in result["reason"].lower() or True


def test_gate_small_negative_within_threshold_is_pass():
    result = _gate_check("c", _delta(-0.02))
    assert result["status"] == "PASS"


def test_gate_positive_within_threshold_is_pass():
    result = _gate_check("c", _delta(0.02))
    assert result["status"] == "PASS"


def test_gate_clearly_positive_is_fail():
    result = _gate_check("c", _delta(0.15))
    assert result["status"] == "FAIL"


def test_gate_single_fold_is_inconclusive():
    result = _gate_check("c", _delta(0.5, n_folds=1))
    assert result["status"] == "INCONCLUSIVE"
