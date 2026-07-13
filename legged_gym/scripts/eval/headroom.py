"""Primary headroom formulas and three-seed decision gates (pure numpy).

Implements benchmark_tripler.md §7:

  h_P5(s,c)    = 100 * (err_MLP - err_P5) / err_MLP
  H_P5(s)      = median over the six primary cells
  h_V(s,c)     = 100 * (err_P5 - err_P5+V) / err_P5
  H_V(s)       = median of six cells
  h_total(s,c) = 100 * (err_MLP - err_P5+V) / err_MLP
  H_total(s)   = median of six cells

Gates (P5 and velocity separately):
  pass:        all H(s) > 0, median(H) >= 10, fall guard, checkpoint-robust
  early-fail:  all H(s) <= 0
  expand:      otherwise
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PRIMARY_VX = (0.75, 1.0)
PRIMARY_MASS = (-1.0, 0.0, 1.0)
PRIMARY_CELLS: Tuple[Tuple[float, float], ...] = tuple(
    (m, vx) for vx in PRIMARY_VX for m in PRIMARY_MASS
)  # six cells: mass × vx
FALL_GUARD_ABS = 0.02
HEADROOM_PASS_MEDIAN = 10.0  # percent


def percent_headroom(err_ref: float, err_adv: float) -> float:
    """100 * (err_ref - err_adv) / err_ref. NaN if ref is non-positive or non-finite."""
    err_ref = float(err_ref)
    err_adv = float(err_adv)
    if not np.isfinite(err_ref) or not np.isfinite(err_adv) or err_ref <= 0.0:
        return float("nan")
    return 100.0 * (err_ref - err_adv) / err_ref


def median_headroom(cell_headrooms: Sequence[float]) -> float:
    arr = np.asarray(list(cell_headrooms), dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmedian(arr))


def seed_headrooms(
    err_mlp: Mapping[Tuple[float, float], float],
    err_p5: Mapping[Tuple[float, float], float],
    err_p5v: Mapping[Tuple[float, float], float],
    cells: Sequence[Tuple[float, float]] = PRIMARY_CELLS,
) -> Dict[str, Any]:
    """Compute per-cell and median headrooms for one training seed."""
    h_p5, h_v, h_total = [], [], []
    cell_rows = []
    for c in cells:
        e_m = float(err_mlp[c])
        e_p = float(err_p5[c])
        e_v = float(err_p5v[c])
        hp = percent_headroom(e_m, e_p)
        hv = percent_headroom(e_p, e_v)
        ht = percent_headroom(e_m, e_v)
        h_p5.append(hp)
        h_v.append(hv)
        h_total.append(ht)
        cell_rows.append({
            "added_mass": c[0], "command_vx": c[1],
            "err_mlp": e_m, "err_p5": e_p, "err_p5v": e_v,
            "h_p5": hp, "h_v": hv, "h_total": ht,
        })
    return {
        "cells": cell_rows,
        "H_P5": median_headroom(h_p5),
        "H_V": median_headroom(h_v),
        "H_total": median_headroom(h_total),
    }


def fall_guard_ok(
    fall_adv: float,
    fall_ref: float,
    abs_tol: float = FALL_GUARD_ABS,
) -> bool:
    """Advantageous method must not be more than abs_tol worse in fall_rate."""
    if not np.isfinite(fall_adv) or not np.isfinite(fall_ref):
        return False
    return float(fall_adv) <= float(fall_ref) + abs_tol


def gate_three_seeds(
    H_values: Sequence[float],
    *,
    fall_guard_per_seed: Optional[Sequence[bool]] = None,
    checkpoint_flips_direction: bool = False,
    pass_median: float = HEADROOM_PASS_MEDIAN,
) -> str:
    """Return 'pass' | 'early-fail' | 'expand-to-seeds-4-5'.

    ``checkpoint_flips_direction`` is retained for callers that already folded
    sensitivity into this flag; prefer computing gates raw then applying
    ``finalize_gate_with_checkpoint`` so model_3000 gets its own gate decision.
    """
    H = np.asarray(list(H_values), dtype=np.float64)
    if H.size == 0 or not np.isfinite(H).all():
        return "expand-to-seeds-4-5"
    if np.all(H <= 0.0):
        return "early-fail"
    med = float(np.median(H))
    all_pos = bool(np.all(H > 0.0))
    fall_ok = True if fall_guard_per_seed is None else all(fall_guard_per_seed)
    if all_pos and med >= pass_median and fall_ok and not checkpoint_flips_direction:
        return "pass"
    return "expand-to-seeds-4-5"


def checkpoint_direction_flip(
    H_best: Sequence[float],
    H_3000: Sequence[float],
) -> bool:
    """True if median *sign* of best vs model_3000 headrooms disagree (or non-finite).

    This is only one half of checkpoint sensitivity; also compare gate decisions
    via ``checkpoint_is_sensitive`` / ``build_summary_primary``.
    """
    mb = float(np.median(np.asarray(H_best, dtype=np.float64)))
    m3 = float(np.median(np.asarray(H_3000, dtype=np.float64)))
    if not np.isfinite(mb) or not np.isfinite(m3):
        return True
    return (mb > 0.0) != (m3 > 0.0)


def checkpoint_is_sensitive(
    H_best: Sequence[float],
    H_3000: Sequence[float],
    *,
    fall_guard_per_seed: Optional[Sequence[bool]] = None,
    pass_median: float = HEADROOM_PASS_MEDIAN,
) -> Tuple[bool, str, str]:
    """Compare best vs model_3000 under the same three-seed gate.

    Sensitive when **either**:
      * median headroom sign disagrees, or
      * gate decision differs (e.g. pass @15% vs expand @5% — same positive sign).

    Returns ``(sensitive, gate_best, gate_3000)``.
    """
    gate_best = gate_three_seeds(
        H_best,
        fall_guard_per_seed=fall_guard_per_seed,
        checkpoint_flips_direction=False,
        pass_median=pass_median,
    )
    gate_3000 = gate_three_seeds(
        H_3000,
        fall_guard_per_seed=fall_guard_per_seed,
        checkpoint_flips_direction=False,
        pass_median=pass_median,
    )
    sign_flip = checkpoint_direction_flip(H_best, H_3000)
    sensitive = bool(sign_flip or (gate_best != gate_3000))
    return sensitive, gate_best, gate_3000


def finalize_gate_with_checkpoint(
    gate_best: str,
    gate_3000: Optional[str],
    sensitive: bool,
) -> str:
    """Final campaign gate after checkpoint-robustness check.

    If sensitive, strong ``pass`` claims are demoted to ``expand-to-seeds-4-5``
    (plan: no strong headroom when best↔3000 disagree on sign *or* gate).
    Both-sides ``early-fail`` stays ``early-fail``.
    """
    if not sensitive or gate_3000 is None:
        return gate_best
    if gate_best == "early-fail" and gate_3000 == "early-fail":
        return "early-fail"
    if gate_best == "pass" and gate_3000 == "pass":
        # Sensitive with both pass should be rare (non-finite path); no strong claim.
        return "expand-to-seeds-4-5"
    return "expand-to-seeds-4-5"


def build_summary_primary(
    per_seed: Mapping[int, Mapping[str, Any]],
    *,
    H_p5_3000: Optional[Sequence[float]] = None,
    H_v_3000: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Assemble summary_primary.json content from per-seed headroom dicts.

    ``per_seed[s]`` must contain H_P5, H_V, H_total, fall_mlp, fall_p5, fall_p5v,
    and optionally cells.

    When model_3000 H vectors are provided, each comparison's gate is computed
    independently for best and 3000 (same A1 fall guards), then sensitivity is
    sign-OR-gate disagreement; final ``gate_*`` is robustness-adjusted.
    """
    seeds = sorted(per_seed.keys())
    H_p5 = [float(per_seed[s]["H_P5"]) for s in seeds]
    H_v = [float(per_seed[s]["H_V"]) for s in seeds]
    H_tot = [float(per_seed[s]["H_total"]) for s in seeds]

    fall_p5_ok = [
        fall_guard_ok(per_seed[s]["fall_p5"], per_seed[s]["fall_mlp"]) for s in seeds
    ]
    fall_v_ok = [
        fall_guard_ok(per_seed[s]["fall_p5v"], per_seed[s]["fall_p5"]) for s in seeds
    ]

    # Raw best gates (no checkpoint flag folded in)
    gate_p5_best = gate_three_seeds(H_p5, fall_guard_per_seed=fall_p5_ok)
    gate_v_best = gate_three_seeds(H_v, fall_guard_per_seed=fall_v_ok)

    sens_p5 = False
    sens_v = False
    gate_p5_3000: Optional[str] = None
    gate_v_3000: Optional[str] = None
    if H_p5_3000 is not None:
        sens_p5, gate_p5_best, gate_p5_3000 = checkpoint_is_sensitive(
            H_p5, H_p5_3000, fall_guard_per_seed=fall_p5_ok,
        )
    if H_v_3000 is not None:
        sens_v, gate_v_best, gate_v_3000 = checkpoint_is_sensitive(
            H_v, H_v_3000, fall_guard_per_seed=fall_v_ok,
        )

    gate_p5 = finalize_gate_with_checkpoint(gate_p5_best, gate_p5_3000, sens_p5)
    gate_v = finalize_gate_with_checkpoint(gate_v_best, gate_v_3000, sens_v)

    return {
        "seeds": seeds,
        "per_seed": {str(s): dict(per_seed[s]) for s in seeds},
        "H_P5": H_p5,
        "H_V": H_v,
        "H_total": H_tot,
        "median_H_P5": float(np.median(H_p5)),
        "median_H_V": float(np.median(H_v)),
        "median_H_total": float(np.median(H_tot)),
        "fall_guard_p5_ok": fall_p5_ok,
        "fall_guard_v_ok": fall_v_ok,
        "gate_p5_best": gate_p5_best,
        "gate_v_best": gate_v_best,
        "gate_p5_3000": gate_p5_3000,
        "gate_v_3000": gate_v_3000,
        "checkpoint_sensitive_p5": sens_p5,
        "checkpoint_sensitive_v": sens_v,
        # Final campaign gates (robustness-adjusted)
        "gate_p5": gate_p5,
        "gate_v": gate_v,
        "expand_seeds_4_5": gate_p5 == "expand-to-seeds-4-5"
            or gate_v == "expand-to-seeds-4-5",
        "protocol": {
            "primary_cells": [{"added_mass": m, "command_vx": vx} for m, vx in PRIMARY_CELLS],
            "fall_guard_abs": FALL_GUARD_ABS,
            "pass_median_pct": HEADROOM_PASS_MEDIAN,
            "checkpoint_sensitive_def": "median_sign_flip OR gate_best!=gate_3000",
        },
    }
