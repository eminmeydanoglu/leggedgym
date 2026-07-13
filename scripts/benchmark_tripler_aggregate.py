#!/usr/bin/env python3
"""Aggregate Phase A artifacts into summary_primary.json and decision tables.

Reads the campaign tree written by ``benchmark_tripler_orchestrator``:

  <root>/best/seed_<S>/indist/<task>.npz
  <root>/best/seed_<S>/primary/<task>_mass_vx{0.75|1}.npz
  <root>/model_3000/seed_<S>/primary/<task>_mass_vx{0.75|1}.npz

Emits under ``<root>/aggregate/``:
  summary_primary.json
  primary_cells.csv
  seed_headroom.csv
  safety.csv
  checkpoint_sensitivity.csv

Does not re-derive tracking metrics — loads shipped ``tracking_lin_err_mean`` /
``fall_rate`` fields and applies ``headroom.build_summary_primary``.

Usage:
  .venv/bin/python scripts/benchmark_tripler_aggregate.py \\
      --artifact-root logs/eval/benchmark_tripler_2026-07-13
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from legged_gym.scripts.eval.headroom import (
    PRIMARY_CELLS,
    PRIMARY_MASS,
    PRIMARY_VX,
    build_summary_primary,
    seed_headrooms,
)

# method_key -> (task, label)
METHODS = {
    "mlp": ("go2_bench_mlp", "MLP"),
    "p5": ("go2_bench_oracle_id", "P5"),
    "p5v": ("go2_bench_oracle_id_vel", "P5+V"),
}
SEEDS = (1, 2, 3)


def _scalar(z, *keys, default=None):
    for k in keys:
        if k in z.files:
            arr = np.asarray(z[k]).reshape(-1)
            if arr.size == 0:
                continue
            return float(arr.flat[0]) if arr.size == 1 else float(np.mean(arr))
    return default


def load_indist_fall(path: str) -> float:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing indist artifact: {path}")
    with np.load(path, allow_pickle=True) as z:
        fr = _scalar(z, "fall_rate", "fall_rate_mean")
        if fr is None or not np.isfinite(fr):
            raise ValueError(f"no finite fall_rate in {path}")
        return float(fr)


def load_primary_mass_curve(path: str) -> Dict[float, float]:
    """Map added_mass grid value -> tracking_lin_err_mean."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing primary sweep: {path}")
    with np.load(path, allow_pickle=True) as z:
        if "grid" not in z.files or "tracking_lin_err_mean" not in z.files:
            raise ValueError(f"{path} missing grid/tracking_lin_err_mean")
        grid = np.asarray(z["grid"], dtype=np.float64).reshape(-1)
        err = np.asarray(z["tracking_lin_err_mean"], dtype=np.float64).reshape(-1)
        if grid.shape != err.shape:
            raise ValueError(f"{path}: grid shape {grid.shape} != err {err.shape}")
        if not np.isfinite(err).all():
            raise ValueError(f"{path}: non-finite tracking_lin_err_mean")
        return {float(g): float(e) for g, e in zip(grid, err)}


def primary_path(root: str, tree: str, seed: int, task: str, vx: float) -> str:
    vx_tag = f"{vx:g}"
    return os.path.join(
        root, tree, f"seed_{seed}", "primary", f"{task}_mass_vx{vx_tag}.npz",
    )


def indist_path(root: str, seed: int, task: str) -> str:
    return os.path.join(root, "best", f"seed_{seed}", "indist", f"{task}.npz")


def cell_errors_for_method(
    root: str, tree: str, seed: int, task: str,
) -> Dict[Tuple[float, float], float]:
    """(added_mass, vx) -> tracking_lin_err for the six primary cells."""
    out: Dict[Tuple[float, float], float] = {}
    for vx in PRIMARY_VX:
        curve = load_primary_mass_curve(primary_path(root, tree, seed, task, vx))
        for mass in PRIMARY_MASS:
            # exact float key match with small tolerance
            matched = None
            for g, e in curve.items():
                if abs(g - mass) < 1e-9:
                    matched = e
                    break
            if matched is None:
                raise KeyError(
                    f"mass={mass} not in grid of {primary_path(root, tree, seed, task, vx)}; "
                    f"have {sorted(curve)}"
                )
            out[(float(mass), float(vx))] = matched
    return out


def collect_per_seed(
    root: str, tree: str = "best",
) -> Dict[int, Dict[str, Any]]:
    per_seed: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        errs = {
            mk: cell_errors_for_method(root, tree, seed, METHODS[mk][0])
            for mk in METHODS
        }
        sh = seed_headrooms(errs["mlp"], errs["p5"], errs["p5v"])
        # fall rates from A1 indist (best tree only for safety guard)
        falls = {}
        for mk, (task, _) in METHODS.items():
            falls[mk] = load_indist_fall(indist_path(root, seed, task))
        sh["fall_mlp"] = falls["mlp"]
        sh["fall_p5"] = falls["p5"]
        sh["fall_p5v"] = falls["p5v"]
        sh["seed"] = seed
        sh["tree"] = tree
        per_seed[seed] = sh
    return per_seed


def H_vectors(per_seed: Mapping[int, Mapping[str, Any]], key: str) -> List[float]:
    return [float(per_seed[s][key]) for s in sorted(per_seed.keys())]


def write_csv(path: str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate(root: str, *, require_3000: bool = True) -> Dict[str, Any]:
    agg_dir = os.path.join(root, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    per_best = collect_per_seed(root, "best")

    H_p5_3000 = H_v_3000 = None
    per_3000 = None
    if require_3000 or os.path.isdir(os.path.join(root, "model_3000")):
        try:
            # A3 only has primary/ — fall rates still come from best A1
            per_3000 = {}
            for seed in SEEDS:
                errs = {
                    mk: cell_errors_for_method(root, "model_3000", seed, METHODS[mk][0])
                    for mk in METHODS
                }
                sh = seed_headrooms(errs["mlp"], errs["p5"], errs["p5v"])
                # reuse A1 falls for gate structure when summarizing 3000 H only
                sh["fall_mlp"] = per_best[seed]["fall_mlp"]
                sh["fall_p5"] = per_best[seed]["fall_p5"]
                sh["fall_p5v"] = per_best[seed]["fall_p5v"]
                per_3000[seed] = sh
            H_p5_3000 = H_vectors(per_3000, "H_P5")
            H_v_3000 = H_vectors(per_3000, "H_V")
        except FileNotFoundError:
            if require_3000:
                raise
            per_3000 = None

    summary = build_summary_primary(
        per_best, H_p5_3000=H_p5_3000, H_v_3000=H_v_3000,
    )
    if per_3000 is not None:
        summary["model_3000"] = {
            "H_P5": H_p5_3000,
            "H_V": H_v_3000,
            "H_total": H_vectors(per_3000, "H_total"),
            "per_seed": {str(s): dict(per_3000[s]) for s in sorted(per_3000)},
        }

    # --- tables ---
    cell_rows = []
    for seed in SEEDS:
        for cell in per_best[seed]["cells"]:
            cell_rows.append({"seed": seed, "tree": "best", **cell})
        if per_3000 is not None:
            for cell in per_3000[seed]["cells"]:
                cell_rows.append({"seed": seed, "tree": "model_3000", **cell})
    write_csv(
        os.path.join(agg_dir, "primary_cells.csv"),
        cell_rows,
        ["tree", "seed", "added_mass", "command_vx",
         "err_mlp", "err_p5", "err_p5v", "h_p5", "h_v", "h_total"],
    )

    seed_rows = []
    for seed in SEEDS:
        seed_rows.append({
            "seed": seed,
            "H_P5": per_best[seed]["H_P5"],
            "H_V": per_best[seed]["H_V"],
            "H_total": per_best[seed]["H_total"],
            "fall_mlp": per_best[seed]["fall_mlp"],
            "fall_p5": per_best[seed]["fall_p5"],
            "fall_p5v": per_best[seed]["fall_p5v"],
        })
    write_csv(
        os.path.join(agg_dir, "seed_headroom.csv"),
        seed_rows,
        ["seed", "H_P5", "H_V", "H_total", "fall_mlp", "fall_p5", "fall_p5v"],
    )

    safety_rows = []
    for seed in SEEDS:
        safety_rows.append({
            "seed": seed,
            "fall_mlp": per_best[seed]["fall_mlp"],
            "fall_p5": per_best[seed]["fall_p5"],
            "fall_p5v": per_best[seed]["fall_p5v"],
            "p5_vs_mlp_delta": per_best[seed]["fall_p5"] - per_best[seed]["fall_mlp"],
            "p5v_vs_p5_delta": per_best[seed]["fall_p5v"] - per_best[seed]["fall_p5"],
            "fall_guard_p5_ok": summary["fall_guard_p5_ok"][SEEDS.index(seed)],
            "fall_guard_v_ok": summary["fall_guard_v_ok"][SEEDS.index(seed)],
        })
    write_csv(
        os.path.join(agg_dir, "safety.csv"),
        safety_rows,
        ["seed", "fall_mlp", "fall_p5", "fall_p5v",
         "p5_vs_mlp_delta", "p5v_vs_p5_delta",
         "fall_guard_p5_ok", "fall_guard_v_ok"],
    )

    sens_rows = []
    if per_3000 is not None:
        for i, seed in enumerate(SEEDS):
            sens_rows.append({
                "seed": seed,
                "H_P5_best": per_best[seed]["H_P5"],
                "H_P5_3000": H_p5_3000[i],
                "H_V_best": per_best[seed]["H_V"],
                "H_V_3000": H_v_3000[i],
            })
    write_csv(
        os.path.join(agg_dir, "checkpoint_sensitivity.csv"),
        sens_rows,
        ["seed", "H_P5_best", "H_P5_3000", "H_V_best", "H_V_3000"],
    )

    summary_path = os.path.join(agg_dir, "summary_primary.json")
    # JSON-safe: convert numpy scalars
    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, (np.floating, float)):
            v = float(o)
            if not np.isfinite(v):
                return None
            return v
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    with open(summary_path, "w") as f:
        json.dump(_sanitize(summary), f, indent=2)

    print(f"wrote {summary_path}")
    print(f"gate_p5={summary['gate_p5']}  gate_v={summary['gate_v']}  "
          f"expand_4_5={summary['expand_seeds_4_5']}")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate Phase A → summary_primary.json")
    p.add_argument(
        "--artifact-root", type=str, required=True,
        help="campaign root, e.g. logs/eval/benchmark_tripler_2026-07-13",
    )
    p.add_argument(
        "--allow-missing-3000", action="store_true",
        help="aggregate best.pt only if model_3000 tree is incomplete",
    )
    args = p.parse_args(argv)
    root = os.path.abspath(args.artifact_root)
    if not os.path.isdir(root):
        print(f"artifact root not found: {root}", file=sys.stderr)
        return 2
    aggregate(root, require_3000=not args.allow_missing_3000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
