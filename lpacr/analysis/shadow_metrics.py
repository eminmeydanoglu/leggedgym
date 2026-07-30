"""Pre-registered V5 Uniform shadow-metric analysis.

This module deliberately consumes only immutable training frames and held-out
validation artifacts.  It never reaches into a training checkpoint to select a
model, and only pairs a shadow signal measured at stage ``t`` with a validation
change ending at ``t+h`` for the pre-declared horizons ``h={2, 4}``.

Example::

    python -m lpacr.analysis.shadow_metrics \
      --frames RUN/curriculum_atlas/frames.ndjson \
      --manifests RUN --validation RUN/heldout_validation \
      --output RUN/shadow_horizons.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr


HORIZONS = (2, 4)
PRIMARY_SIGNALS = ("pvl", "abs_gae", "success_ewma4_trend", "frontier")
NUISANCE_SIGNALS = ("completion_count", "mean_episode_length", "mean_raw_gae", "mean_return")


def _array(frame: dict[str, Any], key: str, n_cells: int) -> np.ndarray:
    value = (frame.get("metrics") or {}).get(key)
    if value is None:
        return np.full(n_cells, np.nan)
    out = np.asarray([np.nan if x is None else x for x in value], dtype=float)
    if out.shape != (n_cells,):
        raise ValueError(f"metric {key!r} has shape {out.shape}, expected {(n_cells,)}")
    return out


def load_frames(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load one atomic snapshot per curriculum stage, rejecting duplicates."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no frames in {path}")
    by_stage: dict[int, dict[str, Any]] = {}
    for row in rows:
        stage = (row.get("metadata") or {}).get("frame", {}).get("stage_index")
        if stage is None:
            raise ValueError("shadow analysis requires metadata.frame.stage_index")
        stage = int(stage)
        if stage in by_stage:
            raise ValueError(f"duplicate stage frame {stage}; do not stitch ambiguous runs")
        by_stage[stage] = row
    return by_stage


def _n_cells(frame: dict[str, Any]) -> int:
    values = (frame.get("metrics") or {}).get("pvl")
    if not isinstance(values, list) or not values:
        raise ValueError("frames must include non-empty V5 shadow metric pvl")
    return len(values)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def frame_signals(frames: dict[int, dict[str, Any]]) -> dict[int, dict[str, np.ndarray]]:
    """Derive primary and nuisance predictors using no future stage information."""
    n_cells = _n_cells(next(iter(frames.values())))
    ewma: np.ndarray | None = None
    result: dict[int, dict[str, np.ndarray]] = {}
    for stage in sorted(frames):
        frame = frames[stage]
        count = _array(frame, "completion_count", n_cells)
        gae_count = _array(frame, "gae_timestep_count", n_cells)
        success = _array(frame, "success_rate", n_cells)
        if not np.isfinite(success).any():
            success = _safe_ratio(_array(frame, "success_count", n_cells), count)
        current_ewma = success.copy() if ewma is None else np.where(
            np.isfinite(success), 0.4 * success + 0.6 * ewma, ewma
        )
        trend = np.full(n_cells, np.nan) if ewma is None else current_ewma - ewma
        raw_sum = _array(frame, "raw_gae_sum", n_cells)
        length_sum = _array(frame, "episode_length_sum", n_cells)
        result[stage] = {
            "pvl": _array(frame, "pvl", n_cells),
            "abs_gae": _array(frame, "abs_gae", n_cells),
            "success_ewma4_trend": trend,
            "frontier": _array(frame, "frontier", n_cells),
            "completion_count": count,
            "mean_episode_length": _safe_ratio(length_sum, count),
            "mean_raw_gae": _safe_ratio(raw_sum, gae_count),
            "mean_return": _array(frame, "performance", n_cells),
            "sampling_probability": _array(frame, "sampling_probability", n_cells),
        }
        ewma = current_ewma
    return result


def load_heldout_by_stage(
    manifests: str | Path, validation_dir: str | Path
) -> dict[int, dict[str, np.ndarray]]:
    """Join stage manifests to held-out artifacts by actual PPO iteration.

    A file name alone is intentionally insufficient: each manifest contributes
    the observed stage index and exact completed iteration, which must match the
    validation artifact's embedded checkpoint iteration.
    """
    manifests = Path(manifests)
    validation_dir = Path(validation_dir)
    out: dict[int, dict[str, np.ndarray]] = {}
    fingerprint: str | None = None
    for manifest_path in sorted(manifests.glob("shadow_stage_*.json")):
        manifest = json.loads(manifest_path.read_text())
        stage = int(manifest["snapshot_stage_index"])
        iteration = int(manifest["ppo_completed_iteration"])
        artifact_path = validation_dir / f"model_{iteration}.json"
        if not artifact_path.is_file():
            continue
        artifact = json.loads(artifact_path.read_text())
        if int(artifact.get("checkpoint_iteration", -1)) != iteration:
            raise ValueError(f"iteration mismatch: {manifest_path.name} vs {artifact_path.name}")
        checkpoint_path = manifests / str(manifest.get("checkpoint_file", ""))
        if not checkpoint_path.is_file():
            raise ValueError(f"missing stage checkpoint named by {manifest_path.name}: {checkpoint_path}")
        digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if str(artifact.get("checkpoint_sha256", "")) != digest:
            raise ValueError(f"checkpoint SHA256 mismatch for {manifest_path.name}")
        shadow = (artifact.get("provenance") or {}).get("shadow_stage") or {}
        if int(shadow.get("snapshot_stage_index", -1)) != stage:
            raise ValueError(f"held-out stage provenance mismatch for {manifest_path.name}")
        artifact_fingerprint = str(artifact.get("validation_bank_fingerprint") or "")
        if not artifact_fingerprint:
            raise ValueError(f"missing validation bank fingerprint in {artifact_path}")
        if fingerprint is None:
            fingerprint = artifact_fingerprint
        elif fingerprint != artifact_fingerprint:
            raise ValueError("held-out artifacts have different bank fingerprints")
        cells = sorted((artifact.get("scores") or {}).get("cells") or [], key=lambda x: x["cell_id"])
        if not cells:
            raise ValueError(f"missing per-cell held-out scores in {artifact_path}")
        if stage in out:
            raise ValueError(f"duplicate held-out stage {stage}")
        out[stage] = {
            "spnte_lin": np.asarray([x["spnte_lin"] for x in cells], float),
            "fall_rate": np.asarray([x["fall_rate"] for x in cells], float),
            "cell_success": np.asarray([float(x["cell_success"]) for x in cells], float),
        }
    if not out:
        raise ValueError("no manifests had a matching held-out validation artifact")
    return out


def _rho(signal: np.ndarray, target: np.ndarray) -> tuple[float, int]:
    ok = np.isfinite(signal) & np.isfinite(target)
    n = int(ok.sum())
    if n < 20 or np.unique(signal[ok]).size < 2 or np.unique(target[ok]).size < 2:
        return np.nan, n
    return float(spearmanr(signal[ok], target[ok]).statistic), n


def _top_lift(signal: np.ndarray, target: np.ndarray, k: int, rng: np.random.Generator) -> tuple[float, float]:
    ok = np.isfinite(signal) & np.isfinite(target)
    ids = np.flatnonzero(ok)
    if len(ids) < max(20, k):
        return np.nan, np.nan
    chosen = ids[np.argsort(signal[ids])[-k:]]
    top = float(np.mean(target[chosen]))
    random = float(np.mean(target[rng.choice(ids, size=k, replace=False)]))
    return top - random, top


def _bootstrap(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> list[float | None]:
    if not len(values):
        return [None, None]
    draws = np.asarray([np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)])
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _strata(frame: dict[str, Any], n_cells: int) -> dict[str, np.ndarray]:
    """Recover V5's stable C-order (vx outer, terrain-cell inner) strata."""
    coordinates = ((frame.get("task_space") or {}).get("coordinates") or {})
    vx = list(coordinates.get("vx_bin") or [])
    terrain = list(coordinates.get("terrain_cell") or [])
    if len(vx) * len(terrain) != n_cells:
        return {"all": np.asarray(["all"] * n_cells, dtype=object)}
    vx_values = np.repeat(np.asarray(vx, dtype=object), len(terrain))
    terrain_values = np.tile(np.asarray(terrain, dtype=object), len(vx))
    family = np.asarray([str(value).split(" · ")[0] for value in terrain_values], dtype=object)
    level = np.asarray([str(value).rsplit(" · ", 1)[-1] for value in terrain_values], dtype=object)
    return {"vx_bin": vx_values, "terrain_family": family, "terrain_level": level}


def _sem_from_sums(count: np.ndarray, total: np.ndarray, sq_total: np.ndarray) -> np.ndarray:
    """Unbiased SEM with safe variance clamp; unobserved cells stay NaN."""
    out = np.full(count.shape, np.nan)
    valid = count > 1
    mean = _safe_ratio(total, count)
    variance = np.zeros(count.shape)
    variance[valid] = np.maximum(
        (sq_total[valid] - count[valid] * mean[valid] ** 2) / (count[valid] - 1.0), 0.0
    )
    out[valid] = np.sqrt(variance[valid] / count[valid])
    return out


def telemetry_coverage(frames: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Per-stage count/coverage/SEM summaries; never turns unobserved into zero."""
    n_cells = _n_cells(next(iter(frames.values())))
    rows = []
    for stage, frame in sorted(frames.items()):
        gae_count = _array(frame, "gae_timestep_count", n_cells)
        completion = _array(frame, "completion_count", n_cells)
        pvl_sem = _sem_from_sums(
            gae_count, _array(frame, "positive_gae_sum", n_cells),
            _array(frame, "positive_gae_sq_sum", n_cells),
        )
        abs_sem = _sem_from_sums(
            gae_count, _array(frame, "absolute_gae_sum", n_cells),
            _array(frame, "absolute_gae_sq_sum", n_cells),
        )
        rows.append({
            "stage": stage,
            "moving_gae_timestep_count": int(np.nansum(gae_count)),
            "moving_completion_count": int(np.nansum(completion)),
            "pvl_covered_cells": int(np.isfinite(_array(frame, "pvl", n_cells)).sum()),
            "abs_gae_covered_cells": int(np.isfinite(_array(frame, "abs_gae", n_cells)).sum()),
            "completion_covered_cells": int((completion > 0).sum()),
            "median_pvl_sem": float(np.nanmedian(pvl_sem)) if np.isfinite(pvl_sem).any() else None,
            "median_abs_gae_sem": float(np.nanmedian(abs_sem)) if np.isfinite(abs_sem).any() else None,
        })
    return {"n_cells": n_cells, "per_stage": rows}


def analyze(
    frames: dict[int, dict[str, Any]], heldout: dict[int, dict[str, np.ndarray]],
    *, n_bootstrap: int = 2000, n_permutation: int = 2000, top_k: int = 10, seed: int = 31001,
) -> dict[str, Any]:
    """Evaluate every pre-declared predictor at h=2 and h=4.

    Bootstrap resampling is clustered by stage: a draw resamples complete
    per-stage correlations/lifts, never individual correlated cell records.
    """
    signals = frame_signals(frames)
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {"horizons": {}, "primary_signals": list(PRIMARY_SIGNALS),
                              "nuisance_signals": list(NUISANCE_SIGNALS), "seed": seed,
                              "telemetry_coverage": telemetry_coverage(frames)}
    all_names = PRIMARY_SIGNALS + NUISANCE_SIGNALS + ("sampling_probability",)
    strata = _strata(next(iter(frames.values())), len(next(iter(signals.values()))["pvl"]))
    for horizon in HORIZONS:
        pairs = [stage for stage in sorted(heldout) if stage + horizon in heldout and stage in signals]
        horizon_rows: dict[str, Any] = {"paired_stages": pairs, "signals": {}}
        for name in all_names:
            per_target: dict[str, list[dict[str, Any]]] = {"gain_spnte_lin": [], "gain_cell_success": [], "gain_fall_rate": []}
            for stage in pairs:
                signal = signals[stage][name]
                before, after = heldout[stage], heldout[stage + horizon]
                targets = {
                    "gain_spnte_lin": before["spnte_lin"] - after["spnte_lin"],
                    "gain_cell_success": after["cell_success"] - before["cell_success"],
                    "gain_fall_rate": before["fall_rate"] - after["fall_rate"],
                }
                for target_name, target in targets.items():
                    rho, n = _rho(signal, target)
                    lift, top = _top_lift(signal, target, top_k, rng)
                    per_target[target_name].append({"stage": stage, "rho": rho, "n": n, "top_k_lift": lift, "top_k_gain": top})
            summarized: dict[str, Any] = {}
            for target_name, rows in per_target.items():
                rho = np.asarray([r["rho"] for r in rows if np.isfinite(r["rho"])], float)
                lift = np.asarray([r["top_k_lift"] for r in rows if np.isfinite(r["top_k_lift"])], float)
                null_rho: list[float] = []
                for _ in range(n_permutation):
                    shuffled = []
                    for stage in pairs:
                        signal = signals[stage][name]
                        before, after = heldout[stage], heldout[stage + horizon]
                        target = {"gain_spnte_lin": before["spnte_lin"] - after["spnte_lin"],
                                  "gain_cell_success": after["cell_success"] - before["cell_success"],
                                  "gain_fall_rate": before["fall_rate"] - after["fall_rate"]}[target_name]
                        value, _ = _rho(signal, rng.permutation(target))
                        if np.isfinite(value):
                            shuffled.append(value)
                    if shuffled:
                        null_rho.append(float(np.median(shuffled)))
                summarized[target_name] = {
                    "n_stage_clusters": int(len(rho)), "median_spearman": float(np.median(rho)) if len(rho) else None,
                    "stage_clustered_bootstrap_ci": _bootstrap(rho, rng, n_bootstrap),
                    "median_top_k_lift": float(np.median(lift)) if len(lift) else None,
                    "top_k_lift_ci": _bootstrap(lift, rng, n_bootstrap),
                    "permutation_null_spearman_p975": float(np.percentile(null_rho, 97.5)) if null_rho else None,
                    "per_stage": rows,
                }
            horizon_rows["signals"][name] = summarized
        # Predeclared direction/leakage negative control: the same signal may
        # not be interpreted as prospective if it only associates with an
        # already-realised gain in the preceding h-stage interval.
        past_control: dict[str, Any] = {}
        for name in PRIMARY_SIGNALS:
            values = []
            for stage in pairs:
                if stage - horizon not in heldout:
                    continue
                rho, n = _rho(
                    signals[stage][name],
                    heldout[stage - horizon]["spnte_lin"] - heldout[stage]["spnte_lin"],
                )
                values.append({"stage": stage, "spearman": rho, "n": n})
            finite = np.asarray([r["spearman"] for r in values if np.isfinite(r["spearman"])], float)
            past_control[name] = {
                "per_stage": values,
                "median_spearman": float(np.median(finite)) if len(finite) else None,
                "stage_clustered_bootstrap_ci": _bootstrap(finite, rng, n_bootstrap),
            }
        horizon_rows["negative_controls"] = {
            "past_gain_spnte_lin": past_control,
            "uniform_sampling_probability": "reported as a nuisance/null predictor; any non-constant value is a protocol violation",
        }
        # Regime and terrain summaries are descriptive (not separate winner
        # tests): small family/level bins are explicitly retained with their n.
        split = pairs[len(pairs) // 2:]
        horizon_rows["regime_split"] = {
            "early_stages": pairs[:len(pairs) // 2], "late_stages": split,
        }
        stratified: dict[str, Any] = {}
        for dimension, labels in strata.items():
            groups: dict[str, Any] = {}
            for label in sorted(set(str(x) for x in labels)):
                mask = labels == label
                group_rows = []
                for stage in pairs:
                    target = heldout[stage]["spnte_lin"] - heldout[stage + horizon]["spnte_lin"]
                    # PVL is the registered example; other candidates retain
                    # full per-stage data above and can be inspected identically.
                    rho, n = _rho(signals[stage]["pvl"][mask], target[mask])
                    group_rows.append({"stage": stage, "n_cells": int(mask.sum()), "spearman": rho, "n": n})
                groups[label] = group_rows
            stratified[dimension] = groups
        horizon_rows["pvl_strata_gain_spnte_lin"] = stratified
        report["horizons"][str(horizon)] = horizon_rows
    return report


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutation", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    frames = load_frames(args.frames)
    heldout = load_heldout_by_stage(args.manifests, args.validation)
    report = analyze(frames, heldout, n_bootstrap=args.n_bootstrap, n_permutation=args.n_permutation, top_k=args.top_k)
    report["provenance"] = {"frames": str(Path(args.frames).resolve()), "manifests": str(Path(args.manifests).resolve()),
                            "validation": str(Path(args.validation).resolve()), "horizons": list(HORIZONS)}
    Path(args.output).write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
