"""CPU-only decodability analysis for controlled MoE-CTS latent banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Direct script-directory import avoids legged_gym.__init__ and therefore does
# not initialize or require a simulator for analysis.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_lib import (  # noqa: E402
    NumpyLogisticProbe, _best_ridge_numpy, classification_metrics,
    compute_classifier_metrics, compute_decoder_metrics,
    delta_balanced_accuracy, delta_r2, group_kfold_splits,
)


# Faz A scale (80-100k rows x 10 feature sets x ~19 targets) makes the
# original exhaustive inner alpha-search (5 alphas x 3 inner folds = ~16
# ridge refits per outer fold) and the default max_iter=1000 logistic GD the
# dominant cost -- both scale ~linearly in n per refit/iteration, so at
# 100k rows (166x this module's 600-row profiling run) the exhaustive path
# would take on the order of hours. Fixing alpha and capping GD iterations
# removes the redundant refitting multiplier without changing what's being
# estimated (features are standardized, so a fixed alpha=1.0 is a reasonable
# single choice; see probe_lib._best_ridge_numpy's docstring). This does NOT
# touch probe_lib's *defaults*, so probe_rma_analyze.py's exact-match
# regression against the published RMA probe_metrics.json is unaffected.
FAST_RIDGE_KWARGS = {"alphas": (1.0,), "n_inner_folds": 1}
FAST_LOGISTIC_MAX_ITER = 300
# Caps each fold's logistic-probe *fit* set; OOF test coverage over all ID
# rows is unaffected. This is what actually bounds runtime at Faz A scale --
# without it, GD cost grows ~linearly with n and dominates the analysis
# (profiling: with alpha search removed, NumpyLogisticProbe.fit is ~80% of
# remaining runtime on a 600-row bank).
FAST_LOGISTIC_MAX_TRAIN_ROWS = 5000

# Must stay in lockstep with probe_moe_latent.AXIS_NAMES -- these index the
# physics_raw columns the collector writes.  control_delay is excluded there
# (go2_moects trains in substep delay mode, which has no per-episode delay
# value); see the comment on probe_moe_latent.AXIS_NAMES.
PHYSICS = ("friction", "added_mass", "com_x", "com_y", "com_z",
           "pd_gain_scale")
REGRESSION_TARGETS = {
    "base_lin_vel_x": ("base_lin_vel", 0), "base_lin_vel_y": ("base_lin_vel", 1),
    "base_lin_vel_z": ("base_lin_vel", 2), "base_ang_vel_z": ("base_ang_vel", 2),
    "terrain_level": ("terrain_level", None), "command_x": ("command", 0),
    "command_y": ("command", 1), "command_yaw": ("command", 2),
    "command_change_time": ("command_change_time", None), "gait_phase": ("gait_phase", None),
    "torque_norm": ("torque_norm", None), "dof_acc_norm": ("dof_acc_norm", None),
}
CATEGORICAL_TARGETS = ("contact_pattern", "terrain_id", "command_id")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def _groups(data):
    # episode_id is reset to 0 at the start of every (axis, command) block in
    # the collector (probe_moe_latent.py:170: `episode_id[:] = 0` runs per
    # command, not just per axis), so the same env_id+episode_id pair recurs
    # across different commands within one axis sweep. command_id must be
    # part of the group key or GroupKFold conflates two temporally disjoint
    # rollouts as one group, weakening its leakage protection.
    fields = [data[k].astype(str) for k in
              ("axis_code", "val_index", "env_id", "command_id", "episode_id")]
    out = fields[0]
    for field in fields[1:]:
        out = np.char.add(np.char.add(out, ":"), field)
    return out


def _pca32(history, train_mask):
    if not np.any(train_mask):
        # Smoke banks may intentionally contain only OOD endpoints. Keep the
        # transform defined, while all ID metrics remain NaN and OOD is never
        # included in the training mask below.
        train_mask = np.ones(len(history), dtype=bool)
    mean = history[train_mask].mean(0)
    _, _, vt = np.linalg.svd(history[train_mask] - mean, full_matrices=False)
    return (history - mean) @ vt[:32].T


def _feature_sets(data, train_mask, seed):
    history = data["obs_history"].astype(np.float64)
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(history.shape[1], 32)) / np.sqrt(history.shape[1])
    return {
        "z_s": data["z_s"], "z_t": data["z_t"], "g": data["g"],
        "obs45": data["obs"], "history225": history,
        "obs45+z_s": np.concatenate([data["obs"], data["z_s"]], 1),
        "obs45+z_t": np.concatenate([data["obs"], data["z_t"]], 1),
        "pca32_history": _pca32(history, train_mask),
        "rp32_history": history @ projection,
        "random_init_latent": data["z_random"],
    }


def _target(data, spec):
    key, col = spec
    value = data[key]
    return value if col is None else value[:, col]


def _ood_regression(X, y, id_mask, ood_mask, groups, ridge_kwargs=None):
    """R2 of a model fit on all ID rows, evaluated on the held-out OOD rows.

    Previously this ran its own 5-way GroupKFold *inside* the ID set and
    averaged predictions across folds' models -- redundant CV nested inside
    the outer analysis loop's own CV, and a large share of the runtime at
    scale (Faz A profiling: ~70 calls x 5 folds x full inner alpha-search
    each). OOD rows never overlap the ID set by construction (see
    ``ood_flag`` in the collector), so there is no leakage risk in fitting
    once on the *entire* ID split rather than folding it again -- the OOD
    rows were never available for alpha selection either way. One fit here
    is both cheaper and uses strictly more training data per model.
    """
    ridge_kwargs = ridge_kwargs or {}
    id_indices = np.flatnonzero(id_mask)
    if len(id_indices) < 10 or not np.any(ood_mask):
        return {"R2": np.nan, "n": int(np.sum(ood_mask))}
    pred, _ = _best_ridge_numpy(X[id_indices], y[id_indices], X[ood_mask], **ridge_kwargs)
    truth = y[ood_mask]
    ss = np.sum((truth - truth.mean()) ** 2)
    return {"R2": float(1 - np.sum((truth - pred) ** 2) / max(ss, 1e-12)), "n": len(truth)}


def _gate_check(name, delta, threshold=0.05):
    """Three-state, one-sided leakage-control gate.

    Controls (random-init encoder latent, shuffled physics label) carry no
    real signal, so their ΔR2 vs obs45 should sit near 0. Only a clearly
    POSITIVE ΔR2 (> +threshold) is evidence of leakage. NaN (empty ID set,
    <2 groups) is INCONCLUSIVE, never FAIL -- "no data" must not read as
    "leak detected". A strongly NEGATIVE ΔR2 is also not leakage: with
    GroupKFold, every fold holds out an entire physics cell, so the probe
    extrapolates outside the training range and high-dimensional features
    (e.g. history225) can score far below the obs45-only baseline purely
    from small-sample / extrapolation noise (observed on a real N=576
    friction-axis smoke bank: random_init_latent ΔR2=-0.15, history225
    ΔR2=-5.50, no leakage involved). Such cases are reported as
    INCONCLUSIVE with an explicit "insufficient sample / extrapolation"
    reason rather than silently passing or wrongly failing.
    """
    mean = delta.get("mean", float("nan"))
    n_folds = delta.get("n_folds", len(delta.get("fold_values", []) or []))
    std = delta.get("std", float("nan"))
    if not np.isfinite(mean) or n_folds < 2:
        return {"name": name, "status": "INCONCLUSIVE", "delta_r2_mean": mean, "delta_r2_std": std,
                "n_folds": n_folds,
                "reason": "delta_r2 NaN veya <2 fold (ID kumesi bos ya da yetersiz grup sayisi)"}
    if mean > threshold:
        return {"name": name, "status": "FAIL", "delta_r2_mean": mean, "delta_r2_std": std,
                "n_folds": n_folds,
                "reason": f"delta_r2={mean:.4f} > +{threshold}: olasi sizinti (kontrol sinyalsiz olmali)"}
    if mean < -threshold:
        return {"name": name, "status": "INCONCLUSIVE", "delta_r2_mean": mean, "delta_r2_std": std,
                "n_folds": n_folds,
                "reason": (f"delta_r2={mean:.4f} < -{threshold}: kucuk orneklem / GroupKFold "
                           "ekstrapolasyon gurultusu, sizinti KANITI DEGIL")}
    return {"name": name, "status": "PASS", "delta_r2_mean": mean, "delta_r2_std": std, "n_folds": n_folds,
            "reason": f"|delta_r2|={mean:.4f} <= {threshold}"}


def analyze_one(path, out_dir, seed=1, fast=True):
    ridge_kwargs = FAST_RIDGE_KWARGS if fast else {}
    logistic_max_iter = FAST_LOGISTIC_MAX_ITER if fast else 1000
    logistic_max_train = FAST_LOGISTIC_MAX_TRAIN_ROWS if fast else None
    data = dict(np.load(path, allow_pickle=True))
    required = {"obs", "obs_history", "z_s", "z_t", "z_random", "g", "physics_raw",
                "ood_flag", "axis_code", "val_index", "env_id", "episode_id"}
    missing = sorted(required - data.keys())
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    groups = _groups(data)
    id_mask = ~data["ood_flag"].astype(bool)
    ood_mask = ~id_mask
    features = _feature_sets(data, id_mask, seed)
    metrics = {"samples": str(path), "n_samples": len(groups), "n_id": int(id_mask.sum()),
               "n_ood": int(ood_mask.sum()), "group_key": "axis_code,val_index,env_id,command_id,episode_id",
               "fast_mode": fast, "features": {}, "controls": {}, "intervention": {}}

    regression = dict(REGRESSION_TARGETS)
    for index, name in enumerate(PHYSICS):
        regression[name] = ("physics_raw", index)
    for feature_name, X in features.items():
        block = {"regression": {}, "classification": {}}
        for target_name, spec in regression.items():
            y = _target(data, spec).astype(float)
            # A physics target only varies inside its own axis sweep -- the
            # collector pins it to nominal everywhere else (that pinning is
            # exactly what GATE 1 verifies). Scoring friction on the whole bank
            # would therefore let the probe earn most of its R2 by emitting the
            # nominal constant on the 5/6 of rows belonging to other sweeps,
            # which says nothing about decodability. Restrict each physics
            # target to the rows where it was actually swept.
            rows = id_mask
            ood_rows = ood_mask
            if target_name in PHYSICS:
                on_axis = data["axis_code"] == PHYSICS.index(target_name)
                rows = id_mask & on_axis
                ood_rows = ood_mask & on_axis
            result = compute_decoder_metrics(X[rows], y[rows], groups[rows], ridge_kwargs=ridge_kwargs,
                                             shuffle_seed=seed)
            item = {"raw": result["linear"], "fold_summary": result["fold_summary"],
                    "n_rows": int(rows.sum())}
            if target_name in PHYSICS:
                item["delta_r2_vs_obs45"] = delta_r2(
                    data["obs"][rows], X[rows], y[rows], groups[rows], ridge_kwargs=ridge_kwargs,
                    shuffle_seed=seed)
                item["ood_unseen_level"] = _ood_regression(X, y, rows, ood_rows, groups,
                                                           ridge_kwargs=ridge_kwargs)
            block["regression"][target_name] = item
        for target_name in CATEGORICAL_TARGETS:
            y = data[target_name]
            result = compute_classifier_metrics(X[id_mask], y[id_mask], groups[id_mask],
                                                 max_train_rows=logistic_max_train,
                                                 shuffle_seed=seed,
                                                 max_iter=logistic_max_iter)
            block["classification"][target_name] = {
                "raw": result["oof"], "fold_summary": result["fold_summary"],
                "delta_balanced_accuracy_vs_obs45": delta_balanced_accuracy(
                    data["obs"][id_mask], X[id_mask], y[id_mask], groups[id_mask],
                    max_train_rows=logistic_max_train, shuffle_seed=seed,
                    max_iter=logistic_max_iter),
            }
        metrics["features"][feature_name] = block

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(data["physics_raw"][:, 0])
    # Both controls are stated "vs friction", so they must be scored on exactly
    # the rows the real friction probe uses -- otherwise a control computed on
    # the full bank is not comparable to the number it is meant to control.
    control_rows = id_mask & (data["axis_code"] == PHYSICS.index("friction"))
    shuffled_raw = compute_decoder_metrics(data["z_s"][control_rows], shuffled[control_rows],
                                            groups[control_rows], ridge_kwargs=ridge_kwargs,
                                            shuffle_seed=seed)
    shuffled_delta = delta_r2(data["obs"][control_rows], data["z_s"][control_rows],
                               shuffled[control_rows], groups[control_rows],
                               ridge_kwargs=ridge_kwargs, shuffle_seed=seed)
    random_target = metrics["features"]["random_init_latent"]["regression"]["friction"]
    random_delta = random_target["delta_r2_vs_obs45"]
    random_raw = random_target["raw"]

    metrics["controls"] = {
        **metrics["controls"],
        "threshold": 0.05,
        "shuffled_label_friction": {
            "delta_r2_vs_obs45": shuffled_delta,
            "raw_R2_z_s_to_shuffled_label": shuffled_raw["linear"],
        },
        "random_init_latent_friction": {
            "delta_r2_vs_obs45": random_delta,
            "raw_R2_obs45_plus_random_latent": random_raw,
        },
    }
    checks = [
        _gate_check("shuffled_label_friction", shuffled_delta, threshold=0.05),
        _gate_check("random_init_latent_friction", random_delta, threshold=0.05),
    ]
    statuses = [c["status"] for c in checks]
    overall = "FAIL" if "FAIL" in statuses else ("INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "PASS")
    metrics["controls"]["checks"] = checks
    metrics["controls"]["status"] = overall
    # Backward-compat boolean: only true when every control is a clean PASS.
    metrics["controls"]["pass"] = overall == "PASS"

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probe_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)
    _figures(metrics, out_dir)
    _report(metrics, out_dir)
    return metrics


def _figures(metrics, out_dir):
    features = list(metrics["features"])
    matrix = np.full((len(features), len(PHYSICS)), np.nan)
    for i, feature in enumerate(features):
        for j, target in enumerate(PHYSICS):
            matrix[i, j] = metrics["features"][feature]["regression"][target]["delta_r2_vs_obs45"]["mean"]
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-max(.05, np.nanmax(abs(matrix))),
                      vmax=max(.05, np.nanmax(abs(matrix))), aspect="auto")
    ax.set_xticks(range(len(PHYSICS)), PHYSICS, rotation=35, ha="right")
    ax.set_yticks(range(len(features)), features)
    fig.colorbar(image, ax=ax, label="Delta R2 vs obs45")
    fig.tight_layout(); fig.savefig(out_dir / "delta_r2_heatmap.png", dpi=150); plt.close(fig)


def _fmt(value):
    return "nan" if value is None or not np.isfinite(value) else f"{value:.4f}"


def _report(metrics, out_dir):
    controls = metrics["controls"]
    lines = ["# MoE-CTS Latent Probe Report", "",
             f"Samples: {metrics['n_samples']} (ID {metrics['n_id']}, OOD {metrics['n_ood']})", "",
             "## Leakage controls", "",
             f"Control gate: **{controls['status']}** (threshold {controls['threshold']}, one-sided: only "
             "delta_r2 > +threshold fails; NaN or delta_r2 < -threshold is INCONCLUSIVE, not FAIL)", "",
             "| Check | Status | delta_r2 mean | delta_r2 std | n_folds | raw R2 (full model) | Reason |",
             "|---|---|---:|---:|---:|---:|---|"]
    raw_lookup = {
        "shuffled_label_friction": controls["shuffled_label_friction"]["raw_R2_z_s_to_shuffled_label"]["R2"],
        "random_init_latent_friction": controls["random_init_latent_friction"]["raw_R2_obs45_plus_random_latent"]["R2"],
    }
    for check in controls["checks"]:
        lines.append(
            f"| {check['name']} | {check['status']} | {_fmt(check['delta_r2_mean'])} | "
            f"{_fmt(check['delta_r2_std'])} | {check['n_folds']} | {_fmt(raw_lookup.get(check['name']))} | "
            f"{check['reason']} |")
    lines += ["", "## Physics decodability (Delta R2 vs obs45)", "",
             "| Feature | " + " | ".join(PHYSICS) + " |",
             "|---|" + "---:|" * len(PHYSICS)]
    for feature, block in metrics["features"].items():
        values = [block["regression"][target]["delta_r2_vs_obs45"]["mean"] for target in PHYSICS]
        lines.append("| " + feature + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "Motor strength and zero offset are excluded because this checkout has no validated live setter/readback contract for them."]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def _trend(results, out_dir):
    tags, values = [], {name: [] for name in PHYSICS}
    for result in results:
        tag = Path(result["samples"]).parent.name; tags.append(tag)
        for target in PHYSICS:
            values[target].append(result["features"]["z_s"]["regression"][target]["delta_r2_vs_obs45"]["mean"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for target, series in values.items():
        ax.plot(range(len(tags)), series, marker="o", label=target)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(range(len(tags)), tags, rotation=30)
    ax.set_ylabel("Delta R2 (z_s over obs45)"); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "checkpoint_trend.png", dpi=150); plt.close(fig)


def _intervention(path):
    data = dict(np.load(path, allow_pickle=True)); result = {}
    for mode in np.unique(data["mode"]):
        mask = data["mode"] == mode
        result[str(mode)] = {key: float(np.mean(data[key][mask])) for key in
                             ("tracking_lin_err", "tracking_ang_err", "achieved_speed", "fall", "action_mae")}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--intervene")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--full_search", action="store_true",
                        help="Disable the fixed-alpha/capped-GD fast path and use the original "
                             "exhaustive inner alpha-search + max_iter=1000 logistic GD. Only "
                             "tractable on small (smoke-scale) banks; Faz A (80-100k rows) needs "
                             "the default fast path.")
    cli = parser.parse_args(); out = Path(cli.out_dir)
    results = []
    for path in cli.samples:
        result = analyze_one(Path(path), out / Path(path).parent.name, cli.seed, fast=not cli.full_search)
        if cli.intervene:
            result["intervention"] = _intervention(cli.intervene)
            with open(out / Path(path).parent.name / "probe_metrics.json", "w") as handle:
                json.dump(result, handle, indent=2, default=_json_default)
        results.append(result)
    if len(results) > 1:
        _trend(results, out)


if __name__ == "__main__":
    main()
