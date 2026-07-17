"""
FILE 2 — probe_rma_analyze.py
Pure CPU analysis of RMA latent-probe samples.

NO legged_gym / genesis imports at module top.
Primary deps: numpy, torch (for tiny MLP probe), scipy.stats, matplotlib (Agg).
sklearn is optional and used as a drop-in when present.

Usage:
    python probe_rma_analyze.py --samples <dir>/samples.npz --out_dir <dir>
    python probe_rma_analyze.py --samples samples.npz --switch switch_raw.npz \
                                 --intervene intervene.npz --out_dir results/

samples.npz schema (see PROBE_SPEC.md):
    P5_raw     (N, 5)   friction, added_mass, com_x, com_y, com_z  [physical units]
    P5_norm    (N, 5)   same, normalised & clamped to [-1,1]
    vel_raw    (N, 3)   base_lin_vel in m/s
    z_t        (N, 8)   teacher latent
    z_s        (N, 8)   student latent
    teacher_action (N, 12)
    student_action (N, 12)
    obs        (N, 45)  current observation — control probe (optional; omitted → skipped)
    command    (N, 3)
    command_id (N,)     int
    axis_code  (N,)     int  0=friction 1=added_mass 2=com_x 3=com_y 4=com_z
    physics_combo_id (N,)  int  CV group key = axis_code*100 + value_index
    fall       (N,)     bool
    tracking_lin_err (N,)
    tracking_ang_err (N,)
    step       (N,)     int
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Optional sklearn — gracefully absent on a115
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import Ridge as _SklearnRidge
    from sklearn.neural_network import MLPRegressor as _SklearnMLP
    from sklearn.model_selection import GroupKFold as _SklearnGKF
    from sklearn.preprocessing import StandardScaler as _SklearnScaler
    from sklearn.pipeline import Pipeline as _SklearnPipeline
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# torch is always present
import torch
import torch.nn as nn

# scipy — always present
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NORM_RANGES = {
    "friction":    (0.50, 1.25),
    "added_mass":  (-2.0, 5.0),
    "com_x":       (-0.08, 0.08),
    "com_y":       (-0.08, 0.08),
    "com_z":       (-0.08, 0.08),
}
TARGET_NAMES = ["friction", "added_mass", "com_x", "com_y", "com_z", "vx", "vy", "vz"]
AXIS_NAMES   = ["friction", "added_mass", "com_x", "com_y", "com_z"]
FEATURE_SETS = ["z_s", "z_t", "current_obs", "shuffled_label"]


# ---------------------------------------------------------------------------
# Normalization helpers (pure numpy — importable by tests)
# ---------------------------------------------------------------------------

def dr_normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalise raw physics value to [-1, 1] with clamp."""
    normed = 2.0 * (x - lo) / (hi - lo) - 1.0
    return np.clip(normed, -1.0, 1.0)


def dr_unnormalize(x_norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse of dr_normalize (no clamp applied — for round-trip testing)."""
    return (x_norm + 1.0) / 2.0 * (hi - lo) + lo


# ---------------------------------------------------------------------------
# GroupKFold — pure numpy implementation (always used; sklearn variant below)
# ---------------------------------------------------------------------------

def group_kfold_splits(groups: np.ndarray, n_splits: int):
    """
    Yield (train_indices, test_indices) for GroupKFold.
    Groups in each test fold are disjoint from train groups.
    Returns at most n_splits folds (may be fewer if n_groups < n_splits).
    """
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        return  # nothing to yield

    # Partition unique groups into actual_splits roughly equal buckets
    fold_assignments = np.array_split(unique_groups, actual_splits)
    for fold_idx in range(actual_splits):
        test_groups = set(fold_assignments[fold_idx].tolist())
        test_mask   = np.array([g in test_groups for g in groups], dtype=bool)
        train_mask  = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        # Assert disjointness (invariant)
        tr_g = set(groups[train_idx].tolist())
        te_g = set(groups[test_idx].tolist())
        assert tr_g.isdisjoint(te_g), (
            f"GroupKFold violation: groups {tr_g & te_g} in both train and test"
        )
        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Numpy ridge (closed-form)
# ---------------------------------------------------------------------------

class _NumpyRidgePipeline:
    """StandardScaler + closed-form ridge: w = (XᵀX + αI)⁻¹ Xᵀy."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None
        self.b_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyRidgePipeline":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_  = np.where(self.std_ < 1e-12, 1.0, self.std_)
        Xs = (X - self.mean_) / self.std_
        # Augment with bias column
        n, d = Xs.shape
        A = Xs.T @ Xs + self.alpha * np.eye(d)
        b = Xs.T @ y
        try:
            self.w_ = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.w_ = np.linalg.lstsq(A, b, rcond=None)[0]
        self.b_  = float(np.mean(y) - (self.mean_ / self.std_) @ self.w_)
        # Recompute bias properly: yhat = Xs @ w; b = mean(y - Xs@w)
        yhat = Xs @ self.w_
        self.b_ = float(np.mean(y - yhat))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        return Xs @ self.w_ + self.b_


def _best_ridge_numpy(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray,
    alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
) -> np.ndarray:
    """Pick best alpha by in-fold LOO estimate, then predict on test set."""
    best_alpha = 1.0
    best_val_err = float("inf")
    # Quick internal CV on training set (3-fold)
    n_tr = len(y_tr)
    if n_tr >= 10:
        fold_sz = max(1, n_tr // 3)
        for alpha in alphas:
            errs = []
            for start in range(0, n_tr, fold_sz):
                mask = np.zeros(n_tr, dtype=bool)
                mask[start:start+fold_sz] = True
                xi, xv = X_tr[~mask], X_tr[mask]
                yi, yv = y_tr[~mask], y_tr[mask]
                if len(xi) < 2:
                    continue
                pipe = _NumpyRidgePipeline(alpha=alpha)
                pipe.fit(xi, yi)
                errs.append(np.mean((pipe.predict(xv) - yv)**2))
            if errs and np.mean(errs) < best_val_err:
                best_val_err = np.mean(errs)
                best_alpha = alpha
    pipe = _NumpyRidgePipeline(alpha=best_alpha)
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te), pipe


# ---------------------------------------------------------------------------
# Torch MLP probe
# ---------------------------------------------------------------------------

class _TorchMLPProbe(nn.Module):
    def __init__(self, in_features: int, hidden: Tuple[int, ...] = (32, 8)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_torch_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    hidden: Tuple[int, ...] = (32, 8),
    epochs: int = 300,
    lr: float = 1e-3,
    patience: int = 20,
) -> np.ndarray:
    """Train a small MLP regressor and return predictions on X_te."""
    # Normalize
    mu_x  = X_tr.mean(axis=0)
    sig_x = X_tr.std(axis=0)
    sig_x = np.where(sig_x < 1e-12, 1.0, sig_x)
    mu_y  = float(y_tr.mean())
    sig_y = float(y_tr.std())
    if sig_y < 1e-12:
        sig_y = 1.0

    Xtr_n = torch.tensor((X_tr - mu_x) / sig_x, dtype=torch.float32)
    ytr_n = torch.tensor((y_tr - mu_y) / sig_y, dtype=torch.float32)
    Xte_n = torch.tensor((X_te - mu_x) / sig_x, dtype=torch.float32)

    model = _TorchMLPProbe(X_tr.shape[1], hidden=hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Simple train/val split (last 10%)
    n_val = max(1, int(0.1 * len(Xtr_n)))
    Xv, yv = Xtr_n[-n_val:], ytr_n[-n_val:]
    Xt, yt = Xtr_n[:-n_val], ytr_n[:-n_val]

    if len(Xt) < 2:
        Xt, yt, Xv, yv = Xtr_n, ytr_n, Xtr_n, ytr_n

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_n = model(Xte_n).numpy()
    return preds_n * sig_y + mu_y


# ---------------------------------------------------------------------------
# Latent metrics
# ---------------------------------------------------------------------------

def latent_r2(z_s: np.ndarray, z_t: np.ndarray) -> float:
    """
    latent_R2 = 1 - MSE(z_s, z_t) / Var(z_t)
    z_s == z_t  → 1.0;   z_s == mean(z_t)  → ~0.0
    """
    var_zt = float(np.var(z_t))
    if var_zt < 1e-12:
        return 1.0 if np.allclose(z_s, z_t) else 0.0
    mse = float(np.mean((z_s - z_t) ** 2))
    return float(1.0 - mse / var_zt)


def latent_mse(z_s: np.ndarray, z_t: np.ndarray) -> float:
    return float(np.mean((z_s - z_t) ** 2))


# ---------------------------------------------------------------------------
# Core decoder metrics
# ---------------------------------------------------------------------------

def compute_decoder_metrics(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Dict:
    """
    Train frozen out-of-fold Ridge and MLP decoders on feature matrix X
    predicting target y, using GroupKFold split on `groups`.

    Returns dict with keys:
        "linear"  / "nonlinear"  — dicts of R2, nRMSE, MAE, spearman_rho,
                                    calib_slope, calib_intercept
        "y_true_oof"             — OOF true values (valid entries only)
        "y_pred_linear"          — OOF linear predictions
        "y_pred_nonlinear"       — OOF MLP predictions
        "idx_oof"                — indices into original arrays
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)

    n_groups = len(np.unique(groups))
    n_splits_actual = min(n_splits, n_groups)

    nan_metric = {
        "R2": float("nan"), "nRMSE": float("nan"), "MAE": float("nan"),
        "spearman_rho": float("nan"), "calib_slope": float("nan"),
        "calib_intercept": float("nan"),
    }
    if n_splits_actual < 2:
        logging.warning("compute_decoder_metrics: fewer than 2 groups — returning NaN")
        return {
            "linear": nan_metric, "nonlinear": nan_metric,
            "y_true_oof": y, "idx_oof": np.arange(len(y)),
            "y_pred_linear": np.full_like(y, float("nan")),
            "y_pred_nonlinear": np.full_like(y, float("nan")),
        }

    y_pred_lin = np.full(len(y), float("nan"), dtype=np.float64)
    y_pred_mlp = np.full(len(y), float("nan"), dtype=np.float64)

    for train_idx, test_idx in group_kfold_splits(groups, n_splits_actual):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        # Linear ridge
        preds_lin, _ = _best_ridge_numpy(X_tr, y_tr, X_te)
        y_pred_lin[test_idx] = preds_lin

        # Nonlinear MLP
        try:
            preds_mlp = _train_torch_mlp(X_tr, y_tr, X_te, hidden=(32, 8))
        except Exception as e:
            logging.warning(f"MLP failed in fold: {e}")
            preds_mlp = preds_lin  # fallback to linear
        y_pred_mlp[test_idx] = preds_mlp

    # Select valid OOF entries
    valid = np.isfinite(y_pred_lin) & np.isfinite(y_pred_mlp)
    yt    = y[valid]
    idx_v = np.where(valid)[0]

    def _metrics_from(yt_: np.ndarray, yp_: np.ndarray) -> Dict:
        if len(yt_) < 2:
            return dict(nan_metric)
        ss_res = float(np.sum((yt_ - yp_) ** 2))
        ss_tot = float(np.sum((yt_ - yt_.mean()) ** 2))
        r2     = 1.0 - ss_res / max(ss_tot, 1e-12)
        rmse   = float(np.sqrt(np.mean((yt_ - yp_) ** 2)))
        nrmse  = rmse / float(np.std(yt_)) if float(np.std(yt_)) > 1e-12 else float("inf")
        mae    = float(np.mean(np.abs(yt_ - yp_)))
        rho, _ = spearmanr(yt_, yp_)
        if np.std(yp_) > 1e-12:
            calib = np.polyfit(yp_, yt_, 1)
            cs, ci = float(calib[0]), float(calib[1])
        else:
            cs, ci = float("nan"), float("nan")
        return {
            "R2": float(r2), "nRMSE": float(nrmse), "MAE": float(mae),
            "spearman_rho": float(rho), "calib_slope": cs, "calib_intercept": ci,
        }

    return {
        "linear":    _metrics_from(yt, y_pred_lin[valid]),
        "nonlinear": _metrics_from(yt, y_pred_mlp[valid]),
        "y_true_oof":      yt,
        "y_pred_linear":   y_pred_lin[valid],
        "y_pred_nonlinear": y_pred_mlp[valid],
        "idx_oof":         idx_v,
    }


# ---------------------------------------------------------------------------
# Switch curve helpers
# ---------------------------------------------------------------------------

def enter_band_time(
    decoded: np.ndarray,
    target: float,
    band_frac: float = 0.20,
    t0_idx: int = 0,
) -> Tuple[Optional[int], float]:
    """
    Return (first_index_in_band_after_t0, dwell_fraction_from_there).
    Band = target ± max(band_frac*|target|, 0.05).
    Returns (None, 0.0) if never enters band.
    """
    tol = max(abs(target) * band_frac, 0.05)
    in_band = np.abs(decoded[t0_idx:] - target) < tol
    first_enter = None
    for i, v in enumerate(in_band):
        if v:
            first_enter = t0_idx + i
            break
    if first_enter is None:
        return None, 0.0
    dwell = float(np.mean(in_band[first_enter - t0_idx:]))
    return first_enter, dwell


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    required = [
        "P5_raw", "P5_norm", "vel_raw", "z_t", "z_s",
        "teacher_action", "student_action",
        "command_id", "axis_code", "physics_combo_id",
    ]
    d: Dict[str, np.ndarray] = {k: data[k] for k in data.files}
    missing = [r for r in required if r not in d]
    if missing:
        raise KeyError(f"samples.npz missing required keys: {missing}")
    return d


def _get_feature(d: Dict[str, np.ndarray], fset: str) -> Optional[np.ndarray]:
    if fset == "z_s":
        return d["z_s"].astype(np.float64)
    if fset == "z_t":
        return d["z_t"].astype(np.float64)
    if fset == "current_obs":
        if "obs" not in d:
            logging.warning("current_obs requested but 'obs' not in samples.npz — skipping")
            return None
        return d["obs"].astype(np.float64)
    if fset == "shuffled_label":
        return d["z_s"].astype(np.float64)
    raise ValueError(f"Unknown feature set: {fset}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(
    samples_path: str,
    out_dir: str,
    switch_path: Optional[str] = None,
    intervene_path: Optional[str] = None,
    seed_label: str = "unknown",
    n_splits: int = 5,
):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    logging.info(f"Loading samples from {samples_path}")
    d = load_samples(samples_path)
    N = d["P5_raw"].shape[0]
    logging.info(f"  {N} samples loaded")

    targets_raw  = np.concatenate([d["P5_raw"], d["vel_raw"]], axis=1)  # (N, 8)
    groups       = d["physics_combo_id"].astype(np.int32).ravel()
    axis_codes   = d["axis_code"].astype(np.int32).ravel()
    command_ids  = d["command_id"].astype(np.int32).ravel()
    z_s          = d["z_s"].astype(np.float64)
    z_t          = d["z_t"].astype(np.float64)
    stu_actions  = d["student_action"].astype(np.float64)
    tea_actions  = d["teacher_action"].astype(np.float64)

    # ---- Latent metrics ------------------------------------------------
    lmse  = latent_mse(z_s, z_t)
    lr2   = latent_r2(z_s, z_t)
    a_mae = float(np.mean(np.abs(stu_actions - tea_actions)))

    by_axis: Dict[str, float] = {}
    for ac, ac_name in enumerate(AXIS_NAMES):
        m = axis_codes == ac
        if m.sum() > 0:
            by_axis[ac_name] = latent_r2(z_s[m], z_t[m])
    by_cmd: Dict[str, float] = {}
    for cid in np.unique(command_ids):
        m = command_ids == cid
        if m.sum() > 0:
            by_cmd[str(int(cid))] = latent_r2(z_s[m], z_t[m])
    amae_by_axis: Dict[str, float] = {}
    for ac, ac_name in enumerate(AXIS_NAMES):
        m = axis_codes == ac
        if m.sum() > 0:
            amae_by_axis[ac_name] = float(np.mean(np.abs(stu_actions[m] - tea_actions[m])))
    amae_by_cmd: Dict[str, float] = {}
    for cid in np.unique(command_ids):
        m = command_ids == cid
        if m.sum() > 0:
            amae_by_cmd[str(int(cid))] = float(np.mean(np.abs(stu_actions[m] - tea_actions[m])))

    latent_block = {
        "latent_mse":            lmse,
        "latent_R2":             lr2,
        "action_mae":            a_mae,
        "by_axis":               by_axis,
        "by_command":            by_cmd,
        "action_mae_by_axis":    amae_by_axis,
        "action_mae_by_command": amae_by_cmd,
    }
    logging.info(f"  latent_R2={lr2:.4f}  action_mae={a_mae:.4f}")

    # ---- Decoder probes ------------------------------------------------
    rng = np.random.default_rng(42)
    decoders_block: Dict = {}
    scatter_data: Dict  = {}

    for fset in FEATURE_SETS:
        X = _get_feature(d, fset)
        if X is None:
            continue

        fset_metrics: Dict = {}
        scatter_data[fset] = {}

        for ti, tname in enumerate(TARGET_NAMES):
            y_raw = targets_raw[:, ti].astype(np.float64)
            y = y_raw if fset != "shuffled_label" else y_raw[rng.permutation(len(y_raw))]

            try:
                res = compute_decoder_metrics(X, y, groups, n_splits=n_splits)
            except Exception as e:
                logging.warning(f"Decoder {fset}/{tname}: {e}")
                res = {"linear": {}, "nonlinear": {}}

            fset_metrics[tname] = {
                "linear":    res.get("linear",    {}),
                "nonlinear": res.get("nonlinear", {}),
            }
            if fset in ("z_s", "z_t"):
                scatter_data[fset][tname] = (
                    res.get("y_true_oof",       np.array([])),
                    res.get("y_pred_linear",    np.array([])),
                    res.get("y_pred_nonlinear", np.array([])),
                )

        decoders_block[fset] = fset_metrics

    # ---- Figures -------------------------------------------------------
    _make_scatter_figures(scatter_data, fig_dir)
    _make_r2_bar_figure(decoders_block, fig_dir)

    # ---- Switch analysis -----------------------------------------------
    switch_block: Optional[Dict] = None
    if switch_path is not None:
        switch_block = _analyze_switch(switch_path, d, groups, out_dir, fig_dir)

    # ---- Intervene analysis --------------------------------------------
    intervene_block: Optional[Dict] = None
    if intervene_path is not None:
        intervene_block = _analyze_intervene(intervene_path, out_dir, fig_dir)

    # ---- Assemble metrics ----------------------------------------------
    metrics = {
        "seed_label": seed_label,
        "n_samples":  int(N),
        "latent":     latent_block,
        "decoders":   decoders_block,
    }
    if switch_block is not None:
        metrics["switch"] = switch_block
    if intervene_block is not None:
        metrics["intervene"] = intervene_block

    mpath = os.path.join(out_dir, "probe_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    logging.info(f"Saved probe_metrics.json → {mpath}")

    _write_report(metrics, out_dir)
    return metrics


def _json_default(obj):
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.float32, np.float64)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.int32, np.int64, np.intp, np.int_)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_scatter_figures(scatter_data: Dict, fig_dir: str):
    for tname in TARGET_NAMES:
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(f"Decoder scatter — {tname}")
        combos = [
            ("z_s", "linear",    axes[0, 0]),
            ("z_s", "nonlinear", axes[0, 1]),
            ("z_t", "linear",    axes[1, 0]),
            ("z_t", "nonlinear", axes[1, 1]),
        ]
        for fset, dtype, ax in combos:
            ax.set_title(f"{fset} {dtype}")
            ax.set_xlabel("y_pred")
            ax.set_ylabel("y_true")
            if fset not in scatter_data or tname not in scatter_data[fset]:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
                continue
            yt, yp_lin, yp_mlp = scatter_data[fset][tname]
            yp = yp_lin if dtype == "linear" else yp_mlp
            if len(yt) == 0:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
                continue
            ax.scatter(yp, yt, s=2, alpha=0.3, rasterized=True)
            lo = min(yp.min(), yt.min()); hi = max(yp.max(), yt.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="ideal")
            if np.std(yp) > 1e-12:
                calib = np.polyfit(yp, yt, 1)
                xs = np.linspace(yp.min(), yp.max(), 50)
                ax.plot(xs, np.polyval(calib, xs), "g-", lw=1,
                        label=f"fit {calib[0]:.2f}x+{calib[1]:.2f}")
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"scatter_{tname}.png"), dpi=100)
        plt.close(fig)


def _make_r2_bar_figure(decoders_block: Dict, fig_dir: str):
    fsets = ["z_s", "z_t", "shuffled_label"]
    avail = [f for f in fsets if f in decoders_block]
    if not avail:
        return
    x = np.arange(len(TARGET_NAMES))
    w = 0.25
    colors = ["steelblue", "darkorange", "gray"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, fset in enumerate(avail):
        vals = [
            decoders_block[fset].get(t, {}).get("linear", {}).get("R2", float("nan"))
            for t in TARGET_NAMES
        ]
        vals = [v if v is not None else float("nan") for v in vals]
        ax.bar(x + i * w, vals, w, label=fset, color=colors[i % len(colors)], alpha=0.8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(TARGET_NAMES, rotation=30, ha="right")
    ax.set_ylabel("R² (linear decoder, GroupKFold OOF)")
    ax.set_title("Probe R² by feature set and target")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "r2_bar.png"), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Switch analysis
# ---------------------------------------------------------------------------

def _analyze_switch(
    switch_path: str,
    grid_d: Dict[str, np.ndarray],
    grid_groups: np.ndarray,
    out_dir: str,
    fig_dir: str,
) -> Dict:
    import csv

    sw = np.load(switch_path, allow_pickle=True)
    if "z_s_mean" not in sw:
        logging.warning("switch_raw.npz missing z_s_mean — skipping switch analysis")
        return {"error": "no z_s_mean in switch_raw.npz"}

    T             = sw["z_s_mean"].shape[0]
    z_s_sw        = sw["z_s_mean"].astype(np.float64)
    real_mass_raw = sw["real_mass_raw"].astype(np.float64) if "real_mass_raw" in sw else np.zeros(T)
    latent_err_sw = sw["latent_err"].astype(np.float64)    if "latent_err"   in sw else np.zeros(T)
    action_mae_sw = sw["action_mae"].astype(np.float64)    if "action_mae"   in sw else np.zeros(T)
    tracking_sw   = sw["tracking_lin_err"].astype(np.float64) if "tracking_lin_err" in sw else np.zeros(T)
    switch_step   = int(sw["switch_step"]) if "switch_step" in sw else T // 2

    # Fit best z_s → added_mass decoder on grid samples
    X_grid = grid_d["z_s"].astype(np.float64)
    y_grid = grid_d["P5_raw"][:, 1].astype(np.float64)   # added_mass index=1
    _, ridge = _best_ridge_numpy(X_grid, y_grid, X_grid)  # fit on full grid
    decoded_mass = ridge.predict(z_s_sw)                  # (T,)

    t_arr     = np.arange(T)
    pre_mask  = t_arr < switch_step
    post_mask = t_arr >= switch_step

    pre_steady_err  = float(np.mean(np.abs(decoded_mass[pre_mask]  - real_mass_raw[pre_mask])))  if pre_mask.any()  else float("nan")
    post_max_err    = float(np.max (np.abs(decoded_mass[post_mask] - real_mass_raw[post_mask])))  if post_mask.any() else float("nan")
    late_steady_err = float(np.mean(np.abs(decoded_mass[-20:]      - real_mass_raw[-20:])))

    new_mass         = float(real_mass_raw[-1]) if post_mask.any() else 4.0
    first_in_band, dwell = enter_band_time(decoded_mass, new_mass, band_frac=0.20, t0_idx=switch_step)
    time_to_band     = (first_in_band - switch_step) if first_in_band is not None else None
    overshoot        = float(np.max(decoded_mass[post_mask]) - new_mass) if post_mask.any() else float("nan")

    pre_le_mean = float(np.mean(latent_err_sw[pre_mask])) if pre_mask.any() else 0.0
    le_recovery = None
    for t in range(switch_step, T):
        if latent_err_sw[t] <= pre_le_mean * 1.2:
            le_recovery = int(t - switch_step)
            break

    sw_metrics = {
        "pre_steady_err":              pre_steady_err,
        "post_max_err":                post_max_err,
        "late_steady_err":             late_steady_err,
        "time_to_enter_band_steps":    time_to_band,
        "dwell_frac":                  dwell,
        "overshoot_kg":                overshoot,
        "latent_err_recovery_steps":   le_recovery,
    }

    csv_path = os.path.join(out_dir, "switch_curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "real_mass", "decoded_mass", "latent_err", "action_mae", "tracking_err"])
        for i in range(T):
            writer.writerow([i, real_mass_raw[i], decoded_mass[i],
                             latent_err_sw[i], action_mae_sw[i], tracking_sw[i]])
    logging.info(f"Saved switch_curves.csv → {csv_path}")

    _make_switch_figure(t_arr, real_mass_raw, decoded_mass, latent_err_sw,
                        action_mae_sw, tracking_sw, switch_step, fig_dir)
    return sw_metrics


def _make_switch_figure(t, real_mass, decoded_mass, latent_err, action_mae, tracking_err,
                        switch_step: int, fig_dir: str):
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    fig.suptitle("Switch 0→+4 kg: per-step evolution")
    axes[0].plot(t, real_mass,    label="real mass", color="k")
    axes[0].plot(t, decoded_mass, label="decoded mass", color="steelblue")
    axes[0].axvline(switch_step, color="r", ls="--", lw=0.8, label="switch")
    axes[0].set_ylabel("added mass (kg)")
    axes[0].legend(fontsize=7)
    axes[1].plot(t, np.abs(decoded_mass - real_mass), color="orange")
    axes[1].set_ylabel("|decode err| (kg)")
    axes[2].plot(t, latent_err, color="purple")
    axes[2].set_ylabel("||z_s - z_t||")
    axes[3].plot(t, action_mae, color="green")
    axes[3].set_ylabel("action MAE")
    axes[4].plot(t, tracking_err, color="red")
    axes[4].set_ylabel("tracking_lin_err")
    axes[4].set_xlabel("step")
    for ax in axes:
        ax.axvline(switch_step, color="r", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "switch_curves.png"), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Intervene analysis
# ---------------------------------------------------------------------------

def _analyze_intervene(intervene_path: str, out_dir: str, fig_dir: str) -> Dict:
    iv  = np.load(intervene_path, allow_pickle=True)
    modes = ["student", "teacher_true", "teacher_nominal", "teacher_wrong"]
    block: Dict[str, object] = {}
    for mode in modes:
        entry: Dict[str, float] = {}
        for metric in ["tracking_lin_err", "fall_rate", "achieved_speed_ratio", "action_rate"]:
            # collector flattens aggregate() output as "<mode>__<metric>_mean"
            key = f"{mode}__{metric}_mean"
            if key in iv.files:
                entry[metric] = float(np.mean(iv[key]))
        if not entry and mode in iv.files:
            arr = iv[mode]
            if arr.ndim == 0:
                val = arr.item()
                if isinstance(val, dict):
                    entry = {k: float(v) for k, v in val.items()}
        block[mode] = entry

    valid_modes = [m for m in modes if isinstance(block.get(m), dict)
                   and "tracking_lin_err" in block[m]]
    if valid_modes:
        ordering = " < ".join(
            sorted(valid_modes, key=lambda m: block[m]["tracking_lin_err"])
        )
    else:
        ordering = "no tracking data available"
    block["ordering_note"] = (
        "Expected: teacher_true ≲ student < nominal < wrong. "
        f"Observed (low→high tracking err): {ordering}"
    )

    _make_intervene_figure(block, modes, fig_dir)
    logging.info(f"Intervene ordering: {ordering}")
    return block


def _make_intervene_figure(block: Dict, modes: List[str], fig_dir: str):
    metrics_to_plot = ["tracking_lin_err", "fall_rate", "achieved_speed_ratio"]
    colors = ["steelblue", "darkorange", "gray", "crimson"]
    x = np.arange(len(modes))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Causal intervention: per-mode metrics")
    for mi, metric in enumerate(metrics_to_plot):
        ax = axes[mi]
        vals = []
        for m in modes:
            ent = block.get(m, {})
            vals.append(ent.get(metric, float("nan")) if isinstance(ent, dict) else float("nan"))
        ax.bar(x, vals, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric.replace("_", " "))
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "intervene_bar.png"), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(metrics: Dict, out_dir: str):
    lines = []
    a = lines.append
    a("# RMA Latent Probe Report")
    a("")
    a(f"**Seed label**: {metrics.get('seed_label', '?')}   "
      f"**N samples**: {metrics.get('n_samples', '?')}")
    a("")
    a("## Gate 1 — Distillation fidelity (does z_s track z_t?)")
    lat = metrics.get("latent", {})
    lr2  = lat.get("latent_R2",  "N/A")
    lmse_ = lat.get("latent_mse", "N/A")
    amae  = lat.get("action_mae", "N/A")
    a(f"- latent R² = {lr2}")
    a(f"- latent MSE = {lmse_}")
    a(f"- action MAE (student vs teacher) = {amae}")
    a("")
    a("By axis:")
    for k, v in lat.get("by_axis", {}).items():
        a(f"  - {k}: R²={v:.4f}")
    a("")
    a("## Gate 2 — ID decodability (does z_s encode physics?)")
    dec = metrics.get("decoders", {})
    for fset in ["z_s", "z_t", "shuffled_label"]:
        if fset not in dec:
            continue
        a(f"\n### Feature set: {fset}")
        a(f"{'Target':<14} {'Ridge R²':>10} {'MLP R²':>10} {'Ridge nRMSE':>12}")
        a("-" * 50)
        for tname in TARGET_NAMES:
            if tname not in dec[fset]:
                continue
            lin  = dec[fset][tname].get("linear",    {})
            mlp  = dec[fset][tname].get("nonlinear", {})
            r2l  = lin.get("R2",    float("nan"))
            r2m  = mlp.get("R2",    float("nan"))
            nrms = lin.get("nRMSE", float("nan"))
            f_r2l  = f"{r2l:.4f}"  if isinstance(r2l,  float) and not np.isnan(r2l)  else "NaN"
            f_r2m  = f"{r2m:.4f}"  if isinstance(r2m,  float) and not np.isnan(r2m)  else "NaN"
            f_nrms = f"{nrms:.4f}" if isinstance(nrms, float) and not np.isnan(nrms) else "NaN"
            a(f"{tname:<14} {f_r2l:>10} {f_r2m:>10} {f_nrms:>12}")
    a("")
    a("## Gate 3 — Dynamic tracking (switch 0→+4 kg)")
    sw = metrics.get("switch", {})
    if sw and "error" not in sw:
        for k, v in sw.items():
            a(f"- {k}: {v}")
    else:
        a("*(switch data not provided or errored)*")
    a("")
    a("## Gate 4 — Causal use (intervention experiment)")
    iv = metrics.get("intervene", {})
    if iv:
        a(iv.get("ordering_note", ""))
        a("")
        for mode in ["teacher_true", "student", "teacher_nominal", "teacher_wrong"]:
            if mode in iv and isinstance(iv[mode], dict):
                te = iv[mode].get("tracking_lin_err", "N/A")
                fr = iv[mode].get("fall_rate",        "N/A")
                a(f"- **{mode}**: tracking_err={te}  fall_rate={fr}")
    else:
        a("*(intervene data not provided)*")
    a("")
    a("## Interpretation table")
    a("| Gate | Pass criterion | Value |")
    a("|------|----------------|-------|")
    g1 = "PASS" if isinstance(lr2, float) and lr2 > 0.7 else "CHECK"
    a(f"| Distillation     | latent_R2 > 0.7     | {g1} ({lr2}) |")
    zs_r2 = dec.get("z_s", {}).get("added_mass", {}).get("linear", {}).get("R2", float("nan"))
    g2 = "PASS" if isinstance(zs_r2, float) and not np.isnan(zs_r2) and zs_r2 > 0.5 else "CHECK"
    a(f"| ID decodability  | z_s mass R2 > 0.5   | {g2} ({zs_r2}) |")
    ttb = metrics.get("switch", {}).get("time_to_enter_band_steps")
    g3 = f"PASS ({ttb} steps)" if ttb is not None and ttb < 50 else f"CHECK ({ttb})"
    a(f"| Dynamic tracking | time_to_band < 50   | {g3} |")
    order = iv.get("ordering_note", "N/A") if iv else "N/A"
    a(f"| Causal use       | student ≲ teacher_true >> wrong | {order} |")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logging.info(f"Saved report.md → {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="RMA latent probe analysis (numpy/torch/scipy, no env import)"
    )
    p.add_argument("--samples",    required=True)
    p.add_argument("--switch",     default=None)
    p.add_argument("--intervene",  default=None)
    p.add_argument("--out_dir",    default="probe_results")
    p.add_argument("--seed_label", default="unknown")
    p.add_argument("--n_splits",   type=int, default=5)
    p.add_argument("--verbose",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    run_analysis(
        samples_path=args.samples,
        out_dir=args.out_dir,
        switch_path=args.switch,
        intervene_path=args.intervene,
        seed_label=args.seed_label,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()
