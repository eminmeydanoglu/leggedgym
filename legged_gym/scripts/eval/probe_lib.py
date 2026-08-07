"""CPU-only utilities shared by latent probing scripts.

This module deliberately imports no simulator or legged_gym environment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata, spearmanr


NORM_RANGES = {
    "friction": (0.50, 1.25),
    "added_mass": (-2.0, 5.0),
    "com_x": (-0.08, 0.08),
    "com_y": (-0.08, 0.08),
    "com_z": (-0.08, 0.08),
    "pd_gain_scale": (0.50, 1.50),
    "control_delay": (0.0, 6.0),
}


def dr_normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(x)
    return np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def dr_unnormalize(x_norm: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.asarray(x_norm) + 1.0) * 0.5 * (hi - lo) + lo


def group_kfold_splits(groups: np.ndarray, n_splits: int, shuffle_seed=None):
    """Deterministic GroupKFold with explicit train/test disjointness checks.

    ``shuffle_seed=None`` keeps the historical behaviour: ``np.unique`` returns
    the group keys *sorted*, and the folds are contiguous blocks of that sorted
    order. That is safe only when the sort order carries no information about
    the target. It does not hold for the MoE controlled-grid banks, whose group
    key starts with ``axis_code:val_index``: sorting then orders groups by
    swept axis and physics cell, so a contiguous block is (nearly) one axis's
    sweep. Probing e.g. friction, most test folds then contain no friction-sweep
    rows at all -- friction sits pinned at its nominal value, the fold's target
    variance collapses, and R2 = 1 - SS_res/SS_tot explodes (observed: -1e12).
    Passing a ``shuffle_seed`` permutes the unique group keys before splitting,
    so every fold mixes axes and cells while whole rollouts stay intact.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    actual = min(int(n_splits), len(unique))
    if actual < 2:
        return
    if shuffle_seed is not None:
        unique = np.random.default_rng(shuffle_seed).permutation(unique)
    for test_groups in np.array_split(unique, actual):
        test_mask = np.isin(groups, np.asarray(test_groups, dtype=groups.dtype))
        train_idx, test_idx = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        if not len(train_idx) or not len(test_idx):
            continue
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        yield train_idx, test_idx


class _NumpyRidgePipeline:
    """StandardScaler plus closed-form ridge regression."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = X.mean(0)
        self.std_ = np.where(X.std(0) < 1e-12, 1.0, X.std(0))
        Xs = (X - self.mean_) / self.std_
        A = Xs.T @ Xs + self.alpha * np.eye(X.shape[1])
        B = Xs.T @ y
        try:
            self.w_ = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.w_ = np.linalg.lstsq(A, B, rcond=None)[0]
        self.b_ = np.mean(y - Xs @ self.w_, axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X) - self.mean_) / self.std_) @ self.w_ + self.b_


def _best_ridge_numpy(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    n_inner_folds: int = 3,
):
    """Choose alpha using deterministic inner ``n_inner_folds``-fold CV.

    ``alphas``/``n_inner_folds`` default to the original grid (5 alphas x 3
    inner folds) so existing callers -- in particular ``probe_rma_analyze.py``,
    whose published ``probe_metrics.json`` regression baselines this default
    must keep reproducing exactly -- are unaffected. Callers operating at
    much larger row counts (the MoE-CTS controlled-grid banks, tens to
    hundreds of thousands of rows) can pass a smaller ``alphas`` sequence
    (e.g. a single fixed value) and/or ``n_inner_folds=1`` to skip the O(15x)
    inner-CV multiplier, which is the dominant cost at that scale (ridge
    itself is O(n*d^2) and cheap; it's *refitting it ~16x per outer fold*
    for alpha selection that doesn't scale).
    """
    n = len(y_tr)
    best_alpha, best_error = 1.0, float("inf")
    if n >= 10 and len(alphas) > 1:
        for alpha in alphas:
            errors = []
            fold_size = max(1, n // max(1, n_inner_folds))
            for start in range(0, n, fold_size):
                val = np.zeros(n, dtype=bool)
                val[start:start + fold_size] = True
                if np.sum(~val) < 2:
                    continue
                model = _NumpyRidgePipeline(alpha).fit(X_tr[~val], y_tr[~val])
                errors.append(float(np.mean((model.predict(X_tr[val]) - y_tr[val]) ** 2)))
            error = float(np.mean(errors))
            if error < best_error:
                best_alpha, best_error = float(alpha), error
    elif len(alphas) == 1:
        best_alpha = float(alphas[0])
    model = _NumpyRidgePipeline(best_alpha).fit(X_tr, y_tr)
    return model.predict(X_te), model


def _regression_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y, pred = np.asarray(y), np.asarray(pred)
    variance = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / max(variance, 1e-12)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    std = float(np.std(y))
    rho = spearmanr(y, pred).statistic if len(y) > 1 else np.nan
    slope, intercept = (np.polyfit(pred, y, 1) if np.std(pred) > 1e-12 else (np.nan, np.nan))
    return {"R2": r2, "nRMSE": rmse / std if std > 1e-12 else np.inf,
            "MAE": float(np.mean(np.abs(y - pred))), "spearman_rho": float(rho),
            "calib_slope": float(slope), "calib_intercept": float(intercept)}


def compute_decoder_metrics(X, y, groups, n_splits: int = 5,
                            nonlinear_predictor=None, ridge_kwargs: Optional[Dict] = None,
                            shuffle_seed=None) -> Dict:
    """Group-aware OOF metrics, including fold mean and std.

    ``nonlinear_predictor`` is an optional ``(X_train, y_train, X_test)``
    callable used by the legacy RMA MLP path. ``ridge_kwargs`` (e.g.
    ``{"alphas": (1.0,), "n_inner_folds": 1}``) is forwarded to
    ``_best_ridge_numpy`` to skip its inner alpha-search CV at large row
    counts; omit it to keep the exact historical RMA behavior.
    """
    ridge_kwargs = ridge_kwargs or {}
    X, y, groups = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups)
    pred = np.full(len(y), np.nan)
    nonlinear_pred = np.full(len(y), np.nan)
    folds = []
    for fold, (tr, te) in enumerate(group_kfold_splits(groups, n_splits, shuffle_seed) or []):
        pred[te], model = _best_ridge_numpy(X[tr], y[tr], X[te], **ridge_kwargs)
        if nonlinear_predictor is None:
            nonlinear_pred[te] = pred[te]
        else:
            try:
                nonlinear_pred[te] = nonlinear_predictor(X[tr], y[tr], X[te])
            except Exception:
                nonlinear_pred[te] = pred[te]
        folds.append({"fold": fold, "n_train": len(tr), "n_test": len(te),
                      "alpha": model.alpha, **_regression_metrics(y[te], pred[te])})
    valid = np.isfinite(pred) & np.isfinite(nonlinear_pred)
    nan = {k: np.nan for k in ("R2", "nRMSE", "MAE", "spearman_rho", "calib_slope", "calib_intercept")}
    linear = _regression_metrics(y[valid], pred[valid]) if valid.sum() > 1 else nan
    fold_summary = {}
    for key in nan:
        vals = np.asarray([f[key] for f in folds], dtype=float)
        fold_summary[key] = {"mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                             "std": float(np.nanstd(vals)) if len(vals) else np.nan}
    nonlinear = _regression_metrics(y[valid], nonlinear_pred[valid]) if valid.sum() > 1 else nan
    return {"linear": linear, "nonlinear": nonlinear, "folds": folds,
            "fold_summary": fold_summary, "y_true_oof": y[valid],
            "y_pred_linear": pred[valid], "y_pred_nonlinear": nonlinear_pred[valid],
            "idx_oof": np.flatnonzero(valid)}


def delta_r2(feat_a, feat_b, y, groups, n_splits: int = 5, ridge_kwargs: Optional[Dict] = None,
             shuffle_seed=None) -> Dict:
    """Foldwise R2([feat_a, feat_b]) - R2(feat_a), without OOF leakage.

    ``ridge_kwargs`` is forwarded to ``_best_ridge_numpy``; see
    ``compute_decoder_metrics`` for the large-scale-run rationale.
    """
    ridge_kwargs = ridge_kwargs or {}
    a, b, y = np.asarray(feat_a), np.asarray(feat_b), np.asarray(y)
    values = []
    for tr, te in group_kfold_splits(groups, n_splits, shuffle_seed) or []:
        pa, _ = _best_ridge_numpy(a[tr], y[tr], a[te], **ridge_kwargs)
        pab, _ = _best_ridge_numpy(np.concatenate([a[tr], b[tr]], 1), y[tr],
                                   np.concatenate([a[te], b[te]], 1), **ridge_kwargs)
        values.append(_regression_metrics(y[te], pab)["R2"] - _regression_metrics(y[te], pa)["R2"])
    return {"mean": float(np.mean(values)) if values else np.nan,
            "std": float(np.std(values)) if values else np.nan,
            "fold_values": [float(v) for v in values], "n_folds": len(values)}


def _softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


class NumpyLogisticProbe:
    """Multiclass softmax regression with L2 and full-batch Adam."""

    def __init__(self, l2: float = 1e-3, lr: float = 0.05, max_iter: int = 1000,
                 tol: float = 1e-8):
        self.l2, self.lr, self.max_iter, self.tol = l2, lr, max_iter, tol

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.mean_, self.std_ = X.mean(0), np.where(X.std(0) < 1e-12, 1.0, X.std(0))
        Xs = (X - self.mean_) / self.std_
        Xb = np.concatenate([Xs, np.ones((len(Xs), 1))], 1)
        W = np.zeros((Xb.shape[1], len(self.classes_)))
        m = np.zeros_like(W); v = np.zeros_like(W)
        onehot = np.eye(len(self.classes_))[yi]
        previous = np.inf
        for t in range(1, self.max_iter + 1):
            p = _softmax(Xb @ W)
            loss = -np.mean(np.log(p[np.arange(len(y)), yi] + 1e-12)) + 0.5 * self.l2 * np.sum(W[:-1] ** 2)
            grad = Xb.T @ (p - onehot) / len(y)
            grad[:-1] += self.l2 * W[:-1]
            m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad * grad
            W -= self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            if abs(previous - loss) < self.tol:
                break
            previous = loss
        self.coef_ = W
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X) - self.mean_) / self.std_
        return _softmax(np.concatenate([Xs, np.ones((len(Xs), 1))], 1) @ self.coef_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _macro_auc(y, proba, classes):
    aucs = []
    for col, cls in enumerate(classes):
        pos = y == cls
        if not pos.any() or pos.all():
            continue
        ranks = rankdata(proba[:, col])
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        aucs.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else np.nan


def classification_metrics(y, pred, proba=None, classes=None) -> Dict[str, float]:
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) == 0:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan,
                "macro_auc": np.nan, "majority_baseline": np.nan}
    classes = np.unique(y) if classes is None else np.asarray(classes)
    recalls = [np.mean(pred[y == c] == c) for c in classes if np.any(y == c)]
    counts = np.asarray([np.sum(y == c) for c in classes])
    return {"accuracy": float(np.mean(y == pred)), "balanced_accuracy": float(np.mean(recalls)),
            "macro_auc": _macro_auc(y, proba, classes) if proba is not None else np.nan,
            "majority_baseline": float(counts.max() / len(y))}


def _subsample_train(tr, max_train_rows, seed):
    """Cap a training-fold index array to at most ``max_train_rows``.

    The (test-fold) OOF evaluation set is untouched -- coverage over all
    rows is preserved -- only the more expensive *fit* cost is bounded, so
    this makes iterative-GD probe cost roughly independent of total bank
    size instead of growing linearly with it. ``max_train_rows=None`` keeps
    the historical unbounded behavior (used by probe_rma_analyze.py, whose
    published probe_metrics.json this must keep reproducing exactly).
    """
    if max_train_rows is None or len(tr) <= max_train_rows:
        return tr
    rng = np.random.default_rng(seed)
    return rng.choice(tr, size=max_train_rows, replace=False)


def compute_classifier_metrics(X, y, groups, n_splits: int = 5, max_train_rows: Optional[int] = None,
                               shuffle_seed=None, **probe_kwargs) -> Dict:
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    pred = np.empty(len(y), dtype=y.dtype); valid = np.zeros(len(y), bool)
    fold_metrics = []
    for fold, (tr, te) in enumerate(group_kfold_splits(groups, n_splits, shuffle_seed) or []):
        tr = _subsample_train(tr, max_train_rows, seed=fold)
        model = NumpyLogisticProbe(**probe_kwargs).fit(X[tr], y[tr])
        p = model.predict(X[te]); prob = model.predict_proba(X[te])
        pred[te], valid[te] = p, True
        fold_metrics.append({"fold": fold, **classification_metrics(y[te], p, prob, model.classes_)})
    summary = {}
    for key in ("accuracy", "balanced_accuracy", "macro_auc", "majority_baseline"):
        vals = np.asarray([f[key] for f in fold_metrics])
        summary[key] = {"mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                        "std": float(np.nanstd(vals)) if len(vals) else np.nan}
    return {"oof": classification_metrics(y[valid], pred[valid]), "folds": fold_metrics,
            "fold_summary": summary, "idx_oof": np.flatnonzero(valid), "y_pred_oof": pred[valid]}


def delta_balanced_accuracy(feat_a, feat_b, y, groups, n_splits: int = 5,
                            max_train_rows: Optional[int] = None, shuffle_seed=None,
                            **probe_kwargs) -> Dict:
    """``**probe_kwargs`` (e.g. ``max_iter=200``) is forwarded to
    ``NumpyLogisticProbe``; previously this always used the class default
    (max_iter=1000), silently ignoring any caller override and making this
    one of the dominant costs at large row counts. ``max_train_rows`` caps
    the per-fold fit size (see ``_subsample_train``)."""
    values = []
    a, b, y = np.asarray(feat_a), np.asarray(feat_b), np.asarray(y)
    for fold, (tr, te) in enumerate(group_kfold_splits(groups, n_splits, shuffle_seed) or []):
        tr = _subsample_train(tr, max_train_rows, seed=fold)
        ma = NumpyLogisticProbe(**probe_kwargs).fit(a[tr], y[tr])
        mab = NumpyLogisticProbe(**probe_kwargs).fit(np.concatenate([a[tr], b[tr]], 1), y[tr])
        ba = classification_metrics(y[te], ma.predict(a[te]))["balanced_accuracy"]
        bab = classification_metrics(y[te], mab.predict(np.concatenate([a[te], b[te]], 1)))["balanced_accuracy"]
        values.append(bab - ba)
    return {"mean": float(np.mean(values)) if values else np.nan,
            "std": float(np.std(values)) if values else np.nan,
            "fold_values": [float(v) for v in values], "n_folds": len(values)}
