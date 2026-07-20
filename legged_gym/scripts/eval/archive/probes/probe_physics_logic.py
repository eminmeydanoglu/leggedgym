"""Pure logic for the added-mass learn+use probe.

No Genesis / legged_gym env imports. Unit-testable with numpy/torch only.

Scientific claim under test:
  Does the latent encode added mass, and does the policy use that information
  under lateral command vy=+1.0?

Pieces:
  - opposite-end mass mapping for RMA teacher_wrong
  - live-mass invariant check
  - trajectory-level train/test split (no traj leakage; every mass in both)
  - small frozen MLP decoder + R² / shuffled control
  - paired Δuse aggregation (+ bootstrap CI)
  - result classification
  - eval cfg physics contract helpers (dict-like / SimpleNamespace mutation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


# ---------------------------------------------------------------------------
# Probe protocol constants (shared defaults)
# ---------------------------------------------------------------------------

MASS_GRID_KG: Tuple[float, ...] = (-2.0, 0.0, 3.0, 5.0)
LATERAL_CMD: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # vy = +1.0
MASS_NORM_RANGE: Tuple[float, float] = (-2.0, 5.0)
DEFAULT_WARMUP = 100
DEFAULT_MEASURE_STEPS = 400
DEFAULT_PER_POINT = 128
DECODER_HIDDEN = 32
LEARNED_R2_GATE = 0.5
SHUFFLE_R2_GATE = 0.15  # "≈ 0" practical ceiling
# ITT composite: after fall, remaining horizon uses this tracking penalty (m/s).
FALL_TRACKING_PENALTY = 2.0


# ---------------------------------------------------------------------------
# Mass mapping
# ---------------------------------------------------------------------------

# Opposite-end pairing for the default mass grid (plan §C):
#   real −2 → given +5
#   real  0 → given +3
#   real +3 → given  0
#   real +5 → given −2
_DEFAULT_OPPOSITE: Dict[float, float] = {
    -2.0: 5.0,
    0.0: 3.0,
    3.0: 0.0,
    5.0: -2.0,
}


def opposite_mass_map(
    mass: Union[float, np.ndarray, Sequence[float]],
    mass_grid: Sequence[float] = MASS_GRID_KG,
) -> Union[float, np.ndarray]:
    """Map each real mass to the opposite-end mass for teacher_wrong.

    For the canonical grid [-2, 0, 3, 5] uses the fixed pairing above.
    For arbitrary grids, pairs index i with index n-1-i.
    """
    grid = [float(m) for m in mass_grid]
    if list(grid) == list(MASS_GRID_KG) or set(grid) == set(MASS_GRID_KG):
        lookup = dict(_DEFAULT_OPPOSITE)
    else:
        # pair extremes: sorted unique then reverse
        sorted_g = sorted(set(grid))
        lookup = {sorted_g[i]: sorted_g[len(sorted_g) - 1 - i] for i in range(len(sorted_g))}

    def _one(m: float) -> float:
        key = float(m)
        if key in lookup:
            return float(lookup[key])
        # nearest-grid fallback
        nearest = min(lookup.keys(), key=lambda g: abs(g - key))
        return float(lookup[nearest])

    if isinstance(mass, (float, int, np.floating, np.integer)):
        return _one(float(mass))
    arr = np.asarray(mass, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    flat = arr.ravel()
    for i, v in enumerate(flat):
        out.ravel()[i] = _one(float(v))
    return out


def opposite_mass_index_pairs(mass_grid: Sequence[float] = MASS_GRID_KG) -> List[Tuple[int, int]]:
    """Return (src_idx, wrong_idx) pairs over mass_grid positions."""
    grid = list(mass_grid)
    pairs = []
    for i, m in enumerate(grid):
        wrong = float(opposite_mass_map(float(m), mass_grid))
        j = int(np.argmin([abs(float(g) - wrong) for g in grid]))
        pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Live mass invariant
# ---------------------------------------------------------------------------


def check_mass_invariant(
    recorded_mass: np.ndarray,
    target_mass: Union[float, np.ndarray],
    *,
    atol: float = 1e-3,
    env_ids: Optional[np.ndarray] = None,
) -> None:
    """Raise ValueError unless every recorded sample matches its target mass.

    recorded_mass: (N,) live simulator mass at each sample.
    target_mass: scalar or (N,) intended mass for that sample/env.
    """
    rec = np.asarray(recorded_mass, dtype=np.float64).ravel()
    if np.isscalar(target_mass) or (isinstance(target_mass, np.ndarray) and target_mass.ndim == 0):
        tgt = np.full_like(rec, float(target_mass))
    else:
        tgt = np.asarray(target_mass, dtype=np.float64).ravel()
    if rec.shape != tgt.shape:
        raise ValueError(
            f"mass invariant shape mismatch: recorded {rec.shape} vs target {tgt.shape}"
        )
    bad = np.abs(rec - tgt) > atol
    if not np.any(bad):
        return
    idx = np.where(bad)[0]
    n_show = min(5, len(idx))
    details = []
    for i in idx[:n_show]:
        eid = int(env_ids[i]) if env_ids is not None else i
        details.append(f"i={i} env={eid} recorded={rec[i]:.6g} target={tgt[i]:.6g}")
    raise ValueError(
        f"mass invariant violated on {int(bad.sum())}/{len(rec)} samples "
        f"(atol={atol}). Examples: " + "; ".join(details)
    )


def assert_mass_groups_constant(
    mass_by_group: Mapping[Any, Sequence[float]],
    *,
    atol: float = 1e-3,
) -> None:
    """Each group must be constant at its own target (mean of samples)."""
    for g, vals in mass_by_group.items():
        arr = np.asarray(list(vals), dtype=np.float64).ravel()
        if arr.size == 0:
            raise ValueError(f"mass group {g!r} has no samples")
        target = float(np.median(arr))
        check_mass_invariant(arr, target, atol=atol)


# ---------------------------------------------------------------------------
# Trajectory split
# ---------------------------------------------------------------------------


@dataclass
class TrajSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_trajs: np.ndarray
    test_trajs: np.ndarray


def trajectory_train_test_split(
    traj_ids: np.ndarray,
    mass: np.ndarray,
    *,
    test_frac: float = 0.25,
    seed: int = 0,
    min_per_mass_train: int = 1,
    min_per_mass_test: int = 1,
) -> TrajSplit:
    """Split samples by trajectory so no traj appears on both sides.

    Guarantees every unique mass level that has ≥2 trajectories contributes
    at least one traj to train and one to test when possible.
    """
    traj_ids = np.asarray(traj_ids).ravel()
    mass = np.asarray(mass, dtype=np.float64).ravel()
    if traj_ids.shape != mass.shape:
        raise ValueError("traj_ids and mass must have same length")
    if len(traj_ids) == 0:
        raise ValueError("empty dataset")

    rng = np.random.default_rng(seed)
    unique_masses = np.unique(mass)
    train_traj_set: set = set()
    test_traj_set: set = set()

    for m in unique_masses:
        trajs_m = np.unique(traj_ids[mass == m])
        rng.shuffle(trajs_m)
        n = len(trajs_m)
        if n < 2:
            # put the only traj in train; test may lack this mass
            train_traj_set.update(trajs_m.tolist())
            continue
        n_test = max(min_per_mass_test, int(round(n * test_frac)))
        n_test = min(n_test, n - min_per_mass_train)
        n_test = max(n_test, min_per_mass_test)
        n_test = min(n_test, n - 1)
        test_traj_set.update(trajs_m[:n_test].tolist())
        train_traj_set.update(trajs_m[n_test:].tolist())

    # resolve any accidental dual assignment (should not happen)
    both = train_traj_set & test_traj_set
    if both:
        for t in both:
            test_traj_set.discard(t)

    train_mask = np.isin(traj_ids, list(train_traj_set))
    test_mask = np.isin(traj_ids, list(test_traj_set))
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"trajectory split produced empty side "
            f"(train={len(train_idx)}, test={len(test_idx)}); need more trajs"
        )

    # leakage check
    tr = set(traj_ids[train_idx].tolist())
    te = set(traj_ids[test_idx].tolist())
    if tr & te:
        raise RuntimeError(f"trajectory leakage: {tr & te}")

    # every mass that has ≥2 trajs must appear in both
    for m in unique_masses:
        trajs_m = set(traj_ids[mass == m].tolist())
        if len(trajs_m) < 2:
            continue
        if not (trajs_m & tr) or not (trajs_m & te):
            raise RuntimeError(
                f"mass {m} with {len(trajs_m)} trajs not present in both splits"
            )

    return TrajSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        train_trajs=np.array(sorted(tr)),
        test_trajs=np.array(sorted(te)),
    )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class MassDecoderMLP(nn.Module if nn is not None else object):  # type: ignore[misc]
    """latent → hidden → 1. Policy weights stay frozen; this is a probe head."""

    def __init__(self, in_dim: int, hidden: int = DECODER_HIDDEN):
        if torch is None:
            raise RuntimeError("torch required for MassDecoderMLP")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class DecoderResult:
    r2: float
    mae: float
    y_true: np.ndarray
    y_pred: np.ndarray
    n_train: int
    n_test: int


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _traj_aware_train_val(
    X: "torch.Tensor",
    y: "torch.Tensor",
    traj_ids: Optional[np.ndarray],
    *,
    seed: int,
    val_frac: float = 0.1,
):
    """Split train tensors by trajectory for early-stopping val (no sample tail)."""
    n = X.shape[0]
    if traj_ids is None or len(traj_ids) != n:
        # fallback: random sample split (not last 10%)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_val = max(1, int(val_frac * n))
        if n_val >= n:
            return X, y, X, y
        val_i, tr_i = perm[:n_val], perm[n_val:]
        return X[tr_i], y[tr_i], X[val_i], y[val_i]

    traj_ids = np.asarray(traj_ids).ravel()
    uniq = np.unique(traj_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val_t = max(1, int(round(len(uniq) * val_frac))) if len(uniq) > 1 else 0
    if n_val_t < 1 or n_val_t >= len(uniq):
        return X, y, X, y
    val_set = set(uniq[:n_val_t].tolist())
    val_mask = np.array([t in val_set for t in traj_ids], dtype=bool)
    tr_mask = ~val_mask
    if not tr_mask.any() or not val_mask.any():
        return X, y, X, y
    return X[tr_mask], y[tr_mask], X[val_mask], y[val_mask]


def shuffle_labels_by_trajectory(
    labels: np.ndarray,
    traj_ids: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Permute labels at trajectory level (constant label per traj preserved).

    Each trajectory keeps one shared label; traj→label mapping is shuffled.
    Harder null than frame-level permutation when labels are constant in traj.
    """
    labels = np.asarray(labels)
    traj_ids = np.asarray(traj_ids).ravel()
    if labels.shape[0] != traj_ids.shape[0]:
        raise ValueError("labels/traj_ids length mismatch")
    uniq = np.unique(traj_ids)
    # representative label per traj (mode/mean of that traj)
    traj_label = {}
    for t in uniq:
        vals = labels[traj_ids == t]
        if vals.ndim == 1:
            traj_label[int(t)] = float(vals[0])  # mass constant within traj
        else:
            traj_label[int(t)] = vals[0].copy()
    rng = np.random.default_rng(seed)
    keys = list(traj_label.keys())
    vals = [traj_label[k] for k in keys]
    order = rng.permutation(len(keys))
    mapped = {keys[i]: vals[order[i]] for i in range(len(keys))}
    out = np.empty_like(labels)
    for i, t in enumerate(traj_ids):
        out[i] = mapped[int(t)]
    return out


def fit_mass_decoder(
    z: np.ndarray,
    mass: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    traj_ids: Optional[np.ndarray] = None,
    hidden: int = DECODER_HIDDEN,
    epochs: int = 400,
    lr: float = 1e-3,
    patience: int = 40,
    seed: int = 0,
    device: str = "cpu",
) -> DecoderResult:
    """Train a small MLP on train trajs; evaluate R² on held-out trajs."""
    if torch is None:
        raise RuntimeError("torch required for fit_mass_decoder")

    z = np.asarray(z, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64).ravel()
    if z.ndim != 2:
        raise ValueError(f"z must be (N, D), got {z.shape}")
    if z.shape[0] != mass.shape[0]:
        raise ValueError("z and mass length mismatch")

    X_tr, y_tr = z[train_idx], mass[train_idx]
    X_te, y_te = z[test_idx], mass[test_idx]
    traj_ids_train = None
    if traj_ids is not None:
        traj_ids_train = np.asarray(traj_ids).ravel()[train_idx]

    mu_x = X_tr.mean(axis=0)
    sig_x = X_tr.std(axis=0)
    sig_x = np.where(sig_x < 1e-12, 1.0, sig_x)
    mu_y = float(y_tr.mean())
    sig_y = float(y_tr.std())
    if sig_y < 1e-12:
        sig_y = 1.0

    torch.manual_seed(seed)
    dev = torch.device(device)
    model = MassDecoderMLP(z.shape[1], hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor((X_tr - mu_x) / sig_x, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor((y_tr - mu_y) / sig_y, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor((X_te - mu_x) / sig_x, dtype=torch.float32, device=dev)

    Xt, yt, Xv, yv = _traj_aware_train_val(
        Xtr_t, ytr_t, traj_ids_train, seed=seed + 3
    )

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    no_improve = 0
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val = float(loss_fn(model(Xv), yv).item())
        if val < best_val - 1e-6:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_n = model(Xte_t).cpu().numpy()
    y_pred = preds_n * sig_y + mu_y
    y_true = y_te.astype(np.float64)
    return DecoderResult(
        r2=r2_score(y_true, y_pred),
        mae=float(np.mean(np.abs(y_true - y_pred))),
        y_true=y_true,
        y_pred=y_pred.astype(np.float64),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
    )


def mass_decode_with_shuffle_control(
    z: np.ndarray,
    mass: np.ndarray,
    traj_ids: np.ndarray,
    *,
    seed: int = 0,
    hidden: int = DECODER_HIDDEN,
    **decoder_kwargs,
) -> Dict[str, Any]:
    """Trajectory split + real decode + trajectory-level shuffled-label control."""
    split = trajectory_train_test_split(traj_ids, mass, seed=seed)
    real = fit_mass_decoder(
        z, mass, split.train_idx, split.test_idx,
        traj_ids=traj_ids, seed=seed, hidden=hidden, **decoder_kwargs
    )
    mass_shuf = shuffle_labels_by_trajectory(mass, traj_ids, seed=seed + 17)
    shuf = fit_mass_decoder(
        z, mass_shuf, split.train_idx, split.test_idx,
        traj_ids=traj_ids, seed=seed + 1, hidden=hidden, **decoder_kwargs
    )
    return {
        "r2": real.r2,
        "mae": real.mae,
        "shuffled_r2": shuf.r2,
        "shuffled_mae": shuf.mae,
        "n_train": real.n_train,
        "n_test": real.n_test,
        "n_train_trajs": int(len(split.train_trajs)),
        "n_test_trajs": int(len(split.test_trajs)),
        "split": split,
        "real": real,
        "shuffled": shuf,
    }


# ---------------------------------------------------------------------------
# Use-test metrics
# ---------------------------------------------------------------------------


@dataclass
class UseTestResult:
    normal_err: float
    control_err: float
    wrong_err: float
    delta_use: float
    delta_use_ci_lo: float
    delta_use_ci_hi: float
    normal_fall: float
    control_fall: float
    wrong_fall: float
    delta_fall: float
    delta_fall_ci_lo: float
    delta_fall_ci_hi: float
    n_pairs: int
    metric_kind: str = "itt_composite"  # intention-to-treat tracking + fall penalty
    per_env_delta: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    per_env_delta_fall: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def apply_fall_penalty_itt(
    lin_err_sum: np.ndarray,
    step_count: np.ndarray,
    ever_fell: np.ndarray,
    n_steps: int,
    *,
    penalty: float = FALL_TRACKING_PENALTY,
) -> np.ndarray:
    """Intention-to-treat composite per env: full horizon mean with fall penalty.

    Fall step uses measured tracking error (included in lin_err_sum / step_count).
    Subsequent remaining horizon steps use `penalty`. Envs are never dropped
    for falling — avoids survivor selection bias.
    """
    lin_err_sum = np.asarray(lin_err_sum, dtype=np.float64).ravel()
    step_count = np.asarray(step_count, dtype=np.float64).ravel()
    ever_fell = np.asarray(ever_fell, dtype=bool).ravel()
    n = len(lin_err_sum)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        alive = int(step_count[i])
        alive = max(0, min(alive, n_steps))
        dead = n_steps - alive if ever_fell[i] else 0
        if not ever_fell[i]:
            if alive > 0:
                out[i] = lin_err_sum[i] / alive
            else:
                out[i] = float("nan")
        else:
            # measured steps (incl. fall step) + penalty for remaining horizon
            total = lin_err_sum[i] + dead * penalty
            denom = max(n_steps, alive + dead)
            out[i] = total / denom
    return out


def paired_delta_use(
    err_wrong: np.ndarray,
    err_control: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> Tuple[float, float, float, np.ndarray]:
    """Δuse = wrong − control; paired bootstrap CI on mean Δuse."""
    w = np.asarray(err_wrong, dtype=np.float64).ravel()
    c = np.asarray(err_control, dtype=np.float64).ravel()
    if w.shape != c.shape:
        raise ValueError("err_wrong and err_control must match shape")
    mask = np.isfinite(w) & np.isfinite(c)
    d = w[mask] - c[mask]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan"), d
    mean = float(d.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    n = len(d)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = d[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return mean, lo, hi, d


def aggregate_use_test(
    normal_err: np.ndarray,
    control_err: np.ndarray,
    wrong_err: np.ndarray,
    *,
    normal_fall: Optional[np.ndarray] = None,
    control_fall: Optional[np.ndarray] = None,
    wrong_fall: Optional[np.ndarray] = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
    metric_kind: str = "itt_composite",
) -> UseTestResult:
    # ITT: keep all envs with finite composite scores (no survivor filter)
    n = np.asarray(normal_err).ravel()
    c = np.asarray(control_err).ravel()
    w = np.asarray(wrong_err).ravel()
    mask = np.isfinite(n) & np.isfinite(c) & np.isfinite(w)
    mean_d, lo, hi, d = paired_delta_use(
        w[mask], c[mask], n_bootstrap=n_bootstrap, seed=seed
    )

    def _mean(x, fallback=float("nan")):
        if x is None:
            return fallback
        a = np.asarray(x, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        return float(a.mean()) if a.size else fallback

    nf = _mean(normal_fall, 0.0)
    cf = _mean(control_fall, 0.0)
    wf = _mean(wrong_fall, 0.0)

    # Paired binary fall difference per env ∈ {-1,0,1}, then bootstrap CI
    if wrong_fall is not None and control_fall is not None:
        wf_e = np.asarray(wrong_fall, dtype=np.float64).ravel()
        cf_e = np.asarray(control_fall, dtype=np.float64).ravel()
        # align with error mask length if same n
        if wf_e.shape == w.shape:
            fmask = mask
            d_fall_env = wf_e[fmask] - cf_e[fmask]
        else:
            d_fall_env = wf_e - cf_e
        mean_f, flo, fhi, d_f = paired_delta_use(
            d_fall_env + 0.0,  # already a difference; bootstrap mean of paired diffs
            np.zeros_like(d_fall_env),
            n_bootstrap=n_bootstrap,
            seed=seed + 11,
        )
        # paired_delta_use does wrong-control; we passed (diff, 0) so mean_f = mean(diff)
        delta_fall = float(d_fall_env.mean()) if d_fall_env.size else (wf - cf)
        delta_fall_ci_lo, delta_fall_ci_hi = flo, fhi
        per_env_df = d_fall_env
    else:
        delta_fall = wf - cf
        delta_fall_ci_lo = float("nan")
        delta_fall_ci_hi = float("nan")
        per_env_df = np.array([])

    return UseTestResult(
        normal_err=_mean(n),
        control_err=_mean(c),
        wrong_err=_mean(w),
        delta_use=mean_d,
        delta_use_ci_lo=lo,
        delta_use_ci_hi=hi,
        normal_fall=nf,
        control_fall=cf,
        wrong_fall=wf,
        delta_fall=delta_fall,
        delta_fall_ci_lo=delta_fall_ci_lo,
        delta_fall_ci_hi=delta_fall_ci_hi,
        n_pairs=int(d.size),
        metric_kind=metric_kind,
        per_env_delta=d,
        per_env_delta_fall=per_env_df,
    )


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------


def use_evidence_flags(
    delta_use: float,
    delta_use_ci_lo: float,
    *,
    delta_fall: float = 0.0,
    delta_fall_ci_lo: float = float("nan"),
    fall_margin: float = 0.05,
) -> Dict[str, bool]:
    """Strict CI gates for tracking and fall co-primaries (NaN CI = no pass)."""
    used_track = (
        np.isfinite(delta_use)
        and delta_use > 0.0
        and np.isfinite(delta_use_ci_lo)
        and delta_use_ci_lo > 0.0
    )
    used_fall = (
        np.isfinite(delta_fall)
        and delta_fall > fall_margin
        and np.isfinite(delta_fall_ci_lo)
        and delta_fall_ci_lo > 0.0
    )
    return {"track": bool(used_track), "fall": bool(used_fall)}


def used_via_label(flags: Mapping[str, bool]) -> str:
    t, f = flags.get("track", False), flags.get("fall", False)
    if t and f:
        return "both"
    if t:
        return "tracking"
    if f:
        return "fall"
    return "neither"


def classify_result(
    mass_r2: float,
    shuffled_r2: float,
    delta_use: float,
    delta_use_ci_lo: float,
    *,
    learned_gate: float = LEARNED_R2_GATE,
    shuffle_gate: float = SHUFFLE_R2_GATE,
    delta_fall: float = 0.0,
    delta_fall_ci_lo: float = float("nan"),
    fall_margin: float = 0.05,
) -> str:
    """Three-way outcome for student latent use.

    'used' if either channel has a positive effect AND a finite positive CI_lo:
      - tracking: Δuse > 0 and CI_lo > 0 (NaN CI does NOT pass)
      - fall: Δfall > fall_margin and CI_lo > 0
    Note: co-primary OR has multiplicity; report used_via for which channel fired.
    """
    learned = (
        np.isfinite(mass_r2)
        and mass_r2 > learned_gate
        and (not np.isfinite(shuffled_r2) or shuffled_r2 < shuffle_gate)
    )
    if not learned:
        return "öğrenildiği gösterilemedi"
    flags = use_evidence_flags(
        delta_use, delta_use_ci_lo,
        delta_fall=delta_fall, delta_fall_ci_lo=delta_fall_ci_lo,
        fall_margin=fall_margin,
    )
    if flags["track"] or flags["fall"]:
        return "öğrendi ve kullandı"
    return "öğrendi ama kullanmadı"


def build_table_row(
    method: str,
    seed: Any,
    mass_r2: float,
    shuffled_r2: float,
    normal_err: float,
    control_err: float,
    wrong_err: float,
    delta_use: float,
    *,
    delta_use_ci_lo: float = float("nan"),
    teacher_r2: Optional[float] = None,
    delta_fall: float = 0.0,
    delta_fall_ci_lo: float = float("nan"),
    control_fall: float = float("nan"),
    wrong_fall: float = float("nan"),
    use_test_kind: str = "student_latent_swap",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    flags = use_evidence_flags(
        delta_use, delta_use_ci_lo,
        delta_fall=delta_fall, delta_fall_ci_lo=delta_fall_ci_lo,
    )
    row = {
        "method": method,
        "seed": seed,
        "mass_r2": float(mass_r2),
        "shuffled_r2": float(shuffled_r2),
        "teacher_mass_r2": float(teacher_r2) if teacher_r2 is not None else None,
        "normal_err": float(normal_err),
        "control_err": float(control_err),
        "wrong_err": float(wrong_err),
        "delta_use": float(delta_use),
        "delta_use_ci_lo": float(delta_use_ci_lo),
        "control_fall": float(control_fall),
        "wrong_fall": float(wrong_fall),
        "delta_fall": float(delta_fall),
        "delta_fall_ci_lo": float(delta_fall_ci_lo),
        "use_test_kind": use_test_kind,
        "used_via": used_via_label(flags),
        "ci_kind": "within_run_paired_bootstrap",
        "result": classify_result(
            mass_r2, shuffled_r2, delta_use, delta_use_ci_lo,
            delta_fall=delta_fall, delta_fall_ci_lo=delta_fall_ci_lo,
        ),
    }
    if extra:
        row.update(extra)
    return row


def format_comparison_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Method", "Seed", "Mass R²", "Shuffled R²",
        "Normal err", "Control err", "Wrong err", "Δuse", "Δfall", "via", "Sonuç",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append(
            "| {method} | {seed} | {mass_r2:.3f} | {shuffled_r2:.3f} | "
            "{normal_err:.4f} | {control_err:.4f} | {wrong_err:.4f} | "
            "{delta_use:.4f} | {delta_fall:.3f} | {via} | {result} |".format(
                method=r["method"],
                seed=r["seed"],
                mass_r2=r["mass_r2"],
                shuffled_r2=r["shuffled_r2"],
                normal_err=r["normal_err"],
                control_err=r["control_err"],
                wrong_err=r["wrong_err"],
                delta_use=r["delta_use"],
                delta_fall=float(r.get("delta_fall", 0.0)),
                via=r.get("used_via", "neither"),
                result=r["result"],
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Eval physics contract (cfg mutation — no env needed)
# ---------------------------------------------------------------------------


def apply_probe_physics_contract(env_cfg: Any) -> Dict[str, Any]:
    """Disable mid-episode V3 switch, push, and non-mass DR axes.

    Mutates env_cfg in place. Returns a dict of what was set for logging/tests.
    """
    applied: Dict[str, Any] = {}
    dr = getattr(env_cfg, "domain_rand", None)
    if dr is None:
        raise ValueError("env_cfg has no domain_rand")

    if hasattr(dr, "resample_physics_within_episode"):
        dr.resample_physics_within_episode = False
        applied["resample_physics_within_episode"] = False
    if hasattr(dr, "push_robots"):
        dr.push_robots = False
        applied["push_robots"] = False
    if hasattr(dr, "randomize_friction"):
        dr.randomize_friction = False
        applied["randomize_friction"] = False
    if hasattr(dr, "randomize_base_mass"):
        dr.randomize_base_mass = False
        applied["randomize_base_mass"] = False
    if hasattr(dr, "randomize_com_displacement"):
        dr.randomize_com_displacement = False
        applied["randomize_com_displacement"] = False

    # freeze lateral command on cfg if present
    cmds = getattr(env_cfg, "commands", None)
    if cmds is not None:
        if hasattr(cmds, "curriculum"):
            cmds.curriculum = False
        if hasattr(cmds, "heading_command"):
            cmds.heading_command = False
        if hasattr(cmds, "zero_cmd_prob"):
            cmds.zero_cmd_prob = 0.0
        ranges = getattr(cmds, "ranges", None)
        if ranges is not None:
            if hasattr(ranges, "lin_vel_x"):
                ranges.lin_vel_x = [LATERAL_CMD[0], LATERAL_CMD[0]]
            if hasattr(ranges, "lin_vel_y"):
                ranges.lin_vel_y = [LATERAL_CMD[1], LATERAL_CMD[1]]
            if hasattr(ranges, "ang_vel_yaw"):
                ranges.ang_vel_yaw = [LATERAL_CMD[2], LATERAL_CMD[2]]
            applied["command"] = list(LATERAL_CMD)

    applied["command_default"] = list(LATERAL_CMD)
    applied["mass_grid"] = list(MASS_GRID_KG)
    return applied


def dr_normalize(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip-normalized label to [-1, 1] (matches eval harness)."""
    rng = hi - lo
    if abs(rng) < 1e-12:
        return np.zeros_like(np.asarray(x, dtype=np.float32))
    return np.clip(
        2.0 * (np.asarray(x, dtype=np.float32) - lo) / rng - 1.0, -1.0, 1.0
    )


# ---------------------------------------------------------------------------
# RMA privilege construction (pure tensor path)
# ---------------------------------------------------------------------------


def build_rma_wrong_privilege(
    priv_obs: "torch.Tensor",
    real_mass_raw: "torch.Tensor",
    *,
    mass_grid: Sequence[float] = MASS_GRID_KG,
    mass_range: Tuple[float, float] = MASS_NORM_RANGE,
    mass_slot: int = 1,
) -> "torch.Tensor":
    """Overwrite only the added-mass slot with opposite-end mass; keep velocity.

    priv_obs layout (RMA V3): [friction, added_mass, com_x, com_y, com_z, vx, vy, vz]
    """
    if torch is None:
        raise RuntimeError("torch required")
    wrong = priv_obs.clone()
    real_np = real_mass_raw.detach().cpu().numpy().astype(np.float64).ravel()
    wrong_raw = opposite_mass_map(real_np, mass_grid)
    if not isinstance(wrong_raw, np.ndarray):
        wrong_raw = np.asarray([wrong_raw], dtype=np.float64)
    wrong_norm = dr_normalize(wrong_raw, mass_range[0], mass_range[1])
    wrong[:, mass_slot] = torch.as_tensor(
        wrong_norm, dtype=wrong.dtype, device=wrong.device
    )
    return wrong


def build_rma_correct_privilege(priv_obs: "torch.Tensor") -> "torch.Tensor":
    """Teacher correct: use true privilege as-is (mass + velocity)."""
    return priv_obs


# ---------------------------------------------------------------------------
# Latent swap helpers (DreamWaQ / HIM) — pure tensor
# ---------------------------------------------------------------------------


def swap_implicit_latent(
    own_latent: "torch.Tensor",
    donor_latent: "torch.Tensor",
    donor_index: "torch.Tensor",
) -> "torch.Tensor":
    """Replace each env's latent with donor_index[env]'s latent.

    own_latent / donor_latent: (N, D). donor_index: (N,) long.
    vel is intentionally NOT passed — callers keep own vel separately.
    """
    if torch is None:
        raise RuntimeError("torch required")
    idx = donor_index.long()
    if idx.min() < 0 or idx.max() >= donor_latent.shape[0]:
        raise ValueError("donor_index out of range")
    return donor_latent[idx]


def update_frozen_latent_bank(
    frozen_bank: "torch.Tensor",
    live_bank: "torch.Tensor",
    ever_fell: np.ndarray,
) -> "torch.Tensor":
    """Refresh frozen bank only for envs that have not yet fallen.

    Fallen slots keep their last pre-fall latent (no post-reset overwrite).
    Returns a new tensor (does not mutate inputs in-place).
    """
    if torch is None:
        raise RuntimeError("torch required")
    out = frozen_bank.detach().clone()
    alive = torch.as_tensor(~np.asarray(ever_fell, dtype=bool), device=out.device)
    if alive.any():
        out[alive] = live_bank[alive]
    return out


def donor_map_stats(donor_idx: np.ndarray) -> Dict[str, float]:
    """Clustering diagnostics for a fixed donor map."""
    d = np.asarray(donor_idx, dtype=np.int64).ravel()
    uniq, counts = np.unique(d, return_counts=True)
    return {
        "unique_donor_count": float(len(uniq)),
        "max_receivers_per_donor": float(counts.max()) if len(counts) else 0.0,
        "mean_receivers_per_donor": float(counts.mean()) if len(counts) else 0.0,
    }


def make_within_and_cross_donors(
    mass_levels: np.ndarray,
    *,
    seed: int = 0,
    mass_grid: Sequence[float] = MASS_GRID_KG,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each env index, pick a within-mass donor and a cross-mass (opposite) donor.

    Returns (within_idx, cross_idx) as int64 arrays length N.
    Within-mass: derangement (cyclic) when group size > 1.
    Cross-mass: prefer one-to-one permutation against opposite mass when
    |group| == |opp| (default equal per_point); otherwise sample with
    replacement as fallback.
    """
    mass = np.asarray(mass_levels, dtype=np.float64).ravel()
    n = len(mass)
    rng = np.random.default_rng(seed)
    within = np.arange(n, dtype=np.int64)
    cross = np.arange(n, dtype=np.int64)
    valid = (
        np.ones(n, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool).ravel()
    )
    if valid.shape != mass.shape:
        raise ValueError("valid_mask shape must match mass_levels")

    # process each mass once; for equal-size opposite pairs use bipartite matching
    done_cross: set = set()
    for m in np.unique(mass):
        idxs = np.where(np.isclose(mass, m))[0]
        if len(idxs) == 0:
            continue
        pool_same = idxs[valid[idxs]] if np.any(valid[idxs]) else idxs
        # within: bijective derangement on valid pool (shuffle + cyclic roll)
        if len(pool_same) == 1:
            within[idxs] = pool_same[0]
        else:
            order = pool_same.copy()
            rng.shuffle(order)
            donors = np.roll(order, 1)
            within[order] = donors
            # invalid receivers in this mass group: assign from valid pool
            if valid_mask is not None:
                invalid_src = idxs[~valid[idxs]]
                if len(invalid_src) and len(pool_same):
                    within[invalid_src] = rng.choice(
                        pool_same, size=len(invalid_src), replace=True
                    )

        if float(m) in done_cross:
            continue
        wrong_m = float(opposite_mass_map(float(m), mass_grid))
        opp = np.where(np.isclose(mass, wrong_m))[0]
        if len(opp) == 0:
            opp = np.where(~np.isclose(mass, m))[0]
        if len(opp) == 0:
            opp = idxs
        opp_valid = opp[valid[opp]] if np.any(valid[opp]) else opp
        src_valid = idxs[valid[idxs]] if np.any(valid[idxs]) else idxs
        src = idxs
        # one-to-one bijection when equal valid cardinality (default equal per_point)
        if len(src_valid) == len(opp_valid) and len(src_valid) > 0:
            order_src = src_valid.copy()
            order_opp = opp_valid.copy()
            rng.shuffle(order_src)
            rng.shuffle(order_opp)
            for a, b in zip(order_src, order_opp):
                cross[a] = b
                if abs(wrong_m - float(m)) > 1e-9:
                    cross[b] = a
            # invalid receivers still need a donor: sample from valid opposite
            invalid_src = idxs[~valid[idxs]] if valid_mask is not None else np.array([], dtype=int)
            if len(invalid_src) and len(opp_valid):
                cross[invalid_src] = rng.choice(opp_valid, size=len(invalid_src), replace=True)
            done_cross.add(float(m))
            done_cross.add(wrong_m)
        else:
            chosen = rng.choice(opp_valid, size=len(src), replace=True)
            cross[src] = chosen

    return within, cross


def mask_valid_measurement(
    fall: np.ndarray,
    done: Optional[np.ndarray] = None,
    *,
    ever_invalid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Samples valid for decode/metrics: not a fall step; optional ever-invalid mask.

    For per-step datasets: mark step invalid if fall/done this step.
    ever_invalid: (N,) if provided, OR'ed in (post-reset contamination).
    Returns boolean mask True = keep.
    """
    fall = np.asarray(fall).astype(bool).ravel()
    keep = ~fall
    if done is not None:
        keep = keep & ~np.asarray(done).astype(bool).ravel()
    if ever_invalid is not None:
        keep = keep & ~np.asarray(ever_invalid).astype(bool).ravel()
    return keep


def accumulate_ever_invalid(
    n_envs: int,
    steps_fall: Sequence[np.ndarray],
) -> np.ndarray:
    """Given per-step fall arrays (T of (N,)), mark env contaminated after first fall.

    Returns ever_invalid stacked as (T, N) bool — True from first fall inclusive onward.
    """
    T = len(steps_fall)
    out = np.zeros((T, n_envs), dtype=bool)
    live = np.zeros(n_envs, dtype=bool)
    for t in range(T):
        f = np.asarray(steps_fall[t]).astype(bool).ravel()
        live = live | f
        out[t] = live
    return out
