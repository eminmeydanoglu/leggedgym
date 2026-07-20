"""Pure logic for the base-velocity estimation probe.

Separate from the added-mass learn+use probe. Scientific question:

  How well does each method represent base linear velocity?

Method contracts (intentionally different mechanisms):
  - RMA: frozen decoder  z_s → [vx,vy,vz]  and  z_t → [vx,vy,vz]
    (velocity is mixed into the privilege/history latent; no explicit head)
  - DreamWaQ: direct  vel_mu  vs true base_lin_vel  (explicit head)
  - HIM:      direct  vel_hat vs true base_lin_vel  (explicit head)

No Genesis imports. Unit-testable with numpy/torch only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore

# Shared protocol defaults (kept local so this module imports without Genesis)
DECODER_HIDDEN = 32
DEFAULT_MEASURE_STEPS = 400
DEFAULT_PER_POINT = 128
DEFAULT_WARMUP = 100
LATERAL_CMD: Tuple[float, float, float] = (0.0, 1.0, 0.0)
MASS_GRID_KG: Tuple[float, ...] = (-2.0, 0.0, 3.0, 5.0)

# Velocity probe gates
VEL_R2_GATE = 0.5
VEL_SHUFFLE_GATE = 0.15
VEL_DIMS = ("vx", "vy", "vz")
# Min target std (m/s) to treat a dim as identifiable under the protocol
VEL_MIN_TARGET_STD = 0.05

# Default multi-command schedule for richer velocity excitation (not single lateral).
# Each entry: [vx, vy, yaw]
DEFAULT_VEL_COMMAND_SCHEDULE = (
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (-0.5, 0.0, 0.0),
    (0.75, 0.5, 0.0),
    (0.0, 0.0, 0.0),
)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Trajectory split (no mass stratification — velocity varies within traj)
# ---------------------------------------------------------------------------


@dataclass
class TrajSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_trajs: np.ndarray
    test_trajs: np.ndarray


def trajectory_split(
    traj_ids: np.ndarray,
    *,
    test_frac: float = 0.25,
    seed: int = 0,
    strata: Optional[np.ndarray] = None,
) -> TrajSplit:
    """Split samples by trajectory so no traj appears on both sides.

    If `strata` is provided (same length as traj_ids), split is stratified so
    each stratum contributes trajs to train and test when it has ≥2 trajs
    (e.g. command_id for multi-command velocity probe).
    """
    traj_ids = np.asarray(traj_ids).ravel()
    if traj_ids.size == 0:
        raise ValueError("empty traj_ids")
    rng = np.random.default_rng(seed)

    if strata is None:
        uniq = np.unique(traj_ids)
        rng.shuffle(uniq)
        n_test = max(1, int(round(len(uniq) * test_frac)))
        n_test = min(n_test, len(uniq) - 1) if len(uniq) > 1 else 0
        if len(uniq) < 2 or n_test < 1:
            raise ValueError(
                f"need ≥2 trajectories for a train/test split, got {len(uniq)}"
            )
        test_set = set(uniq[:n_test].tolist())
        train_set = set(uniq[n_test:].tolist())
    else:
        strata = np.asarray(strata).ravel()
        if strata.shape != traj_ids.shape:
            raise ValueError("strata must match traj_ids length")
        # one stratum label per traj
        traj_to_stratum = {}
        for t, s in zip(traj_ids, strata):
            traj_to_stratum[int(t)] = s
        train_set: set = set()
        test_set: set = set()
        for s in np.unique(strata):
            trajs_s = np.array(
                [t for t, ss in traj_to_stratum.items() if ss == s], dtype=np.int64
            )
            rng.shuffle(trajs_s)
            if len(trajs_s) < 2:
                train_set.update(trajs_s.tolist())
                continue
            n_te = max(1, int(round(len(trajs_s) * test_frac)))
            n_te = min(n_te, len(trajs_s) - 1)
            test_set.update(trajs_s[:n_te].tolist())
            train_set.update(trajs_s[n_te:].tolist())
        both = train_set & test_set
        for t in both:
            test_set.discard(t)
        if not train_set or not test_set:
            raise ValueError("stratified trajectory split produced empty side")

    train_idx = np.where(np.isin(traj_ids, list(train_set)))[0]
    test_idx = np.where(np.isin(traj_ids, list(test_set)))[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("trajectory split produced empty side")
    tr = set(traj_ids[train_idx].tolist())
    te = set(traj_ids[test_idx].tolist())
    if tr & te:
        raise RuntimeError(f"trajectory leakage: {tr & te}")
    return TrajSplit(
        train_idx=train_idx,
        test_idx=test_idx,
        train_trajs=np.array(sorted(tr)),
        test_trajs=np.array(sorted(te)),
    )


def mean_r2_identifiable(
    r2_per_dim: Mapping[str, float],
    target_std: Mapping[str, Any],
) -> Tuple[float, List[str]]:
    """Mean R² over identifiable dims only; returns (mean, dim_names)."""
    dims = [
        k for k in VEL_DIMS
        if k in r2_per_dim and target_std.get(k, {}).get("identifiable", True)
    ]
    if not dims:
        return float("nan"), []
    return float(np.mean([r2_per_dim[k] for k in dims])), dims


# ---------------------------------------------------------------------------
# Multi-output decoder (RMA latent → velocity)
# ---------------------------------------------------------------------------


class VelDecoderMLP(nn.Module if nn is not None else object):  # type: ignore[misc]
    """latent → hidden → 3 (vx, vy, vz). Policy stays frozen."""

    def __init__(self, in_dim: int, out_dim: int = 3, hidden: int = DECODER_HIDDEN):
        if torch is None:
            raise RuntimeError("torch required for VelDecoderMLP")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class VelMetrics:
    r2_mean: float
    r2_per_dim: Dict[str, float]
    mae: float
    mae_per_dim: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    n_train: int
    n_test: int


def multi_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Per-component R² and mean over finite components."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
        yp = yp.reshape(-1, 1)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch {yt.shape} vs {yp.shape}")
    per: Dict[str, float] = {}
    vals = []
    for d in range(yt.shape[1]):
        name = VEL_DIMS[d] if d < len(VEL_DIMS) else f"d{d}"
        r = r2_score(yt[:, d], yp[:, d])
        per[name] = r
        vals.append(r)
    return float(np.mean(vals)), per


def multi_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict[str, float]]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
        yp = yp.reshape(-1, 1)
    per: Dict[str, float] = {}
    for d in range(yt.shape[1]):
        name = VEL_DIMS[d] if d < len(VEL_DIMS) else f"d{d}"
        per[name] = float(np.mean(np.abs(yt[:, d] - yp[:, d])))
    return float(np.mean(list(per.values()))), per


def _vel_train_val(X, y, traj_ids, *, seed: int, val_frac: float = 0.1):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if traj_ids is not None and len(traj_ids) == n:
        uniq = np.unique(traj_ids)
        rng.shuffle(uniq)
        n_val_t = max(1, int(round(len(uniq) * val_frac))) if len(uniq) > 1 else 0
        if 0 < n_val_t < len(uniq):
            val_set = set(uniq[:n_val_t].tolist())
            val_mask = np.array([t in val_set for t in traj_ids], dtype=bool)
            if val_mask.any() and (~val_mask).any():
                return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    if n_val >= n:
        return X, y, X, y
    return X[perm[n_val:]], y[perm[n_val:]], X[perm[:n_val]], y[perm[:n_val]]


def shuffle_vel_by_trajectory(
    vel: np.ndarray, traj_ids: np.ndarray, *, seed: int = 0
) -> np.ndarray:
    """Block-permute velocity labels by trajectory (whole traj vector reassigned)."""
    vel = np.asarray(vel, dtype=np.float64)
    traj_ids = np.asarray(traj_ids).ravel()
    uniq = np.unique(traj_ids)
    # one representative velocity sequence per traj — reassign whole traj blocks
    blocks = {int(t): vel[traj_ids == t] for t in uniq}
    keys = list(blocks.keys())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    donor = {keys[i]: keys[order[i]] for i in range(len(keys))}
    out = np.empty_like(vel)
    for t in keys:
        src = donor[t]
        src_block = blocks[src]
        n = (traj_ids == t).sum()
        # tile or truncate if lengths differ (same steps usually)
        if len(src_block) >= n:
            out[traj_ids == t] = src_block[:n]
        else:
            reps = int(np.ceil(n / len(src_block)))
            out[traj_ids == t] = np.tile(src_block, (reps, 1))[:n]
    return out


def fit_velocity_decoder(
    z: np.ndarray,
    vel: np.ndarray,
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
) -> VelMetrics:
    """Train latent → velocity MLP; evaluate multi-dim R² on held-out trajs."""
    if torch is None:
        raise RuntimeError("torch required")
    z = np.asarray(z, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    if z.ndim != 2 or vel.ndim != 2:
        raise ValueError(f"z and vel must be 2-D, got {z.shape}, {vel.shape}")
    if z.shape[0] != vel.shape[0]:
        raise ValueError("z/vel length mismatch")
    out_dim = vel.shape[1]

    X_tr, y_tr = z[train_idx], vel[train_idx]
    X_te, y_te = z[test_idx], vel[test_idx]

    mu_x = X_tr.mean(axis=0)
    sig_x = np.where(X_tr.std(axis=0) < 1e-12, 1.0, X_tr.std(axis=0))
    mu_y = y_tr.mean(axis=0)
    sig_y = np.where(y_tr.std(axis=0) < 1e-12, 1.0, y_tr.std(axis=0))

    torch.manual_seed(seed)
    dev = torch.device(device)
    model = VelDecoderMLP(z.shape[1], out_dim=out_dim, hidden=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor((X_tr - mu_x) / sig_x, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor((y_tr - mu_y) / sig_y, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor((X_te - mu_x) / sig_x, dtype=torch.float32, device=dev)

    traj_tr = None
    if traj_ids is not None:
        traj_tr = np.asarray(traj_ids).ravel()[train_idx]
    Xt, yt, Xv, yv = _vel_train_val(Xtr_t, ytr_t, traj_tr, seed=seed + 3)

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
    r2_m, r2_d = multi_r2(y_true, y_pred)
    mae_m, mae_d = multi_mae(y_true, y_pred)
    return VelMetrics(
        r2_mean=r2_m,
        r2_per_dim=r2_d,
        mae=mae_m,
        mae_per_dim=mae_d,
        y_true=y_true,
        y_pred=y_pred.astype(np.float64),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
    )


def target_std_report(vel: np.ndarray) -> Dict[str, Any]:
    """Per-dim target std/range and identifiability flags."""
    vel = np.asarray(vel, dtype=np.float64)
    if vel.ndim == 1:
        vel = vel.reshape(-1, 1)
    out: Dict[str, Any] = {}
    identifiable = []
    for d in range(vel.shape[1]):
        name = VEL_DIMS[d] if d < len(VEL_DIMS) else f"d{d}"
        std = float(vel[:, d].std())
        lo, hi = float(vel[:, d].min()), float(vel[:, d].max())
        ok = std >= VEL_MIN_TARGET_STD
        out[name] = {
            "std": std,
            "range": [lo, hi],
            "identifiable": ok,
        }
        identifiable.append(ok)
    out["n_identifiable"] = int(sum(identifiable))
    return out


def decode_velocity_with_shuffle(
    z: np.ndarray,
    vel: np.ndarray,
    traj_ids: np.ndarray,
    *,
    seed: int = 0,
    hidden: int = DECODER_HIDDEN,
    strata: Optional[np.ndarray] = None,
    **kw,
) -> Dict[str, Any]:
    """Traj split + real decode + trajectory-block shuffled control for RMA."""
    split = trajectory_split(traj_ids, seed=seed, strata=strata)
    real = fit_velocity_decoder(
        z, vel, split.train_idx, split.test_idx,
        traj_ids=traj_ids, seed=seed, hidden=hidden, **kw
    )
    vel_shuf = shuffle_vel_by_trajectory(vel, traj_ids, seed=seed + 19)
    shuf = fit_velocity_decoder(
        z, vel_shuf, split.train_idx, split.test_idx,
        traj_ids=traj_ids, seed=seed + 1, hidden=hidden, **kw
    )
    tgt = target_std_report(vel[split.test_idx])
    r2_report, id_dims = mean_r2_identifiable(real.r2_per_dim, tgt)
    shuf_report, _ = mean_r2_identifiable(shuf.r2_per_dim, tgt)
    if not id_dims:
        r2_report = real.r2_mean
        shuf_report = shuf.r2_mean
    return {
        "r2_mean": r2_report,
        "r2_mean_all_dims": real.r2_mean,
        "r2_per_dim": real.r2_per_dim,
        "shuffled_r2_per_dim": shuf.r2_per_dim,
        "mae": real.mae,
        "mae_per_dim": real.mae_per_dim,
        "shuffled_r2": shuf_report,
        "shuffled_r2_all_dims": shuf.r2_mean,
        "identifiable_dims": id_dims,
        "target_std": tgt,
        "n_train": real.n_train,
        "n_test": real.n_test,
        "split": split,
        "real": real,
        "shuffled": shuf,
        "kind": "decoder",
    }


# ---------------------------------------------------------------------------
# Explicit head metrics (DreamWaQ vel_mu / HIM vel_hat)
# ---------------------------------------------------------------------------


def explicit_velocity_metrics(
    vel_hat: np.ndarray,
    vel_true: np.ndarray,
    traj_ids: np.ndarray,
    *,
    seed: int = 0,
    strata: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Direct comparison on held-out trajectories (no trained decoder).

    Shuffle control: trajectory-block permute true velocity so correspondence
    is broken while keeping within-traj structure. Real and shuffle R² use the
    same identifiable-dimension set.
    """
    split = trajectory_split(traj_ids, seed=seed, strata=strata)
    pred = np.asarray(vel_hat, dtype=np.float64)[split.test_idx]
    true = np.asarray(vel_true, dtype=np.float64)[split.test_idx]
    r2_m, r2_d = multi_r2(true, pred)
    mae_m, mae_d = multi_mae(true, pred)

    true_full = np.asarray(vel_true, dtype=np.float64)
    true_shuf_full = shuffle_vel_by_trajectory(true_full, traj_ids, seed=seed + 23)
    true_shuf = true_shuf_full[split.test_idx]
    shuf_r2_all, shuf_r2_d = multi_r2(true_shuf, pred)
    tgt = target_std_report(true)
    r2_report, id_dims = mean_r2_identifiable(r2_d, tgt)
    shuf_report, _ = mean_r2_identifiable(shuf_r2_d, tgt)
    if not id_dims:
        r2_report, shuf_report = r2_m, shuf_r2_all

    return {
        "r2_mean": r2_report,
        "r2_mean_all_dims": r2_m,
        "r2_per_dim": r2_d,
        "shuffled_r2_per_dim": shuf_r2_d,
        "mae": mae_m,
        "mae_per_dim": mae_d,
        "shuffled_r2": shuf_report,
        "shuffled_r2_all_dims": shuf_r2_all,
        "identifiable_dims": id_dims,
        "target_std": tgt,
        "n_train": int(len(split.train_idx)),
        "n_test": int(len(split.test_idx)),
        "split": split,
        "kind": "explicit",
    }


# ---------------------------------------------------------------------------
# Result classification + table
# ---------------------------------------------------------------------------


def classify_velocity_result(
    r2_mean: float,
    shuffled_r2: float,
    *,
    r2_gate: float = VEL_R2_GATE,
    shuffle_gate: float = VEL_SHUFFLE_GATE,
    n_identifiable: int = 3,
) -> str:
    if n_identifiable <= 0:
        return "protocol yetersiz (hedef varyans düşük)"
    ok = (
        np.isfinite(r2_mean)
        and r2_mean > r2_gate
        and (not np.isfinite(shuffled_r2) or shuffled_r2 < shuffle_gate)
    )
    if not ok:
        return "velocity gösterilemedi"
    if n_identifiable >= 3:
        return "velocity tahmin ediyor"
    return "identifiable boyutlarda tahmin ediyor"


def build_velocity_row(
    method: str,
    seed: Any,
    source: str,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    r2 = float(metrics["r2_mean"])
    shuf = float(metrics["shuffled_r2"])
    r2_d = metrics.get("r2_per_dim") or {}
    tgt = metrics.get("target_std") or {}
    n_id = int(tgt.get("n_identifiable", 3)) if isinstance(tgt, dict) else 3
    std_vx = float(tgt.get("vx", {}).get("std", float("nan"))) if isinstance(tgt, dict) else float("nan")
    std_vy = float(tgt.get("vy", {}).get("std", float("nan"))) if isinstance(tgt, dict) else float("nan")
    std_vz = float(tgt.get("vz", {}).get("std", float("nan"))) if isinstance(tgt, dict) else float("nan")
    return {
        "method": method,
        "seed": seed,
        "source": source,
        "r2_mean": r2,
        "shuffled_r2": shuf,
        "mae": float(metrics["mae"]),
        "r2_vx": float(r2_d.get("vx", float("nan"))),
        "r2_vy": float(r2_d.get("vy", float("nan"))),
        "r2_vz": float(r2_d.get("vz", float("nan"))),
        "std_vx": std_vx,
        "std_vy": std_vy,
        "std_vz": std_vz,
        "n_identifiable": n_id,
        "kind": metrics.get("kind", ""),
        "result": classify_velocity_result(r2, shuf, n_identifiable=n_id),
    }


def format_velocity_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Method", "Seed", "Source", "Vel R²", "Shuffled R²", "MAE",
        "vx R²", "vy R²", "vz R²", "std vx/vy/vz", "Sonuç",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        stds = "{:.2f}/{:.2f}/{:.2f}".format(
            float(r.get("std_vx", float("nan"))),
            float(r.get("std_vy", float("nan"))),
            float(r.get("std_vz", float("nan"))),
        )
        lines.append(
            "| {method} | {seed} | {source} | {r2_mean:.3f} | {shuffled_r2:.3f} | "
            "{mae:.4f} | {r2_vx:.3f} | {r2_vy:.3f} | {r2_vz:.3f} | {stds} | {result} |".format(
                method=r["method"],
                seed=r["seed"],
                source=r["source"],
                r2_mean=r["r2_mean"],
                shuffled_r2=r["shuffled_r2"],
                mae=r["mae"],
                r2_vx=r["r2_vx"],
                r2_vy=r["r2_vy"],
                r2_vz=r["r2_vz"],
                stds=stds,
                result=r["result"],
            )
        )
    return "\n".join(lines)


def analyze_velocity_samples(
    samples: Dict[str, np.ndarray],
    *,
    method: str,
    seed_label: str = "1",
    seed: int = 0,
) -> Dict[str, Any]:
    """Offline analysis from a velocity samples.npz dict.

    Expected keys:
      vel_true (N,3), traj_id (N,)
      optional command (N,3) for command-stratified split
      RMA: z_s (N,D), optional z_t (N,D)
      DreamWaQ/HIM: vel_hat (N,3)  — explicit estimate
    """
    method_l = method.lower()
    vel_true = np.asarray(samples["vel_true"], dtype=np.float64)
    traj = np.asarray(samples["traj_id"]).ravel()
    # Prefer explicit command_id; fall back to stable command tuple encoding
    strata = None
    if "command_id" in samples:
        strata = np.asarray(samples["command_id"]).ravel()
    elif "command" in samples:
        cmd = np.asarray(samples["command"], dtype=np.float64)
        # stable encoding: include yaw; round to avoid float noise
        strata = (
            np.round(cmd[:, 0] * 1000)
            + np.round(cmd[:, 1] * 100) * 10000
            + np.round(cmd[:, 2] * 10) * 1_000_000
        ).astype(np.int64)

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    is_rma = method_l in ("rma", "go2_v3_rma", "v3_rma")
    is_dw = method_l in ("dreamwaq", "go2_v3_dreamwaq", "dw", "v3_dreamwaq")
    is_him = method_l in ("him", "go2_v3_him_fixed", "him_fixed", "v3_him")

    display = "RMA" if is_rma else ("DreamWaQ" if is_dw else ("HIM" if is_him else method))

    if is_rma:
        if "z_s" not in samples:
            raise KeyError("RMA velocity samples need z_s")
        m_s = decode_velocity_with_shuffle(
            samples["z_s"], vel_true, traj, seed=seed, strata=strata
        )
        rows.append(build_velocity_row(display, seed_label, "z_s→v", m_s))
        details["z_s"] = {k: m_s[k] for k in m_s if k not in ("split", "real", "shuffled")}
        if "z_t" in samples:
            m_t = decode_velocity_with_shuffle(
                samples["z_t"], vel_true, traj, seed=seed + 7, strata=strata
            )
            rows.append(build_velocity_row(display, seed_label, "z_t→v", m_t))
            details["z_t"] = {k: m_t[k] for k in m_t if k not in ("split", "real", "shuffled")}
    else:
        # DreamWaQ / HIM explicit head
        if "vel_hat" not in samples:
            raise KeyError(f"{display} velocity samples need vel_hat")
        src = "vel_mu" if is_dw else "vel_hat"
        m = explicit_velocity_metrics(
            samples["vel_hat"], vel_true, traj, seed=seed, strata=strata
        )
        rows.append(build_velocity_row(display, seed_label, src, m))
        details[src] = {k: m[k] for k in m if k != "split"}

    return {
        "rows": rows,
        "details": details,
        "table_md": format_velocity_table(rows),
    }
