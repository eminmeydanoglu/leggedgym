"""Per-cell / per-replica validation bank loader (run #1 only on disk).

Each ``model_{iter}.json`` holds 1008 measurements = 84 cells × 12 replicas with
matched ``(cell_id, replica_id) → (command_vx, command_vy, command_yaw)`` across
checkpoints and arms (shared ``validation_bank_fingerprint``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import atlas as atlas_mod

REPO_ROOT = atlas_mod.REPO_ROOT
CHECKPOINTS = (1000, 1400, 1800, 2200, 2600, 3000)


@dataclass(frozen=True)
class ValidationBank:
    """Stacked validation measurements for one arm."""

    arm: str
    path: Path
    iterations: tuple[int, ...]
    # shape (n_iter, n_meas)
    cell_id: np.ndarray
    replica_id: np.ndarray
    spnte_lin: np.ndarray
    spnte_yaw: np.ndarray
    fall_rate: np.ndarray
    survival_steps: np.ndarray
    command_vx: np.ndarray
    command_vy: np.ndarray
    command_yaw: np.ndarray
    fingerprint: str
    # shape (n_iter, 84)
    cell_spnte_lin: np.ndarray
    cell_fall_rate: np.ndarray
    macro_spnte_lin: np.ndarray

    @property
    def n_cells(self) -> int:
        return self.cell_spnte_lin.shape[1]

    @property
    def n_replicas(self) -> int:
        return int(self.replica_id[0].max()) + 1 if self.replica_id.size else 0


def _load_one(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_bank(directory: str | Path, arm: str = "") -> ValidationBank | None:
    """Load all model_*.json under a ued_validation directory, or None if empty."""
    directory = Path(directory)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    if not directory.is_dir():
        return None
    files = sorted(directory.glob("model_*.json"))
    # skip shard files
    files = [f for f in files if "shard" not in f.name]
    if not files:
        return None

    iters: list[int] = []
    meas_blocks: list[dict[str, np.ndarray]] = []
    cell_spnte: list[np.ndarray] = []
    cell_fall: list[np.ndarray] = []
    macro: list[float] = []
    fingerprint = ""

    for f in files:
        d = _load_one(f)
        it = int(d.get("checkpoint_iteration", 0))
        if it == 0:
            # parse from name
            try:
                it = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
        iters.append(it)
        fingerprint = d.get("validation_bank_fingerprint", fingerprint) or fingerprint
        m = d["measurements"]
        block = {
            "cell_id": np.array([x["cell_id"] for x in m], dtype=int),
            "replica_id": np.array([x["replica_id"] for x in m], dtype=int),
            "spnte_lin": np.array([x["spnte_lin"] for x in m], dtype=float),
            "spnte_yaw": np.array([x.get("spnte_yaw", np.nan) for x in m], dtype=float),
            "fall_rate": np.array([x.get("fall_rate", np.nan) for x in m], dtype=float),
            "survival_steps": np.array(
                [x.get("survival_steps", np.nan) for x in m], dtype=float
            ),
            "command_vx": np.array([x["command_vx"] for x in m], dtype=float),
            "command_vy": np.array([x["command_vy"] for x in m], dtype=float),
            "command_yaw": np.array([x["command_yaw"] for x in m], dtype=float),
        }
        meas_blocks.append(block)
        cells = d.get("scores", {}).get("cells") or []
        if cells:
            # sort by cell_id
            cells_sorted = sorted(cells, key=lambda c: c["cell_id"])
            cell_spnte.append(
                np.array([c["spnte_lin"] for c in cells_sorted], dtype=float)
            )
            cell_fall.append(
                np.array([c.get("fall_rate", np.nan) for c in cells_sorted], dtype=float)
            )
        else:
            # aggregate from measurements
            n_cells = int(block["cell_id"].max()) + 1
            sp = np.full(n_cells, np.nan)
            fr = np.full(n_cells, np.nan)
            for c in range(n_cells):
                msk = block["cell_id"] == c
                sp[c] = np.nanmean(block["spnte_lin"][msk])
                fr[c] = np.nanmean(block["fall_rate"][msk])
            cell_spnte.append(sp)
            cell_fall.append(fr)
        macro.append(float(d.get("scores", {}).get("macro_mean_spnte_lin", np.nanmean(cell_spnte[-1]))))

    # sort by iteration
    order = np.argsort(iters)
    iters_t = tuple(int(iters[i]) for i in order)
    stack = lambda key: np.stack([meas_blocks[i][key] for i in order], axis=0)

    return ValidationBank(
        arm=arm,
        path=directory,
        iterations=iters_t,
        cell_id=stack("cell_id"),
        replica_id=stack("replica_id"),
        spnte_lin=stack("spnte_lin"),
        spnte_yaw=stack("spnte_yaw"),
        fall_rate=stack("fall_rate"),
        survival_steps=stack("survival_steps"),
        command_vx=stack("command_vx"),
        command_vy=stack("command_vy"),
        command_yaw=stack("command_yaw"),
        fingerprint=fingerprint,
        cell_spnte_lin=np.stack([cell_spnte[i] for i in order], axis=0),
        cell_fall_rate=np.stack([cell_fall[i] for i in order], axis=0),
        macro_spnte_lin=np.array([macro[i] for i in order], dtype=float),
    )


def load_run1_banks() -> dict[str, ValidationBank | None]:
    """Load run #1 LP and UNI validation banks (only ones present on disk)."""
    out: dict[str, ValidationBank | None] = {}
    for key in ("run1_lp", "run1_uni"):
        meta = atlas_mod.RUN_REGISTRY[key]
        vb = meta.get("validation_bank")
        if vb is None:
            out[key] = None
        else:
            out[key] = load_bank(vb, arm=key)
    return out


def verify_matched_design(
    bank_a: ValidationBank, bank_b: ValidationBank
) -> dict[str, Any]:
    """Assert (cell, replica) → command triples match across banks/checkpoints."""
    # compare first checkpoint of each
    same_fp = bank_a.fingerprint == bank_b.fingerprint and bool(bank_a.fingerprint)
    cmd_keys = ("command_vx", "command_vy", "command_yaw", "cell_id", "replica_id")
    per_ckpt: list[dict[str, Any]] = []
    all_match = True
    n = min(len(bank_a.iterations), len(bank_b.iterations))
    for i in range(n):
        match = True
        for k in cmd_keys:
            a = getattr(bank_a, k)[i]
            b = getattr(bank_b, k)[i]
            if a.shape != b.shape or not np.allclose(a, b, equal_nan=True):
                match = False
                all_match = False
        per_ckpt.append(
            {
                "iter_a": bank_a.iterations[i],
                "iter_b": bank_b.iterations[i],
                "commands_match": match,
            }
        )
    # within-bank stability across checkpoints
    within_a = True
    for i in range(1, len(bank_a.iterations)):
        for k in ("command_vx", "command_vy", "command_yaw", "cell_id", "replica_id"):
            if not np.allclose(getattr(bank_a, k)[0], getattr(bank_a, k)[i], equal_nan=True):
                within_a = False
    return {
        "fingerprints_equal": same_fp,
        "fingerprint": bank_a.fingerprint,
        "cross_arm_all_match": all_match,
        "within_arm_commands_stable": within_a,
        "per_checkpoint": per_ckpt,
    }


def cell_improvement(
    bank: ValidationBank, i_from: int, i_to: int, metric: str = "spnte_lin"
) -> np.ndarray:
    """Per-cell improvement. For spnte_lin lower is better ⇒ improvement = from - to."""
    if metric == "spnte_lin":
        return bank.cell_spnte_lin[i_from] - bank.cell_spnte_lin[i_to]
    if metric == "fall_rate":
        return bank.cell_fall_rate[i_from] - bank.cell_fall_rate[i_to]
    raise ValueError(metric)


def within_cell_command_r2(
    bank: ValidationBank, ckpt_index: int = -1, nonlinear: bool = False
) -> dict[str, Any]:
    """Fraction of within-cell spnte_lin variance explained by commands.

    Model: y = cell_FE + β·cmd (+ |vy|, |yaw| if nonlinear).
    Returns partial R² shares and residual SEM reduction estimate.
    """
    i = ckpt_index if ckpt_index >= 0 else len(bank.iterations) + ckpt_index
    y = bank.spnte_lin[i]
    cid = bank.cell_id[i]
    vx = bank.command_vx[i]
    vy = bank.command_vy[i]
    yaw = bank.command_yaw[i]
    n_cells = int(cid.max()) + 1

    # demean within cell
    y_dm = np.empty_like(y)
    for c in range(n_cells):
        m = cid == c
        y_dm[m] = y[m] - np.mean(y[m])
    ss_tot = float(np.sum(y_dm ** 2))
    if ss_tot < 1e-15:
        return {"r2_total": 0.0, "ss_tot": ss_tot, "n": int(y.size)}

    def _ols(X: np.ndarray, yv: np.ndarray) -> tuple[float, np.ndarray]:
        # demean X within cell as well for within estimator
        Xd = X.copy()
        for c in range(n_cells):
            m = cid == c
            Xd[m] = X[m] - X[m].mean(axis=0)
        # add intercept of 0 after demeaning
        beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
        resid = yv - Xd @ beta
        ss_res = float(np.sum(resid ** 2))
        r2 = 1.0 - ss_res / ss_tot
        return r2, beta

    X_all = np.column_stack([vx, vy, yaw])
    if nonlinear:
        X_all = np.column_stack([vx, vy, yaw, np.abs(vy), np.abs(yaw), vx ** 2])
    r2_all, _ = _ols(X_all, y_dm)

    shares = {}
    for name, col in (("vx", vx), ("vy", vy), ("yaw", yaw)):
        r2_k, _ = _ols(col.reshape(-1, 1), y_dm)
        shares[name] = float(r2_k)

    # residual within-cell var vs raw
    raw_vars = []
    resid_vars = []
    Xd = X_all.copy()
    for c in range(n_cells):
        m = cid == c
        Xd[m] = X_all[m] - X_all[m].mean(axis=0)
    beta, *_ = np.linalg.lstsq(Xd, y_dm, rcond=None)
    resid = y_dm - Xd @ beta
    for c in range(n_cells):
        m = cid == c
        if m.sum() < 2:
            continue
        raw_vars.append(float(np.var(y[m], ddof=1)))
        resid_vars.append(float(np.var(resid[m] + np.mean(y[m]), ddof=1)))
    mean_raw = float(np.mean(raw_vars)) if raw_vars else np.nan
    mean_resid = float(np.mean(resid_vars)) if resid_vars else np.nan
    sem_ratio = (
        float(np.sqrt(mean_resid / mean_raw))
        if mean_raw and mean_raw > 0
        else np.nan
    )

    return {
        "checkpoint": int(bank.iterations[i]),
        "nonlinear": nonlinear,
        "r2_total": float(r2_all),
        "partial_r2": shares,
        "mean_within_cell_var_raw": mean_raw,
        "mean_within_cell_var_resid": mean_resid,
        "sem_reduction_factor": sem_ratio,
        "n_measurements": int(y.size),
        "ss_tot": ss_tot,
    }
