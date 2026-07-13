"""Shared artifact provenance for sweep / indist / transient eval outputs."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import numpy as np

from legged_gym.scripts.eval.ckpt_utils import (
    CkptSpec,
    ckpt_kind,
    extract_ckpt_meta,
    sha256_file,
)


def git_commit(cwd: Optional[str] = None) -> str:
    try:
        from legged_gym import LEGGED_GYM_ROOT_DIR
        root = cwd or LEGGED_GYM_ROOT_DIR
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _genesis_version() -> str:
    try:
        import genesis as gs
        return str(getattr(gs, "__version__", "unknown"))
    except Exception:
        return "unknown"


def load_run_manifest(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def collect_eval_meta(
    *,
    task: str,
    method: Optional[str] = None,
    chosen_run: str,
    log_root: str,
    ckpt_spec: CkptSpec,
    ckpt_path: str,
    seed: int,
    warmup: int,
    steps: Optional[int] = None,
    per_point: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the provenance dict written into every eval ``.npz``."""
    try:
        from legged_gym import SIMULATOR
        simulator = str(SIMULATOR)
    except Exception:
        simulator = os.environ.get("SIMULATOR", "unknown")

    kind = ckpt_kind(ckpt_spec)
    digest = sha256_file(ckpt_path) if os.path.isfile(ckpt_path) else "missing"
    ck_meta = {}
    if os.path.isfile(ckpt_path):
        try:
            ck_meta = extract_ckpt_meta(ckpt_path)
        except Exception as e:
            ck_meta = {"ckpt_meta_error": str(e)}

    run_dir = os.path.join(log_root, chosen_run) if not os.path.isabs(chosen_run) else chosen_run
    # If chosen_run is already absolute (some call sites), keep it.
    if not os.path.isdir(run_dir) and os.path.isdir(os.path.dirname(ckpt_path)):
        run_dir = os.path.dirname(ckpt_path)
    man = load_run_manifest(run_dir)

    import torch

    meta: Dict[str, Any] = dict(
        task=task,
        method=method or task,
        training_seed=ck_meta.get("training_seed", man.get("training_seed")),
        eval_seed=seed,
        load_run=chosen_run,
        run_folder=os.path.basename(chosen_run.rstrip("/")),
        ckpt_kind=kind,
        ckpt=str(ckpt_spec),
        ckpt_path=ckpt_path,
        ckpt_sha256=digest,
        ckpt_iter=ck_meta.get("ckpt_iter"),
        best_eval_score=ck_meta.get("eval_score", ck_meta.get("best_eval_score")),
        best_eval_it=ck_meta.get("eval_it"),
        schedule_stage_start=ck_meta.get("schedule_stage_start"),
        schedule_lin_vel_x=ck_meta.get("schedule_lin_vel_x"),
        training_commit=man.get("git_commit", "unknown"),
        eval_commit=git_commit(),
        run_manifest_task=man.get("task"),
        run_manifest_seed=man.get("training_seed"),
        simulator=simulator,
        genesis_version=_genesis_version(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        gpu=_gpu_name(),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        warmup=warmup,
    )
    if steps is not None:
        meta["steps"] = steps
    if per_point is not None:
        meta["per_point"] = per_point
    if extra:
        meta.update(extra)
    return meta


def merge_npz_payload(*parts) -> Dict[str, Any]:
    """Merge dict parts into one npz payload; raise if any key appears twice.

    Prevents ``atomic_savez(path, axis=..., **meta)`` TypeError when ``meta``
    already carries the same keys (the sweep.py P0 bug class).
    """
    out: Dict[str, Any] = {}
    for i, part in enumerate(parts):
        if part is None:
            continue
        if not isinstance(part, dict):
            raise TypeError(f"merge_npz_payload part {i} must be dict, got {type(part)}")
        for k, v in part.items():
            if k in out:
                raise ValueError(
                    f"duplicate npz payload key {k!r} (already set; refusing silent override)"
                )
            out[k] = v
    return out


def verify_run_identity(
    run_dir: str,
    *,
    expected_task: str,
    expected_run_name: Optional[str] = None,
    expected_training_seed: Optional[int] = None,
    require_manifest: bool = False,
) -> Dict[str, Any]:
    """Fail-loud if run folder / run_manifest.json does not match the eval task.

    Checks (when data is available):
      * folder name contains expected_run_name (e.g. ``bench_mlp``)
      * manifest ``task`` == expected_task
      * manifest ``training_seed`` == expected_training_seed (if both set)
      * folder ``_seedN`` suffix matches manifest / expected seed when present
    """
    import re

    base = os.path.basename(run_dir.rstrip(os.sep))
    if expected_run_name and expected_run_name not in base:
        raise ValueError(
            f"run folder {base!r} does not contain expected run_name "
            f"{expected_run_name!r} for task {expected_task!r}"
        )

    man = load_run_manifest(run_dir)
    if not man:
        if require_manifest:
            raise FileNotFoundError(
                f"run_manifest.json missing under {run_dir} (required for identity check)"
            )
        return {}

    man_task = man.get("task")
    if man_task is not None and man_task != expected_task:
        raise ValueError(
            f"run_manifest task {man_task!r} != expected task {expected_task!r} "
            f"(run={base})"
        )

    man_seed = man.get("training_seed")
    folder_seed = None
    m = re.search(r"_seed(\d+)$", base)
    if m:
        folder_seed = int(m.group(1))

    if expected_training_seed is not None:
        if man_seed is not None and int(man_seed) != int(expected_training_seed):
            raise ValueError(
                f"run_manifest training_seed {man_seed} != expected {expected_training_seed} "
                f"(run={base})"
            )
        if folder_seed is not None and folder_seed != int(expected_training_seed):
            raise ValueError(
                f"run folder seed {folder_seed} != expected {expected_training_seed} "
                f"(run={base})"
            )

    if man_seed is not None and folder_seed is not None and int(man_seed) != folder_seed:
        raise ValueError(
            f"run_manifest training_seed {man_seed} != folder _seed{folder_seed} "
            f"(run={base})"
        )

    return man


def atomic_savez(path: str, **arrays) -> str:
    """Write arrays via a sibling temp ``.npz`` then ``os.replace`` (atomic on same FS).

    Important: ``numpy.savez`` appends ``.npz`` unless the path already ends with
    ``.npz``. The temp name must therefore end in ``.npz``, otherwise this would
    create ``path.tmp.npz`` and ``os.replace(path.tmp, path)`` would FileNotFoundError.
    The final ``path`` itself need not end in ``.npz`` (orchestrator may use
    ``*.partial.npz`` or similar); the file is still a valid npz archive.
    """
    import tempfile

    path = os.path.abspath(path)
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)

    packed = {}
    for k, v in arrays.items():
        if v is None:
            packed[k] = np.asarray(None)
        elif isinstance(v, (dict, list, tuple)):
            packed[k] = np.asarray(v, dtype=object)
        else:
            packed[k] = v

    # Same directory so os.replace is atomic; suffix MUST be .npz for np.savez.
    fd, tmp = tempfile.mkstemp(prefix=".atomic_savez_", suffix=".npz", dir=parent)
    os.close(fd)
    try:
        np.savez(tmp, **packed)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def npz_is_valid(path: str, required_keys=()) -> bool:
    """True if path exists, is a readable npz, and has required keys with finite primary metrics."""
    if not path or not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            keys = set(z.files)
            for k in required_keys:
                if k not in keys:
                    return False
            # Fail-loud on NaN/Inf in common primary metrics if present
            for k in ("tracking_lin_err_mean", "tracking_lin_err", "fall_rate_mean", "fall_rate"):
                if k in keys:
                    arr = np.asarray(z[k], dtype=np.float64)
                    if not np.isfinite(arr).all():
                        return False
        return True
    except Exception:
        return False
