"""Checkpoint selection + provenance helpers for the eval harness.

Pure helpers (filesystem + optional torch load of metadata only). Unit-testable
without a simulator. Supports:

  * ``best``     -> ``best.pt``
  * ``latest`` / ``-1`` -> highest ``model_<iter>.pt``
  * integer / digit string -> ``model_<iter>.pt``
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional, Union

CkptSpec = Union[int, str]


def normalize_ckpt_spec(checkpoint: CkptSpec) -> Union[int, str]:
    """Normalize a CLI / config checkpoint spec to int or 'best'/'latest'."""
    if isinstance(checkpoint, bool):  # bool is a subclass of int; reject
        raise TypeError(f"Invalid checkpoint spec: {checkpoint!r}")
    if isinstance(checkpoint, int):
        return checkpoint
    if isinstance(checkpoint, str):
        s = checkpoint.strip()
        low = s.lower()
        if low in ("best", "latest"):
            return low
        try:
            return int(s)
        except ValueError as e:
            raise ValueError(
                f"Invalid checkpoint spec {checkpoint!r}: "
                "expected 'best', 'latest', or an integer iteration"
            ) from e
    raise TypeError(f"Invalid checkpoint spec type: {type(checkpoint)!r}")


def ckpt_kind(checkpoint: CkptSpec) -> str:
    """Short label for artifact metadata: 'best' | 'latest' | '<iter>'."""
    spec = normalize_ckpt_spec(checkpoint)
    if spec == "best":
        return "best"
    if spec == "latest" or spec == -1:
        return "latest"
    return str(int(spec))


def _model_iter(filename: str) -> Optional[int]:
    """Parse iteration from ``model_<iter>.pt``; return None if not matching."""
    if not (filename.startswith("model_") and filename.endswith(".pt")):
        return None
    mid = filename[len("model_"):-len(".pt")]
    try:
        return int(mid)
    except ValueError:
        return None


def resolve_checkpoint_path(run_dir: str, checkpoint: CkptSpec = -1) -> str:
    """Resolve the absolute path of the checkpoint file inside ``run_dir``.

    Raises FileNotFoundError if the run dir or chosen file is missing.
    """
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"No run directory: {run_dir}")

    spec = normalize_ckpt_spec(checkpoint)

    if spec == "best":
        path = os.path.join(run_dir, "best.pt")
    elif spec == "latest" or spec == -1:
        models = []
        for f in os.listdir(run_dir):
            it = _model_iter(f)
            if it is not None:
                models.append((it, f))
        if not models:
            raise FileNotFoundError(f"No model_*.pt checkpoints in {run_dir}")
        models.sort(key=lambda t: t[0])
        path = os.path.join(run_dir, models[-1][1])
    else:
        path = os.path.join(run_dir, f"model_{int(spec)}.pt")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return os.path.abspath(path)


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_ckpt_meta(path: str) -> Dict[str, Any]:
    """Read provenance fields from a checkpoint without requiring GPU.

    Fields (may be None if older format):
      iter, training_seed, schedule_stage_start, schedule_lin_vel_x,
      eval_score, eval_it, best_eval_score
    """
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    infos = ck.get("infos") or {}
    eval_score = infos.get("eval_score", ck.get("best_eval_score"))
    if hasattr(eval_score, "item"):
        try:
            eval_score = float(eval_score.item())
        except Exception:
            pass
    return {
        "ckpt_iter": ck.get("iter"),
        "training_seed": ck.get("training_seed"),
        "schedule_stage_start": ck.get("schedule_stage_start"),
        "schedule_lin_vel_x": ck.get("schedule_lin_vel_x"),
        "eval_score": eval_score,
        "eval_it": infos.get("eval_it"),
        "best_eval_score": ck.get("best_eval_score"),
    }


def parse_ckpt_cli(value: str) -> Union[int, str]:
    """argparse type for --ckpt: 'best' | 'latest' | int."""
    try:
        return normalize_ckpt_spec(value)
    except (ValueError, TypeError) as e:
        raise ValueError(str(e)) from e
