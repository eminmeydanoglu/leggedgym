"""Minimal TensorBoard scalar reader (local events only, no network)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import atlas as atlas_mod

REPO_ROOT = atlas_mod.REPO_ROOT

# Scalars of interest
DEFAULT_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Policy/mean_noise_std",
)


def find_event_file(directory: str | Path) -> Path | None:
    directory = Path(directory)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    if not directory.exists():
        return None
    files = list(directory.rglob("events.out.tfevents*"))
    return files[0] if files else None


def load_scalars(
    directory: str | Path,
    tags: tuple[str, ...] = DEFAULT_TAGS,
) -> dict[str, Any]:
    """Return {tag: {"steps": [...], "values": [...]}} or unavailable."""
    path = find_event_file(directory)
    if path is None:
        return {"unavailable": f"no events file under {directory}"}
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {"unavailable": "tensorboard package not importable"}

    try:
        ea = EventAccumulator(str(path.parent), size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        out: dict[str, Any] = {"path": str(path), "tags_available": sorted(available)}
        for tag in tags:
            if tag not in available:
                out[tag] = {"unavailable": "tag missing"}
                continue
            events = ea.Scalars(tag)
            out[tag] = {
                "steps": [int(e.step) for e in events],
                "values": [float(e.value) for e in events],
            }
        # also grab Episode/rew_* if present
        rew_tags = sorted(t for t in available if t.startswith("Episode/rew_"))
        out["reward_components"] = {}
        for tag in rew_tags[:20]:
            events = ea.Scalars(tag)
            out["reward_components"][tag] = {
                "steps": [int(e.step) for e in events],
                "values": [float(e.value) for e in events],
            }
        return out
    except Exception as e:  # noqa: BLE001 — mark unavailable, don't fail report
        return {"unavailable": f"TB read failed: {e}", "path": str(path)}
