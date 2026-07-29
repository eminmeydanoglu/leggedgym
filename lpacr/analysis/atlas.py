"""Shared loader for Curriculum Atlas frame dumps.

Every analysis in this package reads the same artifact: the NDJSON frame stream
written by ``lpacr/dashboard/plugger.py`` during a V5 run (one line per closed
curriculum stage).  Loading is centralised here so that a schema change breaks
one file instead of six, and so every script agrees on which frame index counts
as the bootstrap stage.

Run #1 (Jul24) predates ``performance_sem`` / ``eligible_for_lp`` /
``previous_stage_episode_count`` instrumentation.  ``load`` fills those with
NaN/False so multi-run analysis can proceed; callers must check
``Atlas.available_fields`` (or ``has_sem``) and mark analyses unavailable
rather than treating NaN as zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

# Frame 0 closes the bootstrap stage: it aggregates every episode completed
# before the first LP update, so its episode counts are ~5x a steady-state
# stage and its learning progress is undefined.  No analysis should treat it as
# a normal stage; helpers below therefore start at index 1.
BOOTSTRAP_FRAME = 0

# Regime split established in HISTORY.md 10.4: the vx curriculum signal is alive
# through stage 16 and dead from stage 17 on.  Analyses report both regimes
# separately because they turned out to have opposite pathologies (11.6).
# Indices are into the post-bootstrap stage-pair array (frame i vs i-1, i>=1),
# so EARLY covers stage pairs starting at frames 1..15 (stages 2-16).
EARLY_STAGES = slice(0, 15)
LATE_STAGES = slice(15, None)

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DATA = REPO_ROOT / "logs/lpacr_dashboard_data"

DEFAULT_ATLAS = (
    REPO_ROOT
    / "logs/curriculum_atlas_local/505449_20260727_155936"
    / "lpacrl-fixedbeta-seed1/frames.ndjson"
)

# Full-schema metric keys written by modern plugger.
FULL_METRIC_KEYS = (
    "performance",
    "performance_sem",
    "learning_progress",
    "effective_learning_progress",
    "eligible_for_lp",
    "previous_stage_episode_count",
    "sampling_probability",
    "stage_episode_count",
    "task_assignment_count",
    "task_completion_count",
)

# Analyses that require each optional field.
FIELD_ANALYSIS_GATES: dict[str, list[str]] = {
    "performance_sem": [
        "A1_alpha_sem",
        "B1_episode_sampling_noise",
        "D1_more_episodes_sem_scaling",
        "lp_reliability_fixed_point",
    ],
    "eligible_for_lp": ["eligible_mask_strict"],
    "previous_stage_episode_count": ["harmonic_N"],
    "effective_learning_progress": ["effective_lp_vs_raw"],
}

# Registry of V5 campaign atlas dumps under logs/lpacr_dashboard_data/.
# resume segments of run #2 are listed separately; loaders may stitch them.
RUN_REGISTRY: dict[str, dict[str, Any]] = {
    "run1_lp": {
        "label": "#1 LP (β=5)",
        "campaign_run": 1,
        "arm": "lp_acrl",
        "algo": "lp_acrl",
        "dashboard_dir": "01_Jul24_go2_v5_lpacrl-seed1",
        "notes": "partial schema: no performance_sem / eligible_for_lp / previous_stage_episode_count",
        "partial_schema": True,
        "validation_bank": (
            "logs/go2_v5_ued/Jul24_13-55-22_v5_lp_acrl_genesis_seed1/ued_validation"
        ),
        "tb_events": (
            "logs/go2_v5_ued/Jul24_13-55-22_v5_lp_acrl_genesis_seed1"
        ),
        "pair_group": "run1",
    },
    "run1_uni": {
        "label": "#1 UNI",
        "campaign_run": 1,
        "arm": "uniform",
        "algo": "uniform",
        "dashboard_dir": "01_Jul24_go2_v5_uniform-seed1",
        "notes": "partial schema (same as run1_lp)",
        "partial_schema": True,
        "validation_bank": (
            "logs/go2_v5_ued/Jul24_13-55-38_v5_uniform_genesis_seed1/ued_validation"
        ),
        "tb_events": (
            "logs/go2_v5_ued/Jul24_13-55-38_v5_uniform_genesis_seed1"
        ),
        "pair_group": "run1",
    },
    "run2_lp": {
        "label": "#2 LP (adaptive) + resume",
        "campaign_run": 2,
        "arm": "lp_acrl",
        "algo": "lp_acrl",
        "dashboard_dir": "02_507739_20260727_100500_lpacrl-seed1",
        "resume_dir": "02_507758_20260727_110022_lpacrl-seed1",
        "notes": "full schema; stitch primary+resume segments by step",
        "partial_schema": False,
        "validation_bank": None,  # empty on disk
        "tb_events": "logs/go2_v5_ued_tb/Jul27_11-01-35_v5_lp_acrl_genesis_seed1",
        "pair_group": "run2",
    },
    "run2_uni": {
        "label": "#2 UNI + resume",
        "campaign_run": 2,
        "arm": "uniform",
        "algo": "uniform",
        "dashboard_dir": "02_507739_20260727_100500_uniform-seed1",
        "resume_dir": "02_507758_20260727_110022_uniform-seed1",
        "notes": "full schema; stitch primary+resume",
        "partial_schema": False,
        "validation_bank": None,
        "tb_events": "logs/go2_v5_ued_tb/Jul27_11-01-19_v5_uniform_genesis_seed1",
        "pair_group": "run2",
    },
    "run3_crash": {
        "label": "#3 LP crash (β=1)",
        "campaign_run": 3,
        "arm": "lp_acrl",
        "algo": "lp_acrl",
        "dashboard_dir": "03_Jul27_lpacrl_beta1_crash",
        "notes": "crash run; extra diagnostics (top10_overlap, tv, lp_reliability)",
        "partial_schema": False,
        "validation_bank": None,
        "tb_events": "logs/go2_v5_ued_tb/Jul27_15-08-12_v5_lp_acrl_genesis_seed1",
        "pair_group": None,
    },
    "run4_fixed": {
        "label": "#4 LP fixed β=1",
        "campaign_run": 4,
        "arm": "lp_acrl",
        "algo": "lp_acrl",
        "dashboard_dir": "04_Jul27_lpacrl_beta1_fixed",
        "notes": "§11 sole source; full schema",
        "partial_schema": False,
        "validation_bank": None,  # empty on disk
        "tb_events": "logs/go2_v5_ued_tb/Jul27_16-00-38_v5_lp_acrl_genesis_seed1",
        "pair_group": None,
    },
    "v6_frontier": {
        "label": "V6 frontier reference",
        "campaign_run": 6,
        "arm": "frontier",
        "algo": "frontier",
        "dashboard_dir": "v6-frontier-1500-seed1-20260728",
        "notes": "different schema (success-gated); optional comparison only",
        "partial_schema": True,
        "validation_bank": None,
        "tb_events": None,
        "pair_group": None,
        "optional": True,
    },
}

PRIMARY_V5_RUNS = (
    "run1_lp",
    "run1_uni",
    "run2_lp",
    "run2_uni",
    "run3_crash",
    "run4_fixed",
)


@dataclass(frozen=True)
class Atlas:
    """Per-cell metric arrays stacked over frames, shape ``(frames, cells)``.

    Frame 0 is the bootstrap stage and is retained for cumulative counters, but
    ``defined()`` and most diagnostics start at index 1.  Prefer
    ``load_run`` / ``load`` which drop bootstrap when ``skip_bootstrap=True``.
    """

    step: np.ndarray
    performance: np.ndarray
    performance_sem: np.ndarray
    learning_progress: np.ndarray
    eligible: np.ndarray
    stage_episode_count: np.ndarray
    previous_stage_episode_count: np.ndarray
    sampling_probability: np.ndarray
    task_assignment_count: np.ndarray
    task_completion_count: np.ndarray
    diagnostics: list[dict]
    vx_labels: list[str]
    terrain_labels: list[str]
    # Extended metadata for multi-run analysis
    run_id: str = ""
    available_fields: tuple[str, ...] = FULL_METRIC_KEYS
    missing_fields: tuple[str, ...] = ()
    unavailable_analyses: tuple[str, ...] = ()
    raw_frame_count_including_bootstrap: int = 0
    source_paths: tuple[str, ...] = ()

    @property
    def n_frames(self) -> int:
        return self.performance.shape[0]

    @property
    def n_cells(self) -> int:
        return self.performance.shape[1]

    @property
    def n_terrain(self) -> int:
        return len(self.terrain_labels) if self.terrain_labels else 21

    @property
    def has_sem(self) -> bool:
        return "performance_sem" in self.available_fields and np.isfinite(
            self.performance_sem
        ).any()

    @property
    def has_eligible(self) -> bool:
        return "eligible_for_lp" in self.available_fields

    def cell_name(self, task_id: int) -> str:
        """``task_id = vx_bin * n_terrain + terrain_cell`` (see frame metadata)."""
        vx, terrain = divmod(task_id, self.n_terrain)
        vx_lab = self.vx_labels[vx] if vx < len(self.vx_labels) else str(vx)
        te_lab = (
            self.terrain_labels[terrain]
            if terrain < len(self.terrain_labels)
            else str(terrain)
        )
        return f"{vx_lab} | {te_lab}"

    def group_index(self, axis: str) -> np.ndarray:
        """Marginalisation index over one task-space axis.

        ``"vx"`` collapses the 21 terrain cells inside each velocity band;
        ``"terrain"`` collapses the 4 velocity bands inside each terrain cell.
        Used by the pooling test in 11.3.
        """
        ids = np.arange(self.n_cells)
        if axis == "vx":
            return ids // self.n_terrain
        if axis == "terrain":
            return ids % self.n_terrain
        raise ValueError("axis must be 'vx' or 'terrain'")

    def lp_se(self, t: int) -> np.ndarray:
        """SE of ``LP_t = perf_t - perf_{t-1}``, treating the stages as independent.

        NOTE (11.6): this overstates the true noise in LP.  Part of each stage's
        ``performance_sem`` comes from within-cell command heterogeneity, which
        is present in BOTH stages and largely cancels in the difference -- but
        this formula adds it twice.  Covariate adjustment against the logged
        per-episode commands is the intended fix; until the new instrumentation
        lands, treat every ``lp_se`` here as an upper bound.
        """
        return np.sqrt(
            self.performance_sem[t] ** 2 + self.performance_sem[t - 1] ** 2
        )

    def defined(self, t: int, *, require_eligible: bool = True) -> np.ndarray:
        """Cells whose stage-``t`` LP is usable.

        When ``performance_sem`` is missing (run #1), SEM finiteness is not
        required; eligibility is treated as all-True if the field is absent.
        """
        mask = np.isfinite(self.learning_progress[t])
        if self.has_sem:
            mask = (
                mask
                & np.isfinite(self.performance_sem[t])
                & np.isfinite(self.performance_sem[t - 1])
            )
        if require_eligible and self.has_eligible:
            mask = mask & self.eligible[t]
        return mask

    def quality_require_eligible(self) -> bool:
        """Whether observational quality metrics should AND the eligible mask.

        Uniform arms (and some crash dumps) ship ``eligible_for_lp`` as all-False
        even though LP is finite on every cell.  For those runs, requiring
        eligibility blanks the entire scorecard.  Quality metrics (A1/A3/A5)
        fall back to finite-LP (±SEM) with an explicit mask note instead.
        """
        if not self.has_eligible:
            return False
        # overall eligible fraction across non-bootstrap frames
        if self.n_frames <= 1:
            return bool(self.eligible.any())
        frac = float(np.mean(self.eligible[1:]))
        return frac > 0.05

    def quality_mask_note(self) -> str:
        if not self.has_eligible:
            return "eligible_for_lp absent; quality metrics use finite LP (±SEM)"
        if self.quality_require_eligible():
            frac = float(np.mean(self.eligible[1:])) if self.n_frames > 1 else float(np.mean(self.eligible))
            return f"eligible_for_lp applied (mean frac={frac:.3f})"
        frac = float(np.mean(self.eligible)) if self.eligible.size else 0.0
        return (
            f"eligible_for_lp present but near-zero (mean frac={frac:.3f}); "
            "quality metrics use finite LP (±SEM) — observational null, not sampler gate"
        )

    def to_inventory(self) -> dict[str, Any]:
        elig_frac = None
        if self.has_eligible and self.eligible.size:
            elig_frac = float(np.mean(self.eligible[1:] if self.n_frames > 1 else self.eligible))
        notes = list(self.unavailable_analyses)
        if self.has_eligible and not self.quality_require_eligible():
            notes = list(notes) + [
                "eligible_mask_all_false_or_near_zero: quality metrics use finite LP"
            ]
        return {
            "run_id": self.run_id,
            "n_frames": self.n_frames,
            "n_cells": self.n_cells,
            "step_range": (
                int(self.step[0]) if self.n_frames else None,
                int(self.step[-1]) if self.n_frames else None,
            ),
            "available_fields": list(self.available_fields),
            "missing_fields": list(self.missing_fields),
            "unavailable_analyses": notes,
            "eligible_fraction_mean": elig_frac,
            "quality_mask_note": self.quality_mask_note(),
            "quality_require_eligible": self.quality_require_eligible(),
            "has_sem": self.has_sem,
            "source_paths": list(self.source_paths),
            "raw_frame_count_including_bootstrap": self.raw_frame_count_including_bootstrap,
        }


def _stack_metric(rows: list[dict], key: str, dtype, n_cells: int) -> np.ndarray:
    """Stack a metric; missing key → all-NaN (or all-False for bool)."""
    out = []
    for r in rows:
        m = r.get("metrics") or {}
        if key not in m or m[key] is None:
            if dtype is bool or np.dtype(dtype) == np.bool_:
                out.append(np.ones(n_cells, dtype=bool))  # eligible default True
            else:
                out.append(np.full(n_cells, np.nan, dtype=float))
        else:
            arr = np.asarray(m[key], dtype=dtype if dtype is not bool else float)
            if dtype is bool or np.dtype(dtype) == np.bool_:
                arr = arr.astype(bool)
            out.append(arr)
    return np.array(out)


def _detect_fields(rows: list[dict]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present: set[str] = set()
    for r in rows:
        m = r.get("metrics") or {}
        for k, v in m.items():
            if v is not None:
                present.add(k)
    available = tuple(k for k in FULL_METRIC_KEYS if k in present)
    # also keep any extra keys
    extra = tuple(sorted(present - set(FULL_METRIC_KEYS)))
    available = available + extra
    missing = tuple(k for k in FULL_METRIC_KEYS if k not in present)
    return available, missing


def _unavailable_for(missing: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for field in missing:
        out.extend(FIELD_ANALYSIS_GATES.get(field, []))
    return tuple(dict.fromkeys(out))


def _coords(rows: list[dict]) -> tuple[list[str], list[str]]:
    for r in reversed(rows):
        ts = r.get("task_space") or {}
        coords = ts.get("coordinates") or {}
        if "vx_bin" in coords and "terrain_cell" in coords:
            return list(coords["vx_bin"]), list(coords["terrain_cell"])
    # fallback for frontier / partial dumps
    n = len((rows[-1].get("metrics") or {}).get("performance") or [])
    if n == 84:
        return [f"vx{i}" for i in range(4)], [f"t{i}" for i in range(21)]
    return [f"c{i}" for i in range(max(n, 1))], ["0"]


def _n_cells(rows: list[dict]) -> int:
    for r in rows:
        perf = (r.get("metrics") or {}).get("performance")
        if perf is not None:
            return len(perf)
    raise ValueError("no performance metric in any frame")


def _diagnostics(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        try:
            out.append(r["metadata"]["frame"]["diagnostics"])
        except (KeyError, TypeError):
            out.append({})
    return out


def load(
    path: str | Path = DEFAULT_ATLAS,
    *,
    skip_bootstrap: bool = False,
    run_id: str = "",
) -> Atlas:
    """Load a single frames.ndjson path.

    Missing full-schema fields are filled with NaN (or True for eligibility).

    Bootstrap frame 0 is **kept** by default so ``lp_se(1)`` still sees the
    previous stage's SEM.  All diagnostics must start at index 1
    (``BOOTSTRAP_FRAME``); set ``skip_bootstrap=True`` only for display
    inventories that do not need SEM continuity.
    """
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no frames in {path}")
    raw_count = len(rows)
    if skip_bootstrap and len(rows) > 1:
        rows = rows[1:]
    n_cells = _n_cells(rows)
    available, missing = _detect_fields(rows)
    # If performance_sem never present, mark missing even if we filled NaN
    if not np.any(
        [
            "performance_sem" in (r.get("metrics") or {})
            for r in rows
        ]
    ):
        if "performance_sem" not in missing:
            missing = missing + ("performance_sem",)
        available = tuple(k for k in available if k != "performance_sem")

    vx_labels, terrain_labels = _coords(rows)
    # recompute learning_progress from performance if needed
    performance = _stack_metric(rows, "performance", float, n_cells)
    lp = _stack_metric(rows, "learning_progress", float, n_cells)
    # fill LP from performance deltas where file LP is nan but perf is finite
    for t in range(1, len(rows)):
        need = ~np.isfinite(lp[t]) & np.isfinite(performance[t]) & np.isfinite(
            performance[t - 1]
        )
        if need.any():
            lp[t] = np.where(need, performance[t] - performance[t - 1], lp[t])

    return Atlas(
        step=np.array([r.get("step", 0) for r in rows], dtype=np.int64),
        performance=performance,
        performance_sem=_stack_metric(rows, "performance_sem", float, n_cells),
        learning_progress=lp,
        eligible=_stack_metric(rows, "eligible_for_lp", bool, n_cells),
        stage_episode_count=_stack_metric(
            rows, "stage_episode_count", float, n_cells
        ),
        previous_stage_episode_count=_stack_metric(
            rows, "previous_stage_episode_count", float, n_cells
        ),
        sampling_probability=_stack_metric(
            rows, "sampling_probability", float, n_cells
        ),
        task_assignment_count=_stack_metric(
            rows, "task_assignment_count", float, n_cells
        ),
        task_completion_count=_stack_metric(
            rows, "task_completion_count", float, n_cells
        ),
        diagnostics=_diagnostics(rows),
        vx_labels=vx_labels,
        terrain_labels=terrain_labels,
        run_id=run_id,
        available_fields=available,
        missing_fields=missing,
        unavailable_analyses=_unavailable_for(missing),
        raw_frame_count_including_bootstrap=raw_count,
        source_paths=(str(path),),
    )


def _stitch_rows(paths: list[Path]) -> list[dict]:
    """Concatenate frame streams sorted by step, dropping duplicate steps."""
    all_rows: list[dict] = []
    for p in paths:
        part = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        all_rows.extend(part)
    all_rows.sort(key=lambda r: int(r.get("step", 0)))
    seen: set[int] = set()
    uniq: list[dict] = []
    for r in all_rows:
        s = int(r.get("step", 0))
        if s in seen:
            continue
        seen.add(s)
        uniq.append(r)
    return uniq


def load_rows(
    rows: list[dict],
    *,
    skip_bootstrap: bool = False,
    run_id: str = "",
    source_paths: tuple[str, ...] = (),
) -> Atlas:
    """Build Atlas from already-parsed row dicts (used after stitch)."""
    if not rows:
        raise ValueError("no frames")
    raw_count = len(rows)
    if skip_bootstrap and len(rows) > 1:
        rows = rows[1:]
    n_cells = _n_cells(rows)
    available, missing = _detect_fields(rows)
    if not any("performance_sem" in (r.get("metrics") or {}) for r in rows):
        if "performance_sem" not in missing:
            missing = tuple(list(missing) + ["performance_sem"])
        available = tuple(k for k in available if k != "performance_sem")
    vx_labels, terrain_labels = _coords(rows)
    performance = _stack_metric(rows, "performance", float, n_cells)
    lp = _stack_metric(rows, "learning_progress", float, n_cells)
    for t in range(1, len(rows)):
        need = ~np.isfinite(lp[t]) & np.isfinite(performance[t]) & np.isfinite(
            performance[t - 1]
        )
        if need.any():
            lp[t] = np.where(need, performance[t] - performance[t - 1], lp[t])
    return Atlas(
        step=np.array([r.get("step", 0) for r in rows], dtype=np.int64),
        performance=performance,
        performance_sem=_stack_metric(rows, "performance_sem", float, n_cells),
        learning_progress=lp,
        eligible=_stack_metric(rows, "eligible_for_lp", bool, n_cells),
        stage_episode_count=_stack_metric(
            rows, "stage_episode_count", float, n_cells
        ),
        previous_stage_episode_count=_stack_metric(
            rows, "previous_stage_episode_count", float, n_cells
        ),
        sampling_probability=_stack_metric(
            rows, "sampling_probability", float, n_cells
        ),
        task_assignment_count=_stack_metric(
            rows, "task_assignment_count", float, n_cells
        ),
        task_completion_count=_stack_metric(
            rows, "task_completion_count", float, n_cells
        ),
        diagnostics=_diagnostics(rows),
        vx_labels=vx_labels,
        terrain_labels=terrain_labels,
        run_id=run_id,
        available_fields=available,
        missing_fields=missing,
        unavailable_analyses=_unavailable_for(missing),
        raw_frame_count_including_bootstrap=raw_count,
        source_paths=source_paths,
    )


def dashboard_path(run_key: str) -> Path:
    meta = RUN_REGISTRY[run_key]
    return DASHBOARD_DATA / meta["dashboard_dir"] / "frames.ndjson"


def load_run(
    run_key: str,
    *,
    skip_bootstrap: bool = False,
    stitch_resume: bool = True,
) -> Atlas:
    """Load a registered campaign run (optionally stitching resume segments)."""
    if run_key not in RUN_REGISTRY:
        raise KeyError(f"unknown run_key {run_key!r}; known={list(RUN_REGISTRY)}")
    meta = RUN_REGISTRY[run_key]
    paths = [DASHBOARD_DATA / meta["dashboard_dir"] / "frames.ndjson"]
    if stitch_resume and meta.get("resume_dir"):
        resume = DASHBOARD_DATA / meta["resume_dir"] / "frames.ndjson"
        if resume.is_file():
            paths.append(resume)
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    if len(paths) == 1:
        return load(paths[0], skip_bootstrap=skip_bootstrap, run_id=run_key)
    rows = _stitch_rows(paths)
    return load_rows(
        rows,
        skip_bootstrap=skip_bootstrap,
        run_id=run_key,
        source_paths=tuple(str(p) for p in paths),
    )


def load_all_primary(
    *,
    skip_bootstrap: bool = False,
    include_optional: bool = False,
) -> dict[str, Atlas]:
    """Load the six primary V5 conditions (+ optional V6 if requested).

    Bootstrap frame is retained (see ``load``); analyses skip index 0.
    """
    keys = list(PRIMARY_V5_RUNS)
    if include_optional:
        keys.append("v6_frontier")
    out: dict[str, Atlas] = {}
    for k in keys:
        try:
            out[k] = load_run(k, skip_bootstrap=skip_bootstrap)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            # leave absent rather than fabricate (e.g. V6 frontier schema)
            continue
    return out


def regime_label(stage_pair_index: int) -> str:
    """Name the regime for an index into a per-stage-pair result array."""
    return "early" if stage_pair_index < EARLY_STAGES.stop else "late"


def regime_masks(n_stage_pairs: int) -> dict[str, np.ndarray]:
    """Boolean masks for early/late given number of consecutive stage pairs."""
    idx = np.arange(n_stage_pairs)
    return {
        "early": idx < EARLY_STAGES.stop,
        "late": idx >= EARLY_STAGES.stop,
        "all": np.ones(n_stage_pairs, dtype=bool),
    }
