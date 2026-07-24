"""Offline V5 UED checkpoint selection from a frozen validation bank.

Only existing ``model_<iteration>.pt`` files participate.  The selector is
intentionally independent of the trainer: it consumes artifacts written by
``ued_validation`` / ``ued_rollout`` and materialises a separate
``best_spnte.pt`` without ever changing ``best_tracking.pt``.

Offline evaluation is **not** launched from training.  After a run, score the
scheduled iterations manually, then call this selector.

Selection schedule (from ``configs/eval/v5_ued.yaml``):
  * start target = ``min_iteration`` (default 1000)
  * for each target T, take the largest existing checkpoint ``<= T``
    (strictly after the previous pick, and ``>= min_iteration``)
  * next target = chosen_iteration + ``iteration_stride`` (default 500)
  * if nothing floors to T, advance T by the stride
  * always include the final periodic checkpoint when configured
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import torch

from legged_gym.scripts.eval.ued_validation import (
    aggregate_measurements,
    bank_fingerprint,
    build_validation_bank,
    geometry_pins,
    load_config,
    protocol_fingerprint,
    support_scale,
    validate_measurements,
)


PERIODIC_CHECKPOINT = re.compile(r"^model_(\d+)\.pt$")


@dataclass(frozen=True)
class Candidate:
    iteration: int
    checkpoint_path: Path
    artifact_path: Path
    artifact: Mapping[str, Any]
    scores: Mapping[str, Any]


def periodic_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """Return existing periodic checkpoints in iteration order (never best/latest)."""
    directory = Path(run_dir)
    candidates = []
    for path in directory.glob("model_*.pt"):
        match = PERIODIC_CHECKPOINT.match(path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    return sorted(candidates)


def scheduled_iterations(
    available: Sequence[int],
    *,
    min_iteration: int = 1000,
    iteration_stride: int = 500,
    always_include_final: bool = True,
) -> list[int]:
    """Choose offline-eval iterations from existing periodic checkpoints.

    Example with saves every 200 through 3000:
      1000 → 1400 (floor of 1500) → 1800 → 2200 → 2600 → 3000 (and final).
    """
    iters = sorted({int(value) for value in available})
    if not iters:
        return []
    if iteration_stride <= 0:
        raise ValueError("iteration_stride must be positive")
    if min_iteration < 0:
        raise ValueError("min_iteration must be non-negative")

    final = iters[-1]
    picks: list[int] = []
    target = int(min_iteration)

    while target <= final:
        cands = [i for i in iters if min_iteration <= i <= target]
        if picks:
            cands = [i for i in cands if i > picks[-1]]
        if not cands:
            target += int(iteration_stride)
            continue
        pick = max(cands)
        picks.append(pick)
        target = pick + int(iteration_stride)

    if always_include_final and final not in picks:
        picks.append(final)
    return picks


def selection_schedule_from_config(
    available: Sequence[int], config: Mapping[str, Any],
) -> list[int]:
    """Apply the frozen selection schedule fields from a loaded config."""
    selection = config["selection"]
    return scheduled_iterations(
        available,
        min_iteration=int(selection.get("min_iteration", 1000)),
        iteration_stride=int(selection.get("iteration_stride", 500)),
        always_include_final=bool(selection.get("always_include_final", True)),
    )


def artifact_path_for(run_dir: str | Path, iteration: int, config: Mapping[str, Any]) -> Path:
    selection = config["selection"]
    return Path(run_dir) / str(selection["artifact_dir"]) / str(selection["artifact_pattern"]).format(iteration=iteration)


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"missing fixed-bank validation artifact: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid validation artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"validation artifact {path} must be a mapping")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_from_artifact(
    iteration: int, checkpoint: Path, artifact_path: Path, config: Mapping[str, Any],
) -> Candidate:
    artifact = _read_artifact(artifact_path)
    if int(artifact.get("checkpoint_iteration", -1)) != iteration:
        raise ValueError(f"{artifact_path}: checkpoint_iteration does not match {checkpoint.name}")
    # Leakage guard: only the validation (selection) bank may pick a checkpoint;
    # the held-out final bank must never influence selection (§7).
    bank_kind = str(artifact.get("bank_kind", "validation"))
    if bank_kind != "validation":
        raise ValueError(
            f"{artifact_path}: refusing '{bank_kind}' bank for checkpoint selection "
            "(the held-out bank must not feed selection)"
        )
    if artifact.get("checkpoint_sha256") != _sha256_file(checkpoint):
        raise ValueError(f"{artifact_path}: checkpoint SHA-256 mismatch")
    expected_fingerprint = bank_fingerprint(config, kind="validation")
    if artifact.get("validation_bank_fingerprint") != expected_fingerprint:
        raise ValueError(f"{artifact_path}: validation-bank fingerprint mismatch")
    # Protocol binding: a short smoke override (or any changed horizon/success/
    # selection knob) fingerprints differently and must never select (reviewer 8b).
    if artifact.get("protocol_fingerprint") != protocol_fingerprint(config):
        raise ValueError(f"{artifact_path}: protocol fingerprint mismatch")
    if artifact.get("geometry_hashes") != geometry_pins(config):
        raise ValueError(f"{artifact_path}: geometry hash pins mismatch")
    if float(artifact.get("spnte_v_scale", -1.0)) != support_scale(config):
        raise ValueError(f"{artifact_path}: SPNTE support scale mismatch")
    # Do not trust precomputed scores: completeness and macro aggregation are
    # re-derived before every selection.
    rows = validate_measurements(artifact.get("measurements", ()), config, build_validation_bank(config))
    scores = aggregate_measurements(rows, config)
    return Candidate(iteration=iteration, checkpoint_path=checkpoint, artifact_path=artifact_path, artifact=artifact, scores=scores)


def load_candidates(run_dir: str | Path, config: Mapping[str, Any]) -> list[Candidate]:
    """Load scheduled periodic candidates; refuse if any scheduled bank is incomplete."""
    checkpoints = periodic_checkpoints(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"{run_dir}: no existing periodic model_<iteration>.pt checkpoints")
    by_iteration = {iteration: path for iteration, path in checkpoints}
    schedule = selection_schedule_from_config(list(by_iteration), config)
    if not schedule:
        raise FileNotFoundError(f"{run_dir}: selection schedule produced no candidate iterations")
    missing = [iteration for iteration in schedule if iteration not in by_iteration]
    if missing:
        raise FileNotFoundError(f"{run_dir}: scheduled iterations missing checkpoints: {missing}")
    return [
        _candidate_from_artifact(
            iteration, by_iteration[iteration], artifact_path_for(run_dir, iteration, config), config,
        )
        for iteration in schedule
    ]


def _tie_break_key(candidate: Candidate) -> tuple[float, float, float, int]:
    """Total order for candidates already inside the primary-tolerance band.

    Smaller tuples win: lower worst-10% CVaR, then lower fall rate, then higher
    success rate (negated), then the earlier iteration.  A total sort key makes
    the winner independent of candidate ordering (reviewer 9).
    """
    scores = candidate.scores
    return (
        float(scores["worst_10pct_cvar_spnte_lin"]),
        float(scores["macro_fall_rate"]),
        -float(scores["macro_success_rate"]),
        int(candidate.iteration),
    )


def select_best(candidates: Iterable[Candidate], config: Mapping[str, Any]) -> Candidate:
    """Permutation-invariant selection: tie-break only inside a global band.

    First find the global best (lowest) primary score, then apply the secondary
    tie-break ONLY among candidates within ``primary + tolerance``.  This avoids
    the non-transitive pairwise-tolerance chaining that let a candidate more than
    a tolerance worse than the field win by riding a chain of near-ties
    (reviewer 9).
    """
    candidates = list(candidates)
    if not candidates:
        raise ValueError("cannot select from no complete validation candidates")
    tolerance = float(config["selection"]["primary_tolerance"])
    best_primary = min(float(c.scores["macro_mean_spnte_lin"]) for c in candidates)
    band = [
        c for c in candidates
        if float(c.scores["macro_mean_spnte_lin"]) <= best_primary + tolerance
    ]
    return min(band, key=_tie_break_key)


def selection_metadata(candidate: Candidate, config: Mapping[str, Any]) -> dict[str, Any]:
    """Stable provenance written into ``best_spnte.pt`` and resume metadata."""
    return {
        "selection_metric": str(config["selection"]["metric"]),
        "selection_version": str(config["selection"]["version"]),
        "score_components": dict(candidate.scores),
        "validation_bank_fingerprint": bank_fingerprint(config, kind="validation"),
        "geometry_hashes": geometry_pins(config),
        "spnte_v_scale": support_scale(config),
        "selected_iteration": int(candidate.iteration),
        "source_checkpoint": candidate.checkpoint_path.name,
        "validation_artifact": candidate.artifact_path.name,
        "selection_min_iteration": int(config["selection"].get("min_iteration", 1000)),
        "selection_iteration_stride": int(config["selection"].get("iteration_stride", 500)),
        # Keep the two distributions visibly distinct in the selection record.
        "assignment_distribution": candidate.artifact.get("assignment_distribution"),
        "ppo_transition_occupancy": candidate.artifact.get("ppo_transition_occupancy"),
    }


def _merge_best_spnte_metadata(checkpoint: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Update selection fields without discarding resume or teacher state."""
    result = dict(checkpoint)
    previous = result.get("best_spnte")
    best_spnte = dict(previous) if isinstance(previous, Mapping) else {}
    best_spnte.update(metadata)
    result["best_spnte"] = best_spnte

    resume = result.get("resume_metadata")
    resume_metadata = dict(resume) if isinstance(resume, Mapping) else {}
    old_resume_selection = resume_metadata.get("best_spnte")
    preserved = dict(old_resume_selection) if isinstance(old_resume_selection, Mapping) else {}
    preserved.update(metadata)
    resume_metadata["best_spnte"] = preserved
    result["resume_metadata"] = resume_metadata
    return result


def materialize_best_spnte(run_dir: str | Path, candidate: Candidate, config: Mapping[str, Any]) -> Path:
    """Copy a selected periodic checkpoint and attach immutable selection metadata."""
    run = Path(run_dir)
    target = run / str(config["selection"]["best_artifact"])
    if candidate.checkpoint_path.resolve() == target.resolve():
        raise ValueError("best_spnte artifact must be materialised from a periodic model checkpoint")
    # Atomic publish: build the fully-annotated checkpoint in a sibling temp file
    # and os.replace() it into place only on success.  An interrupted selection
    # can never leave a truncated best_spnte.pt or clobber a good one mid-write
    # (reviewer 8c).  os.replace is atomic within the same directory/filesystem.
    tmp = target.with_name(target.name + ".tmp")
    try:
        shutil.copy2(candidate.checkpoint_path, tmp)
        checkpoint = torch.load(tmp, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{target}: checkpoint payload must be a mapping")
        torch.save(_merge_best_spnte_metadata(checkpoint, selection_metadata(candidate, config)), tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


def select_run(run_dir: str | Path, config: Mapping[str, Any]) -> tuple[Candidate, Path]:
    """Select and materialise one run's offline SPNTE winner."""
    winner = select_best(load_candidates(run_dir, config), config)
    return winner, materialize_best_spnte(run_dir, winner, config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select V5 UED best_spnte.pt from complete scheduled validations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    winner, target = select_run(args.run_dir, config)
    schedule = selection_schedule_from_config(
        [iteration for iteration, _ in periodic_checkpoints(args.run_dir)], config,
    )
    print(json.dumps({
        "best_spnte": str(target),
        "selected_iteration": winner.iteration,
        "schedule": schedule,
        "scores": winner.scores,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
