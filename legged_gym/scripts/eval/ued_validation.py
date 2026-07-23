"""Frozen V5 UED validation-bank construction and aggregation.

The module deliberately separates the deterministic bank and aggregation
contract from simulator rollout.  A Genesis/V4 caller supplies a ``rollout``
callable that consumes the generated rows and returns one measurement per row;
tests can use the same boundary without constructing a full GPU scene.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml


BANK_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1

# Frozen V5 selection bank size (84 cells × replicas_per_cell).
FROZEN_REPLICAS_PER_CELL = 12
FROZEN_NUM_CELLS = 84


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BankRow:
    """One deterministic command replica in a fixed terrain/velocity cell."""

    cell_id: int
    terrain_type: str
    terrain_level: int
    vx_bin: int
    replica_id: int
    command_vx: float
    geometry_hash: str

    def key(self) -> tuple[int, int]:
        return self.cell_id, self.replica_id


def replicas_per_cell(config: Mapping[str, Any]) -> int:
    return int(config["validation_bank"]["replicas_per_cell"])


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen UED evaluation configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("UED evaluation configuration must be a mapping")
    validate_config(config)
    pinned = config["validation_bank"].get("bank_fingerprint")
    if pinned is not None and str(pinned) != bank_fingerprint(config):
        raise ValueError("validation bank fingerprint pin does not match deterministic bank")
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"validation_seed", "eval_seed", "validation_bank", "rollout", "success", "selection"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"UED evaluation configuration is missing {sorted(missing)}")
    bank = config["validation_bank"]
    if not isinstance(bank, Mapping):
        raise ValueError("validation_bank must be a mapping")
    terrain_types = tuple(bank.get("terrain_types", ()))
    levels = tuple(bank.get("terrain_levels", ()))
    edges = tuple(float(value) for value in bank.get("velocity_bin_edges", ()))
    if len(terrain_types) != 5 or len(set(terrain_types)) != 5:
        raise ValueError("validation bank requires exactly five distinct moving terrain types")
    if len(levels) != 4 or len(set(levels)) != 4:
        raise ValueError("validation bank requires exactly four terrain levels")
    if len(edges) != 5 or any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("validation bank requires four increasing velocity bins")
    if int(bank.get("replicas_per_cell", 0)) != FROZEN_REPLICAS_PER_CELL:
        raise ValueError(
            f"frozen validation bank requires {FROZEN_REPLICAS_PER_CELL} replicas per cell"
        )
    pinned_fingerprint = bank.get("bank_fingerprint")
    if not isinstance(pinned_fingerprint, str) or len(pinned_fingerprint) != 64:
        raise ValueError("validation_bank must pin its deterministic fingerprint")
    hashes = bank.get("geometry_hashes")
    expected_hash_keys = set(terrain_types) | {str(bank.get("flat_terrain_type", "flat"))}
    if not isinstance(hashes, Mapping) or set(hashes) != expected_hash_keys:
        raise ValueError("geometry_hashes must pin every moving terrain and flat")
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("geometry hashes must be 64-character SHA-256 strings")
    if int(config["validation_seed"]) != 31001 or int(config["eval_seed"]) != 41001:
        raise ValueError("frozen UED validation and held-out seeds are 31001 and 41001")

    selection = config["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("selection must be a mapping")
    min_iteration = int(selection.get("min_iteration", 1000))
    iteration_stride = int(selection.get("iteration_stride", 500))
    if min_iteration < 0:
        raise ValueError("selection.min_iteration must be non-negative")
    if iteration_stride <= 0:
        raise ValueError("selection.iteration_stride must be positive")
    if "always_include_final" in selection and not isinstance(selection["always_include_final"], bool):
        raise ValueError("selection.always_include_final must be a boolean")


def validation_cells(config: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    """Return the stable 84-cell order: terrain/level (then flat) × velocity."""
    validate_config(config)
    bank = config["validation_bank"]
    terrain = [(str(kind), int(level)) for kind in bank["terrain_types"] for level in bank["terrain_levels"]]
    terrain.append((str(bank["flat_terrain_type"]), int(bank["flat_terrain_level"])))
    return [(kind, level, vx_bin) for vx_bin in range(4) for kind, level in terrain]


def build_validation_bank(config: Mapping[str, Any]) -> list[BankRow]:
    """Precompute every deterministic in-bin command draw in the 84 × R bank."""
    cells = validation_cells(config)
    if len(cells) != FROZEN_NUM_CELLS:
        raise AssertionError(f"frozen UED bank must have {FROZEN_NUM_CELLS} cells, got {len(cells)}")
    bank = config["validation_bank"]
    n_replicas = replicas_per_cell(config)
    edges = np.asarray(bank["velocity_bin_edges"], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(int(config["validation_seed"])))
    rows: list[BankRow] = []
    for cell_id, (terrain_type, terrain_level, vx_bin) in enumerate(cells):
        draws = rng.uniform(edges[vx_bin], edges[vx_bin + 1], size=n_replicas)
        for replica_id, command_vx in enumerate(draws):
            rows.append(BankRow(
                cell_id=cell_id, terrain_type=terrain_type, terrain_level=terrain_level,
                vx_bin=vx_bin, replica_id=replica_id, command_vx=float(command_vx),
                geometry_hash=str(bank["geometry_hashes"][terrain_type]),
            ))
    expected = FROZEN_NUM_CELLS * n_replicas
    if len(rows) != expected:
        raise AssertionError(f"frozen UED bank must contain {expected} replicas, got {len(rows)}")
    return rows


def bank_fingerprint(config: Mapping[str, Any], bank: Sequence[BankRow] | None = None) -> str:
    """Fingerprint the frozen config and the actual precomputed command matrix."""
    rows = build_validation_bank(config) if bank is None else list(bank)
    frozen = {
        "schema_version": BANK_SCHEMA_VERSION,
        "validation_seed": int(config["validation_seed"]),
        "eval_seed": int(config["eval_seed"]),
        "geometry_hash_version": config["validation_bank"]["geometry_hash_version"],
        "geometry_hashes": dict(config["validation_bank"]["geometry_hashes"]),
        "rows": [asdict(row) for row in rows],
    }
    return _sha256(frozen)


def support_scale(config: Mapping[str, Any]) -> float:
    edges = [float(value) for value in config["validation_bank"]["velocity_bin_edges"]]
    return max(abs(min(edges)), abs(max(edges)))


def bank_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = build_validation_bank(config)
    return {
        "schema_version": BANK_SCHEMA_VERSION,
        "validation_seed": int(config["validation_seed"]),
        "eval_seed": int(config["eval_seed"]),
        "bank_fingerprint": bank_fingerprint(config, rows),
        "geometry_hashes": dict(config["validation_bank"]["geometry_hashes"]),
        "spnte_v_scale": support_scale(config),
        "rows": [asdict(row) for row in rows],
    }


def _require_number(row: Mapping[str, Any], key: str, *, lower: float | None = None, upper: float | None = None) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"validation measurement requires finite {key}") from error
    if not np.isfinite(value):
        raise ValueError(f"validation measurement has non-finite {key}")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"validation measurement has out-of-range {key}")
    return value


def validate_measurements(
    measurements: Iterable[Mapping[str, Any]], config: Mapping[str, Any], bank: Sequence[BankRow] | None = None,
) -> list[dict[str, Any]]:
    """Fail closed unless a candidate reports each frozen replica exactly once."""
    rows = build_validation_bank(config) if bank is None else list(bank)
    required = {row.key(): row for row in rows}
    seen: dict[tuple[int, int], dict[str, Any]] = {}
    for input_row in measurements:
        row = dict(input_row)
        try:
            key = int(row["cell_id"]), int(row["replica_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("validation measurement requires cell_id and replica_id") from error
        if key not in required:
            raise ValueError(f"measurement references unknown bank replica {key}")
        if key in seen:
            raise ValueError(f"duplicate measurement for bank replica {key}")
        expected = required[key]
        if "command_vx" not in row:
            raise ValueError(f"validation measurement requires frozen command_vx at {key}")
        if not np.isclose(float(row["command_vx"]), expected.command_vx, rtol=0.0, atol=1e-12):
            raise ValueError(f"measurement command diverges from frozen bank at {key}")
        if "geometry_hash" not in row:
            raise ValueError(f"validation measurement requires geometry_hash at {key}")
        if str(row["geometry_hash"]) != expected.geometry_hash:
            raise ValueError(f"measurement geometry hash diverges from frozen bank at {key}")
        row["cell_id"], row["replica_id"] = key
        row["spnte_lin"] = _require_number(row, "spnte_lin", lower=0.0, upper=1.0)
        row["fall_rate"] = _require_number(row, "fall_rate", lower=0.0, upper=1.0)
        row["survival_steps"] = _require_number(
            row, "survival_steps", lower=0.0, upper=float(config["rollout"]["steps"]),
        )
        row["geometry_hash"] = expected.geometry_hash
        row["command_vx"] = expected.command_vx
        seen[key] = row
    missing = sorted(set(required).difference(seen))
    if missing:
        raise ValueError(f"incomplete validation bank: missing {len(missing)} of {len(required)} replicas")
    return [seen[row.key()] for row in rows]


def aggregate_measurements(
    measurements: Iterable[Mapping[str, Any]], config: Mapping[str, Any], bank: Sequence[BankRow] | None = None,
) -> dict[str, Any]:
    """Macro aggregate replica SPNTE to 84 cells and then equally weight cells."""
    rows = build_validation_bank(config) if bank is None else list(bank)
    measured = validate_measurements(measurements, config, rows)
    n_replicas = replicas_per_cell(config)
    success = config["success"]
    min_survival, max_spnte = float(success["minimum_survival_steps"]), float(success["spnte_lin_lt"])
    grouped: dict[int, list[dict[str, Any]]] = {cell_id: [] for cell_id in range(FROZEN_NUM_CELLS)}
    for row in measured:
        grouped[int(row["cell_id"])].append(row)
    cells: list[dict[str, Any]] = []
    for cell_id, members in grouped.items():
        if len(members) != n_replicas:
            raise AssertionError(
                f"complete measurement validation must produce {n_replicas} replicas per cell"
            )
        spnte_values = np.asarray([member["spnte_lin"] for member in members], dtype=np.float64)
        fall_values = np.asarray([member["fall_rate"] for member in members], dtype=np.float64)
        replica_success = np.asarray([
            member["survival_steps"] >= min_survival and member["spnte_lin"] < max_spnte for member in members
        ], dtype=np.float64)
        first = rows[cell_id * n_replicas]
        cells.append({
            "cell_id": cell_id, "terrain_type": first.terrain_type, "terrain_level": first.terrain_level,
            "vx_bin": first.vx_bin, "spnte_lin": float(spnte_values.mean()),
            "fall_rate": float(fall_values.mean()), "replica_success_rate": float(replica_success.mean()),
            "cell_success": bool(replica_success.mean() >= float(success["replica_success_rate_threshold"])),
        })
    cell_spnte = np.asarray([cell["spnte_lin"] for cell in cells], dtype=np.float64)
    cvar_count = max(1, int(np.ceil(len(cells) * float(config["selection"]["worst_task_fraction"]))))
    worst = np.sort(cell_spnte)[-cvar_count:]
    return {
        "macro_mean_spnte_lin": float(cell_spnte.mean()),
        "worst_10pct_cvar_spnte_lin": float(worst.mean()),
        "macro_fall_rate": float(np.mean([cell["fall_rate"] for cell in cells])),
        "macro_success_rate": float(np.mean([cell["replica_success_rate"] for cell in cells])),
        "cell_success_rate": float(np.mean([cell["cell_success"] for cell in cells])),
        "cells": cells,
    }


def make_validation_artifact(
    *, checkpoint_iteration: int, measurements: Iterable[Mapping[str, Any]], config: Mapping[str, Any],
    checkpoint_sha256: str = "", assignment_distribution: Any = None, ppo_transition_occupancy: Any = None,
) -> dict[str, Any]:
    """Create a serialisable complete-bank artifact from a rollout boundary."""
    rows = build_validation_bank(config)
    canonical = validate_measurements(measurements, config, rows)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION, "checkpoint_iteration": int(checkpoint_iteration),
        "checkpoint_sha256": checkpoint_sha256, "validation_bank_fingerprint": bank_fingerprint(config, rows),
        "geometry_hashes": dict(config["validation_bank"]["geometry_hashes"]),
        "spnte_v_scale": support_scale(config), "measurements": canonical,
        "scores": aggregate_measurements(canonical, config, rows),
        # Never merge the teacher's sampling distribution with buffer occupancy:
        # early failures make their transition exposure inherently different.
        "assignment_distribution": assignment_distribution,
        "ppo_transition_occupancy": ppo_transition_occupancy,
    }


def evaluate_with_rollout(
    checkpoint_iteration: int, config: Mapping[str, Any],
    rollout: Callable[[Sequence[BankRow]], Iterable[Mapping[str, Any]]], **metadata: Any,
) -> dict[str, Any]:
    """Reusable test/V4 seam: invoke a rollout once with the fixed bank."""
    bank = build_validation_bank(config)
    return make_validation_artifact(
        checkpoint_iteration=checkpoint_iteration, measurements=rollout(bank), config=config, **metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the frozen V5 UED validation bank")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="write the complete deterministic bank JSON")
    args = parser.parse_args(argv)
    payload = bank_payload(load_config(args.config))
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
