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


BANK_SCHEMA_VERSION = 3  # v3: per-(type, level) heightfield-byte geometry pins
ARTIFACT_SCHEMA_VERSION = 3  # v3: protocol_fingerprint binds horizon/success/selection knobs
# Protocol fingerprint covers the frozen scoring knobs the bank fingerprint does
# NOT: rollout horizon (steps/warmup), DR mode, success thresholds, CVaR
# fraction, selection metric/version and primary tolerance.  It is computed from
# the *effective* horizon actually rolled out (from provenance), so a short smoke
# override can never masquerade as the frozen headline artifact (reviewer 8b).
# Merge only requires shards to *agree* on a protocol (so a short-horizon smoke
# shard set can still produce a complete artifact); selection refuses any
# protocol_fingerprint that is not the frozen headline one.
PROTOCOL_SCHEMA_VERSION = 1
# Shard partials carry a provenance header binding them to a checkpoint SHA,
# identity and protocol so a merge can never legitimise stale/foreign shards or
# silently mix protocols (reviewer 8a).
SHARD_SCHEMA_VERSION = 1

# Runtime-geometry pin version.  MUST equal
# ``legged_gym.utils.terrain.GEOMETRY_HASH_VERSION`` -- asserted in
# ``tests/test_ued_checkpoint_selection.py`` -- but kept as a local literal so
# this module stays simulator-neutral (no Genesis import at load time).
GEOMETRY_HASH_VERSION = "v5_ued_geometry_v2_hfbytes"

# Command-draw seed streams: the validation bank selects checkpoints, the
# held-out bank measures the winner ONCE and never feeds selection (§7).
VALIDATION_SEED_KEY = "validation_seed"
HOLDOUT_SEED_KEY = "eval_seed"
BANK_KINDS = {"validation": VALIDATION_SEED_KEY, "holdout": HOLDOUT_SEED_KEY}

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
    command_vy: float
    command_yaw: float
    geometry_hash: str

    def key(self) -> tuple[int, int]:
        return self.cell_id, self.replica_id


def replicas_per_cell(config: Mapping[str, Any]) -> int:
    return int(config["validation_bank"]["replicas_per_cell"])


def geometry_pins(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Frozen per-(terrain_type, level) heightfield hashes, JSON-stable.

    Level keys are normalised to strings so the pins survive a JSON round-trip
    (YAML parses ``0:`` as an int key; a serialised artifact carries ``"0"``) and
    compare equal on both sides of the boundary."""
    raw = config["validation_bank"]["geometry_hashes"]
    return {
        str(terrain_type): {str(level): str(value) for level, value in dict(levels).items()}
        for terrain_type, levels in dict(raw).items()
    }


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a frozen UED evaluation configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("UED evaluation configuration must be a mapping")
    validate_config(config)
    for kind, pin_key in (("validation", "bank_fingerprint"), ("holdout", "holdout_bank_fingerprint")):
        pinned = config["validation_bank"].get(pin_key)
        if pinned is not None and str(pinned) != bank_fingerprint(config, kind=kind):
            raise ValueError(f"{kind} bank fingerprint pin does not match deterministic bank")
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
    support = bank.get("command_support")
    if not isinstance(support, Mapping) or set(support) != {"lin_vel_y", "ang_vel_yaw"}:
        raise ValueError(
            "validation_bank.command_support must pin exactly lin_vel_y and ang_vel_yaw ranges"
        )
    for axis in ("lin_vel_y", "ang_vel_yaw"):
        pair = support[axis]
        if (not isinstance(pair, Sequence) or isinstance(pair, str) or len(pair) != 2
                or float(pair[1]) <= float(pair[0])):
            raise ValueError(f"command_support.{axis} must be an increasing [lo, hi] pair")
    for pin_key in ("bank_fingerprint", "holdout_bank_fingerprint"):
        pinned_fingerprint = bank.get(pin_key)
        if not isinstance(pinned_fingerprint, str) or len(pinned_fingerprint) != 64:
            raise ValueError(f"validation_bank must pin its deterministic {pin_key}")
    if bank["bank_fingerprint"] == bank["holdout_bank_fingerprint"]:
        raise ValueError("validation and held-out banks must have distinct fingerprints")
    flat_type = str(bank.get("flat_terrain_type", "flat"))
    flat_level = int(bank.get("flat_terrain_level", 0))
    hashes = bank.get("geometry_hashes")
    expected_hash_keys = set(terrain_types) | {flat_type}
    if not isinstance(hashes, Mapping) or set(map(str, hashes)) != expected_hash_keys:
        raise ValueError("geometry_hashes must pin every moving terrain and flat")
    moving_levels = {int(level) for level in levels}
    for terrain_type, level_hashes in hashes.items():
        if not isinstance(level_hashes, Mapping):
            raise ValueError(f"geometry_hashes.{terrain_type} must map level -> SHA-256")
        present = {int(level) for level in level_hashes}
        expected_levels = {flat_level} if str(terrain_type) == flat_type else moving_levels
        if present != expected_levels:
            raise ValueError(f"geometry_hashes.{terrain_type} must pin exactly levels {sorted(expected_levels)}")
        if any(not isinstance(value, str) or len(value) != 64 for value in level_hashes.values()):
            raise ValueError("geometry hashes must be 64-character SHA-256 strings")
    if str(bank.get("geometry_hash_version")) != GEOMETRY_HASH_VERSION:
        raise ValueError(f"geometry_hash_version must be {GEOMETRY_HASH_VERSION}")
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


def command_support(config: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """Return the frozen (lo, hi) support for the vy / omega_z nuisance axes.

    These MUST equal the training command ranges (``Go2V5CommonCfg.commands.
    ranges.lin_vel_y`` / ``ang_vel_yaw``) so the offline bank and the online
    sampler share one 3-axis command distribution (solun_plani.md §2/§14.1).
    """
    support = config["validation_bank"]["command_support"]
    out: dict[str, tuple[float, float]] = {}
    for axis in ("lin_vel_y", "ang_vel_yaw"):
        lo, hi = (float(v) for v in support[axis])
        if hi <= lo:
            raise ValueError(f"command_support.{axis} must be an increasing [lo, hi] pair")
        out[axis] = (lo, hi)
    return out


def yaw_support_scale(config: Mapping[str, Any]) -> float:
    """SPNTE yaw-tracking scale = |bound| of the omega_z support."""
    lo, hi = command_support(config)["ang_vel_yaw"]
    return max(abs(lo), abs(hi))


def build_bank(config: Mapping[str, Any], *, seed_key: str = VALIDATION_SEED_KEY) -> list[BankRow]:
    """Precompute every deterministic in-bin command draw in the 84 × R bank.

    Each replica freezes a full 3-axis command: ``command_vx`` uniform within
    its velocity bin, plus ``command_vy`` and ``command_yaw`` (omega_z) uniform
    over the shared nuisance support.  All three are drawn from ``config[seed_key]``
    so the whole matrix is byte-reproducible and the fingerprint pins it.  The
    validation bank (``validation_seed``) selects the checkpoint; the held-out
    bank (``eval_seed``) shares the SAME terrain grid but draws INDEPENDENT
    commands, giving genuinely disjoint rows for the final unbiased measurement.
    ``geometry_hash`` is the frozen per-(type, level) pin the rollout must
    reproduce from the real heightfield -- it is NOT a source of truth here.
    """
    if seed_key not in (VALIDATION_SEED_KEY, HOLDOUT_SEED_KEY):
        raise ValueError(f"unknown bank seed_key {seed_key!r}")
    cells = validation_cells(config)
    if len(cells) != FROZEN_NUM_CELLS:
        raise AssertionError(f"frozen UED bank must have {FROZEN_NUM_CELLS} cells, got {len(cells)}")
    bank = config["validation_bank"]
    pins = geometry_pins(config)
    n_replicas = replicas_per_cell(config)
    edges = np.asarray(bank["velocity_bin_edges"], dtype=np.float64)
    (vy_lo, vy_hi) = command_support(config)["lin_vel_y"]
    (yaw_lo, yaw_hi) = command_support(config)["ang_vel_yaw"]
    rng = np.random.Generator(np.random.PCG64(int(config[seed_key])))
    rows: list[BankRow] = []
    for cell_id, (terrain_type, terrain_level, vx_bin) in enumerate(cells):
        vx_draws = rng.uniform(edges[vx_bin], edges[vx_bin + 1], size=n_replicas)
        vy_draws = rng.uniform(vy_lo, vy_hi, size=n_replicas)
        yaw_draws = rng.uniform(yaw_lo, yaw_hi, size=n_replicas)
        for replica_id in range(n_replicas):
            rows.append(BankRow(
                cell_id=cell_id, terrain_type=terrain_type, terrain_level=terrain_level,
                vx_bin=vx_bin, replica_id=replica_id,
                command_vx=float(vx_draws[replica_id]),
                command_vy=float(vy_draws[replica_id]),
                command_yaw=float(yaw_draws[replica_id]),
                geometry_hash=pins[str(terrain_type)][str(terrain_level)],
            ))
    expected = FROZEN_NUM_CELLS * n_replicas
    if len(rows) != expected:
        raise AssertionError(f"frozen UED bank must contain {expected} replicas, got {len(rows)}")
    return rows


def build_validation_bank(config: Mapping[str, Any]) -> list[BankRow]:
    """The checkpoint-selection bank (``validation_seed`` command stream)."""
    return build_bank(config, seed_key=VALIDATION_SEED_KEY)


def build_holdout_bank(config: Mapping[str, Any]) -> list[BankRow]:
    """The held-out final bank (``eval_seed`` command stream); never selects."""
    return build_bank(config, seed_key=HOLDOUT_SEED_KEY)


def bank_fingerprint(
    config: Mapping[str, Any], bank: Sequence[BankRow] | None = None, *, kind: str = "validation",
) -> str:
    """Fingerprint the frozen config and the actual precomputed command matrix.

    ``kind`` selects the validation or held-out command stream; both banks share
    the geometry pins but their disjoint rows (and the ``bank_kind`` tag) give
    each a distinct fingerprint so an artifact can never masquerade as the other.
    """
    if kind not in BANK_KINDS:
        raise ValueError(f"unknown bank kind {kind!r}")
    rows = build_bank(config, seed_key=BANK_KINDS[kind]) if bank is None else list(bank)
    frozen = {
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_kind": kind,
        "validation_seed": int(config["validation_seed"]),
        "eval_seed": int(config["eval_seed"]),
        "geometry_hash_version": config["validation_bank"]["geometry_hash_version"],
        "geometry_hashes": geometry_pins(config),
        "rows": [asdict(row) for row in rows],
    }
    return _sha256(frozen)


def support_scale(config: Mapping[str, Any]) -> float:
    edges = [float(value) for value in config["validation_bank"]["velocity_bin_edges"]]
    return max(abs(min(edges)), abs(max(edges)))


def protocol_fingerprint(
    config: Mapping[str, Any], *, rollout_steps: int | None = None, warmup_steps: int | None = None,
) -> str:
    """Fingerprint the frozen scoring knobs the bank fingerprint does not cover.

    ``rollout_steps`` / ``warmup_steps`` default to the frozen config horizon; a
    caller that actually rolled out a shorter (smoke) horizon must pass the
    effective values so the resulting fingerprint diverges from the frozen one
    and the selector rejects it (reviewer 8b).
    """
    rollout = config["rollout"]
    success = config["success"]
    selection = config["selection"]
    frozen = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "rollout_steps": int(rollout["steps"] if rollout_steps is None else rollout_steps),
        "warmup_steps": int(rollout["warmup_steps"] if warmup_steps is None else warmup_steps),
        "domain_randomization": str(rollout.get("domain_randomization", "pinned_nominal")),
        "minimum_survival_steps": float(success["minimum_survival_steps"]),
        "spnte_lin_lt": float(success["spnte_lin_lt"]),
        "replica_success_rate_threshold": float(success["replica_success_rate_threshold"]),
        "selection_metric": str(selection["metric"]),
        "selection_version": str(selection["version"]),
        "primary_tolerance": float(selection["primary_tolerance"]),
        "worst_task_fraction": float(selection["worst_task_fraction"]),
    }
    return _sha256(frozen)


def make_shard_payload(
    *, checkpoint_iteration: int, checkpoint_sha256: str, bank_kind: str,
    shard_index: int, num_shards: int, config: Mapping[str, Any],
    measurements: Iterable[Mapping[str, Any]],
    rollout_steps: int | None = None, warmup_steps: int | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap one shard's raw measurements in a provenance header (reviewer 8a).

    The header binds the measurements to a specific checkpoint SHA, identity,
    protocol and shard layout so ``merge_shard_payloads`` can refuse to merge
    stale, foreign or protocol-mismatched partials.
    """
    if bank_kind not in BANK_KINDS:
        raise ValueError(f"unknown bank kind {bank_kind!r}")
    if not (0 <= int(shard_index) < int(num_shards)):
        raise ValueError(f"shard_index {shard_index} out of range for num_shards {num_shards}")
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "checkpoint_iteration": int(checkpoint_iteration),
        "checkpoint_sha256": str(checkpoint_sha256),
        "bank_kind": str(bank_kind),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "protocol_fingerprint": protocol_fingerprint(
            config, rollout_steps=rollout_steps, warmup_steps=warmup_steps,
        ),
        "validation_bank_fingerprint": bank_fingerprint(config, kind=bank_kind),
        "provenance": dict(provenance) if provenance is not None else None,
        "measurements": list(measurements),
    }


def merge_shard_payloads(
    payloads: Sequence[Mapping[str, Any]], *, checkpoint_iteration: int, checkpoint_sha256: str,
    bank_kind: str, num_shards: int, config: Mapping[str, Any],
    rollout_steps: int | None = None, warmup_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Verify shard headers agree with each other and the live checkpoint.

    Fails closed on a bare (headerless) shard, a SHA/identity mismatch, a wrong
    shard count, or a missing/duplicate shard index -- returning the concatenated
    measurements only when every partial provably belongs to this exact
    checkpoint and a *shared* protocol.

    Protocol policy (reviewer 8a/8b, smoke path):
      * All shards must share one ``protocol_fingerprint`` (consensus).
      * If ``rollout_steps`` / ``warmup_steps`` are given (the effective horizon
        the shards were rolled with), that consensus must equal
        ``protocol_fingerprint(config, ...)`` for that horizon -- so a smoke
        merge can pass while a mixed-horizon set cannot.
      * If no effective horizon is supplied, consensus alone is enough: the
        short-horizon smoke path can produce a complete artifact. Headline
        selection still rejects any non-frozen protocol fingerprint.
    """
    expected_bank = bank_fingerprint(config, kind=bank_kind)
    # None => discover from the first envelope; non-None => force this fingerprint.
    if rollout_steps is not None or warmup_steps is not None:
        expected_protocol: str | None = protocol_fingerprint(
            config, rollout_steps=rollout_steps, warmup_steps=warmup_steps,
        )
    else:
        expected_protocol = None
    seen_indices: dict[int, Mapping[str, Any]] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping) or "measurements" not in payload:
            raise ValueError("shard partial is not a provenance-bound envelope (re-run the rollout)")
        if int(payload.get("schema_version", -1)) != SHARD_SCHEMA_VERSION:
            raise ValueError(f"shard schema_version mismatch: {payload.get('schema_version')!r}")
        if int(payload.get("checkpoint_iteration", -1)) != int(checkpoint_iteration):
            raise ValueError("shard checkpoint_iteration mismatch")
        if str(payload.get("checkpoint_sha256")) != str(checkpoint_sha256):
            raise ValueError("shard checkpoint SHA-256 does not match the live checkpoint")
        if str(payload.get("bank_kind")) != str(bank_kind):
            raise ValueError("shard bank_kind mismatch")
        if int(payload.get("num_shards", -1)) != int(num_shards):
            raise ValueError("shard num_shards mismatch")
        if payload.get("validation_bank_fingerprint") != expected_bank:
            raise ValueError("shard validation-bank fingerprint mismatch")
        shard_protocol = payload.get("protocol_fingerprint")
        if expected_protocol is None:
            expected_protocol = shard_protocol
        elif shard_protocol != expected_protocol:
            raise ValueError(
                "shard protocol fingerprint mismatch "
                "(shards disagree, or do not match the effective merge horizon)"
            )
        index = int(payload.get("shard_index", -1))
        if not (0 <= index < int(num_shards)):
            raise ValueError(f"shard_index {index} out of range for num_shards {num_shards}")
        if index in seen_indices:
            raise ValueError(f"duplicate shard_index {index}")
        seen_indices[index] = payload
    missing = sorted(set(range(int(num_shards))) - set(seen_indices))
    if missing:
        raise ValueError(f"missing shard indices {missing} of {num_shards}")
    measurements: list[dict[str, Any]] = []
    for index in range(int(num_shards)):
        measurements.extend(dict(row) for row in seen_indices[index]["measurements"])
    return measurements


def bank_payload(config: Mapping[str, Any], *, kind: str = "validation") -> dict[str, Any]:
    if kind not in BANK_KINDS:
        raise ValueError(f"unknown bank kind {kind!r}")
    rows = build_bank(config, seed_key=BANK_KINDS[kind])
    return {
        "schema_version": BANK_SCHEMA_VERSION,
        "bank_kind": kind,
        "validation_seed": int(config["validation_seed"]),
        "eval_seed": int(config["eval_seed"]),
        "bank_fingerprint": bank_fingerprint(config, rows, kind=kind),
        "geometry_hash_version": config["validation_bank"]["geometry_hash_version"],
        "geometry_hashes": geometry_pins(config),
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
        for axis, frozen in (
            ("command_vx", expected.command_vx),
            ("command_vy", expected.command_vy),
            ("command_yaw", expected.command_yaw),
        ):
            if axis not in row:
                raise ValueError(f"validation measurement requires frozen {axis} at {key}")
            if not np.isclose(float(row[axis]), frozen, rtol=0.0, atol=1e-12):
                raise ValueError(f"measurement {axis} diverges from frozen bank at {key}")
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
        # Diagnostic yaw-tracking metrics: recorded for transparency (the eval
        # command is now 3-axis) but NOT part of the spnte_lin headline used for
        # selection (solun_plani.md §2 "D3").  Optional so pre-yaw fixtures and
        # partial shards still validate.
        if "spnte_yaw" in row:
            row["spnte_yaw"] = _require_number(row, "spnte_yaw", lower=0.0, upper=1.0)
        if "tracking_ang_err" in row:
            row["tracking_ang_err"] = _require_number(row, "tracking_ang_err", lower=0.0)
        row["geometry_hash"] = expected.geometry_hash
        row["command_vx"] = expected.command_vx
        row["command_vy"] = expected.command_vy
        row["command_yaw"] = expected.command_yaw
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
    checkpoint_sha256: str = "", bank_kind: str = "validation", provenance: Mapping[str, Any] | None = None,
    assignment_distribution: Any = None, ppo_transition_occupancy: Any = None,
) -> dict[str, Any]:
    """Create a serialisable complete-bank artifact from a rollout boundary.

    ``bank_kind`` records whether these measurements came from the validation
    (selection) bank or the held-out final bank; ``provenance`` carries the
    real rollout context (seed, simulator, task, run, commit, horizon, DR) so a
    result can be audited without trusting file names.  The stored
    ``geometry_hashes`` are the frozen pins, while each measurement now carries
    the runtime-computed hash the rollout reported -- ``validate_measurements``
    fails closed if they disagree.
    """
    if bank_kind not in BANK_KINDS:
        raise ValueError(f"unknown bank kind {bank_kind!r}")
    rows = build_bank(config, seed_key=BANK_KINDS[bank_kind])
    canonical = validate_measurements(measurements, config, rows)
    prov = dict(provenance) if provenance is not None else None
    # Fingerprint the horizon actually rolled out (from provenance), not the
    # frozen config default, so a smoke override diverges from the headline.
    proto = protocol_fingerprint(
        config,
        rollout_steps=(prov or {}).get("rollout_steps"),
        warmup_steps=(prov or {}).get("warmup_steps"),
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION, "checkpoint_iteration": int(checkpoint_iteration),
        "bank_kind": bank_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_bank_fingerprint": bank_fingerprint(config, rows, kind=bank_kind),
        "protocol_fingerprint": proto,
        "geometry_hash_version": GEOMETRY_HASH_VERSION,
        "geometry_hashes": geometry_pins(config),
        "provenance": prov,
        "spnte_v_scale": support_scale(config), "measurements": canonical,
        "scores": aggregate_measurements(canonical, config, rows),
        # Never merge the teacher's sampling distribution with buffer occupancy:
        # early failures make their transition exposure inherently different.
        "assignment_distribution": assignment_distribution,
        "ppo_transition_occupancy": ppo_transition_occupancy,
    }


def evaluate_with_rollout(
    checkpoint_iteration: int, config: Mapping[str, Any],
    rollout: Callable[[Sequence[BankRow]], Iterable[Mapping[str, Any]]], *, bank_kind: str = "validation",
    **metadata: Any,
) -> dict[str, Any]:
    """Reusable test/V4 seam: invoke a rollout once with the fixed bank."""
    if bank_kind not in BANK_KINDS:
        raise ValueError(f"unknown bank kind {bank_kind!r}")
    bank = build_bank(config, seed_key=BANK_KINDS[bank_kind])
    return make_validation_artifact(
        checkpoint_iteration=checkpoint_iteration, measurements=rollout(bank), config=config,
        bank_kind=bank_kind, **metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the frozen V5 UED validation bank")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="write the complete deterministic bank JSON")
    parser.add_argument("--kind", choices=sorted(BANK_KINDS), default="validation",
                        help="which command stream to materialise (validation selects; holdout is final)")
    parser.add_argument("--emit-geometry-pins", action="store_true",
                        help="print freshly-built per-(type, level) geometry pins and exit")
    args = parser.parse_args(argv)
    if args.emit_geometry_pins:
        from legged_gym.utils.terrain import build_taxonomy_geometry_hashes
        config = load_config(args.config)
        bank = config["validation_bank"]
        pins = build_taxonomy_geometry_hashes(
            terrain_types=tuple(bank["terrain_types"]),
            flat_terrain_type=str(bank["flat_terrain_type"]),
            flat_terrain_level=int(bank["flat_terrain_level"]),
        )
        print(json.dumps(pins, indent=2, sort_keys=True))
        return 0
    payload = bank_payload(load_config(args.config), kind=args.kind)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
