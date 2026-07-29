"""V6 frontier diagnostic bank, success metrics, merge, and analysis.

Diagnostic-only: never eligible for checkpoint selection.  Separate from the
frozen V5 UED validation bank (``configs/eval/v5_ued.yaml``).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from legged_gym.utils.frontier.task_space import V4FrontierTaskSpace

SCHEMA_VERSION = "v6_frontier_diagnostic_v2"
REGIME_A = "A_baseline"
REGIME_B = "B_vy_only"
REGIME_C = "C_yaw_only"
REGIME_D = "D_joint_fixed"
REGIME_E = "E_training_faithful"
ALL_REGIMES = (REGIME_A, REGIME_B, REGIME_C, REGIME_D, REGIME_E)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {cfg.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if cfg.get("purpose") != "diagnostic_only":
        raise ValueError("config.purpose must be diagnostic_only")
    if cfg.get("eligible_for_checkpoint_selection", True):
        raise ValueError("diagnostic config must set eligible_for_checkpoint_selection=false")
    return cfg


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def bank_fingerprint(rows: Sequence["DiagnosticRow"]) -> str:
    payload = json.dumps(
        [row.to_fingerprint_dict() for row in rows],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def family_for_column(column: int) -> str:
    space = V4FrontierTaskSpace()
    return space.TERRAIN_FAMILIES[space.family_for_column(column)]


def normalized_linear_error(
    cmd_vx: float, cmd_vy: float, act_vx: float, act_vy: float, *, min_scale: float = 0.2
) -> float:
    cmd = np.asarray([cmd_vx, cmd_vy], dtype=np.float64)
    act = np.asarray([act_vx, act_vy], dtype=np.float64)
    scale = max(float(np.linalg.norm(cmd)), float(min_scale))
    return float(np.linalg.norm(cmd - act) / scale)


def normalized_yaw_error(cmd_yaw: float, act_yaw: float, *, min_scale: float = 0.2) -> float:
    scale = max(abs(float(cmd_yaw)), float(min_scale))
    return abs(float(cmd_yaw) - float(act_yaw)) / scale


def frontier_success(
    *,
    timed_out: bool,
    mean_linear_error: float,
    mean_yaw_error: float,
    linear_threshold: float = 0.35,
    yaw_threshold: float = 0.40,
) -> bool:
    return (
        bool(timed_out)
        and float(mean_linear_error) <= float(linear_threshold)
        and float(mean_yaw_error) <= float(yaw_threshold)
    )


def required_success_count(window_size: int, mastery_threshold: float) -> int:
    """Min successes so Beta(1,1) posterior mean (k+1)/(n+2) >= mastery_threshold.

    Matches the training / objective contract (prompt): for window 32,
    0.80→27, 0.75→25, 0.70→23.  This is stricter than raw k/n ≥ threshold.
    """
    n = int(window_size)
    thr = float(mastery_threshold)
    if n <= 0:
        raise ValueError("window_size must be positive")
    if not (0.0 < thr <= 1.0):
        raise ValueError("mastery_threshold must be in (0, 1]")
    # Smallest integer k with (k+1)/(n+2) >= thr  <=>  k >= thr*(n+2) - 1
    return int(math.ceil(thr * (n + 2) - 1.0 - 1e-12))


def _latin_hypercube_pairs(n: int, seed: int) -> list[tuple[float, float]]:
    """Deterministic stratified pairs in [-1, 1]^2 with balanced signs."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(int(seed))
    # One-dimensional LHS on each axis, independently shuffled.
    edges = np.linspace(0.0, 1.0, n + 1)
    u = edges[:-1] + rng.random(n) * (edges[1:] - edges[:-1])
    v = edges[:-1] + rng.random(n) * (edges[1:] - edges[:-1])
    rng.shuffle(u)
    rng.shuffle(v)
    pairs = [(float(2.0 * ui - 1.0), float(2.0 * vi - 1.0)) for ui, vi in zip(u, v)]
    # Ensure both axes contain positive and negative samples when n >= 2.
    signs_u = [1 if p[0] >= 0 else -1 for p in pairs]
    signs_v = [1 if p[1] >= 0 else -1 for p in pairs]
    if n >= 2 and (all(s > 0 for s in signs_u) or all(s < 0 for s in signs_u)):
        pairs[0] = (-abs(pairs[0][0]) - 1e-3, pairs[0][1])
        pairs[0] = (float(np.clip(pairs[0][0], -1.0, 1.0)), pairs[0][1])
    if n >= 2 and (all(s > 0 for s in signs_v) or all(s < 0 for s in signs_v)):
        pairs[1] = (pairs[1][0], -abs(pairs[1][1]) - 1e-3)
        pairs[1] = (pairs[1][0], float(np.clip(pairs[1][1], -1.0, 1.0)))
    return pairs


def _uniform_in_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(float(lo), float(hi)))


@dataclass(frozen=True)
class SegmentCommand:
    start_step: int
    vx: float
    vy: float
    yaw: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "start_step": int(self.start_step),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "yaw": float(self.yaw),
        }


@dataclass(frozen=True)
class DiagnosticRow:
    episode_id: str
    regime: str
    geometry_seed: int
    physical_column: int
    terrain_family: str
    terrain_level: int
    speed_bin: int
    command_vx: float
    command_vy: float
    command_yaw: float
    segment_commands: tuple[SegmentCommand, ...] = field(default_factory=tuple)
    command_design: str | None = None
    command_seed: int | None = None
    pair_index: int | None = None
    schedule_index: int | None = None
    vx_sign: float | None = None

    def to_fingerprint_dict(self) -> dict[str, Any]:
        # Fingerprint identity is the bank plan only.  Display aliases such as
        # terrain_column must NOT enter this dict or existing shards fail merge.
        return {
            "episode_id": self.episode_id,
            "regime": self.regime,
            "geometry_seed": self.geometry_seed,
            "physical_column": self.physical_column,
            "terrain_family": self.terrain_family,
            "terrain_level": self.terrain_level,
            "speed_bin": self.speed_bin,
            "command_vx": self.command_vx,
            "command_vy": self.command_vy,
            "command_yaw": self.command_yaw,
            "segment_commands": [s.to_dict() for s in self.segment_commands],
            "command_design": self.command_design,
            "command_seed": self.command_seed,
            "pair_index": self.pair_index,
            "schedule_index": self.schedule_index,
            "vx_sign": self.vx_sign,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.to_fingerprint_dict()
        # Contract alias for training terminology (not part of bank fingerprint).
        d["terrain_column"] = self.physical_column
        return d


def build_bank(config: Mapping[str, Any], *, geometry_seed: int | None = None) -> list[DiagnosticRow]:
    """Expand the full diagnostic bank, optionally filtered to one geometry seed."""
    geo = config["geometry"]
    cmd = config["commands"]
    seeds = [int(s) for s in geo["seeds"]]
    if geometry_seed is not None:
        geometry_seed = int(geometry_seed)
        if geometry_seed not in seeds:
            raise ValueError(f"geometry_seed {geometry_seed} not in config.geometry.seeds")
        seeds = [geometry_seed]

    columns = list(range(int(geo["num_cols"])))
    if columns != list(range(10)):
        raise ValueError("diagnostic bank requires exactly 10 physical columns")
    level = int(geo["terrain_level"])
    speed_bin = int(geo["speed_bin"])
    if level != 0 or speed_bin != 0:
        raise ValueError("diagnostic bank is fixed to terrain_level=0 and speed_bin=0")

    vx_list = [float(v) for v in cmd["controlled_vx"]]
    expected_vx = [-0.45, -0.35, -0.25, 0.25, 0.35, 0.45]
    if vx_list != expected_vx:
        raise ValueError(f"controlled_vx must be exactly {expected_vx}")

    vy_b = [float(v) for v in cmd["regime_b_vy"]]
    yaw_c = [float(v) for v in cmd["regime_c_yaw"]]
    joint_pairs = _latin_hypercube_pairs(int(cmd["joint_pairs"]), int(cmd["joint_seed"]))
    if len(joint_pairs) != 12:
        raise ValueError("joint_pairs must expand to 12")

    rows: list[DiagnosticRow] = []
    for gseed in seeds:
        for col in columns:
            family = family_for_column(col)
            # Regime A
            for vx in vx_list:
                rows.append(
                    DiagnosticRow(
                        episode_id=f"g{gseed}_c{col}_{REGIME_A}_vx{vx:+.2f}",
                        regime=REGIME_A,
                        geometry_seed=gseed,
                        physical_column=col,
                        terrain_family=family,
                        terrain_level=level,
                        speed_bin=speed_bin,
                        command_vx=vx,
                        command_vy=0.0,
                        command_yaw=0.0,
                        segment_commands=(SegmentCommand(0, vx, 0.0, 0.0),),
                    )
                )
            # Regime B
            for vx in vx_list:
                for vy in vy_b:
                    rows.append(
                        DiagnosticRow(
                            episode_id=f"g{gseed}_c{col}_{REGIME_B}_vx{vx:+.2f}_vy{vy:+.2f}",
                            regime=REGIME_B,
                            geometry_seed=gseed,
                            physical_column=col,
                            terrain_family=family,
                            terrain_level=level,
                            speed_bin=speed_bin,
                            command_vx=vx,
                            command_vy=vy,
                            command_yaw=0.0,
                            segment_commands=(SegmentCommand(0, vx, vy, 0.0),),
                        )
                    )
            # Regime C
            for vx in vx_list:
                for yaw in yaw_c:
                    rows.append(
                        DiagnosticRow(
                            episode_id=f"g{gseed}_c{col}_{REGIME_C}_vx{vx:+.2f}_yaw{yaw:+.2f}",
                            regime=REGIME_C,
                            geometry_seed=gseed,
                            physical_column=col,
                            terrain_family=family,
                            terrain_level=level,
                            speed_bin=speed_bin,
                            command_vx=vx,
                            command_vy=0.0,
                            command_yaw=yaw,
                            segment_commands=(SegmentCommand(0, vx, 0.0, yaw),),
                        )
                    )
            # Regime D
            for vx in vx_list:
                for pair_i, (vy, yaw) in enumerate(joint_pairs):
                    rows.append(
                        DiagnosticRow(
                            episode_id=(
                                f"g{gseed}_c{col}_{REGIME_D}_vx{vx:+.2f}_p{pair_i:02d}"
                            ),
                            regime=REGIME_D,
                            geometry_seed=gseed,
                            physical_column=col,
                            terrain_family=family,
                            terrain_level=level,
                            speed_bin=speed_bin,
                            command_vx=vx,
                            command_vy=vy,
                            command_yaw=yaw,
                            segment_commands=(SegmentCommand(0, vx, vy, yaw),),
                            command_design="latin_hypercube",
                            command_seed=int(cmd["joint_seed"]),
                            pair_index=pair_i,
                        )
                    )
            # Regime E
            e_seed = int(cmd["training_faithful_seed"])
            e_n = int(cmd["training_faithful_episodes_per_column_sign"])
            seg_steps = int(cmd["training_faithful_segment_steps"])
            vx_lo, vx_hi = [float(x) for x in cmd["training_faithful_vx_bin"]]
            vy_lo, vy_hi = [float(x) for x in cmd["training_faithful_vy_range"]]
            yaw_lo, yaw_hi = [float(x) for x in cmd["training_faithful_yaw_range"]]
            for sign in [float(s) for s in cmd["training_faithful_vx_signs"]]:
                for sched_i in range(e_n):
                    # SeedSequence requires non-negative ints; encode sign as 0/1.
                    rng = np.random.default_rng(
                        [
                            e_seed,
                            gseed,
                            col,
                            0 if float(sign) < 0.0 else 1,
                            sched_i,
                            17,
                        ]
                    )
                    mag0 = _uniform_in_range(rng, vx_lo, vx_hi)
                    mag1 = _uniform_in_range(rng, vx_lo, vx_hi)
                    vy0 = _uniform_in_range(rng, vy_lo, vy_hi)
                    yaw0 = _uniform_in_range(rng, yaw_lo, yaw_hi)
                    vy1 = _uniform_in_range(rng, vy_lo, vy_hi)
                    yaw1 = _uniform_in_range(rng, yaw_lo, yaw_hi)
                    vx0 = float(sign) * mag0
                    vx1 = float(sign) * mag1
                    rows.append(
                        DiagnosticRow(
                            episode_id=(
                                f"g{gseed}_c{col}_{REGIME_E}_sign{sign:+.0f}_s{sched_i:02d}"
                            ),
                            regime=REGIME_E,
                            geometry_seed=gseed,
                            physical_column=col,
                            terrain_family=family,
                            terrain_level=level,
                            speed_bin=speed_bin,
                            command_vx=vx0,
                            command_vy=vy0,
                            command_yaw=yaw0,
                            segment_commands=(
                                SegmentCommand(0, vx0, vy0, yaw0),
                                SegmentCommand(seg_steps, vx1, vy1, yaw1),
                            ),
                            command_design="training_faithful_two_segment",
                            command_seed=e_seed,
                            schedule_index=sched_i,
                            vx_sign=float(sign),
                        )
                    )

    _validate_bank_counts(rows, n_geometry=len(seeds))
    return rows


def _validate_bank_counts(rows: Sequence[DiagnosticRow], *, n_geometry: int) -> None:
    by_regime: dict[str, int] = {}
    for row in rows:
        by_regime[row.regime] = by_regime.get(row.regime, 0) + 1
    # Per geometry seed expectations
    expected_per_seed = {
        REGIME_A: 60,
        REGIME_B: 240,
        REGIME_C: 360,
        REGIME_D: 720,
        REGIME_E: 240,
    }
    for regime, per in expected_per_seed.items():
        got = by_regime.get(regime, 0)
        if got != per * n_geometry:
            raise ValueError(
                f"bank count for {regime}: expected {per * n_geometry}, got {got}"
            )
    total = sum(by_regime.values())
    if total != 1620 * n_geometry:
        raise ValueError(f"bank total expected {1620 * n_geometry}, got {total}")
    columns = {r.physical_column for r in rows}
    if columns != set(range(10)):
        raise ValueError(f"bank must cover columns 0..9, got {sorted(columns)}")
    for row in rows:
        if row.terrain_level != 0 or row.speed_bin != 0:
            raise ValueError("bank row escaped level0/speed_bin0")
        if row.regime == REGIME_E:
            if len(row.segment_commands) != 2:
                raise ValueError("regime E must have two segments")
            if row.vx_sign is None:
                raise ValueError("regime E requires vx_sign")
            s0, s1 = row.segment_commands
            if np.sign(s0.vx) != np.sign(row.vx_sign) or np.sign(s1.vx) != np.sign(row.vx_sign):
                raise ValueError("regime E vx sign must stay fixed across segments")


def expected_episode_count(config: Mapping[str, Any], *, geometry_seed: int | None = None) -> int:
    n_geo = 1 if geometry_seed is not None else len(config["geometry"]["seeds"])
    return 1620 * n_geo


def artifact_dir(run_dir: Path, iteration: int, config: Mapping[str, Any]) -> Path:
    rel = str(config.get("artifact", {}).get("relative_root", "frontier_diagnostic"))
    return Path(run_dir) / rel / f"model_{int(iteration)}"


def _load_existing_artifact_identity(out_dir: Path) -> dict[str, Any] | None:
    """Return existing meta identity if any artifact is already present."""
    bank_path = out_dir / "bank.json"
    fp_path = out_dir / "bank_fingerprint.txt"
    if not bank_path.is_file() and not fp_path.is_file():
        return None
    identity: dict[str, Any] = {}
    if bank_path.is_file():
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
        meta = payload.get("meta") or {}
        identity.update(
            {
                "schema_version": meta.get("schema_version"),
                "config_fingerprint": meta.get("config_fingerprint"),
                "bank_fingerprint": meta.get("bank_fingerprint"),
                "checkpoint_sha256": meta.get("checkpoint_sha256"),
            }
        )
    if fp_path.is_file():
        identity["bank_fingerprint"] = fp_path.read_text(encoding="utf-8").strip()
    return identity


def write_bank_artifacts(
    out_dir: Path,
    config: Mapping[str, Any],
    rows: Sequence[DiagnosticRow],
    *,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bank = [r.to_dict() for r in rows]
    fp = bank_fingerprint(rows)
    cfg_fp = config_fingerprint(config)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "diagnostic_only",
        "eligible_for_checkpoint_selection": False,
        "config_fingerprint": cfg_fp,
        "bank_fingerprint": fp,
        "episode_count": len(rows),
        "checkpoint_sha256": checkpoint_sha256,
        "joint_pairs": _latin_hypercube_pairs(
            int(config["commands"]["joint_pairs"]),
            int(config["commands"]["joint_seed"]),
        ),
    }
    existing = _load_existing_artifact_identity(out_dir)
    if existing is not None:
        conflicts = []
        if existing.get("schema_version") not in (None, SCHEMA_VERSION):
            conflicts.append(
                f"schema_version {existing.get('schema_version')!r} != {SCHEMA_VERSION!r}"
            )
        if existing.get("config_fingerprint") not in (None, cfg_fp):
            conflicts.append("config_fingerprint mismatch")
        if existing.get("bank_fingerprint") not in (None, fp):
            conflicts.append("bank_fingerprint mismatch")
        if (
            checkpoint_sha256 is not None
            and existing.get("checkpoint_sha256") not in (None, checkpoint_sha256)
        ):
            conflicts.append("checkpoint_sha256 mismatch")
        if conflicts:
            raise ValueError(
                f"refuse to overwrite conflicting artifacts under {out_dir}: "
                + "; ".join(conflicts)
                + ". Use a new artifact directory / protocol revision."
            )
        # Identical identity: idempotent rewrite of the same content is allowed.
    (out_dir / "bank.json").write_text(
        json.dumps({"meta": meta, "episodes": bank}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "bank_fingerprint.txt").write_text(fp + "\n", encoding="utf-8")
    resolved = dict(config)
    resolved["_resolved"] = meta
    (out_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    return meta


# Bank identity fields that every measured episode must reproduce exactly.
BANK_IDENTITY_FIELDS = (
    "episode_id",
    "regime",
    "geometry_seed",
    "physical_column",
    "terrain_family",
    "terrain_level",
    "speed_bin",
    "command_vx",
    "command_vy",
    "command_yaw",
    "segment_commands",
    "command_design",
    "command_seed",
    "pair_index",
    "schedule_index",
    "vx_sign",
)


def bank_row_identity(row: DiagnosticRow | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, DiagnosticRow):
        raw = row.to_fingerprint_dict()
    else:
        raw = dict(row)
    out = {}
    for key in BANK_IDENTITY_FIELDS:
        val = raw.get(key)
        if key == "segment_commands" and val is not None:
            # Normalize SegmentCommand / dict list to plain dicts.
            norm = []
            for seg in val:
                if hasattr(seg, "to_dict"):
                    norm.append(seg.to_dict())
                else:
                    norm.append(
                        {
                            "start_step": int(seg["start_step"]),
                            "vx": float(seg["vx"]),
                            "vy": float(seg["vy"]),
                            "yaw": float(seg["yaw"]),
                        }
                    )
            out[key] = norm
        else:
            out[key] = val
    return out


def assert_episode_matches_bank_row(
    rec: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    got = bank_row_identity(rec)
    exp = bank_row_identity(expected)
    for key in BANK_IDENTITY_FIELDS:
        g, e = got.get(key), exp.get(key)
        if key in ("command_vx", "command_vy", "command_yaw", "vx_sign") and g is not None and e is not None:
            if abs(float(g) - float(e)) > 1e-9:
                raise ValueError(
                    f"episode {rec.get('episode_id')}: field {key}={g!r} != bank {e!r}"
                )
            continue
        if key == "segment_commands":
            if json.dumps(g, sort_keys=True) != json.dumps(e, sort_keys=True):
                raise ValueError(
                    f"episode {rec.get('episode_id')}: segment_commands mismatch bank row"
                )
            continue
        if g != e:
            raise ValueError(
                f"episode {rec.get('episode_id')}: field {key}={g!r} != bank {e!r}"
            )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def episode_success_from_record(
    rec: Mapping[str, Any],
    *,
    linear_threshold: float,
    yaw_threshold: float,
) -> bool:
    # Prefer explicit observed_timeout_event (v2); fall back to timed_out.
    if "observed_timeout_event" in rec:
        timed_out = bool(rec.get("observed_timeout_event"))
    else:
        timed_out = bool(rec.get("timed_out"))
    return frontier_success(
        timed_out=timed_out,
        mean_linear_error=float(rec["mean_linear_error"]),
        mean_yaw_error=float(rec["mean_yaw_error"]),
        linear_threshold=linear_threshold,
        yaw_threshold=yaw_threshold,
    )


def failure_decomposition(
    records: Sequence[Mapping[str, Any]],
    *,
    linear_threshold: float = 0.35,
    yaw_threshold: float = 0.40,
) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "survival_fail": 0.0,
            "linear_fail_only": 0.0,
            "yaw_fail_only": 0.0,
            "both_tracking_fail": 0.0,
            "success": 0.0,
            "linear_threshold": float(linear_threshold),
            "yaw_threshold": float(yaw_threshold),
        }
    survival_fail = linear_only = yaw_only = both = success = 0
    for rec in records:
        timed_out = bool(rec.get("timed_out"))
        lin_ok = float(rec["mean_linear_error"]) <= float(linear_threshold)
        yaw_ok = float(rec["mean_yaw_error"]) <= float(yaw_threshold)
        if not timed_out:
            survival_fail += 1
        elif lin_ok and yaw_ok:
            success += 1
        elif (not lin_ok) and yaw_ok:
            linear_only += 1
        elif lin_ok and (not yaw_ok):
            yaw_only += 1
        else:
            both += 1
    return {
        "n": n,
        "survival_fail": survival_fail / n,
        "linear_fail_only": linear_only / n,
        "yaw_fail_only": yaw_only / n,
        "both_tracking_fail": both / n,
        "success": success / n,
        "linear_threshold": float(linear_threshold),
        "yaw_threshold": float(yaw_threshold),
    }


def _rate(records: Sequence[Mapping[str, Any]], *, lin_thr: float, yaw_thr: float) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0, "success_rate": float("nan"), "ci95": [float("nan"), float("nan")]}
    k = sum(
        1
        for r in records
        if episode_success_from_record(r, linear_threshold=lin_thr, yaw_threshold=yaw_thr)
    )
    lo, hi = wilson_interval(k, n)
    return {"n": n, "successes": k, "success_rate": k / n, "ci95": [lo, hi]}


def aggregate_macro_micro(
    records: Sequence[Mapping[str, Any]],
    *,
    linear_threshold: float = 0.35,
    yaw_threshold: float = 0.40,
) -> dict[str, Any]:
    by_col: dict[int, list[Mapping[str, Any]]] = {}
    by_fam: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        col = int(rec["physical_column"])
        fam = str(rec["terrain_family"])
        by_col.setdefault(col, []).append(rec)
        by_fam.setdefault(fam, []).append(rec)

    micro_rates = []
    micro = {}
    for col in sorted(by_col):
        r = _rate(by_col[col], lin_thr=linear_threshold, yaw_thr=yaw_threshold)
        micro[str(col)] = r
        if r["n"]:
            micro_rates.append(r["success_rate"])

    # Semantic family macro: average column rates within family, then average families.
    space = V4FrontierTaskSpace()
    family_rates = []
    family = {}
    for fam_i, fam_name in enumerate(space.TERRAIN_FAMILIES):
        cols = space.FAMILY_COLUMNS[fam_i]
        col_rates = []
        for col in cols:
            r = micro.get(str(col))
            if r and r["n"]:
                col_rates.append(r["success_rate"])
        if col_rates:
            fam_rate = float(np.mean(col_rates))
            family_rates.append(fam_rate)
            family[fam_name] = {
                "columns": list(cols),
                "column_mean_success_rate": fam_rate,
                "n_columns_with_data": len(col_rates),
            }
        else:
            family[fam_name] = {
                "columns": list(cols),
                "column_mean_success_rate": float("nan"),
                "n_columns_with_data": 0,
            }

    return {
        "physical_column_micro": {
            "per_column": micro,
            "mean_success_rate": float(np.mean(micro_rates)) if micro_rates else float("nan"),
        },
        "semantic_family_macro": {
            "per_family": family,
            "mean_success_rate": float(np.mean(family_rates)) if family_rates else float("nan"),
        },
        "overall": _rate(records, lin_thr=linear_threshold, yaw_thr=yaw_threshold),
    }


def threshold_sweep(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    sweep = config["threshold_sweep"]
    lin_grid = [float(x) for x in sweep["linear_error_threshold"]]
    yaw_grid = [float(x) for x in sweep["yaw_error_threshold"]]
    mastery_grid = [float(x) for x in sweep["mastery_threshold"]]
    window = int(config["frontier_success"]["window_size"])

    cells = []
    for lin_thr in lin_grid:
        for yaw_thr in yaw_grid:
            for mastery in mastery_grid:
                agg = aggregate_macro_micro(
                    records, linear_threshold=lin_thr, yaw_threshold=yaw_thr
                )
                fail = failure_decomposition(
                    records, linear_threshold=lin_thr, yaw_threshold=yaw_thr
                )
                # Empirical success treated as Bernoulli p; mastery vote approx.
                p = agg["semantic_family_macro"]["mean_success_rate"]
                req = required_success_count(window, mastery)
                if agg["overall"]["n"]:
                    # Independent Binomial approximation for one window vote.
                    # P(X >= req) with X~Bin(window, p)
                    one_vote = (
                        float(
                            sum(
                                math.comb(window, k)
                                * (p**k)
                                * ((1 - p) ** (window - k))
                                for k in range(req, window + 1)
                            )
                        )
                        if (p == p and 0.0 <= p <= 1.0)
                        else float("nan")
                    )
                else:
                    one_vote = float("nan")
                two_vote = one_vote * one_vote if one_vote == one_vote else float("nan")
                cells.append(
                    {
                        "linear_error_threshold": lin_thr,
                        "yaw_error_threshold": yaw_thr,
                        "mastery_threshold": mastery,
                        "required_success_count": req,
                        "required_success_semantics": "beta11_posterior_mean",
                        "overall": agg["overall"],
                        "semantic_family_macro_success": agg["semantic_family_macro"][
                            "mean_success_rate"
                        ],
                        "physical_column_micro_success": agg["physical_column_micro"][
                            "mean_success_rate"
                        ],
                        "failure_decomposition": fail,
                        "approx_one_vote_probability": one_vote,
                        "approx_two_vote_probability": two_vote,
                        "note": (
                            "two_vote uses independence approximation; "
                            "not a guarantee of unlock"
                        ),
                    }
                )
    return {
        "grid": cells,
        "original_thresholds": {
            "linear_error_threshold": 0.35,
            "yaw_error_threshold": 0.40,
            "mastery_threshold": 0.80,
            "required_success_count": required_success_count(window, 0.80),
            "required_success_semantics": "beta11_posterior_mean",
        },
    }


def group_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        out.setdefault(str(rec.get(key)), []).append(rec)
    return out


def _mag_bin(val: float, edges: Sequence[float]) -> str:
    a = abs(float(val))
    for i in range(len(edges) - 1):
        if edges[i] <= a < edges[i + 1] or (i == len(edges) - 2 and a <= edges[i + 1]):
            return f"[{edges[i]},{edges[i+1]}]"
    return "other"


def _vx_magnitude(rec: Mapping[str, Any]) -> float:
    return abs(float(rec.get("command_vx", 0.0)))


def _vx_sign_label(rec: Mapping[str, Any]) -> str:
    if rec.get("vx_sign") is not None:
        s = float(rec["vx_sign"])
    else:
        s = float(np.sign(float(rec.get("command_vx", 0.0))))
    if s < 0:
        return "-1"
    if s > 0:
        return "+1"
    return "0"


def summarize(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    by_regime = group_by(records, "regime")
    regime_summary = {}
    for regime in ALL_REGIMES:
        recs = by_regime.get(regime, [])
        regime_summary[regime] = {
            **aggregate_macro_micro(recs),
            "failure_decomposition": failure_decomposition(recs),
        }

    yaw_bins: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        b = _mag_bin(rec.get("command_yaw", 0.0), [0.0, 0.25, 0.5, 0.75, 1.01])
        yaw_bins.setdefault(b, []).append(rec)
    yaw_bin_summary = {
        b: {
            **aggregate_macro_micro(rs)["overall"],
            "failure_decomposition": failure_decomposition(rs),
        }
        for b, rs in sorted(yaw_bins.items())
    }

    vy_bins: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        b = _mag_bin(rec.get("command_vy", 0.0), [0.0, 0.25, 0.5, 0.75, 1.01])
        vy_bins.setdefault(b, []).append(rec)
    vy_bin_summary = {
        b: {
            **aggregate_macro_micro(rs)["overall"],
            "failure_decomposition": failure_decomposition(rs),
        }
        for b, rs in sorted(vy_bins.items())
    }

    # family × |vx| magnitude (controlled anchors 0.25/0.35/0.45)
    fam_vx: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for rec in records:
        fam = str(rec.get("terrain_family", "unknown"))
        mag = f"{_vx_magnitude(rec):.2f}"
        fam_vx.setdefault(fam, {}).setdefault(mag, []).append(rec)
    family_x_vx_magnitude = {
        fam: {
            mag: {
                **aggregate_macro_micro(rs)["overall"],
                "failure_decomposition": failure_decomposition(rs),
            }
            for mag, rs in sorted(mags.items())
        }
        for fam, mags in sorted(fam_vx.items())
    }

    # family × regime (macro already has per-family; add success + fail decomp)
    family_x_regime: dict[str, dict[str, Any]] = {}
    for regime, recs in by_regime.items():
        by_fam = group_by(recs, "terrain_family")
        for fam, frs in by_fam.items():
            family_x_regime.setdefault(fam, {})[regime] = {
                **aggregate_macro_micro(frs)["overall"],
                "failure_decomposition": failure_decomposition(frs),
            }

    # vx_sign slice
    by_sign: dict[str, list[Mapping[str, Any]]] = {}
    for rec in records:
        by_sign.setdefault(_vx_sign_label(rec), []).append(rec)
    vx_sign_summary = {
        sign: {
            **aggregate_macro_micro(rs)["overall"],
            "failure_decomposition": failure_decomposition(rs),
        }
        for sign, rs in sorted(by_sign.items())
    }

    return {
        "purpose": "diagnostic_only",
        "eligible_for_checkpoint_selection": False,
        "n_episodes": len(records),
        "by_regime": regime_summary,
        "overall": {
            **aggregate_macro_micro(records),
            "failure_decomposition": failure_decomposition(records),
        },
        "by_geometry_seed": {
            k: {
                **aggregate_macro_micro(v)["overall"],
                "failure_decomposition": failure_decomposition(v),
            }
            for k, v in sorted(group_by(records, "geometry_seed").items())
        },
        "by_abs_yaw_bin": yaw_bin_summary,
        "by_abs_vy_bin": vy_bin_summary,
        "by_family_x_vx_magnitude": family_x_vx_magnitude,
        "by_family_x_regime": family_x_regime,
        "by_vx_sign": vx_sign_summary,
        "frontier_success_is_training_semantics": True,
        "spnte_is_secondary_diagnostic_only": True,
    }


def recommend_v61(summary: Mapping[str, Any], sweep: Mapping[str, Any]) -> dict[str, Any]:
    """Data-driven single-variable recommendation for the next V6.1 experiment."""
    by_regime = summary.get("by_regime", {})
    a = by_regime.get(REGIME_A, {}).get("semantic_family_macro", {}).get("mean_success_rate")
    b = by_regime.get(REGIME_B, {}).get("semantic_family_macro", {}).get("mean_success_rate")
    c = by_regime.get(REGIME_C, {}).get("semantic_family_macro", {}).get("mean_success_rate")
    d = by_regime.get(REGIME_D, {}).get("semantic_family_macro", {}).get("mean_success_rate")
    e = by_regime.get(REGIME_E, {}).get("semantic_family_macro", {}).get("mean_success_rate")
    fail = summary.get("overall", {}).get("failure_decomposition", {})
    a_fail = by_regime.get(REGIME_A, {}).get("failure_decomposition", {})
    stairs_a = (
        by_regime.get(REGIME_A, {})
        .get("semantic_family_macro", {})
        .get("per_family", {})
        .get("stairs_up", {})
        .get("column_mean_success_rate")
    )

    choice = "keep_thresholds"
    rationale = []
    if a is not None and a == a and a >= 0.80:
        rationale.append(f"A_baseline macro success {a:.3f} already clears mastery 0.80")
    elif a is not None and a == a:
        rationale.append(f"A_baseline macro success {a:.3f} is below mastery 0.80")
    if stairs_a is not None and stairs_a == stairs_a and stairs_a < 0.30:
        rationale.append(
            f"stairs_up under A is only {stairs_a:.3f}: terrain/linear bottleneck, not yaw"
        )

    yaw_dom = float(fail.get("yaw_fail_only", 0.0)) + 0.5 * float(fail.get("both_tracking_fail", 0.0))
    lin_dom = float(fail.get("linear_fail_only", 0.0)) + 0.5 * float(fail.get("both_tracking_fail", 0.0))
    a_lin = float(a_fail.get("linear_fail_only", 0.0))
    surv = float(fail.get("survival_fail", 0.0))

    # Prefer yaw calibration only when A is already strong and pure yaw hurts.
    if (
        a is not None
        and a == a
        and a >= 0.70
        and c is not None
        and c == c
        and (a - c) >= 0.25
        and yaw_dom >= lin_dom
    ):
        choice = "calibrate_yaw_threshold_only"
        rationale.append(
            f"yaw-dominated failure (yawish={yaw_dom:.3f}, linearish={lin_dom:.3f}); "
            f"A-C gap={a - c:.3f}"
        )
    elif e is not None and e == e and e < 0.55 and a is not None and a == a and a >= 0.70:
        choice = "stage_vy_yaw_nuisance"
        rationale.append(
            f"training-faithful E success {e:.3f} stays low while baseline A is usable"
        )
    elif a is not None and a == a and a < 0.70 and a_lin >= 0.20:
        choice = "keep_thresholds"
        rationale.append(
            f"baseline linear failures dominate A (linear_fail_only={a_lin:.3f}); "
            "loosening yaw or staging nuisance will not unlock stairs-first frontier"
        )
    elif surv >= 0.40:
        choice = "keep_thresholds"
        rationale.append(f"survival failures dominate ({surv:.3f}); threshold tweaks won't unlock")
    else:
        rationale.append("default: keep original thresholds until a clearer single lever appears")

    # Smallest yaw thr at fixed lin=0.35, mastery=0.80 that lifts E macro to ≥0.55
    # without requiring A to collapse (report only; does not change recommendation alone).
    suggested_yaw = 0.40
    e_by_yaw: list[tuple[float, float]] = []
    for cell in sweep.get("grid", []):
        if (
            abs(cell["linear_error_threshold"] - 0.35) < 1e-9
            and abs(cell["mastery_threshold"] - 0.80) < 1e-9
        ):
            e_by_yaw.append(
                (float(cell["yaw_error_threshold"]), float(cell.get("semantic_family_macro_success", float("nan"))))
            )
    e_by_yaw.sort()
    for yaw_thr, e_macro in e_by_yaw:
        if e_macro == e_macro and e_macro >= 0.55:
            suggested_yaw = yaw_thr
            break

    return {
        "recommendation": choice,
        "options": [
            "keep_thresholds",
            "calibrate_yaw_threshold_only",
            "stage_vy_yaw_nuisance",
        ],
        "rationale": rationale,
        "suggested_yaw_threshold_if_calibrating": suggested_yaw,
        "empirical_A_macro": a,
        "empirical_B_macro": b,
        "empirical_C_macro": c,
        "empirical_D_macro": d,
        "empirical_E_macro": e,
        "empirical_A_stairs_up": stairs_a,
        "delta_B_minus_A": (b - a) if (a == a and b == b and a is not None and b is not None) else None,
        "delta_C_minus_A": (c - a) if (a == a and c == c and a is not None and c is not None) else None,
        "delta_E_minus_D": (e - d) if (e == e and d == d and e is not None and d is not None) else None,
        "failure_decomposition": fail,
    }


def render_report_md(
    summary: Mapping[str, Any],
    sweep: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    bank_fp: str,
    config_fp: str,
) -> str:
    by_reg = summary.get("by_regime", {})

    def _macro(regime: str) -> Any:
        return by_reg.get(regime, {}).get("semantic_family_macro", {}).get("mean_success_rate")

    def _micro(regime: str) -> Any:
        return by_reg.get(regime, {}).get("physical_column_micro", {}).get("mean_success_rate")

    def _n(regime: str) -> int:
        return int(by_reg.get(regime, {}).get("overall", {}).get("n", 0))

    a, b, c, d, e = (
        _macro(REGIME_A),
        _macro(REGIME_B),
        _macro(REGIME_C),
        _macro(REGIME_D),
        _macro(REGIME_E),
    )
    delta_ba = (b - a) if (a is not None and b is not None and a == a and b == b) else float("nan")
    delta_ca = (c - a) if (a is not None and c is not None and a == a and c == c) else float("nan")
    delta_ed = (e - d) if (e is not None and d is not None and e == e and d == d) else float("nan")

    lines = [
        "# V6 frontier diagnostic report",
        "",
        "Purpose: **diagnostic_only** (not eligible for checkpoint selection).",
        "",
        f"- checkpoint_sha256: `{checkpoint_sha256}`",
        f"- bank_fingerprint: `{bank_fp}`",
        f"- config_fingerprint: `{config_fp}`",
        f"- n_episodes: {summary.get('n_episodes')}",
        f"- mastery count semantics: Beta(1,1) posterior mean "
        f"`(k+1)/(n+2)`; window 32 → 0.80 needs **27/32**",
        "",
        "## Regime success (semantic-family macro)",
        "",
        "| Regime | Macro success | Micro success | N |",
        "|---|---:|---:|---:|",
    ]
    for regime in ALL_REGIMES:
        lines.append(
            f"| `{regime}` | {_fmt(_macro(regime))} | {_fmt(_micro(regime))} | {_n(regime)} |"
        )

    fail = summary.get("overall", {}).get("failure_decomposition", {})
    a_fail = by_reg.get(REGIME_A, {}).get("failure_decomposition", {})
    lines += [
        "",
        "## Failure decomposition (overall @ lin≤0.35, yaw≤0.40)",
        "",
        f"- survival_fail: {_fmt(fail.get('survival_fail'))}",
        f"- linear_fail_only: {_fmt(fail.get('linear_fail_only'))}",
        f"- yaw_fail_only: {_fmt(fail.get('yaw_fail_only'))}",
        f"- both_tracking_fail: {_fmt(fail.get('both_tracking_fail'))}",
        f"- success: {_fmt(fail.get('success'))}",
        "",
        "## Answers to diagnostic questions",
        "",
        "### Q1 — Is A_baseline solved at level0 / speed_bin0?",
        "",
        f"- Semantic-family macro success **{_fmt(a)}** "
        f"(micro {_fmt(_micro(REGIME_A))}, n={_n(REGIME_A)}).",
        f"- Mastery bar is 0.80; A is **{'above' if (a is not None and a == a and a >= 0.80) else 'below'}** mastery.",
        f"- A failure mix: survival={_fmt(a_fail.get('survival_fail'))}, "
        f"linear_only={_fmt(a_fail.get('linear_fail_only'))}, "
        f"yaw_only={_fmt(a_fail.get('yaw_fail_only'))}, "
        f"both={_fmt(a_fail.get('both_tracking_fail'))}.",
        "",
        "### Q2 — Pure vy impact (B − A)",
        "",
        f"- B macro = {_fmt(b)} (n={_n(REGIME_B)}).",
        f"- **Δ(B−A) = {_fmt(delta_ba)}** absolute success points.",
        f"- B failure mix: "
        f"survival={_fmt(by_reg.get(REGIME_B, {}).get('failure_decomposition', {}).get('survival_fail'))}, "
        f"linear_only={_fmt(by_reg.get(REGIME_B, {}).get('failure_decomposition', {}).get('linear_fail_only'))}, "
        f"yaw_only={_fmt(by_reg.get(REGIME_B, {}).get('failure_decomposition', {}).get('yaw_fail_only'))}, "
        f"both={_fmt(by_reg.get(REGIME_B, {}).get('failure_decomposition', {}).get('both_tracking_fail'))}.",
        "",
        "### Q3 — Pure yaw impact (C − A)",
        "",
        f"- C macro = {_fmt(c)} (n={_n(REGIME_C)}).",
        f"- **Δ(C−A) = {_fmt(delta_ca)}** absolute success points.",
        f"- C failure mix: "
        f"survival={_fmt(by_reg.get(REGIME_C, {}).get('failure_decomposition', {}).get('survival_fail'))}, "
        f"linear_only={_fmt(by_reg.get(REGIME_C, {}).get('failure_decomposition', {}).get('linear_fail_only'))}, "
        f"yaw_only={_fmt(by_reg.get(REGIME_C, {}).get('failure_decomposition', {}).get('yaw_fail_only'))}, "
        f"both={_fmt(by_reg.get(REGIME_C, {}).get('failure_decomposition', {}).get('both_tracking_fail'))}.",
        "",
        "### Q4 — Failure decomposition (overall)",
        "",
        f"- survival: {_fmt(fail.get('survival_fail'))}",
        f"- linear_only: {_fmt(fail.get('linear_fail_only'))}",
        f"- yaw_only: {_fmt(fail.get('yaw_fail_only'))}",
        f"- both tracking: {_fmt(fail.get('both_tracking_fail'))}",
        f"- success: {_fmt(fail.get('success'))}",
        "",
        "### Q5 — Where does yaw failure concentrate?",
        "",
        "| |yaw| bin | success_rate | n | yaw_fail_only | linear_fail_only |",
        "|---|---:|---:|---:|---:|",
    ]
    for bin_name, r in sorted((summary.get("by_abs_yaw_bin") or {}).items()):
        fd = r.get("failure_decomposition") or {}
        lines.append(
            f"| {bin_name} | {_fmt(r.get('success_rate'))} | {r.get('n')} | "
            f"{_fmt(fd.get('yaw_fail_only'))} | {_fmt(fd.get('linear_fail_only'))} |"
        )
    lines += [
        "",
        "### Q6 — Controlled D vs training-faithful E",
        "",
        f"- D macro = {_fmt(d)} (n={_n(REGIME_D)}); E macro = {_fmt(e)} (n={_n(REGIME_E)}).",
        f"- **Δ(E−D) = {_fmt(delta_ed)}**.",
        "",
        "### Q7 — Does E reproduce the training ring ~35–50% band?",
        "",
        f"- E macro = **{_fmt(e)}**.",
        (
            f"- {'YES' if (e is not None and e == e and 0.35 <= e <= 0.50) else 'NO'}: "
            "training rings were reported near 35–50%; offline E "
            f"{'falls inside' if (e is not None and e == e and 0.35 <= e <= 0.50) else 'does not fall inside'} "
            "that band."
        ),
        "",
        "### Q8 — Which yaw threshold helps without free-riding baseline?",
        "",
        "| yaw_thr (lin=0.35, mastery=0.80) | overall success | macro | "
        "yaw_fail_only | linear_fail_only | P(one vote) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    yaw_rows = []
    for cell in sweep.get("grid", []):
        if (
            abs(float(cell["linear_error_threshold"]) - 0.35) < 1e-9
            and abs(float(cell["mastery_threshold"]) - 0.80) < 1e-9
        ):
            yaw_rows.append(cell)
    yaw_rows.sort(key=lambda x: float(x["yaw_error_threshold"]))
    for cell in yaw_rows:
        fd = cell.get("failure_decomposition") or {}
        lines.append(
            f"| {cell['yaw_error_threshold']:.2f} | "
            f"{_fmt((cell.get('overall') or {}).get('success_rate'))} | "
            f"{_fmt(cell.get('semantic_family_macro_success'))} | "
            f"{_fmt(fd.get('yaw_fail_only'))} | "
            f"{_fmt(fd.get('linear_fail_only'))} | "
            f"{_fmt(cell.get('approx_one_vote_probability'))} |"
        )
    lines += [
        "",
        f"- Suggested yaw thr if calibrating: "
        f"**{recommendation.get('suggested_yaw_threshold_if_calibrating')}** "
        "(smallest thr in the lin=0.35 grid with macro≥0.55, else highest).",
        "",
        "### Q9 — Threshold tweak vs staged vy/yaw nuisance?",
        "",
        f"- Recommendation: **`{recommendation.get('recommendation')}`**.",
    ]
    for r in recommendation.get("rationale", []):
        lines.append(f"- {r}")
    lines += [
        "",
        "### Q10 — Family-specific terrain problems (A_baseline macro)",
        "",
        "| Family | columns | A success |",
        "|---|---|---:|",
    ]
    per_fam = (
        by_reg.get(REGIME_A, {})
        .get("semantic_family_macro", {})
        .get("per_family", {})
    )
    for fam, info in sorted(per_fam.items()):
        lines.append(
            f"| `{fam}` | {info.get('columns')} | "
            f"{_fmt(info.get('column_mean_success_rate'))} |"
        )
    # Also show family × |vx| for A anchors if present in by_family_x_regime
    lines += [
        "",
        "#### Family × regime success (overall rate)",
        "",
        "| Family | A | B | C | D | E |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    fxr = summary.get("by_family_x_regime") or {}
    for fam in sorted(fxr):
        row = fxr[fam]
        lines.append(
            "| `{fam}` | {a} | {b} | {c} | {d} | {e} |".format(
                fam=fam,
                a=_fmt((row.get(REGIME_A) or {}).get("success_rate")),
                b=_fmt((row.get(REGIME_B) or {}).get("success_rate")),
                c=_fmt((row.get(REGIME_C) or {}).get("success_rate")),
                d=_fmt((row.get(REGIME_D) or {}).get("success_rate")),
                e=_fmt((row.get(REGIME_E) or {}).get("success_rate")),
            )
        )
    lines += [
        "",
        "#### vx_sign slice",
        "",
        "| vx_sign | success_rate | n |",
        "|---|---:|---:|",
    ]
    for sign, r in sorted((summary.get("by_vx_sign") or {}).items()):
        lines.append(f"| {sign} | {_fmt(r.get('success_rate'))} | {r.get('n')} |")
    lines += [
        "",
        "## V6.1 recommendation",
        "",
        f"**{recommendation.get('recommendation')}**",
        "",
    ]
    for r in recommendation.get("rationale", []):
        lines.append(f"- {r}")
    lines.append("")
    lines.append(
        "Note: mastery / unlock probabilities in the sweep use an independence "
        "approximation over a single window and Beta(1,1) required-success counts; "
        "they are not unlock guarantees."
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    try:
        if x is None or (isinstance(x, float) and x != x):
            return "nan"
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


# Mandatory episode provenance keys required by the diagnostic contract (v2).
MANDATORY_EPISODE_FIELDS = (
    "schema_version",
    "purpose",
    "eligible_for_checkpoint_selection",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_iteration",
    "training_seed",
    "git_commit",
    "working_tree_dirty",
    "config_fingerprint",
    "bank_fingerprint",
    "geometry_seed",
    "scene_geometry_hash",
    "runtime_tile_geometry_hash",
    "requested_terrain_column",
    "requested_terrain_level",
    "runtime_terrain_column",
    "runtime_terrain_level",
    "terrain_family",
    "terrain_column",
    "terrain_level",
    "speed_bin",
    "regime",
    "episode_id",
    "mean_linear_error",
    "mean_yaw_error",
    "timed_out",
    "observed_timeout_event",
    "survived_measurement_horizon",
    "max_episode_length",
    "timeout_comparison",
    "frontier_success_at_original_thresholds",
)


def policy_training_seed_from_config(config: Mapping[str, Any] | None) -> int:
    """Policy training seed for the evaluated run (independent of geometry_seed)."""
    if config is None:
        return 1
    if "training_seed" in config:
        return int(config["training_seed"])
    # Prefer explicit geometry.training_seed if present; else the seed that
    # matches the training scene (training_geometry_seed) for this pilot.
    geo = config.get("geometry") or {}
    if "training_seed" in geo:
        return int(geo["training_seed"])
    return int(geo.get("training_geometry_seed", 1))


def normalize_episode_record(
    rec: Mapping[str, Any],
    *,
    default_training_seed: int | None = 1,
) -> dict[str, Any]:
    """Backfill contract fields so older shards remain analyzable and complete.

    ``training_seed`` is the *policy* training seed of the evaluated checkpoint.
    It is independent of ``geometry_seed`` (offline terrain scene).  Missing
    values are filled from ``default_training_seed`` for every geometry seed.
    """
    out = dict(rec)
    if "terrain_column" not in out and "physical_column" in out:
        out["terrain_column"] = int(out["physical_column"])
    if "physical_column" not in out and "terrain_column" in out:
        out["physical_column"] = int(out["terrain_column"])
    if "checkpoint_path" not in out and out.get("run_dir") and out.get("checkpoint"):
        out["checkpoint_path"] = str(Path(str(out["run_dir"])) / str(out["checkpoint"]))
    if "checkpoint_iteration" not in out or out.get("checkpoint_iteration") is None:
        if out.get("checkpoint"):
            m = re.search(r"model_(\d+)\.pt", str(out["checkpoint"]))
            if m:
                out["checkpoint_iteration"] = int(m.group(1))
    # Always fill training_seed when absent/None — do NOT gate on training_seed_matched.
    if out.get("training_seed") is None and default_training_seed is not None:
        out["training_seed"] = int(default_training_seed)
    if "working_tree_dirty" not in out:
        out["working_tree_dirty"] = None
    if "schema_version" not in out:
        out["schema_version"] = SCHEMA_VERSION
    if "purpose" not in out:
        out["purpose"] = "diagnostic_only"
    if "eligible_for_checkpoint_selection" not in out:
        out["eligible_for_checkpoint_selection"] = False
    return out


def episode_missing_mandatory_fields(rec: Mapping[str, Any]) -> list[str]:
    missing = []
    for key in MANDATORY_EPISODE_FIELDS:
        if key not in rec or rec[key] is None:
            # working_tree_dirty may legitimately be None (unknown); still require key.
            if key == "working_tree_dirty" and key in rec:
                continue
            missing.append(key)
    return missing


def read_ndjson(
    path: Path, *, default_training_seed: int | None = 1
) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(
            normalize_episode_record(
                json.loads(line), default_training_seed=default_training_seed
            )
        )
    return records


def write_ndjson(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def rewrite_episode_artifacts_with_provenance(
    out_dir: Path,
    *,
    default_training_seed: int = 1,
) -> dict[str, int]:
    """Rewrite every geometry_seed_*.episodes.ndjson (+ merged if present) in place.

    Ensures on-disk shards carry the full mandatory provenance schema, not only
    the merge-time view.
    """
    counts: dict[str, int] = {}
    paths = sorted(out_dir.glob("geometry_seed_*.episodes.ndjson"))
    # Prefer raw per-seed files; skip shard partials already merged naming.
    paths = [p for p in paths if ".shard" not in p.name]
    for path in paths:
        records = read_ndjson(path, default_training_seed=default_training_seed)
        for rec in records:
            missing = episode_missing_mandatory_fields(rec)
            if missing:
                raise ValueError(f"{path}: still missing {missing} after normalize")
        write_ndjson(path, records)
        counts[path.name] = len(records)
    merged_path = out_dir / "episodes.merged.ndjson"
    if merged_path.is_file() or paths:
        # Rebuild merged from rewritten per-seed files when present.
        if paths:
            merged: list[dict[str, Any]] = []
            for path in paths:
                merged.extend(read_ndjson(path, default_training_seed=default_training_seed))
            write_ndjson(merged_path, merged)
            counts[merged_path.name] = len(merged)
        else:
            records = read_ndjson(merged_path, default_training_seed=default_training_seed)
            for rec in records:
                missing = episode_missing_mandatory_fields(rec)
                if missing:
                    raise ValueError(f"{merged_path}: still missing {missing}")
            write_ndjson(merged_path, records)
            counts[merged_path.name] = len(records)
    return counts


def merge_geometry_episode_files(
    paths: Sequence[Path],
    *,
    expected_bank_fp: str,
    expected_config_fp: str,
    expected_checkpoint_sha: str,
    expected_count: int,
    default_training_seed: int = 1,
    bank_rows: Sequence[DiagnosticRow] | None = None,
) -> list[dict[str, Any]]:
    """Merge shard/geometry episode files with fail-closed provenance checks.

    When ``bank_rows`` is provided (recommended), every measured episode is
    compared field-by-field against the fingerprint-stable bank identity for
    its ``episode_id``.  Header fingerprint match alone is not enough.
    """
    expected_by_id: dict[str, dict[str, Any]] | None = None
    expected_ids: set[str] | None = None
    if bank_rows is not None:
        expected_by_id = {
            row.episode_id: bank_row_identity(row) for row in bank_rows
        }
        expected_ids = set(expected_by_id)
        if len(expected_ids) != expected_count:
            # Caller may pass a geometry-seed subset; count must still match.
            pass

    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        for rec in read_ndjson(path, default_training_seed=default_training_seed):
            if rec.get("bank_fingerprint") != expected_bank_fp:
                raise ValueError(f"{path}: bank_fingerprint mismatch")
            if rec.get("config_fingerprint") != expected_config_fp:
                raise ValueError(f"{path}: config_fingerprint mismatch")
            if rec.get("checkpoint_sha256") != expected_checkpoint_sha:
                raise ValueError(f"{path}: checkpoint_sha256 mismatch")
            missing = episode_missing_mandatory_fields(rec)
            if missing:
                raise ValueError(f"{path}: episode missing mandatory fields {missing}")
            eid = rec["episode_id"]
            if eid in seen_ids:
                raise ValueError(f"duplicate episode_id {eid}")
            if expected_by_id is not None:
                if eid not in expected_by_id:
                    raise ValueError(f"{path}: unknown episode_id {eid} not in bank")
                assert_episode_matches_bank_row(rec, expected_by_id[eid])
            seen_ids.add(eid)
            merged.append(rec)
    if len(merged) != expected_count:
        raise ValueError(f"merged episode count {len(merged)} != expected {expected_count}")
    if expected_ids is not None and seen_ids != expected_ids:
        missing_ids = sorted(expected_ids - seen_ids)
        extra_ids = sorted(seen_ids - expected_ids)
        raise ValueError(
            f"episode_id set mismatch vs bank: missing={missing_ids[:5]} "
            f"extra={extra_ids[:5]}"
        )
    return merged


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Merge/analyze V6 frontier diagnostic artifacts")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = artifact_dir(run_dir, args.iteration, config)
    ckpt = run_dir / f"model_{args.iteration}.pt"
    from legged_gym.scripts.eval.ckpt_utils import sha256_file

    ckpt_sha = sha256_file(str(ckpt))
    full_bank = build_bank(config)
    meta = write_bank_artifacts(out_dir, config, full_bank, checkpoint_sha256=ckpt_sha)
    train_seed = policy_training_seed_from_config(config)

    # Always rewrite per-seed shards so on-disk NDJSON carries full provenance
    # (not only the merge-time in-memory view).
    if args.merge or args.analyze:
        counts = rewrite_episode_artifacts_with_provenance(
            out_dir, default_training_seed=train_seed
        )
        if counts:
            print(f"[frontier_diagnostic] rewrote provenance: {counts}")

    if args.merge:
        paths = sorted(
            p
            for p in out_dir.glob("geometry_seed_*.episodes.ndjson")
            if ".shard" not in p.name
        )
        if not paths:
            raise FileNotFoundError(f"no geometry_seed_*.episodes.ndjson under {out_dir}")
        merged = merge_geometry_episode_files(
            paths,
            expected_bank_fp=meta["bank_fingerprint"],
            expected_config_fp=meta["config_fingerprint"],
            expected_checkpoint_sha=ckpt_sha,
            expected_count=expected_episode_count(config),
            default_training_seed=train_seed,
            bank_rows=full_bank,
        )
        write_ndjson(out_dir / "episodes.merged.ndjson", merged)
        print(f"[frontier_diagnostic] merged {len(merged)} episodes")
    else:
        merged_path = out_dir / "episodes.merged.ndjson"
        if not merged_path.is_file():
            raise FileNotFoundError("episodes.merged.ndjson missing; pass --merge first")
        merged = read_ndjson(merged_path, default_training_seed=train_seed)

    if args.analyze:
        summary = summarize(merged, config)
        sweep = threshold_sweep(merged, config)
        rec = recommend_v61(summary, sweep)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out_dir / "threshold_sweep.json").write_text(
            json.dumps(sweep, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (out_dir / "recommendation.json").write_text(
            json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = render_report_md(
            summary,
            sweep,
            rec,
            checkpoint_sha256=ckpt_sha,
            bank_fp=meta["bank_fingerprint"],
            config_fp=meta["config_fingerprint"],
        )
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        print(f"[frontier_diagnostic] wrote analysis under {out_dir}")
        print(json.dumps({"recommendation": rec["recommendation"], "n": len(merged)}, indent=2))


if __name__ == "__main__":
    main()
