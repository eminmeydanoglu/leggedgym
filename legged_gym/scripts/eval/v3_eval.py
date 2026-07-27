"""V3 generalisation scorecard runner.

This is intentionally separate from :mod:`campaign`: Eval V2 describes the
historical axis/hard-suite artifact contract, while V3 evaluates one frozen
training family with static payload cells, deterministic mid-episode switches,
and an explicit oracle-gated scorecard.  It reuses the V2 strict deployment
loader and atomic artifact writer, but never changes their existing output
shape or interpretation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml

from legged_gym.scripts.eval.campaign import (
    _artifact_valid,
    _meta,
    _root,
    _save_cell,
    _set_nominal,
    _terrain_hash,
    _terrain_raw,
    build_session,
    build_terrain_session,
    parse_shard,
    shard_items,
)
from legged_gym.scripts.eval.ckpt_utils import resolve_checkpoint_path, sha256_file
from legged_gym.scripts.eval.dr_axes import get_axis
from legged_gym.scripts.eval.metrics import MetricAccumulator
from legged_gym.scripts.eval.provenance import verify_run_identity


RAW_METRICS = ("tracking_lin", "tracking_yaw", "fall_rate", "return_per_step")
SCORED_SUITES = ("s1", "s2", "t1", "t2")
TERRAIN_SUITES = ("t0", "t1", "t2")
V4_MATRIX_SUITE = "h4"

# Canonical column->type map for the pinned eval grid.  With num_cols=5 and the
# benchmark proportions [0.2, 0.1, 0.25, 0.25, 0.2] the deterministic
# ``curiculum()`` generator assigns ``choice = col / num_cols`` to, in order:
# smooth slope, random uniform, pyramid stairs (down), pyramid stairs (up),
# discrete obstacles.  Falls back to a stable ``type{n}`` label off-grid.
_TERRAIN_TYPE_NAMES = {0: "slope", 1: "random_uniform", 2: "stairs_down", 3: "stairs_up", 4: "discrete"}


def _terrain_type_name(index: int, num_cols: int = 5) -> str:
    if num_cols == 5 and int(index) in _TERRAIN_TYPE_NAMES:
        return _TERRAIN_TYPE_NAMES[int(index)]
    return f"type{int(index)}"


@dataclass(frozen=True)
class Cell:
    suite: str
    model: str
    seed: int
    name: str
    scenario: Mapping[str, Any]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("V3 eval config must be a mapping")
    required = ("campaign", "artifact_root", "simulator", "training_seeds", "models", "protocol", "scorecard")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"V3 eval config missing required keys: {missing}")
    if not isinstance(cfg["campaign"], str) or not cfg["campaign"].strip():
        raise ValueError("campaign must be a non-empty name, e.g. 'v3_ilk_deneme'")
    if cfg["simulator"] != "genesis":
        raise ValueError("V3 eval supports simulator: genesis only")
    labels = [model.get("label") for model in cfg["models"]]
    if len(labels) != len(set(labels)) or any(not label for label in labels):
        raise ValueError("models must have unique non-empty labels")
    score = cfg["scorecard"]
    for key in ("baseline_label", "oracle_label", "method_labels", "relative_headroom", "absolute_headroom", "fall_gate_pp", "achieved_speed_ratio"):
        if key not in score:
            raise ValueError(f"scorecard missing {key!r}")
    known = set(labels)
    if score["baseline_label"] not in known or score["oracle_label"] not in known:
        raise ValueError("scorecard baseline/oracle labels must be declared models")
    if not set(score["method_labels"]).issubset(known - {score["baseline_label"], score["oracle_label"]}):
        raise ValueError("scorecard method_labels must exclude baseline and oracle")
    return cfg


def artifact_root(cfg: Mapping[str, Any]) -> Path:
    path = Path(str(cfg["artifact_root"]))
    return path if path.is_absolute() else _root() / path


def _cell_rel(cell: Cell) -> str:
    return f"{cell.suite}/{cell.model}/seed_{cell.seed}/{cell.name}"


def _seed_list(cfg: Mapping[str, Any], model: Mapping[str, Any]) -> List[int]:
    return [int(seed) for seed in model.get("training_seeds", cfg["training_seeds"])]


def _filter_models(cfg: Mapping[str, Any], labels: Sequence[str] | None) -> List[Mapping[str, Any]]:
    """Return a caller-selected subset without silently accepting a typo."""
    if not labels:
        return list(cfg["models"])
    wanted = {str(label) for label in labels}
    known = {str(model["label"]) for model in cfg["models"]}
    unknown = wanted - known
    if unknown:
        raise ValueError(f"unknown --models label(s): {sorted(unknown)}; known={sorted(known)}")
    return [model for model in cfg["models"] if model["label"] in wanted]


def payload_com_x(mass_kg: float, payload_cfg: Mapping[str, Any]) -> float:
    """Pre-registered deterministic mass -> forward-CoM payload mapping."""
    raw = float(mass_kg) * float(payload_cfg["com_x_per_kg"])
    limit = abs(float(payload_cfg["com_x_limit_m"]))
    return float(np.clip(raw, -limit, limit))


def _command_name(command: Mapping[str, Any]) -> str:
    return str(command["name"]).replace("/", "_").replace(" ", "_")


def _is_v4_headroom_matrix(cfg: Mapping[str, Any]) -> bool:
    interface = cfg.get("execution", {}).get("runner_interface", cfg.get("protocol_kind", ""))
    return cfg.get("schema_version") == "v4_headroom_matrix_v1" or interface == "v4_headroom_matrix_v1"


def _v4_value_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "neg").replace(".", "p")


def _v4_followup_manifest(cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Load an immutable discovery-selected world manifest, when configured."""
    raw_path = cfg.get("execution", {}).get("followup_manifest")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = _root() / path
    if not path.is_file():
        raise FileNotFoundError(f"V4 follow-up manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "v4_adaptive_followup_v1":
        raise ValueError(f"unsupported V4 follow-up manifest: {path}")
    return manifest


def _validate_v4_headroom_matrix(cfg: Mapping[str, Any]) -> None:
    if not _is_v4_headroom_matrix(cfg):
        return
    protocol = cfg["protocol"]
    terrain = protocol["terrain"]
    if cfg.get("simulator") != "genesis":
        raise ValueError("v4_headroom_matrix_v1 supports Genesis only")
    required_nominal = {"mass_kg", "com_x_m", "friction"}
    if set(protocol["nominal_physics"]) != required_nominal:
        raise ValueError("v4_headroom_matrix_v1 nominal_physics must contain exactly mass_kg, com_x_m, friction")
    if set(protocol.get("training_support", {})) != required_nominal:
        raise ValueError("v4_headroom_matrix_v1 training_support must contain exactly mass_kg, com_x_m, friction")
    for axis_name, bounds in protocol["training_support"].items():
        if len(bounds) != 2 or float(bounds[0]) >= float(bounds[1]):
            raise ValueError(f"training_support.{axis_name} must be [lower, upper]")
        nominal = float(protocol["nominal_physics"][axis_name])
        if not float(bounds[0]) <= nominal <= float(bounds[1]):
            raise ValueError(f"nominal_physics.{axis_name} must lie inside training_support")
    if len(terrain["types"]) > int(terrain["num_cols"]):
        raise ValueError("declared terrain.types exceeds terrain.num_cols")
    if not terrain["types"] or not terrain["levels"] or not protocol["commands"]["vx"]:
        raise ValueError("v4_headroom_matrix_v1 needs non-empty terrain types/levels and command vx")
    known_types = {_terrain_type_name(index, int(terrain["num_cols"])) for index in range(int(terrain["num_cols"]))}
    unknown_types = set(terrain["types"]) - known_types
    if unknown_types:
        raise ValueError(f"unknown terrain type(s) for configured terrain proportions: {sorted(unknown_types)}")
    if int(protocol["replicas_per_cell"]) < 1:
        raise ValueError("replicas_per_cell must be positive")
    if int(protocol["warmup_steps"]) < 0 or int(protocol["measured_steps"]) < 1:
        raise ValueError("warmup_steps must be >= 0 and measured_steps must be positive")
    axes = protocol["primary_isolated_axes"]["axes"]
    expected = {"mass_kg": "added_mass", "com_x_m": "com_x", "friction": "friction"}
    names = [str(axis["name"]) for axis in axes]
    if not names or len(names) != len(set(names)) or not set(names).issubset(expected):
        raise ValueError("primary_isolated_axes must declare a non-empty unique subset of mass_kg, com_x_m, friction")
    for axis in axes:
        if axis["dr_axis"] != expected[axis["name"]]:
            raise ValueError(f"{axis['name']} must use dr_axis={expected[axis['name']]}")
        if not axis["id"] or not axis["ood"] or set(axis["id"]) & set(axis["ood"]):
            raise ValueError(f"{axis['name']} must have non-empty, disjoint id/ood values")
        lower, upper = map(float, protocol["training_support"][axis["name"]])
        if not all(lower <= float(value) <= upper for value in axis["id"]):
            raise ValueError(f"{axis['name']} ID values must lie inside training_support")
        if not all(float(value) < lower or float(value) > upper for value in axis["ood"]):
            raise ValueError(f"{axis['name']} OOD values must lie outside training_support")
    if not bool(protocol["primary_isolated_axes"].get("pin_unswept_axes_to_nominal")):
        raise ValueError("v4 primary cells require pin_unswept_axes_to_nominal=true")
    manifest = _v4_followup_manifest(cfg)
    if manifest is not None:
        requested = {str(item) for item in cfg.get("execution", {}).get("followup_model_labels", [])}
        known = {str(model["label"]) for model in cfg["models"]}
        if not requested or not requested.issubset(known):
            raise ValueError("V4 follow-up must declare non-empty execution.followup_model_labels from models")
        if requested != set(manifest.get("method_labels", [])):
            raise ValueError("V4 follow-up config method labels differ from frozen manifest")
        if not manifest.get("worlds"):
            raise ValueError("V4 follow-up manifest has no eligible worlds; do not run adaptation policies")


def _v4_headroom_cells(cfg: Mapping[str, Any]) -> List[Cell]:
    """Expand the V4 matrix without conflating primary and stress tiers."""
    _validate_v4_headroom_matrix(cfg)
    protocol = cfg["protocol"]
    terrain = protocol["terrain"]
    nominal = {key: float(value) for key, value in protocol["nominal_physics"].items()}
    cells: List[Cell] = []
    type_to_index = {_terrain_type_name(index, int(terrain["num_cols"])): index for index in range(int(terrain["num_cols"]))}
    manifest = _v4_followup_manifest(cfg)
    if manifest is not None:
        labels = {str(item) for item in cfg["execution"]["followup_model_labels"]}
        for model in cfg["models"]:
            if str(model["label"]) not in labels:
                continue
            model_seeds = set(_seed_list(cfg, model))
            for world in manifest["worlds"]:
                identity = world["identity"]
                terrain_type = str(identity["terrain_type"])
                if terrain_type not in type_to_index:
                    raise ValueError(f"follow-up terrain type is not present in config: {terrain_type}")
                physics = {key: float(value) for key, value in json.loads(str(identity["physics_signature"])).items()}
                for seed in world["training_seeds"]:
                    if int(seed) not in model_seeds:
                        continue
                    name = str(identity["scenario"])
                    cells.append(Cell(V4_MATRIX_SUITE, str(model["label"]), int(seed), name, {
                        "kind": "v4_headroom", "tier": str(identity["physics_tier"]),
                        "terrain_type": terrain_type, "terrain_type_index": int(type_to_index[terrain_type]),
                        "terrain_level": int(identity["terrain_level"]), "command_vx": float(identity["command_vx"]),
                        "physics_axis": str(identity["physics_axis"]),
                        "physics_dr_axis": {"mass_kg": "added_mass", "com_x_m": "com_x", "friction": "friction"}.get(str(identity["physics_axis"]), "combined"),
                        "physics_band": str(identity["physics_band"]), "physics": physics,
                        "physics_scenario": "" if str(identity["physics_axis"]) != "combined" else name.split("__secondary__", 1)[-1],
                    }))
        return cells
    primary = protocol["primary_isolated_axes"]
    for model in cfg["models"]:
        for seed in _seed_list(cfg, model):
            for terrain_type in terrain["types"]:
                for level in terrain["levels"]:
                    for vx in protocol["commands"]["vx"]:
                        for axis in primary["axes"]:
                            for band in ("id", "ood"):
                                for value in axis[band]:
                                    physics = dict(nominal); physics[axis["name"]] = float(value)
                                    name = (f"{terrain_type}__L{int(level)}__vx{_v4_value_name(vx)}__"
                                            f"primary__{axis['name']}__{band}__{_v4_value_name(value)}")
                                    cells.append(Cell(V4_MATRIX_SUITE, model["label"], seed, name, {
                                        "kind": "v4_headroom", "tier": str(primary["tier"]), "terrain_type": str(terrain_type),
                                        "terrain_type_index": int(type_to_index[str(terrain_type)]), "terrain_level": int(level),
                                        "command_vx": float(vx), "physics_axis": str(axis["name"]),
                                        "physics_dr_axis": str(axis["dr_axis"]), "physics_band": band,
                                        "physics": physics,
                                    }))
            secondary = protocol["secondary_combined_stress_payload"]
            for terrain_type in terrain["types"]:
                for level in secondary["terrain_levels"]:
                    for vx in secondary["vx"]:
                        for stress in secondary["scenarios"]:
                            physics = {key: float(stress[key]) for key in nominal}
                            name = (f"{terrain_type}__L{int(level)}__vx{_v4_value_name(vx)}__secondary__"
                                    f"{stress['name']}")
                            cells.append(Cell(V4_MATRIX_SUITE, model["label"], seed, name, {
                                "kind": "v4_headroom", "tier": str(secondary["tier"]), "terrain_type": str(terrain_type),
                                "terrain_type_index": int(type_to_index[str(terrain_type)]), "terrain_level": int(level),
                                "command_vx": float(vx), "physics_axis": "combined", "physics_dr_axis": "combined",
                                "physics_band": str(stress["band"]), "physics": physics,
                                "physics_scenario": str(stress["name"]),
                            }))
    budget = protocol.get("planned_cell_budget")
    if budget:
        primary_count = sum(cell.scenario["tier"] == primary["tier"] for cell in cells)
        secondary_count = len(cells) - primary_count
        actual = {"primary": primary_count, "secondary": secondary_count, "total": len(cells)}
        if {key: int(budget[key]) for key in actual} != actual:
            raise ValueError(f"v4 planned_cell_budget mismatch: declared={budget}, expanded={actual}")
    return cells


def _planned_cells(cfg: Mapping[str, Any], suite: str, *, include_diagnostics: bool = False) -> List[Cell]:
    if _is_v4_headroom_matrix(cfg):
        if suite != "all":
            raise ValueError("v4_headroom_matrix_v1 accepts --suite all only; tiers are declared inside the protocol")
        return _v4_headroom_cells(cfg)
    protocol = cfg["protocol"]
    terrain_cfg = protocol.get("terrain")
    if suite == "all":
        # ``all`` covers whichever families this campaign actually configures:
        # the historical flat s-suites and/or the terrain-native t-suites.  A
        # terrain-only config never plans empty s-cells and vice versa.
        requested = []
        if "payload" in protocol and "s1_commands" in protocol:
            requested += ["s0", "s1", "s2", "s3"]
        if terrain_cfg:
            requested += ["t0", "t1", "t2"]
    else:
        requested = (suite,)
    model_by_label = {model["label"]: model for model in cfg["models"]}
    out: List[Cell] = []
    payload_cfg = protocol.get("payload")
    if "s0" in requested:
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                out.append(Cell("s0", model["label"], seed, "static_id", {"kind": "static_id"}))
                out.append(Cell("s0", model["label"], seed, "dynamic_id_in_band", {
                    "kind": "dynamic_id", "target": cfg["protocol"]["s0_dynamic_target"],
                }))
    if "s1" in requested:
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                for payload in payload_cfg["static"]:
                    for command in cfg["protocol"]["s1_commands"]:
                        if not bool(command.get("headline", True)) and not include_diagnostics:
                            continue
                        name = f"payload_{payload['name']}__{_command_name(command)}"
                        out.append(Cell("s1", model["label"], seed, name, {
                            "kind": "static_payload", "payload": payload, "command": command,
                        }))
    if "s2" in requested:
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                for switch in payload_cfg["switches"]:
                    for command in cfg["protocol"]["s2_commands"]:
                        if not bool(command.get("headline", True)) and not include_diagnostics:
                            continue
                        name = f"switch_{switch['name']}__{_command_name(command)}"
                        out.append(Cell("s2", model["label"], seed, name, {
                            "kind": "payload_switch", "switch": switch, "command": command,
                        }))
    if "s3" in requested:
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                for kick in cfg["protocol"]["s3"]["kick_dvy"]:
                    out.append(Cell("s3", model["label"], seed, f"kick_{str(kick).replace('.', '_')}", {
                        "kind": "kick", "kick_dvy": float(kick),
                    }))
                for terrain_cm in cfg["protocol"]["s3"]["terrain_cm"]:
                    out.append(Cell("s3", model["label"], seed, f"terrain_{str(terrain_cm).replace('.', '_')}cm", {
                        "kind": "terrain", "terrain_cm": float(terrain_cm),
                    }))
    if terrain_cfg and ("t0" in requested or "t1" in requested or "t2" in requested):
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                if "t0" in requested:
                    id_level = int(terrain_cfg["id_level"])
                    out.append(Cell("t0", model["label"], seed, f"terrain_id_level_{id_level}", {
                        "kind": "terrain_id", "level": id_level,
                    }))
                if "t1" in requested:
                    for level in terrain_cfg["severity_levels"]:
                        out.append(Cell("t1", model["label"], seed, f"terrain_severity_level_{int(level)}", {
                            "kind": "terrain_severity", "level": int(level),
                        }))
                if "t2" in requested:
                    payload_level = int(terrain_cfg["payload_level"])
                    for payload in terrain_cfg["payloads"]:
                        out.append(Cell("t2", model["label"], seed, f"terrain_payload_{payload['name']}", {
                            "kind": "terrain_payload", "level": payload_level, "payload": payload,
                        }))
    # Detect typoed method labels in a config while producing a useful error.
    if {cell.model for cell in out} - set(model_by_label):
        raise RuntimeError("planned an undeclared model")
    return out


def _selection_path(root: Path) -> Path:
    return root / "run_selection.json"


def _discover_run(model: Mapping[str, Any], seed: int, *, require_final: bool) -> Path:
    root = Path(str(model["log_root"]))
    root = root if root.is_absolute() else _root() / root
    pattern = str(model["run_pattern"]).format(seed=seed)
    pinned_paths = model.get("run_paths", {})
    pinned = pinned_paths.get(seed, pinned_paths.get(str(seed))) if isinstance(pinned_paths, Mapping) else None
    candidate_dirs = [root / str(pinned)] if pinned else list(root.glob(pattern))
    candidates = []
    for run_dir in candidate_dirs:
        if not run_dir.is_dir():
            continue
        if require_final and not (run_dir / "model_3000.pt").is_file():
            continue
        try:
            checkpoint = Path(resolve_checkpoint_path(str(run_dir), model.get("checkpoint", "best_tracking")))
            verify_run_identity(str(run_dir), expected_task=str(model["task"]),
                                expected_training_seed=seed, require_manifest=True)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append((checkpoint.stat().st_mtime, run_dir, checkpoint))
    if not candidates:
        suffix = " with model_3000.pt and best_tracking.pt" if require_final else ""
        raise FileNotFoundError(
            f"{model['label']} seed {seed}: no verified V3 run matching {root / pattern}{suffix}"
        )
    if len(candidates) > 1:
        names = sorted(str(item[1].name) for item in candidates)
        raise RuntimeError(f"{model['label']} seed {seed}: multiple verified runs match {pattern}: {names}; pin run_paths in config")
    return candidates[0][1]


def cmd_resolve_runs(cfg: Mapping[str, Any], *, labels: Sequence[str] | None = None, strict: bool = False) -> int:
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    # V4 curriculum runs select the explicit best_spnte deploy checkpoint; they
    # are not required to retain V3's historical model_3000.pt final-save.
    require_final = bool(cfg.get("require_final_checkpoint", not _is_v4_headroom_matrix(cfg)))
    path = _selection_path(root)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"selections": []}
    selection_by_key = {(str(row["label"]), int(row["training_seed"])): row
                        for row in previous.get("selections", [])}
    discovered, skipped = [], []
    for model in _filter_models(cfg, labels):
        for seed in _seed_list(cfg, model):
            try:
                run_dir = _discover_run(model, seed, require_final=require_final)
            except FileNotFoundError as exc:
                if strict:
                    raise
                skipped.append({"label": model["label"], "training_seed": seed, "reason": str(exc)})
                continue
            checkpoint = Path(resolve_checkpoint_path(str(run_dir), model.get("checkpoint", "best_tracking")))
            row = {
                "label": model["label"], "task": model["task"], "training_seed": seed,
                "run_folder": run_dir.name, "run_path": str(run_dir),
                "checkpoint": model.get("checkpoint", "best_tracking"),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(str(checkpoint)),
            }
            key = (str(model["label"]), int(seed))
            old = selection_by_key.get(key)
            if old and old.get("checkpoint_sha256") != row["checkpoint_sha256"]:
                raise RuntimeError(
                    f"{model['label']} seed {seed}: existing campaign selection points to a different checkpoint. "
                    "Start a new campaign instead of silently mixing policies."
                )
            selection_by_key[key] = row
            discovered.append(row)
    selections = [selection_by_key[key] for key in sorted(selection_by_key)]
    payload = {
        "campaign": cfg["campaign"], "generated_at": datetime.now(timezone.utc).isoformat(),
        "require_final_checkpoint": require_final, "selections": selections, "skipped": skipped,
    }
    path.write_text(_json(payload), encoding="utf-8")
    print(f"[v3_eval] added/verified {len(discovered)} run(s); campaign now has {len(selections)} selected policy-seed(s) -> {path}")
    for row in skipped:
        print(f"[v3_eval] pending {row['label']} seed{row['training_seed']}: no final checkpoint yet")
    return 0


def _selected_models(cfg: Mapping[str, Any], root: Path, labels: Sequence[str] | None = None) -> Dict[str, Dict[str, Any]]:
    path = _selection_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run 'resolve-runs' after V3 training completes")
    data = json.loads(path.read_text(encoding="utf-8"))
    selected: Dict[str, Dict[str, Any]] = {}
    for model in _filter_models(cfg, labels):
        entries = [entry for entry in data.get("selections", []) if entry.get("label") == model["label"]]
        by_seed = {int(entry["training_seed"]): entry for entry in entries}
        if not by_seed:
            continue
        expected = [seed for seed in _seed_list(cfg, model) if seed in by_seed]
        copied = dict(model)
        copied["run_folders"] = {seed: by_seed[seed]["run_folder"] for seed in expected}
        copied["selected_checkpoints"] = {seed: by_seed[seed] for seed in expected}
        copied["training_seeds"] = expected
        selected[model["label"]] = copied
    if labels and not selected:
        raise FileNotFoundError("none of the requested models are selected yet; run resolve-runs once their checkpoints exist")
    return selected


def _id_commands(num_envs: int, seed: int, bank: Mapping[str, Any]) -> np.ndarray:
    """Fixed V3 direct-velocity bank, with support declared in campaign YAML."""
    rng = np.random.default_rng(seed)
    x_values, y_values, yaw_values = (tuple(float(x) for x in bank[key]) for key in ("anchor_x", "anchor_y", "anchor_yaw_rate"))
    anchors = np.asarray([[x, y, z] for x in x_values for y in y_values for z in yaw_values], dtype=np.float32)
    commands = np.empty((num_envs, 3), dtype=np.float32)
    commands[:, 0] = rng.uniform(*map(float, bank["lin_vel_x"]), size=num_envs)
    commands[:, 1] = rng.uniform(*map(float, bank["lin_vel_y"]), size=num_envs)
    commands[:, 2] = rng.uniform(*map(float, bank["ang_vel_yaw"]), size=num_envs)
    commands[:min(len(anchors), num_envs)] = anchors[:min(len(anchors), num_envs)]
    return commands


def spnte_scales_from_command_bank(bank: Mapping[str, Any]) -> Tuple[float, float]:
    """Return linear-x and yaw command-support scales for SPNTE.

    A campaign support is fixed for an artifact; it is deliberately not the
    instantaneous command magnitude, so standstill commands remain well-defined.
    A degenerate all-zero support is represented by machine epsilon solely to
    keep its zero-error normalized score finite.
    """
    def scale(key: str) -> float:
        values = np.asarray(bank[key], dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"SPNTE command support {key!r} must be finite and non-empty")
        return max(float(np.max(np.abs(values))), float(np.finfo(np.float32).eps))

    return scale("lin_vel_x"), scale("ang_vel_yaw")


def _spnte_scales(commands: np.ndarray, bank: Mapping[str, Any] | None = None) -> Tuple[float, float]:
    if bank is not None:
        return spnte_scales_from_command_bank(bank)
    return spnte_scales_from_command_bank({"lin_vel_x": commands[:, 0], "ang_vel_yaw": commands[:, 2]})


def _command_array(command: Mapping[str, Any], num_envs: int) -> np.ndarray:
    vector = np.asarray(command["vector"], dtype=np.float32)
    if vector.shape != (3,):
        raise ValueError(f"command {command.get('name')!r} vector must have length 3")
    return np.repeat(vector[None, :], num_envs, axis=0)


def _apply_payload(env, mass_kg: float, payload_cfg: Mapping[str, Any], *, pin_nominal: bool) -> Dict[str, np.ndarray]:
    if pin_nominal:
        _set_nominal(env)
    mass = torch.full((env.num_envs,), float(mass_kg), device=env.device)
    com_x = torch.full((env.num_envs,), payload_com_x(float(mass_kg), payload_cfg), device=env.device)
    mass_axis, com_axis = get_axis("added_mass"), get_axis("com_x")
    mass_axis.apply(env, mass)
    com_axis.apply(env, com_x)
    return {
        "payload_mass_requested": mass.detach().cpu().numpy(),
        "payload_mass_actual": mass_axis.validate(env, mass).detach().cpu().numpy(),
        "payload_com_x_requested": com_x.detach().cpu().numpy(),
        "payload_com_x_actual": com_axis.validate(env, com_x).detach().cpu().numpy(),
    }


def _warmup(session, commands: np.ndarray, warmup: int):
    env, adapter = session.env, session.adapter
    command_t = torch.as_tensor(commands, dtype=torch.float, device=env.device)
    state = adapter.reset(env)
    for _ in range(int(warmup)):
        env.commands[:, :3] = command_t
        state, _, _ = adapter.step(env, adapter.act(state).detach())
    env.episode_length_buf.zero_()
    return state, command_t


def _kinematics(env, command_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lin = torch.linalg.vector_norm(command_t[:, :2] - env.simulator.base_lin_vel[:, :2], dim=1)
    yaw = (command_t[:, 2] - env.simulator.base_ang_vel[:, 2]).abs()
    speed_target = torch.linalg.vector_norm(command_t[:, :2], dim=1)
    unit = torch.where(speed_target[:, None] > 1e-6, command_t[:, :2] / speed_target.clamp_min(1e-6)[:, None], torch.zeros_like(command_t[:, :2]))
    achieved = (env.simulator.base_lin_vel[:, :2] * unit).sum(dim=1)
    return lin, yaw, achieved


@torch.no_grad()
def _rollout_fixed(
    session,
    commands: np.ndarray,
    *,
    warmup: int,
    steps: int,
    command_bank: Mapping[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    env, adapter = session.env, session.adapter
    state, command_t = _warmup(session, commands, warmup)
    v_scale, v_scale_yaw = _spnte_scales(commands, command_bank)
    acc = MetricAccumulator(env.num_envs, env.device, v_scale=v_scale, v_scale_yaw=v_scale_yaw)
    achieved_sum = torch.zeros(env.num_envs, device=env.device)
    for _ in range(int(steps)):
        env.commands[:, :3] = command_t
        state, rewards, dones = adapter.step(env, adapter.act(state).detach())
        lin, yaw, achieved = _kinematics(env, command_t)
        lin_x = (command_t[:, 0] - env.simulator.base_lin_vel[:, 0]).abs()
        acc.update(rewards, dones, env.time_out_buf, lin, yaw, lin_x_err=lin_x)
        achieved_sum += achieved
    raw = acc.compute()
    target = torch.linalg.vector_norm(command_t[:, :2], dim=1)
    achieved = achieved_sum / float(steps)
    return {
        "tracking_lin": raw["tracking_lin_err"].cpu().numpy(),
        "tracking_yaw": raw["tracking_ang_err"].cpu().numpy(),
        "fall_rate": raw["ever_fell"].cpu().numpy(),
        "return_per_step": raw["return_per_step"].cpu().numpy(),
        "spnte_lin": raw["spnte_lin"].cpu().numpy(),
        "spnte_yaw": raw["spnte_yaw"].cpu().numpy(),
        "first_fall_step": raw["first_fall_step"].cpu().numpy(),
        "spnte_v_scale": np.asarray(v_scale, dtype=np.float32),
        "spnte_yaw_scale": np.asarray(v_scale_yaw, dtype=np.float32),
        "achieved_speed": achieved.cpu().numpy(),
        "achieved_speed_ratio": torch.where(target > 1e-6, achieved / target, torch.ones_like(target)).cpu().numpy(),
    }


def _sysid_trace(session, state) -> Tuple[torch.Tensor, torch.Tensor] | None:
    actor_critic = getattr(getattr(session.runner, "alg", None), "actor_critic", None)
    estimator = getattr(actor_critic, "estimator", None)
    truth = getattr(session.env, "estimator_labels_buf", None)
    if estimator is None or truth is None or not isinstance(state, torch.Tensor):
        return None
    return estimator(state.detach()), truth.detach()


def _sysid_trace_summary(pred: torch.Tensor, truth: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compact estimator diagnostics without cancellation across environments."""
    error = pred - truth
    return {
        "pred_mean": pred.mean(dim=0),
        "truth_mean": truth.mean(dim=0),
        "error_mean": error.mean(dim=0),
        "mae": error.abs().mean(dim=0),
        "rmse": error.square().mean(dim=0).sqrt(),
    }


@torch.no_grad()
def _rollout_switch(
    session, commands: np.ndarray, *, warmup: int, pre_steps: int, post_steps: int,
    target_mass: float, payload_cfg: Mapping[str, Any], command_bank: Mapping[str, Any] | None = None,
) -> Dict[str, np.ndarray]:
    """Run an externally scheduled, identical V3 payload switch for every env."""
    env, adapter = session.env, session.adapter
    state, command_t = _warmup(session, commands, warmup)
    total = int(pre_steps) + int(post_steps)
    v_scale, v_scale_yaw = _spnte_scales(commands, command_bank)
    spnte = MetricAccumulator(env.num_envs, env.device, v_scale=v_scale, v_scale_yaw=v_scale_yaw)
    lin_errors: List[torch.Tensor] = []
    yaw_errors: List[torch.Tensor] = []
    achieved: List[torch.Tensor] = []
    rewards_sum = torch.zeros(env.num_envs, device=env.device)
    ever_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    post_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    trace_summaries: Dict[str, List[torch.Tensor]] = defaultdict(list)
    switch_payload: Dict[str, np.ndarray] | None = None
    for step in range(total):
        if step == int(pre_steps):
            switch_payload = _apply_payload(env, target_mass, payload_cfg, pin_nominal=False)
        env.commands[:, :3] = command_t
        trace = _sysid_trace(session, state)
        state, rewards, dones = adapter.step(env, adapter.act(state).detach())
        lin, yaw, current_speed = _kinematics(env, command_t)
        lin_x = (command_t[:, 0] - env.simulator.base_lin_vel[:, 0]).abs()
        # Keep the established stream metrics below untouched.  This separate
        # accumulator makes the first payload-switch failure terminal for
        # SPNTE while preserving auto-reset behavior for historical fields.
        spnte.update(rewards, dones, env.time_out_buf, lin, yaw, lin_x_err=lin_x)
        fell = dones.bool() & ~env.time_out_buf.bool()
        ever_fell |= fell
        if step >= int(pre_steps):
            post_fell |= fell
        rewards_sum += rewards
        lin_errors.append(lin)
        yaw_errors.append(yaw)
        achieved.append(current_speed)
        if trace is not None:
            pred, truth = trace
            for key, value in _sysid_trace_summary(pred, truth).items():
                trace_summaries[key].append(value.detach().cpu())
    lin_error_t = torch.stack(lin_errors)
    yaw_error_t = torch.stack(yaw_errors)
    speed_t = torch.stack(achieved)
    pre_lin = lin_error_t[:pre_steps]
    post_lin = lin_error_t[pre_steps:]
    post_yaw = yaw_error_t[pre_steps:]
    target_speed = torch.linalg.vector_norm(command_t[:, :2], dim=1)
    spnte_raw = spnte.compute()
    result: Dict[str, np.ndarray] = {
        "tracking_lin": lin_error_t.mean(dim=0).cpu().numpy(),
        "tracking_yaw": yaw_error_t.mean(dim=0).cpu().numpy(),
        "fall_rate": ever_fell.float().cpu().numpy(),
        "return_per_step": (rewards_sum / float(total)).cpu().numpy(),
        "pre_switch_tracking_error": pre_lin.mean(dim=0).cpu().numpy(),
        "post_switch_tracking_error": post_lin.mean(dim=0).cpu().numpy(),
        "post_switch_tracking_yaw": post_yaw.mean(dim=0).cpu().numpy(),
        "post_switch_fall_rate": post_fell.float().cpu().numpy(),
        "achieved_speed": speed_t[pre_steps:].mean(dim=0).cpu().numpy(),
        "achieved_speed_ratio": torch.where(target_speed > 1e-6, speed_t[pre_steps:].mean(dim=0) / target_speed, torch.ones_like(target_speed)).cpu().numpy(),
        "spnte_lin": spnte_raw["spnte_lin"].cpu().numpy(),
        "spnte_yaw": spnte_raw["spnte_yaw"].cpu().numpy(),
        "first_fall_step": spnte_raw["first_fall_step"].cpu().numpy(),
        "spnte_v_scale": np.asarray(v_scale, dtype=np.float32),
        "spnte_yaw_scale": np.asarray(v_scale_yaw, dtype=np.float32),
    }
    if switch_payload is None:
        raise RuntimeError("switch rollout ended before its scheduled switch")
    result.update(switch_payload)
    if trace_summaries:
        result["p_hat_trace"] = torch.stack(trace_summaries["pred_mean"]).numpy()
        result["p_true_trace"] = torch.stack(trace_summaries["truth_mean"]).numpy()
        result["p_hat_error_mean_trace"] = torch.stack(trace_summaries["error_mean"]).numpy()
        result["p_hat_mae_trace"] = torch.stack(trace_summaries["mae"]).numpy()
        result["p_hat_rmse_trace"] = torch.stack(trace_summaries["rmse"]).numpy()
    return result


@torch.no_grad()
def _rollout_kick(session, commands: np.ndarray, *, warmup: int, pre_steps: int, post_steps: int, dvy: float) -> Dict[str, np.ndarray]:
    env, adapter = session.env, session.adapter
    state, command_t = _warmup(session, commands, warmup)
    errors: List[torch.Tensor] = []
    ever_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    rewards_sum = torch.zeros(env.num_envs, device=env.device)
    for step in range(int(pre_steps) + int(post_steps)):
        env.commands[:, :3] = command_t
        if step == int(pre_steps):
            qvel = env.simulator._robot.get_dofs_velocity()
            qvel[:, 1] += float(dvy)
            env.simulator._robot.set_dofs_velocity(qvel)
        state, rewards, dones = adapter.step(env, adapter.act(state).detach())
        lin, _, _ = _kinematics(env, command_t)
        errors.append(lin)
        rewards_sum += rewards
        ever_fell |= dones.bool() & ~env.time_out_buf.bool()
    error_t = torch.stack(errors)
    post = error_t[int(pre_steps):]
    return {
        "tracking_lin": error_t.mean(dim=0).cpu().numpy(),
        "tracking_yaw": np.zeros(env.num_envs, dtype=np.float32),
        "fall_rate": ever_fell.float().cpu().numpy(),
        "return_per_step": (rewards_sum / float(len(errors))).cpu().numpy(),
        "post_kick_tracking_error_integral": (post.sum(dim=0) * float(env.dt)).cpu().numpy(),
        "kick_dvy": np.full(env.num_envs, float(dvy), dtype=np.float32),
    }


def _payload_metadata(payload: Mapping[str, Any], payload_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    mass = float(payload["mass_kg"])
    return {
        "payload_name": str(payload["name"]), "payload_tier": str(payload["tier"]),
        "payload_mass_kg": mass, "payload_com_x_m": payload_com_x(mass, payload_cfg),
    }


def _base_payload(session, suite: str, scenario: str, eval_seed: int, warmup: int, steps: int, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = _meta(session, suite, scenario, eval_seed, warmup, steps)
    out["campaign"] = cfg["campaign"]
    return out


def _v4_protocol_fingerprint(cfg: Mapping[str, Any]) -> str:
    """Stable provenance fingerprint for all semantics that define a matrix world."""
    payload = {
        "schema_version": cfg.get("schema_version"), "protocol": cfg["protocol"],
        "scorecard": {key: cfg["scorecard"].get(key) for key in ("baseline_label", "oracle_label", "headline_tier", "relative_headroom", "absolute_headroom", "fall_gate_pp", "achieved_speed_ratio")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _v4_campaign_fingerprint(cfg: Mapping[str, Any], root: Path | None = None) -> str:
    """Fingerprint the policy population and selection policy around a V4 world."""
    root = root or artifact_root(cfg)
    selection = _selection_path(root)
    selection_sha = hashlib.sha256(selection.read_bytes()).hexdigest() if selection.is_file() else ""
    payload = {
        "world_protocol_fingerprint": _v4_protocol_fingerprint(cfg),
        "training_seeds": [int(seed) for seed in cfg["training_seeds"]],
        "comparison_scope": cfg.get("comparison_scope", ""),
        "models": [{key: model.get(key) for key in ("label", "task", "checkpoint", "run_pattern", "run_paths")}
                   for model in cfg["models"]],
        "run_selection_sha256": selection_sha,
        "followup_manifest": cfg.get("execution", {}).get("followup_manifest", ""),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _v4_artifact_valid(path: Path, cfg: Mapping[str, Any], cell: Cell) -> bool:
    """Resume only a finite artifact from the exact same V4 matrix world."""
    if not _artifact_valid(path):
        return False
    scenario = cell.scenario
    required = {
        "protocol_kind": "v4_headroom_matrix_v1",
        "protocol_fingerprint": _v4_protocol_fingerprint(cfg),
        "campaign_fingerprint": _v4_campaign_fingerprint(cfg),
        "training_seed": int(cell.seed), "model": str(cell.model), "tier": str(scenario["tier"]),
        "terrain_type_name": str(scenario["terrain_type"]), "terrain_level": int(scenario["terrain_level"]),
        "command_vx": float(scenario["command_vx"]), "physics_axis": str(scenario["physics_axis"]),
        "physics_band": str(scenario["physics_band"]),
    }
    try:
        with np.load(path, allow_pickle=False) as z:
            for key, expected in required.items():
                actual = _scalar(z, key, object())
                if isinstance(expected, float):
                    if not np.isfinite(float(actual)) or not np.isclose(float(actual), expected, rtol=0.0, atol=1e-9):
                        return False
                elif actual != expected:
                    return False
            return str(_scalar(z, "physics_signature", "")) == json.dumps(scenario["physics"], sort_keys=True, separators=(",", ":"))
    except (OSError, ValueError, TypeError):
        return False


def _apply_v4_physics(env, physics: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Write and validate all three V4 P5 physics values, every cell."""
    expected_keys = {"mass_kg", "com_x_m", "friction"}
    if set(physics) != expected_keys:
        raise ValueError(f"V4 physics must contain exactly {sorted(expected_keys)}, got {sorted(physics)}")
    _set_nominal(env)
    mapping = (("mass_kg", "added_mass"), ("com_x_m", "com_x"), ("friction", "friction"))
    output: Dict[str, np.ndarray] = {}
    for physical_name, axis_name in mapping:
        requested = torch.full((env.num_envs,), float(physics[physical_name]), device=env.device)
        axis = get_axis(axis_name)
        axis.apply(env, requested)
        actual = axis.validate(env, requested)
        output[f"physics_{physical_name}_requested"] = requested.detach().cpu().numpy()
        output[f"physics_{physical_name}_actual"] = actual.detach().cpu().numpy()
    # ``com_x``'s setter must leave the unswept lateral/vertical components
    # nominal too. Its readback covers x only, so make that invariant explicit.
    com_bias = env.simulator._base_com_bias
    if not torch.allclose(com_bias[:, 1:], torch.zeros_like(com_bias[:, 1:]), atol=1e-6, rtol=0.0):
        raise RuntimeError("V4 physics pinning leaked non-zero com_y/com_z")
    return output


def _v4_terrain_hash(terrain_info: Mapping[str, Any], terrain_type: str, level: int, eval_seed: int) -> str:
    # The base hash covers the actual generated full heightfield. Appending the
    # selected tile identity prevents reports from labelling two tiles as one
    # world while retaining a direct cross-model equality check.
    text = f"{terrain_info['terrain_hash']}|{terrain_type}|{int(level)}|{int(eval_seed)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_v4_headroom_cell(root: Path, cfg: Mapping[str, Any], model: Mapping[str, Any], cell: Cell, *, resume: bool) -> None:
    rel = _cell_rel(cell)
    final = root / "raw" / f"{rel}.npz"
    if resume and _v4_artifact_valid(final, cfg, cell):
        return
    scenario = dict(cell.scenario)
    protocol = cfg["protocol"]
    terrain = protocol["terrain"]
    session, terrain_info = build_terrain_session(
        model, cell.seed, int(protocol["replicas_per_cell"]), int(protocol["eval_seed"]),
        fixed_level=int(scenario["terrain_level"]), fixed_type=int(scenario["terrain_type_index"]),
        num_cols=int(terrain["num_cols"]), terrain_proportions=terrain["proportions"],
        num_rows=int(terrain["num_rows"]), measure_heights=bool(terrain["measure_heights"]),
        disable_v3_physics_switch=True,
    )
    selected_checkpoint = model.get("selected_checkpoints", {}).get(int(cell.seed))
    if selected_checkpoint and sha256_file(str(session.checkpoint)) != str(selected_checkpoint["checkpoint_sha256"]):
        raise RuntimeError(
            f"V4 deploy checkpoint drift for {cell.model} seed {cell.seed}: selection recorded "
            f"{selected_checkpoint['checkpoint_sha256']}, runner resolved {sha256_file(str(session.checkpoint))}"
        )
    actual_types = terrain_info["terrain_type"]
    actual_levels = terrain_info["terrain_level"]
    if not np.all(actual_types == int(scenario["terrain_type_index"])):
        raise RuntimeError(f"V4 terrain type pin failed for {cell.name}: got {np.unique(actual_types).tolist()}")
    if not np.all(actual_levels == int(scenario["terrain_level"])):
        raise RuntimeError(f"V4 terrain level pin failed for {cell.name}: got {np.unique(actual_levels).tolist()}")
    actual_type_name = _terrain_type_name(int(scenario["terrain_type_index"]), int(terrain["num_cols"]))
    if actual_type_name != str(scenario["terrain_type"]):
        raise RuntimeError(f"V4 terrain type mapping drift: config={scenario['terrain_type']}, runtime={actual_type_name}")
    physics = _apply_v4_physics(session.env, scenario["physics"])
    vx, vy, yaw = float(scenario["command_vx"]), float(protocol["commands"]["vy"]), float(protocol["commands"]["yaw_rate"])
    commands = np.tile(np.asarray([[vx, vy, yaw]], dtype=np.float32), (session.env.num_envs, 1))
    payload = _base_payload(session, V4_MATRIX_SUITE, cell.name, int(protocol["eval_seed"]),
                            int(protocol["warmup_steps"]), int(protocol["measured_steps"]), cfg)
    physics_signature = json.dumps(scenario["physics"], sort_keys=True, separators=(",", ":"))
    payload.update(
        protocol_kind="v4_headroom_matrix_v1", protocol_fingerprint=_v4_protocol_fingerprint(cfg),
        campaign_fingerprint=_v4_campaign_fingerprint(cfg, root),
        tier=str(scenario["tier"]), physics_tier=str(scenario["tier"]),
        physics_axis=str(scenario["physics_axis"]), physics_dr_axis=str(scenario["physics_dr_axis"]),
        physics_band=str(scenario["physics_band"]), physics_scenario=str(scenario.get("physics_scenario", "")),
        physics_signature=physics_signature, terrain_type_name=actual_type_name,
        terrain_type_index=int(scenario["terrain_type_index"]), terrain_level=int(scenario["terrain_level"]),
        terrain_num_cols=int(terrain["num_cols"]), terrain_num_rows=int(terrain["num_rows"]),
        terrain_hash=_v4_terrain_hash(terrain_info, actual_type_name, int(scenario["terrain_level"]), int(protocol["eval_seed"])),
        terrain_full_hash=str(terrain_info["terrain_hash"]), terrain_type=actual_types, terrain_level_per_env=actual_levels,
        command_vx=vx, command_vy=vy, command_yaw_rate=yaw, command_speed=float(np.hypot(vx, vy)),
        headline_command=True, **physics,
        **_rollout_fixed(session, commands, warmup=int(protocol["warmup_steps"]), steps=int(protocol["measured_steps"])),
    )
    _save_cell(root, rel, payload)


def _run_cell(root: Path, cfg: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]], cell: Cell, *, resume: bool) -> None:
    rel = _cell_rel(cell)
    final = root / "raw" / f"{rel}.npz"
    model = models[cell.model]
    protocol = cfg["protocol"]
    scenario = dict(cell.scenario)
    kind = scenario["kind"]
    if kind == "v4_headroom":
        _run_v4_headroom_cell(root, cfg, model, cell, resume=resume)
        return
    if resume and _artifact_valid(final):
        return
    num_envs, eval_seed = int(protocol["num_envs"]), int(protocol["eval_seed"])
    if kind == "static_id":
        p = protocol["s0"]
        session = build_session(model, cell.seed, num_envs, eval_seed, id_domain_rand=True, disable_v3_physics_switch=True)
        commands = _id_commands(num_envs, eval_seed, p["command_bank"])
        payload = _base_payload(session, "s0", "static_id", eval_seed, int(p["warmup"]), int(p["steps"]), cfg)
        payload.update(s0_variant="static", heading_command=False, yaw_semantics="direct_yaw_rate",
                       command_bank_json=json.dumps(p["command_bank"], sort_keys=True), command_matrix=commands,
                       scenario_hash=hashlib.sha256(commands.tobytes()).hexdigest(),
                       **_rollout_fixed(session, commands, warmup=int(p["warmup"]), steps=int(p["steps"]),
                                         command_bank=p["command_bank"]))
    elif kind == "dynamic_id":
        p, target = protocol["s0"], scenario["target"]
        session = build_session(model, cell.seed, num_envs, eval_seed, disable_v3_physics_switch=True)
        initial = _apply_payload(session.env, 0.0, protocol["payload"], pin_nominal=True)
        commands = _id_commands(num_envs, eval_seed, p["command_bank"])
        payload = _base_payload(session, "s0", "dynamic_id_in_band", eval_seed, int(p["warmup"]), int(p["pre_steps"]) + int(p["post_steps"]), cfg)
        payload.update(s0_variant="dynamic", heading_command=False, yaw_semantics="direct_yaw_rate",
                       command_bank_json=json.dumps(p["command_bank"], sort_keys=True), initial_payload_mass_kg=0.0, initial_payload_com_x_m=0.0,
                       **{f"initial_{key}": value for key, value in initial.items()},
                       **_payload_metadata(target, protocol["payload"]), command_matrix=commands,
                       **_rollout_switch(session, commands, warmup=int(p["warmup"]), pre_steps=int(p["pre_steps"]), post_steps=int(p["post_steps"]),
                                         target_mass=float(target["mass_kg"]), payload_cfg=protocol["payload"], command_bank=p["command_bank"]))
    elif kind == "static_payload":
        p, target, command = protocol["s1"], scenario["payload"], scenario["command"]
        session = build_session(model, cell.seed, num_envs, eval_seed, disable_v3_physics_switch=True)
        physics = _apply_payload(session.env, float(target["mass_kg"]), protocol["payload"], pin_nominal=True)
        commands = _command_array(command, num_envs)
        payload = _base_payload(session, "s1", cell.name, eval_seed, int(p["warmup"]), int(p["steps"]), cfg)
        payload.update(**_payload_metadata(target, protocol["payload"]), command_name=_command_name(command),
                       command_mode=str(command["mode"]), command_vector=np.asarray(command["vector"], dtype=np.float32),
                       command_speed=float(np.linalg.norm(np.asarray(command["vector"], dtype=float)[:2])),
                       headline_command=bool(command.get("headline", True)),
                       diagnostic_kind=str(command.get("diagnostic_kind", "")), **physics,
                       **_rollout_fixed(session, commands, warmup=int(p["warmup"]), steps=int(p["steps"]),
                                         command_bank=protocol["s0"]["command_bank"]))
    elif kind == "payload_switch":
        p, target, command = protocol["s2"], scenario["switch"], scenario["command"]
        session = build_session(model, cell.seed, num_envs, eval_seed, disable_v3_physics_switch=True)
        initial = _apply_payload(session.env, 0.0, protocol["payload"], pin_nominal=True)
        commands = _command_array(command, num_envs)
        payload = _base_payload(session, "s2", cell.name, eval_seed, int(p["warmup"]), int(p["pre_steps"]) + int(p["post_steps"]), cfg)
        payload.update(switch_name=str(target["name"]), switch_tier=str(target["tier"]), command_name=_command_name(command),
                       command_mode=str(command["mode"]), command_vector=np.asarray(command["vector"], dtype=np.float32),
                       command_speed=float(np.linalg.norm(np.asarray(command["vector"], dtype=float)[:2])),
                       headline_command=bool(command.get("headline", True)),
                       diagnostic_kind=str(command.get("diagnostic_kind", "")),
                       **_payload_metadata(target, protocol["payload"]),
                       **{f"initial_{key}": value for key, value in initial.items()},
                       **_rollout_switch(session, commands, warmup=int(p["warmup"]), pre_steps=int(p["pre_steps"]), post_steps=int(p["post_steps"]),
                                         target_mass=float(target["mass_kg"]), payload_cfg=protocol["payload"], command_bank=protocol["s0"]["command_bank"]))
    elif kind == "kick":
        p = protocol["s3"]
        session = build_session(model, cell.seed, num_envs, eval_seed, disable_v3_physics_switch=True)
        _set_nominal(session.env)
        commands = np.tile(np.asarray([[1., 0., 0.]], dtype=np.float32), (num_envs, 1))
        payload = _base_payload(session, "s3", "kick_calibration", eval_seed, int(p["warmup"]), int(p["kick_pre_steps"]) + int(p["kick_post_steps"]), cfg)
        payload.update(s3_kind="kick", command_mode="forward", command_speed=1.0,
                       **_rollout_kick(session, commands, warmup=int(p["warmup"]), pre_steps=int(p["kick_pre_steps"]),
                                       post_steps=int(p["kick_post_steps"]), dvy=float(scenario["kick_dvy"])))
    elif kind == "terrain":
        p, cm = protocol["s3"], float(scenario["terrain_cm"])
        hscale = float(p["terrain_horizontal_scale"])
        raw = _terrain_raw(cm / 100.0, int(eval_seed), hscale)
        session = build_session(model, cell.seed, num_envs, eval_seed, terrain_raw=raw, terrain_horizontal_scale=hscale,
                                disable_v3_physics_switch=True)
        _set_nominal(session.env)
        commands = np.tile(np.asarray([[1., 0., 0.]], dtype=np.float32), (num_envs, 1))
        payload = _base_payload(session, "s3", "terrain_raw", eval_seed, int(p["warmup"]), int(p["terrain_steps"]), cfg)
        payload.update(s3_kind="terrain", terrain_cm=cm, terrain_hash=_terrain_hash(raw), terrain_max_abs_m=cm / 100.0,
                       command_mode="forward", command_speed=1.0,
                       **_rollout_fixed(session, commands, warmup=int(p["warmup"]), steps=int(p["terrain_steps"])))
    elif kind in ("terrain_id", "terrain_severity", "terrain_payload"):
        t = protocol["terrain"]
        level = int(scenario["level"])
        vx = float(t["command"]["vx"])
        session, terrain_info = build_terrain_session(
            model, cell.seed, num_envs, eval_seed,
            fixed_level=level, num_cols=int(t["num_cols"]),
            terrain_proportions=t["terrain_proportions"], num_rows=t.get("num_rows"),
            measure_heights=bool(t.get("measure_heights", True)),
            disable_v3_physics_switch=True,
        )
        _set_nominal(session.env)
        extra: Dict[str, Any] = {}
        if kind == "terrain_payload":
            payload_item = scenario["payload"]
            physics = _apply_payload(session.env, float(payload_item["mass_kg"]), protocol["payload"], pin_nominal=False)
            extra.update(_payload_metadata(payload_item, protocol["payload"]), **physics)
        commands = np.tile(np.asarray([[vx, 0., 0.]], dtype=np.float32), (num_envs, 1))
        scenario_name = {"terrain_id": "terrain_id", "terrain_severity": "terrain_severity",
                         "terrain_payload": "terrain_payload"}[kind]
        payload = _base_payload(session, cell.suite, scenario_name, eval_seed, int(t["warmup"]), int(t["steps"]), cfg)
        payload.update(terrain_kind=kind, terrain_level=level, terrain_hash=terrain_info["terrain_hash"],
                       terrain_num_cols=terrain_info["num_cols"], terrain_num_rows=terrain_info["num_rows"],
                       terrain_type=terrain_info["terrain_type"], terrain_level_per_env=terrain_info["terrain_level"],
                       command_mode="forward", command_speed=vx, command_vx=vx,
                       physics_tier=(str(extra.get("payload_tier", "")) or "nominal"),
                       physics_signature=str(extra.get("payload_name", "")), headline_command=True, **extra,
                       **_rollout_fixed(session, commands, warmup=int(t["warmup"]), steps=int(t["steps"])))
    else:
        raise ValueError(f"unknown V3 eval cell kind {kind!r}")
    _save_cell(root, rel, payload)


def _write_manifest(cfg: Mapping[str, Any], cells: Sequence[Cell]) -> None:
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in cells:
        rel = _cell_rel(cell) + ".npz"
        path = root / "raw" / rel
        rows.append({"path": f"raw/{rel}", "complete": _artifact_valid(path),
                     "sha256": sha256_file(str(path)) if _artifact_valid(path) else None})
    payload = {"campaign": cfg["campaign"], "generated_at": datetime.now(timezone.utc).isoformat(),
               "expected_raw_artifacts": len(rows), "completed_raw_artifacts": sum(row["complete"] for row in rows),
               "complete": all(row["complete"] for row in rows), "cells": rows,
               "note": "Expected artifacts cover only policy-seeds currently selected in run_selection.json; pending policies are intentionally not missing cells."}
    (root / "manifest.json").write_text(_json(payload), encoding="utf-8")
    (root / "campaign.yaml").write_text(yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8")


def _selected_cells(cfg: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]], suite: str, *, include_diagnostics: bool = False) -> List[Cell]:
    allowed = {(label, int(seed)) for label, model in models.items() for seed in model["training_seeds"]}
    return [cell for cell in _planned_cells(cfg, suite, include_diagnostics=include_diagnostics) if (cell.model, cell.seed) in allowed]


def cmd_plan(cfg: Mapping[str, Any], suite: str, *, labels: Sequence[str] | None = None, include_diagnostics: bool = False) -> int:
    all_cells = _planned_cells(cfg, suite, include_diagnostics=include_diagnostics)
    root = artifact_root(cfg)
    selected = _selected_models(cfg, root, labels) if _selection_path(root).is_file() else {}
    cells = _selected_cells(cfg, selected, suite, include_diagnostics=include_diagnostics) if selected else all_cells
    by_suite: Dict[str, int] = defaultdict(int)
    for cell in cells:
        by_suite[cell.suite] += 1
    num_envs_per_cell = cfg["protocol"].get("num_envs", cfg["protocol"].get("replicas_per_cell"))
    print(_json({"campaign": cfg["campaign"], "suite": suite, "process_cells": len(cells), "configured_cells": len(all_cells),
                 "selected_policy_seeds": {label: model["training_seeds"] for label, model in selected.items()}, "by_suite": dict(by_suite),
                 "num_envs_per_cell": num_envs_per_cell, "artifact_root": str(artifact_root(cfg)),
                 "preview": [f"{cell.suite}:{cell.model}:seed{cell.seed}:{cell.name}" for cell in cells[:12]]}))
    return 0


def cmd_run(cfg: Mapping[str, Any], suite: str, *, resume: bool, shard: Tuple[int, int], labels: Sequence[str] | None = None,
            include_diagnostics: bool = False) -> int:
    from legged_gym import SIMULATOR, gs
    if SIMULATOR != "genesis":
        raise RuntimeError(f"SIMULATOR={SIMULATOR!r}; v3_eval needs genesis")
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    models = _selected_models(cfg, root, labels)
    if not models:
        print("[v3_eval] no finalised policy-seed is selected yet; nothing to run")
        _write_manifest(cfg, [])
        return 0
    gs.init(backend=gs.gpu, logging_level="warning")
    selected_cells = _selected_cells(cfg, models, suite, include_diagnostics=include_diagnostics)
    assigned = shard_items(selected_cells, shard)
    print(f"[v3_eval] run suite={suite} shard={shard[0]}/{shard[1]} cells={len(assigned)} resume={resume}")
    for index, cell in enumerate(assigned, start=1):
        print(f"[v3_eval] cell {index}/{len(assigned)} {cell.suite}:{cell.model}:seed{cell.seed}:{cell.name}", flush=True)
        try:
            _run_cell(root, cfg, models, cell, resume=resume)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    _write_manifest(cfg, _selected_cells(cfg, _selected_models(cfg, root), "all", include_diagnostics=include_diagnostics))
    return 0


def _scalar(z, key: str, default: Any = None) -> Any:
    if key not in z:
        return default
    value = z[key]
    if np.asarray(value).shape == ():
        return np.asarray(value).item()
    return value


def _mean(z, key: str) -> float:
    return float(np.mean(np.asarray(z[key], dtype=float)))


def _raw_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((root / "raw").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            suite = str(_scalar(z, "suite"))
            if suite in TERRAIN_SUITES:
                # Terrain cells pack several terrain types into one env dimension;
                # they are expanded per (type x level) by ``_terrain_raw_rows``.
                continue
            scenario = str(_scalar(z, "scenario"))
            error_key = "post_switch_tracking_error" if suite == "s2" else "tracking_lin"
            row = {
                "path": str(path.relative_to(root)), "suite": suite, "scenario": scenario,
                "model": str(_scalar(z, "model")), "training_seed": int(_scalar(z, "training_seed")),
                "tracking_error": _mean(z, error_key), "tracking_lin": _mean(z, "tracking_lin"),
                "fall_rate": _mean(z, "post_switch_fall_rate" if suite == "s2" and "post_switch_fall_rate" in z else "fall_rate"),
                "achieved_speed": _mean(z, "achieved_speed") if "achieved_speed" in z else float("nan"),
                "achieved_speed_ratio": _mean(z, "achieved_speed_ratio") if "achieved_speed_ratio" in z else float("nan"),
                "return_per_step": _mean(z, "return_per_step"),
                # Append-only: historical V3 artifacts remain readable, while
                # newly written V3/V4 fixed rollouts surface SPNTE alongside
                # their unchanged tracking/fall scorecard inputs.
                "spnte_lin": _mean(z, "spnte_lin") if "spnte_lin" in z else float("nan"),
                "spnte_yaw": _mean(z, "spnte_yaw") if "spnte_yaw" in z else float("nan"),
                "spnte_v_scale": float(_scalar(z, "spnte_v_scale", float("nan"))),
                "command_speed": float(_scalar(z, "command_speed", 0.0)),
                "headline_command": bool(_scalar(z, "headline_command", True)),
                "diagnostic_kind": str(_scalar(z, "diagnostic_kind", "")),
                "payload_tier": str(_scalar(z, "payload_tier", _scalar(z, "switch_tier", ""))),
                "payload_name": str(_scalar(z, "payload_name", _scalar(z, "switch_name", ""))),
                "command_name": str(_scalar(z, "command_name", "")),
                # Keep enough provenance to form a V4 world key even for the
                # older flat artifacts.  New V4 writers persist command_vx and
                # physics_tier explicitly; the fallbacks make the reader
                # append-only for V3/V4 terrain artifacts written before that
                # schema existed.
                "command_vx": float(_scalar(z, "command_vx", _scalar(z, "command_speed", 0.0))),
                "physics_tier": str(_scalar(z, "physics_tier", _scalar(z, "payload_tier", "nominal"))) or "nominal",
                "physics_signature": str(_scalar(z, "physics_signature", _scalar(z, "payload_name", ""))),
                "eval_seed": int(_scalar(z, "eval_seed", -1)),
                "physics_axis": str(_scalar(z, "physics_axis", "")),
                "physics_band": str(_scalar(z, "physics_band", "")),
                "protocol_kind": str(_scalar(z, "protocol_kind", "")),
                "protocol_fingerprint": str(_scalar(z, "protocol_fingerprint", "")),
                "campaign_fingerprint": str(_scalar(z, "campaign_fingerprint", "")),
                "terrain_type_name": str(_scalar(z, "terrain_type_name", "")),
                "terrain_level": int(_scalar(z, "terrain_level", 0)),
                "terrain_hash": str(_scalar(z, "terrain_hash", "")),
            }
            if suite == "s2":
                row["post_switch_tracking_error"] = _mean(z, "post_switch_tracking_error")
                row["post_switch_tracking_yaw"] = _mean(z, "post_switch_tracking_yaw")
            rows.append(row)
    return rows


def _followup_reference_rows(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Import only frozen MLP/Oracle references from the discovery artifact root.

    Adaptation policies must never decide their own evaluation population.  The
    manifest pins that population and hashes the two discovery summary inputs;
    this loader verifies both before allowing the old reference artifacts into
    the follow-up scorecard.
    """
    manifest = _v4_followup_manifest(cfg)
    if manifest is None:
        return []
    source = manifest["source"]
    root_value = str(source["artifact_root"])
    source_root = Path(root_value) if Path(root_value).is_absolute() else _root() / root_value
    headline = source_root / "tables" / "headline.json"
    scorecard_worlds = source_root / "tables" / "scorecard_worlds.csv"
    raw_table = source_root / "tables" / "raw_cells.csv"
    run_selection = source_root / "run_selection.json"
    for path, expected, name in ((headline, source["headline_sha256"], "headline"),
                                 (scorecard_worlds, source["scorecard_worlds_sha256"], "scorecard worlds"),
                                 (raw_table, source["raw_cells_sha256"], "raw-cell table"),
                                 (run_selection, source["run_selection_sha256"], "checkpoint selection")):
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        if actual != str(expected):
            raise RuntimeError(f"frozen follow-up {name} hash changed; regenerate the manifest instead of mixing discovery results")
    allowed = {
        (json.dumps(world["identity"], sort_keys=True, separators=(",", ":")), int(seed))
        for world in manifest["worlds"] for seed in world["training_seeds"]
    }
    score = cfg["scorecard"]
    references = {str(score["baseline_label"]), str(score["oracle_label"])}
    expected_fingerprint = str(source["protocol_fingerprint"])
    expected_campaign = str(source["campaign_fingerprint"])
    rows = []
    for row in _raw_rows(source_root):
        if (str(row["model"]) not in references or str(row.get("protocol_fingerprint", "")) != expected_fingerprint
                or str(row.get("campaign_fingerprint", "")) != expected_campaign):
            continue
        key = (json.dumps(_v4_world_identity(row), sort_keys=True, separators=(",", ":")), int(row["training_seed"]))
        if key in allowed:
            world = next(item for item in manifest["worlds"]
                         if json.dumps(item["identity"], sort_keys=True, separators=(",", ":")) == key[0])
            reference = world["references"][str(row["training_seed"])]
            if str(row.get("terrain_hash", "")) != str(reference["terrain_hash"]):
                raise RuntimeError("frozen follow-up terrain hash drift; regenerate the manifest")
            artifact = source_root / str(row["path"])
            expected_artifact = reference["artifacts"][str(row["model"])]
            actual_sha = hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.is_file() else ""
            if actual_sha != str(expected_artifact["npz_sha256"]):
                raise RuntimeError("frozen follow-up reference artifact drift; regenerate the manifest")
            for field in ("tracking_error", "fall_rate", "achieved_speed_ratio"):
                if not np.isclose(float(row[field]), float(expected_artifact[field]), rtol=0.0, atol=1e-12):
                    raise RuntimeError(f"frozen follow-up reference {field} drift; regenerate the manifest")
            copied = dict(row)
            copied["frozen_discovery_reference"] = True
            rows.append(copied)
    expected_count = len(allowed) * len(references)
    if len(rows) != expected_count:
        raise RuntimeError(f"frozen follow-up references incomplete: expected {expected_count}, found {len(rows)}")
    return rows


def _terrain_raw_rows(root: Path) -> List[Dict[str, Any]]:
    """Expand each terrain cell into one raw row per terrain type present.

    A terrain artifact evaluates every env at one pinned difficulty ``level``
    but splits the env dimension across ``num_cols`` types.  Each unique type is
    reduced over its replica envs here, before any cross-seed aggregation, so a
    seed with more replicas never outweighs another.  The emitted ``scenario``
    encodes both the type and the controlled second axis (level for t1, payload
    for t2), which is exactly the key ``cmd_aggregate`` groups baseline/oracle/
    method on.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted((root / "raw").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            suite = str(_scalar(z, "suite"))
            if suite not in TERRAIN_SUITES:
                continue
            model = str(_scalar(z, "model"))
            training_seed = int(_scalar(z, "training_seed"))
            level = int(_scalar(z, "terrain_level", 0))
            num_cols = int(_scalar(z, "terrain_num_cols", 5))
            terrain_hash = str(_scalar(z, "terrain_hash", ""))
            payload_tier = str(_scalar(z, "payload_tier", ""))
            payload_name = str(_scalar(z, "payload_name", ""))
            command_speed = float(_scalar(z, "command_speed", 0.0))
            command_vx = float(_scalar(z, "command_vx", command_speed))
            physics_tier = str(_scalar(z, "physics_tier", payload_tier)) or "nominal"
            physics_signature = str(_scalar(z, "physics_signature", payload_name))
            terrain_type = np.asarray(z["terrain_type"]).astype(int)
            tracking = np.asarray(z["tracking_lin"], dtype=float)
            fall = np.asarray(z["fall_rate"], dtype=float)
            ratio = np.asarray(z["achieved_speed_ratio"], dtype=float) if "achieved_speed_ratio" in z \
                else np.full(tracking.shape, np.nan)
            speed = np.asarray(z["achieved_speed"], dtype=float) if "achieved_speed" in z \
                else np.full(tracking.shape, np.nan)
            per_step = np.asarray(z["return_per_step"], dtype=float)
            spnte_lin = np.asarray(z["spnte_lin"], dtype=float) if "spnte_lin" in z else np.full(tracking.shape, np.nan)
            spnte_yaw = np.asarray(z["spnte_yaw"], dtype=float) if "spnte_yaw" in z else np.full(tracking.shape, np.nan)
            spnte_v_scale = float(_scalar(z, "spnte_v_scale", float("nan")))
            for type_index in sorted(np.unique(terrain_type).tolist()):
                mask = terrain_type == type_index
                type_name = _terrain_type_name(type_index, num_cols)
                if suite == "t2":
                    scenario = f"terrain_type_{type_name}_payload_{payload_name}"
                else:
                    scenario = f"terrain_type_{type_name}_level_{level}"
                rows.append({
                    "path": str(path.relative_to(root)), "suite": suite, "scenario": scenario,
                    "model": model, "training_seed": training_seed,
                    "tracking_error": float(np.mean(tracking[mask])), "tracking_lin": float(np.mean(tracking[mask])),
                    "fall_rate": float(np.mean(fall[mask])),
                    "achieved_speed": float(np.mean(speed[mask])),
                    "achieved_speed_ratio": float(np.mean(ratio[mask])),
                    "return_per_step": float(np.mean(per_step[mask])),
                    "spnte_lin": float(np.mean(spnte_lin[mask])),
                    "spnte_yaw": float(np.mean(spnte_yaw[mask])),
                    "spnte_v_scale": spnte_v_scale,
                    "command_speed": command_speed, "command_vx": command_vx,
                    "physics_tier": physics_tier, "physics_signature": physics_signature,
                    "headline_command": True, "diagnostic_kind": "",
                    "payload_tier": payload_tier, "payload_name": payload_name, "command_name": "",
                    "terrain_type": int(type_index), "terrain_type_name": type_name, "terrain_level": level,
                    "terrain_hash": terrain_hash, "eval_seed": int(_scalar(z, "eval_seed", -1)),
                    "num_replicas": int(mask.sum()),
                })
    return rows


def score_cell(
    baseline: Mapping[str, float], oracle_candidates: Sequence[Mapping[str, float]], method: Mapping[str, float], score_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pure scorecard rule used by aggregation and CPU-only contract tests."""
    fall_gate = float(score_cfg["fall_gate_pp"])
    speed_gate = float(score_cfg["achieved_speed_ratio"])
    stable = [row for row in oracle_candidates if float(row["fall_rate"]) <= float(baseline["fall_rate"]) + fall_gate]
    speed_ok = [row for row in stable if float(row["achieved_speed_ratio"]) >= speed_gate]
    if not stable:
        return {"headline_status": "oracle_unstable", "headline_include": False, "reasons": "oracle_fall_gate"}
    if not speed_ok:
        return {"headline_status": "oracle_speed_saturated", "headline_include": False, "reasons": "oracle_achieved_speed"}
    oracle = min(speed_ok, key=lambda row: float(row["tracking_error"]))
    numerator = float(baseline["tracking_error"]) - float(method["tracking_error"])
    denominator = float(baseline["tracking_error"]) - float(oracle["tracking_error"])
    rel = denominator / max(abs(float(baseline["tracking_error"])), 1e-12)
    if denominator < float(score_cfg["absolute_headroom"]) or rel < float(score_cfg["relative_headroom"]):
        return {"headline_status": "no_oracle_headroom", "headline_include": False, "reasons": "oracle_headroom",
                "oracle_seed": int(oracle["training_seed"]), "oracle_error": float(oracle["tracking_error"]),
                "oracle_headroom_relative": rel, "oracle_headroom_absolute": denominator}
    raw = numerator / denominator
    clipped = float(np.clip(raw, -0.5, 1.5))
    if float(method["fall_rate"]) > float(baseline["fall_rate"]) + fall_gate:
        return {"headline_status": "method_fall_gated", "headline_include": True, "reasons": "method_fall_gate",
                "raw_gap_closed": raw, "headline_gap_closed": 0.0, "oracle_seed": int(oracle["training_seed"]),
                "oracle_error": float(oracle["tracking_error"]), "oracle_headroom_relative": rel,
                "oracle_headroom_absolute": denominator}
    return {"headline_status": "eligible", "headline_include": True, "reasons": "",
            "raw_gap_closed": raw, "headline_gap_closed": clipped, "oracle_seed": int(oracle["training_seed"]),
            "oracle_error": float(oracle["tracking_error"]), "oracle_headroom_relative": rel,
            "oracle_headroom_absolute": denominator}


def _error_view(row: Mapping[str, Any], error_field: str) -> Dict[str, Any]:
    """Row view for :func:`score_cell`, closing the gap on ``error_field``."""
    return {**row, "tracking_error": float(row[error_field])}


def score_cell_on(
    error_field: str, baseline: Mapping[str, float], oracle_candidates: Sequence[Mapping[str, float]],
    method: Mapping[str, float], score_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Score one cell by closing the gap on an arbitrary lower-is-better ``error_field``.

    A thin remap in front of :func:`score_cell`: every gate (oracle stability
    via fall_rate, the speed gate, the headroom gate, and the method
    fall-gate) is reused byte-for-byte from the single scorecard rule -- only
    the field being closed changes.  ``error_field="tracking_error"`` is a
    no-op remap and is exactly the historical scorecard; SPNTE scoring calls
    this with ``error_field="spnte_lin"``.
    """
    if error_field == "tracking_error":
        return score_cell(baseline, oracle_candidates, method, score_cfg)
    return score_cell(
        _error_view(baseline, error_field),
        [_error_view(row, error_field) for row in oracle_candidates],
        _error_view(method, error_field),
        score_cfg,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _scope(row: Mapping[str, Any]) -> str:
    if row["suite"] == "s1":
        return "GapClosed_static"
    if row["suite"] == "s2":
        return f"GapClosed_dynamic_{row['payload_tier']}"
    if row["suite"] == "t1":
        return f"GapClosed_terrain_{row['terrain_type_name']}"
    if row["suite"] == "t2":
        return f"GapClosed_terrain_payload_{row['payload_tier']}"
    raise ValueError(f"no gap-closed scope for {row['suite']}")


def _uses_v4_world_headroom(cfg: Mapping[str, Any]) -> bool:
    """Whether aggregation must use the V4 per-world contract.

    V3's historical scorecard intentionally pools its reference seeds for
    backwards compatibility.  That is not scientifically valid for the V4
    terrain matrix: a training seed is a replicate, not an additional
    parallel-env sample.  The explicit runner interface is the preferred
    switch; the campaign-name fallback keeps the existing V4 terrain campaign
    on the stricter path without requiring an artifact migration.
    """
    interface = cfg.get(
        "runner_interface",
        cfg.get("execution", {}).get("runner_interface", cfg.get("protocol", {}).get("runner_interface", "")),
    )
    return interface == "v4_headroom_matrix_v1" or str(cfg.get("campaign", "")).startswith("v4_")


def _v4_world_identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Physical world identity, deliberately excluding ``training_seed``.

    A training seed is retained alongside this key during scoring and only
    aggregated after a per-seed, per-world comparison.  ``physics_signature``
    prevents two distinct settings in the same broad tier from being merged.
    """
    return {
        "suite": str(row.get("suite", "")),
        "terrain_type": str(row.get("terrain_type_name", "")),
        "terrain_level": int(row.get("terrain_level", 0)),
        "command_vx": float(row.get("command_vx", row.get("command_speed", 0.0))),
        "physics_tier": str(row.get("physics_tier", row.get("payload_tier", "nominal"))) or "nominal",
        "physics_signature": str(row.get("physics_signature", row.get("payload_name", ""))),
        "eval_seed": int(row.get("eval_seed", -1)),
        "physics_axis": str(row.get("physics_axis", "")),
        "physics_band": str(row.get("physics_band", "")),
        # ``scenario`` remains part of the key for legacy artifacts that did
        # not yet emit a complete physics signature.
        "scenario": str(row.get("scenario", "")),
    }


def _v4_world_key(row: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(_v4_world_identity(row).items()))


def _v4_scope(row: Mapping[str, Any], score_cfg: Mapping[str, Any]) -> str:
    """Keep secondary stress worlds out of the declared primary headline."""
    tier = str(row.get("physics_tier", row.get("payload_tier", "nominal"))) or "nominal"
    if row.get("suite") == V4_MATRIX_SUITE:
        if tier == str(score_cfg.get("headline_tier", tier)):
            return f"GapClosed_primary_{row.get('physics_axis', 'unknown')}_{row.get('physics_band', 'unknown')}"
        return f"GapClosed_secondary_{tier}_{row.get('physics_band', 'unknown')}"
    if tier == str(score_cfg.get("headline_tier", tier)):
        return _scope(row)
    return f"GapClosed_secondary_{tier}_{row.get('terrain_type_name', 'flat')}"


def classify_v4_tracking_world(mlp: Mapping[str, Any], oracle: Mapping[str, Any], score_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify a V4 world using *only* the MLP and Oracle references.

    A survival world is useful for survival reporting but has no tracking
    GapClosed value. Gates are explicit scorecard contract fields and therefore
    part of the V4 protocol fingerprint.
    """
    mlp_fall = float(mlp["fall_rate"])
    oracle_fall = float(oracle["fall_rate"])
    oracle_speed = float(oracle["achieved_speed_ratio"])
    absolute_headroom = float(mlp["tracking_error"]) - float(oracle["tracking_error"])
    reasons: List[str] = []
    fall_gate = float(score_cfg["fall_gate_pp"])
    speed_gate = float(score_cfg["achieved_speed_ratio"])
    absolute_gate = float(score_cfg["absolute_headroom"])
    relative_gate = float(score_cfg["relative_headroom"])
    relative_headroom = absolute_headroom / float(mlp["tracking_error"]) if float(mlp["tracking_error"]) > 0 else float("nan")
    if not np.isfinite(mlp_fall) or mlp_fall > fall_gate:
        reasons.append(f"mlp_fall_rate_gt_{fall_gate:.2f}")
    if not np.isfinite(oracle_fall) or oracle_fall > fall_gate:
        reasons.append(f"oracle_fall_rate_gt_{fall_gate:.2f}")
    if reasons:
        return {
            "world_classification": "survival",
            "survival_status": "survival_world",
            "tracking_status": "excluded_survival_world",
            "tracking_include": False,
            "tracking_exclusion_reasons": ";".join(reasons),
            "absolute_tracking_headroom": absolute_headroom, "relative_tracking_headroom": relative_headroom,
        }
    if not np.isfinite(oracle_speed) or oracle_speed < speed_gate:
        reasons.append(f"oracle_achieved_speed_ratio_lt_{speed_gate:.2f}")
    if not np.isfinite(absolute_headroom) or absolute_headroom < absolute_gate:
        reasons.append(f"absolute_tracking_headroom_lt_{absolute_gate:.2f}")
    if not np.isfinite(relative_headroom) or relative_headroom < relative_gate:
        reasons.append(f"relative_tracking_headroom_lt_{relative_gate:.2f}")
    if reasons:
        return {
            "world_classification": "tracking_excluded",
            "survival_status": "survival_eligible",
            "tracking_status": "excluded_tracking_gate",
            "tracking_include": False,
            "tracking_exclusion_reasons": ";".join(reasons),
            "absolute_tracking_headroom": absolute_headroom, "relative_tracking_headroom": relative_headroom,
        }
    return {
        "world_classification": "tracking",
        "survival_status": "survival_eligible",
        "tracking_status": "eligible",
        "tracking_include": True,
        "tracking_exclusion_reasons": "",
        "absolute_tracking_headroom": absolute_headroom, "relative_tracking_headroom": relative_headroom,
    }


def _v4_normalized_headroom_payload(cfg: Mapping[str, Any], world_rows: Sequence[Mapping[str, Any]], score_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Adapt scorecard rows to ``headroom_report.py``'s deliberately small schema."""
    score_cfg = cfg["scorecard"]
    primary_tier = str(score_cfg.get("headline_tier", ""))
    by_world: Dict[Tuple[Tuple[str, Any], ...], List[Mapping[str, Any]]] = defaultdict(list)
    identity_fields = ("suite", "terrain_type", "terrain_level", "command_vx", "physics_tier", "physics_signature", "eval_seed", "physics_axis", "physics_band")
    for row in world_rows:
        by_world[tuple((field, row.get(field, "")) for field in identity_fields)].append(row)
    scores_by_world: Dict[Tuple[Tuple[str, Any], ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        scores_by_world[tuple((field, row.get(field, "")) for field in identity_fields)].append(row)

    worlds = []
    for key, rows in sorted(by_world.items()):
        identity = dict(key)
        def median(field: str) -> float | None:
            values = [float(row[field]) for row in rows if np.isfinite(float(row.get(field, float("nan"))))]
            return float(np.median(values)) if values else None
        mlp_error, oracle_error = median("mlp_tracking_error"), median("oracle_tracking_error")
        # The renderer requires reference tracking values. An incomplete
        # append-only campaign is still fully retained in scorecard_worlds.csv;
        # omit it here rather than serialize NaN into a renderer contract.
        if mlp_error is None or oracle_error is None:
            continue
        all_tracking = all(bool(row["tracking_include"]) for row in rows)
        reasons = sorted({
            reason
            for row in rows
            for reason in str(row.get("tracking_exclusion_reasons", "")).split(";")
            if reason
        })
        if str(identity["physics_tier"]) != primary_tier:
            all_tracking = False
            if "secondary_tier_not_primary_headline" not in reasons:
                reasons.append("secondary_tier_not_primary_headline")
        band = str(identity["physics_band"]).lower()
        id_ood = "ID" if band.startswith("id") or band in {"", "nominal", "in_band"} else "OOD"
        terrain_label = {
            "slope": "Eğim",
            "random_uniform": "Rastgele pürüz",
            "stairs_down": "İnen merdiven",
            "stairs_up": "Çıkan merdiven",
            "discrete": "Ayrık engeller",
        }.get(str(identity["terrain_type"]), str(identity["terrain_type"]))
        axis_label = {
            "mass_kg": "kütle",
            "com_x_m": "CoM-x",
            "friction": "sürtünme",
            "combined": "birleşik stress",
        }.get(str(identity["physics_axis"]), str(identity["physics_axis"]))
        try:
            physics = json.loads(str(identity["physics_signature"]))
        except json.JSONDecodeError:
            physics = None
        axis = str(identity["physics_axis"])
        if not isinstance(physics, Mapping):
            physical_label = f"{axis_label} = {identity['physics_signature']}"
        elif axis == "mass_kg":
            physical_label = f"kütle = {float(physics['mass_kg']):+g} kg"
        elif axis == "com_x_m":
            physical_label = f"CoM-x = {float(physics['com_x_m']):+g} m"
        elif axis == "friction":
            physical_label = f"sürtünme = {float(physics['friction']):g}"
        else:
            physical_label = (f"kütle = {float(physics['mass_kg']):+g} kg · "
                              f"CoM-x = {float(physics['com_x_m']):+g} m · μ = {float(physics['friction']):g}")
        name = (f"{terrain_label} · L{identity['terrain_level']} · {float(identity['command_vx']):g} m/s · "
                f"{physical_label} · {str(identity['physics_band']).upper()}")
        tracking = {"MLP": mlp_error, "Oracle": oracle_error}
        fall = {"MLP": median("mlp_fall_rate"), "Oracle": median("oracle_fall_rate")}
        speed = {"MLP": None, "Oracle": median("oracle_achieved_speed_ratio")}
        consistency: Dict[str, Dict[str, int]] = {}
        gap_closed: Dict[str, float | None] = {}
        method_note: Dict[str, str] = {}
        for label, report_label in (("DreamWaQ", "DreamWaQ"), ("HIM-fixed", "HIM"), ("HIM", "HIM")):
            method_rows = [row for row in scores_by_world.get(key, []) if row.get("model") == label]
            if not method_rows:
                continue
            values = [float(row["method_tracking_error"]) for row in method_rows]
            tracking[report_label] = float(np.median(values))
            fall[report_label] = float(np.median([float(row["method_fall_rate"]) for row in method_rows]))
            speed[report_label] = float(np.median([float(row["method_achieved_speed_ratio"]) for row in method_rows]))
            included = [row for row in method_rows if row.get("headline_include")]
            consistency[report_label] = {"better_seeds": sum(float(row.get("raw_gap_closed", 0.0)) > 0.0 for row in included),
                                         "total_seeds": len({int(row["training_seed"]) for row in method_rows})}
            if any(str(row.get("headline_status")) == "method_fall_gated" for row in method_rows):
                gap_closed[report_label] = None
                method_note[report_label] = "survival-gated"
            else:
                values = [float(row["headline_gap_closed"]) for row in included if "headline_gap_closed" in row]
                gap_closed[report_label] = float(np.median(values)) if values else None
        worlds.append({"world": name, "id_ood": id_ood, "include": all_tracking,
                       "exclusion_reason": ";".join(reasons), "tracking_error": tracking,
                       "fall_rate": fall, "achieved_speed_ratio": speed, "seed_consistency": consistency,
                       "gap_closed": gap_closed, "method_note": method_note})
    seed_count = len({int(row["training_seed"]) for row in world_rows})
    return {"experiment": {"name": str(cfg["campaign"]),
                             "contract": "Eşlenmiş MLP–Oracle; sabit terrain, komut ve fizik kimliği.",
                             "seed_count": max(seed_count, 1),
                             "tracking_metric": "Ortalama doğrusal hız takip hatası (m/s)",
                             "limitations": ["Birleşik stress dünyaları ana tracking sonucuna dahil edilmez.",
                                             "Tracking uygunluğu yalnızca MLP ve Oracle ile belirlenir."]},
            "worlds": worlds}


def _aggregate_v4_world_headroom(cfg: Mapping[str, Any], raw: Sequence[Mapping[str, Any]], diagnostic_raw: Sequence[Mapping[str, Any]]) -> int:
    """Aggregate V4 headroom without pooling worlds or training seeds.

    The reference selection is intentionally made before any adaptation-method
    row is read.  Therefore DreamWaQ/HIM (or any later method) cannot turn a
    survival world into a tracking world, nor select a more favorable oracle.
    """
    root = artifact_root(cfg)
    score_cfg = cfg["scorecard"]
    baseline_label = str(score_cfg["baseline_label"])
    oracle_label = str(score_cfg["oracle_label"])
    scored = [row for row in raw if row["suite"] in (*SCORED_SUITES, V4_MATRIX_SUITE) and bool(row["headline_command"])]
    if _is_v4_headroom_matrix(cfg):
        expected_fingerprint = _v4_protocol_fingerprint(cfg)
        expected_campaign = _v4_campaign_fingerprint(cfg, root)
        for row in scored:
            if row["suite"] != V4_MATRIX_SUITE:
                continue
            if row.get("protocol_kind") != "v4_headroom_matrix_v1" or row.get("protocol_fingerprint") != expected_fingerprint:
                raise RuntimeError(
                    "V4 aggregate found an artifact from a different protocol fingerprint; "
                    "start a new artifact root instead of mixing smoke/full or changed matrix cells"
                )
            if not row.get("frozen_discovery_reference") and row.get("campaign_fingerprint") != expected_campaign:
                raise RuntimeError("V4 aggregate found an artifact from a different campaign fingerprint; start a new artifact root")
    by_world_seed: Dict[Tuple[Tuple[Tuple[str, Any], ...], int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        by_world_seed[(_v4_world_key(row), int(row["training_seed"]))].append(row)

    score_rows: List[Dict[str, Any]] = []
    world_rows: List[Dict[str, Any]] = []
    for (world_key, training_seed), rows in sorted(by_world_seed.items(), key=lambda item: (item[0][0], item[0][1])):
        identity = dict(world_key)
        terrain_hashes = {str(row.get("terrain_hash", "")) for row in rows}
        terrain_hashes.discard("")
        if len(terrain_hashes) > 1:
            offenders = {str(row["model"]): str(row.get("terrain_hash", ""))[:12] for row in rows}
            raise RuntimeError(
                "terrain geometry mismatch in V4 world "
                f"{identity['terrain_type']}:L{identity['terrain_level']}:eval{identity['eval_seed']}: "
                f"{offenders}"
            )
        by_model: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_model[str(row["model"])].append(row)
        mlps = by_model.get(baseline_label, [])
        oracles = by_model.get(oracle_label, [])
        if len(mlps) != 1 or len(oracles) != 1:
            # We do not substitute a different training seed or median several
            # rows.  Emit the missing-reference condition for methods so an
            # incomplete append-only campaign is visible rather than silently
            # changing the comparison population.
            ref_reasons = []
            if len(mlps) != 1:
                ref_reasons.append("missing_or_duplicate_mlp_reference")
            if len(oracles) != 1:
                ref_reasons.append("missing_or_duplicate_oracle_reference")
            classification: Dict[str, Any] = {
                "world_classification": "unscorable",
                "survival_status": "unclassified",
                "tracking_status": "excluded_missing_reference",
                "tracking_include": False,
                "tracking_exclusion_reasons": ";".join(ref_reasons),
                "absolute_tracking_headroom": float("nan"),
            }
            mlp = oracle = None
        else:
            mlp, oracle = mlps[0], oracles[0]
            classification = classify_v4_tracking_world(mlp, oracle, score_cfg)
            if str(identity["physics_tier"]) != str(score_cfg["headline_tier"]):
                reasons = [item for item in str(classification["tracking_exclusion_reasons"]).split(";") if item]
                if "secondary_tier_not_primary_headline" not in reasons:
                    reasons.append("secondary_tier_not_primary_headline")
                classification = {**classification, "tracking_include": False,
                                  "tracking_status": "excluded_secondary_tier",
                                  "tracking_exclusion_reasons": ";".join(reasons)}
        world_rows.append({**identity, "training_seed": training_seed,
                           "baseline_label": baseline_label, "oracle_label": oracle_label,
                           "mlp_tracking_error": float(mlp["tracking_error"]) if mlp else float("nan"),
                           "oracle_tracking_error": float(oracle["tracking_error"]) if oracle else float("nan"),
                           "mlp_fall_rate": float(mlp["fall_rate"]) if mlp else float("nan"),
                           "oracle_fall_rate": float(oracle["fall_rate"]) if oracle else float("nan"),
                           "oracle_achieved_speed_ratio": float(oracle["achieved_speed_ratio"]) if oracle else float("nan"),
                           **classification})
        for label in score_cfg["method_labels"]:
            for method in by_model.get(str(label), []):
                row = {**identity, "training_seed": training_seed, "model": str(label),
                       "scope": _v4_scope(method, score_cfg), "metric_kind": "tracking",
                       "method_tracking_error": float(method["tracking_error"]),
                       "method_fall_rate": float(method["fall_rate"]),
                       "method_achieved_speed_ratio": float(method["achieved_speed_ratio"]),
                       "method_survival_status": ("survived" if float(method["fall_rate"]) <= float(score_cfg["fall_gate_pp"])
                                                  else f"fall_rate_gt_{float(score_cfg['fall_gate_pp']):.2f}"),
                       **classification}
                if mlp is not None:
                    row["baseline_tracking_error"] = float(mlp["tracking_error"])
                if oracle is not None:
                    row["oracle_tracking_error"] = float(oracle["tracking_error"])
                if classification["tracking_include"]:
                    # Never clip the scientific value.  A value outside [0, 1]
                    # is meaningful (regression or beating the oracle) and must
                    # remain visible in the raw and headline seed summaries.
                    raw_gap = (float(mlp["tracking_error"]) - float(method["tracking_error"])) / float(classification["absolute_tracking_headroom"])
                    method_fall_gate = float(score_cfg["fall_gate_pp"])
                    if float(method["fall_rate"]) > method_fall_gate:
                        row.update({"headline_status": "method_fall_gated", "headline_include": True,
                                    "raw_gap_closed": raw_gap, "headline_gap_closed": 0.0,
                                    "reasons": "method_fall_gate"})
                    else:
                        row.update({"headline_status": "eligible", "headline_include": True,
                                    "raw_gap_closed": raw_gap, "headline_gap_closed": raw_gap,
                                    "reasons": ""})
                else:
                    row.update({"headline_status": str(classification["tracking_status"]), "headline_include": False,
                                "reasons": str(classification["tracking_exclusion_reasons"])})
                score_rows.append(row)

    _write_csv(root / "tables" / "scorecard_worlds.csv", world_rows)
    _write_csv(root / "tables" / "scorecard_cells.csv", score_rows)
    seed_scores: List[Dict[str, Any]] = []
    by_seed: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if row["headline_include"]:
            by_seed[(str(row["model"]), str(row["scope"]), int(row["training_seed"]))].append(row)
    for (model, scope, training_seed), rows in sorted(by_seed.items()):
        values = [float(row["headline_gap_closed"]) for row in rows]
        signs = {int(np.sign(value)) for value in values if value != 0.0}
        seed_scores.append({"model": model, "scope": scope, "metric_kind": "tracking", "training_seed": training_seed,
                            "n_cells": len(values), "gap_closed_median": float(np.median(values)),
                            "gap_closed_min": float(np.min(values)), "gap_closed_max": float(np.max(values)),
                            "all_cells_positive": all(value > 0.0 for value in values),
                            "cell_sign_consistent": len(signs) <= 1})
    _write_csv(root / "tables" / "scorecard_seed_scores.csv", seed_scores)
    headlines: List[Dict[str, Any]] = []
    by_headline: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_scores:
        by_headline[(str(row["model"]), str(row["scope"]))].append(row)
    for (model, scope), rows in sorted(by_headline.items()):
        values = [float(row["gap_closed_median"]) for row in rows]
        signs = {int(np.sign(value)) for value in values if value != 0.0}
        headlines.append({"model": model, "metric": scope, "metric_kind": "tracking", "n_training_seeds": len(rows),
                          "value_median": float(np.median(values)), "value_min": float(np.min(values)),
                          "value_max": float(np.max(values)), "all_seeds_positive": all(value > 0.0 for value in values),
                          "seed_sign_consistent": len(signs) <= 1})
    status_counts: Dict[str, int] = defaultdict(int)
    for row in score_rows:
        status_counts[str(row["headline_status"])] += 1
    classification_counts: Dict[str, int] = defaultdict(int)
    for row in world_rows:
        classification_counts[str(row["world_classification"])] += 1
    summary = {"campaign": cfg["campaign"], "headline": headlines, "status_counts": dict(status_counts),
               "world_classification_counts": dict(classification_counts), "raw_cells": len(raw),
               "scorecard_cells": len(score_rows), "scorecard_worlds": len(world_rows), "seed_scores": len(seed_scores),
               "command_ood_diagnostic_raw_cells": len(diagnostic_raw),
               "contract": "v4_headroom_matrix_v1"}
    normalized_path = root / "tables" / "headroom_report_normalized.json"
    normalized_path.write_text(_json(_v4_normalized_headroom_payload(cfg, world_rows, score_rows)), encoding="utf-8")
    summary["normalized_headroom_json"] = str(normalized_path.relative_to(root))
    (root / "tables" / "headline.json").write_text(_json(summary), encoding="utf-8")
    print(f"[v3_eval] V4 aggregate raw={len(raw)} worlds={len(world_rows)} scorecard_cells={len(score_rows)} -> {root / 'tables'}")
    return 0


def cmd_aggregate(cfg: Mapping[str, Any]) -> int:
    root = artifact_root(cfg)
    raw = _raw_rows(root) + _terrain_raw_rows(root)
    frozen_references = _followup_reference_rows(cfg)
    raw = raw + frozen_references
    if not raw:
        raise FileNotFoundError(f"no raw artifacts under {root / 'raw'}")
    _write_csv(root / "tables" / "raw_cells.csv", raw)
    score_cfg = cfg["scorecard"]
    baseline_label, oracle_label = score_cfg["baseline_label"], score_cfg["oracle_label"]
    # A policy training seed is a replicate.  It must be held fixed while the
    # MLP/oracle/method cell is formed; seed aggregation happens only below in
    # ``scorecard_seed_scores.csv`` / ``headline.json``.
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    diagnostic_raw = []
    for row in raw:
        if row["suite"] in SCORED_SUITES:
            if row["headline_command"]:
                grouped[(row["suite"], row["scenario"], int(row["training_seed"]))].append(row)
            else:
                diagnostic_raw.append(row)
    if _uses_v4_world_headroom(cfg):
        return _aggregate_v4_world_headroom(cfg, raw, diagnostic_raw)
    score_rows: List[Dict[str, Any]] = []
    for (suite, scenario, training_seed), rows in grouped.items():
        if suite in TERRAIN_SUITES:
            # The entire scorecard's validity rests on every method seeing a
            # byte-identical heightfield for this (type x level/payload) cell.
            # Fail loud rather than silently compare apples to oranges if any
            # policy's terrain geometry diverged (e.g. a task consumed a
            # different amount of numpy RNG before the terrain was built).
            hashes = {str(row.get("terrain_hash", "")) for row in rows}
            hashes.discard("")
            if len(hashes) > 1:
                offenders = {row["model"]: str(row.get("terrain_hash", ""))[:12] for row in rows}
                raise RuntimeError(
                    f"terrain geometry mismatch in {suite}:{scenario}:seed{training_seed}; methods must share one "
                    f"heightfield but saw {len(hashes)} distinct terrain_hash values: {offenders}"
                )
        baseline_rows = [row for row in rows if row["model"] == baseline_label]
        oracle_rows = [row for row in rows if row["model"] == oracle_label]
        if not baseline_rows or not oracle_rows:
            # Incremental campaigns intentionally accumulate policy-seeds over
            # time. Preserve and report their raw cells now; score a condition
            # only after the baseline and ceiling actually exist together.
            continue
        baseline = {
            "tracking_error": float(np.median([row["tracking_error"] for row in baseline_rows])),
            "fall_rate": float(np.median([row["fall_rate"] for row in baseline_rows])),
        }
        # Parallel SPNTE-scored view of the same (suite, scenario) cell:
        # score_cell_on reuses the identical gate rule, only closing the gap
        # on spnte_lin instead of tracking_error.  Only attempted once the
        # baseline+oracle rows actually carry finite spnte_lin, so historical
        # artifacts without SPNTE fields score exactly as they did before.
        spnte_ready = (
            all(np.isfinite(float(row.get("spnte_lin", float("nan")))) for row in baseline_rows)
            and all(np.isfinite(float(row.get("spnte_lin", float("nan")))) for row in oracle_rows)
        )
        spnte_baseline = {
            "spnte_lin": float(np.median([row["spnte_lin"] for row in baseline_rows])),
            "fall_rate": baseline["fall_rate"],
        } if spnte_ready else None
        for label in score_cfg["method_labels"]:
            for method in [row for row in rows if row["model"] == label]:
                score = score_cell_on("tracking_error", baseline, oracle_rows, method, score_cfg)
                score_rows.append({
                    "suite": suite, "scenario": scenario, "scope": _scope(method), "model": label,
                    "training_seed": method["training_seed"], "payload_tier": method["payload_tier"],
                    "payload_name": method["payload_name"], "command_name": method["command_name"],
                    "metric_kind": "tracking",
                    "baseline_tracking_error": baseline["tracking_error"], "baseline_fall_rate": baseline["fall_rate"],
                    "method_tracking_error": method["tracking_error"], "method_fall_rate": method["fall_rate"],
                    "method_achieved_speed_ratio": method["achieved_speed_ratio"], **score,
                })
                if spnte_ready and np.isfinite(float(method.get("spnte_lin", float("nan")))):
                    spnte_score = score_cell_on("spnte_lin", spnte_baseline, oracle_rows, method, score_cfg)
                    score_rows.append({
                        "suite": suite, "scenario": scenario, "scope": _scope(method), "model": label,
                        "training_seed": method["training_seed"], "payload_tier": method["payload_tier"],
                        "payload_name": method["payload_name"], "command_name": method["command_name"],
                        "metric_kind": "spnte",
                        "baseline_spnte_error": spnte_baseline["spnte_lin"], "baseline_fall_rate": baseline["fall_rate"],
                        "method_spnte_error": float(method["spnte_lin"]), "method_fall_rate": method["fall_rate"],
                        "method_achieved_speed_ratio": method["achieved_speed_ratio"], **spnte_score,
                    })
    _write_csv(root / "tables" / "scorecard_cells.csv", score_rows)
    seed_scores: List[Dict[str, Any]] = []
    by_seed: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if row["headline_include"]:
            by_seed[(row["model"], row["scope"], str(row.get("metric_kind", "tracking")), int(row["training_seed"]))].append(row)
    for (model, scope, metric_kind, seed), rows in by_seed.items():
        values = [float(row["headline_gap_closed"]) for row in rows]
        seed_scores.append({"model": model, "scope": scope, "metric_kind": metric_kind, "training_seed": seed,
                            "n_cells": len(values), "gap_closed_median": float(np.median(values)),
                            "gap_closed_min": float(np.min(values)), "gap_closed_max": float(np.max(values))})
    _write_csv(root / "tables" / "scorecard_seed_scores.csv", seed_scores)
    headlines: List[Dict[str, Any]] = []
    by_headline: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in seed_scores:
        by_headline[(row["model"], row["scope"], str(row.get("metric_kind", "tracking")))].append(row)
    for (model, scope, metric_kind), rows in by_headline.items():
        values = [float(row["gap_closed_median"]) for row in rows]
        headlines.append({"model": model, "metric": scope, "metric_kind": metric_kind, "n_training_seeds": len(rows),
                          "value_median": float(np.median(values)), "value_min": float(np.min(values)),
                          "value_max": float(np.max(values)), "all_seeds_positive": all(value > 0 for value in values)})
    # S0a is deliberately not transformed into gap_closed. It reports the
    # method-minus-MLP tracking and fall difference on static ID.
    s0_static = [row for row in raw if row["suite"] == "s0" and row["scenario"] == "static_id"]
    mlp_static_by_seed: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in s0_static:
        if row["model"] == baseline_label:
            mlp_static_by_seed[int(row["training_seed"])].append(row)
    for label in score_cfg["method_labels"]:
        tracking_delta, fall_delta = [], []
        for row in s0_static:
            if row["model"] != label:
                continue
            same_seed = mlp_static_by_seed.get(int(row["training_seed"]), [])
            if len(same_seed) != 1:
                # A seed is a replicate, not an interchangeable sample.  Do
                # not borrow/median a different seed's MLP for an ID delta.
                continue
            baseline_row = same_seed[0]
            tracking_delta.append(float(row["tracking_lin"]) - float(baseline_row["tracking_lin"]))
            fall_delta.append(float(row["fall_rate"]) - float(baseline_row["fall_rate"]))
        if tracking_delta:
            headlines.append({"model": label, "metric": "ID_delta_tracking", "n_training_seeds": len(tracking_delta),
                              "value_median": float(np.median(tracking_delta)), "value_min": float(np.min(tracking_delta)),
                              "value_max": float(np.max(tracking_delta)), "all_seeds_positive": None,
                              "fall_delta_median": float(np.median(fall_delta))})
    # t0 is descriptive, not a gap-closed suite: report each method's
    # tracking/fall delta versus MLP at the nominal terrain difficulty, per type.
    t0_rows = [row for row in raw if row["suite"] == "t0"]
    for type_name in sorted({str(row["terrain_type_name"]) for row in t0_rows}):
        type_rows = [row for row in t0_rows if str(row["terrain_type_name"]) == type_name]
        mlp_by_seed: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
        for row in type_rows:
            if row["model"] == baseline_label:
                mlp_by_seed[int(row["training_seed"])].append(row)
        for label in score_cfg["method_labels"]:
            tracking_delta, fall_delta = [], []
            for row in type_rows:
                if row["model"] != label:
                    continue
                same_seed = mlp_by_seed.get(int(row["training_seed"]), [])
                if len(same_seed) != 1:
                    continue
                baseline_row = same_seed[0]
                tracking_delta.append(float(row["tracking_lin"]) - float(baseline_row["tracking_lin"]))
                fall_delta.append(float(row["fall_rate"]) - float(baseline_row["fall_rate"]))
            if tracking_delta:
                headlines.append({"model": label, "metric": f"terrain_ID_delta_tracking_{type_name}",
                                  "n_training_seeds": len(tracking_delta), "value_median": float(np.median(tracking_delta)),
                                  "value_min": float(np.min(tracking_delta)), "value_max": float(np.max(tracking_delta)),
                                  "all_seeds_positive": None, "fall_delta_median": float(np.median(fall_delta))})
    # Split by metric_kind so the historical tracking status_counts stays
    # exactly what it was before SPNTE scoring existed; the parallel SPNTE
    # view gets its own counter rather than being silently mixed in.
    status_counts: Dict[str, int] = defaultdict(int)
    spnte_status_counts: Dict[str, int] = defaultdict(int)
    for row in score_rows:
        if str(row.get("metric_kind", "tracking")) == "spnte":
            spnte_status_counts[str(row["headline_status"])] += 1
        else:
            status_counts[str(row["headline_status"])] += 1
    # SPNTE is append-only for V3/V4: it is visible in the report and raw CSV,
    # but does not alter the established tracking/fall gap-closed scorecard.
    spnte_raw = []
    by_spnte: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw:
        if np.isfinite(float(row.get("spnte_lin", float("nan")))):
            by_spnte[(str(row["suite"]), str(row["model"]))].append(row)
    for (suite, model), rows in sorted(by_spnte.items()):
        lin = [float(row["spnte_lin"]) for row in rows]
        yaw = [float(row["spnte_yaw"]) for row in rows if np.isfinite(float(row.get("spnte_yaw", float("nan"))))]
        scales = sorted({float(row["spnte_v_scale"]) for row in rows if np.isfinite(float(row.get("spnte_v_scale", float("nan"))))})
        spnte_raw.append({"suite": suite, "model": model, "n_raw_cells": len(rows),
                          "spnte_lin_mean": float(np.mean(lin)),
                          "spnte_yaw_mean": float(np.mean(yaw)) if yaw else None,
                          "spnte_v_scales": scales})

    summary = {"campaign": cfg["campaign"], "headline": headlines, "status_counts": dict(status_counts),
               "raw_cells": len(raw), "scorecard_cells": len(score_rows), "seed_scores": len(seed_scores)}
    summary["command_ood_diagnostic_raw_cells"] = len(diagnostic_raw)
    summary["spnte_status_counts"] = dict(spnte_status_counts)
    summary["spnte_raw"] = spnte_raw
    (root / "tables" / "headline.json").write_text(_json(summary), encoding="utf-8")
    print(f"[v3_eval] aggregate raw={len(raw)} scorecard_cells={len(score_rows)} -> {root / 'tables'}")
    return 0


def cmd_report(cfg: Mapping[str, Any]) -> int:
    root = artifact_root(cfg)
    summary = json.loads((root / "tables" / "headline.json").read_text(encoding="utf-8"))
    # Older headline.json artifacts (and non gap-closed rows like ID_delta_*)
    # carry no "metric_kind" key at all; default them into the historical
    # tracking table so its rendering is unchanged when SPNTE isn't scored.
    tracking_headline = [row for row in summary["headline"] if row.get("metric_kind", "tracking") == "tracking"]
    spnte_headline = [row for row in summary["headline"] if row.get("metric_kind") == "spnte"]
    lines = ["# V3 Eval Scorecard", "", f"Campaign: `{summary['campaign']}`", "", "## Headline", "",
             "| Method | Metrik | Median | Min–max | Seed | İşaret tutarlılığı |", "|---|---|---:|---:|---:|---:|"]
    for row in tracking_headline:
        lines.append(f"| {row['model']} | {row['metric']} | {row['value_median']:.4f} | {row['value_min']:.4f}–{row['value_max']:.4f} | {row['n_training_seeds']} | {row.get('all_seeds_positive', '')} |")
    if spnte_headline:
        lines += ["", "## SPNTE Gap-Closed (scored view, parallel to Headline)", "",
                  "| Method | Metrik | Median | Min–max | Seed | İşaret tutarlılığı |", "|---|---|---:|---:|---:|---:|"]
        for row in spnte_headline:
            lines.append(f"| {row['model']} | {row['metric']} | {row['value_median']:.4f} | {row['value_min']:.4f}–{row['value_max']:.4f} | {row['n_training_seeds']} | {row.get('all_seeds_positive', '')} |")
    lines += ["", "## Hücre durumları", "", "| Durum | Hücre |", "|---|---:|"]
    for status, count in sorted(summary["status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    if summary.get("spnte_status_counts"):
        lines += ["", "## SPNTE Hücre durumları (scored view)", "", "| Durum | Hücre |", "|---|---:|"]
        for status, count in sorted(summary["spnte_status_counts"].items()):
            lines.append(f"| {status} | {count} |")
    if summary.get("spnte_raw"):
        lines += ["", "## SPNTE (append-only raw cross-check)", "",
                  "| Suite | Method | SPNTE lin | SPNTE yaw | v scale | Raw cells |",
                  "|---|---|---:|---:|---|---:|"]
        for row in summary["spnte_raw"]:
            yaw = "" if row["spnte_yaw_mean"] is None else f"{row['spnte_yaw_mean']:.4f}"
            scales = ", ".join(f"{value:g}" for value in row["spnte_v_scales"])
            lines.append(f"| {row['suite']} | {row['model']} | {row['spnte_lin_mean']:.4f} | {yaw} | {scales} | {row['n_raw_cells']} |")
    lines += ["", f"Command-OOD diagnostic raw hücreleri: `{summary.get('command_ood_diagnostic_raw_cells', 0)}`. Bunlar headline skoruna dahil edilmez.",
              "", "Ham hücreler `tables/raw_cells.csv`, kapı ve skor hücreleri `tables/scorecard_cells.csv`, seed bazlı özetler `tables/scorecard_seed_scores.csv` içindedir.", ""]
    report = root / "report" / "scorecard.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[v3_eval] report -> {report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 generalisation scorecard batch")
    parser.add_argument("command", choices=("plan", "resolve-runs", "run", "aggregate", "report"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", choices=("s0", "s1", "s2", "s3", "t0", "t1", "t2", "all"), default="all")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", default="", help="comma-separated model labels; use one label to append a newly finished policy")
    parser.add_argument("--strict", action="store_true", help="resolve-runs: fail if a requested policy-seed is not final yet")
    parser.add_argument("--diagnostics", action="store_true", help="also run command-OOD diagnostic cells; never adds them to headline scores")
    args = parser.parse_args()
    cfg = load_config(args.config)
    labels = [label.strip() for label in args.models.split(",") if label.strip()]
    if args.command == "plan":
        return cmd_plan(cfg, args.suite, labels=labels, include_diagnostics=args.diagnostics)
    if args.command == "resolve-runs":
        return cmd_resolve_runs(cfg, labels=labels, strict=args.strict)
    if args.command == "run":
        return cmd_run(cfg, args.suite, resume=args.resume, shard=parse_shard(args.shard), labels=labels,
                       include_diagnostics=args.diagnostics)
    if args.command == "aggregate":
        return cmd_aggregate(cfg)
    return cmd_report(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
