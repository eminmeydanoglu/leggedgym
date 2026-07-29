"""Freeze eligible V4 discovery worlds into an adaptation-only follow-up.

The discovery campaign owns the MLP/Oracle comparison.  This tool turns only
worlds that are eligible for *every* requested training seed into an immutable
manifest, then emits a config that runs adaptation policies on exactly those
worlds.  The follow-up aggregate imports the pinned MLP/Oracle artifacts from
the discovery root; it never reselects worlds from adaptation results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from legged_gym.scripts.eval.v3_eval import _root, _v4_world_identity, load_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _root() / path


def load_method_template(path: str | Path) -> Dict[str, Dict[str, Any]]:
    template_path = _path(str(path))
    data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    entries = data.get("models", []) if isinstance(data, Mapping) else []
    by_label = {str(entry.get("label")): dict(entry) for entry in entries if isinstance(entry, Mapping) and entry.get("label")}
    if not by_label:
        raise ValueError(f"adaptive method template has no models: {template_path}")
    return by_label


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _as_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise CSV strings just enough for the common V4 identity helper."""
    out = dict(row)
    for key in ("training_seed", "terrain_level", "eval_seed"):
        out[key] = int(out[key])
    out["command_vx"] = float(out["command_vx"])
    out["terrain_type_name"] = out.pop("terrain_type")
    return out


def _identity_key(identity: Mapping[str, Any]) -> str:
    return json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))


def build_manifest(source_cfg: Mapping[str, Any], source_root: Path, methods: Iterable[str]) -> Dict[str, Any]:
    table = source_root / "tables" / "scorecard_worlds.csv"
    headline = source_root / "tables" / "headline.json"
    raw_table = source_root / "tables" / "raw_cells.csv"
    run_selection = source_root / "run_selection.json"
    for path in (table, headline, raw_table, run_selection):
        if not path.is_file():
            raise FileNotFoundError(f"discovery artifact is missing: {path}")
    raw = _read_csv(raw_table)
    fingerprints = {row["protocol_fingerprint"] for row in raw if row.get("protocol_kind") == "v4_headroom_matrix_v1"}
    campaign_fingerprints = {row.get("campaign_fingerprint", "") for row in raw if row.get("protocol_kind") == "v4_headroom_matrix_v1"}
    if len(fingerprints) != 1:
        raise RuntimeError(f"discovery must contain exactly one V4 protocol fingerprint, got {sorted(fingerprints)}")
    if len(campaign_fingerprints) != 1 or not next(iter(campaign_fingerprints)):
        raise RuntimeError("discovery must contain exactly one non-empty V4 campaign fingerprint")
    selection = json.loads(run_selection.read_text(encoding="utf-8"))
    checkpoint_by_label_seed = {
        (str(row["label"]), int(row["training_seed"])): {
            "run_path": str(row["run_path"]), "checkpoint_path": str(row["checkpoint_path"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
        }
        for row in selection.get("selections", [])
    }
    raw_by_key = {}
    for row in raw:
        try:
            identity = {
                "suite": str(row["suite"]), "terrain_type": str(row["terrain_type_name"]),
                "terrain_level": int(row["terrain_level"]), "command_vx": float(row["command_vx"]),
                "physics_tier": str(row["physics_tier"]), "physics_signature": str(row["physics_signature"]),
                "eval_seed": int(row["eval_seed"]), "physics_axis": str(row["physics_axis"]),
                "physics_band": str(row["physics_band"]), "scenario": str(row["scenario"]),
            }
            raw_by_key[(_identity_key(identity), int(row["training_seed"]), str(row["model"]))] = row
        except (KeyError, TypeError, ValueError):
            continue
    worlds = [_as_row(row) for row in _read_csv(table)]
    expected_seeds = {int(seed) for seed in source_cfg["training_seeds"]}
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for row in worlds:
        identity = _v4_world_identity(row)
        key = _identity_key(identity)
        grouped.setdefault(key, []).append(row)

    selected = []
    for key, rows in sorted(grouped.items()):
        by_seed = {int(row["training_seed"]): row for row in rows}
        # Requiring the whole paired training-seed population prevents a
        # follow-up from silently reporting a friendlier subset of seeds.
        if (json.loads(key)["physics_tier"] != str(source_cfg["scorecard"]["headline_tier"])
                or set(by_seed) != expected_seeds
                or not all(str(row["tracking_include"]).lower() == "true" for row in by_seed.values())):
            continue
        identity = json.loads(key)
        references = {}
        for seed, row in sorted(by_seed.items()):
            baseline, oracle = source_cfg["scorecard"]["baseline_label"], source_cfg["scorecard"]["oracle_label"]
            mlp_raw = raw_by_key.get((key, seed, str(baseline)))
            oracle_raw = raw_by_key.get((key, seed, str(oracle)))
            if mlp_raw is None or oracle_raw is None:
                raise RuntimeError(f"discovery references incomplete for frozen world seed {seed}")
            hashes = {str(mlp_raw["terrain_hash"]), str(oracle_raw["terrain_hash"])}
            if len(hashes) != 1:
                raise RuntimeError(f"discovery terrain hash mismatch for frozen world seed {seed}")
            checkpoints = {}
            for label in (baseline, oracle):
                checkpoint = checkpoint_by_label_seed.get((str(label), seed))
                if checkpoint is None:
                    raise RuntimeError(f"discovery checkpoint selection is missing for {label} seed {seed}")
                checkpoints[str(label)] = checkpoint
            artifacts = {}
            for label, raw_item in ((str(baseline), mlp_raw), (str(oracle), oracle_raw)):
                artifact = source_root / str(raw_item["path"])
                if not artifact.is_file():
                    raise FileNotFoundError(f"discovery raw artifact is missing: {artifact}")
                artifacts[label] = {
                    "npz_sha256": _sha256(artifact),
                    "tracking_error": float(raw_item["tracking_error"]),
                    "fall_rate": float(raw_item["fall_rate"]),
                    "achieved_speed_ratio": float(raw_item["achieved_speed_ratio"]),
                }
            references[str(seed)] = {
                "mlp_tracking_error": float(row["mlp_tracking_error"]),
                "oracle_tracking_error": float(row["oracle_tracking_error"]),
                "mlp_fall_rate": float(row["mlp_fall_rate"]),
                "oracle_fall_rate": float(row["oracle_fall_rate"]),
                "oracle_achieved_speed_ratio": float(row["oracle_achieved_speed_ratio"]),
                "terrain_hash": next(iter(hashes)), "checkpoints": checkpoints, "artifacts": artifacts,
            }
        selected.append({
            "identity": identity,
            "training_seeds": sorted(expected_seeds),
            "references": references,
        })
    return {
        "schema_version": "v4_adaptive_followup_v1",
        "source": {
            "campaign": str(source_cfg["campaign"]),
            "artifact_root": str(source_root),
            "headline_sha256": _sha256(headline),
            "scorecard_worlds_sha256": _sha256(table),
            "raw_cells_sha256": _sha256(raw_table),
            "run_selection_sha256": _sha256(run_selection),
            "protocol_fingerprint": next(iter(fingerprints)),
            "campaign_fingerprint": next(iter(campaign_fingerprints)),
            "training_seeds": sorted(expected_seeds),
        },
        "method_labels": list(methods),
        "selection_rule": "all configured training seeds have tracking_include=true in discovery",
        "worlds": selected,
    }


def build_followup_config(source_cfg: Mapping[str, Any], manifest_path: Path, output_root: str, methods: list[str],
                          method_entries: Mapping[str, Mapping[str, Any]] | None = None) -> Dict[str, Any]:
    cfg = deepcopy(dict(source_cfg))
    cfg["campaign"] = f"{source_cfg['campaign']}_adaptive_followup"
    cfg["artifact_root"] = output_root
    cfg["execution"] = {
        "runner_interface": "v4_headroom_matrix_v1",
        "status": "ready_for_adaptive_followup",
        "do_not_execute_with_current_v3_eval": False,
        "geometry_identity": "terrain_hash_per_type_level",
        "followup_manifest": str(manifest_path.resolve()),
        "reference_artifact_root": str(source_cfg["artifact_root"]),
        "followup_model_labels": methods,
    }
    cfg["scorecard"]["method_labels"] = methods
    template = dict(method_entries or {})
    known = {str(model["label"]) for model in cfg["models"]}
    for label in methods:
        if label in known:
            continue
        if label not in template:
            raise ValueError(f"missing canonical deploy entry for follow-up method {label}")
        cfg["models"].append(dict(template[label]))
    # The frozen manifest, not the Cartesian discovery matrix, defines this
    # phase's *executed* cells.  Keep the declared discovery budget in the
    # protocol fingerprint: it is provenance for the reference worlds and the
    # V4 planner bypasses this check when a follow-up manifest is present.
    # Removing it would make frozen MLP/Oracle references appear to belong to
    # a different world protocol during aggregate.
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze V4 discovery worlds for adaptation follow-up")
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-root", default="", help="defaults to source config artifact_root")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", default="DreamWaQ,HIM-fixed")
    parser.add_argument("--methods-template", default="configs/eval/v4_adaptive_methods.yaml")
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    source_cfg = load_config(args.source_config)
    template = load_method_template(args.methods_template)
    if not methods or not set(methods).issubset(template):
        raise ValueError(f"--methods must be declared in the adaptive template; known={sorted(template)}")
    source_root = _path(args.source_root or str(source_cfg["artifact_root"]))
    manifest = build_manifest(source_cfg, source_root, methods)
    manifest_path, config_path = Path(args.output_manifest), Path(args.output_config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    config = build_followup_config(source_cfg, manifest_path, args.output_root, methods, template)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "config": str(config_path),
                      "eligible_worlds": len(manifest["worlds"]), "methods": methods}, indent=2))
    return 0 if manifest["worlds"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
