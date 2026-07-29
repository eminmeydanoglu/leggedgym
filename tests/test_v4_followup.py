"""Contract tests for discovery -> frozen adaptation follow-up handoff."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from legged_gym.scripts.eval.v3_eval import _planned_cells, _v4_campaign_fingerprint, _v4_protocol_fingerprint, cmd_aggregate, load_config
from legged_gym.scripts.eval.v4_followup import build_followup_config, build_manifest, load_method_template


ROOT = Path(__file__).resolve().parents[1]


class TestV4Followup(unittest.TestCase):
    def _source_cfg(self, root: Path):
        cfg = load_config(str(ROOT / "configs/eval/v4_headroom_smoke.yaml"))
        cfg = json.loads(json.dumps(cfg))
        cfg["artifact_root"] = str(root)
        return cfg

    @staticmethod
    def _world_row(seed: int):
        return {
            "suite": "h4", "terrain_type": "stairs_up", "terrain_level": 5, "command_vx": 0.8,
            "physics_tier": "primary_nominal_headroom", "physics_signature": json.dumps({"com_x_m": 0.0, "friction": 1.0, "mass_kg": 0.0}, separators=(",", ":")),
            "eval_seed": 1, "physics_axis": "mass_kg", "physics_band": "id",
            "scenario": "stairs_up__L5__vx0p8__primary__mass_kg__id__0", "training_seed": seed,
            "tracking_include": True, "mlp_tracking_error": 1.0, "oracle_tracking_error": 0.5,
            "mlp_fall_rate": 0.0, "oracle_fall_rate": 0.0, "oracle_achieved_speed_ratio": 1.0,
        }

    @staticmethod
    def _save(root: Path, model: str, seed: int, error: float, fingerprint: str, campaign: str = ""):
        name = "stairs_up__L5__vx0p8__primary__mass_kg__id__0"
        path = root / "raw" / "h4" / model / f"seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / f"{name}.npz", suite="h4", scenario=name, model=model, training_seed=seed,
                 tracking_lin=np.array([error]), tracking_yaw=np.array([0.0]), fall_rate=np.array([0.0]),
                 return_per_step=np.array([1.0]), achieved_speed=np.array([0.8]), achieved_speed_ratio=np.array([1.0]),
                 command_vx=0.8, physics_tier="primary_nominal_headroom",
                 physics_signature=json.dumps({"com_x_m": 0.0, "friction": 1.0, "mass_kg": 0.0}, separators=(",", ":")),
                 eval_seed=1, physics_axis="mass_kg", physics_band="id", protocol_kind="v4_headroom_matrix_v1",
                 protocol_fingerprint=fingerprint, campaign_fingerprint=campaign,
                 terrain_type_name="stairs_up", terrain_level=5, terrain_hash="same")

    def _discovery(self, root: Path, cfg):
        tables = root / "tables"; tables.mkdir(parents=True)
        rows = [self._world_row(1), self._world_row(2)]
        with (tables / "scorecard_worlds.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(rows[0])); writer.writeheader(); writer.writerows(rows)
        (tables / "headline.json").write_text("{}")
        selections = []
        for seed in (1, 2):
            for label in ("MLP", "Superset-Oracle"):
                selections.append({"label": label, "training_seed": seed, "run_path": f"/runs/{label}/{seed}",
                                   "checkpoint_path": f"/runs/{label}/{seed}/best.pt", "checkpoint_sha256": f"sha-{label}-{seed}"})
        (root / "run_selection.json").write_text(json.dumps({"selections": selections}))
        fingerprint = _v4_protocol_fingerprint(cfg)
        campaign = _v4_campaign_fingerprint(cfg, root)
        for seed in (1, 2):
            self._save(root, "MLP", seed, 1.0, fingerprint, campaign)
            self._save(root, "Superset-Oracle", seed, 0.5, fingerprint, campaign)
        raw_rows = []
        signature = json.dumps({"com_x_m": 0.0, "friction": 1.0, "mass_kg": 0.0}, separators=(",", ":"))
        for seed in (1, 2):
            for model, error in (("MLP", 1.0), ("Superset-Oracle", 0.5)):
                rel = f"raw/h4/{model}/seed_{seed}/stairs_up__L5__vx0p8__primary__mass_kg__id__0.npz"
                raw_rows.append({"protocol_kind": "v4_headroom_matrix_v1", "protocol_fingerprint": fingerprint,
                                 "campaign_fingerprint": campaign, "suite": "h4", "terrain_type_name": "stairs_up", "terrain_level": 5,
                                 "command_vx": 0.8, "physics_tier": "primary_nominal_headroom", "physics_signature": signature,
                                 "eval_seed": 1, "physics_axis": "mass_kg", "physics_band": "id", "scenario": "stairs_up__L5__vx0p8__primary__mass_kg__id__0",
                                 "training_seed": seed, "model": model, "terrain_hash": "same", "path": rel,
                                 "tracking_error": error, "fall_rate": 0.0, "achieved_speed_ratio": 1.0})
        with (tables / "raw_cells.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(raw_rows[0])); writer.writeheader(); writer.writerows(raw_rows)

    def test_manifest_selects_only_paired_eligible_worlds_and_followup_plans_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; cfg = self._source_cfg(source); self._discovery(source, cfg)
            manifest = build_manifest(cfg, source, ["DreamWaQ", "HIM-fixed"])
            self.assertEqual(len(manifest["worlds"]), 1)
            manifest_path = Path(tmp) / "followup.json"; manifest_path.write_text(json.dumps(manifest))
            follow = build_followup_config(cfg, manifest_path, str(Path(tmp) / "follow"), ["DreamWaQ", "HIM-fixed"])
            cells = _planned_cells(follow, "all")
            self.assertEqual(len(cells), 4)
            self.assertEqual({cell.model for cell in cells}, {"DreamWaQ", "HIM-fixed"})
            self.assertEqual({cell.seed for cell in cells}, {1, 2})
            self.assertEqual(_v4_protocol_fingerprint(follow), _v4_protocol_fingerprint(cfg))
            self.assertEqual(follow["protocol"].get("planned_cell_budget"), cfg["protocol"].get("planned_cell_budget"))

    def test_followup_aggregate_uses_hash_pinned_discovery_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; cfg = self._source_cfg(source); self._discovery(source, cfg)
            manifest = build_manifest(cfg, source, ["DreamWaQ", "HIM-fixed"])
            manifest_path = Path(tmp) / "followup.json"; manifest_path.write_text(json.dumps(manifest))
            follow_root = Path(tmp) / "follow"
            follow = build_followup_config(cfg, manifest_path, str(follow_root), ["DreamWaQ", "HIM-fixed"])
            fingerprint = _v4_protocol_fingerprint(follow)
            for seed in (1, 2):
                self._save(follow_root, "DreamWaQ", seed, 0.75, fingerprint, _v4_campaign_fingerprint(follow, follow_root))
                self._save(follow_root, "HIM-fixed", seed, 0.6, fingerprint, _v4_campaign_fingerprint(follow, follow_root))
            self.assertEqual(cmd_aggregate(follow), 0)
            headline = json.loads((follow_root / "tables" / "headline.json").read_text())
            self.assertEqual({row["model"] for row in headline["headline"]}, {"DreamWaQ", "HIM-fixed"})
            self.assertTrue(all(row["all_seeds_positive"] for row in headline["headline"]))
            (source / "run_selection.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "checkpoint selection"):
                cmd_aggregate(follow)

    def test_followup_rejects_scorecard_or_raw_reference_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; cfg = self._source_cfg(source); self._discovery(source, cfg)
            manifest = build_manifest(cfg, source, ["DreamWaQ"])
            manifest_path = Path(tmp) / "followup.json"; manifest_path.write_text(json.dumps(manifest))
            follow_root = Path(tmp) / "follow"
            follow = build_followup_config(cfg, manifest_path, str(follow_root), ["DreamWaQ"])
            fingerprint = _v4_protocol_fingerprint(follow)
            for seed in (1, 2):
                self._save(follow_root, "DreamWaQ", seed, 0.75, fingerprint, _v4_campaign_fingerprint(follow, follow_root))
            # Same compact raw table, but one frozen NPZ was changed after the
            # manifest: table hashes alone must not silently permit it.
            self._save(source, "MLP", 1, 1.2, _v4_protocol_fingerprint(cfg), _v4_campaign_fingerprint(cfg, source))
            with self.assertRaisesRegex(RuntimeError, "reference artifact drift"):
                cmd_aggregate(follow)

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; cfg = self._source_cfg(source); self._discovery(source, cfg)
            manifest = build_manifest(cfg, source, ["DreamWaQ"])
            manifest_path = Path(tmp) / "followup.json"; manifest_path.write_text(json.dumps(manifest))
            follow_root = Path(tmp) / "follow"
            follow = build_followup_config(cfg, manifest_path, str(follow_root), ["DreamWaQ"])
            fingerprint = _v4_protocol_fingerprint(follow)
            for seed in (1, 2):
                self._save(follow_root, "DreamWaQ", seed, 0.75, fingerprint, _v4_campaign_fingerprint(follow, follow_root))
            (source / "tables" / "scorecard_worlds.csv").write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "scorecard worlds"):
                cmd_aggregate(follow)

    def test_manifest_rejects_an_eligible_subset_of_training_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; cfg = self._source_cfg(source); self._discovery(source, cfg)
            path = source / "tables" / "scorecard_worlds.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[1]["tracking_include"] = "False"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
            self.assertEqual(build_manifest(cfg, source, ["DreamWaQ", "HIM-fixed"])["worlds"], [])

    def test_manifest_excludes_secondary_even_if_its_discovery_rows_are_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"; cfg = self._source_cfg(source); self._discovery(source, cfg)
            path = source / "tables" / "scorecard_worlds.csv"
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            secondary = []
            for seed in (1, 2):
                row = dict(self._world_row(seed))
                row["physics_tier"] = "secondary_combined_stress_payload"
                row["physics_axis"] = "combined"
                row["physics_band"] = "payload_stress"
                row["physics_signature"] = json.dumps({"mass_kg": 4.0, "com_x_m": 0.04, "friction": 0.75}, separators=(",", ":"))
                secondary.append(row)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows + secondary)
            manifest = build_manifest(cfg, source, ["DreamWaQ"])
            self.assertEqual(len(manifest["worlds"]), 1)
            self.assertEqual(manifest["worlds"][0]["identity"]["physics_tier"], "primary_nominal_headroom")

    def test_full_discovery_injects_canonical_adaptive_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_cfg = load_config(str(ROOT / "configs/eval/v4_headroom.yaml"))
            template = load_method_template(ROOT / "configs/eval/v4_adaptive_methods.yaml")
            manifest_path = Path(tmp) / "manifest.json"; manifest_path.write_text(json.dumps({"schema_version": "fixture"}))
            follow = build_followup_config(source_cfg, manifest_path, str(Path(tmp) / "follow"), ["DreamWaQ", "HIM-fixed"], template)
            config_path = Path(tmp) / "follow.yaml"
            import yaml
            config_path.write_text(yaml.safe_dump(follow, sort_keys=False), encoding="utf-8")
            loaded = load_config(str(config_path))
            models = {model["label"]: model for model in loaded["models"]}
            self.assertEqual(set(models), {"MLP", "Superset-Oracle", "DreamWaQ", "HIM-fixed"})
            self.assertTrue(models["DreamWaQ"]["run_paths"])
            self.assertTrue(models["HIM-fixed"]["run_paths"])
            self.assertEqual(_v4_protocol_fingerprint(loaded), _v4_protocol_fingerprint(source_cfg))


if __name__ == "__main__":
    unittest.main()
