"""Declarative contract tests for the executable V4 nominal-headroom campaign.

These tests intentionally do not import ``v3_eval`` or score artifacts. They
guard the protocol surface consumed by the dedicated matrix runner, while
leaving rollout/scoring behavior to its focused tests.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = ROOT / "configs/eval/v4_headroom.yaml"
SMOKE_PATH = ROOT / "configs/eval/v4_headroom_smoke.yaml"
CONFIRM_PATH = ROOT / "configs/eval/v4_headroom_confirm.yaml"
INVENTORY_PATH = ROOT / "configs/eval/v4_spnte_checkpoint_inventory.yaml"


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class TestV4HeadroomProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = load(FULL_PATH)
        cls.smoke = load(SMOKE_PATH)
        cls.confirm = load(CONFIRM_PATH)
        cls.inventory = load(INVENTORY_PATH)

    def test_eight_spnte_deployments_are_hash_locked_and_match_config_pins(self):
        self.assertEqual(self.inventory["schema_version"], "v4_spnte_checkpoint_inventory_v1")
        self.assertEqual(self.inventory["checkpoint"], "best_spnte")
        entries = {item["label"]: item for item in self.inventory["models"]}
        self.assertEqual(set(entries), {"MLP", "Superset-Oracle", "DreamWaQ", "HIM-fixed"})
        self.assertTrue(all(set(item["seeds"]) == {1, 2} for item in entries.values()))
        for item in entries.values():
            for seed, deployment in item["seeds"].items():
                self.assertEqual(len(deployment["checkpoint_sha256"]), 64)
                self.assertIsInstance(deployment["selected_iteration"], int)
                self.assertIn(deployment["selection_metric"], {"spnte_v1", "spnte_v1_offline"})
        for cfg in (self.full, self.smoke):
            self.assertEqual(cfg["checkpoint_inventory"], "configs/eval/v4_spnte_checkpoint_inventory.yaml")
            for model in cfg["models"]:
                expected = entries[model["label"]]
                self.assertEqual(model["task"], expected["task"])
                self.assertEqual(model["checkpoint"], self.inventory["checkpoint"])
                self.assertEqual(model["run_paths"],
                                 {seed: item["run_folder"] for seed, item in expected["seeds"].items()})

    def test_full_primary_grid_is_small_but_discriminative(self):
        protocol = self.full["protocol"]
        self.assertEqual(self.full["schema_version"], "v4_headroom_matrix_v1")
        self.assertEqual(protocol["eval_seed"], 1)
        self.assertEqual(protocol["terrain"]["types"], ["random_uniform", "stairs_up"])
        self.assertEqual(protocol["terrain"]["levels"], [3])
        self.assertEqual(protocol["commands"], {"vx": [0.6, 1.0], "vy": 0.0, "yaw_rate": 0.0})
        self.assertEqual(protocol["warmup_steps"], 50)
        self.assertEqual(protocol["measured_steps"], 500)
        self.assertEqual(protocol["replicas_per_cell"], 16)
        self.assertEqual([model["label"] for model in self.full["models"]], ["MLP", "Superset-Oracle"])
        self.assertEqual(self.full["comparison_scope"], "matched_mlp_vs_superset_oracle_headroom")
        self.assertTrue(all(model.get("run_paths") for model in self.full["models"]))

    def test_primary_axes_are_isolated_with_disjoint_id_ood_bands(self):
        protocol = self.full["protocol"]
        primary = protocol["primary_isolated_axes"]
        self.assertTrue(primary["pin_unswept_axes_to_nominal"])
        self.assertEqual(protocol["nominal_physics"],
                         {"mass_kg": 0.0, "com_x_m": 0.0, "friction": 1.0})
        self.assertEqual(protocol["training_support"],
                         {"mass_kg": [-2.0, 5.0], "com_x_m": [-0.08, 0.08], "friction": [0.5, 1.25]})
        axes = {axis["name"]: axis for axis in primary["axes"]}
        self.assertEqual(set(axes), {"mass_kg", "com_x_m", "friction"})
        self.assertEqual({name: axis["dr_axis"] for name, axis in axes.items()},
                         {"mass_kg": "added_mass", "com_x_m": "com_x", "friction": "friction"})
        for axis in axes.values():
            self.assertTrue(axis["id"])
            self.assertTrue(axis["ood"])
            self.assertFalse(set(axis["id"]) & set(axis["ood"]))
            lower, upper = protocol["training_support"][axis["name"]]
            self.assertTrue(all(lower <= value <= upper for value in axis["id"]))
            self.assertTrue(all(value < lower or value > upper for value in axis["ood"]))
        self.assertEqual(axes["mass_kg"]["id"], [0.0])
        self.assertEqual(axes["mass_kg"]["ood"], [6.0])
        self.assertEqual(axes["com_x_m"]["id"], [0.08])
        self.assertEqual(axes["com_x_m"]["ood"], [0.10])
        self.assertEqual(axes["friction"]["id"], [0.50])
        self.assertEqual(axes["friction"]["ood"], [0.35])

    def test_discovery_explicitly_excludes_secondary_stress_fishing(self):
        protocol = self.full["protocol"]
        secondary = protocol["secondary_combined_stress_payload"]
        self.assertEqual(secondary["tier"], "secondary_combined_stress_payload")
        self.assertEqual(secondary["terrain_levels"], [])
        self.assertEqual(secondary["vx"], [])
        self.assertEqual(secondary["scenarios"], [])
        self.assertEqual(self.full["scorecard"]["headline_tier"], "primary_nominal_headroom")
        self.assertFalse(self.full["scorecard"]["require_oracle_speed"])
        self.assertFalse(self.smoke["scorecard"]["require_oracle_speed"])

    def test_declared_full_cell_budget_matches_the_matrix(self):
        protocol = self.full["protocol"]
        n_models = len(self.full["models"])
        n_seeds = len(self.full["training_seeds"])
        n_primary_points = sum(len(axis["id"]) + len(axis["ood"])
                               for axis in protocol["primary_isolated_axes"]["axes"])
        expected_primary = (n_models * n_seeds * len(protocol["terrain"]["types"])
                            * len(protocol["terrain"]["levels"])
                            * len(protocol["commands"]["vx"]) * n_primary_points)
        secondary = protocol["secondary_combined_stress_payload"]
        expected_secondary = (n_models * n_seeds * len(protocol["terrain"]["types"])
                              * len(secondary["terrain_levels"]) * len(secondary["vx"])
                              * len(secondary["scenarios"]))
        self.assertEqual(protocol["planned_cell_budget"],
                         {"primary": expected_primary, "secondary": expected_secondary,
                          "total": expected_primary + expected_secondary})
        self.assertEqual(expected_primary, 96)
        self.assertEqual(expected_secondary, 0)

    def test_smoke_uses_discovery_terrain_and_checkpoint_contract(self):
        full, smoke = self.full, self.smoke
        self.assertEqual(smoke["schema_version"], full["schema_version"])
        self.assertEqual(smoke["protocol"]["eval_seed"], full["protocol"]["eval_seed"])
        self.assertLess(smoke["protocol"]["measured_steps"], full["protocol"]["measured_steps"])
        self.assertLess(smoke["protocol"]["replicas_per_cell"], full["protocol"]["replicas_per_cell"])
        self.assertTrue(set(smoke["protocol"]["terrain"]["types"]).issubset(full["protocol"]["terrain"]["types"]))
        self.assertTrue(set(smoke["protocol"]["terrain"]["levels"]).issubset(full["protocol"]["terrain"]["levels"]))
        self.assertTrue(set(smoke["protocol"]["commands"]["vx"]).issubset(full["protocol"]["commands"]["vx"]))
        smoke_labels = [model["label"] for model in smoke["models"]]
        self.assertEqual(smoke_labels[:2], ["MLP", "Superset-Oracle"])
        self.assertEqual(set(smoke_labels[2:]), {"DreamWaQ", "HIM-fixed"})
        self.assertEqual(smoke["training_seeds"], [1, 2])
        self.assertTrue(all(model.get("run_paths") for model in smoke["models"]))
        self.assertEqual(smoke["protocol"]["terrain"]["levels"], [3])
        self.assertEqual(smoke["protocol"]["commands"]["vx"], [0.6])

    def test_configs_target_the_dedicated_matrix_runner(self):
        for cfg in (self.full, self.smoke, self.confirm):
            execution = cfg["execution"]
            self.assertEqual(execution["runner_interface"], "v4_headroom_matrix_v1")
            self.assertTrue(str(execution["status"]).startswith("ready_for_"))
            self.assertFalse(execution["do_not_execute_with_current_v3_eval"])

    def test_confirmation_is_only_the_two_preidentified_paired_worlds(self):
        protocol = self.confirm["protocol"]
        self.assertEqual(self.confirm["training_seeds"], [1, 2])
        self.assertEqual(self.confirm["checkpoint_inventory"], self.full["checkpoint_inventory"])
        self.assertEqual(protocol["terrain"]["types"], ["stairs_up"])
        self.assertEqual(protocol["terrain"]["levels"], [3])
        self.assertEqual(protocol["commands"]["vx"], [1.0])
        self.assertEqual(protocol["primary_isolated_axes"]["axes"], [{
            "name": "com_x_m", "dr_axis": "com_x", "id": [0.08], "ood": [0.10],
        }])
        self.assertEqual(protocol["planned_cell_budget"], {"primary": 8, "secondary": 0, "total": 8})
        self.assertFalse(self.confirm["scorecard"]["require_oracle_speed"])


if __name__ == "__main__":
    unittest.main()
