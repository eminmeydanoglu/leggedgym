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


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class TestV4HeadroomProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = load(FULL_PATH)
        cls.smoke = load(SMOKE_PATH)

    def test_full_primary_grid_is_fixed_and_bounded(self):
        protocol = self.full["protocol"]
        self.assertEqual(self.full["schema_version"], "v4_headroom_matrix_v1")
        self.assertEqual(protocol["eval_seed"], 1)
        self.assertEqual(protocol["terrain"]["types"],
                         ["slope", "random_uniform", "stairs_down", "stairs_up", "discrete"])
        self.assertEqual(protocol["terrain"]["levels"], [1, 3, 5, 7, 9])
        self.assertEqual(protocol["commands"], {"vx": [0.4, 0.6, 0.8, 1.0], "vy": 0.0, "yaw_rate": 0.0})
        self.assertEqual(protocol["replicas_per_cell"], 76)
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

    def test_secondary_stress_is_explicit_and_not_a_headline_input(self):
        protocol = self.full["protocol"]
        secondary = protocol["secondary_combined_stress_payload"]
        self.assertEqual(secondary["tier"], "secondary_combined_stress_payload")
        self.assertEqual(secondary["terrain_levels"], [5, 9])
        self.assertEqual(secondary["vx"], [0.6, 1.0])
        self.assertEqual([item["name"] for item in secondary["scenarios"]],
                         ["id_edge_forward_payload", "ood_mass_forward_payload", "payload_plus4", "payload_plus6"])
        self.assertEqual(self.full["scorecard"]["headline_tier"], "primary_nominal_headroom")

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

    def test_smoke_is_a_true_subset_of_full_contract(self):
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

    def test_configs_target_the_dedicated_matrix_runner(self):
        for cfg in (self.full, self.smoke):
            execution = cfg["execution"]
            self.assertEqual(execution["runner_interface"], "v4_headroom_matrix_v1")
            self.assertTrue(str(execution["status"]).startswith("ready_for_"))
            self.assertFalse(execution["do_not_execute_with_current_v3_eval"])


if __name__ == "__main__":
    unittest.main()
