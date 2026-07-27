"""Focused CPU-only tests for the standalone headroom HTML report."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SIMULATOR", "genesis")

from legged_gym.scripts.eval.headroom_report import load_input, main, render_report, write_report


def sample_payload() -> dict:
    return {
        "experiment": {
            "name": "Sabit terrain karşılaştırması",
            "contract": "Aynı terrain ve command bank.",
            "seed_count": 3,
            "tracking_metric": "Planar tracking error (m/s)",
            "limitations": ["Küçük örnek fixture."],
        },
        "worlds": [
            {
                "world": "Stairs / L7", "id_ood": "OOD", "include": True,
                "tracking_error": {"MLP": 1.0, "Oracle": 0.5, "DreamWaQ": 0.75, "HIM": 0.6},
                "fall_rate": {"MLP": 0.04, "Oracle": 0.01, "DreamWaQ": 0.02, "HIM": 0.01},
                "achieved_speed_ratio": {"MLP": 0.87, "Oracle": 0.98, "DreamWaQ": 0.94, "HIM": 0.96},
                "seed_consistency": {"DreamWaQ": {"better_seeds": 3, "total_seeds": 3}, "HIM": {"better_seeds": 2, "total_seeds": 3}},
            },
            {
                "world": "Smooth / L0", "id_ood": "ID", "include": True,
                "tracking_error": {"MLP": 0.4, "Oracle": 0.4, "DreamWaQ": 0.38, "HIM": 0.39},
            },
        ],
    }


class TestHeadroomReport(unittest.TestCase):
    def test_json_renders_transparent_tracking_and_separate_survival(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path, output_path = Path(tmp) / "input.json", Path(tmp) / "report.html"
            input_path.write_text(json.dumps(sample_payload()), encoding="utf-8")
            write_report(input_path, output_path, name="", contract="", seed_count=0, tracking_metric="", limitations=[])
            report = output_path.read_text(encoding="utf-8")
        self.assertIn("Sabit terrain karşılaştırması", report)
        self.assertIn("Deney sözleşmesi", report)
        self.assertIn("Training seed", report)
        self.assertIn("Tracking headroom", report)
        self.assertIn("Survival ve komut gerçekleşmesi", report)
        self.assertIn("0.500", report)  # absolute MLP–Oracle headroom
        self.assertIn("50.0% kapandı", report)  # DreamWaQ: (1 - .75) / .5
        self.assertIn("80.0% kapandı", report)  # HIM: (1 - .6) / .5
        self.assertIn("MLP→Oracle tracking headroom pozitif değil", report)
        self.assertIn("DW 3/3", report)
        self.assertIn("HIM 2/3", report)
        self.assertIn("rail-marker dw", report)
        self.assertIn("rail-marker him", report)

    def test_csv_normalises_metadata_and_cli_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path, output_path = Path(tmp) / "worlds.csv", Path(tmp) / "report.html"
            columns = ["world", "id_ood", "include", "tracking_error_mlp", "tracking_error_oracle", "tracking_error_dreamwaq", "tracking_error_him", "seed_consistency_dreamwaq", "seed_consistency_him"]
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow({"world": "Rough / L5", "id_ood": "OOD", "include": "true", "tracking_error_mlp": "0.9", "tracking_error_oracle": "0.6", "tracking_error_dreamwaq": "0.75", "tracking_error_him": "0.66", "seed_consistency_dreamwaq": "3/3", "seed_consistency_him": "2/3"})
            rc = main(["--input", str(input_path), "--output", str(output_path), "--name", "CSV deney", "--contract", "Eşleşmiş protokol.", "--seed-count", "3"])
            experiment, worlds = load_input(input_path, name="CSV deney", contract="Eşleşmiş protokol.", seed_count=3, tracking_metric="Planar tracking error", limitations=[])
            report = output_path.read_text(encoding="utf-8")
        self.assertEqual(rc, 0)
        self.assertEqual(experiment.name, "CSV deney")
        self.assertEqual(worlds[0].consistency["DreamWaQ"], "3/3")
        self.assertIn("80.0% kapandı", report)

    def test_rejects_missing_or_duplicate_world_contract(self):
        payload = sample_payload()
        payload["worlds"][0]["tracking_error"].pop("Oracle")
        with self.assertRaisesRegex(ValueError, "MLP and Oracle"):
            from legged_gym.scripts.eval.headroom_report import normalise_json
            normalise_json(payload)

        payload = sample_payload()
        payload["worlds"][1]["world"] = payload["worlds"][0]["world"]
        with self.assertRaisesRegex(ValueError, "unique"):
            from legged_gym.scripts.eval.headroom_report import normalise_json
            normalise_json(payload)

    def test_render_escapes_untrusted_world_label(self):
        payload = sample_payload()
        payload["worlds"][0]["world"] = "<unsafe & world>"
        from legged_gym.scripts.eval.headroom_report import normalise_json
        experiment, worlds = normalise_json(payload)
        report = render_report(experiment, worlds)
        self.assertIn("&lt;unsafe &amp; world&gt;", report)
        self.assertNotIn("<unsafe & world>", report)

    def test_protocol_exclusion_codes_render_as_plain_turkish(self):
        payload = sample_payload()
        payload["worlds"][0]["include"] = False
        payload["worlds"][0]["exclusion_reason"] = (
            "oracle_achieved_speed_ratio_lt_0.90;"
            "absolute_tracking_headroom_lt_0.10;"
            "oracle_achieved_speed_ratio_lt_0.90"
        )
        from legged_gym.scripts.eval.headroom_report import normalise_json
        experiment, worlds = normalise_json(payload)
        report = render_report(experiment, worlds)
        self.assertEqual(report.count("Oracle komut hızının %90’ına ulaşamadı"), 1)
        self.assertIn("MLP–Oracle tracking farkı 0.10’un altında", report)

    def test_explicit_seed_level_gap_closed_is_rendered_as_percent_not_ratio_of_error_medians(self):
        payload = sample_payload()
        # The displayed error medians imply 50%; the protocol-owned seed-level
        # median is deliberately different and must win.
        payload["worlds"][0]["gap_closed"] = {"DreamWaQ": 0.25}
        from legged_gym.scripts.eval.headroom_report import normalise_json
        experiment, worlds = normalise_json(payload)
        report = render_report(experiment, worlds)
        self.assertIn("25.0% kapandı", report)
        self.assertNotIn("50.0% kapandı", report)


if __name__ == "__main__":
    unittest.main()
