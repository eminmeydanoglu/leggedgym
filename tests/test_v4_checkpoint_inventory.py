"""Fail-closed contracts for the committed V4 SPNTE deployment inventory."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from legged_gym.scripts.eval.v3_eval import _verify_v4_inventory_entry


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "configs/eval/v4_spnte_checkpoint_inventory.yaml"


class TestV4CheckpointInventory(unittest.TestCase):
    def _cfg(self):
        return {"checkpoint_inventory": str(INVENTORY)}

    @staticmethod
    def _model():
        return {"label": "MLP", "task": "go2_v4_mlp", "checkpoint": "best_spnte"}

    def test_matching_locked_artifact_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "Jul20_12-02-36_v4_mlp_genesis_seed1"
            run.mkdir()
            checkpoint = run / "best_spnte.pt"
            checkpoint.write_bytes(b"placeholder")
            expected = "da656cc9f87bfa28a7d83a781caf7b5a183a9e06f10eeff2a1c5deb562909c22"
            with patch("legged_gym.scripts.eval.v3_eval.sha256_file", return_value=expected):
                _verify_v4_inventory_entry(self._cfg(), self._model(), 1, run, checkpoint)

    def test_sha_or_run_folder_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "Jul20_12-02-36_v4_mlp_genesis_seed1"
            run.mkdir()
            checkpoint = run / "best_spnte.pt"
            checkpoint.write_bytes(b"placeholder")
            with patch("legged_gym.scripts.eval.v3_eval.sha256_file", return_value="0" * 64):
                with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                    _verify_v4_inventory_entry(self._cfg(), self._model(), 1, run, checkpoint)
            other = Path(tmp) / "wrong_run"
            other.mkdir()
            with self.assertRaisesRegex(RuntimeError, "run mismatch"):
                _verify_v4_inventory_entry(self._cfg(), self._model(), 1, other, checkpoint)


if __name__ == "__main__":
    unittest.main()
