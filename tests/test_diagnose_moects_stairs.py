import importlib.util
import os
from pathlib import Path

import pytest

os.environ.setdefault("SIMULATOR", "genesis")

SCRIPT = Path(__file__).parents[1] / "legged_gym/scripts/eval/diagnose_moects_stairs.py"
SPEC = importlib.util.spec_from_file_location("diagnose_moects_stairs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_training_moe_stairs_heights_and_width_contract():
    assert [MODULE.stairs_step_height(i) for i in range(2, 7)] == pytest.approx(
        [0.096, 0.119, 0.142, 0.165, 0.188], abs=1e-12)
    assert MODULE.STEP_WIDTH_M == 0.31


def test_matrix_is_exactly_four_isolating_conditions():
    assert MODULE.CONDITIONS == {
        "A": ("straight", "nominal"), "B": ("training", "nominal"),
        "C": ("straight", "training_dr"), "D": ("training", "training_dr"),
    }
