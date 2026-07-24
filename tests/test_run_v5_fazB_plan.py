"""run_v5_fazB.sh mode defaults must not be clobbered by headline pre-assignment.

Regression for the operational bug where ARMS/SEEDS/NUM_ENVS/MAX_ITERATIONS were
set to the full campaign first, so --smoke's ${ARMS:-go2_v5_lpacrl} etc. never
applied and a "smoke" run launched 4 arms x 3 seeds x 3000 x 4096.
"""
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_v5_fazB.sh"


def _print_plan(*args: str, env: dict | None = None) -> dict[str, str]:
    merged = os.environ.copy()
    # Clear campaign knobs so mode defaults are visible; caller may re-set.
    for key in (
        "ARMS",
        "SEEDS",
        "NUM_ENVS",
        "MAX_ITERATIONS",
        "NUM_SHARDS",
        "SKIP_VALIDATION",
        "VALIDATION_CONFIG",
    ):
        merged.pop(key, None)
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args, "--print-plan"],
        cwd=str(REPO),
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"print-plan failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    plan = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        plan[key.strip()] = value.strip()
    return plan


class TestRunV5FazBPlan(unittest.TestCase):
    def test_script_is_executable_path(self):
        self.assertTrue(SCRIPT.is_file(), SCRIPT)

    def test_smoke_defaults_are_tiny(self):
        plan = _print_plan("--smoke")
        self.assertEqual(plan["mode"], "smoke")
        self.assertEqual(plan["arms"], "go2_v5_lpacrl")
        self.assertEqual(plan["seeds"], "1")
        self.assertEqual(plan["num_envs"], "64")
        self.assertEqual(plan["max_iterations"], "3")
        self.assertEqual(plan["skip_validation"], "0")

    def test_headline_defaults_are_full_campaign(self):
        plan = _print_plan()
        self.assertEqual(plan["mode"], "headline")
        self.assertEqual(
            plan["arms"],
            "go2_v5_handcrafted go2_v5_uniform go2_v5_lpacrl go2_v5_alp",
        )
        self.assertEqual(plan["seeds"], "1 2 3")
        self.assertEqual(plan["num_envs"], "4096")
        self.assertEqual(plan["max_iterations"], "3000")

    def test_beta_pilot_defaults(self):
        plan = _print_plan("--beta-pilot")
        self.assertEqual(plan["mode"], "beta_pilot")
        self.assertEqual(plan["arms"], "go2_v5_lpacrl")
        self.assertEqual(plan["seeds"], "1")
        self.assertEqual(plan["num_envs"], "4096")
        self.assertEqual(plan["max_iterations"], "400")
        self.assertEqual(plan["skip_validation"], "1")

    def test_env_overrides_still_win_under_smoke(self):
        plan = _print_plan(
            "--smoke",
            env={
                "ARMS": "go2_v5_uniform go2_v5_alp",
                "SEEDS": "7 8",
                "NUM_ENVS": "128",
                "MAX_ITERATIONS": "9",
            },
        )
        self.assertEqual(plan["arms"], "go2_v5_uniform go2_v5_alp")
        self.assertEqual(plan["seeds"], "7 8")
        self.assertEqual(plan["num_envs"], "128")
        self.assertEqual(plan["max_iterations"], "9")


if __name__ == "__main__":
    unittest.main()
