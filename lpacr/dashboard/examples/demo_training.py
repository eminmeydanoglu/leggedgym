#!/usr/bin/env python3
"""Generate a realistic moving curriculum frontier for dashboard development."""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugger import CurriculumDashboardPlugger, TaskSpace


SPACE = TaskSpace(
    dimensions=("terrain_type", "terrain_level", "vx_bin", "yaw_bin"),
    coordinates={
        "terrain_type": ["Descending", "Ascending", "Rough", "Uphill", "Downhill"],
        "terrain_level": ["L1", "L2", "L3", "L4"],
        "vx_bin": ["0–0.5", "0.5–1.0", "1.0–1.5", "1.5–2.0", "2.0–2.5"],
        "yaw_bin": ["0–0.5", "0.5–1.0", "1.0–1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0"],
    },
)


def make_frame(progress: float) -> dict[str, list[float]]:
    reward, lp, raw_probability, success, samples = [], [], [], [], []
    rng = random.Random(round(progress * 10_000))
    for terrain in range(5):
        terrain_bias = [0.18, 0.32, 0.48, 0.24, 0.28][terrain]
        for level in range(4):
            for vx in range(5):
                for yaw in range(6):
                    difficulty = 0.12 * level + 0.095 * vx + 0.065 * yaw + terrain_bias
                    competence = 1.25 * progress
                    learned = 1 / (1 + math.exp(13 * (difficulty - competence)))
                    frontier = math.exp(-((difficulty - competence) ** 2) / 0.018)
                    forgetting = -0.12 * math.exp(-((difficulty - competence + 0.23) ** 2) / 0.012)
                    noise = rng.uniform(-0.025, 0.025)
                    reward.append(8 + 24 * learned + noise * 10)
                    lp.append(frontier * 2.8 + forgetting + noise)
                    raw_probability.append(math.exp(4.2 * frontier) + 0.035)
                    success.append(max(0, min(1, learned + noise)))
                    samples.append(round(8 + 180 * frontier + rng.random() * 8))
    total = sum(raw_probability)
    probability = [value / total for value in raw_probability]
    return {
        "performance": reward,
        "learning_progress": lp,
        "sampling_probability": probability,
        "success_rate": success,
        "sample_count": samples,
    }


def main() -> None:
    plugger = CurriculumDashboardPlugger("demo-live", SPACE)
    print("Streaming demo-live to http://127.0.0.1:8765")
    try:
        for step in range(0, 3001, 25):
            plugger.log(step, make_frame(step / 3000))
            time.sleep(0.16)
    finally:
        plugger.close()


if __name__ == "__main__":
    main()
