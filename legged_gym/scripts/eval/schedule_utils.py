"""Deterministic step-response command schedule (pure, no simulator).

Canonical six-phase schedule for the GO2 triple benchmark campaign:

    stand → forward(vx) → reverse(-vx) → lateral(vy) → yaw(yaw) → stop

Each phase lasts ``phase_steps`` measured steps. Yaw is written to command
channel index 2 (``env.commands[:, 2]``).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Campaign defaults (benchmark_tripler.md §4.2 / B4)
DEFAULT_PHASE_STEPS = 150
DEFAULT_STEP_VX = 1.0
DEFAULT_STEP_VY = 0.75
DEFAULT_STEP_YAW = 0.75

PHASE_NAMES: Tuple[str, ...] = (
    "stand",
    "forward",
    "reverse",
    "lateral",
    "yaw",
    "stop",
)


def build_step_schedule(
    phase_steps: int = DEFAULT_PHASE_STEPS,
    step_vx: float = DEFAULT_STEP_VX,
    step_vy: float = DEFAULT_STEP_VY,
    step_yaw: float = DEFAULT_STEP_YAW,
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """Return ``(schedule, phase_bounds)``.

    schedule: (T, 3) float64 array of [vx, vy, yaw] per measured step.
    phase_bounds: list of (name, start_step) for each command change.
    """
    if phase_steps <= 0:
        raise ValueError(f"phase_steps must be positive, got {phase_steps}")

    phases: Sequence[Tuple[str, Tuple[float, float, float]]] = (
        ("stand", (0.0, 0.0, 0.0)),
        ("forward", (float(step_vx), 0.0, 0.0)),
        ("reverse", (-float(step_vx), 0.0, 0.0)),
        ("lateral", (0.0, float(step_vy), 0.0)),
        ("yaw", (0.0, 0.0, float(step_yaw))),
        ("stop", (0.0, 0.0, 0.0)),
    )
    assert tuple(n for n, _ in phases) == PHASE_NAMES

    schedule = np.zeros((len(phases) * phase_steps, 3), dtype=np.float64)
    bounds: List[Tuple[str, int]] = []
    for i, (name, cmd) in enumerate(phases):
        s = i * phase_steps
        schedule[s : s + phase_steps] = cmd
        bounds.append((name, s))
    return schedule, bounds


def phase_command(name: str, step_vx: float = DEFAULT_STEP_VX,
                  step_vy: float = DEFAULT_STEP_VY,
                  step_yaw: float = DEFAULT_STEP_YAW) -> Tuple[float, float, float]:
    """Return the (vx, vy, yaw) triple for a named phase."""
    table = {
        "stand": (0.0, 0.0, 0.0),
        "forward": (float(step_vx), 0.0, 0.0),
        "reverse": (-float(step_vx), 0.0, 0.0),
        "lateral": (0.0, float(step_vy), 0.0),
        "yaw": (0.0, 0.0, float(step_yaw)),
        "stop": (0.0, 0.0, 0.0),
    }
    if name not in table:
        raise KeyError(f"Unknown phase {name!r}; expected one of {list(table)}")
    return table[name]
