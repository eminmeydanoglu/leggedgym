"""Evaluation harness (Faz 1).

Sweep a single domain-randomization axis over a grid, run every method with the
SAME protocol, and collect distributional metrics (survival, tracking, return).

The pipeline is deliberately method-agnostic: it builds an env from a registered
task, loads its checkpoint, tiles a grid of the swept axis across the parallel
envs (one grid value per contiguous block => N envs per point == N seeds), runs a
fixed number of steps with a fixed forward command, and bins the per-env metrics
back by grid value.

See `sweep.py` for the entry point.
"""
