#!/usr/bin/env python3
"""Idempotent orchestrator for the GO2 triple benchmark campaign.

Encodes the fixed nine Altay runs, planned Phase A/B cells, skip-if-valid `.npz`,
serial one-eval-at-a-time execution, atomic write-then-rename, and ledger logging.

Does NOT reimplement rollout logic — shells out to:
  python -m legged_gym.scripts.eval.{indist,sweep,transient}

Usage examples (on a GPU node, after activate):
  # list planned cells only (no GPU)
  .venv/bin/python scripts/benchmark_tripler_orchestrator.py --dry-run --phase A

  # run Phase A serially (skip valid existing artifacts)
  env SIMULATOR=genesis WANDB_MODE=disabled \\
    .venv/bin/python scripts/benchmark_tripler_orchestrator.py --phase A

  # smoke-scale (tiny envs/steps) for three tasks
  .venv/bin/python scripts/benchmark_tripler_orchestrator.py --phase smoke --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Fixed campaign constants (benchmark_tripler.md)
# --------------------------------------------------------------------------- #
TRAINING_COMMIT = "d138a2de490d0daef28841e848c74584aa51f147"
EVAL_SEED = 12345
ARTIFACT_ROOT_NAME = "benchmark_tripler_2026-07-13"
DEFAULT_ARTIFACT_ROOT = os.path.join("logs", "eval", ARTIFACT_ROOT_NAME)

# method_key -> (task, short_label)
METHODS: Dict[str, Tuple[str, str]] = {
    "mlp": ("go2_bench_mlp", "MLP"),
    "p5": ("go2_bench_oracle_id", "P5"),
    "p5v": ("go2_bench_oracle_id_vel", "P5+V"),
}

# (method_key, seed) -> run folder name under logs/go2_benchmark/
RUN_MAP: Dict[Tuple[str, int], str] = {
    ("mlp", 1): "Jul13_09-47-39_bench_mlp_genesis_seed1",
    ("mlp", 2): "Jul13_09-48-03_bench_mlp_genesis_seed2",
    ("mlp", 3): "Jul13_09-48-28_bench_mlp_genesis_seed3",
    ("p5", 1): "Jul13_09-48-26_bench_oracle_id_genesis_seed1",
    ("p5", 2): "Jul13_09-48-51_bench_oracle_id_genesis_seed2",
    ("p5", 3): "Jul13_09-49-17_bench_oracle_id_genesis_seed3",
    ("p5v", 1): "Jul13_10-34-00_bench_oracle_id_vel_genesis_seed1",
    ("p5v", 2): "Jul13_10-34-29_bench_oracle_id_vel_genesis_seed2",
    ("p5v", 3): "Jul13_10-35-00_bench_oracle_id_vel_genesis_seed3",
}

SEEDS = (1, 2, 3)
PRIMARY_MASSES = (-1.0, 0.0, 1.0)
PRIMARY_VX = (0.75, 1.0)

# Full protocol defaults
FULL_NUM_ENVS = 256
FULL_PER_POINT = 256
FULL_WARMUP = 100
FULL_STEPS = 2000


@dataclass
class EvalCell:
    """One planned eval subcommand."""
    cell_id: str
    phase: str  # A1 | A2 | A3 | B1 | ... | smoke
    module: str  # indist | sweep | transient
    task: str
    method_key: str
    seed: int
    load_run: str
    ckpt: str
    out_rel: str  # relative to artifact root
    args: List[str] = field(default_factory=list)
    required_keys: Tuple[str, ...] = ()

    def out_path(self, root: str) -> str:
        return os.path.join(root, self.out_rel)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def python_bin() -> str:
    # Prefer active venv / current interpreter
    return sys.executable


def npz_is_valid(path: str, required_keys: Sequence[str] = ()) -> bool:
    """Reuse shipped validity helper when available."""
    try:
        from legged_gym.scripts.eval.provenance import npz_is_valid as _v
        return _v(path, required_keys=required_keys)
    except Exception:
        return os.path.isfile(path)


# --------------------------------------------------------------------------- #
# Cell planners
# --------------------------------------------------------------------------- #
def plan_smoke_cells(artifact_root: str) -> List[EvalCell]:
    """3-task smoke: 1 ckpt each, small envs/steps."""
    cells = []
    for mk, (task, label) in METHODS.items():
        seed = 1
        run = RUN_MAP[(mk, seed)]
        cells.append(EvalCell(
            cell_id=f"smoke_indist_{mk}_s{seed}",
            phase="smoke",
            module="indist",
            task=task,
            method_key=mk,
            seed=seed,
            load_run=run,
            ckpt="best",
            out_rel=os.path.join("smoke", f"{task}_best.npz"),
            args=[
                "--num_envs", "8", "--warmup", "5", "--steps", "50",
                "--seed", str(EVAL_SEED), "--label", label,
            ],
            required_keys=("tracking_lin_err", "fall_rate", "ckpt_sha256", "task"),
        ))
    return cells


def plan_phase_a1(artifact_root: str) -> List[EvalCell]:
    """9 full in-distribution evals on best.pt."""
    cells = []
    for mk, (task, label) in METHODS.items():
        for seed in SEEDS:
            run = RUN_MAP[(mk, seed)]
            cells.append(EvalCell(
                cell_id=f"A1_{mk}_s{seed}",
                phase="A1",
                module="indist",
                task=task,
                method_key=mk,
                seed=seed,
                load_run=run,
                ckpt="best",
                out_rel=os.path.join("best", f"seed_{seed}", "indist", f"{task}.npz"),
                args=[
                    "--num_envs", str(FULL_NUM_ENVS),
                    "--warmup", str(FULL_WARMUP),
                    "--steps", str(FULL_STEPS),
                    "--seed", str(EVAL_SEED),
                    "--label", label,
                ],
                required_keys=("tracking_lin_err", "fall_rate", "ckpt_sha256"),
            ))
    return cells


def plan_phase_a2(artifact_root: str, ckpt: str = "best") -> List[EvalCell]:
    """18 primary mass×vx sweeps. ckpt 'best' or '3000'."""
    cells = []
    tree = "best" if ckpt == "best" else "model_3000"
    phase = "A2" if ckpt == "best" else "A3"
    for mk, (task, label) in METHODS.items():
        for seed in SEEDS:
            run = RUN_MAP[(mk, seed)]
            for vx in PRIMARY_VX:
                vx_tag = f"{vx:g}"
                cells.append(EvalCell(
                    cell_id=f"{phase}_{mk}_s{seed}_vx{vx_tag}",
                    phase=phase,
                    module="sweep",
                    task=task,
                    method_key=mk,
                    seed=seed,
                    load_run=run,
                    ckpt=ckpt,
                    out_rel=os.path.join(
                        tree, f"seed_{seed}", "primary",
                        f"{task}_mass_vx{vx_tag}.npz",
                    ),
                    args=[
                        "--axis", "added_mass",
                        "--grid", "-1", "0", "1",
                        "--command_vx", str(vx),
                        "--command_vy", "0", "--command_yaw", "0",
                        "--per_point", str(FULL_PER_POINT),
                        "--warmup", str(FULL_WARMUP),
                        "--steps", str(FULL_STEPS),
                        "--seed", str(EVAL_SEED),
                        "--label", label,
                    ],
                    required_keys=("tracking_lin_err_mean", "fall_rate_mean", "ckpt_sha256"),
                ))
    return cells


def plan_phase_b(artifact_root: str) -> List[EvalCell]:
    """B1–B5 cell list (best.pt only)."""
    cells: List[EvalCell] = []
    mass_grid = ["-2", "-1", "0", "1", "2", "3", "4", "5"]
    friction_grid = [
        "0.1", "0.25", "0.4", "0.5", "0.7", "0.9", "1.1", "1.25",
        "1.5", "1.75", "2.0", "2.5",
    ]
    com_grid = ["-0.08", "-0.05", "-0.03", "0", "0.03", "0.05", "0.08"]

    for mk, (task, label) in METHODS.items():
        for seed in SEEDS:
            run = RUN_MAP[(mk, seed)]
            base = ["--seed", str(EVAL_SEED), "--label", label,
                    "--per_point", str(FULL_PER_POINT),
                    "--warmup", str(FULL_WARMUP), "--steps", str(FULL_STEPS)]

            # B1 full mass at vx=0.5 and 1.0
            for vx in (0.5, 1.0):
                cells.append(EvalCell(
                    cell_id=f"B1_{mk}_s{seed}_vx{vx:g}",
                    phase="B1", module="sweep", task=task, method_key=mk, seed=seed,
                    load_run=run, ckpt="best",
                    out_rel=os.path.join("best", f"seed_{seed}", "sweeps",
                                         f"{task}_mass_full_vx{vx:g}.npz"),
                    args=base + ["--axis", "added_mass", "--grid", *mass_grid,
                                 "--command_vx", str(vx), "--command_vy", "0",
                                 "--command_yaw", "0"],
                    required_keys=("tracking_lin_err_mean", "fall_rate_mean"),
                ))

            # B2 friction × 3 commands
            for cmd_name, vx, vy, yaw in (
                ("fwd", 1.0, 0.0, 0.0),
                ("lat", 0.0, 0.75, 0.0),
                ("yaw", 0.0, 0.0, 0.75),
            ):
                cells.append(EvalCell(
                    cell_id=f"B2_{mk}_s{seed}_{cmd_name}",
                    phase="B2", module="sweep", task=task, method_key=mk, seed=seed,
                    load_run=run, ckpt="best",
                    out_rel=os.path.join("best", f"seed_{seed}", "sweeps",
                                         f"{task}_friction_{cmd_name}.npz"),
                    args=base + ["--axis", "friction", "--grid", *friction_grid,
                                 "--command_vx", str(vx), "--command_vy", str(vy),
                                 "--command_yaw", str(yaw)],
                    required_keys=("tracking_lin_err_mean", "fall_rate_mean"),
                ))

            # B3 CoM axes
            for axis in ("com_x", "com_y", "com_z"):
                cells.append(EvalCell(
                    cell_id=f"B3_{mk}_s{seed}_{axis}",
                    phase="B3", module="sweep", task=task, method_key=mk, seed=seed,
                    load_run=run, ckpt="best",
                    out_rel=os.path.join("best", f"seed_{seed}", "sweeps",
                                         f"{task}_{axis}.npz"),
                    args=base + ["--axis", axis, "--grid", *com_grid,
                                 "--command_vx", "1.0", "--command_vy", "0",
                                 "--command_yaw", "0"],
                    required_keys=("tracking_lin_err_mean", "fall_rate_mean"),
                ))

            # B4 step response
            cells.append(EvalCell(
                cell_id=f"B4_{mk}_s{seed}",
                phase="B4", module="transient", task=task, method_key=mk, seed=seed,
                load_run=run, ckpt="best",
                out_rel=os.path.join("best", f"seed_{seed}", "transient",
                                     f"{task}_step_response.npz"),
                args=["--scenario", "step_response",
                      "--per_point", str(FULL_PER_POINT),
                      "--warmup", str(FULL_WARMUP),
                      "--seed", str(EVAL_SEED), "--label", label,
                      "--phase_steps", "150",
                      "--step_vx", "1.0", "--step_vy", "0.75", "--step_yaw", "0.75"],
                required_keys=("err_integral_mean", "fall_mean", "phase_names"),
            ))

            # B5 push recovery: nominal + +1 kg
            for axis_tag, axis_args in (
                ("nominal", []),
                ("mass_p1", ["--axis", "added_mass", "--axis_value", "1.0"]),
            ):
                cells.append(EvalCell(
                    cell_id=f"B5_{mk}_s{seed}_{axis_tag}",
                    phase="B5", module="transient", task=task, method_key=mk, seed=seed,
                    load_run=run, ckpt="best",
                    out_rel=os.path.join("best", f"seed_{seed}", "transient",
                                         f"{task}_push_{axis_tag}.npz"),
                    args=["--scenario", "push_recovery",
                          "--per_point", str(FULL_PER_POINT),
                          "--warmup", str(FULL_WARMUP),
                          "--seed", str(EVAL_SEED), "--label", label,
                          "--command_vx", "1.0", "--command_vy", "0", "--command_yaw", "0",
                          "--pre_push_steps", "50", "--recovery_steps", "150",
                          "--dvx", "0", "--dvy", "1.0", *axis_args],
                    required_keys=("recover_err_integral_mean", "fall_mean"),
                ))
    return cells


def plan_cells(phase: str, artifact_root: str) -> List[EvalCell]:
    phase = phase.upper() if phase.lower() != "smoke" else "smoke"
    if phase == "smoke":
        return plan_smoke_cells(artifact_root)
    if phase == "A":
        return plan_phase_a1(artifact_root) + plan_phase_a2(artifact_root, "best") + \
            plan_phase_a2(artifact_root, "3000")
    if phase == "A1":
        return plan_phase_a1(artifact_root)
    if phase == "A2":
        return plan_phase_a2(artifact_root, "best")
    if phase == "A3":
        return plan_phase_a2(artifact_root, "3000")
    if phase == "B":
        return plan_phase_b(artifact_root)
    if phase == "ALL":
        return plan_cells("A", artifact_root) + plan_cells("B", artifact_root)
    raise ValueError(f"Unknown phase {phase!r}; use smoke|A|A1|A2|A3|B|ALL")


def build_cmd(cell: EvalCell, out_path: str) -> List[str]:
    cmd = [
        python_bin(), "-m", f"legged_gym.scripts.eval.{cell.module}",
        "--task", cell.task,
        "--load_run", cell.load_run,
        "--ckpt", str(cell.ckpt),
        "--out", out_path,
        *cell.args,
    ]
    return cmd


# --------------------------------------------------------------------------- #
# Ledger / run
# --------------------------------------------------------------------------- #
def ensure_artifact_tree(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "aggregate"), exist_ok=True)
    # write run_map.tsv once
    path = os.path.join(root, "run_map.tsv")
    if not os.path.isfile(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["method", "seed", "task", "run_folder", "training_commit"])
            for (mk, seed), run in sorted(RUN_MAP.items()):
                task, _ = METHODS[mk]
                w.writerow([mk, seed, task, run, TRAINING_COMMIT[:7]])


def append_ledger(root: str, row: dict) -> None:
    path = os.path.join(root, "ledger.tsv")
    write_header = not os.path.isfile(path)
    keys = ["timestamp", "cell_id", "phase", "status", "exit_code", "seconds",
            "out_path", "command"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_commands_log(root: str, line: str) -> None:
    with open(os.path.join(root, "commands.log"), "a") as f:
        f.write(line.rstrip() + "\n")


def partial_out_path(final_out: str) -> str:
    """Sibling temp artifact path that still ends in ``.npz`` for np.savez / loaders.

    ``foo/bar.npz`` -> ``foo/bar.partial.npz`` (not ``foo/bar.npz.partial``, which
    would make numpy append another ``.npz`` under broken writers).
    """
    if final_out.endswith(".npz"):
        return final_out[: -len(".npz")] + ".partial.npz"
    return final_out + ".partial.npz"


def run_cell(cell: EvalCell, root: str, *, dry_run: bool = False,
             force: bool = False, cwd: Optional[str] = None) -> str:
    """Run one cell serially. Returns status: skipped|ok|fail|dry."""
    out = cell.out_path(root)
    partial = partial_out_path(out)
    cmd = build_cmd(cell, partial)
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    ts = datetime.now(timezone.utc).isoformat()

    if not force and npz_is_valid(out, cell.required_keys):
        append_ledger(root, dict(
            timestamp=ts, cell_id=cell.cell_id, phase=cell.phase, status="skipped",
            exit_code=0, seconds=0, out_path=out, command=cmd_str,
        ))
        append_commands_log(root, f"{ts} SKIP {cell.cell_id} -> {out}")
        return "skipped"

    if dry_run:
        append_commands_log(root, f"{ts} DRY  {cell.cell_id}: {cmd_str}")
        append_ledger(root, dict(
            timestamp=ts, cell_id=cell.cell_id, phase=cell.phase, status="dry",
            exit_code="", seconds="", out_path=out, command=cmd_str,
        ))
        return "dry"

    os.makedirs(os.path.dirname(out), exist_ok=True)
    # if a previous partial exists, drop it
    if os.path.isfile(partial):
        try:
            os.remove(partial)
        except OSError:
            pass

    env = os.environ.copy()
    env.setdefault("SIMULATOR", "genesis")
    env.setdefault("WANDB_MODE", "disabled")

    append_commands_log(root, f"{ts} RUN  {cell.cell_id}: {cmd_str}")
    t0 = time.time()
    # Serial: one subprocess at a time (caller must not parallelize)
    proc = subprocess.run(cmd, cwd=cwd or repo_root(), env=env)
    elapsed = time.time() - t0
    code = proc.returncode

    if code != 0:
        append_ledger(root, dict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cell_id=cell.cell_id, phase=cell.phase, status="fail",
            exit_code=code, seconds=f"{elapsed:.1f}", out_path=out, command=cmd_str,
        ))
        append_commands_log(root, f"{ts} FAIL {cell.cell_id} exit={code} t={elapsed:.1f}s")
        # fail-loud: leave partial if any; do not promote
        raise RuntimeError(
            f"Cell {cell.cell_id} failed with exit {code}. See commands.log / ledger.tsv"
        )

    # Eval modules write to --out (= partial via atomic_savez); promote to final name.
    written = partial if os.path.isfile(partial) else out
    if not os.path.isfile(written):
        raise RuntimeError(f"Cell {cell.cell_id} exit 0 but no output at {written}")
    if written != out:
        os.replace(written, out)

    if not npz_is_valid(out, cell.required_keys):
        raise RuntimeError(
            f"Cell {cell.cell_id}: output invalid/missing keys after run: {out}"
        )

    append_ledger(root, dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        cell_id=cell.cell_id, phase=cell.phase, status="ok",
        exit_code=0, seconds=f"{elapsed:.1f}", out_path=out, command=cmd_str,
    ))
    append_commands_log(root, f"{ts} OK   {cell.cell_id} t={elapsed:.1f}s -> {out}")
    return "ok"


def write_manifest(root: str, cells: Sequence[EvalCell], results: Dict[str, str]) -> None:
    expected = [c.out_rel for c in cells]
    complete = [
        c.out_rel for c in cells
        if results.get(c.cell_id) in ("ok", "skipped")
        and npz_is_valid(c.out_path(root), c.required_keys)
    ]
    man = {
        "artifact_root": ARTIFACT_ROOT_NAME,
        "training_commit": TRAINING_COMMIT,
        "eval_seed": EVAL_SEED,
        "n_expected": len(expected),
        "n_complete": len(complete),
        "phase_a_complete": all(
            results.get(c.cell_id) in ("ok", "skipped")
            for c in cells if c.phase in ("A1", "A2", "A3")
        ) if any(c.phase.startswith("A") for c in cells) else False,
        "cells": [
            {
                "cell_id": c.cell_id,
                "phase": c.phase,
                "out": c.out_rel,
                "status": results.get(c.cell_id, "pending"),
            }
            for c in cells
        ],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)


def dry_map_summary(cells: Sequence[EvalCell]) -> str:
    lines = [
        f"n_cells={len(cells)}",
        f"run_map_entries={len(RUN_MAP)}",
        f"methods={list(METHODS.keys())}",
        f"seeds={list(SEEDS)}",
        f"training_commit={TRAINING_COMMIT[:7]}",
        f"artifact_root={DEFAULT_ARTIFACT_ROOT}",
        "--- cells ---",
    ]
    for c in cells:
        lines.append(
            f"{c.phase:5} {c.cell_id:40} {c.module:10} ckpt={c.ckpt:6} "
            f"run={c.load_run} -> {c.out_rel}"
        )
    return "\n".join(lines)


# task run_name substrings as embedded in Jul13 folder names
_RUN_NAME_SUBSTR = {
    "mlp": "bench_mlp",
    "p5": "bench_oracle_id",
    "p5v": "bench_oracle_id_vel",
}


def preflight_run_map(
    log_root: str,
    *,
    require_commit: bool = True,
    require_ckpt_files: bool = True,
) -> List[str]:
    """Fail-loud identity check for all nine fixed runs before any GPU work.

    For each RUN_MAP entry verifies:
      * run directory exists
      * ``best.pt`` and ``model_3000.pt`` exist (optional via flag)
      * ``run_manifest.json`` present with matching task + training_seed
      * folder ``_seedN`` matches expected seed
      * training commit is ``d138a2d…`` (optional via flag)
    Returns a list of human-readable OK lines for logging.
    """
    from legged_gym.scripts.eval.provenance import load_run_manifest, verify_run_identity

    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"experiment log root missing: {log_root}")

    lines: List[str] = []
    for (mk, seed), run in sorted(RUN_MAP.items()):
        task, label = METHODS[mk]
        run_dir = os.path.join(log_root, run)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"[{label} seed={seed}] run dir missing: {run_dir}")

        if require_ckpt_files:
            for fname in ("best.pt", "model_3000.pt"):
                p = os.path.join(run_dir, fname)
                if not os.path.isfile(p):
                    raise FileNotFoundError(
                        f"[{label} seed={seed}] missing {fname} under {run_dir}"
                    )

        man = verify_run_identity(
            run_dir,
            expected_task=task,
            expected_run_name=_RUN_NAME_SUBSTR[mk],
            expected_training_seed=seed,
            require_manifest=True,
        )
        # verify_run_identity already checked task/seed; enforce commit if present
        commit = str(man.get("git_commit") or man.get("training_commit") or "")
        if require_commit:
            if not commit:
                raise ValueError(
                    f"[{label} seed={seed}] run_manifest missing git_commit "
                    f"(expected {TRAINING_COMMIT[:7]})"
                )
            if not (
                commit.startswith(TRAINING_COMMIT[:7])
                or commit.startswith(TRAINING_COMMIT)
                or TRAINING_COMMIT.startswith(commit[:7])
            ):
                raise ValueError(
                    f"[{label} seed={seed}] training commit {commit!r} "
                    f"!= expected {TRAINING_COMMIT[:7]}"
                )
        lines.append(
            f"OK {label:5} seed={seed} run={run} task={task} "
            f"commit={commit[:7] if commit else 'n/a'}"
        )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark tripler idempotent orchestrator")
    p.add_argument("--phase", type=str, default="A",
                   help="smoke|A|A1|A2|A3|B|ALL")
    p.add_argument("--artifact-root", type=str, default=None,
                   help=f"default: <repo>/{DEFAULT_ARTIFACT_ROOT}")
    p.add_argument("--dry-run", action="store_true",
                   help="list/log planned commands; do not spawn evals")
    p.add_argument("--force", action="store_true", help="re-run even if valid npz exists")
    p.add_argument("--list-run-map", action="store_true",
                   help="print fixed RUN_MAP and exit")
    p.add_argument("--preflight", action="store_true",
                   help="run manifest/ckpt/commit checks on RUN_MAP and exit")
    p.add_argument("--log-root", type=str, default=None,
                   help="experiment log root for preflight "
                        "(default: <repo>/logs/go2_benchmark)")
    p.add_argument("--skip-preflight", action="store_true",
                   help="skip automatic preflight before real (non-dry) runs")
    args = p.parse_args(argv)

    if args.list_run_map:
        for (mk, seed), run in sorted(RUN_MAP.items()):
            print(f"{mk}\t{seed}\t{METHODS[mk][0]}\t{run}")
        return 0

    log_root = args.log_root or os.path.join(repo_root(), "logs", "go2_benchmark")

    if args.preflight:
        for line in preflight_run_map(log_root):
            print(line)
        print(f"preflight OK: {len(RUN_MAP)} runs under {log_root}")
        return 0

    root = args.artifact_root or os.path.join(repo_root(), DEFAULT_ARTIFACT_ROOT)
    ensure_artifact_tree(root)
    cells = plan_cells(args.phase, root)

    if args.dry_run:
        print(dry_map_summary(cells))
        results = {c.cell_id: "dry" for c in cells}
        # still write a dry ledger trail
        for c in cells:
            run_cell(c, root, dry_run=True)
        write_manifest(root, cells, results)
        return 0

    # Real GPU path: fail-loud preflight unless explicitly skipped
    if not args.skip_preflight:
        ok_lines = preflight_run_map(log_root)
        for line in ok_lines:
            append_commands_log(root, f"PREFLIGHT {line}")
        print(f"preflight OK ({len(ok_lines)} runs)")

    results: Dict[str, str] = {}
    for c in cells:
        # One GPU eval at a time — strict serial loop
        status = run_cell(c, root, dry_run=False, force=args.force)
        results[c.cell_id] = status
    write_manifest(root, cells, results)
    print(f"done phase={args.phase}: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
