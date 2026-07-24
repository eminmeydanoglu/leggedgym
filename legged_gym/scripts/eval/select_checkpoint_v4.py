"""Offline ``best_spnte.pt`` selector for v4-terrain runs (indist-eval based).

``select_checkpoint.py`` is hardwired to the v5 UED validation-bank flow: it
consumes ``ued_validation`` / ``ued_rollout`` artifacts scored against a frozen
84-cell bank, and refuses to run without them. The v4-terrain campaign
(``configs/eval/v4_terrain.yaml``) has no such bank -- every model entry there
resolves ``checkpoint: best_spnte``, but the v4 seed-1 runs were trained
*without* ever writing a ``best_spnte.pt`` (only periodic ``model_<iter>.pt``
saves and a ``best_tracking.pt``). This module fills that gap purely offline:
score each existing periodic checkpoint with the SAME in-distribution eval the
training loop already uses (``legged_gym.scripts.eval.indist.run_indist_eval``,
which returns ``spnte_lin`` / ``spnte_yaw`` alongside tracking/fall metrics),
then publish the checkpoint with the lowest ``spnte_lin`` as ``best_spnte.pt``.

Nothing here duplicates the rollout: :func:`build_run_evaluator` builds one
Genesis env + actor-critic per run (mirroring ``indist.py``'s standalone
harness for env construction and ``campaign.py``'s ``build_session`` for
strict per-checkpoint deploy-state loading via ``runner.load_deploy_state`` /
``runner.get_eval_adapter``, so RMA/DreamWaQ/SysID's multi-tensor obs contract
is handled correctly, not just the single-tensor MLP/P5/HIM default), then
calls ``run_indist_eval`` once per candidate iteration, reusing the same
simulator instance instead of paying ``gs.init`` + env-build per checkpoint.

Selection rule (deliberately simpler than v5's UED tie-break chain, since there
is no per-cell bank here): minimum ``spnte_lin``, earliest iteration breaks
ties. ``best_tracking.pt`` is never read or written.

Candidate schedule: by default this mirrors the v5 UED offline schedule
(``configs/eval/v5_ued.yaml``'s ``selection.min_iteration`` / ``iteration_stride``
= 1000 / 500) purely as a familiar, battle-tested default -- floor each stride
target to the largest existing checkpoint ``<=`` it, always including the
final periodic save. Pass ``--iterations`` to evaluate an explicit list instead
(overrides ``--min-iteration`` / ``--stride`` entirely).

CLI:
    python -m legged_gym.scripts.eval.select_checkpoint_v4 \\
        --run-dir logs/go2_v4_terrain_curriculum/Jul20_12-02-36_v4_mlp_genesis_seed1 \\
        --task go2_v4_mlp
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

# Pure, method-agnostic schedule helpers -- reused as-is (not duplicated) from
# the v5 selector; neither function touches the UED bank.
from legged_gym.scripts.eval.select_checkpoint import (
    periodic_checkpoints,
    scheduled_iterations,
)

BEST_SPNTE_NAME = "best_spnte.pt"
SELECTION_SIDECAR_NAME = "best_spnte_selection.json"
SELECTION_METRIC = "spnte_v1_offline"

# eval_fn(checkpoint_path, iteration) -> metrics dict; MUST contain 'spnte_lin'.
EvalFn = Callable[[Path, int], Dict[str, float]]


@dataclass(frozen=True)
class CheckpointScore:
    iteration: int
    checkpoint_path: Path
    metrics: Dict[str, float]


# =============================================================================
# Candidate schedule
# =============================================================================

def candidate_iterations(
    run_dir: str | Path,
    *,
    iterations: Optional[Sequence[int]] = None,
    min_iteration: int = 1000,
    stride: int = 500,
    always_include_final: bool = True,
) -> List[int]:
    """Iterations to score: an explicit list, or the floor/stride schedule.

    ``iterations`` (when given) wins outright -- ``min_iteration`` / ``stride``
    are ignored. Otherwise defers to ``scheduled_iterations`` (same floor/
    stride/always-include-final semantics as the v5 selector).
    """
    available = [it for it, _ in periodic_checkpoints(run_dir)]
    if not available:
        raise FileNotFoundError(f"{run_dir}: no existing periodic model_<iteration>.pt checkpoints")

    if iterations is not None:
        explicit = sorted({int(i) for i in iterations})
        missing = [i for i in explicit if i not in available]
        if missing:
            raise FileNotFoundError(f"{run_dir}: requested --iterations missing checkpoints: {missing}")
        return explicit

    schedule = scheduled_iterations(
        available,
        min_iteration=min_iteration,
        iteration_stride=stride,
        always_include_final=always_include_final,
    )
    if not schedule:
        raise FileNotFoundError(f"{run_dir}: selection schedule produced no candidate iterations")
    return schedule


# =============================================================================
# Scoring + selection
# =============================================================================

def score_candidates(run_dir: str | Path, iterations: Sequence[int], eval_fn: EvalFn) -> List[CheckpointScore]:
    """Evaluate every candidate iteration through ``eval_fn`` (the injection point)."""
    by_iteration = dict(periodic_checkpoints(run_dir))
    scores: List[CheckpointScore] = []
    for iteration in iterations:
        path = by_iteration.get(iteration)
        if path is None:
            raise FileNotFoundError(f"{run_dir}: no model_{iteration}.pt checkpoint")
        metrics = dict(eval_fn(path, iteration))
        if "spnte_lin" not in metrics:
            raise ValueError(
                f"eval_fn for iteration {iteration} did not return 'spnte_lin' "
                f"(got keys={sorted(metrics)})"
            )
        scores.append(CheckpointScore(iteration=int(iteration), checkpoint_path=path, metrics=metrics))
    return scores


def select_best(scores: Sequence[CheckpointScore]) -> CheckpointScore:
    """Minimum ``spnte_lin`` wins; earliest iteration is the deterministic tie-break."""
    if not scores:
        raise ValueError("cannot select best_spnte from zero scored checkpoints")
    return min(scores, key=lambda s: (float(s.metrics["spnte_lin"]), int(s.iteration)))


# =============================================================================
# Provenance + atomic publish
# =============================================================================

def selection_metadata(
    winner: CheckpointScore,
    all_scores: Sequence[CheckpointScore],
    *,
    spnte_v_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Provenance dict merged into ``best_spnte.pt['infos']`` and the sidecar JSON."""
    v_scale = spnte_v_scale if spnte_v_scale is not None else winner.metrics.get("spnte_v_scale")
    return {
        "selection_metric": SELECTION_METRIC,
        "selected_iteration": int(winner.iteration),
        "spnte_lin": float(winner.metrics["spnte_lin"]),
        "spnte_yaw": (float(winner.metrics["spnte_yaw"]) if "spnte_yaw" in winner.metrics else None),
        "spnte_v_scale": (float(v_scale) if v_scale is not None else None),
        "source_checkpoint": winner.checkpoint_path.name,
        "eval_metrics_per_iter": {str(s.iteration): dict(s.metrics) for s in all_scores},
    }


def _merge_infos(checkpoint: Mapping[str, Any], metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Attach selection provenance under checkpoint['infos'] without discarding
    anything already there (e.g. training-time eval_score / eval_it)."""
    result = dict(checkpoint)
    infos = dict(result.get("infos") or {})
    infos.update(metadata)
    result["infos"] = infos
    return result


def materialize_best_spnte(
    run_dir: str | Path,
    winner: CheckpointScore,
    all_scores: Sequence[CheckpointScore],
    *,
    spnte_v_scale: Optional[float] = None,
) -> Path:
    """Copy the winning periodic checkpoint to ``best_spnte.pt``, atomically.

    Never touches ``best_tracking.pt``. Builds the fully-annotated file in a
    sibling ``.tmp`` and ``os.replace``s it into place only on success, so an
    interrupted selection can never leave a truncated ``best_spnte.pt`` or a
    stray ``.tmp`` behind.
    """
    run = Path(run_dir)
    target = run / BEST_SPNTE_NAME
    if winner.checkpoint_path.resolve() == target.resolve():
        raise ValueError("best_spnte artifact must be materialised from a periodic model checkpoint")
    tmp = target.with_name(target.name + ".tmp")
    try:
        shutil.copy2(winner.checkpoint_path, tmp)
        checkpoint = torch.load(tmp, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{winner.checkpoint_path}: checkpoint payload must be a mapping")
        metadata = selection_metadata(winner, all_scores, spnte_v_scale=spnte_v_scale)
        torch.save(_merge_infos(checkpoint, metadata), tmp)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


def write_selection_sidecar(
    run_dir: str | Path,
    task: str,
    winner: CheckpointScore,
    all_scores: Sequence[CheckpointScore],
    *,
    spnte_v_scale: Optional[float] = None,
) -> Path:
    """Write ``best_spnte_selection.json``: the human-auditable twin of the
    provenance baked into ``best_spnte.pt['infos']``."""
    run = Path(run_dir)
    path = run / SELECTION_SIDECAR_NAME
    metadata = selection_metadata(winner, all_scores, spnte_v_scale=spnte_v_scale)
    payload = {
        "task": task,
        "run_dir": str(run),
        "best_spnte_path": str(run / BEST_SPNTE_NAME),
        **metadata,
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def format_summary_table(scores: Sequence[CheckpointScore], winner: CheckpointScore) -> str:
    header = f"{'iter':>8}  {'spnte_lin':>12}  {'spnte_yaw':>12}  {'fall_rate':>10}  {'tracking_lin_err':>18}"
    lines = [header, "-" * len(header)]
    for s in sorted(scores, key=lambda s: s.iteration):
        m = s.metrics
        mark = "  <== winner" if s.iteration == winner.iteration else ""
        lines.append(
            f"{s.iteration:>8}  {float(m.get('spnte_lin', float('nan'))):>12.6f}  "
            f"{float(m.get('spnte_yaw', float('nan'))):>12.6f}  "
            f"{float(m.get('fall_rate', float('nan'))):>10.4f}  "
            f"{float(m.get('tracking_lin_err', float('nan'))):>18.4f}{mark}"
        )
    return "\n".join(lines)


# =============================================================================
# Orchestration
# =============================================================================

def run_selection(
    run_dir: str | Path,
    task: str,
    *,
    num_envs: Optional[int] = None,
    eval_seed: int = 12345,
    steps: int = 2000,
    warmup: int = 100,
    cpu: bool = False,
    iterations: Optional[Sequence[int]] = None,
    min_iteration: int = 1000,
    stride: int = 500,
    always_include_final: bool = True,
    eval_fn: Optional[EvalFn] = None,
):
    """Score, select, and publish. Pass ``eval_fn`` to skip the Genesis build
    entirely (the monkeypatch seam used by the offline-selection unit tests)."""
    run_dir = Path(run_dir)
    iters = candidate_iterations(
        run_dir, iterations=iterations, min_iteration=min_iteration,
        stride=stride, always_include_final=always_include_final,
    )
    if eval_fn is None:
        eval_fn = build_run_evaluator(
            run_dir, task, num_envs=num_envs, eval_seed=eval_seed,
            steps=steps, warmup=warmup, cpu=cpu,
        )
    scores = score_candidates(run_dir, iters, eval_fn)
    winner = select_best(scores)
    spnte_v_scale = winner.metrics.get("spnte_v_scale")
    best_path = materialize_best_spnte(run_dir, winner, scores, spnte_v_scale=spnte_v_scale)
    sidecar_path = write_selection_sidecar(run_dir, task, winner, scores, spnte_v_scale=spnte_v_scale)
    return winner, best_path, sidecar_path, scores


# =============================================================================
# Genesis-backed eval_fn builder (the only piece requiring a real simulator)
# =============================================================================

def build_run_evaluator(
    run_dir: str | Path,
    task: str,
    *,
    num_envs: Optional[int] = None,
    eval_seed: int = 12345,
    steps: int = 2000,
    warmup: int = 100,
    cpu: bool = False,
) -> EvalFn:
    """Build one Genesis env + actor-critic for ``task`` and return an
    ``eval_fn(checkpoint_path, iteration) -> metrics`` closure that reloads
    just the deploy-relevant weights per call.

    Mirrors two existing, unedited call sites rather than reinventing them:
      * env construction / seeding -- ``indist.py``'s standalone harness
        (``_build_and_eval``): ``auto_reset=True``, ``debug=False``, fixed
        ``env_cfg.seed``; command pinning + curriculum freeze happen inside
        ``run_indist_eval`` itself, unchanged.
      * per-checkpoint weight loading -- ``campaign.py``'s ``build_session``:
        ``train_cfg.runner.resume = False`` (skip the auto latest/best load),
        then ``runner.load_deploy_state(path, ...)`` (strict, method-scoped:
        actor-only for MLP/P5/HIM, +history/VAE/estimator for adaptive
        methods) and ``runner.get_eval_adapter(device=...)`` so RMA/DreamWaQ/
        SysID's multi-tensor obs contract is respected, not just the
        single-tensor default ``run_indist_eval`` falls back to.

    One env/actor-critic is built ONCE and reused across every candidate
    iteration of this run (``gs.init`` + env construction dominate cost;
    reloading deploy weights per iteration is cheap), rather than paying the
    full build per checkpoint as ``indist.py --ckpt <iter>`` invoked in a loop
    would.
    """
    import re
    from types import SimpleNamespace

    from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR  # noqa: F401
    import legged_gym.envs  # noqa: F401  (import side effect: registers tasks)
    from legged_gym.scripts.eval.indist import (
        _command_support_scale,
        in_dist_command_ranges,
        run_indist_eval,
    )
    from legged_gym.scripts.eval.provenance import verify_run_identity
    from legged_gym.utils import task_registry

    try:
        import genesis as gs
    except Exception:
        gs = None

    if SIMULATOR == "genesis" and gs is not None:
        gs.init(backend=gs.cpu if cpu else gs.gpu, logging_level="warning")

    run_dir = Path(run_dir)
    seed_match = re.search(r"_seed(\d+)$", run_dir.name)
    verify_run_identity(
        str(run_dir),
        expected_task=task,
        expected_training_seed=int(seed_match.group(1)) if seed_match else None,
    )

    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    if num_envs is not None:
        env_cfg.env.num_envs = int(num_envs)
    env_cfg.env.auto_reset = True
    env_cfg.env.debug = False
    env_cfg.seed = int(eval_seed)

    reg_args = SimpleNamespace(
        task=task, headless=True, cpu=cpu, num_envs=None, max_iterations=None,
        resume=False, sync_wandb=False, export_onnx=False, debug=False,
        load_run=run_dir.name, ckpt=-1, use_joystick=False, joystick_type="xbox",
        follow_robot=False, viewer="native", viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )
    env, _ = task_registry.make_env(name=task, args=reg_args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(env=env, name=task, args=reg_args, train_cfg=train_cfg)

    v_scale = _command_support_scale(in_dist_command_ranges(env), "lin_vel_x")

    def eval_fn(checkpoint_path: Path, iteration: int) -> Dict[str, float]:
        runner.load_deploy_state(str(checkpoint_path), map_location=env.device)
        adapter = runner.get_eval_adapter(device=env.device)
        metrics = run_indist_eval(env, steps=steps, warmup=warmup, seed=eval_seed, adapter=adapter)
        metrics["spnte_v_scale"] = v_scale
        return metrics

    return eval_fn


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline best_spnte.pt selector for v4-terrain runs: scores each "
            "existing model_<iter>.pt with the indist eval (spnte_lin), "
            "publishes the minimum as best_spnte.pt. Never touches "
            "best_tracking.pt or the v5 UED validation bank."
        ),
    )
    parser.add_argument("--run-dir", required=True, help="specific run directory, e.g. "
                         "logs/go2_v4_terrain_curriculum/<run>_seed1 (NOT the experiment root)")
    parser.add_argument("--task", required=True, help="registered task, e.g. go2_v4_mlp")
    parser.add_argument("--num-envs", type=int, default=None, help="override env count (default: cfg value)")
    parser.add_argument("--eval-seed", type=int, default=12345, help="fixed eval seed (default: 12345)")
    parser.add_argument("--steps", type=int, default=2000, help="measured steps per candidate (default: 2000)")
    parser.add_argument("--warmup", type=int, default=100, help="unrecorded settling steps (default: 100)")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--min-iteration", type=int, default=1000,
                         help="schedule floor, mirrors v5_ued.yaml selection.min_iteration "
                              "(default: 1000; ignored if --iterations is given)")
    parser.add_argument("--stride", type=int, default=500,
                         help="schedule stride, mirrors v5_ued.yaml selection.iteration_stride "
                              "(default: 500; ignored if --iterations is given)")
    parser.add_argument("--iterations", type=int, nargs="+", default=None,
                         help="explicit iteration list; overrides --min-iteration/--stride entirely")
    args = parser.parse_args(argv)

    winner, best_path, sidecar_path, scores = run_selection(
        args.run_dir, args.task,
        num_envs=args.num_envs, eval_seed=args.eval_seed, steps=args.steps, warmup=args.warmup,
        cpu=args.cpu, iterations=args.iterations, min_iteration=args.min_iteration, stride=args.stride,
    )

    print(f"\n=== best_spnte selection | {args.task} | run={args.run_dir} ===")
    print(format_summary_table(scores, winner))
    print(f"\nwinner: iter {winner.iteration}  spnte_lin={winner.metrics['spnte_lin']:.6f}")
    print(f"best_spnte.pt -> {best_path}")
    print(f"selection sidecar -> {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
