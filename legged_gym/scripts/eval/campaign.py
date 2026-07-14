"""Deterministic, provenance-first Eval V2 campaign runner for Genesis.

The module intentionally owns the V2 artifact contract instead of trying to
reinterpret Phase A/B files.  A cell is always one selected checkpoint and one
scenario bundle; it writes per-environment values to a sibling ``.partial.npz``
and promotes it only after a finite-value and identity check.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml

from legged_gym.scripts.eval.ckpt_utils import resolve_checkpoint_path, sha256_file
from legged_gym.scripts.eval.dr_axes import AXES, get_axis, pin_others_to_nominal
from legged_gym.scripts.eval.metrics import MetricAccumulator
from legged_gym.scripts.eval.provenance import atomic_savez, git_commit, verify_run_identity


METRICS = ("tracking_lin", "tracking_yaw", "fall_rate", "return_per_step")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("campaign config must be a mapping")
    required = ("campaign", "artifact_root", "simulator", "models", "protocol", "axes")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"campaign config missing required keys: {missing}")
    if cfg["simulator"] != "genesis":
        raise ValueError("Eval V2 currently supports simulator: genesis only")
    labels = [m.get("label") for m in cfg["models"]]
    if len(labels) != len(set(labels)) or any(not x for x in labels):
        raise ValueError("models must have unique non-empty labels")
    if set(cfg["axes"]) != set(AXES):
        raise ValueError(f"axes must be exactly {sorted(AXES)}, got {sorted(cfg['axes'])}")
    return cfg


def artifact_root(cfg: Dict[str, Any]) -> Path:
    p = Path(cfg["artifact_root"])
    return p if p.is_absolute() else _root() / p


def parse_shard(value: str) -> Tuple[int, int]:
    """Parse the deterministic ``i/n`` partition used by campaign commands."""
    m = re.fullmatch(r"(\d+)/(\d+)", str(value).strip())
    if not m:
        raise ValueError(f"invalid shard {value!r}; expected i/n, e.g. 0/2")
    index, count = (int(x) for x in m.groups())
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"invalid shard {value!r}; need 0 <= i < n and n >= 1")
    return index, count


def shard_items(items: List[Any], shard: Tuple[int, int]) -> List[Any]:
    """Stable round-robin partition; adding workers never reorders cell IDs."""
    index, count = shard
    return [item for i, item in enumerate(items) if i % count == index]


def _artifact_valid(path: Path) -> bool:
    """Minimal resume gate: only complete, finite V2 payloads are reusable."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return all(key in z and np.isfinite(np.asarray(z[key], dtype=float)).all()
                       for key in METRICS)
    except (OSError, ValueError, KeyError):
        return False


def _expected_raw_relpaths(cfg: Dict[str, Any]) -> List[str]:
    """Every final raw artifact, including terrain's one-file-per-severity shape."""
    paths: List[str] = []
    terrain_cm = cfg["protocol"]["hard"]["terrain_cm"]
    for kind, model, seed, name in _planned_cells(cfg, "all"):
        if kind == "hard" and name == "terrain":
            paths.extend(_cell_path("hard", model["label"], seed,
                                    f"terrain_{str(cm).replace('.', '_')}cm") + ".npz"
                         for cm in terrain_cm)
        else:
            output = (name.replace(":", "_") if kind == "axes" else
                      "step_reversal" if kind == "hard" and name == "step" else name)
            paths.append(_cell_path(kind, model["label"], seed, output) + ".npz")
    return paths


def _validation_relpath(model: str, seed: int, iteration: int) -> str:
    return _cell_path("validation", model, seed, f"model_{iteration}") + ".npz"


def _model_seeds(cfg: Dict[str, Any], model: Dict[str, Any]) -> List[int]:
    """Return model-local seeds when supplied, otherwise the campaign seeds.

    Adaptive smoke runs are often produced sequentially with distinct seed IDs;
    allowing a model-local list keeps one campaign config honest without fake
    run-folder aliases. Full campaigns continue to use the shared global list.
    """
    return [int(seed) for seed in model.get("training_seeds", cfg["training_seeds"])]


def _expected_validation_relpaths(cfg: Dict[str, Any]) -> List[str]:
    """Checkpoint-validation artifacts discovered from the declared run folders."""
    paths = []
    for model in cfg["models"]:
        for seed in _model_seeds(cfg, model):
            run = _run_dir(model, seed)
            candidates = sorted(run.glob("model_*.pt"), key=_checkpoint_iter)
            if not candidates and cfg.get("smoke_only"):
                candidates = [run / "best.pt"]
            if not candidates:
                # Selection itself gives the detailed fail-loud diagnostic.
                continue
            paths.extend(_validation_relpath(model["label"], seed, _checkpoint_iter(path))
                         for path in candidates)
    return paths


def _write_campaign_state(cfg: Dict[str, Any]) -> None:
    """Materialize reproducibility inventory after every execution-stage command."""
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    (root / "campaign.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    expected_final = _expected_raw_relpaths(cfg)
    expected_validation = _expected_validation_relpaths(cfg)
    expected = expected_validation + expected_final
    cells = []
    for rel in expected:
        path = root / "raw" / rel
        exists = _artifact_valid(path)
        cells.append(dict(path=f"raw/{rel}", complete=exists,
                          sha256=sha256_file(str(path)) if exists else None))
    manifest = dict(campaign=cfg["campaign"], generated_at=datetime.now(timezone.utc).isoformat(),
                    expected_validation_artifacts=len(expected_validation),
                    expected_final_artifacts=len(expected_final),
                    expected_raw_artifacts=len(expected), completed_raw_artifacts=sum(c["complete"] for c in cells),
                    complete=all(c["complete"] for c in cells), cells=cells)
    (root / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    with (root / "ledger.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("path", "complete", "sha256"))
        writer.writeheader(); writer.writerows(cells)


def _record_command(cfg: Dict[str, Any], command: str, shard: str, exit_code: int) -> None:
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    row = dict(timestamp=datetime.now(timezone.utc).isoformat(), command=command,
               shard=shard, exit_code=exit_code, argv=sys.argv[1:])
    with (root / "commands.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _git_dirty() -> Tuple[bool, str]:
    raw = subprocess.check_output(["git", "status", "--porcelain"], cwd=_root()).decode()
    return bool(raw.strip()), hashlib.sha256(raw.encode()).hexdigest()


def _run_dir(model: Dict[str, Any], seed: int) -> Path:
    value = (model.get("run_folders") or {}).get(seed, (model.get("run_folders") or {}).get(str(seed)))
    if not value or str(value).startswith("TODO"):
        raise FileNotFoundError(f"{model['label']} seed {seed}: run_folders entry is missing/TODO")
    p = Path(value)
    return p if p.is_absolute() else _root() / "logs" / "go2_benchmark" / p


def _checkpoint_iter(path: Path) -> int:
    m = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(m.group(1)) if m else -1


def selection_order(rows: Iterable[Dict[str, float]], fall_threshold: float = 0.05) -> Dict[str, float]:
    """V2 lexicographic rule; exposed for CPU-only unit tests."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot select from no checkpoint rows")
    safe = [r for r in rows if float(r["fall_rate"]) <= fall_threshold]
    key = (lambda r: (float(r["tracking_score"]), int(r["iteration"]))) if safe else \
        (lambda r: (float(r["fall_rate"]), float(r["tracking_score"]), int(r["iteration"])))
    return min(safe or rows, key=key)


def delay_ms(control_steps: float, control_dt: float = 0.02) -> float:
    return float(control_steps) * control_dt * 1000.0


def _scenario_commands(n: int, seed: int) -> np.ndarray:
    """A fixed stratified ID bank: corners, near-zero points, then seeded fill."""
    rng = np.random.default_rng(seed)
    anchors = np.asarray([[x, y, z] for x in (-1., 0., 1.) for y in (-1., 0., 1.) for z in (-1., 0., 1.)])
    out = rng.uniform(-1., 1., size=(n, 3)).astype(np.float32)
    out[:min(n, len(anchors))] = anchors[:min(n, len(anchors))]
    return out


def _registry_args(task: str, run_folder: str, ckpt: int | str) -> SimpleNamespace:
    return SimpleNamespace(task=task, headless=True, cpu=False, num_envs=None, max_iterations=None,
        resume=False, sync_wandb=False, export_onnx=False, debug=False, load_run=run_folder,
        ckpt=ckpt, use_joystick=False, joystick_type="xbox", follow_robot=False,
        viewer="native", viser_port=8080, motion_file=None, motion_out_dir=None, num_student=None)


@dataclass
class Session:
    env: Any
    adapter: Any
    runner: Any
    checkpoint: Path
    run_dir: Path
    model: Dict[str, Any]
    seed: int


def _load_actor(actor, checkpoint: Path, device) -> None:
    """Strict actor-only load; critic layouts may legitimately differ over time."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"]
    actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
    if not actor_state:
        raise RuntimeError(f"{checkpoint}: checkpoint contains no actor.* tensors")
    missing, unexpected = actor.load_state_dict(actor_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"{checkpoint}: actor mismatch missing={missing} unexpected={unexpected}")


def build_session(
    model: Dict[str, Any], seed: int, num_envs: int, eval_seed: int, *,
    delay: bool = False, id_domain_rand: bool = False, terrain_raw: np.ndarray | None = None,
    terrain_horizontal_scale: float = 0.1,
) -> Session:
    """Build exactly one task/policy pair, preserving checkpoint/task isolation."""
    import legged_gym.envs  # noqa: F401  (task-registration side effect)
    from legged_gym.utils import task_registry

    run_dir = _run_dir(model, seed)
    verify_run_identity(str(run_dir), expected_task=model["task"], expected_training_seed=seed, require_manifest=True)
    ckpt_spec = model.get("checkpoint", "best")
    checkpoint = Path(resolve_checkpoint_path(str(run_dir), ckpt_spec))
    # Registry configs are mutable singleton instances.  A campaign cell must
    # never leak terrain, delay, or randomization overrides into the next cell.
    env_cfg, train_cfg = (copy.deepcopy(x) for x in task_registry.get_cfgs(name=model["task"]))
    env_cfg.env.num_envs = int(num_envs)
    env_cfg.env.auto_reset = True
    env_cfg.env.debug = False
    env_cfg.seed = int(eval_seed)
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1e6
    if not id_domain_rand:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_com_displacement = False
        env_cfg.domain_rand.push_robots = False
    if terrain_raw is not None:
        env_cfg.terrain.mesh_type = "heightfield"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.terrain_length = 8.0
        env_cfg.terrain.terrain_width = 8.0
        env_cfg.terrain.border_size = 2.0
        env_cfg.terrain.horizontal_scale = float(terrain_horizontal_scale)
        env_cfg.terrain.vertical_scale = 0.0025
        env_cfg.terrain.height_field_raw_override = terrain_raw
    if delay:
        get_axis("control_delay").prepare_cfg(env_cfg)
    run_folder = run_dir.name
    args = _registry_args(model["task"], run_folder, ckpt_spec)
    env, _ = task_registry.make_env(name=model["task"], args=args, env_cfg=env_cfg)
    # Build without runner resume, then strictly load every component consumed by
    # deployed inference. The runner owns the method contract (actor-only for
    # MLP/P5, actor+history/VAE/estimator for adaptive methods), while critic
    # layout changes remain irrelevant to evaluation.
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(env=env, name=model["task"], args=args, train_cfg=train_cfg)
    runner.load_deploy_state(str(checkpoint), map_location=env.device)
    adapter = runner.get_eval_adapter(device=env.device)
    return Session(env, adapter, runner, checkpoint, run_dir, model, seed)


def _set_nominal(env) -> None:
    # Pin privileged P dimensions first, then the held-out axes. This keeps the
    # P5 actor's labels consistent even in a non-P sweep.
    pin_others_to_nominal(env, "friction")
    get_axis("friction").pin_nominal(env)
    get_axis("pd_gain_scale").pin_nominal(env)
    if hasattr(env.simulator, "_action_queue"):
        get_axis("control_delay").pin_nominal(env)


def _terrain_raw(amplitude_m: float, seed: int, horizontal_scale: float = 0.1) -> np.ndarray:
    """One deterministic bumpy map, scaled exactly to ``amplitude_m``.

    The 8m terrain plus 2m border is represented at the requested resolution.
    A coarse fixed field is nearest-neighbour expanded so every severity shares
    the same spatial pattern.  Values are expressed in 2.5mm raw units.
    """
    rng = np.random.default_rng(seed)
    interior = int(round(8.0 / horizontal_scale))
    border = int(round(2.0 / horizontal_scale))
    coarse_side = max(4, interior // 4)
    # Generate a coarse base field then expand it exactly to the interior.
    coarse = rng.uniform(-1.0, 1.0, size=(coarse_side, coarse_side))
    coarse /= np.max(np.abs(coarse))
    field = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)[:interior, :interior]
    raw = np.zeros((interior + 2 * border, interior + 2 * border), dtype=np.int16)
    raw[border:-border, border:-border] = np.rint(field * (amplitude_m / 0.0025)).astype(np.int16)
    return raw


def _terrain_hash(raw: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()


@torch.no_grad()
def rollout(session: Session, commands: np.ndarray, *, warmup: int, steps: int, schedule_every: int = 500, seed: int | None = None) -> Dict[str, np.ndarray]:
    env, adapter = session.env, session.adapter
    command_t = torch.as_tensor(commands, device=env.device, dtype=torch.float)
    if command_t.shape != (env.num_envs, 3):
        raise ValueError(f"commands must be ({env.num_envs}, 3), got {tuple(command_t.shape)}")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    state = adapter.reset(env)
    for _ in range(warmup):
        env.commands[:, :3] = command_t
        state, _, _ = adapter.step(env, adapter.act(state).detach())
    env.episode_length_buf.zero_()
    acc = MetricAccumulator(env.num_envs, env.device)
    for t in range(steps):
        if t and t % schedule_every == 0:
            # deterministic circular schedule: no policy-dependent RNG draws.
            command_t = command_t.roll(shifts=1, dims=0)
        env.commands[:, :3] = command_t
        state, rew, dones = adapter.step(env, adapter.act(state).detach())
        lin = torch.linalg.vector_norm(env.commands[:, :2] - env.simulator.base_lin_vel[:, :2], dim=1)
        yaw = (env.commands[:, 2] - env.simulator.base_ang_vel[:, 2]).abs()
        acc.update(rew, dones, env.time_out_buf, lin, yaw)
    raw = acc.compute()
    return {
        "tracking_lin": raw["tracking_lin_err"].cpu().numpy(),
        "tracking_yaw": raw["tracking_ang_err"].cpu().numpy(),
        "fall_rate": raw["ever_fell"].cpu().numpy(),
        "return_per_step": raw["return_per_step"].cpu().numpy(),
    }


@torch.no_grad()
def _rollout_schedule(
    session: Session, command_schedule: torch.Tensor, *, warmup_command: np.ndarray,
    warmup: int, kick_at: int | None = None, dvy: torch.Tensor | None = None,
    phase_bounds: List[Tuple[int, int]] | None = None,
) -> Dict[str, np.ndarray]:
    """V2 scheduler for hard scenarios, with optional per-env lateral kick."""
    env, adapter = session.env, session.adapter
    if command_schedule.ndim != 3 or command_schedule.shape[1:] != (env.num_envs, 3):
        raise ValueError(f"command schedule must be (T,{env.num_envs},3), got {tuple(command_schedule.shape)}")
    warm = torch.as_tensor(warmup_command, dtype=torch.float, device=env.device)
    state = adapter.reset(env)
    for _ in range(warmup):
        env.commands[:, :3] = warm
        state, _, _ = adapter.step(env, adapter.act(state).detach())
    env.episode_length_buf.zero_()
    acc = MetricAccumulator(env.num_envs, env.device)
    phase_accs = [MetricAccumulator(env.num_envs, env.device) for _ in (phase_bounds or [])]
    for t in range(command_schedule.shape[0]):
        env.commands[:, :3] = command_schedule[t]
        if kick_at is not None and t == kick_at:
            sim = env.simulator
            qvel = sim._robot.get_dofs_velocity()
            qvel[:, 1] += dvy
            sim._robot.set_dofs_velocity(qvel)
        state, rew, dones = adapter.step(env, adapter.act(state).detach())
        lin = torch.linalg.vector_norm(env.commands[:, :2] - env.simulator.base_lin_vel[:, :2], dim=1)
        yaw = (env.commands[:, 2] - env.simulator.base_ang_vel[:, 2]).abs()
        acc.update(rew, dones, env.time_out_buf, lin, yaw)
        for (lo, hi), phase_acc in zip(phase_bounds or [], phase_accs):
            if lo <= t < hi:
                phase_acc.update(rew, dones, env.time_out_buf, lin, yaw)
    raw = acc.compute()
    out = {"tracking_lin": raw["tracking_lin_err"].cpu().numpy(), "tracking_yaw": raw["tracking_ang_err"].cpu().numpy(),
           "fall_rate": raw["ever_fell"].cpu().numpy(), "return_per_step": raw["return_per_step"].cpu().numpy()}
    if phase_accs:
        for key, raw_key in (("tracking_lin", "tracking_lin_err"), ("tracking_yaw", "tracking_ang_err"),
                             ("fall_rate", "ever_fell"), ("return_per_step", "return_per_step")):
            out[f"phase_{key}"] = np.stack([a.compute()[raw_key].cpu().numpy() for a in phase_accs])
    return out


def _meta(session: Session, suite: str, scenario: str, eval_seed: int, warmup: int, steps: int) -> Dict[str, Any]:
    dirty, diff_hash = _git_dirty()
    return dict(campaign="", suite=suite, scenario=scenario, model=session.model["label"], task=session.model["task"],
        training_seed=session.seed, checkpoint_path=str(session.checkpoint), checkpoint_iter=_checkpoint_iter(session.checkpoint),
        checkpoint_sha256=sha256_file(str(session.checkpoint)), run_folder=session.run_dir.name, eval_seed=eval_seed,
        warmup=warmup, steps=steps, eval_commit=git_commit(), git_dirty=dirty, git_diff_sha256=diff_hash,
        simulator="genesis")


def _save_cell(root: Path, rel: str, payload: Dict[str, Any]) -> Path:
    final = root / "raw" / f"{rel}.npz"
    partial = final.with_suffix(".partial.npz")
    for key in ("tracking_lin", "tracking_yaw", "fall_rate", "return_per_step"):
        if key not in payload or not np.isfinite(np.asarray(payload[key], dtype=float)).all():
            raise RuntimeError(f"refusing invalid artifact {rel}: missing/non-finite {key}")
    atomic_savez(str(partial), **payload)
    os.replace(partial, final)
    return final


def _cell_path(suite: str, model: str, seed: int, name: str) -> str:
    return f"{suite}/{model}/seed_{seed}/{name}"


def _planned_cells(cfg: Dict[str, Any], suite: str) -> List[Tuple[str, Dict[str, Any], int, str]]:
    out = []
    suites = ("id", "axes", "hard") if suite == "all" else (suite,)
    for model in cfg["models"]:
        for seed in _model_seeds(cfg, model):
            if "id" in suites:
                out.append(("id", model, seed, "id"))
            if "axes" in suites:
                for axis in cfg["axes"]:
                    for movement in ("forward", "lateral", "yaw"):
                        out.append(("axes", model, seed, f"{axis}:{movement}"))
            if "hard" in suites:
                out.extend(("hard", model, seed, name) for name in ("terrain", "kick", "step"))
    return out


def cmd_plan(cfg: Dict[str, Any], suite: str) -> int:
    cells = _planned_cells(cfg, suite)
    expected = _expected_raw_relpaths(cfg) if suite == "all" else None
    print(_json(dict(campaign=cfg["campaign"], process_cells=len(cells), raw_artifacts=len(expected) if expected else None, suites=suite,
        id_envs=cfg["protocol"]["id"]["num_envs"], axis_max_envs=max(len(v["grid"]) for v in cfg["axes"].values()) * 4 * cfg["protocol"]["axes"]["replicas"],
        cells_preview=[f"{a}:{b['label']}:seed{s}:{n}" for a,b,s,n in cells[:12]])))
    return 0


def cmd_select(cfg: Dict[str, Any], shard: Tuple[int, int]) -> int:
    """Selection is deliberately conservative: this command will not silently use best.pt."""
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    rows = []; protocol = cfg["protocol"]["validation"]
    work = [(model, seed) for model in cfg["models"] for seed in _model_seeds(cfg, model)]
    assigned = shard_items(work, shard)
    print(f"[selection] shard={shard[0]}/{shard[1]} runs={len(assigned)} "
          f"protocol={protocol['num_envs']}env×{protocol['steps']}steps")
    for run_index, (model, seed) in enumerate(assigned, start=1):
        run = _run_dir(model, seed)
        candidates = sorted(run.glob("model_*.pt"), key=_checkpoint_iter)
        if not candidates and cfg.get("smoke_only"):
            candidates = [run / "best.pt"]
        if not candidates:
            raise FileNotFoundError(f"{run}: no model_*.pt checkpoints for offline V2 selection")
        if any(not p.is_file() for p in candidates):
            raise FileNotFoundError(f"{run}: smoke checkpoint best.pt is missing")
        print(f"[selection] run {run_index}/{len(assigned)} {model['label']} seed={seed} "
              f"candidates={len(candidates)}")
        # A single Genesis environment serves every checkpoint in this run.
        # Loading only the actor is strict and avoids paying scene-build cost
        # 16 times; the deterministic bank/reset seed is reapplied below.
        first = candidates[0]
        first_spec = "best" if first.name == "best.pt" else _checkpoint_iter(first)
        initial = dict(model); initial["checkpoint"] = first_spec
        from legged_gym import gs
        if not getattr(cmd_select, "_gs_ready", False):
            gs.init(backend=gs.gpu, logging_level="warning"); cmd_select._gs_ready = True
        session = build_session(initial, seed, protocol["num_envs"], protocol["seed"], id_domain_rand=True)
        commands = _scenario_commands(session.env.num_envs, protocol["seed"])
        for candidate_index, ckpt in enumerate(candidates, start=1):
            # Re-use the V2 deterministic ID bank but force this exact model file.
            spec = "best" if ckpt.name == "best.pt" else _checkpoint_iter(ckpt)
            iteration = _checkpoint_iter(ckpt)
            rel = _validation_relpath(model["label"], seed, iteration)
            print(f"[selection] {model['label']} seed={seed} checkpoint "
                  f"{candidate_index}/{len(candidates)} iter={spec}", flush=True)
            artifact = root / "raw" / rel
            expected_sha = sha256_file(str(ckpt))
            if _artifact_valid(artifact):
                with np.load(artifact, allow_pickle=False) as z:
                    recorded_sha = str(_npz_scalar(z, "checkpoint_sha256"))
                    if recorded_sha != expected_sha:
                        raise RuntimeError(f"{artifact}: checkpoint SHA mismatch; refusing stale validation")
                    vals = {key: np.asarray(z[key]) for key in METRICS}
                print(f"[selection] resume {artifact.name}", flush=True)
            else:
                session.runner.load_deploy_state(str(ckpt), map_location=session.env.device)
                session.checkpoint = ckpt
                vals = rollout(session, commands, warmup=protocol["warmup"], steps=protocol["steps"], seed=protocol["seed"])
                payload = _meta(session, "validation", "checkpoint_selection", protocol["seed"],
                                protocol["warmup"], protocol["steps"])
                payload.update(campaign=cfg["campaign"], checkpoint_spec=spec,
                               tracking_score=float(.5 * (vals["tracking_lin"].mean() / np.sqrt(2.)
                                                     + vals["tracking_yaw"].mean())),
                               command_matrix=commands,
                               scenario_hash=hashlib.sha256(commands.tobytes()).hexdigest(),
                               validation_bank="stratified_v1", replicas=1, **vals)
                _save_cell(root, rel[:-4], payload)
            lin, yaw, fall = (float(vals[k].mean()) for k in ("tracking_lin", "tracking_yaw", "fall_rate"))
            rows.append(dict(model=model["label"], training_seed=seed, path=str(ckpt), checkpoint=spec, iteration=iteration, sha256=expected_sha,
                tracking_lin=lin, tracking_yaw=yaw, fall_rate=fall, tracking_score=.5 * (lin / np.sqrt(2.) + yaw), protocol=protocol))
    selected = [selection_order([r for r in rows if r["model"] == model["label"] and r["training_seed"] == seed])
                for model, seed in shard_items(work, shard)]
    part = root / f"checkpoint_selection.shard_{shard[0]}_of_{shard[1]}.json"
    part.write_text(_json(dict(campaign=cfg["campaign"], shard=f"{shard[0]}/{shard[1]}", fall_threshold=.05,
                               selections=selected)), encoding="utf-8")
    parts = sorted(root.glob(f"checkpoint_selection.shard_*_of_{shard[1]}.json"))
    merged = [row for path in parts for row in json.loads(path.read_text())["selections"]]
    unique = {(row["model"], int(row["training_seed"])): row for row in merged}
    canonical = dict(campaign=cfg["campaign"], fall_threshold=.05,
                     expected_selections=len(work), completed_selections=len(unique),
                     complete=len(unique) == len(work), selections=[unique[k] for k in sorted(unique)])
    (root / "checkpoint_selection.json").write_text(_json(canonical), encoding="utf-8")
    print(f"[selection] selections={len(selected)} canonical="
          f"{canonical['completed_selections']}/{canonical['expected_selections']}")
    _write_campaign_state(cfg)
    return 0


def _materialize_training_best_selection(cfg: Dict[str, Any], root: Path) -> Path:
    """Write checkpoint_selection.json from each run's training-time best without offline re-eval.

    Offline ``select-checkpoints`` remains available when explicitly requested.  By
    default the final suite should use the already-selected ``best_tracking.pt``
    (or the model config's ``checkpoint`` field) from training.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / "checkpoint_selection.json"
    work = [(model, seed) for model in cfg["models"] for seed in _model_seeds(cfg, model)]
    selections = []
    for model, seed in work:
        run = _run_dir(model, seed)
        ckpt_spec = model.get("checkpoint", "best")
        ckpt = Path(resolve_checkpoint_path(str(run), ckpt_spec))
        iteration = _checkpoint_iter(ckpt)
        if iteration < 0:
            # best_tracking.pt / best.pt: prefer embedded training selection iter
            try:
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                iteration = int(payload.get("iter", -1))
            except Exception:
                iteration = -1
        selections.append(dict(
            model=model["label"],
            training_seed=int(seed),
            path=str(ckpt),
            checkpoint=ckpt_spec if isinstance(ckpt_spec, str) else int(ckpt_spec),
            iteration=iteration,
            sha256=sha256_file(str(ckpt)),
            source="training_best",
            note="materialized without offline select-checkpoints re-eval",
        ))
    canonical = dict(
        campaign=cfg["campaign"],
        fall_threshold=float(cfg.get("fall_threshold", 0.05)),
        expected_selections=len(work),
        completed_selections=len(selections),
        complete=True,
        source="training_best",
        selections=selections,
    )
    path.write_text(_json(canonical), encoding="utf-8")
    print(f"[selection] using training-time best checkpoints "
          f"({len(selections)} models×seeds) -> {path}")
    return path


def _selected_config(cfg: Dict[str, Any], root: Path, model: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Resolve the deploy checkpoint for one model/seed.

    Default: use training-time ``best`` / ``best_tracking.pt`` from the model
    config (no offline select).  If ``checkpoint_selection.json`` already exists
    (from an explicit ``select-checkpoints`` run or a prior materialize), honor it.
    """
    path = root / "checkpoint_selection.json"
    # Offline select is opt-in.  Absent a selection file, materialize from
    # training best so ``campaign run`` works without re-validating every model_*.pt.
    if not path.exists():
        if cfg.get("require_offline_select", False):
            raise FileNotFoundError(
                f"{path}: offline select required (require_offline_select=true); "
                "run select-checkpoints first")
        _materialize_training_best_selection(cfg, root)
    data = json.loads(path.read_text())
    found = [x for x in data["selections"] if x["model"] == model["label"] and int(x["training_seed"]) == int(seed)]
    if len(found) != 1:
        raise RuntimeError(f"selection manifest has {len(found)} entries for {model['label']} seed {seed}")
    m = dict(model); m["checkpoint"] = found[0].get("checkpoint", int(found[0]["iteration"]))
    return m


def _run_id(root: Path, cfg: Dict[str, Any], model: Dict[str, Any], seed: int, *, resume: bool) -> None:
    p = cfg["protocol"]["id"]; m = _selected_config(cfg, root, model, seed)
    rel = _cell_path("id", m["label"], seed, "id")
    if resume and _artifact_valid(root / "raw" / f"{rel}.npz"): return
    session = build_session(m, seed, p["num_envs"], p["seed"], id_domain_rand=True)
    payload = _meta(session, "id", "id", p["seed"], p["warmup"], p["steps"])
    payload["campaign"] = cfg["campaign"]
    payload.update(rollout(session, _scenario_commands(session.env.num_envs, p["seed"]), warmup=p["warmup"], steps=p["steps"]))
    payload.update(command_matrix=_scenario_commands(session.env.num_envs, p["seed"]), scenario_hash=hashlib.sha256(_scenario_commands(session.env.num_envs, p["seed"]).tobytes()).hexdigest())
    _save_cell(root, rel, payload)


def _run_axis(root: Path, cfg: Dict[str, Any], model: Dict[str, Any], seed: int, axis_name: str, movement: str, *, resume: bool) -> None:
    p = cfg["protocol"]["axes"]; axis_cfg = cfg["axes"][axis_name]; m = _selected_config(cfg, root, model, seed)
    rel = _cell_path("axes", m["label"], seed, f"{axis_name}_{movement}")
    if resume and _artifact_valid(root / "raw" / f"{rel}.npz"): return
    grid = np.asarray(axis_cfg["grid"], dtype=np.float32); mags = np.asarray(p["magnitudes"], dtype=np.float32)
    directions = {"forward": (1,0,0), "lateral": (0,1,0), "yaw": (0,0,1)}
    if movement not in directions: raise ValueError(f"unknown axis movement {movement}")
    n = len(grid) * len(mags) * p["replicas"]
    session = build_session(m, seed, n, p["seed"], delay=axis_name == "control_delay")
    _set_nominal(session.env)
    vals = np.repeat(grid, len(mags) * p["replicas"])
    axis = get_axis(axis_name); requested = torch.as_tensor(vals, device=session.env.device)
    axis.apply(session.env, requested); actual = axis.validate(session.env, requested).cpu().numpy()
    commands = np.zeros((n,3), np.float32); modes_col=[]; mag_col=[]
    i=0
    for value in grid:
        for mag in mags:
            commands[i:i+p["replicas"]] = np.asarray(directions[movement], np.float32)*mag; modes_col.extend([movement]*p["replicas"]); mag_col.extend([mag]*p["replicas"]); i += p["replicas"]
    payload = _meta(session, "axes", axis_name, p["seed"], p["warmup"], p["steps"])
    payload["campaign"] = cfg["campaign"]
    payload.update(rollout(session, commands, warmup=p["warmup"], steps=p["steps"]), axis=axis_name, axis_value=vals, axis_actual=actual,
        axis_is_id=(vals >= axis_cfg["in_distribution"][0]) & (vals <= axis_cfg["in_distribution"][1]), command_mode=np.asarray(modes_col), command_magnitude=np.asarray(mag_col), command_is_id=np.asarray(mag_col) <= 1.0,
        replicas=p["replicas"], delay_ms=np.asarray([delay_ms(v) if axis_name == "control_delay" else np.nan for v in vals]))
    _save_cell(root, rel, payload)


def _run_hard_terrain(root: Path, cfg: Dict[str, Any], model: Dict[str, Any], seed: int, *, resume: bool) -> None:
    p, m = cfg["protocol"]["hard"], _selected_config(cfg, root, model, seed)
    replicas = int(p.get("terrain_replicas", 256))
    for cm in p["terrain_cm"]:
        rel = _cell_path("hard", m["label"], seed, f"terrain_{str(cm).replace('.', '_')}cm")
        if resume and _artifact_valid(root / "raw" / f"{rel}.npz"):
            continue
        hscale = float(p.get("terrain_horizontal_scale", 0.1))
        raw = _terrain_raw(float(cm) / 100.0, int(p["seed"]), hscale)
        session = build_session(m, seed, replicas, p["seed"], terrain_raw=raw, terrain_horizontal_scale=hscale)
        _set_nominal(session.env)
        commands = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (replicas, 1))
        payload = _meta(session, "hard", "terrain", p["seed"], p["warmup"], p["steps"])
        payload.update(campaign=cfg["campaign"], severity=float(cm), terrain_max_abs_m=float(cm) / 100.0,
            terrain_hash=_terrain_hash(raw), terrain_raw=raw, command_mode="forward", command_magnitude=1.0,
            **rollout(session, commands, warmup=p["warmup"], steps=p["steps"]))
        _save_cell(root, rel, payload)


def _run_hard_kick(root: Path, cfg: Dict[str, Any], model: Dict[str, Any], seed: int, *, resume: bool) -> None:
    p, m = cfg["protocol"]["hard"], _selected_config(cfg, root, model, seed)
    rel = _cell_path("hard", m["label"], seed, "kick")
    if resume and _artifact_valid(root / "raw" / f"{rel}.npz"):
        return
    levels = np.asarray(p["lateral_kick_mps"], dtype=np.float32)
    replicas = int(p.get("kick_replicas", 256))
    n = len(levels) * replicas
    pre = int(p.get("kick_step", 50)); recovery = int(p.get("recovery_steps", 250)); total = pre + recovery
    session = build_session(m, seed, n, p["seed"])
    _set_nominal(session.env)
    schedule = torch.zeros((total, n, 3), device=session.env.device)
    schedule[:, :, 0] = 1.0
    dvy = torch.as_tensor(np.repeat(levels, replicas), device=session.env.device)
    out = _rollout_schedule(session, schedule, warmup_command=np.asarray([1., 0., 0.]), warmup=p["warmup"], kick_at=pre, dvy=dvy)
    payload = _meta(session, "hard", "lateral_velocity_kick", p["seed"], p["warmup"], total)
    payload.update(campaign=cfg["campaign"], severity=np.repeat(levels, replicas), lateral_velocity_kick_mps=np.repeat(levels, replicas),
        recovery_window_steps=recovery, kick_step=pre, command_mode="forward", command_magnitude=1.0, **out)
    _save_cell(root, rel, payload)


def _run_hard_step(root: Path, cfg: Dict[str, Any], model: Dict[str, Any], seed: int, *, resume: bool) -> None:
    p, m = cfg["protocol"]["hard"], _selected_config(cfg, root, model, seed)
    rel = _cell_path("hard", m["label"], seed, "step_reversal")
    if resume and _artifact_valid(root / "raw" / f"{rel}.npz"):
        return
    amplitudes = np.asarray(p["step_amplitudes"], dtype=np.float32)
    replicas = int(p.get("step_replicas", 256)); phase_steps = int(p.get("phase_steps", 150))
    names = ("stand", "forward", "reverse", "lateral", "yaw", "stop")
    n, total = len(amplitudes) * replicas, len(names) * phase_steps
    session = build_session(m, seed, n, p["seed"])
    schedule = torch.zeros((total, n, 3), device=session.env.device)
    for i, amp in enumerate(amplitudes):
        sl = slice(i * replicas, (i + 1) * replicas)
        commands = ((0., 0., 0.), (float(amp), 0., 0.), (-float(amp), 0., 0.),
                    (0., float(amp), 0.), (0., 0., float(amp)), (0., 0., 0.))
        for phase, command in enumerate(commands):
            schedule[phase * phase_steps:(phase + 1) * phase_steps, sl, :] = torch.tensor(command, device=session.env.device)
    _set_nominal(session.env)
    bounds = [(i * phase_steps, (i + 1) * phase_steps) for i in range(len(names))]
    out = _rollout_schedule(session, schedule, warmup_command=np.asarray([0., 0., 0.]), warmup=p["warmup"], phase_bounds=bounds)
    payload = _meta(session, "hard", "step_reversal", p["seed"], p["warmup"], total)
    payload.update(campaign=cfg["campaign"], severity=np.repeat(amplitudes, replicas), command_amplitude=np.repeat(amplitudes, replicas),
        command_is_id=np.repeat(amplitudes, replicas) <= 1.0, phase_names=np.asarray(names), phase_bounds=np.asarray(bounds), phase_steps=phase_steps, **out)
    _save_cell(root, rel, payload)


def _run_hard(root: Path, cfg: Dict[str, Any], model: Dict[str, Any], seed: int, name: str, *, resume: bool) -> None:
    if name == "terrain":
        _run_hard_terrain(root, cfg, model, seed, resume=resume)
    elif name == "kick":
        _run_hard_kick(root, cfg, model, seed, resume=resume)
    elif name == "step":
        _run_hard_step(root, cfg, model, seed, resume=resume)
    else:
        raise ValueError(f"unknown hard scenario {name!r}")


def cmd_run(cfg: Dict[str, Any], suite: str, resume: bool, shard: Tuple[int, int]) -> int:
    from legged_gym import SIMULATOR, gs
    if SIMULATOR != "genesis": raise RuntimeError(f"SIMULATOR={SIMULATOR!r}; Eval V2 needs genesis")
    gs.init(backend=gs.gpu, logging_level="warning")
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    assigned = shard_items(_planned_cells(cfg, suite), shard)
    print(f"[run] suite={suite} shard={shard[0]}/{shard[1]} process_cells={len(assigned)} "
          f"resume={resume}")
    for cell_index, (kind, model, seed, name) in enumerate(assigned, start=1):
        print(f"[run] cell {cell_index}/{len(assigned)} {kind}:{model['label']}:seed{seed}:{name}", flush=True)
        try:
            if kind == "id": _run_id(root, cfg, model, seed, resume=resume)
            elif kind == "axes":
                axis, movement = name.split(":", 1)
                _run_axis(root, cfg, model, seed, axis, movement, resume=resume)
            else:
                _run_hard(root, cfg, model, seed, name, resume=resume)
        finally:
            # Genesis scenes are created per cell; force reclamation before the
            # next cell so a long single-GPU campaign stays serial and bounded.
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        print(f"[run] complete {cell_index}/{len(assigned)}", flush=True)
    _write_campaign_state(cfg)
    return 0


def _npz_scalar(z, key: str):
    v = z[key]
    return v.item() if getattr(v, "ndim", 0) == 0 else v.tolist()


def _npz_cell(z, key: str, index: int, default):
    """Read a per-env field or broadcast an artifact scalar to every row."""
    if key not in z:
        return default
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value[index]


def _metric_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), p25=float(np.percentile(values, 25)),
                p50=float(np.percentile(values, 50)), p75=float(np.percentile(values, 75)),
                num_replicas=int(values.size))


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _raw_groups(z, metric: str) -> Iterable[Tuple[Dict[str, Any], np.ndarray]]:
    """Yield one statistical scenario-cell per raw artifact.

    Axis and hard kick artifacts pack several values into their env dimension;
    each unique scenario descriptor is reduced across its replica environments
    here.  It is deliberately before cross-seed aggregation, so a seed with more
    replicas never receives a larger weight than another training seed.
    """
    scenario = str(_npz_scalar(z, "scenario"))
    n = len(z["tracking_lin"])
    if scenario == "step_reversal" and "phase_tracking_lin" in z:
        names = [str(x) for x in np.asarray(z["phase_names"]).tolist()]
        amplitudes = np.asarray(z["command_amplitude"], dtype=float)
        for phase_i, phase in enumerate(names):
            values = (np.asarray(z["phase_tracking_yaw"][phase_i], dtype=float)
                      if phase == "yaw" else np.asarray(z["phase_tracking_lin"][phase_i], dtype=float))
            if metric == "fall_rate": values = np.asarray(z["phase_fall_rate"][phase_i], dtype=float)
            elif metric == "return_per_step": values = np.asarray(z["phase_return_per_step"][phase_i], dtype=float)
            for amplitude in np.unique(amplitudes):
                mask = amplitudes == amplitude
                yield dict(axis="", axis_value=np.nan, axis_is_id="", command_mode=phase,
                           command_magnitude=float(amplitude), command_is_id=bool(amplitude <= 1.0),
                           severity=float(amplitude), phase=phase), values[mask]
        return

    values = np.asarray(z["tracking_lin" if metric == "tracking" else metric], dtype=float)
    descriptor = []
    for i in range(n):
        axis = str(_npz_cell(z, "axis", i, ""))
        axis_value = _npz_cell(z, "axis_value", i, np.nan)
        axis_is_id = _npz_cell(z, "axis_is_id", i, "")
        mode = str(_npz_cell(z, "command_mode", i, ""))
        magnitude = _npz_cell(z, "command_magnitude", i, np.nan)
        command_is_id = _npz_cell(z, "command_is_id", i, "")
        severity = _npz_cell(z, "severity", i, np.nan)
        descriptor.append((axis, _number_or_none(axis_value), str(axis_is_id), mode,
                           _number_or_none(magnitude), str(command_is_id),
                           _number_or_none(severity)))
    for key in sorted(set(descriptor)):
        mask = np.asarray([item == key for item in descriptor])
        axis, axis_value, axis_is_id, mode, magnitude, command_is_id, severity = key
        yield dict(axis=axis, axis_value=axis_value, axis_is_id=axis_is_id,
                   command_mode=mode, command_magnitude=magnitude,
                   command_is_id=command_is_id, severity=severity, phase=""), values[mask]


def cmd_aggregate(cfg: Dict[str, Any]) -> int:
    root = artifact_root(cfg); raw = sorted((root / "raw").rglob("*.npz")); rows=[]
    for path in raw:
        with np.load(path, allow_pickle=True) as z:
            meta={k:_npz_scalar(z,k) for k in ("campaign","suite","scenario","model","task","training_seed","checkpoint_iter","checkpoint_sha256","eval_seed") if k in z}
            for metric in ("tracking", "fall_rate", "return_per_step"):
                for cell, values in _raw_groups(z, metric):
                    rows.append(dict(**meta, **cell, metric=metric, **_metric_stats(values),
                                     raw_path=str(path.relative_to(root)), raw_sha256=sha256_file(str(path))))
    if not rows: raise FileNotFoundError(f"no raw artifacts under {root / 'raw'}")
    out=root/"tables"; out.mkdir(parents=True, exist_ok=True)
    with open(out/"results.csv", "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=sorted(rows[0])); w.writeheader(); w.writerows(rows)
    # Aggregate across training seeds by experimental condition. Checkpoint
    # iteration/hash and raw artifact provenance describe the selected seed run;
    # they must not split an otherwise identical condition into n=1 rows.
    provenance_fields = {
        "training_seed", "checkpoint_sha256",
        "mean", "p25", "p50", "p75", "num_replicas",
        "raw_path", "raw_sha256",
    }
    groups={}
    for r in rows:
        # Final suites aggregate selected checkpoints across training seeds, so
        # their per-seed checkpoint iteration is provenance. Validation contains
        # one artifact per candidate checkpoint: iteration remains a condition or
        # different learning stages would be miscounted as extra training seeds.
        excluded = provenance_fields | ({"checkpoint_iter"} if r.get("suite") != "validation" else set())
        k=tuple((x,r[x]) for x in r if x not in excluded)
        seed = int(r["training_seed"])
        by_seed = groups.setdefault(k, {})
        if seed in by_seed:
            raise RuntimeError(f"duplicate aggregate cell for training seed {seed}: {dict(k)}")
        by_seed[seed] = r["mean"]
    summary=[]
    for k,by_seed in groups.items():
        vals=list(by_seed.values())
        r=dict(k); r.update(value_median=float(np.median(vals)), value_min=float(np.min(vals)), value_max=float(np.max(vals)), n_training_seeds=len(by_seed)); summary.append(r)
    with open(out/"summary.csv", "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=sorted(summary[0])); w.writeheader(); w.writerows(summary)
    # Paired seed consistency, never a pseudo-statistical confidence interval.
    consistency = []
    comparisons = (("P5 better than MLP", "P5", "MLP"),
                   ("P5+V better than P5", "P5+V", "P5"))
    # Pair different model tasks on the shared condition and training seed.
    # Task/checkpoint fields are model-specific provenance, not comparison keys.
    match_keys = ("suite", "scenario", "checkpoint_iter", "eval_seed", "axis", "axis_value",
                  "axis_is_id", "command_mode", "command_magnitude", "command_is_id", "severity", "phase", "metric")
    groups_by_context: Dict[Tuple[Any, ...], Dict[Tuple[str, int], float]] = {}
    for row in rows:
        # As above, compare validation models only at the same candidate
        # iteration. Final-suite checkpoints were independently selected, so their
        # iteration is provenance and is normalized away for paired seed checks.
        key = tuple((row.get(k) if k != "checkpoint_iter" or row.get("suite") == "validation" else None)
                    for k in match_keys)
        cell = (str(row["model"]), int(row["training_seed"]))
        context_values = groups_by_context.setdefault(key, {})
        if cell in context_values:
            raise RuntimeError(f"duplicate consistency cell for model/seed {cell}: {key}")
        context_values[cell] = float(row["mean"])
    for context, values in groups_by_context.items():
        metric = context[-1]
        lower_is_better = metric in ("tracking", "fall_rate")
        for label, candidate, baseline in comparisons:
            seeds = sorted({seed for model, seed in values if model == candidate}
                           & {seed for model, seed in values if model == baseline})
            if not seeds:
                continue
            wins = sum((values[(candidate, seed)] < values[(baseline, seed)]) if lower_is_better
                       else (values[(candidate, seed)] > values[(baseline, seed)]) for seed in seeds)
            row = dict(zip(match_keys, context))
            row.update(comparison=label, better_k=wins, n_training_seeds=len(seeds))
            consistency.append(row)
    with open(out / "consistency.csv", "w", newline="") as f:
        fields = sorted(consistency[0]) if consistency else [*match_keys, "comparison", "better_k", "n_training_seeds"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(consistency)
    _write_campaign_state(cfg)
    return 0


def cmd_report(cfg: Dict[str, Any]) -> int:
    root=artifact_root(cfg); summary=root/"tables"/"summary.csv"
    if not summary.exists(): raise FileNotFoundError("aggregate must run before report")
    rows=list(csv.DictReader(summary.open()))
    consistency_path = root / "tables" / "consistency.csv"
    consistency = list(csv.DictReader(consistency_path.open())) if consistency_path.exists() else []
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    report=root/"report"; report.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(summary=rows, consistency=consistency, manifest=manifest), separators=(",", ":"))
    colors = {m["label"]: m.get("color", color) for m, color in zip(cfg["models"], ("#2563eb", "#dc2626", "#059669"))}
    html_doc = f'''<!doctype html><meta charset="utf-8"><title>{html.escape(cfg['campaign'])}</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#18212f}}select,button{{margin:3px;padding:5px}}#plot{{width:100%;height:360px;border:1px solid #dbe1ea}}table{{border-collapse:collapse;margin-top:12px}}td,th{{border:1px solid #dbe1ea;padding:5px;text-align:left}}.note{{color:#536273}}.badge{{background:#fff3cd;border-radius:4px;padding:2px 5px}}</style>
<h1>{html.escape(cfg['campaign'])}</h1><p class=note>Descriptive n=3 training-seed summary. Validation and final scenario banks are distinct. Raw completeness: {manifest.get('completed_raw_artifacts','?')}/{manifest.get('expected_raw_artifacts','?')}.</p>
<h2>Axis explorer</h2><label>Axis <select id=axis></select></label><label>Movement <select id=movement></select></label><label>Speed <select id=speed></select></label><label>Metric <select id=metric></select></label><span id=ood class=badge></span><svg id=plot viewBox="0 0 900 360"></svg><div id=raw></div>
<h2>Hard scenarios</h2><label>Scenario <select id=hard></select></label><div id=hardtable></div><h2>Paired consistency</h2><div id=consistency></div>
<script>const D={payload}, C={json.dumps(colors)}; const q=id=>document.getElementById(id), S=D.summary;
function options(id, vals){{let e=q(id),old=e.value;e.innerHTML=[...new Set(vals)].filter(x=>x!==''&&x!=='None').map(x=>`<option>${{x}}</option>`).join('');if([...e.options].some(o=>o.value===old))e.value=old;}}
function num(x){{let n=Number(x);return Number.isFinite(n)?n:null}};
function draw(){{let a=q('axis').value,m=q('movement').value,s=q('speed').value,metric=q('metric').value;let R=S.filter(r=>r.suite==='axes'&&r.axis===a&&r.command_mode===m&&String(r.command_magnitude)===s&&r.metric===metric);q('ood').textContent=Number(s)>1?'command-OOD':'';let xs=[...new Set(R.map(r=>num(r.axis_value)))].sort((x,y)=>x-y), ys=R.flatMap(r=>[num(r.value_min),num(r.value_max)]).filter(x=>x!==null),lo=Math.min(...ys),hi=Math.max(...ys),pad=(hi-lo||1)*.08;lo-=pad;hi+=pad;let X=x=>60+(x-xs[0])*780/(xs[xs.length-1]-xs[0]||1),Y=y=>320-(y-lo)*270/(hi-lo);let svg=q('plot');svg.innerHTML=`<line x1=60 y1=320 x2=850 y2=320 stroke=#789/><line x1=60 y1=30 x2=60 y2=320 stroke=#789/><text x=60 y=345>${{xs.map(x=>x.toFixed(2)).join('     ')}}</text>`;for(let model of [...new Set(R.map(r=>r.model))]){{let p=R.filter(r=>r.model===model).sort((u,v)=>num(u.axis_value)-num(v.axis_value));let band=p.map(r=>`${{X(num(r.axis_value))}},${{Y(num(r.value_min))}}`).join(' ')+' '+p.slice().reverse().map(r=>`${{X(num(r.axis_value))}},${{Y(num(r.value_max))}}`).join(' ');let line=p.map(r=>`${{X(num(r.axis_value))}},${{Y(num(r.value_median))}}`).join(' ');svg.innerHTML+=`<polygon points="${{band}}" fill="${{C[model]||'#555'}}" opacity=.15/><polyline points="${{line}}" fill=none stroke="${{C[model]||'#555'}}" stroke-width=3/><text x=680 y=${{45+20*[...new Set(R.map(r=>r.model))].indexOf(model)}} fill="${{C[model]||'#555'}}">${{model}}</text>`;}}q('raw').innerHTML='<table><tr><th>model</th><th>median</th><th>min-max</th><th>n</th></tr>'+R.map(r=>`<tr><td>${{r.model}}</td><td>${{r.value_median}}</td><td>${{r.value_min}}–${{r.value_max}}</td><td>${{r.n_training_seeds}}</td></tr>`).join('')+'</table>';}}
function hard(){{let sc=q('hard').value,R=S.filter(r=>r.suite==='hard'&&r.scenario===sc);q('hardtable').innerHTML='<table><tr><th>model</th><th>severity</th><th>phase</th><th>metric</th><th>median</th><th>min-max</th></tr>'+R.map(r=>`<tr><td>${{r.model}}</td><td>${{r.severity}}</td><td>${{r.phase}}</td><td>${{r.metric}}</td><td>${{r.value_median}}</td><td>${{r.value_min}}–${{r.value_max}}</td></tr>`).join('')+'</table>';}}
options('axis',S.filter(r=>r.suite==='axes').map(r=>r.axis));options('movement',S.filter(r=>r.suite==='axes').map(r=>r.command_mode));options('speed',S.filter(r=>r.suite==='axes').map(r=>r.command_magnitude));options('metric',['tracking','fall_rate','return_per_step']);options('hard',S.filter(r=>r.suite==='hard').map(r=>r.scenario));['axis','movement','speed','metric'].forEach(id=>q(id).onchange=draw);q('hard').onchange=hard;draw();hard();q('consistency').innerHTML='<table><tr><th>comparison</th><th>metric</th><th>k/n</th></tr>'+D.consistency.slice(0,100).map(r=>`<tr><td>${{r.comparison}}</td><td>${{r.metric}}</td><td>${{r.better_k}}/${{r.n_training_seeds}}</td></tr>`).join('')+'</table>';</script>'''
    (report/"index.html").write_text(html_doc, encoding="utf-8")
    return 0


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("command", choices=("plan","select-checkpoints","run","aggregate","report")); p.add_argument("--config", required=True); p.add_argument("--suite", choices=("id","axes","hard","all"), default="all"); p.add_argument("--shard", default="0/1"); p.add_argument("--resume", action="store_true")
    a=p.parse_args(); cfg=load_config(a.config)
    shard = parse_shard(a.shard)
    exit_code = 1
    try:
        if a.command=="plan": result = cmd_plan(cfg,a.suite)
        elif a.command=="select-checkpoints": result = cmd_select(cfg, shard)
        elif a.command=="run": result = cmd_run(cfg,a.suite,a.resume, shard)
        elif a.command=="aggregate": result = cmd_aggregate(cfg)
        else: result = cmd_report(cfg)
        exit_code = int(result)
        return exit_code
    finally:
        _record_command(cfg, a.command, a.shard, exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
