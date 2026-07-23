"""V3 generalisation scorecard runner.

This is intentionally separate from :mod:`campaign`: Eval V2 describes the
historical axis/hard-suite artifact contract, while V3 evaluates one frozen
training family with static payload cells, deterministic mid-episode switches,
and an explicit oracle-gated scorecard.  It reuses the V2 strict deployment
loader and atomic artifact writer, but never changes their existing output
shape or interpretation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml

from legged_gym.scripts.eval.campaign import (
    _artifact_valid,
    _meta,
    _root,
    _save_cell,
    _set_nominal,
    _terrain_hash,
    _terrain_raw,
    build_session,
    build_terrain_session,
    parse_shard,
    shard_items,
)
from legged_gym.scripts.eval.ckpt_utils import resolve_checkpoint_path, sha256_file
from legged_gym.scripts.eval.dr_axes import get_axis
from legged_gym.scripts.eval.metrics import MetricAccumulator
from legged_gym.scripts.eval.provenance import verify_run_identity


RAW_METRICS = ("tracking_lin", "tracking_yaw", "fall_rate", "return_per_step")
SCORED_SUITES = ("s1", "s2", "t1", "t2")
TERRAIN_SUITES = ("t0", "t1", "t2")

# Canonical column->type map for the pinned eval grid.  With num_cols=5 and the
# benchmark proportions [0.2, 0.1, 0.25, 0.25, 0.2] the deterministic
# ``curiculum()`` generator assigns ``choice = col / num_cols`` to, in order:
# smooth slope, random uniform, pyramid stairs (down), pyramid stairs (up),
# discrete obstacles.  Falls back to a stable ``type{n}`` label off-grid.
_TERRAIN_TYPE_NAMES = {0: "slope", 1: "random_uniform", 2: "stairs_down", 3: "stairs_up", 4: "discrete"}


def _terrain_type_name(index: int, num_cols: int = 5) -> str:
    if num_cols == 5 and int(index) in _TERRAIN_TYPE_NAMES:
        return _TERRAIN_TYPE_NAMES[int(index)]
    return f"type{int(index)}"


@dataclass(frozen=True)
class Cell:
    suite: str
    model: str
    seed: int
    name: str
    scenario: Mapping[str, Any]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("V3 eval config must be a mapping")
    required = ("campaign", "artifact_root", "simulator", "training_seeds", "models", "protocol", "scorecard")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"V3 eval config missing required keys: {missing}")
    if not isinstance(cfg["campaign"], str) or not cfg["campaign"].strip():
        raise ValueError("campaign must be a non-empty name, e.g. 'v3_ilk_deneme'")
    if cfg["simulator"] != "genesis":
        raise ValueError("V3 eval supports simulator: genesis only")
    labels = [model.get("label") for model in cfg["models"]]
    if len(labels) != len(set(labels)) or any(not label for label in labels):
        raise ValueError("models must have unique non-empty labels")
    score = cfg["scorecard"]
    for key in ("baseline_label", "oracle_label", "method_labels", "relative_headroom", "absolute_headroom", "fall_gate_pp", "achieved_speed_ratio"):
        if key not in score:
            raise ValueError(f"scorecard missing {key!r}")
    known = set(labels)
    if score["baseline_label"] not in known or score["oracle_label"] not in known:
        raise ValueError("scorecard baseline/oracle labels must be declared models")
    if not set(score["method_labels"]).issubset(known - {score["baseline_label"], score["oracle_label"]}):
        raise ValueError("scorecard method_labels must exclude baseline and oracle")
    return cfg


def artifact_root(cfg: Mapping[str, Any]) -> Path:
    path = Path(str(cfg["artifact_root"]))
    return path if path.is_absolute() else _root() / path


def _cell_rel(cell: Cell) -> str:
    return f"{cell.suite}/{cell.model}/seed_{cell.seed}/{cell.name}"


def _seed_list(cfg: Mapping[str, Any], model: Mapping[str, Any]) -> List[int]:
    return [int(seed) for seed in model.get("training_seeds", cfg["training_seeds"])]


def _filter_models(cfg: Mapping[str, Any], labels: Sequence[str] | None) -> List[Mapping[str, Any]]:
    """Return a caller-selected subset without silently accepting a typo."""
    if not labels:
        return list(cfg["models"])
    wanted = {str(label) for label in labels}
    known = {str(model["label"]) for model in cfg["models"]}
    unknown = wanted - known
    if unknown:
        raise ValueError(f"unknown --models label(s): {sorted(unknown)}; known={sorted(known)}")
    return [model for model in cfg["models"] if model["label"] in wanted]


def payload_com_x(mass_kg: float, payload_cfg: Mapping[str, Any]) -> float:
    """Pre-registered deterministic mass -> forward-CoM payload mapping."""
    raw = float(mass_kg) * float(payload_cfg["com_x_per_kg"])
    limit = abs(float(payload_cfg["com_x_limit_m"]))
    return float(np.clip(raw, -limit, limit))


def _command_name(command: Mapping[str, Any]) -> str:
    return str(command["name"]).replace("/", "_").replace(" ", "_")


def _planned_cells(cfg: Mapping[str, Any], suite: str, *, include_diagnostics: bool = False) -> List[Cell]:
    protocol = cfg["protocol"]
    terrain_cfg = protocol.get("terrain")
    if suite == "all":
        # ``all`` covers whichever families this campaign actually configures:
        # the historical flat s-suites and/or the terrain-native t-suites.  A
        # terrain-only config never plans empty s-cells and vice versa.
        requested = []
        if "payload" in protocol and "s1_commands" in protocol:
            requested += ["s0", "s1", "s2", "s3"]
        if terrain_cfg:
            requested += ["t0", "t1", "t2"]
    else:
        requested = (suite,)
    model_by_label = {model["label"]: model for model in cfg["models"]}
    out: List[Cell] = []
    payload_cfg = protocol.get("payload")
    if "s0" in requested:
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                out.append(Cell("s0", model["label"], seed, "static_id", {"kind": "static_id"}))
                out.append(Cell("s0", model["label"], seed, "dynamic_id_in_band", {
                    "kind": "dynamic_id", "target": cfg["protocol"]["s0_dynamic_target"],
                }))
    if "s1" in requested:
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                for payload in payload_cfg["static"]:
                    for command in cfg["protocol"]["s1_commands"]:
                        if not bool(command.get("headline", True)) and not include_diagnostics:
                            continue
                        name = f"payload_{payload['name']}__{_command_name(command)}"
                        out.append(Cell("s1", model["label"], seed, name, {
                            "kind": "static_payload", "payload": payload, "command": command,
                        }))
    if "s2" in requested:
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                for switch in payload_cfg["switches"]:
                    for command in cfg["protocol"]["s2_commands"]:
                        if not bool(command.get("headline", True)) and not include_diagnostics:
                            continue
                        name = f"switch_{switch['name']}__{_command_name(command)}"
                        out.append(Cell("s2", model["label"], seed, name, {
                            "kind": "payload_switch", "switch": switch, "command": command,
                        }))
    if "s3" in requested:
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                for kick in cfg["protocol"]["s3"]["kick_dvy"]:
                    out.append(Cell("s3", model["label"], seed, f"kick_{str(kick).replace('.', '_')}", {
                        "kind": "kick", "kick_dvy": float(kick),
                    }))
                for terrain_cm in cfg["protocol"]["s3"]["terrain_cm"]:
                    out.append(Cell("s3", model["label"], seed, f"terrain_{str(terrain_cm).replace('.', '_')}cm", {
                        "kind": "terrain", "terrain_cm": float(terrain_cm),
                    }))
    if terrain_cfg and ("t0" in requested or "t1" in requested or "t2" in requested):
        for model in cfg["models"]:
            for seed in _seed_list(cfg, model):
                if "t0" in requested:
                    id_level = int(terrain_cfg["id_level"])
                    out.append(Cell("t0", model["label"], seed, f"terrain_id_level_{id_level}", {
                        "kind": "terrain_id", "level": id_level,
                    }))
                if "t1" in requested:
                    for level in terrain_cfg["severity_levels"]:
                        out.append(Cell("t1", model["label"], seed, f"terrain_severity_level_{int(level)}", {
                            "kind": "terrain_severity", "level": int(level),
                        }))
                if "t2" in requested:
                    payload_level = int(terrain_cfg["payload_level"])
                    for payload in terrain_cfg["payloads"]:
                        out.append(Cell("t2", model["label"], seed, f"terrain_payload_{payload['name']}", {
                            "kind": "terrain_payload", "level": payload_level, "payload": payload,
                        }))
    # Detect typoed method labels in a config while producing a useful error.
    if {cell.model for cell in out} - set(model_by_label):
        raise RuntimeError("planned an undeclared model")
    return out


def _selection_path(root: Path) -> Path:
    return root / "run_selection.json"


def _discover_run(model: Mapping[str, Any], seed: int, *, require_final: bool) -> Path:
    root = Path(str(model["log_root"]))
    root = root if root.is_absolute() else _root() / root
    pattern = str(model["run_pattern"]).format(seed=seed)
    candidates = []
    for run_dir in root.glob(pattern):
        if not run_dir.is_dir():
            continue
        if require_final and not (run_dir / "model_3000.pt").is_file():
            continue
        try:
            checkpoint = Path(resolve_checkpoint_path(str(run_dir), model.get("checkpoint", "best_tracking")))
            verify_run_identity(str(run_dir), expected_task=str(model["task"]),
                                expected_training_seed=seed, require_manifest=True)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append((checkpoint.stat().st_mtime, run_dir, checkpoint))
    if not candidates:
        suffix = " with model_3000.pt and best_tracking.pt" if require_final else ""
        raise FileNotFoundError(
            f"{model['label']} seed {seed}: no verified V3 run matching {root / pattern}{suffix}"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def cmd_resolve_runs(cfg: Mapping[str, Any], *, labels: Sequence[str] | None = None, strict: bool = False) -> int:
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    require_final = bool(cfg.get("require_final_checkpoint", True))
    path = _selection_path(root)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"selections": []}
    selection_by_key = {(str(row["label"]), int(row["training_seed"])): row
                        for row in previous.get("selections", [])}
    discovered, skipped = [], []
    for model in _filter_models(cfg, labels):
        for seed in _seed_list(cfg, model):
            try:
                run_dir = _discover_run(model, seed, require_final=require_final)
            except FileNotFoundError as exc:
                if strict:
                    raise
                skipped.append({"label": model["label"], "training_seed": seed, "reason": str(exc)})
                continue
            checkpoint = Path(resolve_checkpoint_path(str(run_dir), model.get("checkpoint", "best_tracking")))
            row = {
                "label": model["label"], "task": model["task"], "training_seed": seed,
                "run_folder": run_dir.name, "run_path": str(run_dir),
                "checkpoint": model.get("checkpoint", "best_tracking"),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(str(checkpoint)),
            }
            key = (str(model["label"]), int(seed))
            old = selection_by_key.get(key)
            if old and old.get("checkpoint_sha256") != row["checkpoint_sha256"]:
                raise RuntimeError(
                    f"{model['label']} seed {seed}: existing campaign selection points to a different checkpoint. "
                    "Start a new campaign instead of silently mixing policies."
                )
            selection_by_key[key] = row
            discovered.append(row)
    selections = [selection_by_key[key] for key in sorted(selection_by_key)]
    payload = {
        "campaign": cfg["campaign"], "generated_at": datetime.now(timezone.utc).isoformat(),
        "require_final_checkpoint": require_final, "selections": selections, "skipped": skipped,
    }
    path.write_text(_json(payload), encoding="utf-8")
    print(f"[v3_eval] added/verified {len(discovered)} run(s); campaign now has {len(selections)} selected policy-seed(s) -> {path}")
    for row in skipped:
        print(f"[v3_eval] pending {row['label']} seed{row['training_seed']}: no final checkpoint yet")
    return 0


def _selected_models(cfg: Mapping[str, Any], root: Path, labels: Sequence[str] | None = None) -> Dict[str, Dict[str, Any]]:
    path = _selection_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run 'resolve-runs' after V3 training completes")
    data = json.loads(path.read_text(encoding="utf-8"))
    selected: Dict[str, Dict[str, Any]] = {}
    for model in _filter_models(cfg, labels):
        entries = [entry for entry in data.get("selections", []) if entry.get("label") == model["label"]]
        by_seed = {int(entry["training_seed"]): entry for entry in entries}
        if not by_seed:
            continue
        expected = [seed for seed in _seed_list(cfg, model) if seed in by_seed]
        copied = dict(model)
        copied["run_folders"] = {seed: by_seed[seed]["run_folder"] for seed in expected}
        copied["training_seeds"] = expected
        selected[model["label"]] = copied
    if labels and not selected:
        raise FileNotFoundError("none of the requested models are selected yet; run resolve-runs once their checkpoints exist")
    return selected


def _id_commands(num_envs: int, seed: int, bank: Mapping[str, Any]) -> np.ndarray:
    """Fixed V3 direct-velocity bank, with support declared in campaign YAML."""
    rng = np.random.default_rng(seed)
    x_values, y_values, yaw_values = (tuple(float(x) for x in bank[key]) for key in ("anchor_x", "anchor_y", "anchor_yaw_rate"))
    anchors = np.asarray([[x, y, z] for x in x_values for y in y_values for z in yaw_values], dtype=np.float32)
    commands = np.empty((num_envs, 3), dtype=np.float32)
    commands[:, 0] = rng.uniform(*map(float, bank["lin_vel_x"]), size=num_envs)
    commands[:, 1] = rng.uniform(*map(float, bank["lin_vel_y"]), size=num_envs)
    commands[:, 2] = rng.uniform(*map(float, bank["ang_vel_yaw"]), size=num_envs)
    commands[:min(len(anchors), num_envs)] = anchors[:min(len(anchors), num_envs)]
    return commands


def _command_array(command: Mapping[str, Any], num_envs: int) -> np.ndarray:
    vector = np.asarray(command["vector"], dtype=np.float32)
    if vector.shape != (3,):
        raise ValueError(f"command {command.get('name')!r} vector must have length 3")
    return np.repeat(vector[None, :], num_envs, axis=0)


def _apply_payload(env, mass_kg: float, payload_cfg: Mapping[str, Any], *, pin_nominal: bool) -> Dict[str, np.ndarray]:
    if pin_nominal:
        _set_nominal(env)
    mass = torch.full((env.num_envs,), float(mass_kg), device=env.device)
    com_x = torch.full((env.num_envs,), payload_com_x(float(mass_kg), payload_cfg), device=env.device)
    mass_axis, com_axis = get_axis("added_mass"), get_axis("com_x")
    mass_axis.apply(env, mass)
    com_axis.apply(env, com_x)
    return {
        "payload_mass_requested": mass.detach().cpu().numpy(),
        "payload_mass_actual": mass_axis.validate(env, mass).detach().cpu().numpy(),
        "payload_com_x_requested": com_x.detach().cpu().numpy(),
        "payload_com_x_actual": com_axis.validate(env, com_x).detach().cpu().numpy(),
    }


def _warmup(session, commands: np.ndarray, warmup: int):
    env, adapter = session.env, session.adapter
    command_t = torch.as_tensor(commands, dtype=torch.float, device=env.device)
    state = adapter.reset(env)
    for _ in range(int(warmup)):
        env.commands[:, :3] = command_t
        state, _, _ = adapter.step(env, adapter.act(state).detach())
    env.episode_length_buf.zero_()
    return state, command_t


def _kinematics(env, command_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lin = torch.linalg.vector_norm(command_t[:, :2] - env.simulator.base_lin_vel[:, :2], dim=1)
    yaw = (command_t[:, 2] - env.simulator.base_ang_vel[:, 2]).abs()
    speed_target = torch.linalg.vector_norm(command_t[:, :2], dim=1)
    unit = torch.where(speed_target[:, None] > 1e-6, command_t[:, :2] / speed_target.clamp_min(1e-6)[:, None], torch.zeros_like(command_t[:, :2]))
    achieved = (env.simulator.base_lin_vel[:, :2] * unit).sum(dim=1)
    return lin, yaw, achieved


@torch.no_grad()
def _rollout_fixed(session, commands: np.ndarray, *, warmup: int, steps: int) -> Dict[str, np.ndarray]:
    env, adapter = session.env, session.adapter
    state, command_t = _warmup(session, commands, warmup)
    acc = MetricAccumulator(env.num_envs, env.device)
    achieved_sum = torch.zeros(env.num_envs, device=env.device)
    for _ in range(int(steps)):
        env.commands[:, :3] = command_t
        state, rewards, dones = adapter.step(env, adapter.act(state).detach())
        lin, yaw, achieved = _kinematics(env, command_t)
        acc.update(rewards, dones, env.time_out_buf, lin, yaw)
        achieved_sum += achieved
    raw = acc.compute()
    target = torch.linalg.vector_norm(command_t[:, :2], dim=1)
    achieved = achieved_sum / float(steps)
    return {
        "tracking_lin": raw["tracking_lin_err"].cpu().numpy(),
        "tracking_yaw": raw["tracking_ang_err"].cpu().numpy(),
        "fall_rate": raw["ever_fell"].cpu().numpy(),
        "return_per_step": raw["return_per_step"].cpu().numpy(),
        "achieved_speed": achieved.cpu().numpy(),
        "achieved_speed_ratio": torch.where(target > 1e-6, achieved / target, torch.ones_like(target)).cpu().numpy(),
    }


def _sysid_trace(session, state) -> Tuple[torch.Tensor, torch.Tensor] | None:
    actor_critic = getattr(getattr(session.runner, "alg", None), "actor_critic", None)
    estimator = getattr(actor_critic, "estimator", None)
    truth = getattr(session.env, "estimator_labels_buf", None)
    if estimator is None or truth is None or not isinstance(state, torch.Tensor):
        return None
    return estimator(state.detach()), truth.detach()


def _sysid_trace_summary(pred: torch.Tensor, truth: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compact estimator diagnostics without cancellation across environments."""
    error = pred - truth
    return {
        "pred_mean": pred.mean(dim=0),
        "truth_mean": truth.mean(dim=0),
        "error_mean": error.mean(dim=0),
        "mae": error.abs().mean(dim=0),
        "rmse": error.square().mean(dim=0).sqrt(),
    }


@torch.no_grad()
def _rollout_switch(
    session, commands: np.ndarray, *, warmup: int, pre_steps: int, post_steps: int,
    target_mass: float, payload_cfg: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Run an externally scheduled, identical V3 payload switch for every env."""
    env, adapter = session.env, session.adapter
    state, command_t = _warmup(session, commands, warmup)
    total = int(pre_steps) + int(post_steps)
    lin_errors: List[torch.Tensor] = []
    yaw_errors: List[torch.Tensor] = []
    achieved: List[torch.Tensor] = []
    rewards_sum = torch.zeros(env.num_envs, device=env.device)
    ever_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    post_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    trace_summaries: Dict[str, List[torch.Tensor]] = defaultdict(list)
    switch_payload: Dict[str, np.ndarray] | None = None
    for step in range(total):
        if step == int(pre_steps):
            switch_payload = _apply_payload(env, target_mass, payload_cfg, pin_nominal=False)
        env.commands[:, :3] = command_t
        trace = _sysid_trace(session, state)
        state, rewards, dones = adapter.step(env, adapter.act(state).detach())
        lin, yaw, current_speed = _kinematics(env, command_t)
        fell = dones.bool() & ~env.time_out_buf.bool()
        ever_fell |= fell
        if step >= int(pre_steps):
            post_fell |= fell
        rewards_sum += rewards
        lin_errors.append(lin)
        yaw_errors.append(yaw)
        achieved.append(current_speed)
        if trace is not None:
            pred, truth = trace
            for key, value in _sysid_trace_summary(pred, truth).items():
                trace_summaries[key].append(value.detach().cpu())
    lin_error_t = torch.stack(lin_errors)
    yaw_error_t = torch.stack(yaw_errors)
    speed_t = torch.stack(achieved)
    pre_lin = lin_error_t[:pre_steps]
    post_lin = lin_error_t[pre_steps:]
    post_yaw = yaw_error_t[pre_steps:]
    target_speed = torch.linalg.vector_norm(command_t[:, :2], dim=1)
    result: Dict[str, np.ndarray] = {
        "tracking_lin": lin_error_t.mean(dim=0).cpu().numpy(),
        "tracking_yaw": yaw_error_t.mean(dim=0).cpu().numpy(),
        "fall_rate": ever_fell.float().cpu().numpy(),
        "return_per_step": (rewards_sum / float(total)).cpu().numpy(),
        "pre_switch_tracking_error": pre_lin.mean(dim=0).cpu().numpy(),
        "post_switch_tracking_error": post_lin.mean(dim=0).cpu().numpy(),
        "post_switch_tracking_yaw": post_yaw.mean(dim=0).cpu().numpy(),
        "post_switch_fall_rate": post_fell.float().cpu().numpy(),
        "achieved_speed": speed_t[pre_steps:].mean(dim=0).cpu().numpy(),
        "achieved_speed_ratio": torch.where(target_speed > 1e-6, speed_t[pre_steps:].mean(dim=0) / target_speed, torch.ones_like(target_speed)).cpu().numpy(),
    }
    if switch_payload is None:
        raise RuntimeError("switch rollout ended before its scheduled switch")
    result.update(switch_payload)
    if trace_summaries:
        result["p_hat_trace"] = torch.stack(trace_summaries["pred_mean"]).numpy()
        result["p_true_trace"] = torch.stack(trace_summaries["truth_mean"]).numpy()
        result["p_hat_error_mean_trace"] = torch.stack(trace_summaries["error_mean"]).numpy()
        result["p_hat_mae_trace"] = torch.stack(trace_summaries["mae"]).numpy()
        result["p_hat_rmse_trace"] = torch.stack(trace_summaries["rmse"]).numpy()
    return result


@torch.no_grad()
def _rollout_kick(session, commands: np.ndarray, *, warmup: int, pre_steps: int, post_steps: int, dvy: float) -> Dict[str, np.ndarray]:
    env, adapter = session.env, session.adapter
    state, command_t = _warmup(session, commands, warmup)
    errors: List[torch.Tensor] = []
    ever_fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    rewards_sum = torch.zeros(env.num_envs, device=env.device)
    for step in range(int(pre_steps) + int(post_steps)):
        env.commands[:, :3] = command_t
        if step == int(pre_steps):
            qvel = env.simulator._robot.get_dofs_velocity()
            qvel[:, 1] += float(dvy)
            env.simulator._robot.set_dofs_velocity(qvel)
        state, rewards, dones = adapter.step(env, adapter.act(state).detach())
        lin, _, _ = _kinematics(env, command_t)
        errors.append(lin)
        rewards_sum += rewards
        ever_fell |= dones.bool() & ~env.time_out_buf.bool()
    error_t = torch.stack(errors)
    post = error_t[int(pre_steps):]
    return {
        "tracking_lin": error_t.mean(dim=0).cpu().numpy(),
        "tracking_yaw": np.zeros(env.num_envs, dtype=np.float32),
        "fall_rate": ever_fell.float().cpu().numpy(),
        "return_per_step": (rewards_sum / float(len(errors))).cpu().numpy(),
        "post_kick_tracking_error_integral": (post.sum(dim=0) * float(env.dt)).cpu().numpy(),
        "kick_dvy": np.full(env.num_envs, float(dvy), dtype=np.float32),
    }


def _payload_metadata(payload: Mapping[str, Any], payload_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    mass = float(payload["mass_kg"])
    return {
        "payload_name": str(payload["name"]), "payload_tier": str(payload["tier"]),
        "payload_mass_kg": mass, "payload_com_x_m": payload_com_x(mass, payload_cfg),
    }


def _base_payload(session, suite: str, scenario: str, eval_seed: int, warmup: int, steps: int, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    out = _meta(session, suite, scenario, eval_seed, warmup, steps)
    out["campaign"] = cfg["campaign"]
    return out


def _run_cell(root: Path, cfg: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]], cell: Cell, *, resume: bool) -> None:
    rel = _cell_rel(cell)
    final = root / "raw" / f"{rel}.npz"
    if resume and _artifact_valid(final):
        return
    model = models[cell.model]
    protocol = cfg["protocol"]
    num_envs, eval_seed = int(protocol["num_envs"]), int(protocol["eval_seed"])
    scenario = dict(cell.scenario)
    kind = scenario["kind"]
    if kind == "static_id":
        p = protocol["s0"]
        session = build_session(model, cell.seed, num_envs, eval_seed, id_domain_rand=True, disable_v3_physics_switch=True)
        commands = _id_commands(num_envs, eval_seed, p["command_bank"])
        payload = _base_payload(session, "s0", "static_id", eval_seed, int(p["warmup"]), int(p["steps"]), cfg)
        payload.update(s0_variant="static", heading_command=False, yaw_semantics="direct_yaw_rate",
                       command_bank_json=json.dumps(p["command_bank"], sort_keys=True), command_matrix=commands,
                       scenario_hash=hashlib.sha256(commands.tobytes()).hexdigest(),
                       **_rollout_fixed(session, commands, warmup=int(p["warmup"]), steps=int(p["steps"])))
    elif kind == "dynamic_id":
        p, target = protocol["s0"], scenario["target"]
        session = build_session(model, cell.seed, num_envs, eval_seed, disable_v3_physics_switch=True)
        initial = _apply_payload(session.env, 0.0, protocol["payload"], pin_nominal=True)
        commands = _id_commands(num_envs, eval_seed, p["command_bank"])
        payload = _base_payload(session, "s0", "dynamic_id_in_band", eval_seed, int(p["warmup"]), int(p["pre_steps"]) + int(p["post_steps"]), cfg)
        payload.update(s0_variant="dynamic", heading_command=False, yaw_semantics="direct_yaw_rate",
                       command_bank_json=json.dumps(p["command_bank"], sort_keys=True), initial_payload_mass_kg=0.0, initial_payload_com_x_m=0.0,
                       **{f"initial_{key}": value for key, value in initial.items()},
                       **_payload_metadata(target, protocol["payload"]), command_matrix=commands,
                       **_rollout_switch(session, commands, warmup=int(p["warmup"]), pre_steps=int(p["pre_steps"]), post_steps=int(p["post_steps"]),
                                         target_mass=float(target["mass_kg"]), payload_cfg=protocol["payload"]))
    elif kind == "static_payload":
        p, target, command = protocol["s1"], scenario["payload"], scenario["command"]
        session = build_session(model, cell.seed, num_envs, eval_seed, disable_v3_physics_switch=True)
        physics = _apply_payload(session.env, float(target["mass_kg"]), protocol["payload"], pin_nominal=True)
        commands = _command_array(command, num_envs)
        payload = _base_payload(session, "s1", cell.name, eval_seed, int(p["warmup"]), int(p["steps"]), cfg)
        payload.update(**_payload_metadata(target, protocol["payload"]), command_name=_command_name(command),
                       command_mode=str(command["mode"]), command_vector=np.asarray(command["vector"], dtype=np.float32),
                       command_speed=float(np.linalg.norm(np.asarray(command["vector"], dtype=float)[:2])),
                       headline_command=bool(command.get("headline", True)),
                       diagnostic_kind=str(command.get("diagnostic_kind", "")), **physics,
                       **_rollout_fixed(session, commands, warmup=int(p["warmup"]), steps=int(p["steps"])))
    elif kind == "payload_switch":
        p, target, command = protocol["s2"], scenario["switch"], scenario["command"]
        session = build_session(model, cell.seed, num_envs, eval_seed, disable_v3_physics_switch=True)
        initial = _apply_payload(session.env, 0.0, protocol["payload"], pin_nominal=True)
        commands = _command_array(command, num_envs)
        payload = _base_payload(session, "s2", cell.name, eval_seed, int(p["warmup"]), int(p["pre_steps"]) + int(p["post_steps"]), cfg)
        payload.update(switch_name=str(target["name"]), switch_tier=str(target["tier"]), command_name=_command_name(command),
                       command_mode=str(command["mode"]), command_vector=np.asarray(command["vector"], dtype=np.float32),
                       command_speed=float(np.linalg.norm(np.asarray(command["vector"], dtype=float)[:2])),
                       headline_command=bool(command.get("headline", True)),
                       diagnostic_kind=str(command.get("diagnostic_kind", "")),
                       **_payload_metadata(target, protocol["payload"]),
                       **{f"initial_{key}": value for key, value in initial.items()},
                       **_rollout_switch(session, commands, warmup=int(p["warmup"]), pre_steps=int(p["pre_steps"]), post_steps=int(p["post_steps"]),
                                         target_mass=float(target["mass_kg"]), payload_cfg=protocol["payload"]))
    elif kind == "kick":
        p = protocol["s3"]
        session = build_session(model, cell.seed, num_envs, eval_seed, disable_v3_physics_switch=True)
        _set_nominal(session.env)
        commands = np.tile(np.asarray([[1., 0., 0.]], dtype=np.float32), (num_envs, 1))
        payload = _base_payload(session, "s3", "kick_calibration", eval_seed, int(p["warmup"]), int(p["kick_pre_steps"]) + int(p["kick_post_steps"]), cfg)
        payload.update(s3_kind="kick", command_mode="forward", command_speed=1.0,
                       **_rollout_kick(session, commands, warmup=int(p["warmup"]), pre_steps=int(p["kick_pre_steps"]),
                                       post_steps=int(p["kick_post_steps"]), dvy=float(scenario["kick_dvy"])))
    elif kind == "terrain":
        p, cm = protocol["s3"], float(scenario["terrain_cm"])
        hscale = float(p["terrain_horizontal_scale"])
        raw = _terrain_raw(cm / 100.0, int(eval_seed), hscale)
        session = build_session(model, cell.seed, num_envs, eval_seed, terrain_raw=raw, terrain_horizontal_scale=hscale,
                                disable_v3_physics_switch=True)
        _set_nominal(session.env)
        commands = np.tile(np.asarray([[1., 0., 0.]], dtype=np.float32), (num_envs, 1))
        payload = _base_payload(session, "s3", "terrain_raw", eval_seed, int(p["warmup"]), int(p["terrain_steps"]), cfg)
        payload.update(s3_kind="terrain", terrain_cm=cm, terrain_hash=_terrain_hash(raw), terrain_max_abs_m=cm / 100.0,
                       command_mode="forward", command_speed=1.0,
                       **_rollout_fixed(session, commands, warmup=int(p["warmup"]), steps=int(p["terrain_steps"])))
    elif kind in ("terrain_id", "terrain_severity", "terrain_payload"):
        t = protocol["terrain"]
        level = int(scenario["level"])
        vx = float(t["command"]["vx"])
        session, terrain_info = build_terrain_session(
            model, cell.seed, num_envs, eval_seed,
            fixed_level=level, num_cols=int(t["num_cols"]),
            terrain_proportions=t["terrain_proportions"], num_rows=t.get("num_rows"),
            measure_heights=bool(t.get("measure_heights", True)),
            disable_v3_physics_switch=True,
        )
        _set_nominal(session.env)
        extra: Dict[str, Any] = {}
        if kind == "terrain_payload":
            payload_item = scenario["payload"]
            physics = _apply_payload(session.env, float(payload_item["mass_kg"]), protocol["payload"], pin_nominal=False)
            extra.update(_payload_metadata(payload_item, protocol["payload"]), **physics)
        commands = np.tile(np.asarray([[vx, 0., 0.]], dtype=np.float32), (num_envs, 1))
        scenario_name = {"terrain_id": "terrain_id", "terrain_severity": "terrain_severity",
                         "terrain_payload": "terrain_payload"}[kind]
        payload = _base_payload(session, cell.suite, scenario_name, eval_seed, int(t["warmup"]), int(t["steps"]), cfg)
        payload.update(terrain_kind=kind, terrain_level=level, terrain_hash=terrain_info["terrain_hash"],
                       terrain_num_cols=terrain_info["num_cols"], terrain_num_rows=terrain_info["num_rows"],
                       terrain_type=terrain_info["terrain_type"], terrain_level_per_env=terrain_info["terrain_level"],
                       command_mode="forward", command_speed=vx, headline_command=True, **extra,
                       **_rollout_fixed(session, commands, warmup=int(t["warmup"]), steps=int(t["steps"])))
    else:
        raise ValueError(f"unknown V3 eval cell kind {kind!r}")
    _save_cell(root, rel, payload)


def _write_manifest(cfg: Mapping[str, Any], cells: Sequence[Cell]) -> None:
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in cells:
        rel = _cell_rel(cell) + ".npz"
        path = root / "raw" / rel
        rows.append({"path": f"raw/{rel}", "complete": _artifact_valid(path),
                     "sha256": sha256_file(str(path)) if _artifact_valid(path) else None})
    payload = {"campaign": cfg["campaign"], "generated_at": datetime.now(timezone.utc).isoformat(),
               "expected_raw_artifacts": len(rows), "completed_raw_artifacts": sum(row["complete"] for row in rows),
               "complete": all(row["complete"] for row in rows), "cells": rows,
               "note": "Expected artifacts cover only policy-seeds currently selected in run_selection.json; pending policies are intentionally not missing cells."}
    (root / "manifest.json").write_text(_json(payload), encoding="utf-8")
    (root / "campaign.yaml").write_text(yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8")


def _selected_cells(cfg: Mapping[str, Any], models: Mapping[str, Mapping[str, Any]], suite: str, *, include_diagnostics: bool = False) -> List[Cell]:
    allowed = {(label, int(seed)) for label, model in models.items() for seed in model["training_seeds"]}
    return [cell for cell in _planned_cells(cfg, suite, include_diagnostics=include_diagnostics) if (cell.model, cell.seed) in allowed]


def cmd_plan(cfg: Mapping[str, Any], suite: str, *, labels: Sequence[str] | None = None, include_diagnostics: bool = False) -> int:
    all_cells = _planned_cells(cfg, suite, include_diagnostics=include_diagnostics)
    root = artifact_root(cfg)
    selected = _selected_models(cfg, root, labels) if _selection_path(root).is_file() else {}
    cells = _selected_cells(cfg, selected, suite, include_diagnostics=include_diagnostics) if selected else all_cells
    by_suite: Dict[str, int] = defaultdict(int)
    for cell in cells:
        by_suite[cell.suite] += 1
    print(_json({"campaign": cfg["campaign"], "suite": suite, "process_cells": len(cells), "configured_cells": len(all_cells),
                 "selected_policy_seeds": {label: model["training_seeds"] for label, model in selected.items()}, "by_suite": dict(by_suite),
                 "num_envs_per_cell": cfg["protocol"]["num_envs"], "artifact_root": str(artifact_root(cfg)),
                 "preview": [f"{cell.suite}:{cell.model}:seed{cell.seed}:{cell.name}" for cell in cells[:12]]}))
    return 0


def cmd_run(cfg: Mapping[str, Any], suite: str, *, resume: bool, shard: Tuple[int, int], labels: Sequence[str] | None = None,
            include_diagnostics: bool = False) -> int:
    from legged_gym import SIMULATOR, gs
    if SIMULATOR != "genesis":
        raise RuntimeError(f"SIMULATOR={SIMULATOR!r}; v3_eval needs genesis")
    root = artifact_root(cfg); root.mkdir(parents=True, exist_ok=True)
    models = _selected_models(cfg, root, labels)
    if not models:
        print("[v3_eval] no finalised policy-seed is selected yet; nothing to run")
        _write_manifest(cfg, [])
        return 0
    gs.init(backend=gs.gpu, logging_level="warning")
    selected_cells = _selected_cells(cfg, models, suite, include_diagnostics=include_diagnostics)
    assigned = shard_items(selected_cells, shard)
    print(f"[v3_eval] run suite={suite} shard={shard[0]}/{shard[1]} cells={len(assigned)} resume={resume}")
    for index, cell in enumerate(assigned, start=1):
        print(f"[v3_eval] cell {index}/{len(assigned)} {cell.suite}:{cell.model}:seed{cell.seed}:{cell.name}", flush=True)
        try:
            _run_cell(root, cfg, models, cell, resume=resume)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    _write_manifest(cfg, _selected_cells(cfg, _selected_models(cfg, root), "all", include_diagnostics=include_diagnostics))
    return 0


def _scalar(z, key: str, default: Any = None) -> Any:
    if key not in z:
        return default
    value = z[key]
    if np.asarray(value).shape == ():
        return np.asarray(value).item()
    return value


def _mean(z, key: str) -> float:
    return float(np.mean(np.asarray(z[key], dtype=float)))


def _raw_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((root / "raw").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            suite = str(_scalar(z, "suite"))
            if suite in TERRAIN_SUITES:
                # Terrain cells pack several terrain types into one env dimension;
                # they are expanded per (type x level) by ``_terrain_raw_rows``.
                continue
            scenario = str(_scalar(z, "scenario"))
            error_key = "post_switch_tracking_error" if suite == "s2" else "tracking_lin"
            row = {
                "path": str(path.relative_to(root)), "suite": suite, "scenario": scenario,
                "model": str(_scalar(z, "model")), "training_seed": int(_scalar(z, "training_seed")),
                "tracking_error": _mean(z, error_key), "tracking_lin": _mean(z, "tracking_lin"),
                "fall_rate": _mean(z, "post_switch_fall_rate" if suite == "s2" and "post_switch_fall_rate" in z else "fall_rate"),
                "achieved_speed": _mean(z, "achieved_speed") if "achieved_speed" in z else float("nan"),
                "achieved_speed_ratio": _mean(z, "achieved_speed_ratio") if "achieved_speed_ratio" in z else float("nan"),
                "return_per_step": _mean(z, "return_per_step"),
                "command_speed": float(_scalar(z, "command_speed", 0.0)),
                "headline_command": bool(_scalar(z, "headline_command", True)),
                "diagnostic_kind": str(_scalar(z, "diagnostic_kind", "")),
                "payload_tier": str(_scalar(z, "payload_tier", _scalar(z, "switch_tier", ""))),
                "payload_name": str(_scalar(z, "payload_name", _scalar(z, "switch_name", ""))),
                "command_name": str(_scalar(z, "command_name", "")),
            }
            if suite == "s2":
                row["post_switch_tracking_error"] = _mean(z, "post_switch_tracking_error")
                row["post_switch_tracking_yaw"] = _mean(z, "post_switch_tracking_yaw")
            rows.append(row)
    return rows


def _terrain_raw_rows(root: Path) -> List[Dict[str, Any]]:
    """Expand each terrain cell into one raw row per terrain type present.

    A terrain artifact evaluates every env at one pinned difficulty ``level``
    but splits the env dimension across ``num_cols`` types.  Each unique type is
    reduced over its replica envs here, before any cross-seed aggregation, so a
    seed with more replicas never outweighs another.  The emitted ``scenario``
    encodes both the type and the controlled second axis (level for t1, payload
    for t2), which is exactly the key ``cmd_aggregate`` groups baseline/oracle/
    method on.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted((root / "raw").rglob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            suite = str(_scalar(z, "suite"))
            if suite not in TERRAIN_SUITES:
                continue
            model = str(_scalar(z, "model"))
            training_seed = int(_scalar(z, "training_seed"))
            level = int(_scalar(z, "terrain_level", 0))
            num_cols = int(_scalar(z, "terrain_num_cols", 5))
            terrain_hash = str(_scalar(z, "terrain_hash", ""))
            payload_tier = str(_scalar(z, "payload_tier", ""))
            payload_name = str(_scalar(z, "payload_name", ""))
            command_speed = float(_scalar(z, "command_speed", 0.0))
            terrain_type = np.asarray(z["terrain_type"]).astype(int)
            tracking = np.asarray(z["tracking_lin"], dtype=float)
            fall = np.asarray(z["fall_rate"], dtype=float)
            ratio = np.asarray(z["achieved_speed_ratio"], dtype=float) if "achieved_speed_ratio" in z \
                else np.full(tracking.shape, np.nan)
            speed = np.asarray(z["achieved_speed"], dtype=float) if "achieved_speed" in z \
                else np.full(tracking.shape, np.nan)
            per_step = np.asarray(z["return_per_step"], dtype=float)
            for type_index in sorted(np.unique(terrain_type).tolist()):
                mask = terrain_type == type_index
                type_name = _terrain_type_name(type_index, num_cols)
                if suite == "t2":
                    scenario = f"terrain_type_{type_name}_payload_{payload_name}"
                else:
                    scenario = f"terrain_type_{type_name}_level_{level}"
                rows.append({
                    "path": str(path.relative_to(root)), "suite": suite, "scenario": scenario,
                    "model": model, "training_seed": training_seed,
                    "tracking_error": float(np.mean(tracking[mask])), "tracking_lin": float(np.mean(tracking[mask])),
                    "fall_rate": float(np.mean(fall[mask])),
                    "achieved_speed": float(np.mean(speed[mask])),
                    "achieved_speed_ratio": float(np.mean(ratio[mask])),
                    "return_per_step": float(np.mean(per_step[mask])),
                    "command_speed": command_speed, "headline_command": True, "diagnostic_kind": "",
                    "payload_tier": payload_tier, "payload_name": payload_name, "command_name": "",
                    "terrain_type": int(type_index), "terrain_type_name": type_name, "terrain_level": level,
                    "terrain_hash": terrain_hash, "num_replicas": int(mask.sum()),
                })
    return rows


def score_cell(
    baseline: Mapping[str, float], oracle_candidates: Sequence[Mapping[str, float]], method: Mapping[str, float], score_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pure scorecard rule used by aggregation and CPU-only contract tests."""
    fall_gate = float(score_cfg["fall_gate_pp"])
    speed_gate = float(score_cfg["achieved_speed_ratio"])
    stable = [row for row in oracle_candidates if float(row["fall_rate"]) <= float(baseline["fall_rate"]) + fall_gate]
    speed_ok = [row for row in stable if float(row["achieved_speed_ratio"]) >= speed_gate]
    if not stable:
        return {"headline_status": "oracle_unstable", "headline_include": False, "reasons": "oracle_fall_gate"}
    if not speed_ok:
        return {"headline_status": "oracle_speed_saturated", "headline_include": False, "reasons": "oracle_achieved_speed"}
    oracle = min(speed_ok, key=lambda row: float(row["tracking_error"]))
    numerator = float(baseline["tracking_error"]) - float(method["tracking_error"])
    denominator = float(baseline["tracking_error"]) - float(oracle["tracking_error"])
    rel = denominator / max(abs(float(baseline["tracking_error"])), 1e-12)
    if denominator < float(score_cfg["absolute_headroom"]) or rel < float(score_cfg["relative_headroom"]):
        return {"headline_status": "no_oracle_headroom", "headline_include": False, "reasons": "oracle_headroom",
                "oracle_seed": int(oracle["training_seed"]), "oracle_error": float(oracle["tracking_error"]),
                "oracle_headroom_relative": rel, "oracle_headroom_absolute": denominator}
    raw = numerator / denominator
    clipped = float(np.clip(raw, -0.5, 1.5))
    if float(method["fall_rate"]) > float(baseline["fall_rate"]) + fall_gate:
        return {"headline_status": "method_fall_gated", "headline_include": True, "reasons": "method_fall_gate",
                "raw_gap_closed": raw, "headline_gap_closed": 0.0, "oracle_seed": int(oracle["training_seed"]),
                "oracle_error": float(oracle["tracking_error"]), "oracle_headroom_relative": rel,
                "oracle_headroom_absolute": denominator}
    return {"headline_status": "eligible", "headline_include": True, "reasons": "",
            "raw_gap_closed": raw, "headline_gap_closed": clipped, "oracle_seed": int(oracle["training_seed"]),
            "oracle_error": float(oracle["tracking_error"]), "oracle_headroom_relative": rel,
            "oracle_headroom_absolute": denominator}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _scope(row: Mapping[str, Any]) -> str:
    if row["suite"] == "s1":
        return "GapClosed_static"
    if row["suite"] == "s2":
        return f"GapClosed_dynamic_{row['payload_tier']}"
    if row["suite"] == "t1":
        return f"GapClosed_terrain_{row['terrain_type_name']}"
    if row["suite"] == "t2":
        return f"GapClosed_terrain_payload_{row['payload_tier']}"
    raise ValueError(f"no gap-closed scope for {row['suite']}")


def cmd_aggregate(cfg: Mapping[str, Any]) -> int:
    root = artifact_root(cfg)
    raw = _raw_rows(root) + _terrain_raw_rows(root)
    if not raw:
        raise FileNotFoundError(f"no raw artifacts under {root / 'raw'}")
    _write_csv(root / "tables" / "raw_cells.csv", raw)
    score_cfg = cfg["scorecard"]
    baseline_label, oracle_label = score_cfg["baseline_label"], score_cfg["oracle_label"]
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    diagnostic_raw = []
    for row in raw:
        if row["suite"] in SCORED_SUITES:
            if row["headline_command"]:
                grouped[(row["suite"], row["scenario"])].append(row)
            else:
                diagnostic_raw.append(row)
    score_rows: List[Dict[str, Any]] = []
    for (suite, scenario), rows in grouped.items():
        if suite in TERRAIN_SUITES:
            # The entire scorecard's validity rests on every method seeing a
            # byte-identical heightfield for this (type x level/payload) cell.
            # Fail loud rather than silently compare apples to oranges if any
            # policy's terrain geometry diverged (e.g. a task consumed a
            # different amount of numpy RNG before the terrain was built).
            hashes = {str(row.get("terrain_hash", "")) for row in rows}
            hashes.discard("")
            if len(hashes) > 1:
                offenders = {row["model"]: str(row.get("terrain_hash", ""))[:12] for row in rows}
                raise RuntimeError(
                    f"terrain geometry mismatch in {suite}:{scenario}; methods must share one "
                    f"heightfield but saw {len(hashes)} distinct terrain_hash values: {offenders}"
                )
        baseline_rows = [row for row in rows if row["model"] == baseline_label]
        oracle_rows = [row for row in rows if row["model"] == oracle_label]
        if not baseline_rows or not oracle_rows:
            # Incremental campaigns intentionally accumulate policy-seeds over
            # time. Preserve and report their raw cells now; score a condition
            # only after the baseline and ceiling actually exist together.
            continue
        baseline = {
            "tracking_error": float(np.median([row["tracking_error"] for row in baseline_rows])),
            "fall_rate": float(np.median([row["fall_rate"] for row in baseline_rows])),
        }
        for label in score_cfg["method_labels"]:
            for method in [row for row in rows if row["model"] == label]:
                score = score_cell(baseline, oracle_rows, method, score_cfg)
                score_rows.append({
                    "suite": suite, "scenario": scenario, "scope": _scope(method), "model": label,
                    "training_seed": method["training_seed"], "payload_tier": method["payload_tier"],
                    "payload_name": method["payload_name"], "command_name": method["command_name"],
                    "baseline_tracking_error": baseline["tracking_error"], "baseline_fall_rate": baseline["fall_rate"],
                    "method_tracking_error": method["tracking_error"], "method_fall_rate": method["fall_rate"],
                    "method_achieved_speed_ratio": method["achieved_speed_ratio"], **score,
                })
    _write_csv(root / "tables" / "scorecard_cells.csv", score_rows)
    seed_scores: List[Dict[str, Any]] = []
    by_seed: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if row["headline_include"]:
            by_seed[(row["model"], row["scope"], int(row["training_seed"]))].append(row)
    for (model, scope, seed), rows in by_seed.items():
        values = [float(row["headline_gap_closed"]) for row in rows]
        seed_scores.append({"model": model, "scope": scope, "training_seed": seed, "n_cells": len(values),
                            "gap_closed_median": float(np.median(values)), "gap_closed_min": float(np.min(values)),
                            "gap_closed_max": float(np.max(values))})
    _write_csv(root / "tables" / "scorecard_seed_scores.csv", seed_scores)
    headlines: List[Dict[str, Any]] = []
    by_headline: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in seed_scores:
        by_headline[(row["model"], row["scope"])].append(row)
    for (model, scope), rows in by_headline.items():
        values = [float(row["gap_closed_median"]) for row in rows]
        headlines.append({"model": model, "metric": scope, "n_training_seeds": len(rows),
                          "value_median": float(np.median(values)), "value_min": float(np.min(values)),
                          "value_max": float(np.max(values)), "all_seeds_positive": all(value > 0 for value in values)})
    # S0a is deliberately not transformed into gap_closed. It reports the
    # method-minus-MLP tracking and fall difference on static ID.
    s0_static = [row for row in raw if row["suite"] == "s0" and row["scenario"] == "static_id"]
    mlp_static = [row for row in s0_static if row["model"] == baseline_label]
    if mlp_static:
        baseline_tracking = float(np.median([row["tracking_lin"] for row in mlp_static]))
        baseline_fall = float(np.median([row["fall_rate"] for row in mlp_static]))
        for label in score_cfg["method_labels"]:
            values = [row for row in s0_static if row["model"] == label]
            if values:
                tracking_delta = [row["tracking_lin"] - baseline_tracking for row in values]
                fall_delta = [row["fall_rate"] - baseline_fall for row in values]
                headlines.append({"model": label, "metric": "ID_delta_tracking", "n_training_seeds": len(values),
                                  "value_median": float(np.median(tracking_delta)), "value_min": float(np.min(tracking_delta)),
                                  "value_max": float(np.max(tracking_delta)), "all_seeds_positive": None,
                                  "fall_delta_median": float(np.median(fall_delta))})
    # t0 is descriptive, not a gap-closed suite: report each method's
    # tracking/fall delta versus MLP at the nominal terrain difficulty, per type.
    t0_rows = [row for row in raw if row["suite"] == "t0"]
    for type_name in sorted({str(row["terrain_type_name"]) for row in t0_rows}):
        type_rows = [row for row in t0_rows if str(row["terrain_type_name"]) == type_name]
        mlp_rows = [row for row in type_rows if row["model"] == baseline_label]
        if not mlp_rows:
            continue
        baseline_tracking = float(np.median([row["tracking_lin"] for row in mlp_rows]))
        baseline_fall = float(np.median([row["fall_rate"] for row in mlp_rows]))
        for label in score_cfg["method_labels"]:
            values = [row for row in type_rows if row["model"] == label]
            if not values:
                continue
            tracking_delta = [row["tracking_lin"] - baseline_tracking for row in values]
            fall_delta = [row["fall_rate"] - baseline_fall for row in values]
            headlines.append({"model": label, "metric": f"terrain_ID_delta_tracking_{type_name}",
                              "n_training_seeds": len(values), "value_median": float(np.median(tracking_delta)),
                              "value_min": float(np.min(tracking_delta)), "value_max": float(np.max(tracking_delta)),
                              "all_seeds_positive": None, "fall_delta_median": float(np.median(fall_delta))})
    status_counts: Dict[str, int] = defaultdict(int)
    for row in score_rows:
        status_counts[str(row["headline_status"])] += 1
    summary = {"campaign": cfg["campaign"], "headline": headlines, "status_counts": dict(status_counts),
               "raw_cells": len(raw), "scorecard_cells": len(score_rows), "seed_scores": len(seed_scores)}
    summary["command_ood_diagnostic_raw_cells"] = len(diagnostic_raw)
    (root / "tables" / "headline.json").write_text(_json(summary), encoding="utf-8")
    print(f"[v3_eval] aggregate raw={len(raw)} scorecard_cells={len(score_rows)} -> {root / 'tables'}")
    return 0


def cmd_report(cfg: Mapping[str, Any]) -> int:
    root = artifact_root(cfg)
    summary = json.loads((root / "tables" / "headline.json").read_text(encoding="utf-8"))
    lines = ["# V3 Eval Scorecard", "", f"Campaign: `{summary['campaign']}`", "", "## Headline", "",
             "| Method | Metrik | Median | Min–max | Seed | İşaret tutarlılığı |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["headline"]:
        lines.append(f"| {row['model']} | {row['metric']} | {row['value_median']:.4f} | {row['value_min']:.4f}–{row['value_max']:.4f} | {row['n_training_seeds']} | {row.get('all_seeds_positive', '')} |")
    lines += ["", "## Hücre durumları", "", "| Durum | Hücre |", "|---|---:|"]
    for status, count in sorted(summary["status_counts"].items()):
        lines.append(f"| {status} | {count} |")
    lines += ["", f"Command-OOD diagnostic raw hücreleri: `{summary.get('command_ood_diagnostic_raw_cells', 0)}`. Bunlar headline skoruna dahil edilmez.",
              "", "Ham hücreler `tables/raw_cells.csv`, kapı ve skor hücreleri `tables/scorecard_cells.csv`, seed bazlı özetler `tables/scorecard_seed_scores.csv` içindedir.", ""]
    report = root / "report" / "scorecard.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[v3_eval] report -> {report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="V3 generalisation scorecard batch")
    parser.add_argument("command", choices=("plan", "resolve-runs", "run", "aggregate", "report"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", choices=("s0", "s1", "s2", "s3", "t0", "t1", "t2", "all"), default="all")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", default="", help="comma-separated model labels; use one label to append a newly finished policy")
    parser.add_argument("--strict", action="store_true", help="resolve-runs: fail if a requested policy-seed is not final yet")
    parser.add_argument("--diagnostics", action="store_true", help="also run command-OOD diagnostic cells; never adds them to headline scores")
    args = parser.parse_args()
    cfg = load_config(args.config)
    labels = [label.strip() for label in args.models.split(",") if label.strip()]
    if args.command == "plan":
        return cmd_plan(cfg, args.suite, labels=labels, include_diagnostics=args.diagnostics)
    if args.command == "resolve-runs":
        return cmd_resolve_runs(cfg, labels=labels, strict=args.strict)
    if args.command == "run":
        return cmd_run(cfg, args.suite, resume=args.resume, shard=parse_shard(args.shard), labels=labels,
                       include_diagnostics=args.diagnostics)
    if args.command == "aggregate":
        return cmd_aggregate(cfg)
    return cmd_report(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
