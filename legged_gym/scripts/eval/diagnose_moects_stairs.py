#!/usr/bin/env python3
"""Controlled, training-resolution stairs-up diagnosis for a MoE-CTS policy.

This is deliberately an *evaluation-only* entry point.  It never resumes an
optimizer, never restores the checkpoint's per-environment curriculum state,
and never writes into the training run.  Instead it builds the normal 10 x 20
WTY ``moe_grid`` at its training resolution and pins every evaluation replica
to one existing ``stairs_up`` tile/level.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.envs import *  # noqa: F401,F403 -- registers go2_moects
from legged_gym.utils import get_args, task_registry

if SIMULATOR == "genesis":
    import genesis as gs


STAIRS_UP_ID = 3
STEP_WIDTH_M = 0.31
DEFAULT_LEVELS = (2, 3, 4, 5, 6)
DEFAULT_SEEDS = (101, 202)
CONDITIONS = {
    "A": ("straight", "nominal"),
    "B": ("training", "nominal"),
    "C": ("straight", "training_dr"),
    "D": ("training", "training_dr"),
}


def stairs_step_height(level: int) -> float:
    """Exact IS_HARD stairs height from ``Terrain.make_moe_terrain``."""
    if not 0 <= int(level) <= 9:
        raise ValueError(f"stairs level must be in [0, 9], got {level}")
    return 0.05 + 0.23 * (int(level) / 10.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, str]:
    def read(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=LEGGED_GYM_ROOT_DIR,
                                           text=True).strip()
        except Exception:
            return "unavailable"
    return {"commit": read("git", "rev-parse", "HEAD"),
            "status": read("git", "status", "--short")}


def _parse_args():
    """Keep this script's flags out of the shared strict LeggedGym parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--levels", nargs="+", type=int, default=list(DEFAULT_LEVELS))
    parser.add_argument("--replicas", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS),
                        default=sorted(CONDITIONS))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(LEGGED_GYM_ROOT_DIR) / "logs/eval/moe_cts_7500_stairs_diagnostic")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--include-push", action="store_true",
                        help="include training push perturbations in training_dr (off by default)")
    parser.add_argument("--keep-observation-noise", action="store_true")
    ours, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = get_args()
    args.__dict__.update(vars(ours))
    args.headless = True
    if args.task != "go2_moects":
        raise SystemExit("diagnose_moects_stairs accepts only --task go2_moects")
    if args.replicas < 1:
        raise SystemExit("--replicas must be positive")
    if any(level not in range(10) for level in args.levels):
        raise SystemExit("--levels must be within 0..9")
    if args.smoke:
        args.steps = args.steps or 50
    elif args.steps is None:
        args.steps = 1250
    return args


def configure_eval_cfg(cfg: Any, *, replicas: int, physics: str, aligned_yaw: bool,
                       include_push: bool, keep_observation_noise: bool) -> None:
    """Apply only evaluation overrides; keep the WTY training terrain intact."""
    cfg.env.num_envs = replicas
    cfg.env.auto_reset = False                 # first terminal state is terminal
    cfg.env.debug = False
    cfg.terrain.curriculum = False             # WTY owns it; evaluation pins it below
    cfg.terrain.selected = False
    cfg.terrain.moe_grid = False
    cfg.terrain.moe_showcase = True
    # Build only the one semantic column the diagnosis measures.  Each of its
    # ten rows still calls the *unchanged* training ``make_moe_terrain`` at
    # the training horizontal/vertical scales; omitting the other 19 columns
    # avoids turning an evaluation of one tile into a giant 200-tile Genesis
    # SDF preprocessing job.
    # ``moe_grid_showcase`` keeps its difficulty denominator fixed to the
    # training 10 rows, unlike ``moe_grid``.  Seven rows therefore encode
    # exactly L0..L6 (including all requested L2..L6) while staying below the
    # huge global-heightfield SDF preprocessing threshold.
    cfg.terrain.num_rows = 7
    cfg.terrain.moe_showcase_levels = list(range(7))
    cfg.terrain.num_cols = 1
    # The 25-m training-world outer border is never reachable from an 8-m
    # diagnostic tile within one episode; retain a small safety border without
    # inflating the heightfield/SDF by the otherwise irrelevant outer world.
    cfg.terrain.border_size = 1.0
    cfg.terrain.terrain_proportions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if aligned_yaw:
        # The reset sampler consumes this from init_state (not env).
        cfg.init_state.yaw_random_scale = 0.0
    if not keep_observation_noise:
        cfg.noise.add_noise = False
    if physics == "nominal":
        from legged_gym.scripts.play import _disable_play_domain_rand
        _disable_play_domain_rand(cfg)
    elif physics == "training_dr":
        # The task config remains authoritative. Push is a separate stressor:
        # unlike static DR it injects time-local stochastic disturbances and
        # would confound the main DR-only comparison.
        cfg.domain_rand.push_robots = bool(include_push)
        if hasattr(cfg.domain_rand, "push_links"):
            cfg.domain_rand.push_links = False
    else:
        raise ValueError(f"unknown physics regime {physics!r}")


def pin_stairs_up(env: Any, level: int) -> dict[str, Any]:
    """Pin all envs to one genuine training-grid stairs-up tile and hash it."""
    terrain = env.simulator._terrain
    cols = sorted(int(x) for x in terrain.name2cols["stairs_up"])
    if not cols:
        raise RuntimeError("training moe_grid has no stairs_up column")
    col = cols[0]
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.simulator._terrain_levels[ids] = int(level)
    env.simulator._terrain_types[ids] = col
    env.wty_terrain_ids[ids] = STAIRS_UP_ID
    env.simulator._env_origins[ids] = env.simulator._terrain_origins[level, col]
    env._update_env_command_ranges()
    # Crop exactly the generated training tile, not the global grid/gaps.
    row0 = terrain.border + level * (terrain.length_per_env_pixels + terrain.spacing_pixels)
    col0 = terrain.border + col * (terrain.width_per_env_pixels + terrain.spacing_pixels)
    raw = np.ascontiguousarray(terrain.height_field_raw[
        row0:row0 + terrain.length_per_env_pixels,
        col0:col0 + terrain.width_per_env_pixels])
    return {"level": int(level), "column": col, "step_height_m": stairs_step_height(level),
            "step_width_m": STEP_WIDTH_M, "horizontal_scale": float(env.cfg.terrain.horizontal_scale),
            "vertical_scale": float(env.cfg.terrain.vertical_scale),
            "heightfield_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_path(args: Any) -> Path:
    run = args.load_run
    if not run:
        raise SystemExit("--load_run is required; refusing to select an arbitrary checkpoint")
    root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "go2_moects"
    run_path = Path(run).expanduser()
    run_path = run_path if run_path.is_absolute() else root / run_path
    ckpt = str(args.ckpt)
    name = ckpt if ckpt.endswith(".pt") else f"model_{ckpt}.pt"
    path = run_path / name
    if not path.is_file():
        raise SystemExit(f"checkpoint does not exist: {path}")
    return path.resolve()


def _policy_probe(runner: Any, env: Any) -> None:
    probe = torch.zeros(1, env.cfg.env.num_observations, device=env.device)
    probe[0, 5] = -1.0
    with torch.no_grad():
        value = runner.alg.actor_critic.act_student(
            probe, probe.repeat(1, env.cfg.env.frame_stack)).abs().mean().item()
    if value < 0.1:
        raise RuntimeError(f"loaded policy probe too small ({value:.4f}); refusing random policy")


def _contact_metrics(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    force = env.simulator.link_contact_forces.norm(dim=-1)
    indices = env.simulator.penalized_contact_indices
    penalized = force[:, indices]
    # For go2_moects the configured penalized set is exactly thigh/calf links.
    # Use the canonical index set rather than simulator-private link-name data.
    collision = (penalized > env.cfg.rewards.collision_force_threshold).any(dim=1)
    base = env.terminated_by_base_contact.bool()
    return base, collision


def rollout(env: Any, adapter: Any, *, command_mode: str, steps: int) -> dict[str, np.ndarray]:
    state = adapter.reset(env)
    n = env.num_envs
    dev = env.device
    initial_origin = env.simulator.env_origins.clone()
    initial_z = env.simulator.base_pos[:, 2].clone()
    alive = torch.ones(n, device=dev, dtype=torch.bool)
    first_fall = torch.full((n,), -1, device=dev, dtype=torch.long)
    base_term = torch.zeros(n, device=dev, dtype=torch.bool)
    timeout = torch.zeros(n, device=dev, dtype=torch.bool)
    sum_err = torch.zeros(n, device=dev); sum_err_sq = torch.zeros(n, device=dev)
    bad = torch.zeros(n, device=dev); num_metric = torch.zeros(n, device=dev)
    base_h = torch.zeros(n, device=dev); tilt = torch.zeros(n, device=dev)
    collision = torch.zeros(n, device=dev); torque_sq = torch.zeros(n, device=dev)
    power = torch.zeros(n, device=dev); action_rate = torch.zeros(n, device=dev)
    foot_slip = torch.zeros(n, device=dev); returns = torch.zeros(n, device=dev)
    max_z = env.simulator.base_pos[:, 2].clone()
    max_progress = torch.zeros(n, device=dev)
    previous_actions = None
    initial_commands = torch.zeros(n, 3, device=dev)

    if command_mode == "straight":
        env.commands[:, :3] = torch.tensor((0.5, 0.0, 0.0), device=dev)
        env.commands_resampling_step[:] = float("inf")
    elif command_mode != "training":
        raise ValueError(command_mode)

    for step in range(steps):
        commands = env.commands[:, :3].clone()
        if step == 0:
            initial_commands.copy_(commands)
        with torch.no_grad():
            actions = adapter.act(state)
        state, reward, dones = adapter.step(env, actions)
        mask = alive.float()
        error = torch.norm(commands[:, :2] - env.simulator.base_lin_vel[:, :2], dim=1)
        sum_err += error * mask; sum_err_sq += error.square() * mask
        bad += (error >= 0.30).float() * mask; num_metric += mask
        base_h += env.simulator.base_pos[:, 2] * mask
        tilt += torch.acos(torch.clamp(-env.simulator.projected_gravity[:, 2], -1., 1.)) * mask
        _, limb_collision = _contact_metrics(env)
        collision += limb_collision.float() * mask
        torque_sq += env.simulator.torques.square().mean(dim=1) * mask
        power += torch.abs(env.simulator.torques * env.simulator.dof_vel).mean(dim=1) * mask
        if previous_actions is not None:
            action_rate += (actions - previous_actions).abs().mean(dim=1) * mask
        previous_actions = actions.clone()
        feet = env.simulator.feet_indices
        contact = env.simulator.link_contact_forces[:, feet].norm(dim=-1) > 1.0
        # Genesis feet_vel is already compact ``(N, 4, 3)``; feet contains
        # link indices for the contact-force tensor only.
        speed = env.simulator.feet_vel[:, :, :2].norm(dim=-1)
        foot_slip += ((speed > 0.1) & contact).float().mean(dim=1) * mask
        returns += reward * mask
        delta = env.simulator.base_pos - initial_origin
        # The aligned stairs test uses the tile's world +X exit direction.
        progress = delta[:, 0]
        max_progress = torch.maximum(max_progress, progress)
        max_z = torch.maximum(max_z, env.simulator.base_pos[:, 2])
        done_now = dones.bool() & alive
        if done_now.any():
            first_fall[done_now] = step + 1
            base_term[done_now] = env.terminated_by_base_contact[done_now]
            timeout[done_now] = env.time_out_buf[done_now]
            alive[done_now] = False

    survived = torch.where(first_fall < 0, torch.full_like(first_fall, steps), first_fall)
    denom = torch.clamp(num_metric, min=1.)
    spnte = sum_err / denom
    # Meaningful progress means at least one 0.31-m riser-width forward.  The
    # same distance makes "standing still" or motion parallel to the stairs fail.
    success = ((survived >= int(0.9 * steps)) & ~base_term &
               (max_progress >= STEP_WIDTH_M) & (spnte < 0.30))
    return {k: v.detach().cpu().numpy() for k, v in {
        "success": success, "fall": first_fall >= 0, "first_fall_step": first_fall,
        "survival_steps": survived, "spnte_lin": spnte, "tracking_lin_err": spnte,
        "tracking_lin_rmse": torch.sqrt(sum_err_sq / denom),
        "tracking_lin_bad_frac": bad / denom, "signed_climb_progress_m": max_progress,
        "forward_distance_m": max_progress, "net_base_height_gain_m": max_z - initial_z,
        "max_base_height_m": max_z, "net_steps": (max_progress / STEP_WIDTH_M).floor(),
        "base_contact_termination": base_term, "timeout": timeout,
        "base_height_mean": base_h / denom, "tilt_mean_rad": tilt / denom,
        "thigh_calf_collision_frac": collision / denom, "torque_sq_mean": torque_sq / denom,
        "mechanical_power_mean": power / denom, "action_rate_mean": action_rate / denom,
        "foot_slip_frac": foot_slip / denom, "episode_return": returns,
        "initial_vx": initial_commands[:, 0], "initial_vy": initial_commands[:, 1],
        "initial_yaw": initial_commands[:, 2],
    }.items()}


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["condition"], row["level"]), []).append(row)
    out = []
    for (condition, level), values in sorted(groups.items()):
        item: dict[str, Any] = {"condition": condition, "level": level,
                                "step_height_m": stairs_step_height(level), "replicas": len(values)}
        for key in values[0]:
            if isinstance(values[0][key], (float, int, np.floating, np.integer, bool)) and key not in item:
                item[key] = float(np.mean([x[key] for x in values]))
        out.append(item)
    return out


def main() -> None:
    args = _parse_args()
    if SIMULATOR != "genesis":
        raise SystemExit("this diagnosis currently requires SIMULATOR=genesis")
    checkpoint = _checkpoint_path(args)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing diagnostic output: {output}")
    output.mkdir(parents=True)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    checkpoint_meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
    iteration = int(checkpoint_meta["iter"])
    rows: list[dict[str, Any]] = []
    terrain_hashes: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        for condition in args.conditions:
            command_mode, physics = CONDITIONS[condition]
            _set_seed(seed)
            env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
            configure_eval_cfg(env_cfg, replicas=args.replicas, physics=physics,
                               aligned_yaw=command_mode == "straight",
                               include_push=args.include_push,
                               keep_observation_noise=args.keep_observation_noise)
            env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
            # Restore schedule phase, but never the checkpoint's 8192-env
            # terrain state: evaluation explicitly pins its own level.
            env.common_step_counter = iteration * env.cfg.env.wty_steps_per_iteration
            env.set_wty_total_iterations(train_cfg.runner.max_iterations)
            train_cfg.runner.resume = False
            runner, _ = task_registry.make_alg_runner(env, args.task, args, train_cfg,
                                                       log_root=None, load_env_curriculum=False)
            runner.load_deploy_state(str(checkpoint))
            adapter = runner.get_eval_adapter(device=env.device)
            _policy_probe(runner, env)
            for level in args.levels:
                terrain_hashes[f"L{level}"] = pin_stairs_up(env, level)
                result = rollout(env, adapter, command_mode=command_mode, steps=args.steps)
                for replica in range(args.replicas):
                    rows.append({"seed": seed, "condition": condition, "command_regime": command_mode,
                                 "physics_regime": physics, "level": int(level), "replica": replica,
                                 "step_height_m": stairs_step_height(level),
                                 **{name: (bool(values[replica]) if values.dtype == np.bool_ else float(values[replica]))
                                    for name, values in result.items()}})
            # Genesis owns the scene lifetime; dropping the Python references is
            # sufficient between sequential cells/conditions.
            del runner, env
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = _aggregate(rows)
    with (output / "cells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / "replicas.npz", **{key: np.asarray([r[key] for r in rows]) for key in rows[0]})
    (output / "terrain_hashes.json").write_text(json.dumps(terrain_hashes, indent=2, sort_keys=True) + "\n")
    manifest = {"checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_iteration": iteration, "task": args.task, "simulator": SIMULATOR,
                "torch": torch.__version__, "genesis": getattr(gs, "__version__", "unknown"),
                "git": _git_state(), "levels": args.levels, "seeds": args.seeds,
                "replicas_per_seed": args.replicas, "horizon_steps": args.steps,
                "conditions": {key: CONDITIONS[key] for key in args.conditions},
                "training_dr_push_included": args.include_push,
                "success_contract": {"survival_fraction": 0.90, "min_forward_progress_m": STEP_WIDTH_M,
                                     "spnte_lin_strictly_below": 0.30, "no_base_contact_termination": True},
                "terrain_generator": terrain_hashes}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# MoE-CTS stairs-up diagnostic", "", "| Koşul | Seviye | h (m) | n | Başarı | Düşüş | SPNTE | İlerleme (m) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summary:
        lines.append(f"| {x['condition']} | L{x['level']} | {x['step_height_m']:.3f} | {x['replicas']:.0f} | {x['success']:.3f} | {x['fall']:.3f} | {x['spnte_lin']:.3f} | {x['signed_climb_progress_m']:.3f} |")
    lines += ["", "Ana DR hücreleri push içermez; `--include-push` ayrı push-stres koşusunu üretir.", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "cells": len(summary), "replicas": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
