#!/usr/bin/env python3
"""One-process Genesis smoke for the native V4-style V6 terrain bank.

This is deliberately not a training/evaluation harness: it builds the exact
10x10, 8 m training heightfield, enables the real frontier adapter, takes a short
zero-action rollout, and writes runtime/health/outcome distributions as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SIMULATOR", "genesis")

import genesis as gs
import numpy as np
import torch

import legged_gym.envs  # noqa: F401 - task registration
from legged_gym.envs.go2.go2_v6_frontier_config import build_frontier_teacher
from legged_gym.utils import task_registry


def _args(num_envs: int) -> SimpleNamespace:
    return SimpleNamespace(
        task="go2_v6_frontier", seed=17, debug=False, headless=True, cpu=False,
        num_envs=num_envs, max_iterations=None, resume=False, sync_wandb=False,
        ckpt=None, load_run=None, export_onnx=False, motion_file=None,
        motion_out_dir=None, num_student=None, use_joystick=False,
        joystick_type="xbox", follow_robot=False, viewer="native",
        viser_port=8080,
    )


def _percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "min": float(values.min()), "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _rss_mib() -> float:
    # Linux ru_maxrss is KiB; it is a process-lifetime peak, intentionally.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1100)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.num_envs <= 0 or cli.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")

    gs.init(backend=gs.gpu, logging_level="warning")
    task = "go2_v6_frontier"
    registered, _ = task_registry.get_cfgs(task)
    # The config singleton is shared inside the process.  This script exits
    # after one scene, so changing only the vector count is contained here.
    registered.env.num_envs = cli.num_envs
    t0 = time.perf_counter()
    env, cfg = task_registry.make_env(task, args=_args(cli.num_envs), env_cfg=registered)
    env_init_s = time.perf_counter() - t0
    teacher, task_space = build_frontier_teacher(cfg)
    completed = []
    observe = teacher.observe

    def capture(outcomes):
        completed.append(outcomes)
        observe(outcomes)

    teacher.observe = capture
    t0 = time.perf_counter()
    env.enable_ued(teacher, task_space)
    ued_enable_s = time.perf_counter() - t0

    terrain = env.simulator._terrain
    heightfield = terrain.height_field_raw
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    finite = True
    t0 = time.perf_counter()
    for _ in range(cli.steps):
        obs, privileged, rewards, dones, infos = env.step(actions)
        finite = finite and bool(torch.isfinite(rewards).all())
        if isinstance(obs, tuple):
            finite = finite and all(bool(torch.isfinite(part).all()) for part in obs)
        else:
            finite = finite and bool(torch.isfinite(obs).all())
        if privileged is not None:
            finite = finite and bool(torch.isfinite(privileged).all())
    rollout_s = time.perf_counter() - t0

    def concat(field, dtype=np.float64):
        chunks = [np.asarray(getattr(batch, field), dtype=dtype) for batch in completed]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)

    lengths = concat("episode_lengths", np.int64)
    terminal = concat("terminal_reasons", "U16")
    success = concat("successes", np.bool_)
    linear = concat("mean_linear_errors")
    yaw = concat("mean_yaw_errors")
    on_tile = concat("on_tile_fractions")
    first_exit = concat("first_exit_steps")
    max_dx = concat("max_abs_dx")
    max_dy = concat("max_abs_dy")
    output = {
        "contract": (
            "V6 native 8x8 m full 10x10 starting-terrain bank; "
            "zero-action smoke only, not policy evaluation"
        ),
        "task": task,
        "num_envs": cli.num_envs,
        "control_steps": cli.steps,
        "terrain": {
            "rows": int(cfg.terrain.num_rows), "cols": int(cfg.terrain.num_cols),
            "tile_length_m": float(cfg.terrain.terrain_length),
            "tile_width_m": float(cfg.terrain.terrain_width),
            "heightfield_shape": list(heightfield.shape),
            "heightfield_dtype": str(heightfield.dtype),
            "heightfield_nbytes": int(heightfield.nbytes),
        },
        "timing_s": {
            "env_init_includes_terrain_and_collider": env_init_s,
            "ued_enable": ued_enable_s, "rollout": rollout_s,
            "env_steps_per_s": float(cli.steps * cli.num_envs / rollout_s),
        },
        "peak_rss_mib": _rss_mib(),
        "finite_observations_and_rewards": finite,
        "outcomes": {
            "completed_episodes": int(len(lengths)),
            "survival_rate": float(np.mean(terminal == "timeout")) if len(terminal) else None,
            "success_rate": float(np.mean(success)) if len(success) else None,
            "episode_length_steps": _percentiles(lengths),
            "normalized_linear_error": _percentiles(linear),
            "normalized_yaw_error": _percentiles(yaw),
            "on_tile_fraction": _percentiles(on_tile),
            "left_tile_fraction": float(np.mean(on_tile < 1.0)) if len(on_tile) else None,
            "first_exit_step": _percentiles(first_exit[np.isfinite(first_exit)]),
            "max_abs_dx_m": _percentiles(max_dx),
            "max_abs_dy_m": _percentiles(max_dy),
        },
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
