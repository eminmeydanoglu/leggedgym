"""Minimal base-velocity estimation probe for RMA, DreamWaQ, and HIM.

Separate from the added-mass learn+use probe. Answers only:

  How well does the method represent base linear velocity [vx, vy, vz]?

Method-specific readout (different mechanisms — do not collapse into one mass table):
  RMA:      frozen decoder  z_s → v  and  z_t → v   (latent-embedded)
  DreamWaQ: direct          vel_mu  vs true base_lin_vel  (explicit head)
  HIM:      direct          vel_hat vs true base_lin_vel  (explicit head)

Default velocity protocol (richer than mass probe's single lateral):
  - multi-command schedule ±vx/±vy/diagonal/stand (DEFAULT_VEL_COMMAND_SCHEDULE)
  - --single_command for one (vx,vy,yaw) point only
  - mass grid [-2,0,+3,+5], friction/CoM nominal; push off; V3 switch off
  - command-stratified traj split; identifiable-dim R²; target std reported

Usage:
  python legged_gym/scripts/eval/probe_velocity.py \\
      --method rma --task go2_v3_rma --load_run <run> --seed_label 1 \\
      --out_dir logs/eval/probes/velocity/rma/seed_1

  # offline:
  python legged_gym/scripts/eval/probe_velocity.py \\
      --analyze_only --samples vel_samples.npz --method him --out_dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from legged_gym.scripts.eval.probe_adapters import get_adapter
from legged_gym.scripts.eval.probe_velocity_logic import (
    DEFAULT_MEASURE_STEPS,
    DEFAULT_PER_POINT,
    DEFAULT_VEL_COMMAND_SCHEDULE,
    DEFAULT_WARMUP,
    LATERAL_CMD,
    MASS_GRID_KG,
    analyze_velocity_samples,
)

# Reuse mass-probe env path (physics contract, step/obs normalization)
from legged_gym.scripts.eval.probe_physics_use import (
    _apply_mass_grid,
    _lazy_gym_imports,
    _read_live_mass,
    _set_command,
    _step_env,
    _unpack_obs,
    _warmup,
    build_env_and_policy,
    default_task_for_method,
)
from legged_gym.scripts.eval.probe_physics_logic import (
    check_mass_invariant,
    mask_valid_measurement,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Base-velocity estimation probe (RMA / DreamWaQ / HIM). "
            "RMA: traj-split decoder z→v. DreamWaQ/HIM: explicit vel head vs true v. "
            "Separate from the added-mass learn+use probe."
        )
    )
    p.add_argument(
        "--method",
        type=str,
        default="rma",
        choices=["rma", "dreamwaq", "him", "go2_v3_rma", "go2_v3_dreamwaq", "go2_v3_him_fixed"],
    )
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--load_run", type=str, default=None)
    p.add_argument("--ckpt", type=str, default="best_tracking")
    p.add_argument("--seed_label", type=str, default="1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--per_point", type=int, default=DEFAULT_PER_POINT)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--steps", type=int, default=DEFAULT_MEASURE_STEPS)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--mass_grid", type=float, nargs="+", default=list(MASS_GRID_KG))
    p.add_argument("--command_vx", type=float, default=LATERAL_CMD[0],
                   help="used only with --single_command")
    p.add_argument("--command_vy", type=float, default=LATERAL_CMD[1],
                   help="used only with --single_command")
    p.add_argument("--command_yaw", type=float, default=LATERAL_CMD[2],
                   help="used only with --single_command")
    p.add_argument(
        "--single_command",
        action="store_true",
        default=False,
        help="use only (--command_vx,vy,yaw); default is multi-command schedule",
    )
    p.add_argument("--steps_per_cmd", type=int, default=None,
                   help="measure steps per command in multi schedule (default: steps)")
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--analyze_only", action="store_true")
    p.add_argument("--samples", type=str, default=None)
    return p.parse_args(argv)


def _read_live_base_vel(env) -> np.ndarray:
    return env.simulator.base_lin_vel.detach().cpu().numpy().astype(np.float32)


def _command_schedule(cli) -> list:
    if getattr(cli, "single_command", False):
        return [[cli.command_vx, cli.command_vy, cli.command_yaw]]
    return [list(c) for c in DEFAULT_VEL_COMMAND_SCHEDULE]


def collect_velocity_dataset(cli, env, ac, adapter, g) -> Dict[str, np.ndarray]:
    """Roll out under multi-command schedule; record true vel + method features.

    Default schedule spans forward/lateral/negative/stand to excite vx,vy.
    traj_id encodes (command_index * n_envs + env_id) so splits stay traj-safe.
    """
    adapter_name = adapter.name
    schedule = _command_schedule(cli)
    steps_per = int(cli.steps_per_cmd or cli.steps)
    device = env.device
    n = env.num_envs

    mass_targets = _apply_mass_grid(env, cli.mass_grid, cli.per_point, g)

    rows_vel = []
    rows_traj = []
    rows_zs = []
    rows_zt = []
    rows_vhat = []
    rows_cmd = []
    rows_cmd_id = []

    for cmd_i, cmd_vec in enumerate(schedule):
        print(f"[probe-vel] command[{cmd_i}]={cmd_vec} steps={steps_per}")
        # pin env command ranges so resampling stays at this command
        if hasattr(env, "command_ranges"):
            env.command_ranges["lin_vel_x"] = [cmd_vec[0], cmd_vec[0]]
            env.command_ranges["lin_vel_y"] = [cmd_vec[1], cmd_vec[1]]
            env.command_ranges["ang_vel_yaw"] = [cmd_vec[2], cmd_vec[2]]
        mass_targets = _apply_mass_grid(env, cli.mass_grid, cli.per_point, g)
        state = _warmup(env, ac, adapter, cmd_vec, cli.warmup, adapter_name)
        live0 = _read_live_mass(env)
        check_mass_invariant(live0, mass_targets, atol=1e-2)
        ever_invalid = np.zeros(n, dtype=bool)

        with torch.no_grad():
            for step_idx in range(steps_per):
                _set_command(env, cmd_vec, device)
                live_mass = _read_live_mass(env)
                check_mass_invariant(live_mass, mass_targets, atol=1e-2)
                vel_true = _read_live_base_vel(env)

                state_act = dict(state)
                state_act["real_mass_raw"] = torch.as_tensor(
                    live_mass, device=device, dtype=torch.float32
                )
                actions = adapter.act_normal(ac, state_act)
                latents = adapter.extract_latent(ac, state_act)

                state_new, rew, dones, extras = _step_env(env, actions, adapter_name)
                fall = (dones.bool() & ~env.time_out_buf.bool()).detach().cpu().numpy()
                done_np = dones.bool().detach().cpu().numpy()
                ever_invalid = ever_invalid | fall | done_np

                if step_idx % cli.stride == 0:
                    keep = mask_valid_measurement(
                        fall, done_np, ever_invalid=ever_invalid
                    )
                    keep = keep & (np.abs(live_mass - mass_targets) <= 1e-2)
                    idx = np.where(keep)[0]
                    if idx.size:
                        rows_vel.append(vel_true[idx])
                        # unique traj per (command, env)
                        traj = (cmd_i * n + idx).astype(np.int64)
                        rows_traj.append(traj)
                        rows_cmd.append(
                            np.tile(
                                np.asarray(cmd_vec, dtype=np.float32), (idx.size, 1)
                            )
                        )
                        rows_cmd_id.append(
                            np.full(idx.size, cmd_i, dtype=np.int32)
                        )
                        if "z_s" in latents:
                            rows_zs.append(
                                latents["z_s"].detach().cpu().float().numpy()[idx]
                            )
                        if "z_t" in latents:
                            rows_zt.append(
                                latents["z_t"].detach().cpu().float().numpy()[idx]
                            )
                        if "vel_mu" in latents:
                            rows_vhat.append(
                                latents["vel_mu"].detach().cpu().float().numpy()[idx]
                            )
                        elif "vel_hat" in latents:
                            rows_vhat.append(
                                latents["vel_hat"].detach().cpu().float().numpy()[idx]
                            )
                        elif "vel" in latents and adapter_name != "RMA":
                            rows_vhat.append(
                                latents["vel"].detach().cpu().float().numpy()[idx]
                            )

                if np.any(done_np):
                    _apply_mass_grid(env, cli.mass_grid, cli.per_point, g)
                state = state_new
                _set_command(env, cmd_vec, device)

    if not rows_vel:
        raise RuntimeError("no valid velocity samples collected (all fell?)")

    out: Dict[str, np.ndarray] = {
        "vel_true": np.concatenate(rows_vel, axis=0).astype(np.float32),
        "traj_id": np.concatenate(rows_traj, axis=0).astype(np.int64),
        "command": np.concatenate(rows_cmd, axis=0).astype(np.float32),
        "command_id": np.concatenate(rows_cmd_id, axis=0).astype(np.int32),
        "mass_targets_per_env": mass_targets.astype(np.float32),
        "schedule": np.asarray(schedule, dtype=np.float32),
    }
    if rows_zs:
        out["z_s"] = np.concatenate(rows_zs, axis=0).astype(np.float32)
    if rows_zt:
        out["z_t"] = np.concatenate(rows_zt, axis=0).astype(np.float32)
    if rows_vhat:
        out["vel_hat"] = np.concatenate(rows_vhat, axis=0).astype(np.float32)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    cli = parse_args(argv)
    if cli.task is None:
        cli.task = default_task_for_method(cli.method)
    if cli.analyze_only:
        cli.skip_collect = True

    os.makedirs(cli.out_dir, exist_ok=True)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    adapter = get_adapter(cli.method)
    print(
        f"[probe-vel] method={adapter.name} task={cli.task} "
        f"seed_label={cli.seed_label}"
    )
    sched = _command_schedule(cli)
    print(f"[probe-vel] commands={sched} mass_grid={cli.mass_grid}")

    if cli.skip_collect:
        if not cli.samples:
            print("[probe-vel] --analyze_only requires --samples", file=sys.stderr)
            return 2
        data = np.load(cli.samples, allow_pickle=True)
        samples = {k: data[k] for k in data.files if k != "meta"}
    else:
        g = _lazy_gym_imports()
        num_envs = len(cli.mass_grid) * cli.per_point
        env, ac, env_cfg, train_cfg, chosen_run, ckpt_path, contract = (
            build_env_and_policy(cli, num_envs)
        )
        samples = collect_velocity_dataset(cli, env, ac, adapter, g)
        samples_path = os.path.join(cli.out_dir, "vel_samples.npz")
        meta = {
            "probe": "velocity",
            "method": adapter.name,
            "task": cli.task,
            "seed_label": cli.seed_label,
            "mass_grid": list(cli.mass_grid),
            "command": [cli.command_vx, cli.command_vy, cli.command_yaw],
            "warmup": cli.warmup,
            "steps": cli.steps,
            "stride": cli.stride,
            "per_point": cli.per_point,
            "load_run": getattr(cli, "load_run", None),
            "ckpt": ckpt_path,
            "contract": contract,
        }
        np.savez(samples_path, meta=np.array(json.dumps(meta)), **samples)
        print(
            f"[probe-vel] samples -> {samples_path}  N={samples['vel_true'].shape[0]}"
        )

    result = analyze_velocity_samples(
        samples,
        method=cli.method,
        seed_label=cli.seed_label,
        seed=cli.seed,
    )
    out_json = os.path.join(cli.out_dir, "velocity_result.json")
    def _jsonify(obj):
        if isinstance(obj, (float, np.floating)):
            return float(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (str, bool)) or obj is None:
            return obj
        return str(obj)

    serial = {
        "rows": _jsonify(result["rows"]),
        "details": {
            k: _jsonify({
                kk: vv for kk, vv in det.items()
                if kk not in ("y_true", "y_pred", "split", "real", "shuffled")
            })
            for k, det in result["details"].items()
        },
        "table_md": result["table_md"],
    }
    with open(out_json, "w") as f:
        json.dump(serial, f, indent=2, default=str)
    table_path = os.path.join(cli.out_dir, "velocity_table.md")
    with open(table_path, "w") as f:
        f.write(result["table_md"] + "\n")
    print(result["table_md"])
    print(f"[probe-vel] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
