"""GPU rollout for the V6 frontier diagnostic bank (one geometry seed per process).

Genesis scenes are not safely destroyed/rebuilt in-process, so each
``geometry_seed`` is a separate process.  Shard within a seed with
``--num_shards`` / ``--shard_index`` when VRAM cannot hold the full batch.

Runtime contract (v2):
- ``terrain.curriculum`` is forced off so native level promotion cannot move
  robots off the requested L0 tile during ``env.reset()`` / auto-reset.
- Terrain type/level/origin are assigned *after* reset and asserted.
- Commands are pinned and ``compute_observations()`` is called before the
  first policy action so obs sees the bank command.
- ``timed_out`` is the observed simulator ``time_out_buf`` event
  (``episode_length_buf > max_episode_length``), not merely "did not fall".
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR  # noqa: F401
import legged_gym.envs  # noqa: F401
from legged_gym.utils import task_registry
from legged_gym.utils.terrain import taxonomy_tile_geometry_hash

from legged_gym.scripts.eval.ckpt_utils import sha256_file
from legged_gym.scripts.eval.dr_axes import pin_others_to_nominal
from legged_gym.scripts.eval.frontier_diagnostic import (
    SCHEMA_VERSION,
    DiagnosticRow,
    artifact_dir,
    bank_fingerprint,
    build_bank,
    config_fingerprint,
    expected_episode_count,
    frontier_success,
    load_config,
    write_bank_artifacts,
    write_ndjson,
)
from legged_gym.scripts.eval.metrics import MetricAccumulator

# Far past any episode horizon so bank-pinned commands are never auto-resampled.
_BANK_EVAL_RESAMPLING_TIME_S = 1e6


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", LEGGED_GYM_ROOT_DIR, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _working_tree_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", LEGGED_GYM_ROOT_DIR, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:
        return None


def apply_frontier_diag_env_overrides(env_cfg, *, geometry_seed: int, episode_length_s: float) -> None:
    """Pin a V6 frontier env for offline diagnostic scoring (no teacher).

    Training episode length is preserved so ``max_episode_length`` and the
    ``episode_length_buf > max_episode_length`` timeout match training.

    Terrain generation still uses the curriculum 10x10 builder
    (``terrain.curriculum=True`` at construction).  Runtime level progression
    is disabled *after* ``make_env`` via ``disable_runtime_terrain_curriculum``.
    """
    env_cfg.env.ued_enabled = False
    # CRITICAL: do not auto-reset mid-rollout.  With auto_reset=True the step that
    # sets time_out_buf/fall also calls reset_idx before returning, so post-step
    # tracking / on-tile samples would come from the *new* episode state.
    # Training success is defined on the completed episode terminal state.
    env_cfg.env.auto_reset = False
    env_cfg.env.episode_length_s = float(episode_length_s)
    env_cfg.seed = int(geometry_seed)
    env_cfg.terrain.mesh_type = "heightfield"
    # True only so Terrain.__init__ builds the V4/V6 curriculum heightfield
    # (10 cols x 10 levels), not randomized_terrain().
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.selected = False
    env_cfg.terrain.ued_training_grid = False
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.terrain_length = 8.0
    env_cfg.terrain.terrain_width = 8.0
    env_cfg.terrain.terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]
    env_cfg.terrain.terrain_replica_variation = 0.10
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.legacy_performance_command_curriculum_enabled = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.per_env_standstill = False
    env_cfg.commands.resampling_time = _BANK_EVAL_RESAMPLING_TIME_S


def disable_runtime_terrain_curriculum(env) -> None:
    """Stop native move_up/move_down from rewriting terrain_levels on reset.

    Must run after the terrain mesh is built (which requires curriculum=True).
    ``reset_idx`` gates progression on ``cfg.terrain.curriculum``.
    """
    env.cfg.terrain.curriculum = False
    # Belt-and-suspenders: even if something flips the flag back, no-op update.
    if hasattr(env, "_update_terrain_curriculum"):
        env._update_terrain_curriculum = lambda env_ids: None  # type: ignore[method-assign]


def build_eval_env(task: str, num_envs: int, geometry_seed: int, episode_length_s: float, cpu: bool):
    import genesis as gs

    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if cpu else gs.gpu, logging_level="warning")

    registered_env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    env_cfg = copy.deepcopy(registered_env_cfg)
    env_cfg.env.num_envs = num_envs
    apply_frontier_diag_env_overrides(
        env_cfg, geometry_seed=geometry_seed, episode_length_s=episode_length_s
    )

    reg_args = SimpleNamespace(
        task=task,
        headless=True,
        cpu=cpu,
        num_envs=num_envs,
        max_iterations=None,
        resume=False,
        sync_wandb=False,
        export_onnx=False,
        debug=False,
        load_run=None,
        ckpt=-1,
        use_joystick=False,
        joystick_type="xbox",
        follow_robot=False,
        viewer="native",
        viser_port=8080,
        motion_file=None,
        motion_out_dir=None,
        num_student=None,
        seed=int(geometry_seed),
    )
    env, env_cfg = task_registry.make_env(name=task, args=reg_args, env_cfg=env_cfg)
    pin_others_to_nominal(env, "control_delay")
    disable_runtime_terrain_curriculum(env)
    if bool(getattr(env.cfg.terrain, "curriculum", True)):
        raise RuntimeError(
            "diagnostic env must disable runtime terrain.curriculum after mesh build"
        )
    if bool(getattr(env.cfg.env, "auto_reset", True)):
        raise RuntimeError("diagnostic env must run with env.auto_reset=False")
    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task, args=reg_args, train_cfg=train_cfg, log_root=None
    )
    return env, ppo_runner


def _command_at_step(row: DiagnosticRow, step: int) -> tuple[float, float, float]:
    segs = row.segment_commands
    if not segs:
        return float(row.command_vx), float(row.command_vy), float(row.command_yaw)
    active = segs[0]
    for seg in segs:
        if int(step) >= int(seg.start_step):
            active = seg
    return float(active.vx), float(active.vy), float(active.yaw)


def _pin_commands(env, ids: torch.Tensor, rows: Sequence[DiagnosticRow], *, step: int) -> None:
    for i, row in enumerate(rows):
        vx, vy, yaw = _command_at_step(row, step)
        env.commands[ids[i], 0] = vx
        env.commands[ids[i], 1] = vy
        env.commands[ids[i], 2] = yaw
        env.commands[ids[i], 3] = 0.0


def _requested_tile_tensors(env, rows: Sequence[DiagnosticRow]):
    n = len(rows)
    ids = torch.arange(n, device=env.device)
    columns = torch.as_tensor(
        [row.physical_column for row in rows], dtype=torch.long, device=env.device
    )
    levels = torch.as_tensor(
        [row.terrain_level for row in rows], dtype=torch.long, device=env.device
    )
    origins = env.simulator._terrain_origins
    expected_origins = origins[levels, columns]
    return ids, columns, levels, expected_origins


def _apply_tile_assignment(env, ids, columns, levels, expected_origins) -> None:
    env.simulator.terrain_types[ids] = columns
    env.simulator.terrain_levels[ids] = levels
    env.simulator.env_origins[ids] = expected_origins


def assert_runtime_tiles(env, ids, columns, levels, expected_origins) -> None:
    """Fail closed if simulator tiles drifted from the bank request."""
    n = len(ids)
    rt_levels = env.simulator.terrain_levels[ids]
    rt_types = env.simulator.terrain_types[ids]
    rt_origins = env.simulator.env_origins[ids]
    if not torch.equal(rt_levels, levels):
        raise RuntimeError(
            f"runtime terrain_levels drifted from request: "
            f"got={rt_levels.tolist()} want={levels.tolist()}"
        )
    if not torch.equal(rt_types, columns):
        raise RuntimeError(
            f"runtime terrain_types drifted from request: "
            f"got={rt_types.tolist()} want={columns.tolist()}"
        )
    if not torch.allclose(rt_origins, expected_origins, atol=1e-5, rtol=0.0):
        raise RuntimeError("runtime env_origins drifted from requested tile origins")
    # Also ensure no env was silently moved to another row/col beyond the batch.
    if int(rt_levels.max().item()) != int(levels.max().item()) or int(rt_levels.min().item()) != int(
        levels.min().item()
    ):
        pass  # covered by equal above
    _ = n


def assign_rows(env, rows: Sequence[DiagnosticRow]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reset, force bank tiles, re-pin commands, recompute observations.

    Order is intentional:
    1. Pre-seed tile ids (helps root reset use correct origins on first reset).
    2. ``env.reset()`` (may run zero-action step with random commands).
    3. Re-apply tile assignment so native curriculum / drift cannot stick.
    4. Assert runtime tile identity.
    5. Pin bank commands and recompute observations before policy call.
    """
    n = len(rows)
    if n > env.num_envs:
        raise ValueError(f"{n} rows do not fit in {env.num_envs} envs")
    ids, columns, levels, expected_origins = _requested_tile_tensors(env, rows)
    _apply_tile_assignment(env, ids, columns, levels, expected_origins)
    env.reset()
    # Reset may have advanced native curriculum or resampled commands. Re-pin tiles.
    _apply_tile_assignment(env, ids, columns, levels, expected_origins)
    # Re-teleport roots onto the forced origins (reset_root already ran once).
    if hasattr(env, "_reset_root_states"):
        env._reset_root_states(ids)
        if hasattr(env.simulator, "reset_idx"):
            env.simulator.reset_idx(ids)
    assert_runtime_tiles(env, ids, columns, levels, expected_origins)
    _pin_commands(env, ids, rows, step=0)
    env.episode_length_buf[ids] = 0
    env.fail_buf[ids] = 0
    env.reset_buf[ids] = 0
    env.time_out_buf[ids] = False
    # Observation must reflect pinned commands (get_observations alone is stale).
    env.compute_observations()
    return ids, columns, levels, expected_origins


def scene_geometry_hash(env) -> str:
    terrain = getattr(env.simulator, "_terrain", None)
    if terrain is None or getattr(terrain, "height_field_raw", None) is None:
        raise RuntimeError("no heightfield available for scene geometry hash")
    raw = np.ascontiguousarray(terrain.height_field_raw)
    header = {
        "kind": "full_scene_heightfield",
        "shape": list(raw.shape),
        "horizontal_scale": float(terrain.cfg.horizontal_scale),
        "vertical_scale": float(terrain.cfg.vertical_scale),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\x00")
    digest.update(raw.tobytes(order="C"))
    return digest.hexdigest()


def tile_geometry_hash(env, level: int, column: int) -> str:
    terrain = getattr(env.simulator, "_terrain", None)
    if terrain is None:
        raise RuntimeError("no terrain for tile hash")
    # Reuse the same tile extraction + scale-aware hash as V5 taxonomy.
    return taxonomy_tile_geometry_hash(terrain, int(level), int(column))


def command_obs_slice_matches(
    env, ids: torch.Tensor, rows: Sequence[DiagnosticRow], *, atol: float = 1e-4
) -> bool:
    """Check obs command channels match pinned commands (scaled).

    Default V6 MLP obs layout places commands at indices [9:12] after
    lin_vel(3)+gravity(3)+ang_vel(3).  Noise may be present; we compare the
    *expected scaled command* against the noise-free reconstruction from
    ``env.commands`` after ``compute_observations`` when noise is disabled, or
    against the scaled command tensor itself when we can read commands_scale.
    """
    # Prefer direct tensor check: after compute_observations the obs command
    # channels equal commands * commands_scale (+ optional noise).  With noise
    # off this is exact; with noise on we only verify the env.commands tensor
    # is pinned (obs noise is a separate DR choice).
    for i, row in enumerate(rows):
        vx, vy, yaw = _command_at_step(row, 0)
        cmd = env.commands[ids[i], :3]
        if abs(float(cmd[0]) - vx) > atol or abs(float(cmd[1]) - vy) > atol or abs(float(cmd[2]) - yaw) > atol:
            return False
    # If noise is disabled, also check obs slice.
    add_noise = bool(getattr(getattr(env.cfg, "noise", None), "add_noise", False))
    if not add_noise and hasattr(env, "obs_buf") and hasattr(env, "commands_scale"):
        obs = env.obs_buf[ids]
        # indices 9:12 in default legged_robot compute_observations
        if obs.shape[-1] >= 12:
            expected = env.commands[ids, :3] * env.commands_scale
            if not torch.allclose(obs[:, 9:12], expected, atol=1e-4, rtol=0.0):
                return False
    return True


def rollout_batch(
    env,
    policy,
    rows: Sequence[DiagnosticRow],
    *,
    linear_threshold: float,
    yaw_threshold: float,
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one first-episode diagnostic batch until fall or real timeout."""
    n = len(rows)
    ids, columns, levels, expected_origins = assign_rows(env, rows)
    if not command_obs_slice_matches(env, ids, rows):
        raise RuntimeError("pinned commands not reflected before first policy action")

    max_ep_len = int(env.max_episode_length)
    control_dt = float(env.dt)
    episode_length_s = float(env.max_episode_length_s)
    # Timeout comparison in legged_robot: episode_length_buf > max_episode_length
    # so a surviving episode needs max_episode_length+1 increments to fire.
    guard_steps = max_ep_len + 2
    timeout_comparison = "episode_length_buf > max_episode_length"

    lin_sum = torch.zeros(n, device=env.device)
    yaw_sum = torch.zeros(n, device=env.device)
    abs_vx_sum = torch.zeros(n, device=env.device)
    abs_vy_sum = torch.zeros(n, device=env.device)
    abs_yaw_sum = torch.zeros(n, device=env.device)
    cmd_xy_sum = torch.zeros(n, device=env.device)
    alive_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    ever_fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    observed_timeout = torch.zeros(n, dtype=torch.bool, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    first_fall_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    observed_timeout_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    ret_sum = torch.zeros(n, device=env.device)
    on_tile_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    first_exit = torch.full((n,), -1, dtype=torch.long, device=env.device)
    max_abs_dx = torch.zeros(n, device=env.device)
    max_abs_dy = torch.zeros(n, device=env.device)
    half_l = 0.5 * float(env.cfg.terrain.terrain_length)
    half_w = 0.5 * float(env.cfg.terrain.terrain_width)

    acc = MetricAccumulator(n, env.device, v_scale=2.0, v_scale_yaw=1.0)
    obs = env.get_observations()
    scene_hash = scene_geometry_hash(env)
    tile_hash_cache: dict[tuple[int, int], str] = {}

    # Snapshot runtime tile identity at episode start (post-assert).
    runtime_levels0 = env.simulator.terrain_levels[ids].detach().cpu().tolist()
    runtime_cols0 = env.simulator.terrain_types[ids].detach().cpu().tolist()
    runtime_origins0 = env.simulator.env_origins[ids].detach().cpu().numpy()

    for step in range(guard_steps):
        if bool(finished.all()):
            break
        # Keep bank tiles forced even if auto_reset runs mid-batch on finished envs.
        _apply_tile_assignment(env, ids, columns, levels, expected_origins)
        _pin_commands(env, ids, rows, step=step)
        # Active envs only need fresh obs for policy; recompute for all is fine.
        if step > 0:
            # After previous step, obs already computed inside env.step; still
            # re-pin commands that auto-reset may have resampled, then refresh.
            env.compute_observations()
            obs = env.get_observations()

        actions = policy(obs.detach())
        # Zero actions for already-finished *bank* envs only.  The policy tensor
        # is shaped (num_envs, act); bank rows occupy ids[:n], and the last
        # batch may have n < num_envs (mask size must match the bank slice).
        if finished.any():
            actions = actions.clone()
            actions[ids] = torch.where(
                finished.unsqueeze(-1),
                torch.zeros_like(actions[ids]),
                actions[ids],
            )
        obs, _, rew, dones, _ = env.step(actions.detach())
        # Re-force tiles/commands after step (auto_reset may have run).
        _apply_tile_assignment(env, ids, columns, levels, expected_origins)
        _pin_commands(env, ids, rows, step=step)

        active = ~finished
        if not torch.any(active):
            continue

        cmd = env.commands[:n]
        v = env.simulator.base_lin_vel[:n]
        w = env.simulator.base_ang_vel[:n]
        lin_cmd = cmd[:, :2]
        lin_err = torch.linalg.vector_norm(lin_cmd - v[:, :2], dim=1)
        lin_scale = torch.clamp(torch.linalg.vector_norm(lin_cmd, dim=1), min=0.2)
        yaw_err = torch.abs(cmd[:, 2] - w[:, 2]) / torch.clamp(torch.abs(cmd[:, 2]), min=0.2)

        lin_sum[active] += (lin_err / lin_scale)[active]
        yaw_sum[active] += yaw_err[active]
        abs_vx_sum[active] += torch.abs(cmd[:, 0] - v[:, 0])[active]
        abs_vy_sum[active] += torch.abs(cmd[:, 1] - v[:, 1])[active]
        abs_yaw_sum[active] += torch.abs(cmd[:, 2] - w[:, 2])[active]
        cmd_xy_sum[active] += torch.linalg.vector_norm(lin_cmd, dim=1)[active]
        ret_sum[active] += rew[:n][active]
        alive_steps[active] += 1

        displacement = torch.abs(
            env.simulator.base_pos[:n, :2] - env.simulator.env_origins[:n, :2]
        )
        dx, dy = displacement[:, 0], displacement[:, 1]
        on_tile = (dx < half_l) & (dy < half_w)
        on_tile_steps[active] += on_tile[active].long()
        max_abs_dx = torch.maximum(max_abs_dx, dx)
        max_abs_dy = torch.maximum(max_abs_dy, dy)
        newly_exit = active & (~on_tile) & (first_exit < 0)
        first_exit[newly_exit] = step

        done = dones[:n].bool()
        timeout = env.time_out_buf[:n].bool()
        # Real timeout event observed on the first episode.
        to_now = active & timeout
        observed_timeout |= to_now
        observed_timeout_step[to_now & (observed_timeout_step < 0)] = step
        # Fall = done without timeout (training terminal_reason split).
        fell_now = active & done & (~timeout)
        ever_fell |= fell_now
        first_fall_step[fell_now & (first_fall_step < 0)] = step
        finished |= observed_timeout | ever_fell

        lin_xy = lin_err
        ang = torch.abs(cmd[:, 2] - w[:, 2])
        lin_x = torch.abs(cmd[:, 0] - v[:, 0])
        acc.update(
            rew[:n],
            torch.where(active, done, torch.zeros_like(done)),
            torch.where(active, timeout, torch.zeros_like(timeout)),
            lin_xy,
            ang,
            lin_x_err=lin_x,
        )

    # Still unfinished after guard: treat as measurement failure (not a timeout).
    unfinished = ~finished
    if unfinished.any():
        # Do not invent timeouts; leave observed_timeout False.
        pass

    # Final tile assertion on the batch (should still match bank plan).
    assert_runtime_tiles(env, ids, columns, levels, expected_origins)

    metrics = acc.compute()
    lengths = torch.clamp(alive_steps, min=1).float()
    mean_lin = (lin_sum / lengths).detach().cpu().numpy()
    mean_yaw = (yaw_sum / lengths).detach().cpu().numpy()
    timed_out_np = observed_timeout.detach().cpu().numpy()
    fell_np = ever_fell.detach().cpu().numpy()
    survived_horizon = (~ever_fell).detach().cpu().numpy()

    records = []
    for i, row in enumerate(rows):
        req_col = int(row.physical_column)
        req_lvl = int(row.terrain_level)
        rt_col = int(runtime_cols0[i])
        rt_lvl = int(runtime_levels0[i])
        key = (rt_lvl, rt_col)
        if key not in tile_hash_cache:
            tile_hash_cache[key] = tile_geometry_hash(env, rt_lvl, rt_col)
        success = frontier_success(
            timed_out=bool(timed_out_np[i]),
            mean_linear_error=float(mean_lin[i]),
            mean_yaw_error=float(mean_yaw[i]),
            linear_threshold=linear_threshold,
            yaw_threshold=yaw_threshold,
        )
        fe = int(first_exit[i].item())
        ots = int(observed_timeout_step[i].item())
        ffs = int(first_fall_step[i].item())
        if bool(timed_out_np[i]):
            terminal_reason = "timeout"
        elif bool(fell_np[i]):
            terminal_reason = "fall"
        else:
            terminal_reason = "guard_exhausted"
        records.append(
            {
                **row.to_dict(),
                **provenance,
                "requested_terrain_column": req_col,
                "requested_terrain_level": req_lvl,
                "runtime_terrain_column": rt_col,
                "runtime_terrain_level": rt_lvl,
                "runtime_origin": [float(x) for x in runtime_origins0[i].tolist()],
                "scene_geometry_hash": scene_hash,
                "runtime_tile_geometry_hash": tile_hash_cache[key],
                # Deprecated alias kept only if equal to scene hash for old readers.
                "geometry_hash": scene_hash,
                "episode_length": int(alive_steps[i].item()),
                "configured_episode_length_s": episode_length_s,
                "control_dt": control_dt,
                "max_episode_length": max_ep_len,
                "timeout_comparison": timeout_comparison,
                "observed_timeout_event": bool(timed_out_np[i]),
                "observed_timeout_step": None if ots < 0 else ots,
                "survived_measurement_horizon": bool(survived_horizon[i]),
                "timed_out": bool(timed_out_np[i]),
                "terminated": bool(fell_np[i]),
                "terminal_reason": terminal_reason,
                "episodic_return": float(ret_sum[i].item()),
                "mean_linear_error": float(mean_lin[i]),
                "mean_yaw_error": float(mean_yaw[i]),
                "linear_pass_at_0_35": bool(mean_lin[i] <= 0.35),
                "yaw_pass_at_0_40": bool(mean_yaw[i] <= 0.40),
                "frontier_success_at_original_thresholds": bool(success),
                "on_tile_fraction": float(
                    on_tile_steps[i].item() / max(int(alive_steps[i].item()), 1)
                ),
                "first_exit_step": None if fe < 0 else fe,
                "max_abs_dx": float(max_abs_dx[i].item()),
                "max_abs_dy": float(max_abs_dy[i].item()),
                "mean_abs_vx_error": float((abs_vx_sum[i] / lengths[i]).item()),
                "mean_abs_vy_error": float((abs_vy_sum[i] / lengths[i]).item()),
                "mean_abs_yaw_error": float((abs_yaw_sum[i] / lengths[i]).item()),
                "mean_command_xy_norm": float((cmd_xy_sum[i] / lengths[i]).item()),
                "spnte_lin": float(metrics["spnte_lin"][i]),
                "spnte_yaw": float(metrics["spnte_yaw"][i]),
                "first_fall_step": None if ffs < 0 else ffs,
            }
        )
    return records


def run_geometry(args) -> None:
    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = artifact_dir(run_dir, args.iteration, config)
    ckpt_path = run_dir / f"model_{args.iteration}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt_sha = sha256_file(str(ckpt_path))

    geometry_seed = int(args.geometry_seed)
    full_bank = build_bank(config)
    rows = build_bank(config, geometry_seed=geometry_seed)
    # Optional filters for smoke / focused slices (bank fingerprint still full).
    if args.columns:
        want_cols = {int(x) for x in str(args.columns).split(",") if str(x).strip() != ""}
        rows = [r for r in rows if int(r.physical_column) in want_cols]
        if not rows:
            raise ValueError(f"--columns {args.columns!r} matched zero bank rows")
    if args.regimes:
        want_reg = {x.strip() for x in str(args.regimes).split(",") if x.strip()}
        rows = [r for r in rows if r.regime in want_reg]
        if not rows:
            raise ValueError(f"--regimes {args.regimes!r} matched zero bank rows")
    if args.max_episodes is not None:
        if int(args.max_episodes) <= 0:
            raise ValueError("--max_episodes must be positive")
        rows = rows[: int(args.max_episodes)]
    # Smoke artifacts go under a separate subdir so they cannot clobber a full run.
    if args.smoke or args.max_episodes is not None or args.columns or args.regimes:
        out_dir = out_dir / "smoke"
    meta = write_bank_artifacts(
        out_dir, config, full_bank, checkpoint_sha256=ckpt_sha
    )
    full_bank_fp = meta["bank_fingerprint"]
    cfg_fp = meta["config_fingerprint"]

    if args.num_shards > 1:
        shard_rows = rows[args.shard_index :: args.num_shards]
    else:
        shard_rows = rows
    if not shard_rows:
        raise ValueError("empty shard")

    if int(config["rollout"]["warmup_steps"]) != 0:
        raise ValueError("frontier diagnostic forbids warmup_steps != 0")
    episode_length_s = float(config["rollout"].get("episode_length_s", 20.0))

    batch_size = int(args.num_envs) if args.num_envs else min(256, len(shard_rows))
    batch_size = max(1, min(batch_size, len(shard_rows)))

    env, ppo_runner = build_eval_env(
        args.task, batch_size, geometry_seed, episode_length_s, args.cpu
    )
    ppo_runner.load(str(ckpt_path), load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=env.device)

    lin_thr = float(config["frontier_success"]["linear_error_threshold"])
    yaw_thr = float(config["frontier_success"]["yaw_error_threshold"])
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "diagnostic_only",
        "eligible_for_checkpoint_selection": False,
        "checkpoint_path": str(ckpt_path.resolve()),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_iteration": int(args.iteration),
        "training_seed": int(config["geometry"]["training_geometry_seed"]),
        "bank_fingerprint": full_bank_fp,
        "config_fingerprint": cfg_fp,
        "geometry_seed": geometry_seed,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "git_commit": _git_commit(),
        "working_tree_dirty": _working_tree_dirty(),
        "task": args.task,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": ckpt_path.name,
        "warmup_steps": 0,
        "training_seed_matched": geometry_seed
        == int(config["geometry"]["training_geometry_seed"]),
        "exact_training_geometry_reproduced": None,
        "terrain_curriculum_enabled": False,
        "auto_reset": False,
        "protocol_notes": (
            "timed_out uses observed time_out_buf; "
            "runtime terrain.curriculum=False after curriculum mesh build; "
            "auto_reset=False so terminal metrics are pre-reset; "
            "obs recomputed after command pin"
        ),
    }

    all_records: list[dict[str, Any]] = []
    for start in range(0, len(shard_rows), batch_size):
        batch = shard_rows[start : start + batch_size]
        recs = rollout_batch(
            env,
            policy,
            batch,
            linear_threshold=lin_thr,
            yaw_threshold=yaw_thr,
            provenance=provenance,
        )
        all_records.extend(recs)
        print(
            f"[frontier_diag] geometry_seed={geometry_seed} "
            f"shard={args.shard_index}/{args.num_shards} "
            f"batch {start}:{start+len(batch)} / {len(shard_rows)} done",
            flush=True,
        )

    if args.num_shards > 1:
        out_path = (
            out_dir
            / f"geometry_seed_{geometry_seed}.shard{args.shard_index}of{args.num_shards}.episodes.ndjson"
        )
    else:
        out_path = out_dir / f"geometry_seed_{geometry_seed}.episodes.ndjson"
    write_ndjson(out_path, all_records)
    print(f"[frontier_diag] wrote {len(all_records)} episodes -> {out_path}")


def merge_shards_for_geometry(args) -> None:
    from legged_gym.scripts.eval.frontier_diagnostic import (
        merge_geometry_episode_files,
        policy_training_seed_from_config,
    )

    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = artifact_dir(run_dir, args.iteration, config)
    geometry_seed = int(args.geometry_seed)
    paths = [
        out_dir
        / f"geometry_seed_{geometry_seed}.shard{i}of{args.num_shards}.episodes.ndjson"
        for i in range(args.num_shards)
    ]
    ckpt_sha = sha256_file(str(run_dir / f"model_{args.iteration}.pt"))
    full_bank = build_bank(config)
    records = merge_geometry_episode_files(
        paths,
        expected_bank_fp=bank_fingerprint(full_bank),
        expected_config_fp=config_fingerprint(config),
        expected_checkpoint_sha=ckpt_sha,
        expected_count=expected_episode_count(config, geometry_seed=geometry_seed),
        default_training_seed=policy_training_seed_from_config(config),
        bank_rows=build_bank(config, geometry_seed=geometry_seed),
    )
    out_path = out_dir / f"geometry_seed_{geometry_seed}.episodes.ndjson"
    write_ndjson(out_path, records)
    print(f"[frontier_diag] merged shards -> {out_path} ({len(records)})")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval/v6_frontier_diagnostic.yaml")
    parser.add_argument("--task", default="go2_v6_frontier")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--geometry_seed", type=int, required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Evaluate only the first N bank rows after filters (smoke).",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated physical columns to keep (e.g. 0,2,3,8).",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default=None,
        help="Comma-separated regimes to keep (e.g. A_baseline).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force smoke artifact subdir even without filters.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--merge_shards", action="store_true")
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("invalid shard settings")
    if args.merge_shards:
        merge_shards_for_geometry(args)
    else:
        run_geometry(args)


if __name__ == "__main__":
    main()
