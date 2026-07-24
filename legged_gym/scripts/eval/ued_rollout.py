"""Genesis rollout bridge for the frozen V5 UED validation bank.

``legged_gym.scripts.eval.ued_validation`` and ``select_checkpoint`` (Topic 04)
are deliberately simulator-neutral: they consume a caller-supplied stream of
per-replica measurements and never touch Genesis/Torch themselves (see their
module docstrings and ``lpacr/subplans/04_validation_and_checkpoint.md``).
This module is that caller. It reuses the SAME eval-harness scaffolding as
``sweep.py`` (``MetricAccumulator``, ``pin_others_to_nominal``) rather than
duplicating rollout machinery, and builds the SAME deterministic UED taxonomy
terrain grid used at training time (``go2_v5_config.V5_TERRAIN_SEED``)
regardless of which arm's checkpoint is being scored: the bank is shared
across all four V5 arms, including ``handcrafted_v4`` (solun_plani.md §10).

The frozen bank is 84 cells x ``replicas_per_cell`` deterministic replicas
(currently 12/cell = 1,008 measurements total; see
``configs/eval/v5_ued.yaml``, ``FROZEN_REPLICAS_PER_CELL`` in
``ued_validation.py``). Offline scoring is invoked by
``scripts/run_v5_fazB.sh`` after training, on the SCHEDULED iterations only
(``select_checkpoint.selection_schedule_from_config``: from
``min_iteration``, every ``iteration_stride``, plus the final periodic
checkpoint -- not every single ``model_<iteration>.pt``), then
``select_checkpoint`` materialises ``best_spnte.pt``.

One Genesis scene lives for the life of one process (no ``env.destroy()`` /
rebuild support anywhere in this repo -- see ``tests/v3_runtime_smoke.py`` /
``tests/v4_runtime_smoke.py``'s ``hasattr(env, "destroy")`` guards), so sharding
the bank across a GPU that cannot hold all replicas as parallel envs is done
by re-invoking this script as a separate process per shard
(``--num_shards``/``--shard_index``), each writing a partial measurement file;
``--merge`` then combines the shards into the final artifact.  With one shard
(the default) everything happens in a single process/pass.

Usage (single shard, one iteration):
    python -m legged_gym.scripts.eval.ued_rollout \\
        --task go2_v5_lpacrl --run_dir logs/go2_v5_ued/<run> --iteration 200

Usage (2 shards on a smaller GPU):
    python -m legged_gym.scripts.eval.ued_rollout --task go2_v5_lpacrl \\
        --run_dir <run> --iteration 200 --num_shards 2 --shard_index 0
    python -m legged_gym.scripts.eval.ued_rollout --task go2_v5_lpacrl \\
        --run_dir <run> --iteration 200 --num_shards 2 --shard_index 1
    python -m legged_gym.scripts.eval.ued_rollout --task go2_v5_lpacrl \\
        --run_dir <run> --iteration 200 --num_shards 2 --merge
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR  # noqa: F401  (side effect: exposes gs via legged_gym.__init__)
import legged_gym.envs  # noqa: F401  (registers go2_v5_* tasks)
from legged_gym.utils import task_registry

from legged_gym.scripts.eval.ckpt_utils import sha256_file
from legged_gym.scripts.eval.dr_axes import pin_others_to_nominal
from legged_gym.scripts.eval.metrics import MetricAccumulator
from legged_gym.scripts.eval.ued_validation import (
    BANK_KINDS,
    BankRow,
    build_bank,
    load_config,
    make_shard_payload,
    make_validation_artifact,
    merge_shard_payloads,
    support_scale,
    yaw_support_scale,
)
from legged_gym.utils.terrain import UED_TERRAIN_TYPE_INDEX, taxonomy_tile_geometry_hash
from legged_gym.utils.ued.task_space import TaskSpace


def _artifact_dir(config: dict, bank_kind: str) -> str:
    """Route validation artifacts to the selection dir; held-out to its own dir.

    The held-out final measurements MUST live outside ``selection.artifact_dir``
    so ``select_checkpoint.load_candidates`` can never discover -- let alone
    select on -- them (§7 leakage guard)."""
    if bank_kind == "validation":
        return str(config["selection"]["artifact_dir"])
    return "heldout_final"


def _partial_path(run_dir: Path, iteration: int, shard_index: int, num_shards: int,
                  config: dict, bank_kind: str) -> Path:
    return (run_dir / _artifact_dir(config, bank_kind)
            / f"model_{iteration}.{bank_kind}.shard{shard_index}of{num_shards}.json")


def _final_path(run_dir: Path, iteration: int, config: dict, bank_kind: str) -> Path:
    selection = config["selection"]
    return (run_dir / _artifact_dir(config, bank_kind)
            / str(selection["artifact_pattern"]).format(iteration=iteration))


def _git_commit() -> str:
    """Best-effort short provenance of the working tree; 'unknown' if unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", LEGGED_GYM_ROOT_DIR, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


# Far past any frozen-bank warmup+measure window (see configs/eval/v5_ued.yaml:
# 100 + 1000 steps at dt=0.02 s ≈ 22 s). Same sentinel as campaign.py / transient.py
# / play.py. Training default is 10 s (~500 control steps) and would inject one
# random 3-axis command mid-window while SPNTE is scored against the pinned bank
# command.
_BANK_EVAL_RESAMPLING_TIME_S = 1e6


def apply_bank_eval_env_overrides(env_cfg) -> None:
    """Pin a task env_cfg for frozen-bank scoring (no teacher, fixed command).

    Pure cfg mutation so unit tests can pin the contract without building a
    Genesis scene.  ``build_eval_env`` applies this before ``make_env``.
    """
    from legged_gym.envs.go2.go2_v5_config import V5_TERRAIN_SEED
    from legged_gym.utils.terrain import TAXONOMY_NUM_LEVELS, TAXONOMY_NUM_TYPES

    env_cfg.env.ued_enabled = False  # bank driver owns assignment directly
    env_cfg.terrain.mesh_type = "heightfield"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.ued_training_grid = True
    env_cfg.terrain.ued_training_seed = V5_TERRAIN_SEED
    env_cfg.terrain.num_rows = TAXONOMY_NUM_LEVELS
    env_cfg.terrain.num_cols = TAXONOMY_NUM_TYPES

    # "pinned_nominal" (config rollout.domain_randomization): fixed command,
    # no push/mass/com/friction randomization -- isolate policy behaviour from
    # DR noise so every method is scored under identical physics (mirrors
    # sweep.py's override_cfg_for_eval).
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.legacy_performance_command_curriculum_enabled = False  # renamed from command_curriculum_enabled
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.per_env_standstill = False
    # Disable in-episode random resampling for the whole window: the bank pins
    # (vx, vy, yaw) externally.  Without this, legged_robot._post_physics_step_callback
    # resamples when episode_length % int(resampling_time/dt) == 0 (~step 500 at
    # the training default of 10 s), so the policy briefly sees a random 3-axis
    # command while SPNTE is written against the pinned bank setpoint.
    env_cfg.commands.resampling_time = _BANK_EVAL_RESAMPLING_TIME_S


def build_eval_env(task: str, num_envs: int, seed: int, cpu: bool):
    """Build a V5 env at the frozen UED taxonomy grid, no teacher installed.

    The validation bank is one shared fixed grid across all four arms (§10);
    ``env_cfg.env.ued_enabled`` stays False here regardless of ``task`` -- this
    driver assigns terrain/command directly (see ``assign_rows_to_env``)
    instead of going through an ``EpisodeCurriculum``/``GenesisUEDAdapter``.
    """
    import genesis as gs  # local import: only needed when SIMULATOR == "genesis"

    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if cpu else gs.gpu, logging_level="warning")

    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    env_cfg.env.num_envs = num_envs
    env_cfg.seed = seed
    apply_bank_eval_env_overrides(env_cfg)

    reg_args = SimpleNamespace(
        task=task, headless=True, cpu=cpu, num_envs=num_envs, max_iterations=None,
        resume=False, sync_wandb=False, export_onnx=False, debug=False,
        load_run=None, ckpt=-1, use_joystick=False, joystick_type="xbox",
        follow_robot=False, viewer="native", viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )
    env, env_cfg = task_registry.make_env(name=task, args=reg_args, env_cfg=env_cfg)
    pin_others_to_nominal(env, "control_delay")  # no privileged axis -> pins everything

    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task, args=reg_args, train_cfg=train_cfg, log_root=None,
    )
    return env, ppo_runner


def _command_tensors(rows: Sequence[BankRow], device):
    """Frozen (vx, vy, omega_z) command columns for one shard, in row order."""
    vx = torch.as_tensor([row.command_vx for row in rows], dtype=torch.float, device=device)
    vy = torch.as_tensor([row.command_vy for row in rows], dtype=torch.float, device=device)
    yaw = torch.as_tensor([row.command_yaw for row in rows], dtype=torch.float, device=device)
    return vx, vy, yaw


def _set_commands(env, ids, rows: Sequence[BankRow]) -> None:
    """Pin the frozen 3-axis command (vx, vy, omega_z) onto the given envs.

    heading_command is False for the bank rollout, so commands[:, 2] is the
    directly-tracked yaw-rate setpoint; commands[:, 3] (heading) is unused.
    """
    vx, vy, yaw = _command_tensors(rows, env.device)
    env.commands[ids, 0] = vx
    env.commands[ids, 1] = vy
    env.commands[ids, 2] = yaw
    env.commands[ids, 3] = 0.0


def assign_rows_to_env(env, rows: Sequence[BankRow]) -> None:
    """Force each env onto its bank replica's terrain tile and fixed command."""
    n = len(rows)
    if n > env.num_envs:
        raise ValueError(f"{n} bank rows do not fit in {env.num_envs} envs")
    ids = torch.arange(n, device=env.device)
    type_idx = torch.as_tensor(
        [TaskSpace.TERRAIN_TYPE_NAMES.index(row.terrain_type) for row in rows],
        dtype=torch.long, device=env.device,
    )
    level_idx = torch.as_tensor([row.terrain_level for row in rows], dtype=torch.long, device=env.device)
    origins = env.simulator._terrain_origins
    env.simulator.terrain_types[ids] = type_idx
    env.simulator.terrain_levels[ids] = level_idx
    env.simulator.env_origins[ids] = origins[level_idx, type_idx]
    env.reset()
    _set_commands(env, ids, rows)


def _runtime_geometry_hash(env, terrain_type: str, terrain_level: int, cache: dict) -> str:
    """Hash the ACTUAL built heightfield tile the replica is standing on.

    This is the reviewer item-7 fix: the reported hash is recomputed from the
    live Genesis terrain (``env.simulator._terrain``), NOT copied from the bank
    row, so ``validate_measurements`` compares an independent runtime observation
    against the frozen pin.  Cached per (type, level) since every tile is shared."""
    key = (terrain_type, int(terrain_level))
    if key not in cache:
        terrain = getattr(env.simulator, "_terrain", None)
        if terrain is None or getattr(terrain, "height_field_raw", None) is None:
            raise RuntimeError(
                "cannot verify geometry: simulator exposes no built heightfield terrain"
            )
        cache[key] = taxonomy_tile_geometry_hash(
            terrain, int(terrain_level), UED_TERRAIN_TYPE_INDEX[terrain_type],
        )
    return cache[key]


def rollout_measurements(
    env, policy, rows: Sequence[BankRow], *, warmup_steps: int, steps: int, v_scale: float,
    v_scale_yaw: float,
) -> list[dict[str, Any]]:
    """Run one fixed-command/fixed-terrain rollout and return per-row measurements.

    The command is the frozen 3-axis (vx, vy, omega_z) bank draw, re-pinned
    every step so auto-reset cannot leak a resampled command back in.  spnte_lin
    (forward, the selection headline) plus diagnostic yaw-tracking (spnte_yaw,
    tracking_ang_err) are reported per replica.  ``geometry_hash`` is recomputed
    from the live terrain per replica (see ``_runtime_geometry_hash``).
    """
    n = len(rows)
    ids = torch.arange(n, device=env.device)
    command_vx, command_vy, command_yaw = _command_tensors(rows, env.device)

    def _pin_commands():
        env.commands[ids, 0] = command_vx
        env.commands[ids, 1] = command_vy
        env.commands[ids, 2] = command_yaw
        env.commands[ids, 3] = 0.0

    obs = env.get_observations()
    for _ in range(warmup_steps):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        _pin_commands()
    env.episode_length_buf[ids] = 0

    acc = MetricAccumulator(n, env.device, v_scale=v_scale, v_scale_yaw=v_scale_yaw)
    for _ in range(steps):
        actions = policy(obs.detach())
        obs, _, rew, dones, _ = env.step(actions.detach())
        _pin_commands()
        cmd = env.commands[:n]
        v = env.simulator.base_lin_vel[:n]
        w = env.simulator.base_ang_vel[:n]
        lin_x_err = torch.abs(cmd[:, 0] - v[:, 0])
        lin_err = torch.norm(cmd[:, :2] - v[:, :2], dim=1)
        ang_err = torch.abs(cmd[:, 2] - w[:, 2])
        acc.update(rew[:n], dones[:n], env.time_out_buf[:n], lin_err, ang_err, lin_x_err=lin_x_err)

    metrics = acc.compute()
    geometry_cache: dict = {}
    measurements = []
    for i, row in enumerate(rows):
        measurements.append({
            "cell_id": row.cell_id,
            "replica_id": row.replica_id,
            "spnte_lin": float(metrics["spnte_lin"][i]),
            "spnte_yaw": float(metrics["spnte_yaw"][i]),
            "tracking_ang_err": float(metrics["tracking_ang_err"][i]),
            "fall_rate": float(metrics["ever_fell"][i]),
            "survival_steps": float(metrics["first_fall_step"][i]),
            "command_vx": row.command_vx,
            "command_vy": row.command_vy,
            "command_yaw": row.command_yaw,
            "geometry_hash": _runtime_geometry_hash(env, row.terrain_type, row.terrain_level, geometry_cache),
        })
    return measurements


def _load_checkpoint_into(ppo_runner, checkpoint_path: str) -> None:
    ppo_runner.load(checkpoint_path, load_optimizer=False)


def _rollout_provenance(args, config: dict, checkpoint_path: Path) -> dict[str, Any]:
    """Auditable rollout context stored in the artifact (reviewer item 7)."""
    return {
        "bank_kind": args.bank,
        "seed": int(args.seed),
        "simulator": SIMULATOR,
        "task": str(args.task),
        "run_dir": str(Path(args.run_dir).resolve()),
        "checkpoint": checkpoint_path.name,
        "git_commit": _git_commit(),
        "rollout_steps": int(args.rollout_steps or config["rollout"]["steps"]),
        "warmup_steps": int(args.warmup_steps or config["rollout"]["warmup_steps"]),
        "domain_randomization": str(config["rollout"].get("domain_randomization", "pinned_nominal")),
    }


def run_shard(args) -> None:
    config = load_config(args.config)
    bank = build_bank(config, seed_key=BANK_KINDS[args.bank])
    v_scale = support_scale(config)
    v_scale_yaw = yaw_support_scale(config)
    run_dir = Path(args.run_dir)
    checkpoint_path = str(run_dir / f"model_{args.iteration}.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"missing periodic checkpoint: {checkpoint_path}")

    shard_rows = bank[args.shard_index::args.num_shards] if args.num_shards > 1 else bank
    num_envs = args.num_envs or len(shard_rows)
    if num_envs < len(shard_rows):
        raise ValueError(
            f"--num_envs={num_envs} smaller than this shard's {len(shard_rows)} replicas; "
            "raise --num_envs or increase --num_shards"
        )

    env, ppo_runner = build_eval_env(args.task, num_envs, args.seed, args.cpu)
    _load_checkpoint_into(ppo_runner, checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.device)

    assign_rows_to_env(env, shard_rows)
    measurements = rollout_measurements(
        env, policy, shard_rows,
        warmup_steps=args.warmup_steps or int(config["rollout"]["warmup_steps"]),
        steps=args.rollout_steps or int(config["rollout"]["steps"]),
        v_scale=v_scale, v_scale_yaw=v_scale_yaw,
    )

    payload = make_shard_payload(
        checkpoint_iteration=args.iteration,
        checkpoint_sha256=sha256_file(checkpoint_path),
        bank_kind=args.bank,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        config=config,
        measurements=measurements,
        rollout_steps=args.rollout_steps,
        warmup_steps=args.warmup_steps,
        provenance=_rollout_provenance(args, config, Path(checkpoint_path)),
    )
    out_path = _partial_path(run_dir, args.iteration, args.shard_index, args.num_shards, config, args.bank)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"[ued_rollout] wrote {len(measurements)} {args.bank} measurements -> {out_path}")


def run_merge(args) -> None:
    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / f"model_{args.iteration}.pt"
    checkpoint_sha256 = sha256_file(str(checkpoint_path))
    payloads: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        shard_path = _partial_path(run_dir, args.iteration, shard_index, args.num_shards, config, args.bank)
        payloads.append(json.loads(shard_path.read_text(encoding="utf-8")))
    # Effective horizon the shards already agreed on (CLI smoke override, else
    # the first shard's provenance). Merge checks consensus against this horizon
    # rather than always forcing the frozen headline protocol -- so a short
    # smoke shard set can produce a complete artifact. Selection still rejects
    # any non-frozen protocol_fingerprint (reviewer 8b).
    first_prov = (payloads[0].get("provenance") or {}) if payloads else {}
    merge_rollout_steps = (
        args.rollout_steps if args.rollout_steps is not None
        else first_prov.get("rollout_steps")
    )
    merge_warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None
        else first_prov.get("warmup_steps")
    )
    # Fail closed unless every shard header binds to this exact checkpoint SHA,
    # identity and the agreed protocol before its measurements are trusted.
    measurements = merge_shard_payloads(
        payloads,
        checkpoint_iteration=args.iteration,
        checkpoint_sha256=checkpoint_sha256,
        bank_kind=args.bank,
        num_shards=args.num_shards,
        config=config,
        rollout_steps=merge_rollout_steps,
        warmup_steps=merge_warmup_steps,
    )

    # Final artifact provenance must report the same effective horizon the
    # shards used; never re-default to the frozen config when shards were smoke.
    provenance = _rollout_provenance(args, config, checkpoint_path)
    if args.rollout_steps is None and merge_rollout_steps is not None:
        provenance["rollout_steps"] = int(merge_rollout_steps)
    if args.warmup_steps is None and merge_warmup_steps is not None:
        provenance["warmup_steps"] = int(merge_warmup_steps)

    artifact = make_validation_artifact(
        checkpoint_iteration=args.iteration,
        measurements=measurements,
        config=config,
        checkpoint_sha256=checkpoint_sha256,
        bank_kind=args.bank,
        provenance=provenance,
    )
    # Refuse to rewrite a smoke shard set as a frozen-horizon headline artifact
    # (or the reverse) if provenance was dropped or CLI/horizon disagreed.
    agreed_protocol = payloads[0].get("protocol_fingerprint")
    if artifact.get("protocol_fingerprint") != agreed_protocol:
        raise ValueError(
            "merged artifact protocol_fingerprint diverges from shard consensus; "
            "pass the same --rollout_steps/--warmup_steps used for the shards "
            "(or ensure shard provenance records the effective horizon)"
        )
    out_path = _final_path(run_dir, args.iteration, config, args.bank)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact), encoding="utf-8")
    print(f"[ued_rollout] merged {args.num_shards} {args.bank} shard(s) -> {out_path}")
    print(json.dumps(artifact["scores"], sort_keys=True))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Score one V5 checkpoint on the frozen UED validation bank")
    p.add_argument("--task", required=True, help="a registered go2_v5_* task (policy/PPO hyperparameters only)")
    p.add_argument("--run_dir", required=True, help="training run directory containing model_<iteration>.pt")
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--config", default=os.path.join(LEGGED_GYM_ROOT_DIR, "configs", "eval", "v5_ued.yaml"))
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--merge", action="store_true", help="combine existing shard partials into the final artifact")
    p.add_argument("--bank", choices=sorted(BANK_KINDS), default="validation",
                   help="'validation' scores checkpoints for selection; 'holdout' is the "
                        "single unbiased final measurement (eval_seed) and is never selected on")
    p.add_argument("--num_envs", type=int, default=None, help="override parallel envs (default: this shard's replica count)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true", default=False)
    # Smoke-only overrides: NEVER use for a headline campaign (§11 Faz 0 freezes
    # rollout.steps/warmup_steps at 1000/100 in configs/eval/v5_ued.yaml).
    p.add_argument("--rollout_steps", type=int, default=None, help="smoke-only override of the frozen rollout length")
    p.add_argument("--warmup_steps", type=int, default=None, help="smoke-only override of the frozen warmup length")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.merge:
        run_merge(args)
    else:
        run_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
