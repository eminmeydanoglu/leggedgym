"""Minimal added-mass learn+use probe for RMA, DreamWaQ, and HIM.

Answers only:
  1. Can a frozen decoder read added mass from the method latent? (learned)
  2. Does wrong-mass intervention raise tracking_lin_err? (used)

Protocol (defaults):
  - command: lateral vy=+1.0
  - mass grid: [-2, 0, +3, +5] kg
  - friction / CoM nominal; push off; V3 mid-episode physics switch off
  - 128 envs / mass; 100 warmup + 400 measure steps
  - live mass re-read; invariant enforced; post-fall samples masked

Usage:
  python legged_gym/scripts/eval/probe_physics_use.py \\
      --method rma --task go2_v3_rma --load_run <run> --seed_label 1 \\
      --out_dir logs/eval/probes/physics_use/rma/seed_1

  # offline analysis only (no sim):
  python legged_gym/scripts/eval/probe_physics_use.py \\
      --analyze_only --samples samples.npz --use_npz use_metrics.npz \\
      --method rma --seed_label 1 --out_dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from legged_gym.scripts.eval.probe_adapters import get_adapter
from legged_gym.scripts.eval.probe_physics_logic import (
    DEFAULT_MEASURE_STEPS,
    DEFAULT_PER_POINT,
    DEFAULT_WARMUP,
    FALL_TRACKING_PENALTY,
    LATERAL_CMD,
    MASS_GRID_KG,
    aggregate_use_test,
    apply_fall_penalty_itt,
    apply_probe_physics_contract,
    build_table_row,
    check_mass_invariant,
    donor_map_stats,
    format_comparison_table,
    make_within_and_cross_donors,
    mask_valid_measurement,
    mass_decode_with_shuffle_control,
    update_frozen_latent_bank,
    use_evidence_flags,
    used_via_label,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Added-mass latent learn+use probe (RMA / DreamWaQ / HIM). "
            "Learned: traj-split mass decoder R² + shuffle control. "
            "Used: wrong-mass intervention Δuse on tracking_lin_err."
        )
    )
    p.add_argument(
        "--method",
        type=str,
        default="rma",
        choices=["rma", "dreamwaq", "him", "go2_v3_rma", "go2_v3_dreamwaq", "go2_v3_him_fixed"],
        help="method adapter (default: rma)",
    )
    p.add_argument("--task", type=str, default=None,
                   help="registered task (default: derived from --method)")
    p.add_argument("--load_run", type=str, default=None)
    p.add_argument("--ckpt", type=str, default="best_tracking")
    p.add_argument("--seed_label", type=str, default="1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--per_point", type=int, default=DEFAULT_PER_POINT)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--steps", type=int, default=DEFAULT_MEASURE_STEPS)
    p.add_argument("--stride", type=int, default=5,
                   help="record every stride steps during decode collection")
    p.add_argument("--mass_grid", type=float, nargs="+", default=list(MASS_GRID_KG))
    p.add_argument("--command_vx", type=float, default=LATERAL_CMD[0])
    p.add_argument("--command_vy", type=float, default=LATERAL_CMD[1])
    p.add_argument("--command_yaw", type=float, default=LATERAL_CMD[2])
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--skip_use", action="store_true",
                   help="only collect + decode (skip intervention rollouts)")
    p.add_argument("--skip_collect", action="store_true",
                   help="skip sim; require --samples / --use_npz")
    p.add_argument("--analyze_only", action="store_true",
                   help="alias: offline analysis from npz only")
    p.add_argument("--samples", type=str, default=None,
                   help="path to samples.npz for offline decode")
    p.add_argument("--use_npz", type=str, default=None,
                   help="path to use_metrics.npz for offline Δuse")
    p.add_argument("--max_decoder_samples", type=int, default=0,
                   help="if >0, subsample decode dataset")
    return p.parse_args(argv)


def default_task_for_method(method: str) -> str:
    m = method.lower()
    if m in ("rma", "go2_v3_rma"):
        return "go2_v3_rma"
    if m in ("dreamwaq", "go2_v3_dreamwaq", "dw"):
        return "go2_v3_dreamwaq"
    if m in ("him", "go2_v3_him_fixed", "him_fixed"):
        return "go2_v3_him_fixed"
    return method


# ---------------------------------------------------------------------------
# Env helpers (imported lazily so --help / pure analysis need no Genesis)
# ---------------------------------------------------------------------------


def _lazy_gym_imports():
    import legged_gym as lg
    import legged_gym.envs  # noqa: F401  — task registration side effect
    from legged_gym.utils import task_registry
    from legged_gym.scripts.eval.dr_axes import get_axis, pin_others_to_nominal
    from legged_gym.scripts.eval.sweep import (
        resolve_load_run, make_registry_args, override_cfg_for_eval,
    )
    return SimpleNamespace(
        gs=lg.gs,
        SIMULATOR=lg.SIMULATOR,
        LEGGED_GYM_ROOT_DIR=lg.LEGGED_GYM_ROOT_DIR,
        task_registry=task_registry,
        get_axis=get_axis,
        pin_others_to_nominal=pin_others_to_nominal,
        resolve_load_run=resolve_load_run,
        make_registry_args=make_registry_args,
        override_cfg_for_eval=override_cfg_for_eval,
    )


def build_env_and_policy(cli, num_envs: int):
    g = _lazy_gym_imports()
    if g.SIMULATOR == "genesis":
        g.gs.init(backend=g.gs.cpu if cli.cpu else g.gs.gpu, logging_level="warning")

    env_cfg, train_cfg = g.task_registry.get_cfgs(name=cli.task)
    g.override_cfg_for_eval(env_cfg, cli, num_envs)
    contract = apply_probe_physics_contract(env_cfg)
    print(f"[probe] physics contract: {contract}")

    log_root = os.path.join(g.LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    chosen_run = g.resolve_load_run(log_root, train_cfg.runner.run_name, cli.load_run)
    from legged_gym.utils.helpers import get_load_path
    ckpt_path = get_load_path(log_root, load_run=chosen_run, checkpoint=cli.ckpt)

    cli.load_run = chosen_run
    reg_args = g.make_registry_args(cli)
    env, _ = g.task_registry.make_env(name=cli.task, args=reg_args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = g.task_registry.make_alg_runner(
        env=env, name=cli.task, args=reg_args, train_cfg=train_cfg
    )
    ac = ppo_runner.alg.actor_critic
    ac.eval()
    return env, ac, env_cfg, train_cfg, chosen_run, ckpt_path, contract


def _set_command(env, cmd_vec, device):
    cmd = torch.tensor(cmd_vec, dtype=torch.float32, device=device)
    env.commands[:, :3] = cmd.unsqueeze(0).expand(env.num_envs, -1)


def _read_live_mass(env) -> np.ndarray:
    return (
        env.simulator._added_base_mass[:, 0]
        .detach().cpu().numpy().astype(np.float32)
    )


def _apply_mass_grid(env, mass_grid: Sequence[float], per_point: int, g) -> np.ndarray:
    """Pin non-mass axes nominal; set repeated mass grid. Return per-env targets."""
    num_envs = env.num_envs
    n_vals = len(mass_grid)
    expected = n_vals * per_point
    if num_envs != expected:
        raise ValueError(f"num_envs={num_envs} != len(mass_grid)*per_point={expected}")
    g.pin_others_to_nominal(env, "added_mass")
    per_env = torch.tensor(
        np.repeat(np.asarray(mass_grid, dtype=np.float32), per_point),
        dtype=torch.float32, device=env.device,
    )
    g.get_axis("added_mass").setter(env, per_env)
    return per_env.detach().cpu().numpy().astype(np.float32)


def _compute_lin_err(env) -> torch.Tensor:
    cmd = env.commands
    v = env.simulator.base_lin_vel
    return torch.norm(cmd[:, :2] - v[:, :2], dim=1)


def _unpack_obs(env, adapter_name: str, obs_pack):
    """Normalize env observation pack into a state dict for adapters."""
    # RMA: obs, priv, history, critic
    # DreamWaQ: obs, priv, history, explicit, next_state
    # HIM: single stacked history tensor (or tuple of one)
    if adapter_name == "HIM":
        if isinstance(obs_pack, (tuple, list)):
            hist = obs_pack[0]
        else:
            hist = obs_pack
        return {"obs": hist, "obs_history": hist, "history": hist}
    obs, priv_obs, history = obs_pack[0], obs_pack[1], obs_pack[2]
    return {"obs": obs, "priv_obs": priv_obs, "history": history}


def _step_env(env, actions, adapter_name: str):
    """Step env and normalize to adapter state.

    Contracts (see rsl_rl.runners.eval_adapter):
      HIM:      step -> (obs, priv, rew, dones, extras)  [5]
      RMA:      step -> (obs, priv, history, critic, rew, dones, extras)  [7]
      DreamWaQ: step -> (obs, priv, history, explicit, next, rew, dones, extras)  [8]
    """
    out = env.step(actions.detach())
    if adapter_name == "HIM":
        if len(out) < 5:
            raise RuntimeError(f"HIM env.step expected ≥5 values, got {len(out)}")
        obs_h, rew, dones, extras = out[0], out[2], out[3], out[4]
        return {"obs": obs_h, "obs_history": obs_h, "history": obs_h}, rew, dones, extras
    # RMA / DreamWaQ: history at index 2; rew/dones/extras are the last three
    if len(out) < 7:
        raise RuntimeError(
            f"{adapter_name} env.step expected ≥7 values (obs,priv,hist,...,rew,done,info), "
            f"got {len(out)}"
        )
    return {
        "obs": out[0],
        "priv_obs": out[1],
        "history": out[2],
    }, out[-3], out[-2], out[-1]


def _get_observations_pack(env):
    return env.get_observations()


def _warmup(env, ac, adapter, cmd_vec, warmup: int, adapter_name: str):
    env.reset()
    pack = _get_observations_pack(env)
    state = _unpack_obs(env, adapter_name, pack)
    device = env.device
    _set_command(env, cmd_vec, device)
    with torch.no_grad():
        for _ in range(warmup):
            _set_command(env, cmd_vec, device)
            actions = adapter.act_normal(ac, state)
            state, rew, dones, extras = _step_env(env, actions, adapter_name)
            _set_command(env, cmd_vec, device)
    env.episode_length_buf.zero_()
    return state


# ---------------------------------------------------------------------------
# Collection: decode dataset
# ---------------------------------------------------------------------------


def collect_decode_dataset(cli, env, ac, adapter, mass_targets: np.ndarray, g) -> Dict[str, np.ndarray]:
    """Roll out student/normal policy; record live mass + latent every stride."""
    adapter_name = adapter.name
    cmd_vec = [cli.command_vx, cli.command_vy, cli.command_yaw]
    device = env.device
    n = env.num_envs

    # re-apply mass before condition
    mass_targets = _apply_mass_grid(env, cli.mass_grid, cli.per_point, g)
    state = _warmup(env, ac, adapter, cmd_vec, cli.warmup, adapter_name)

    # verify mass still correct after warmup
    live0 = _read_live_mass(env)
    check_mass_invariant(live0, mass_targets, atol=1e-2)

    rows_z = []
    rows_zt = []
    rows_mass = []
    rows_traj = []
    rows_step = []
    rows_fall = []
    rows_env = []
    ever_invalid = np.zeros(n, dtype=bool)
    step_falls = []

    with torch.no_grad():
        for step_idx in range(cli.steps):
            _set_command(env, cmd_vec, device)
            live_mass = _read_live_mass(env)
            check_mass_invariant(live_mass, mass_targets, atol=1e-2)

            state_act = dict(state)
            state_act["real_mass_raw"] = torch.as_tensor(
                live_mass, device=device, dtype=torch.float32
            )
            actions = adapter.act_normal(ac, state_act)
            latents = adapter.extract_latent(ac, state_act)

            state_new, rew, dones, extras = _step_env(env, actions, adapter_name)
            fall = (dones.bool() & ~env.time_out_buf.bool()).detach().cpu().numpy()
            done_np = dones.bool().detach().cpu().numpy()
            step_falls.append(fall)
            # contamination: once fallen/reset, remaining steps for that env invalid
            ever_invalid = ever_invalid | fall | done_np

            if step_idx % cli.stride == 0:
                z = adapter.decode_features(latents).detach().cpu().float().numpy()
                mass_now = live_mass.copy()
                keep = mask_valid_measurement(fall, done_np, ever_invalid=ever_invalid)
                # also require mass match (should always)
                keep = keep & (np.abs(mass_now - mass_targets) <= 1e-2)
                idx = np.where(keep)[0]
                if idx.size:
                    rows_z.append(z[idx])
                    rows_mass.append(mass_now[idx])
                    # traj id: env_id is stable identity for this rollout
                    rows_traj.append(idx.astype(np.int64))
                    rows_step.append(np.full(idx.size, step_idx, dtype=np.int32))
                    rows_fall.append(fall[idx].astype(np.float32))
                    rows_env.append(idx.astype(np.int32))
                    zt = adapter.teacher_features(latents)
                    if zt is not None:
                        rows_zt.append(zt.detach().cpu().float().numpy()[idx])

            # re-pin mass after possible auto-reset (buffers may stay, but be safe)
            if np.any(done_np):
                _apply_mass_grid(env, cli.mass_grid, cli.per_point, g)
            state = state_new
            _set_command(env, cmd_vec, device)

    if not rows_z:
        raise RuntimeError("no valid decode samples collected (all fell?)")

    out = {
        "z": np.concatenate(rows_z, axis=0).astype(np.float32),
        "mass": np.concatenate(rows_mass, axis=0).astype(np.float32),
        "traj_id": np.concatenate(rows_traj, axis=0).astype(np.int64),
        "step": np.concatenate(rows_step, axis=0).astype(np.int32),
        "fall": np.concatenate(rows_fall, axis=0).astype(np.float32),
        "env_id": np.concatenate(rows_env, axis=0).astype(np.int32),
        "mass_targets_per_env": mass_targets.astype(np.float32),
    }
    if rows_zt:
        out["z_t"] = np.concatenate(rows_zt, axis=0).astype(np.float32)

    # final global invariant on collected labels
    # each sample's mass should equal mass_targets[traj_id]
    expected = mass_targets[out["traj_id"]]
    check_mass_invariant(out["mass"], expected, atol=1e-2, env_ids=out["env_id"])
    return out


# ---------------------------------------------------------------------------
# Use-test interventions
# ---------------------------------------------------------------------------


def _snapshot_rng(seed: int) -> Dict[str, Any]:
    """Capture RNG handles so each intervention mode can restart identically."""
    return {
        "seed": int(seed),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "numpy": np.random.get_state(),
    }


def _restore_rng(snap: Dict[str, Any]) -> None:
    """Restore torch/numpy RNG; re-seed env via caller with snap['seed']."""
    torch.set_rng_state(snap["torch"])
    if snap["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snap["cuda"])
    np.random.set_state(snap["numpy"])
    # Re-seed python/torch convenience generators for any residual draws
    torch.manual_seed(snap["seed"])
    np.random.seed(snap["seed"])


def run_use_test(cli, env, ac, adapter, g) -> Dict[str, Any]:
    """Three modes: student latent normal / within / cross (all methods).

    Pairing: each mode restores the same RNG seed before warmup so env noise
    and reset draws are as aligned as the simulator allows. Donor indices are
    fixed for the whole use-test (not re-sampled every step).

    Metric: intention-to-treat composite tracking — fall does not drop the env;
    remaining horizon is charged FALL_TRACKING_PENALTY. Primary Δuse is
    wrong−control on this composite; Δfall is a co-primary used signal.
    """
    adapter_name = adapter.name
    cmd_vec = [cli.command_vx, cli.command_vy, cli.command_yaw]
    device = env.device
    n = env.num_envs
    n_steps = int(cli.steps)
    mass_targets = _apply_mass_grid(env, cli.mass_grid, cli.per_point, g)

    # Fixed donor matching for the whole use-test (same for every mode).
    within_idx, cross_idx = make_within_and_cross_donors(
        mass_targets, seed=cli.seed, mass_grid=cli.mass_grid
    )
    within_t = torch.as_tensor(within_idx, device=device, dtype=torch.long)
    cross_t = torch.as_tensor(cross_idx, device=device, dtype=torch.long)

    modes = ["normal", "control", "wrong"]
    mode_err_sum = {m: np.zeros(n, dtype=np.float64) for m in modes}
    mode_alive_steps = {m: np.zeros(n, dtype=np.float64) for m in modes}
    mode_fall = {m: np.zeros(n, dtype=np.float64) for m in modes}
    # Donor contamination diagnostics (swap modes only)
    mode_frozen_exposure_steps = {m: 0 for m in ("control", "wrong")}
    mode_recv_weighted_donor_fall = {m: 0.0 for m in ("control", "wrong")}
    mode_unique_donor_fall_frac = {m: 0.0 for m in ("control", "wrong")}
    mode_total_swap_steps = {m: 0 for m in ("control", "wrong")}
    post_reset_refresh_total = 0

    # Shared pairing seed: each mode starts from the same RNG snapshot.
    pair_seed = int(cli.seed) + 10_007
    torch.manual_seed(pair_seed)
    np.random.seed(pair_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(pair_seed)
    rng_snap = _snapshot_rng(pair_seed)

    for mode in modes:
        print(f"[probe-use] mode={mode} (paired seed={pair_seed}, fixed donors, freeze-fallen-bank)")
        _restore_rng(rng_snap)
        if hasattr(env, "cfg") and hasattr(env.cfg, "seed"):
            env.cfg.seed = pair_seed
        mass_targets = _apply_mass_grid(env, cli.mass_grid, cli.per_point, g)
        state = _warmup(env, ac, adapter, cmd_vec, cli.warmup, adapter_name)
        live0 = _read_live_mass(env)
        check_mass_invariant(live0, mass_targets, atol=1e-2)

        ever_fell = np.zeros(n, dtype=bool)
        # Latent bank: update only for non-fallen envs; freeze last pre-fall latent
        # so post-reset donors never contaminate receivers.
        frozen_bank = None
        frozen_exposure_steps = 0
        total_swap_steps = 0
        post_reset_refresh = 0  # measured: fallen slots changed after update

        with torch.no_grad():
            for step_idx in range(n_steps):
                _set_command(env, cmd_vec, device)
                live_mass = _read_live_mass(env)
                check_mass_invariant(live_mass, mass_targets, atol=1e-2)

                state_act = dict(state)
                mass_t = torch.as_tensor(live_mass, device=device, dtype=torch.float32)
                state_act["real_mass_raw"] = mass_t

                latents = adapter.extract_latent(ac, state_act)
                live_bank = adapter.decode_features(latents)
                if frozen_bank is None:
                    frozen_bank = live_bank.detach().clone()
                else:
                    fell_np = ever_fell
                    if fell_np.any():
                        before_fallen = frozen_bank[fell_np].detach().clone()
                    updated = update_frozen_latent_bank(
                        frozen_bank, live_bank, ever_fell
                    )
                    if fell_np.any():
                        # real invariant: fallen slots must be bitwise-equal to pre-update
                        if not torch.equal(updated[fell_np], before_fallen):
                            changed = (
                                (updated[fell_np] != before_fallen)
                                .any(dim=-1)
                                .sum()
                                .item()
                            )
                            post_reset_refresh += int(changed)
                    frozen_bank = updated

                donors = {
                    "within_idx": within_t,
                    "cross_idx": cross_t,
                    "latent_bank": frozen_bank,
                    "real_mass_raw": mass_t,
                    "mass_grid": list(cli.mass_grid),
                }

                if mode == "normal":
                    actions = adapter.act_normal(ac, state_act)
                elif mode == "control":
                    # frozen exposure: alive receiver whose donor already fell
                    # (uses last pre-fall latent — not post-reset contamination)
                    donor_fell = ever_fell[within_idx]
                    recv_alive = ~ever_fell
                    frozen_exposure_steps += int(np.sum(donor_fell & recv_alive))
                    total_swap_steps += int(np.sum(recv_alive))
                    actions = adapter.act_control(ac, state_act, donors)
                else:
                    donor_fell = ever_fell[cross_idx]
                    recv_alive = ~ever_fell
                    frozen_exposure_steps += int(np.sum(donor_fell & recv_alive))
                    total_swap_steps += int(np.sum(recv_alive))
                    actions = adapter.act_wrong(ac, state_act, donors)

                state_new, rew, dones, extras = _step_env(env, actions, adapter_name)
                fall = (dones.bool() & ~env.time_out_buf.bool()).detach().cpu().numpy()
                done_np = dones.bool().detach().cpu().numpy()
                # accumulate tracking only while not yet fallen (ITT fills rest)
                still_alive = ~ever_fell
                lin_err = _compute_lin_err(env).detach().cpu().numpy()
                mode_err_sum[mode][still_alive] += lin_err[still_alive]
                mode_alive_steps[mode][still_alive] += 1.0
                ever_fell = ever_fell | fall
                mode_fall[mode] = ever_fell.astype(np.float64)

                if np.any(done_np):
                    _apply_mass_grid(env, cli.mass_grid, cli.per_point, g)
                state = state_new
                _set_command(env, cmd_vec, device)

        if mode in mode_frozen_exposure_steps:
            mode_frozen_exposure_steps[mode] = frozen_exposure_steps
            mode_total_swap_steps[mode] = max(total_swap_steps, 1)
            d_idx = within_idx if mode == "control" else cross_idx
            mode_recv_weighted_donor_fall[mode] = float(np.mean(ever_fell[d_idx]))
            uniq = np.unique(d_idx)
            mode_unique_donor_fall_frac[mode] = (
                float(np.mean(ever_fell[uniq])) if len(uniq) else 0.0
            )
            post_reset_refresh_total += post_reset_refresh
            if post_reset_refresh != 0:
                raise RuntimeError(
                    f"post-reset donor bank refresh detected: {post_reset_refresh} "
                    "fallen slots were overwritten"
                )

    # ITT composite: all envs retained; fall remainder charged penalty
    normal = apply_fall_penalty_itt(
        mode_err_sum["normal"], mode_alive_steps["normal"],
        mode_fall["normal"] > 0, n_steps, penalty=FALL_TRACKING_PENALTY,
    )
    control = apply_fall_penalty_itt(
        mode_err_sum["control"], mode_alive_steps["control"],
        mode_fall["control"] > 0, n_steps, penalty=FALL_TRACKING_PENALTY,
    )
    wrong = apply_fall_penalty_itt(
        mode_err_sum["wrong"], mode_alive_steps["wrong"],
        mode_fall["wrong"] > 0, n_steps, penalty=FALL_TRACKING_PENALTY,
    )

    use = aggregate_use_test(
        normal, control, wrong,
        normal_fall=mode_fall["normal"],
        control_fall=mode_fall["control"],
        wrong_fall=mode_fall["wrong"],
        seed=cli.seed,
        metric_kind="itt_composite",
    )
    frozen_exp = {
        m: mode_frozen_exposure_steps[m] / max(mode_total_swap_steps[m], 1)
        for m in ("control", "wrong")
    }
    w_stats = donor_map_stats(within_idx)
    c_stats = donor_map_stats(cross_idx)
    return {
        "normal_err_per_env": normal,
        "control_err_per_env": control,
        "wrong_err_per_env": wrong,
        "fall_normal": mode_fall["normal"],
        "fall_control": mode_fall["control"],
        "fall_wrong": mode_fall["wrong"],
        "mass_targets": mass_targets,
        "use": use,
        "use_test_kind": getattr(adapter, "use_test_kind", "student_latent_swap"),
        "pair_seed": pair_seed,
        "pairing_kind": "rng_restore_best_effort",
        "ci_kind": "within_run_paired_bootstrap",
        "fall_tracking_penalty": FALL_TRACKING_PENALTY,
        "donors_within": within_idx,
        "donors_cross": cross_idx,
        "frozen_donor_exposure_rate_control": frozen_exp["control"],
        "frozen_donor_exposure_rate_wrong": frozen_exp["wrong"],
        "post_reset_donor_refresh_count": int(post_reset_refresh_total),
        "receiver_weighted_donor_fall_control": mode_recv_weighted_donor_fall["control"],
        "receiver_weighted_donor_fall_wrong": mode_recv_weighted_donor_fall["wrong"],
        "unique_donor_fall_frac_control": mode_unique_donor_fall_frac["control"],
        "unique_donor_fall_frac_wrong": mode_unique_donor_fall_frac["wrong"],
        "unique_donor_count_within": w_stats["unique_donor_count"],
        "unique_donor_count_cross": c_stats["unique_donor_count"],
        "max_receivers_per_donor_within": w_stats["max_receivers_per_donor"],
        "max_receivers_per_donor_cross": c_stats["max_receivers_per_donor"],
        "mean_receivers_per_donor_cross": c_stats["mean_receivers_per_donor"],
        "latent_bank_policy": "freeze_pre_fall",
    }


# ---------------------------------------------------------------------------
# Offline analysis
# ---------------------------------------------------------------------------


def analyze_samples(
    samples: Dict[str, np.ndarray],
    *,
    method: str,
    seed_label: str,
    seed: int = 0,
    use_result: Optional[Dict[str, Any]] = None,
    max_samples: int = 0,
) -> Dict[str, Any]:
    z = samples["z"]
    mass = samples["mass"]
    traj = samples["traj_id"]
    if max_samples > 0 and len(z) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z), max_samples, replace=False)
        idx.sort()
        z, mass, traj = z[idx], mass[idx], traj[idx]

    # live-mass invariant: mass equals target for traj if present
    if "mass_targets_per_env" in samples:
        expected = samples["mass_targets_per_env"][traj]
        check_mass_invariant(mass, expected, atol=1e-2)

    decode = mass_decode_with_shuffle_control(z, mass, traj, seed=seed)
    teacher_r2 = None
    if "z_t" in samples:
        z_t = samples["z_t"]
        if max_samples > 0 and len(samples["z"]) > max_samples:
            z_t = z_t[idx]  # type: ignore[name-defined]
        # same split indices
        split = decode["split"]
        from legged_gym.scripts.eval.probe_physics_logic import fit_mass_decoder
        treal = fit_mass_decoder(
            z_t, mass,
            split.train_idx, split.test_idx,
            traj_ids=traj, seed=seed,
        )
        teacher_r2 = treal.r2

    if use_result is None:
        use_result = {
            "normal_err": float("nan"),
            "control_err": float("nan"),
            "wrong_err": float("nan"),
            "delta_use": float("nan"),
            "delta_use_ci_lo": float("nan"),
            "delta_fall": 0.0,
            "delta_fall_ci_lo": float("nan"),
            "control_fall": float("nan"),
            "wrong_fall": float("nan"),
        }
    elif "use" in use_result and hasattr(use_result["use"], "delta_use"):
        u = use_result["use"]
        use_result = {
            "normal_err": u.normal_err,
            "control_err": u.control_err,
            "wrong_err": u.wrong_err,
            "delta_use": u.delta_use,
            "delta_use_ci_lo": u.delta_use_ci_lo,
            "normal_fall": u.normal_fall,
            "control_fall": u.control_fall,
            "wrong_fall": u.wrong_fall,
            "delta_fall": u.delta_fall,
            "delta_fall_ci_lo": u.delta_fall_ci_lo,
            "delta_fall_ci_hi": u.delta_fall_ci_hi,
        }

    adapter = get_adapter(method)
    row = build_table_row(
        method=adapter.name,
        seed=seed_label,
        mass_r2=decode["r2"],
        shuffled_r2=decode["shuffled_r2"],
        normal_err=use_result.get("normal_err", float("nan")),
        control_err=use_result.get("control_err", float("nan")),
        wrong_err=use_result.get("wrong_err", float("nan")),
        delta_use=use_result.get("delta_use", float("nan")),
        delta_use_ci_lo=use_result.get("delta_use_ci_lo", float("nan")),
        teacher_r2=teacher_r2,
        delta_fall=float(use_result.get("delta_fall", 0.0)),
        delta_fall_ci_lo=float(use_result.get("delta_fall_ci_lo", float("nan"))),
        control_fall=float(use_result.get("control_fall", float("nan"))),
        wrong_fall=float(use_result.get("wrong_fall", float("nan"))),
        use_test_kind=getattr(adapter, "use_test_kind", "student_latent_swap"),
    )
    return {
        "row": row,
        "decode": {
            "r2": decode["r2"],
            "shuffled_r2": decode["shuffled_r2"],
            "mae": decode["mae"],
            "n_train": decode["n_train"],
            "n_test": decode["n_test"],
            "teacher_r2": teacher_r2,
        },
        "use": use_result,
        "table_md": format_comparison_table([row]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    print(f"[probe] method={adapter.name} task={cli.task} seed_label={cli.seed_label}")
    print(f"[probe] command=({cli.command_vx}, {cli.command_vy}, {cli.command_yaw}) "
          f"mass_grid={cli.mass_grid}")

    samples = None
    use_bundle = None

    if cli.skip_collect:
        if not cli.samples:
            print("[probe] --skip_collect/--analyze_only requires --samples", file=sys.stderr)
            return 2
        data = np.load(cli.samples, allow_pickle=True)
        samples = {k: data[k] for k in data.files if k != "meta"}
        if cli.use_npz and os.path.isfile(cli.use_npz):
            ud = np.load(cli.use_npz, allow_pickle=True)
            def _f(key, default=float("nan")):
                return float(ud[key]) if key in ud else default

            use_bundle = {
                "normal_err": _f("normal_err"),
                "control_err": _f("control_err"),
                "wrong_err": _f("wrong_err"),
                "delta_use": _f("delta_use"),
                "delta_use_ci_lo": _f("delta_use_ci_lo"),
                "delta_fall": _f("delta_fall", 0.0),
                "delta_fall_ci_lo": _f("delta_fall_ci_lo"),
                "control_fall": _f("control_fall"),
                "wrong_fall": _f("wrong_fall"),
            }
    else:
        g = _lazy_gym_imports()
        n_vals = len(cli.mass_grid)
        num_envs = n_vals * cli.per_point
        env, ac, env_cfg, train_cfg, chosen_run, ckpt_path, contract = \
            build_env_and_policy(cli, num_envs)

        samples = collect_decode_dataset(cli, env, ac, adapter, None, g)
        samples_path = os.path.join(cli.out_dir, "samples.npz")
        meta = {
            "method": adapter.name,
            "task": cli.task,
            "seed_label": cli.seed_label,
            "mass_grid": list(cli.mass_grid),
            "command": [cli.command_vx, cli.command_vy, cli.command_yaw],
            "warmup": cli.warmup,
            "steps": cli.steps,
            "stride": cli.stride,
            "per_point": cli.per_point,
            "load_run": cli.load_run,
            "ckpt": ckpt_path,
            "contract": contract,
        }
        np.savez(samples_path, meta=np.array(json.dumps(meta)), **samples)
        print(f"[probe] samples -> {samples_path}  N={samples['z'].shape[0]}")

        if not cli.skip_use:
            use_raw = run_use_test(cli, env, ac, adapter, g)
            u = use_raw["use"]
            use_bundle = {
                "normal_err": u.normal_err,
                "control_err": u.control_err,
                "wrong_err": u.wrong_err,
                "delta_use": u.delta_use,
                "delta_use_ci_lo": u.delta_use_ci_lo,
                "delta_use_ci_hi": u.delta_use_ci_hi,
                "normal_fall": u.normal_fall,
                "control_fall": u.control_fall,
                "wrong_fall": u.wrong_fall,
                "delta_fall": u.delta_fall,
                "delta_fall_ci_lo": u.delta_fall_ci_lo,
                "delta_fall_ci_hi": u.delta_fall_ci_hi,
                "n_pairs": u.n_pairs,
                "metric_kind": u.metric_kind,
                "use_test_kind": use_raw.get("use_test_kind", "student_latent_swap"),
                "pair_seed": use_raw.get("pair_seed", -1),
                "pairing_kind": use_raw.get("pairing_kind", "rng_restore_best_effort"),
                "fall_tracking_penalty": use_raw.get(
                    "fall_tracking_penalty", FALL_TRACKING_PENALTY
                ),
                "latent_bank_policy": use_raw.get("latent_bank_policy", "freeze_pre_fall"),
                "ci_kind": use_raw.get("ci_kind", "within_run_paired_bootstrap"),
                "frozen_donor_exposure_rate_control": use_raw.get(
                    "frozen_donor_exposure_rate_control", 0.0
                ),
                "frozen_donor_exposure_rate_wrong": use_raw.get(
                    "frozen_donor_exposure_rate_wrong", 0.0
                ),
                "post_reset_donor_refresh_count": use_raw.get(
                    "post_reset_donor_refresh_count", 0
                ),
                "receiver_weighted_donor_fall_control": use_raw.get(
                    "receiver_weighted_donor_fall_control", 0.0
                ),
                "receiver_weighted_donor_fall_wrong": use_raw.get(
                    "receiver_weighted_donor_fall_wrong", 0.0
                ),
                "unique_donor_fall_frac_control": use_raw.get(
                    "unique_donor_fall_frac_control", 0.0
                ),
                "unique_donor_fall_frac_wrong": use_raw.get(
                    "unique_donor_fall_frac_wrong", 0.0
                ),
                "unique_donor_count_within": use_raw.get(
                    "unique_donor_count_within", 0.0
                ),
                "unique_donor_count_cross": use_raw.get("unique_donor_count_cross", 0.0),
                "max_receivers_per_donor_within": use_raw.get(
                    "max_receivers_per_donor_within", 0.0
                ),
                "max_receivers_per_donor_cross": use_raw.get(
                    "max_receivers_per_donor_cross", 0.0
                ),
                "mean_receivers_per_donor_cross": use_raw.get(
                    "mean_receivers_per_donor_cross", 0.0
                ),
                "used_via": used_via_label(
                    use_evidence_flags(
                        u.delta_use,
                        u.delta_use_ci_lo,
                        delta_fall=u.delta_fall,
                        delta_fall_ci_lo=u.delta_fall_ci_lo,
                    )
                ),
            }
            use_path = os.path.join(cli.out_dir, "use_metrics.npz")
            np.savez(
                use_path,
                **{
                    k: np.asarray(v) if not isinstance(v, str) else np.array(v)
                    for k, v in use_bundle.items()
                },
                normal_err_per_env=use_raw["normal_err_per_env"],
                control_err_per_env=use_raw["control_err_per_env"],
                wrong_err_per_env=use_raw["wrong_err_per_env"],
                mass_targets=use_raw["mass_targets"],
                donors_within=use_raw["donors_within"],
                donors_cross=use_raw["donors_cross"],
            )
            print(f"[probe] use metrics -> {use_path}")

    result = analyze_samples(
        samples,
        method=cli.method,
        seed_label=cli.seed_label,
        seed=cli.seed,
        use_result=use_bundle,
        max_samples=cli.max_decoder_samples,
    )
    out_json = os.path.join(cli.out_dir, "probe_result.json")
    # JSON-serializable
    serial = {
        "row": result["row"],
        "decode": result["decode"],
        "use": {k: (float(v) if isinstance(v, (float, np.floating, int)) else v)
                for k, v in (result["use"] or {}).items()
                if not isinstance(v, np.ndarray)},
        "table_md": result["table_md"],
    }
    with open(out_json, "w") as f:
        json.dump(serial, f, indent=2)
    table_path = os.path.join(cli.out_dir, "table.md")
    with open(table_path, "w") as f:
        f.write(result["table_md"] + "\n")
    print(result["table_md"])
    print(f"[probe] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
