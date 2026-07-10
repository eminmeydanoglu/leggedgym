"""Experiment A -- does the oracle actually USE its privileged P?

The Wave-1 report shows oracle_id beating the DR-MLP in-band @vx=1.0 and the wide
oracle losing everywhere. Both facts are consistent with two very different
stories: (1) the policy reads P and maps it to better/worse control, or (2) P is
decorative and the score gap is just a different (lucky/unlucky) policy. This
script separates them by CORRUPTING the P block at inference while leaving the
true physics untouched, and by injecting COUNTERFACTUAL P labels into a policy
running on nominal physics.

The privileged block is the LAST 5 dims of the oracle obs:
    P = [friction(1), added_mass(1), com_bias(3)]   (dr_normalize'd to [-1, 1])
A plain MLP obs is 45-dim (no P); interventions are then no-ops (only `true` runs).

Two modes
---------
`intervene`  (A1, decisive/behavioral): sweep an OOD axis (default added_mass @
    vx=1.0, the headline scenario) and re-run the SAME rollout under each P
    intervention. If `true << shuffle ~ mean` the policy uses P; if all modes tie,
    P is decorative. `shuffle - true` = behavioral value of *correct* P.
      true    : P as-is (reproduces the standard eval curve)
      shuffle : permute P rows across the batch -> real but WRONG P (in-dist values,
                mismatched to physics). The clean "wrong info" test.
      mean    : P := 0 (the dr_normalize midpoint) -> "no info" test.
      random  : P ~ U(-1, 1) -> alternative wrong-info control.

`counterfactual` (A2, mechanism): pin physics at nominal, then SWEEP the injected
    added_mass label of P from -2..+5 kg while physics stays put, and watch mean
    feedforward effort (torque_sq) / action_rate / tracking respond. A monotone
    response in the physically-sensible direction = the policy reads P and uses it
    as feedforward. For oracle_id (narrow added_mass_range [-1, 1]) the label
    saturates past +-1 kg via dr_normalize's clamp -> directly visualises why the
    narrow oracle cannot extrapolate in mass.

Example (GPU box, env activated):
    # A1 -- headline scenario, all four modes
    python legged_gym/scripts/eval/probe_p.py --mode intervene \
        --task go2_bench_oracle_id --axis added_mass --command_vx 1.0 \
        --per_point 256 --steps 2000 --out logs/eval/probe/oracle_id_mass_vx1.npz

    # A2 -- counterfactual mass label on nominal physics
    python legged_gym/scripts/eval/probe_p.py --mode counterfactual \
        --task go2_bench_oracle_id --cf_param added_mass --command_vx 1.0 \
        --per_point 256 --steps 400 --out logs/eval/probe/oracle_id_cf_mass.npz

Run the SAME command for --task go2_bench_oracle (wide) and, as a P-free control,
--task go2_bench_mlp (only `true` is meaningful). Compare only within a matched
pair. Single seed -> a screen, not a proof; repeat with --seed to firm up.
"""

import argparse
import os
from types import SimpleNamespace

from legged_gym import *  # noqa: F401,F403  (gs, SIMULATOR, LEGGED_GYM_ROOT_DIR)
from legged_gym.envs import *  # noqa: F401,F403  (registers tasks)
from legged_gym.utils import task_registry

import numpy as np
import torch

from legged_gym.scripts.eval.dr_axes import get_axis, pin_others_to_nominal
from legged_gym.scripts.eval.metrics import MetricAccumulator
from legged_gym.scripts.eval.sweep import (
    resolve_load_run, make_registry_args, override_cfg_for_eval,
    aggregate, collect_run_meta,
)

P_DIM = 5  # [friction(1), added_mass(1), com_bias(3)]
# index of each privileged sub-block inside the P slice
P_INDEX = {"friction": 0, "added_mass": 1, "com_x": 2, "com_y": 3, "com_z": 4}
MODES = ["true", "shuffle", "mean", "random"]


def parse_args():
    p = argparse.ArgumentParser(description="Experiment A: probe whether the oracle uses privileged P")
    p.add_argument("--mode", choices=["intervene", "counterfactual"], default="intervene")
    p.add_argument("--task", type=str, required=True, help="oracle task (go2_bench_oracle / _oracle_id); mlp allowed as P-free control")
    p.add_argument("--load_run", type=str, default=None)
    p.add_argument("--ckpt", type=int, default=-1)
    # --- intervene mode ---
    p.add_argument("--axis", type=str, default="added_mass", help="OOD axis to sweep (intervene mode)")
    p.add_argument("--grid", type=float, nargs="+", default=None)
    p.add_argument("--interventions", nargs="+", default=MODES, choices=MODES)
    # --- counterfactual mode ---
    p.add_argument("--cf_param", type=str, default="added_mass", choices=list(P_INDEX),
                   help="which P sub-dim to sweep as an injected label (counterfactual mode)")
    p.add_argument("--cf_grid", type=float, nargs="+", default=None,
                   help="injected label values in physical units (default: the axis grid)")
    # --- shared ---
    p.add_argument("--per_point", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--command_vx", type=float, default=1.0)
    p.add_argument("--command_vy", type=float, default=0.0)
    p.add_argument("--command_yaw", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


def dr_normalize_np(x, lo, hi):
    """Mirror utils.math_utils.dr_normalize (incl. the [-1,1] clamp) for label injection."""
    rng = hi - lo
    if abs(rng) < 1e-12:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return np.clip(2.0 * (np.asarray(x, dtype=np.float64) - lo) / rng - 1.0, -1.0, 1.0)


def build_env_and_policy(cli, num_envs):
    """Shared build path -- identical protocol to sweep.py so numbers are comparable."""
    if SIMULATOR == "genesis":  # noqa: F405
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level="warning")  # noqa: F405

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)
    override_cfg_for_eval(env_cfg, cli, num_envs)

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)  # noqa: F405
    chosen_run = resolve_load_run(log_root, train_cfg.runner.run_name, cli.load_run)
    from legged_gym.utils.helpers import get_load_path
    ckpt_path = get_load_path(log_root, load_run=chosen_run, checkpoint=cli.ckpt)

    cli.load_run = chosen_run
    reg_args = make_registry_args(cli)
    env, _ = task_registry.make_env(name=cli.task, args=reg_args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cli.task, args=reg_args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    return env, policy, env_cfg, train_cfg, chosen_run, ckpt_path


def has_p(env):
    return env.num_obs >= 45 + P_DIM


def apply_intervention(obs, mode, gen):
    """Return obs with the P block (last P_DIM dims) corrupted per `mode`. Physics untouched."""
    if mode == "true" or obs.shape[1] < 45 + P_DIM:
        return obs
    o = obs.clone()
    s = o.shape[1] - P_DIM
    if mode == "shuffle":
        perm = torch.randperm(o.shape[0], device=o.device, generator=gen)
        o[:, s:] = o[perm, s:]
    elif mode == "mean":
        o[:, s:] = 0.0
    elif mode == "random":
        o[:, s:] = 2.0 * torch.rand((o.shape[0], P_DIM), device=o.device, generator=gen) - 1.0
    return o


def measure(env, policy, cli, warmup, transform):
    """Run warmup + measured rollout; `transform(obs)` maps obs->obs before the policy.

    Rebuilds fresh accumulators each call; caller is responsible for env state
    (we re-warmup every time so each pass starts from a settled gait).
    """
    env.reset()
    obs = env.get_observations()
    for _ in range(warmup):
        actions = policy(transform(obs).detach())
        obs, _, _, _, _ = env.step(actions.detach())
    env.episode_length_buf.zero_()

    acc = MetricAccumulator(env.num_envs, env.device)
    for _ in range(cli.steps):
        actions = policy(transform(obs).detach())
        obs, _, rew, dones, _ = env.step(actions.detach())
        cmd = env.commands
        v = env.simulator.base_lin_vel
        w = env.simulator.base_ang_vel
        lin_err = torch.norm(cmd[:, :2] - v[:, :2], dim=1)
        ang_err = torch.abs(cmd[:, 2] - w[:, 2])
        acc.update(
            rew, dones, env.time_out_buf, lin_err, ang_err,
            tilt=torch.norm(env.simulator.projected_gravity[:, :2], dim=1),
            torques=env.simulator.torques, dof_vel=env.simulator.dof_vel,
            actions=env.actions, last_actions=env.last_actions,
            feet_vel=env.simulator.feet_vel, feet_contact=env.feet_max_force_z > 1.0,
        )
    return acc.compute()


def run_intervene(cli):
    axis = get_axis(cli.axis)
    grid = np.asarray(cli.grid, dtype=np.float64) if cli.grid else axis.grid()
    num_points = len(grid)
    num_envs = num_points * cli.per_point
    label = cli.label or cli.task

    env, policy, env_cfg, train_cfg, chosen_run, ckpt_path = build_env_and_policy(cli, num_envs)

    per_env_vals = torch.tensor(np.repeat(grid, cli.per_point), dtype=torch.float, device=env.device)
    pin_others_to_nominal(env, cli.axis)
    axis.setter(env, per_env_vals)

    if not has_p(env):
        print(f"[probe] '{cli.task}' has no P block (num_obs={env.num_obs}); only 'true' is meaningful.")
        modes = ["true"]
    else:
        modes = cli.interventions

    gen = torch.Generator(device=env.device).manual_seed(cli.seed)
    results = {}
    for mode in modes:
        per_env = measure(env, policy, cli, cli.warmup, lambda o, m=mode: apply_intervention(o, m, gen))
        results[mode] = aggregate(per_env, num_points, cli.per_point)

    # comparison table: tracking_lin_err mean by grid point, one row per mode
    print(f"\n=== {label} | intervene | axis={cli.axis} vx={cli.command_vx} | tracking_lin_err (lower=better) ===")
    print("mode      " + " ".join(f"{g:>7.2f}" for g in grid))
    base = results.get("true", results[modes[0]])["tracking_lin_err_mean"]
    for mode in modes:
        row = results[mode]["tracking_lin_err_mean"]
        print(f"{mode:<9} " + " ".join(f"{x:7.3f}" for x in row))
    for mode in modes:
        if mode == "true":
            continue
        delta = results[mode]["tracking_lin_err_mean"] - base
        print(f"Δ({mode}-true) " + " ".join(f"{x:+7.3f}" for x in delta)
              + f"   [mean {delta.mean():+.3f}]  (+ => corrupting P HURT => P was used)")

    out = cli.out or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "eval", "probe", f"intervene_{label}_{cli.axis}.npz")  # noqa: F405
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = collect_run_meta(cli, chosen_run, ckpt_path)
    flat = {f"{mode}__{k}": v for mode, agg in results.items() for k, v in agg.items()}
    np.savez(out, mode="intervene", task=cli.task, axis=cli.axis, grid=grid,
             modes=np.array(modes), command_vx=cli.command_vx, per_point=cli.per_point,
             steps=cli.steps, seed=cli.seed, **meta, **flat)
    print(f"\nsaved -> {out}")


def run_counterfactual(cli):
    """Pin physics nominal; sweep an INJECTED P label; watch effort/tracking respond."""
    axis = get_axis(cli.cf_param if cli.cf_param in ("friction", "added_mass") else "com_x")
    grid = np.asarray(cli.cf_grid, dtype=np.float64) if cli.cf_grid else axis.grid()
    num_points = len(grid)
    num_envs = num_points * cli.per_point
    label = cli.label or cli.task

    env, policy, env_cfg, train_cfg, chosen_run, ckpt_path = build_env_and_policy(cli, num_envs)
    if not has_p(env):
        raise SystemExit(f"[probe] counterfactual needs a P block; '{cli.task}' has num_obs={env.num_obs}.")

    # physics identical across ALL envs = nominal; only the INJECTED label differs
    pin_others_to_nominal(env, cli.cf_param)  # also pins the cf axis physically to nominal
    axis.setter(env, torch.full((num_envs,), axis.nominal, device=env.device))

    # normalized labels to inject, per grid point (uses THIS policy's own range -> shows clamp)
    if cli.cf_param == "friction":
        lo, hi = env_cfg.domain_rand.friction_range
    elif cli.cf_param == "added_mass":
        lo, hi = env_cfg.domain_rand.added_mass_range
    else:
        lo, hi = -0.1, 0.1  # com bias: no per-dim normalize range in cfg; rough scale for the screen
    norm_labels = dr_normalize_np(grid, lo, hi)
    p_col = (env.num_obs - P_DIM) + P_INDEX[cli.cf_param]

    labels_t = torch.tensor(np.repeat(norm_labels, cli.per_point), dtype=torch.float, device=env.device)

    def inject(o):
        o = o.clone()
        o[:, p_col] = labels_t
        return o

    per_env = measure(env, policy, cli, cli.warmup, inject)
    agg = aggregate(per_env, num_points, cli.per_point)

    print(f"\n=== {label} | counterfactual | inject {cli.cf_param}, physics=nominal, vx={cli.command_vx} ===")
    print(f"{'label':>8} {'norm':>7} {'torque_sq':>10} {'act_rate':>9} {'lin_err':>9} {'return':>9}")
    for i, g in enumerate(grid):
        print(f"{g:8.3f} {norm_labels[i]:7.3f} {agg['torque_sq_mean_mean'][i]:10.3f} "
              f"{agg['action_rate_mean_mean'][i]:9.4f} {agg['tracking_lin_err_mean'][i]:9.4f} "
              f"{agg['mean_return_mean'][i]:9.2f}")
    print("Read: monotone torque_sq vs label => policy maps P->feedforward effort. "
          "Flat past the training range => dr_normalize clamp => cannot extrapolate.")

    out = cli.out or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "eval", "probe", f"cf_{label}_{cli.cf_param}.npz")  # noqa: F405
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = collect_run_meta(cli, chosen_run, ckpt_path)
    np.savez(out, mode="counterfactual", task=cli.task, cf_param=cli.cf_param, grid=grid,
             norm_labels=norm_labels, command_vx=cli.command_vx, per_point=cli.per_point,
             steps=cli.steps, seed=cli.seed, **meta, **agg)
    print(f"\nsaved -> {out}")


def main():
    cli = parse_args()
    if cli.mode == "intervene":
        run_intervene(cli)
    else:
        run_counterfactual(cli)


if __name__ == "__main__":
    main()
