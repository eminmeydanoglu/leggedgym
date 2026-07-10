"""Transient (time-resolved) evaluation scenarios for the Go2 adaptation benchmark.

The steady-state sweep (`sweep.py`) holds ONE fixed forward command and one fixed
physics point and reports distribution-over-episodes summaries. That measures
*local sensitivity* to a mis-specified parameter, but it deliberately averages
away the thing online adaptation is supposed to help with: the TRANSIENT after a
disturbance. A policy that has (implicitly or explicitly) estimated the true
parameter should re-converge faster after a command change or a shove, even when
its steady-state tracking error looks identical on the sweep curve. This module
adds the two transient scenarios the README flagged as still-missing:

  * `step_response`  -- drive a deterministic COMMAND schedule (stand -> forward ->
    reverse -> lateral -> stop) identical for every env, and measure how fast/clean
    the policy tracks each command change (settling time, peak error, error
    integral). This is the "transient command" probe.

  * `push_recovery`  -- hold a fixed forward command, then at a PRE-SCHEDULED step
    apply the SAME fixed velocity impulse to every env's base and measure recovery
    (recovery time, peak tilt, error integral, fall rate in the window). This is the
    "fixed-impulse recovery" probe.

Both scenarios reuse `sweep.py`'s scaffolding verbatim -- `resolve_load_run`
(checkpoint isolation so a 45-dim model is never loaded into a 50-dim oracle slot),
`make_registry_args`, `override_cfg_for_eval` (freeze physics, collapse command
ranges, kill curriculum/heading), and the warmup + `episode_length_buf.zero_()`
episode-clock reset trick. We do NOT re-implement that logic here -- importing it
keeps sweep.py the single source of truth and byte-identical.

Unlike the sweep, transient scenarios measure a SINGLE time window per env, not a
distribution over auto-reset episodes: we tile the window across `per_point`
independent env replicas (== independent seeds, exactly like sweep's per_point) and
aggregate the per-env transient metrics across those replicas (mean/std/quantiles).

Example (on the GPU box, env activated):
    export SIMULATOR=genesis
    python legged_gym/scripts/eval/transient.py --scenario step_response \
        --task go2_bench_mlp --load_run <run> --per_point 64 \
        --out logs/eval/step_mlp.npz

    python legged_gym/scripts/eval/transient.py --scenario push_recovery \
        --task go2_bench_mlp --load_run <run> --per_point 64 --dvy 1.0 \
        --out logs/eval/push_mlp.npz
"""

import argparse
import os
import sys

from legged_gym import *  # noqa: F401,F403  (exposes gs, SIMULATOR, LEGGED_GYM_ROOT_DIR)
from legged_gym.envs import *  # noqa: F401,F403  (registers tasks)
from legged_gym.utils import task_registry

import numpy as np
import torch

from legged_gym.scripts.eval.dr_axes import get_axis, pin_others_to_nominal
# Reuse sweep.py's scaffolding rather than duplicating the (subtle) checkpoint
# isolation / registry-arg / eval-freeze logic. Importing keeps a single source of
# truth so sweep.py stays byte-identical and any future fix lands in both places.
from legged_gym.scripts.eval.sweep import (
    resolve_load_run,
    git_commit,
    make_registry_args,
    override_cfg_for_eval,
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Transient (time-resolved) eval scenarios")
    p.add_argument("--scenario", type=str, required=True,
                   choices=["step_response", "push_recovery"],
                   help="which transient probe to run")

    # --- shared loading / isolation scaffolding (mirrors sweep.py) ---
    p.add_argument("--task", type=str, required=True, help="registered benchmark task, e.g. go2_bench_mlp")
    p.add_argument("--load_run", type=str, default=None, help="run dir to load (default: latest)")
    p.add_argument("--ckpt", type=int, default=-1, help="checkpoint iteration (-1 = latest)")
    p.add_argument("--per_point", type=int, default=64,
                   help="parallel env replicas == independent seeds; transient metrics "
                        "are aggregated (mean/std/quantiles) across them")
    p.add_argument("--warmup", type=int, default=100, help="unrecorded settling steps before the scenario")
    p.add_argument("--seed", type=int, default=1, help="global seed (repeat runs to add more seeds)")
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--out", type=str, default=None, help="output .npz path")
    p.add_argument("--label", type=str, default=None, help="curve label (default: task name)")
    p.add_argument("--tol_lin", type=float, default=0.25,
                   help="lin-vel tracking tolerance [m/s]: error must fall below this to "
                        "count as 'settled'/'recovered' (matches metrics.lin_err_threshold)")

    # --- optional physics point (run the scenario off-nominal) ---
    p.add_argument("--axis", type=str, default=None,
                   help="optional dr_axes axis to hold at a fixed off-nominal value "
                        "(e.g. added_mass) so we can ask 'does a heavy robot recover worse?'")
    p.add_argument("--axis_value", type=float, default=None,
                   help="value for --axis (required if --axis is set)")

    # --- step_response schedule ---
    p.add_argument("--phase_steps", type=int, default=150,
                   help="steps per command phase in step_response")
    p.add_argument("--step_vx", type=float, default=0.5,
                   help="forward command magnitude [m/s]; |vx|<=0.5 stays in training range")
    p.add_argument("--step_vy", type=float, default=0.5,
                   help="lateral command magnitude [m/s]; |vy|<=1.0 stays in training range")
    p.add_argument("--step_yaw", type=float, default=0.0,
                   help="yaw-rate command [rad/s] for any yaw phase; |yaw|<=1.0 in range")

    # --- push_recovery schedule ---
    p.add_argument("--command_vx", type=float, default=0.5, help="held forward command [m/s] during push_recovery")
    p.add_argument("--command_vy", type=float, default=0.0, help="held lateral command [m/s] during push_recovery")
    p.add_argument("--command_yaw", type=float, default=0.0, help="held yaw-rate command [rad/s] during push_recovery")
    p.add_argument("--pre_push_steps", type=int, default=50,
                   help="recorded steady steps before the impulse (baseline window)")
    p.add_argument("--recovery_steps", type=int, default=150,
                   help="recorded steps after the impulse (recovery window)")
    p.add_argument("--dvx", type=float, default=0.0, help="impulse added to base world x-velocity [m/s]")
    p.add_argument("--dvy", type=float, default=1.0, help="impulse added to base world y-velocity [m/s]")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Command schedule for step_response
# --------------------------------------------------------------------------- #
def build_command_schedule(cli):
    """Return (schedule, phase_bounds).

    schedule: (T, 3) array of [vx, vy, yaw] the SAME for every env, one row per
    measured step. phase_bounds: list of (name, start_step) marking each command
    change so per-phase metrics can be sliced out.

    Phases (each `phase_steps` long): stand -> forward -> reverse -> lateral -> stop.
    Values are clamped to the benchmark training range so the probe stays
    confirmatory (|vx|<=0.5, |vy|<=1.0, |yaw|<=1.0); use larger magnitudes only as an
    explicitly-labelled command-OOD diagnostic.
    """
    ps = cli.phase_steps
    phases = [
        ("stand",   (0.0,          0.0,         0.0)),
        ("forward", (cli.step_vx,  0.0,         0.0)),
        ("reverse", (-cli.step_vx, 0.0,         0.0)),
        ("lateral", (0.0,          cli.step_vy, 0.0)),
        ("stop",    (0.0,          0.0,         0.0)),
    ]
    schedule = np.zeros((len(phases) * ps, 3), dtype=np.float64)
    bounds = []
    for i, (name, cmd) in enumerate(phases):
        s = i * ps
        schedule[s:s + ps] = cmd
        bounds.append((name, s))
    return schedule, bounds


# --------------------------------------------------------------------------- #
# Transient accumulators
# --------------------------------------------------------------------------- #
class TransientRecorder:
    """Record per-env time series of tracking error, tilt, and fall flags.

    Unlike `metrics.MetricAccumulator` (episode-distribution stats under
    auto_reset), a transient scenario cares about the SHAPE of a single time window
    per env. We keep full per-step tensors on-device and reduce them into scenario
    metrics at the end. All buffers are (T, num_envs); `T` is known ahead of time.

    "fell" is latched: once an env terminates (fall, not timeout) inside the window
    it stays flagged, so a mid-window recovery followed by a late fall still counts
    as a failure, and a fall isn't double-counted after auto_reset re-spawns it.
    """

    def __init__(self, T: int, num_envs: int, device):
        self.T = T
        self.num_envs = num_envs
        self.device = device
        z = lambda: torch.zeros(T, num_envs, device=device)
        self.lin_err = z()   # |cmd_xy - v_xy|
        self.ang_err = z()   # |cmd_yaw - w_yaw|
        self.tilt = z()      # ||projected_gravity_xy||  (sine of base tilt)
        self.fell = torch.zeros(T, num_envs, dtype=torch.bool, device=device)
        self._ever_fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._t = 0

    @torch.no_grad()
    def record(self, env):
        """Read the current step's transient signals off the env. Call once/step."""
        cmd = env.commands
        v = env.simulator.base_lin_vel
        w = env.simulator.base_ang_vel
        lin_err = torch.norm(cmd[:, :2] - v[:, :2], dim=1)
        ang_err = torch.abs(cmd[:, 2] - w[:, 2])
        tilt = torch.norm(env.simulator.projected_gravity[:, :2], dim=1)
        # a fall is a non-timeout termination this step; latch it for the window
        fell_now = env.reset_buf.bool() & ~env.time_out_buf.bool()
        self._ever_fell |= fell_now

        t = self._t
        self.lin_err[t] = lin_err
        self.ang_err[t] = ang_err
        self.tilt[t] = tilt
        self.fell[t] = self._ever_fell
        self._t += 1

    # ---- reductions ------------------------------------------------------- #
    @torch.no_grad()
    def error_integral(self, dt: float, lo: int = 0, hi: int = None) -> torch.Tensor:
        """Integral of lin-tracking error over [lo, hi) steps, per env (units m)."""
        hi = self.T if hi is None else hi
        return self.lin_err[lo:hi].sum(dim=0) * dt

    @torch.no_grad()
    def peak_error(self, lo: int = 0, hi: int = None) -> torch.Tensor:
        hi = self.T if hi is None else hi
        return self.lin_err[lo:hi].max(dim=0).values

    @torch.no_grad()
    def peak_tilt(self, lo: int = 0, hi: int = None) -> torch.Tensor:
        hi = self.T if hi is None else hi
        return self.tilt[lo:hi].max(dim=0).values

    @torch.no_grad()
    def settling_time(self, tol: float, lo: int, hi: int = None) -> torch.Tensor:
        """Steps after `lo` until lin-error first drops below `tol` and STAYS below.

        We require the error to be below tol at the found step AND for the rest of
        the window, so a single lucky dip during a wild transient doesn't count as
        settled. Envs that never settle in [lo, hi) are right-censored to (hi - lo).
        """
        hi = self.T if hi is None else hi
        win = self.lin_err[lo:hi]                       # (L, num_envs)
        L = win.shape[0]
        below = win < tol                               # (L, num_envs)
        # "stays below from here on": suffix-AND. reverse-cummin over a bool as int.
        stays = torch.flip(torch.cummin(torch.flip(below.int(), dims=[0]), dim=0).values, dims=[0]).bool()
        # first index where it stays below for the rest of the window
        idx = torch.where(
            stays.any(dim=0),
            stays.float().argmax(dim=0),                # first True
            torch.full((self.num_envs,), float(L), device=self.device),  # censored
        )
        return idx.float()

    @torch.no_grad()
    def fall_rate(self, at: int = None) -> torch.Tensor:
        """Fraction (0/1 per env) that had fallen by step `at` (default: window end)."""
        at = self.T - 1 if at is None else at
        return self.fell[at].float()


# --------------------------------------------------------------------------- #
# Aggregation across per_point replicas (mirrors sweep.aggregate's stats)
# --------------------------------------------------------------------------- #
def agg_stats(vec: torch.Tensor) -> dict:
    """mean/std/p25/p50/p75 of a per-env scalar vector across the replicas."""
    arr = vec.detach().cpu().numpy()
    return dict(
        mean=float(arr.mean()), std=float(arr.std()),
        p25=float(np.percentile(arr, 25)), p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
    )


# --------------------------------------------------------------------------- #
# Shared setup: build env + policy, warmup, reset episode clock
# --------------------------------------------------------------------------- #
def build_env_and_policy(cli, num_envs):
    """Mirror sweep.main()'s build path exactly (isolation + freeze), minus the
    grid tiling. Returns (env, policy, chosen_run, ckpt_path)."""
    if SIMULATOR == "genesis":  # noqa: F405
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level="warning")  # noqa: F405

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)

    # For push_recovery we FREEZE the command with override_cfg_for_eval's collapsed
    # ranges (it is held constant, so this is exactly the sweep behaviour). For
    # step_response the command must CHANGE across phases, so we still call
    # override_cfg_for_eval to kill curriculum/heading/push and disable resampling,
    # then overwrite env.commands ourselves each step (see run_step_response).
    cli.command_vx = getattr(cli, "command_vx", 0.5)
    override_cfg_for_eval(env_cfg, cli, num_envs)

    # Disable in-episode command resampling for the whole window: we drive commands
    # externally. Pushing resampling_time far past the window means _resample_commands
    # gets an empty env_ids set every measured step, so it never overwrites our
    # scheduled command (curriculum/heading are already off from override_cfg).
    env_cfg.commands.resampling_time = 1e6

    # resolve the checkpoint BEFORE building, restricted to this task's run_name so a
    # sibling method's model can never be loaded by accident (sweep.resolve_load_run).
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

    # optional off-nominal physics point via the sweep's axis setters
    if cli.axis is not None:
        if cli.axis_value is None:
            raise SystemExit("--axis requires --axis_value")
        axis = get_axis(cli.axis)
        pin_others_to_nominal(env, cli.axis)   # isolate: all other axes at nominal
        vals = torch.full((env.num_envs,), float(cli.axis_value), device=env.device)
        axis.setter(env, vals)
    else:
        # Even with no chosen axis, pin the whole privileged vector to nominal so the
        # oracle reads clean labels (some sim buffers hold nonzero stale build values).
        pin_others_to_nominal(env, "friction")   # pins added_mass + com; friction nominal set below
        axis = get_axis("friction")
        axis.setter(env, torch.full((env.num_envs,), 1.0, device=env.device))

    return env, policy, chosen_run, ckpt_path


def warmup_and_reset_clock(env, policy, cli, hold_cmd):
    """Settle the robots under `hold_cmd`, then zero the episode clock.

    Same trick as sweep.main(): warmup steps are unrecorded so the scenario starts
    from a settled gait, and zeroing `episode_length_buf` keeps the settled physics
    state while restarting the episode timer -- otherwise the timeout (max_episode
    _length) would fire `warmup` steps early and truncate the recovery window.
    """
    hold = torch.tensor(hold_cmd, dtype=torch.float, device=env.device)
    env.reset()
    obs = env.get_observations()
    for _ in range(cli.warmup):
        env.commands[:, :3] = hold          # keep the settling command fixed
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
    env.episode_length_buf.zero_()
    return obs


# --------------------------------------------------------------------------- #
# Scenario 1: step_response
# --------------------------------------------------------------------------- #
def run_step_response(env, policy, cli):
    schedule, bounds = build_command_schedule(cli)
    T = schedule.shape[0]
    sched_t = torch.tensor(schedule, dtype=torch.float, device=env.device)  # (T, 3)

    # settle standing (first phase is stand), then measure
    obs = warmup_and_reset_clock(env, policy, cli, hold_cmd=list(schedule[0]))

    rec = TransientRecorder(T, env.num_envs, env.device)
    for t in range(T):
        # overwrite the command for THIS step so the obs computed at the end of this
        # step (which the next action consumes) reflects the scheduled command. Same
        # row for every env -> a deterministic, identical schedule across all envs.
        env.commands[:, :3] = sched_t[t]
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        rec.record(env)

    dt = env.dt
    # global metrics over the whole schedule
    per_env = {
        "err_integral": rec.error_integral(dt),
        "peak_err": rec.peak_error(),
        "peak_tilt": rec.peak_tilt(),
        "fall": rec.fall_rate(),
    }
    # per-phase settling time + peak error, sliced from each command change onward
    phase_names = [name for name, _ in bounds]
    for i, (name, start) in enumerate(bounds):
        stop = bounds[i + 1][1] if i + 1 < len(bounds) else T
        per_env[f"settle_{name}"] = rec.settling_time(cli.tol_lin, lo=start, hi=stop)
        per_env[f"peakerr_{name}"] = rec.peak_error(lo=start, hi=stop)

    extra = dict(
        schedule=schedule, phase_names=np.asarray(phase_names),
        phase_bounds=np.asarray([b for _, b in bounds]),
        phase_steps=cli.phase_steps, T=T,
        # per-step mean tracking-error time series (across envs) for the plotter
        lin_err_ts=rec.lin_err.mean(dim=1).cpu().numpy(),
        tilt_ts=rec.tilt.mean(dim=1).cpu().numpy(),
    )
    return per_env, extra


# --------------------------------------------------------------------------- #
# Scenario 2: push_recovery  (deterministic, RNG-matched)
# --------------------------------------------------------------------------- #
def apply_fixed_impulse(env, dvx: float, dvy: float):
    """Add the SAME constant velocity impulse to every env's base.

    Mirrors genesis_simulator.push_robots() -- the base link is a 6-DOF free joint,
    so dof indices [0:3] are base world linear velocity -- but with a CONSTANT delta
    instead of a per-env random draw:

        dofs_vel = robot.get_dofs_velocity(); dofs_vel[:,0]+=dvx; dofs_vel[:,1]+=dvy

    WHY DETERMINISTIC MATTERS (do not "improve" this into a random push): the methods
    have different observation dims (45 vs 50) and therefore consume a DIFFERENT
    number of RNG draws per policy forward pass. A *random* push (`torch_rand_float`)
    would draw from an RNG stream that is already desynchronised across methods, so
    each method would get a different push -> an unfair comparison confounding
    "recovers better" with "got a smaller shove". A fixed, pre-scheduled impulse
    applied to ALL envs at the SAME step is identical across methods by construction,
    isolating recovery quality. We also apply it at a fixed step index, not a random
    interval, for the same reason.
    """
    sim = env.simulator
    robot = sim._robot
    dofs_vel = robot.get_dofs_velocity()   # (num_envs, num_dof); [0:3] = base lin vel (world)
    dofs_vel[:, 0] += dvx
    dofs_vel[:, 1] += dvy
    robot.set_dofs_velocity(dofs_vel)
    # keep the cached base_lin_vel consistent with the sim (as push_robots does)
    from legged_gym.utils.math_utils import quat_rotate_inverse
    sim._base_lin_vel[:] = quat_rotate_inverse(sim._base_quat, robot.get_vel())


def run_push_recovery(env, policy, cli):
    hold = [cli.command_vx, cli.command_vy, cli.command_yaw]
    obs = warmup_and_reset_clock(env, policy, cli, hold_cmd=hold)
    hold_t = torch.tensor(hold, dtype=torch.float, device=env.device)

    T = cli.pre_push_steps + cli.recovery_steps
    push_step = cli.pre_push_steps    # impulse applied at this measured step (fixed)
    rec = TransientRecorder(T, env.num_envs, env.device)

    for t in range(T):
        env.commands[:, :3] = hold_t                     # hold the forward command
        if t == push_step:
            apply_fixed_impulse(env, cli.dvx, cli.dvy)   # SAME impulse, all envs, same step
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        rec.record(env)

    dt = env.dt
    lo = push_step   # recovery window starts at the push
    per_env = {
        # recovery-error integral over the post-push window
        "recover_err_integral": rec.error_integral(dt, lo=lo),
        # recovery time: steps after the push until error settles below tol
        "recovery_time": rec.settling_time(cli.tol_lin, lo=lo),
        "peak_tilt": rec.peak_tilt(lo=lo),
        "peak_err": rec.peak_error(lo=lo),
        # fall rate WITHIN the recovery window
        "fall": rec.fall_rate(),
        # baseline (pre-push) error for reference / sanity
        "pre_push_err": rec.lin_err[:lo].mean(dim=0) if lo > 0 else rec.lin_err[:1].mean(dim=0),
    }
    extra = dict(
        push_step=push_step, T=T, dvx=cli.dvx, dvy=cli.dvy,
        pre_push_steps=cli.pre_push_steps, recovery_steps=cli.recovery_steps,
        lin_err_ts=rec.lin_err.mean(dim=1).cpu().numpy(),
        tilt_ts=rec.tilt.mean(dim=1).cpu().numpy(),
    )
    return per_env, extra


# --------------------------------------------------------------------------- #
# Provenance + save
# --------------------------------------------------------------------------- #
def collect_run_meta(cli, chosen_run, ckpt_path):
    try:
        genesis_version = gs.__version__  # noqa: F405
    except Exception:
        genesis_version = "unknown"
    return dict(
        scenario=cli.scenario, warmup=cli.warmup, load_run=chosen_run,
        ckpt_path=ckpt_path, ckpt=cli.ckpt, simulator=str(SIMULATOR),  # noqa: F405
        git_commit=git_commit(), python_version=sys.version.split()[0],
        torch_version=torch.__version__, numpy_version=np.__version__,
        genesis_version=str(genesis_version), seed=cli.seed, per_point=cli.per_point,
        tol_lin=cli.tol_lin, axis="" if cli.axis is None else cli.axis,
        axis_value=float("nan") if cli.axis_value is None else cli.axis_value,
    )


def main():
    cli = parse_args()
    label = cli.label or cli.task
    num_envs = cli.per_point   # one window per env; per_point replicas == seeds

    env, policy, chosen_run, ckpt_path = build_env_and_policy(cli, num_envs)

    if cli.scenario == "step_response":
        per_env, extra = run_step_response(env, policy, cli)
    else:
        per_env, extra = run_push_recovery(env, policy, cli)

    # aggregate every per-env metric to mean/std/quantiles across the replicas
    agg = {}
    for key, vec in per_env.items():
        for stat, val in agg_stats(vec).items():
            agg[f"{key}_{stat}"] = val

    out = cli.out or os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "eval", f"{cli.scenario}_{label}.npz")  # noqa: F405
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = collect_run_meta(cli, chosen_run, ckpt_path)
    np.savez(out, label=label, **meta, **extra,
             **{k: np.asarray(v) for k, v in agg.items()})

    # --- human-readable summary ---
    print(f"\n=== {label} | scenario={cli.scenario} | {cli.per_point} envs (seeds) ===")
    if cli.axis is not None:
        print(f"physics point: {cli.axis} = {cli.axis_value}")
    if cli.scenario == "step_response":
        print(f"schedule: stand->forward(vx={cli.step_vx})->reverse(vx={-cli.step_vx})"
              f"->lateral(vy={cli.step_vy})->stop, {cli.phase_steps} steps/phase")
        print(f"\n{'metric':>20} {'mean':>9} {'std':>9} {'p50':>9}")
        order = ["err_integral", "peak_err", "peak_tilt", "fall"]
        order += [f"settle_{n}" for n in ["stand", "forward", "reverse", "lateral", "stop"]]
        for key in order:
            print(f"{key:>20} {agg[key+'_mean']:9.3f} {agg[key+'_std']:9.3f} {agg[key+'_p50']:9.3f}")
    else:
        print(f"held command: (vx,vy,yaw)=({cli.command_vx},{cli.command_vy},{cli.command_yaw})")
        print(f"DETERMINISTIC impulse (identical across ALL envs & methods): "
              f"dvx={cli.dvx:+.3f}  dvy={cli.dvy:+.3f} [m/s]  @ measured step {extra['push_step']}")
        print(f"\n{'metric':>22} {'mean':>9} {'std':>9} {'p50':>9}")
        for key in ["pre_push_err", "recover_err_integral", "recovery_time",
                    "peak_err", "peak_tilt", "fall"]:
            print(f"{key:>22} {agg[key+'_mean']:9.3f} {agg[key+'_std']:9.3f} {agg[key+'_p50']:9.3f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
