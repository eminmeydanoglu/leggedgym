"""Sweep one OOD axis over a grid and collect distributional metrics.

Runs ONE method (task + checkpoint) with the frozen benchmark protocol, tiles the
grid across the parallel envs (one value per contiguous block => `per_point` envs
== `per_point` seeds), runs `steps` steps with a fixed forward command, then bins
per-env metrics back by grid value. Result is saved as an .npz consumed by
`plot_sweep.py`.

Example (on the GPU box, env activated):
    python legged_gym/scripts/eval/sweep.py \
        --task go2_bench_mlp --load_run <run> --axis friction \
        --per_point 256 --steps 1000 --out logs/eval/friction_mlp.npz

Design notes:
  * only the swept axis varies across envs; base-mass / com / push randomization
    are turned OFF so the degradation curve is purely the axis's effect.
  * a fixed forward command (default 0.5 m/s) makes tracking error meaningful;
    command ranges are collapsed to a point so resampling never perturbs it.
  * auto_reset stays ON so we get multiple episodes per env -> real distributions.
"""

import argparse
import os
import subprocess
import sys
from types import SimpleNamespace

from legged_gym import *  # noqa: F401,F403  (exposes gs, SIMULATOR, LEGGED_GYM_ROOT_DIR)
from legged_gym.envs import *  # noqa: F401,F403  (registers tasks)
from legged_gym.utils import task_registry

import numpy as np
import torch

from legged_gym.scripts.eval.dr_axes import get_axis, pin_others_to_nominal
from legged_gym.scripts.eval.metrics import MetricAccumulator


def parse_args():
    p = argparse.ArgumentParser(description="OOD single-axis sweep for the eval harness")
    p.add_argument("--task", type=str, required=True, help="registered benchmark task, e.g. go2_bench_mlp")
    p.add_argument("--load_run", type=str, default=None, help="run dir to load (default: latest)")
    p.add_argument("--ckpt", type=int, default=-1, help="checkpoint iteration (-1 = latest)")
    p.add_argument("--axis", type=str, default="friction", help="OOD axis name (see dr_axes.AXES)")
    p.add_argument("--grid", type=float, nargs="+", default=None, help="override the axis grid values")
    p.add_argument("--per_point", type=int, default=256, help="parallel env replicas per grid point")
    p.add_argument("--steps", type=int, default=2000,
                   help="measured steps. NOTE: timeout fires at max_episode_length+1 "
                        "(1001 for the 20s/0.02dt default), so a full-survival policy "
                        "logs its FIRST complete episode only at step 1001 -- with "
                        "steps<=1000 `mean_return` stays 0. Use >=1001 (default 2000 "
                        "captures a full episode plus margin; go higher for tighter "
                        "return/fall distributions).")
    p.add_argument("--warmup", type=int, default=100, help="unrecorded settling steps before measuring")
    p.add_argument("--command_vx", type=float, default=0.5, help="fixed forward command [m/s]")
    p.add_argument("--command_vy", type=float, default=0.0, help="fixed lateral command [m/s]")
    p.add_argument("--command_yaw", type=float, default=0.0, help="fixed yaw-rate command [rad/s]")
    p.add_argument("--seed", type=int, default=1, help="global seed (repeat runs to add more seeds)")
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--out", type=str, default=None, help="output .npz path")
    p.add_argument("--label", type=str, default=None, help="curve label (default: task name)")
    return p.parse_args()


def resolve_load_run(root, run_name, load_run):
    """Pick the checkpoint run folder, isolating this task from sibling methods.

    All benchmark methods share `experiment_name='go2_benchmark'`, so the vanilla
    `get_load_path` "latest folder in experiment root" would happily hand a
    go2_bench_mlp eval an oracle checkpoint (50-dim obs vs 45-dim model) -> silent
    wrong-model load / obs-size crash. We restrict candidates to folders that carry
    this task's `run_name`, and fail loud if none match.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No experiment log dir: {root}")
    runs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d != "exported"
    )
    if not runs:
        raise FileNotFoundError(f"No runs in {root}")

    if load_run is not None:
        if load_run not in runs:
            raise FileNotFoundError(
                f"--load_run '{load_run}' not found in {root}. Available: {runs}"
            )
        if run_name not in load_run:
            print(f"[warn] --load_run '{load_run}' does not carry this task's "
                  f"run_name '{run_name}'. Loading another method's checkpoint will "
                  f"fail on an obs-size mismatch -- double-check this is intentional.")
        return load_run

    matching = [d for d in runs if run_name in d]
    if not matching:
        raise FileNotFoundError(
            f"No run matching run_name '{run_name}' in {root}. Available: {runs}. "
            f"Pass --load_run explicitly."
        )
    # pick the calendar-latest by mtime, not by string sort: the `MonDD_HH-MM-SS`
    # folder prefix does not sort chronologically across month/day boundaries.
    chosen = max(matching, key=lambda d: os.path.getmtime(os.path.join(root, d)))
    print(f"[eval] auto-selected latest '{run_name}' run: {chosen}")
    return chosen


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=LEGGED_GYM_ROOT_DIR,  # noqa: F405
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def collect_run_meta(cli, chosen_run, ckpt_path):
    """Provenance for the published artifact: what exactly produced these numbers."""
    try:
        genesis_version = gs.__version__  # noqa: F405
    except Exception:
        genesis_version = "unknown"
    return dict(
        warmup=cli.warmup,
        load_run=chosen_run,
        ckpt_path=ckpt_path,
        ckpt=cli.ckpt,
        simulator=str(SIMULATOR),  # noqa: F405
        git_commit=git_commit(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        genesis_version=str(genesis_version),
    )


def make_registry_args(cli):
    """Build the arg namespace task_registry expects (mirrors helpers.get_args fields)."""
    return SimpleNamespace(
        task=cli.task, headless=True, cpu=cli.cpu, num_envs=None, max_iterations=None,
        resume=True, sync_wandb=False, export_onnx=False, debug=False,
        load_run=cli.load_run, ckpt=cli.ckpt, use_joystick=False, joystick_type="xbox",
        follow_robot=False, viewer="native", viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )


def override_cfg_for_eval(env_cfg, cli, num_envs):
    """Freeze everything except the swept axis; fix a forward command."""
    env_cfg.env.num_envs = num_envs
    env_cfg.env.auto_reset = True
    env_cfg.env.debug = False
    env_cfg.seed = cli.seed

    # isolate the swept axis: no other physics randomization
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.push_robots = False
    # friction is written per-env after build; drop its build-time random draw
    env_cfg.domain_rand.randomize_friction = False

    # constant forward command: collapse ranges to a point so resampling is a no-op.
    # curriculum MUST be off -- a trained policy passes the tracking threshold and the
    # command curriculum would silently widen lin_vel_x back to [0, max] mid-run,
    # un-freezing the "fixed" command and corrupting tracking/return metrics.
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.ranges.lin_vel_x = [cli.command_vx, cli.command_vx]
    env_cfg.commands.ranges.lin_vel_y = [cli.command_vy, cli.command_vy]
    env_cfg.commands.ranges.ang_vel_yaw = [cli.command_yaw, cli.command_yaw]
    if hasattr(env_cfg.commands.ranges, "heading"):
        env_cfg.commands.ranges.heading = [0.0, 0.0]


def aggregate(per_env: dict, num_points: int, per_point: int) -> dict:
    """Bin per-env metric vectors -> per-grid-point mean/std/quantiles."""
    out = {}
    for key, vec in per_env.items():
        arr = vec.detach().cpu().numpy().reshape(num_points, per_point)
        out[f"{key}_mean"] = arr.mean(axis=1)
        out[f"{key}_std"] = arr.std(axis=1)
        out[f"{key}_p25"] = np.percentile(arr, 25, axis=1)
        out[f"{key}_p50"] = np.percentile(arr, 50, axis=1)
        out[f"{key}_p75"] = np.percentile(arr, 75, axis=1)
    return out


def main():
    cli = parse_args()
    axis = get_axis(cli.axis)
    grid = np.asarray(cli.grid, dtype=np.float64) if cli.grid else axis.grid()
    num_points = len(grid)
    num_envs = num_points * cli.per_point
    label = cli.label or cli.task

    if SIMULATOR == "genesis":  # noqa: F405
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level="warning")  # noqa: F405

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)
    override_cfg_for_eval(env_cfg, cli, num_envs)

    # resolve the checkpoint BEFORE building anything, restricted to this task's
    # run_name so a sibling method's model can never be loaded by accident.
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)  # noqa: F405
    chosen_run = resolve_load_run(log_root, train_cfg.runner.run_name, cli.load_run)
    from legged_gym.utils.helpers import get_load_path
    ckpt_path = get_load_path(log_root, load_run=chosen_run, checkpoint=cli.ckpt)

    cli.load_run = chosen_run  # feed the resolved folder into the registry args
    reg_args = make_registry_args(cli)
    env, _ = task_registry.make_env(name=cli.task, args=reg_args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cli.task, args=reg_args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # tile the grid across env blocks: env i in block (i // per_point)
    per_env_vals = torch.tensor(
        np.repeat(grid, cli.per_point), dtype=torch.float, device=env.device
    )
    pin_others_to_nominal(env, cli.axis)  # isolate: all other axes at nominal
    axis.setter(env, per_env_vals)

    env.reset()  # clean start; get_observations() would otherwise return zero-init obs
    obs = env.get_observations()

    # warmup: let robots settle under the fixed command, don't record
    for _ in range(cli.warmup):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

    # Restart the episode clock without teleporting the (now-settled) robots.
    # Otherwise the env's episode_length_buf is already ~warmup steps in, so the
    # first measured episode times out `warmup` steps early -> mean_ep_len caps at
    # steps-warmup (the observed 900 with warmup=100/steps=1000) and its return is
    # only partially counted. Zeroing the clock keeps the settled physics state but
    # makes measurement cover full-length episodes.
    env.episode_length_buf.zero_()

    acc = MetricAccumulator(num_envs, env.device)
    for _ in range(cli.steps):
        actions = policy(obs.detach())
        obs, _, rew, dones, _ = env.step(actions.detach())
        cmd = env.commands
        v = env.simulator.base_lin_vel
        w = env.simulator.base_ang_vel
        lin_err = torch.norm(cmd[:, :2] - v[:, :2], dim=1)
        ang_err = torch.abs(cmd[:, 2] - w[:, 2])
        acc.update(
            rew, dones, env.time_out_buf, lin_err, ang_err,
            tilt=torch.norm(env.simulator.projected_gravity[:, :2], dim=1),
            torques=env.simulator.torques,
            dof_vel=env.simulator.dof_vel,
            actions=env.actions,
            last_actions=env.last_actions,
            feet_vel=env.simulator.feet_vel,
            feet_contact=env.feet_max_force_z > 1.0,
        )

    per_env = acc.compute()
    agg = aggregate(per_env, num_points, cli.per_point)

    out = cli.out or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "eval", f"{cli.axis}_{label}.npz")  # noqa: F405
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = collect_run_meta(cli, chosen_run, ckpt_path)
    np.savez(
        out, axis=cli.axis, grid=grid, in_dist=np.asarray(axis.in_dist),
        unit=axis.unit, label=label, per_point=cli.per_point, steps=cli.steps,
        command_vx=cli.command_vx, command_vy=cli.command_vy,
        command_yaw=cli.command_yaw, seed=cli.seed, **meta, **agg,
    )

    # human-readable summary to stdout
    print(f"\n=== {label} | axis={cli.axis} | {cli.per_point} envs/pt x {cli.steps} steps ===")
    print(f"{'grid':>8} {'fall_rate':>10} {'falls/1k':>10} {'ep_len':>8} {'lin_err':>9} {'return':>9}")
    for i, g in enumerate(grid):
        print(f"{g:8.3f} {agg['fall_rate_mean'][i]:10.3f} {agg['falls_per_1k_mean'][i]:10.2f} "
              f"{agg['mean_ep_len_mean'][i]:8.1f} {agg['tracking_lin_err_mean'][i]:9.3f} "
              f"{agg['mean_return_mean'][i]:9.2f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
