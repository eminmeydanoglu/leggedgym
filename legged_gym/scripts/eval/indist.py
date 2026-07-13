"""In-distribution evaluation: the "classic" eval the OOD sweep is NOT.

`sweep.py` answers "how fast does the policy degrade as ONE physics axis leaves
the training band" (a single forward command, all other DR pinned to nominal).
This module answers the complementary, everyday question: **how good is the
learned gait inside the training distribution** -- the full command set and the
in-distribution DR band the policy was actually trained on.

It exists for two consumers, both driving off the SAME rollout code so their
numbers mean the same thing:

  1. `OnPolicyRunner` calls `run_indist_eval` periodically during training to log
     `Eval/*` and pick `best_tracking.pt` with the deterministic tracking and
     safety rule used by Eval V2 (not noisy, curriculum-drifting reward).
  2. `main()` here is the standalone after-training harness (the in-dist sibling
     of `sweep.py`): load a checkpoint, run the same eval, save an `.npz`.

Protocol (why the numbers are comparable across checkpoints):
  * Deterministic policy -- `act_inference` (mean action, no exploration noise).
  * Command ranges pinned to the FULL configured in-dist ranges and the command
    curriculum frozen, so the eval task does not drift wider as training advances.
  * A fixed RNG seed around the rollout -> identical command draws and obs noise
    on every call within a run (build-time DR is already fixed per env, so the
    per-env friction/mass/CoM sample is held constant too).
  * `auto_reset` stays ON so each env yields many episodes -> real distributions.

Metrics are `MetricAccumulator`'s (fall_rate / mean_ep_len / falls_per_1k /
mean_return / tracking_lin_err / tracking_ang_err), here reduced to scalars by
averaging over envs.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import numpy as np
import torch

from legged_gym.scripts.eval.metrics import MetricAccumulator


def in_dist_command_ranges(env) -> Dict[str, list]:
    """The fixed, full in-distribution command ranges, read from the env config.

    `env.command_ranges` is a live dict the curriculum narrows/widens over
    training; `env.cfg.commands.ranges` holds the original configured (curriculum
    target) values, which is exactly the frozen protocol we pin eval to.
    """
    r = env.cfg.commands.ranges
    ranges = {
        "lin_vel_x": list(r.lin_vel_x),
        "lin_vel_y": list(r.lin_vel_y),
        "ang_vel_yaw": list(r.ang_vel_yaw),
    }
    if hasattr(r, "heading"):
        ranges["heading"] = list(r.heading)
    return ranges


@torch.no_grad()
def run_indist_eval(
    env,
    policy: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int = 1100,
    warmup: int = 50,
    seed: int = 12345,
    command_ranges: Optional[Dict[str, list]] = None,
) -> Dict[str, float]:
    """Run one in-distribution eval rollout and return env-averaged metrics.

    Non-destructive: the env's live command ranges, curriculum flag and the global
    RNG state are saved on entry and restored on exit (even on error), so a caller
    mid-training can resume without side effects. The caller is still responsible
    for `env.reset()`-ing and refreshing its own observation handles afterwards --
    this function leaves the env freshly reset.

    Args:
        env:     a built legged-gym env (auto_reset expected ON).
        policy:  deterministic inference policy, obs -> actions (e.g. act_inference).
        steps:   measured steps. MUST be >= max_episode_length + 1 (1001 for the
                 20s/0.02dt default) or a full-survival policy never completes an
                 episode and `mean_return` reads 0. Default 1100 = one full episode
                 + margin; raise for tighter return/fall distributions.
        warmup:  unrecorded settling steps before measurement.
        seed:    fixed seed applied around the rollout for cross-checkpoint parity.
        command_ranges: override the pinned ranges (default: `in_dist_command_ranges`).

    Returns:
        dict of scalar metrics. ``fall_rate`` is the V2 fixed-window event rate
        (an environment counts once if it falls at least once); the legacy
        completed-episode rate remains available as ``episode_fall_rate``.
    """
    device = env.device

    # --- freeze the eval protocol: fixed command ranges, no curriculum drift ---
    saved_ranges = {k: list(v) for k, v in env.command_ranges.items()}
    saved_curriculum = env.cfg.commands.curriculum
    pin = command_ranges or in_dist_command_ranges(env)
    for k, v in pin.items():
        if k in env.command_ranges:
            env.command_ranges[k] = list(v)
    env.cfg.commands.curriculum = False

    # --- fixed seed so command draws + obs noise are identical across checkpoints;
    #     build-time per-env DR (friction/mass/CoM) is already fixed for the run ---
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    try:
        env.reset()
        obs = env.get_observations().to(device)

        # warmup: settle under the pinned commands, don't record
        for _ in range(warmup):
            obs, _, _, _, _ = env.step(policy(obs.detach()))
            obs = obs.to(device)

        # restart the episode clock WITHOUT teleporting the settled robots, so the
        # first measured episode is full-length (mirrors sweep.py); otherwise it
        # times out `warmup` steps early and its return is only partially counted.
        env.episode_length_buf.zero_()

        acc = MetricAccumulator(env.num_envs, device)
        for _ in range(steps):
            obs, _, rew, dones, _ = env.step(policy(obs.detach()))
            obs = obs.to(device)
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
        metrics = {k: float(t.mean().item()) for k, t in per_env.items()}
        # Eval V2 defines safety over the entire fixed observation window rather
        # than over a policy-dependent number of auto-reset episodes.
        metrics["episode_fall_rate"] = metrics["fall_rate"]
        metrics["fall_rate"] = metrics["ever_fell"]
    finally:
        # restore training state no matter what
        for k, v in saved_ranges.items():
            env.command_ranges[k] = v
        env.cfg.commands.curriculum = saved_curriculum
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        np.random.set_state(np_state)

    # clean episode boundary: reset under the RESTORED (training) command ranges so
    # the caller resumes from fresh episodes, not the pinned wide-command eval state.
    env.reset()
    return metrics


def tracking_score(metrics: Dict[str, float]) -> float:
    """Eval V2 tracking score for the shared [-1, 1] command field.

    The linear error is normalized by the maximum planar command magnitude
    ``sqrt(2)`` and yaw error by its unit maximum. Lower is better.
    """
    return 0.5 * (float(metrics["tracking_lin_err"]) / np.sqrt(2.0)
                  + float(metrics["tracking_ang_err"]))


def tracking_selection_key(
    metrics: Dict[str, float], iteration: int, fall_threshold: float,
) -> tuple[float, ...]:
    """Lexicographic Eval V2 checkpoint key (lower is better).

    Safe checkpoints always win. Among safe candidates tracking is decisive;
    if none are safe, minimize fall rate then tracking. Earlier iterations make
    a deterministic final tie-breaker.
    """
    fall = float(metrics["fall_rate"])
    score = tracking_score(metrics)
    if fall <= float(fall_threshold):
        return (0.0, score, float(iteration))
    return (1.0, fall, score, float(iteration))


def selection_score(metrics: Dict[str, float], fall_guard: float) -> float:
    """Backward-compatible scalar diagnostic; do not use for V2 selection."""
    return -tracking_score(metrics) - 1e4 * max(0.0, metrics["fall_rate"] - fall_guard)


# =============================================================================
# Standalone after-training harness (in-dist sibling of sweep.py)
# =============================================================================

def _build_and_eval(cli):
    from types import SimpleNamespace

    from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR  # noqa: F401
    import legged_gym.envs  # noqa: F401  (import side effect: registers tasks)
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import get_load_path
    from legged_gym.scripts.eval.sweep import resolve_load_run

    try:
        import genesis as gs  # noqa: F401
    except Exception:
        gs = None

    if SIMULATOR == "genesis" and gs is not None:
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level="warning")

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)

    # in-dist eval: keep the training DR band ON (unlike sweep, which zeroes it);
    # only fix num_envs / seed / auto_reset. Commands are pinned inside the rollout.
    if cli.num_envs is not None:
        env_cfg.env.num_envs = cli.num_envs
    env_cfg.env.auto_reset = True
    env_cfg.env.debug = False
    env_cfg.seed = cli.seed

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    chosen_run = resolve_load_run(log_root, train_cfg.runner.run_name, cli.load_run)
    from legged_gym.scripts.eval.provenance import verify_run_identity
    import re
    _sm = re.search(r"_seed(\d+)$", chosen_run)
    verify_run_identity(
        os.path.join(log_root, chosen_run),
        expected_task=cli.task,
        expected_run_name=train_cfg.runner.run_name,
        expected_training_seed=int(_sm.group(1)) if _sm else None,
    )
    ckpt_path = get_load_path(log_root, load_run=chosen_run, checkpoint=cli.ckpt)
    print(f"[eval] loading checkpoint: {ckpt_path} (spec={cli.ckpt!r})")

    reg_args = SimpleNamespace(
        task=cli.task, headless=True, cpu=cli.cpu, num_envs=None, max_iterations=None,
        resume=True, sync_wandb=False, export_onnx=False, debug=False,
        load_run=chosen_run, ckpt=cli.ckpt, use_joystick=False, joystick_type="xbox",
        follow_robot=False, viewer="native", viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )
    env, _ = task_registry.make_env(name=cli.task, args=reg_args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=cli.task, args=reg_args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    metrics = run_indist_eval(env, policy, steps=cli.steps, warmup=cli.warmup, seed=cli.seed)
    return metrics, chosen_run, ckpt_path, log_root, int(env.num_obs)


def main():
    import argparse

    from legged_gym.scripts.eval.ckpt_utils import parse_ckpt_cli
    from legged_gym.scripts.eval.provenance import atomic_savez, collect_eval_meta

    p = argparse.ArgumentParser(description="In-distribution eval for the benchmark harness")
    p.add_argument("--task", type=str, required=True, help="registered task, e.g. go2_bench_mlp")
    p.add_argument("--load_run", type=str, default=None, help="run dir to load (default: latest matching)")
    p.add_argument("--ckpt", type=parse_ckpt_cli, default=-1,
                   help="checkpoint: 'best', 'latest'/-1, or iteration int (e.g. 3000)")
    p.add_argument("--num_envs", type=int, default=None, help="override env count (default: cfg value)")
    p.add_argument("--steps", type=int, default=2000,
                   help="measured steps (>= max_episode_length+1; default 2000 for tight distributions)")
    p.add_argument("--warmup", type=int, default=100, help="unrecorded settling steps")
    p.add_argument("--seed", type=int, default=12345, help="fixed eval seed")
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--out", type=str, default=None, help="output .npz path")
    p.add_argument("--label", type=str, default=None, help="method label (default: task)")
    cli = p.parse_args()

    metrics, chosen_run, ckpt_path, log_root, num_obs = _build_and_eval(cli)

    from legged_gym import LEGGED_GYM_ROOT_DIR
    out = cli.out or os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "eval", f"indist_{cli.task}.npz")
    meta = collect_eval_meta(
        task=cli.task,
        method=cli.label or cli.task,
        chosen_run=chosen_run,
        log_root=log_root,
        ckpt_spec=cli.ckpt,
        ckpt_path=ckpt_path,
        seed=cli.seed,
        warmup=cli.warmup,
        steps=cli.steps,
        per_point=cli.num_envs,
        extra=dict(num_obs=num_obs, seed=cli.seed),
    )
    atomic_savez(out, **meta, **metrics)

    print(f"\n=== in-dist eval | {cli.task} | run={chosen_run} | {cli.steps} steps ===")
    print(f"ckpt: {ckpt_path}")
    for k in ("mean_return", "fall_rate", "falls_per_1k", "mean_ep_len",
              "tracking_lin_err", "tracking_ang_err", "never_fell"):
        print(f"{k:>18}: {metrics[k]:.4f}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
