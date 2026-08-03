"""Measurement script (NOT a pytest test): Genesis contact-force noise floor
on the go2_moects penalized links (thigh/calf), to justify the collision
reward threshold (reference PhysX value 0.1 N, host default 10.0 N).

Builds the real ``go2_moects`` env on a FLAT plane (moe_grid off), pushes off,
no episode timeouts, and logs ``env.penalized_bodies_force_norm`` (the exact
tensor ``_reward_collision`` consumes) plus ``env.feet_force_norm`` as a
calibration reference.

Scenarios:
  A  quasi-static noise floor: zero actions (PD holds default pose), 1000 steps
  B  dynamic noise floor: small random actions U(-0.2, 0.2) rad re-drawn every
     10 control steps (walking-adjacent jitter, no deliberate thigh contact)
  C  genuine-contact signal: scripted crouch (thigh target 2.2 rad, calf
     target -2.8 rad) so thigh/calf links rest on the ground; report LOW
     percentiles of the genuinely-contacting cluster (per-step max over
     penalized links > 1.0 N).

Aggregates are saved to tmp/collision_force_stats.json and printed as a table.

Run:
    SIMULATOR=genesis .venv/bin/python tests/_measure_collision_force_noise.py
"""

import json
import os
import sys
from types import SimpleNamespace

import torch

NUM_ENVS = 64
SETTLE_STEPS_A = 100
RECORD_STEPS_A = 1000
SETTLE_STEPS_B = 100
RECORD_STEPS_B = 1000
REDRAW_EVERY_B = 10
SETTLE_STEPS_C = 300
RECORD_STEPS_C = 700
POST_RESET_BLACKOUT = 25  # steps excluded after an env reset (respawn transient)

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
OUT_JSON = os.path.join(ROOT, "tmp", "collision_force_stats.json")


def make_flat_env():
    import genesis as gs
    import legged_gym.envs  # noqa: F401  (registers go2_moects)
    from legged_gym.utils import task_registry

    gs.init(backend=gs.gpu, logging_level="warning")
    cfg, _ = task_registry.get_cfgs("go2_moects")
    cfg.env.num_envs = NUM_ENVS
    cfg.env.episode_length_s = 1e6  # no timeouts during measurement
    # flat noise-floor terrain: moe_grid requires heightfield, so switch both off
    cfg.terrain.moe_grid = False
    cfg.terrain.mesh_type = "plane"
    # keep external perturbations out of the noise floor (DR at reset stays on)
    cfg.domain_rand.push_robots = False
    args = SimpleNamespace(
        task="go2_moects", seed=7, debug=False, headless=True, cpu=False,
        num_envs=NUM_ENVS, max_iterations=None, resume=False,
        sync_wandb=False, ckpt=None, load_run=None, export_onnx=False,
        motion_file=None, num_student=None)
    env, _ = task_registry.make_env("go2_moects", args=args, env_cfg=cfg)
    return env


def run_scenario(env, action_fn, settle_steps, record_steps):
    """Drive the env; return (pen_norms, feet_norms, post_reset_mask) stacked
    over the recorded steps only. All on CPU float64-ready (kept float32)."""
    pen, feet, blackout = [], [], []
    # explicit CPU: the process default device may be cuda once genesis is up
    last_reset_step = torch.full((env.num_envs,), -10**9, dtype=torch.long,
                                 device="cpu")
    total_steps = settle_steps + record_steps
    resets_total = 0
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for step in range(total_steps):
        actions = action_fn(step, actions)
        _, _, _, _, _, resets, _ = env.step(actions)
        resets = resets.bool()
        resets_total += int(resets.sum().item())
        if resets.any():
            last_reset_step[resets.cpu()] = step
        if step >= settle_steps:
            pen.append(env.penalized_bodies_force_norm.detach().cpu().clone())
            feet.append(env.feet_force_norm.detach().cpu().clone())
            blackout.append((step - last_reset_step) < POST_RESET_BLACKOUT)
    return (torch.stack(pen).cpu(), torch.stack(feet).cpu(),
            torch.stack(blackout).cpu(), resets_total)


def stats(t, mask=None):
    """t: (steps, envs, links) force norms. mask: (steps, envs) True = exclude."""
    t = t.detach().cpu()
    if mask is not None:
        t = t[~mask.cpu().unsqueeze(-1).expand_as(t)].reshape(-1, t.shape[-1])
    flat = t.reshape(-1).double()
    n = flat.numel()
    return {
        "n_samples": n,
        "frac_exactly_zero": float((flat == 0.0).double().mean()),
        "mean": float(flat.mean()),
        "p50": float(torch.quantile(flat, 0.5)),
        "p99": float(torch.quantile(flat, 0.99)),
        "p99_9": float(torch.quantile(flat, 0.999)),
        "max": float(flat.max()),
    }


def per_link_max(t, mask=None):
    t = t.detach().cpu()
    if mask is not None:
        t = t[~mask.cpu().unsqueeze(-1).expand_as(t)].reshape(-1, t.shape[-1])
    return {f"link{i}": float(t[:, i].max()) for i in range(t.shape[-1])}


def main():
    env = make_flat_env()
    sim = env.simulator
    print(f"penalized link indices: {sim.penalized_contact_indices}")
    print(f"feet link indices: {sim.feet_contact_indices}")
    try:
        # find_link_indices already subtracts link_start -> index directly
        names = [l.name for l in sim._robot.links]
        pen_names = [names[i] for i in sim.penalized_contact_indices]
        feet_names = [names[i] for i in sim.feet_contact_indices]
    except Exception as e:  # names are nice-to-have; indices are the contract
        print(f"(link name lookup failed: {e})")
        pen_names = feet_names = None
    print(f"penalized links: {pen_names}")
    print(f"feet links: {feet_names}")
    env.reset()

    zero_actions = lambda step, prev: torch.zeros_like(prev)

    def random_actions(step, prev):
        if step % REDRAW_EVERY_B == 0:
            return torch.rand_like(prev) * 0.4 - 0.2
        return prev

    # crouch targets (rad): hips 0.0, thighs 2.2, calves -2.8 -> legs fold,
    # robot sits on thigh/calf links. action = (target - default) / 0.25
    default = sim.default_dof_pos[0].clone()
    target = default.clone()
    for i in range(0, 12, 3):      # hip joints
        target[i] = 0.0
    for i in range(1, 12, 3):      # thigh joints
        target[i] = 2.2
    for i in range(2, 12, 3):      # calf joints
        target[i] = -2.8
    crouch = ((target - default) / 0.25).unsqueeze(0).repeat(env.num_envs, 1)
    crouch_actions = lambda step, prev: crouch

    out = {
        "config": {
            "task": "go2_moects", "simulator": "genesis", "num_envs": NUM_ENVS,
            "terrain": "plane (moe_grid off)", "push_robots": False,
            "dt": env.dt, "settle_steps": {
                "A": SETTLE_STEPS_A, "B": SETTLE_STEPS_B, "C": SETTLE_STEPS_C},
            "post_reset_blackout_steps": POST_RESET_BLACKOUT,
        },
        "links": {
            "penalized_indices": [int(i) for i in sim.penalized_contact_indices],
            "penalized_names": pen_names,
            "feet_indices": [int(i) for i in sim.feet_contact_indices],
            "feet_names": feet_names,
        },
        "scenarios": {},
    }

    # --- Scenario A: quasi-static noise floor ---
    pen, feet, blackout, resets = run_scenario(
        env, zero_actions, SETTLE_STEPS_A, RECORD_STEPS_A)
    out["scenarios"]["A_quasistatic_zero_actions"] = {
        "recorded_steps": RECORD_STEPS_A, "env_resets_during_scenario": resets,
        "penalized": stats(pen, blackout),
        "penalized_per_link_max": per_link_max(pen, blackout),
        "feet": stats(feet, blackout),
    }
    print("scenario A done")

    # --- Scenario B: dynamic jitter noise floor ---
    pen, feet, blackout, resets = run_scenario(
        env, random_actions, SETTLE_STEPS_B, RECORD_STEPS_B)
    out["scenarios"]["B_dynamic_random_jitter"] = {
        "recorded_steps": RECORD_STEPS_B, "env_resets_during_scenario": resets,
        "action_distribution": "U(-0.2, 0.2) rad, re-drawn every 10 steps",
        "penalized": stats(pen, blackout),
        "penalized_per_link_max": per_link_max(pen, blackout),
        "feet": stats(feet, blackout),
    }
    print("scenario B done")

    # --- Scenario C: scripted crouch -> genuine thigh/calf contact ---
    pen, feet, blackout, resets = run_scenario(
        env, crouch_actions, SETTLE_STEPS_C, RECORD_STEPS_C)
    base_h = env.simulator.base_pos[:, 2].detach().cpu()
    # genuine-contact cluster: (env, step) samples whose max penalized-link
    # force exceeds 1.0 N (loose operational cut, far above any noise candidate)
    valid = ~blackout
    per_step_max = pen.max(dim=-1).values            # (steps, envs)
    cluster = per_step_max[(per_step_max > 1.0) & valid]
    cluster = cluster.double()
    if cluster.numel() > 0:
        p1 = float(torch.quantile(cluster, 0.01))
        p5 = float(torch.quantile(cluster, 0.05))
        p10 = float(torch.quantile(cluster, 0.10))
        p50 = float(torch.quantile(cluster, 0.50))
        cluster_stats = {
            "definition": "per-(env,step) max over penalized links, > 1.0 N",
            "n_samples": int(cluster.numel()),
            "frac_of_valid_samples": float(cluster.numel() / int(valid.sum())),
            "p1": p1, "p5": p5, "p10": p10, "p50": p50,
            "max": float(cluster.max()),
        }
    else:
        cluster_stats = {"n_samples": 0}
    out["scenarios"]["C_crouch_genuine_contact"] = {
        "recorded_steps": RECORD_STEPS_C, "env_resets_during_scenario": resets,
        "crouch_targets_rad": {"hip": 0.0, "thigh": 2.2, "calf": -2.8},
        "final_base_height_z": {
            "mean": float(base_h.mean()), "min": float(base_h.min()),
            "max": float(base_h.max())},
        "penalized": stats(pen, blackout),
        "contact_cluster": cluster_stats,
        "feet": stats(feet, blackout),
    }
    print("scenario C done")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved aggregates -> {OUT_JSON}")

    # ---- compact table ----
    def row(name, s):
        return (f"{name:<38} n={s['n_samples']:<9} zero={s['frac_exactly_zero']:.4f} "
                f"mean={s['mean']:.4g} p50={s['p50']:.4g} p99={s['p99']:.4g} "
                f"p99.9={s['p99_9']:.4g} max={s['max']:.4g}")
    print("\n=== force-norm stats [N] ===")
    for sc, blk in out["scenarios"].items():
        print(f"\n--- {sc} (resets: {blk['env_resets_during_scenario']}) ---")
        print(row("penalized (thigh/calf)", blk["penalized"]))
        print(row("feet", blk["feet"]))
        if "contact_cluster" in blk and blk["contact_cluster"].get("n_samples"):
            c = blk["contact_cluster"]
            print(f"{'genuine-contact cluster (max>1N)':<38} n={c['n_samples']:<9} "
                  f"p1={c['p1']:.4g} p5={c['p5']:.4g} p10={c['p10']:.4g} "
                  f"p50={c['p50']:.4g} max={c['max']:.4g}")

    if hasattr(env, "destroy"):
        env.destroy()


if __name__ == "__main__":
    sys.exit(main())
