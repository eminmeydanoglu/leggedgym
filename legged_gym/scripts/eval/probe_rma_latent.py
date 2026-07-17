"""RMA latent-probe diagnostic for the go2_v3_rma student.

Proves whether the RMA student latent z_s encodes physics, tracks it online,
and is used causally.  Three modes:

  grid       -- single-axis OOD grid; record z_s, z_t, actions per env/step.
  switch     -- 0->+4 kg mid-episode; record per-step aggregated means.
  intervene  -- causal test: swap latent source and measure tracking/fall.

Build & env interface mirrors probe_p.py exactly; latent extraction added.

Example (GPU box, env activated):
    # grid collection -- all axes, 384 envs/point
    python legged_gym/scripts/eval/probe_rma_latent.py --mode grid \\
        --task go2_v3_rma --load_run <run> \\
        --per_point 384 --steps 500 --warmup 100 --stride 5 \\
        --out_dir logs/eval/v3_ilk_deneme/probes/rma/seed_1

    # switch collection
    python legged_gym/scripts/eval/probe_rma_latent.py --mode switch \\
        --task go2_v3_rma --load_run <run> \\
        --per_point 384 --steps 400 --warmup 100 --switch_step 100 \\
        --out_dir logs/eval/v3_ilk_deneme/probes/rma/seed_1

    # intervene collection
    python legged_gym/scripts/eval/probe_rma_latent.py --mode intervene \\
        --task go2_v3_rma --load_run <run> \\
        --per_point 384 --steps 500 --warmup 100 \\
        --out_dir logs/eval/v3_ilk_deneme/probes/rma/seed_1
"""

import argparse
import json
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

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
TASK_DEFAULT = "go2_v3_rma"

# Normalization ranges (spec §confirmed interfaces)
NORM_RANGES = {
    "friction":   (0.5,   1.25),
    "added_mass": (-2.0,  5.0),
    "com_x":      (-0.08, 0.08),
    "com_y":      (-0.08, 0.08),
    "com_z":      (-0.08, 0.08),
}

# Per-axis grids for mode=grid (spec §mode=grid)
AXIS_GRIDS = {
    "friction":   [0.50, 0.65, 0.80, 0.95, 1.10, 1.25],
    "added_mass": [-2.0, -1.0, 0.0,  1.0,  3.0,  5.0],
    "com_x":      [-0.08, -0.04, 0.0, 0.04, 0.08],
    "com_y":      [-0.08, -0.04, 0.0, 0.04, 0.08],
    "com_z":      [-0.08, -0.04, 0.0, 0.04, 0.08],
}
AXIS_CODE = {"friction": 0, "added_mass": 1, "com_x": 2, "com_y": 3, "com_z": 4}

# Command set (spec §command set)
COMMANDS = [
    [0.0,  0.0,  0.0],   # 0 stand
    [1.0,  0.0,  0.0],   # 1 fwd
    [0.0, -1.0,  0.0],   # 2 lat-1.0
    [0.0, -0.5,  0.0],   # 3 lat-0.5
    [0.0,  0.5,  0.0],   # 4 lat+0.5
    [0.0,  1.0,  0.0],   # 5 lat+1.0
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def dr_normalize(x, lo, hi):
    """Clip-normalized label: 2*(x-lo)/(hi-lo)-1, clamped to [-1,1]."""
    rng = hi - lo
    if abs(rng) < 1e-12:
        return np.zeros_like(np.asarray(x, dtype=np.float32))
    return np.clip(2.0 * (np.asarray(x, dtype=np.float32) - lo) / rng - 1.0, -1.0, 1.0)


def _history_encoder_input(ac, history):
    """Return history reshaped for the encoder (TCN needs unsqueeze(1))."""
    if getattr(ac, "history_encoder_type", "MLP") == "TCN":
        return history.unsqueeze(1)
    return history


def _get_latents(ac, obs, priv_obs, history):
    """Return (z_t, z_s) as (N,8) float tensors under no_grad."""
    z_t = ac.privilege_encoder(priv_obs)
    z_s = ac.history_encoder(_history_encoder_input(ac, history))
    return z_t, z_s


def _read_raw_physics(env):
    """Return (friction_raw, mass_raw, comx, comy, comz) each (num_envs,) numpy."""
    sim = env.simulator
    fr   = sim._friction_values[:, 0].detach().cpu().numpy().astype(np.float32)
    mass = sim._added_base_mass[:, 0].detach().cpu().numpy().astype(np.float32)
    com  = sim._base_com_bias.detach().cpu().numpy().astype(np.float32)  # (N,3)
    return fr, mass, com[:, 0], com[:, 1], com[:, 2]


def _set_command(env, cmd_vec, device):
    """Re-assert env.commands[:, :3] = cmd_vec every step (prevents resampling)."""
    cmd = torch.tensor(cmd_vec, dtype=torch.float32, device=device)
    env.commands[:, :3] = cmd.unsqueeze(0).expand(env.num_envs, -1)


def _compute_lin_ang_err(env):
    cmd = env.commands
    v   = env.simulator.base_lin_vel
    w   = env.simulator.base_ang_vel
    lin_err = torch.norm(cmd[:, :2] - v[:, :2], dim=1)
    ang_err = torch.abs(cmd[:, 2] - w[:, 2])
    return lin_err, ang_err


# ---------------------------------------------------------------------------
# build_env_and_policy  (verbatim mirror of probe_p.build_env_and_policy)
# ---------------------------------------------------------------------------

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
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=cli.task, args=reg_args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)  # student inference fn

    # Grab the full model (confirmed interface: ppo_runner.alg.actor_critic)
    ac = ppo_runner.alg.actor_critic  # ActorCriticTS, already .eval()
    ac.eval()

    return env, policy, ac, env_cfg, train_cfg, chosen_run, ckpt_path


# ---------------------------------------------------------------------------
# warmup helper (mirrors sweep.py / probe_p.py pattern for RMA 4-tuple I/O)
# ---------------------------------------------------------------------------

def _warmup_rma(env, ac, cli_warmup, cmd_vec):
    """Run warmup steps with student actions; return (obs, priv_obs, history)."""
    env.reset()
    obs, priv_obs, history, _ = env.get_observations()
    device = env.device
    _set_command(env, cmd_vec, device)

    with torch.no_grad():
        for _ in range(cli_warmup):
            _set_command(env, cmd_vec, device)
            actions = ac.act_student(obs.detach(), history.detach())
            obs, priv_obs, history, _, _, _, _ = env.step(actions.detach())
            _set_command(env, cmd_vec, device)

    env.episode_length_buf.zero_()
    return obs, priv_obs, history


# ---------------------------------------------------------------------------
# mode = grid
# ---------------------------------------------------------------------------

def run_grid(cli):
    """Collect per-sample rows across all axes and commands; save samples.npz.

    Genesis allows a single init/scene per process, so we build the env ONCE
    (sized to the largest axis grid) and reconfigure physics per axis via the
    live setters -- exactly how switch/intervene reconfigure a running env.
    Axes shorter than the max grid leave the trailing envs pinned nominal and
    simply are not recorded.
    """
    os.makedirs(cli.out_dir, exist_ok=True)

    max_nvals = max(len(v) for v in AXIS_GRIDS.values())
    num_envs  = max_nvals * cli.per_point

    # single build -- gs.init() runs exactly once (build command is a placeholder;
    # the per-command range is pinned on the live env below)
    cli_tmp = SimpleNamespace(**vars(cli))
    cli_tmp.command_vx = cli_tmp.command_vy = cli_tmp.command_yaw = 0.0
    env, policy, ac, env_cfg, train_cfg, chosen_run, ckpt_path = \
        build_env_and_policy(cli_tmp, num_envs)
    device = env.device
    ckpt_path_final = ckpt_path
    cli.load_run = chosen_run  # record the resolved run name in meta

    all_rows = []
    for axis_name, grid_vals in AXIS_GRIDS.items():
        grid      = np.asarray(grid_vals, dtype=np.float32)
        n_vals    = len(grid)
        n_active  = n_vals * cli.per_point       # only these envs are recorded
        axis_code = AXIS_CODE[axis_name]

        print(f"\n[probe-grid] axis={axis_name}  grid={grid_vals}  "
              f"active_envs={n_active}/{num_envs}  commands={len(COMMANDS)}")

        # reconfigure physics on the LIVE env: pin all others nominal, set this axis
        axis_obj = get_axis(axis_name)
        pin_others_to_nominal(env, axis_name)
        per_env_vals = torch.full((num_envs,), float(axis_obj.nominal),
                                  dtype=torch.float32, device=device)
        per_env_vals[:n_active] = torch.tensor(
            np.repeat(grid, cli.per_point), dtype=torch.float32, device=device)
        axis_obj.setter(env, per_env_vals)

        # raw physics for the active envs (static across commands for this axis)
        fr_raw, mass_raw, cx_raw, cy_raw, cz_raw = (
            a[:n_active] for a in _read_raw_physics(env))
        p5_raw_active = np.stack([fr_raw, mass_raw, cx_raw, cy_raw, cz_raw], axis=1)
        val_index = np.arange(n_active, dtype=np.int32) // cli.per_point
        combo_id  = (axis_code * 100 + val_index).astype(np.int32)
        env_id    = np.arange(n_active, dtype=np.int32)

        for cmd_id, cmd_vec in enumerate(COMMANDS):
            # Make the env's OWN command resampling reproduce this command.
            # legged_robot resamples every resampling_time steps (and at
            # episode_length_buf==0, right after warmup) by drawing from
            # env.command_ranges (a dict built from cfg at init, legged_robot.py:703).
            # Pin the range to a point == this command so resampling is a no-op.
            env.command_ranges["lin_vel_x"]   = [cmd_vec[0], cmd_vec[0]]
            env.command_ranges["lin_vel_y"]   = [cmd_vec[1], cmd_vec[1]]
            env.command_ranges["ang_vel_yaw"] = [cmd_vec[2], cmd_vec[2]]

            # warmup (resets -> resamples from the pinned range -> our command)
            obs, priv_obs, history, _ = _warmup_step_free(
                env, ac, cli.warmup, cmd_vec, device)

            sl = slice(0, n_active)
            with torch.no_grad():
                for step_idx in range(cli.steps):
                    _set_command(env, cmd_vec, device)
                    actions_student = ac.act_student(obs.detach(), history.detach())
                    obs_new, priv_obs_new, history_new, _, rew, dones, _ = \
                        env.step(actions_student.detach())
                    fall = (dones.bool() & ~env.time_out_buf.bool())

                    if step_idx % cli.stride == 0:
                        z_t, z_s = _get_latents(ac, obs, priv_obs, history)
                        t_action = ac.act_teacher(obs, priv_obs)
                        lin_err, ang_err = _compute_lin_ang_err(env)

                        all_rows.append({
                            "P5_norm":          priv_obs[sl, :5].detach().cpu().float().numpy(),
                            "P5_raw":           p5_raw_active,
                            "vel_norm":         priv_obs[sl, 5:8].detach().cpu().float().numpy(),
                            "vel_raw":          env.simulator.base_lin_vel[sl].detach().cpu().float().numpy(),
                            "z_t":              z_t[sl].detach().cpu().float().numpy(),
                            "z_s":              z_s[sl].detach().cpu().float().numpy(),
                            "teacher_action":   t_action[sl].detach().cpu().float().numpy(),
                            "student_action":   actions_student[sl].detach().cpu().float().numpy(),
                            "obs":              obs[sl].detach().cpu().float().numpy(),
                            "command":          np.tile(np.asarray(cmd_vec, dtype=np.float32), (n_active, 1)),
                            "command_id":       np.full(n_active, cmd_id, dtype=np.int32),
                            "axis_code":        np.full(n_active, axis_code, dtype=np.int32),
                            "physics_combo_id": combo_id,
                            "env_id":           env_id,
                            "step":             np.full(n_active, step_idx, dtype=np.int32),
                            "fall":             fall[sl].detach().cpu().numpy().astype(np.float32),
                            "tracking_lin_err": lin_err[sl].detach().cpu().float().numpy(),
                            "tracking_ang_err": ang_err[sl].detach().cpu().float().numpy(),
                        })

                    obs, priv_obs, history = obs_new, priv_obs_new, history_new

            print(f"[probe-grid]   axis={axis_name} cmd={cmd_id}  "
                  f"rows_so_far={sum(r['z_s'].shape[0] for r in all_rows)}")

    # --- concatenate all rows ---
    if not all_rows:
        raise RuntimeError("[probe-grid] no rows collected")

    keys = list(all_rows[0].keys())
    combined = {k: np.concatenate([r[k] for r in all_rows], axis=0) for k in keys}

    # optional sub-sampling
    N_total = combined["z_s"].shape[0]
    if cli.max_samples > 0 and N_total > cli.max_samples:
        idx = np.random.choice(N_total, cli.max_samples, replace=False)
        idx.sort()
        combined = {k: v[idx] for k, v in combined.items()}
        N_total = cli.max_samples
        print(f"[probe-grid] sub-sampled to {N_total} rows (--max_samples)")

    print(f"[probe-grid] total rows: {N_total}")

    # meta
    meta = {
        "axes":         list(AXIS_GRIDS.keys()),
        "grids":        {ax: AXIS_GRIDS[ax] for ax in AXIS_GRIDS},
        "commands":     COMMANDS,
        "per_point":    cli.per_point,
        "steps":        cli.steps,
        "stride":       cli.stride,
        "warmup":       cli.warmup,
        "seed_label":   cli.seed_label,
        "task":         cli.task,
        "load_run":     cli.load_run,
        "ckpt_path":    ckpt_path_final,
        "norm_ranges":  NORM_RANGES,
    }

    out_path = os.path.join(cli.out_dir, "samples.npz")
    np.savez(
        out_path,
        meta=np.array(json.dumps(meta)),
        **{k: v.astype(np.float32) if v.dtype.kind == "f" else v
           for k, v in combined.items()},
    )
    print(f"[probe-grid] saved -> {out_path}")


def _warmup_step_free(env, ac, warmup, cmd_vec, device):
    """Warmup the env with student actions for RMA (4-tuple I/O); return obs/priv/hist/_."""
    env.reset()
    obs, priv_obs, history, critic_obs = env.get_observations()
    _set_command(env, cmd_vec, device)

    with torch.no_grad():
        for _ in range(warmup):
            _set_command(env, cmd_vec, device)
            actions = ac.act_student(obs.detach(), history.detach())
            obs, priv_obs, history, critic_obs, _, _, _ = env.step(actions.detach())
    env.episode_length_buf.zero_()
    return obs, priv_obs, history, critic_obs


# ---------------------------------------------------------------------------
# mode = switch
# ---------------------------------------------------------------------------

def run_switch(cli):
    """Mass switch 0 -> +switch_to kg mid-episode; per-step aggregated means."""
    os.makedirs(cli.out_dir, exist_ok=True)

    num_envs = cli.per_point
    cmd_vec  = COMMANDS[cli.switch_command_id]

    cli_tmp = SimpleNamespace(**vars(cli))
    cli_tmp.command_vx  = cmd_vec[0]
    cli_tmp.command_vy  = cmd_vec[1]
    cli_tmp.command_yaw = cmd_vec[2]

    env, policy, ac, env_cfg, train_cfg, chosen_run, ckpt_path = \
        build_env_and_policy(cli_tmp, num_envs)
    device = env.device

    mass_axis = get_axis("added_mass")
    pin_others_to_nominal(env, "added_mass")
    mass_axis.setter(env, torch.full((num_envs,), float(cli.switch_from), device=device))

    obs, priv_obs, history, _ = _warmup_step_free(env, ac, cli.warmup, cmd_vec, device)

    # per-step arrays
    T_steps = cli.steps
    real_mass_norm_t  = np.zeros(T_steps, dtype=np.float32)
    real_mass_raw_t   = np.zeros(T_steps, dtype=np.float32)
    z_t_mean_t        = np.zeros((T_steps, 8), dtype=np.float32)
    z_s_mean_t        = np.zeros((T_steps, 8), dtype=np.float32)
    latent_err_t      = np.zeros(T_steps, dtype=np.float32)
    action_mae_t      = np.zeros(T_steps, dtype=np.float32)
    tracking_lin_err_t = np.zeros(T_steps, dtype=np.float32)
    # per-env at each step (keep as compact float32 array -- shape T x N x 8)
    # only z_s and z_t per-env for analysis; skip for large N
    store_per_env = (num_envs <= 512)
    if store_per_env:
        z_s_per_env = np.zeros((T_steps, num_envs, 8), dtype=np.float32)
        z_t_per_env = np.zeros((T_steps, num_envs, 8), dtype=np.float32)

    lo_m, hi_m = NORM_RANGES["added_mass"]

    with torch.no_grad():
        for step_idx in range(T_steps):
            # switch at switch_step
            if step_idx == cli.switch_step:
                mass_axis.setter(
                    env, torch.full((num_envs,), float(cli.switch_to), device=device))
                print(f"[probe-switch] step={step_idx}: switched mass "
                      f"{cli.switch_from} -> {cli.switch_to} kg")

            _set_command(env, cmd_vec, device)
            actions_student = ac.act_student(obs.detach(), history.detach())
            obs_new, priv_obs_new, history_new, _, rew, dones, extras = \
                env.step(actions_student.detach())
            _set_command(env, cmd_vec, device)

            # compute latents + teacher action on the NEW obs after step
            z_t, z_s = _get_latents(ac, obs_new, priv_obs_new, history_new)
            t_action  = ac.act_teacher(obs_new, priv_obs_new)
            lin_err, ang_err = _compute_lin_ang_err(env)

            # read true mass
            mass_raw_now = env.simulator._added_base_mass[:, 0].detach().cpu().float().numpy()
            mass_norm_now = dr_normalize(mass_raw_now, lo_m, hi_m)

            z_t_np = z_t.detach().cpu().float().numpy()
            z_s_np = z_s.detach().cpu().float().numpy()
            ta_np  = t_action.detach().cpu().float().numpy()
            sa_np  = actions_student.detach().cpu().float().numpy()

            real_mass_norm_t[step_idx]   = mass_norm_now.mean()
            real_mass_raw_t[step_idx]    = mass_raw_now.mean()
            z_t_mean_t[step_idx]         = z_t_np.mean(axis=0)
            z_s_mean_t[step_idx]         = z_s_np.mean(axis=0)
            latent_err_t[step_idx]       = np.linalg.norm(z_s_np - z_t_np, axis=1).mean()
            action_mae_t[step_idx]       = np.abs(sa_np - ta_np).mean()
            tracking_lin_err_t[step_idx] = lin_err.detach().cpu().float().numpy().mean()

            if store_per_env:
                z_s_per_env[step_idx] = z_s_np
                z_t_per_env[step_idx] = z_t_np

            obs, priv_obs, history = obs_new, priv_obs_new, history_new

    out_path = os.path.join(cli.out_dir, "switch_raw.npz")
    save_dict = dict(
        real_mass_norm=real_mass_norm_t,
        real_mass_raw=real_mass_raw_t,
        z_t_mean=z_t_mean_t,
        z_s_mean=z_s_mean_t,
        latent_err=latent_err_t,
        action_mae=action_mae_t,
        tracking_lin_err=tracking_lin_err_t,
        switch_step=np.int32(cli.switch_step),
        switch_from=np.float32(cli.switch_from),
        switch_to=np.float32(cli.switch_to),
        command_id=np.int32(cli.switch_command_id),
        steps=np.int32(T_steps),
    )
    if store_per_env:
        save_dict["z_s_per_env"] = z_s_per_env  # (T, N, 8)
        save_dict["z_t_per_env"] = z_t_per_env  # (T, N, 8)

    np.savez(out_path, **save_dict)
    print(f"[probe-switch] saved -> {out_path}")


# ---------------------------------------------------------------------------
# mode = intervene
# ---------------------------------------------------------------------------

def run_intervene(cli):
    """Causal test: swap latent source and measure tracking/fall/achieved speed."""
    os.makedirs(cli.out_dir, exist_ok=True)

    mass_grid = cli.intervene_mass_grid   # e.g. [-2, 0, 3, 5]
    n_vals    = len(mass_grid)
    num_envs  = n_vals * cli.per_point
    cmd_vec   = COMMANDS[1]  # fwd [1,0,0]

    cli_tmp = SimpleNamespace(**vars(cli))
    cli_tmp.command_vx  = cmd_vec[0]
    cli_tmp.command_vy  = cmd_vec[1]
    cli_tmp.command_yaw = cmd_vec[2]

    env, policy, ac, env_cfg, train_cfg, chosen_run, ckpt_path = \
        build_env_and_policy(cli_tmp, num_envs)
    device = env.device

    mass_axis = get_axis("added_mass")
    per_env_mass = torch.tensor(
        np.repeat(np.asarray(mass_grid, dtype=np.float32), cli.per_point),
        dtype=torch.float32, device=device)
    pin_others_to_nominal(env, "added_mass")
    mass_axis.setter(env, per_env_mass)

    lo_m, hi_m = NORM_RANGES["added_mass"]
    wrong_mass_val = float(mass_grid[0]) if float(mass_grid[-1]) > 0 else float(mass_grid[-1])

    # nominal P5 (friction=1->norm, mass=0->norm, com=0->norm)
    nominal_p5_norm = np.array([
        dr_normalize(1.0,  *NORM_RANGES["friction"]),
        dr_normalize(0.0,  *NORM_RANGES["added_mass"]),
        dr_normalize(0.0,  *NORM_RANGES["com_x"]),
        dr_normalize(0.0,  *NORM_RANGES["com_y"]),
        dr_normalize(0.0,  *NORM_RANGES["com_z"]),
    ], dtype=np.float32)

    # wrong mass: if environment has mass in [0,5] range, swap to far negative end
    # default: use the opposite end of the grid
    wrong_mass_norm = dr_normalize(wrong_mass_val, lo_m, hi_m)

    MODES_INTERVENE = ["student", "teacher_true", "teacher_nominal", "teacher_wrong"]

    mode_results = {}

    for mode in MODES_INTERVENE:
        print(f"\n[probe-intervene] mode={mode}")
        # warmup fresh for each mode
        obs, priv_obs, history, _ = _warmup_step_free(
            env, ac, cli.warmup, cmd_vec, device)

        acc = MetricAccumulator(num_envs, device)

        with torch.no_grad():
            for _step in range(cli.steps):
                _set_command(env, cmd_vec, device)

                if mode == "student":
                    z_s = ac.history_encoder(
                        _history_encoder_input(ac, history.detach()))
                    actions = ac.actor(
                        torch.cat([obs.detach(), z_s], dim=-1))

                elif mode == "teacher_true":
                    z_t = ac.privilege_encoder(priv_obs.detach())
                    actions = ac.actor(
                        torch.cat([obs.detach(), z_t], dim=-1))

                elif mode == "teacher_nominal":
                    # build nominal priv_obs: overwrite P5 with nominal-norm, keep vel
                    nom_priv = priv_obs.clone()
                    nom_p5 = torch.tensor(nominal_p5_norm, dtype=torch.float32,
                                          device=device).unsqueeze(0).expand(num_envs, -1)
                    nom_priv[:, :5] = nom_p5
                    # velocity ([:,5:8]) kept true
                    z_nom = ac.privilege_encoder(nom_priv)
                    actions = ac.actor(
                        torch.cat([obs.detach(), z_nom], dim=-1))

                elif mode == "teacher_wrong":
                    # overwrite added_mass slot (index 1 in P5) with wrong value
                    wrong_priv = priv_obs.clone()
                    wrong_priv[:, 1] = float(wrong_mass_norm)
                    z_wrong = ac.privilege_encoder(wrong_priv)
                    actions = ac.actor(
                        torch.cat([obs.detach(), z_wrong], dim=-1))
                else:
                    raise ValueError(f"unknown mode {mode}")

                obs_new, priv_obs_new, history_new, _, rew, dones, _ = \
                    env.step(actions.detach())
                _set_command(env, cmd_vec, device)

                lin_err, ang_err = _compute_lin_ang_err(env)
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
                obs, priv_obs, history = obs_new, priv_obs_new, history_new

        per_env_metrics = acc.compute()
        mode_results[mode] = aggregate(per_env_metrics, n_vals, cli.per_point)

        # also compute achieved_speed_ratio (not in MetricAccumulator)
        # approximate from the final obs; store mean across per_point replicas
        achieved_vx = env.simulator.base_lin_vel[:, 0].detach().cpu().float().numpy()
        ratio = achieved_vx / (cmd_vec[0] if abs(cmd_vec[0]) > 1e-6 else 1.0)
        ratio_grouped = ratio.reshape(n_vals, cli.per_point).mean(axis=1)
        mode_results[mode]["achieved_speed_ratio_mean"] = ratio_grouped

    # print comparison table
    print(f"\n=== intervene | go2_v3_rma | mass_grid={mass_grid} | fwd cmd ===")
    header = "mode              " + " ".join(f"{g:>8.1f}" for g in mass_grid)
    print(header)
    for mode in MODES_INTERVENE:
        row = mode_results[mode]["tracking_lin_err_mean"]
        print(f"{mode:<18} " + " ".join(f"{x:8.4f}" for x in row))
    print("(lower tracking_lin_err = better)")

    # save intervene.npz
    flat = {f"{mode}__{k}": v for mode, agg in mode_results.items() for k, v in agg.items()}
    out_path = os.path.join(cli.out_dir, "intervene.npz")
    np.savez(
        out_path,
        mass_grid=np.asarray(mass_grid, dtype=np.float32),
        modes=np.array(MODES_INTERVENE),
        per_point=np.int32(cli.per_point),
        steps=np.int32(cli.steps),
        command=np.array(cmd_vec, dtype=np.float32),
        wrong_mass_val=np.float32(wrong_mass_val),
        wrong_mass_norm=np.float32(wrong_mass_norm),
        **{k: v.astype(np.float32) if hasattr(v, "astype") else v
           for k, v in flat.items()},
    )
    print(f"[probe-intervene] saved -> {out_path}")


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="RMA latent-probe: collect z_s/z_t samples for physics-encoding analysis")
    p.add_argument("--mode", choices=["grid", "switch", "intervene"], required=True)
    p.add_argument("--task", type=str, default=TASK_DEFAULT,
                   help=f"registered task (default: {TASK_DEFAULT})")
    p.add_argument("--load_run", type=str, default=None,
                   help="run directory to load (default: latest matching run_name)")
    p.add_argument("--ckpt", type=str, default="best_tracking",
                   help="checkpoint spec: 'best_tracking' (deployed policy, default), "
                        "'best', 'latest', or an integer iteration number")
    p.add_argument("--seed_label", type=str, default="seed_1",
                   help="label for this seed (e.g. seed_1); used in meta and output path")
    p.add_argument("--per_point", type=int, default=384,
                   help="parallel env replicas per grid point (grid/switch modes)")
    p.add_argument("--steps", type=int, default=500,
                   help="measured steps per command/condition")
    p.add_argument("--warmup", type=int, default=100,
                   help="unrecorded settling steps before measurement")
    p.add_argument("--stride", type=int, default=5,
                   help="record every stride steps (grid mode)")
    p.add_argument("--out_dir", type=str, required=True,
                   help="output directory for .npz files")
    p.add_argument("--seed", type=int, default=1,
                   help="global RNG seed")
    p.add_argument("--cpu", action="store_true", default=False,
                   help="use Genesis CPU backend (for testing; very slow)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="if >0, sub-sample the grid rows to this count (float32; memory cap)")
    # switch mode
    p.add_argument("--switch_step", type=int, default=100,
                   help="step at which to apply mass switch (switch mode)")
    p.add_argument("--switch_from", type=float, default=0.0,
                   help="initial added_mass [kg] (switch mode)")
    p.add_argument("--switch_to", type=float, default=4.0,
                   help="target added_mass [kg] after switch (switch mode)")
    p.add_argument("--switch_command_id", type=int, default=1,
                   help="command index for switch mode (default 1 = fwd)")
    # intervene mode
    p.add_argument("--intervene_mass_grid", type=float, nargs="+",
                   default=[-2.0, 0.0, 3.0, 5.0],
                   help="added_mass grid for intervene mode")
    p.add_argument("--wrong", choices=["swap_mass", "permute"], default="swap_mass",
                   help="how to construct the wrong latent in intervene mode")
    return p.parse_args()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    cli = parse_args()
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    os.makedirs(cli.out_dir, exist_ok=True)

    if cli.mode == "grid":
        run_grid(cli)
    elif cli.mode == "switch":
        run_switch(cli)
    elif cli.mode == "intervene":
        run_intervene(cli)
    else:
        raise ValueError(f"unknown mode: {cli.mode}")


if __name__ == "__main__":
    main()
