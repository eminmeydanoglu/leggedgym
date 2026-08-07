"""Controlled-grid collection and causal latent intervention for MoE-CTS."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from legged_gym.scripts.eval.dr_axes import AXES, get_axis, pin_others_to_nominal
from legged_gym.scripts.eval.probe_lib import NORM_RANGES, dr_normalize
from legged_gym.scripts.eval.probe_moe_gate import (
    _build_env_and_policy, _restore_curriculum_state,
)


# control_delay is deliberately NOT probed on this policy.  dr_axes' axis
# implements the legacy "queue" delay model (one delay per episode, held in
# simulator._action_queue / _action_delay), but go2_moects trains with
# ctrl_delay_mode="substep" (go2_moects_config.py:315): the delay is re-drawn
# EVERY control step in sim-substep units and blended inside the decimation
# loop, so genesis_simulator.py:759 never allocates the action queue and there
# is no per-episode delay value to label a sample with.  Forcing queue mode
# would take the policy outside its training distribution, which is worse than
# leaving the axis out.  Same exclusion rationale as motor strength / zero
# offset: no validated setter+readback contract on this checkout.
AXIS_NAMES = ("friction", "added_mass", "com_x", "com_y", "com_z",
              "pd_gain_scale")
# Axes with no slot in the oracle's privileged P vector, so pin_others_to_nominal
# does not touch them and they must be pinned explicitly (see run_grid).
PINNED_NON_P_AXES = ("pd_gain_scale",)
PROBE_AXES = AXIS_NAMES
AXIS_CODE = {name: i for i, name in enumerate(AXIS_NAMES)}
COMMANDS = np.asarray([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, -0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0],
], dtype=np.float32)

# Separate from the causal grid's historical command ids.  This profile
# reproduces the six semantic command classes used by the paper's PCA figure.
PAPER_PCA_COMMANDS = np.asarray([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
], dtype=np.float32)


def _set_command(env, command):
    for key, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_yaw"), command):
        env.command_ranges[key] = [float(value), float(value)]
    env.commands[:, :3] = torch.as_tensor(command, device=env.device).view(1, 3)


def _physics_raw(env):
    columns = []
    for name in AXIS_NAMES:
        axis = get_axis(name)
        if axis.readback is None:
            raise RuntimeError(f"{name} has no live readback")
        columns.append(axis.readback(env).reshape(-1).detach().float().cpu().numpy())
    return np.stack(columns, axis=1).astype(np.float32)


def _physics_norm(raw):
    return np.stack([dr_normalize(raw[:, i], *NORM_RANGES[name])
                     for i, name in enumerate(AXIS_NAMES)], axis=1).astype(np.float32)


def _terrain(env):
    ids = getattr(env, "wty_terrain_ids", None)
    if ids is None:
        ids = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    levels = getattr(env.simulator, "terrain_levels", None)
    if levels is None:
        levels = torch.full_like(ids, -1)
    return ids.long(), levels.long()


def _contact_pattern(env):
    contact = (env.feet_max_force_z > 10.0).long()
    weights = 2 ** torch.arange(contact.shape[1], device=contact.device)
    return (contact * weights).sum(1)


def _latents(ac, history, priv):
    z_s = ac.history_encoder(history)
    z_t = ac.privilege_encoder(priv)
    experts = ac.history_encoder.moe.experts(history)
    gate = ac.history_encoder.moe.gating_network(history)
    return z_s, z_t, gate, torch.linalg.vector_norm(experts, dim=2)


def _load_checkpoint(ac, path):
    checkpoint = torch.load(path, map_location=next(ac.parameters()).device, weights_only=False)
    ac.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ac.eval()
    return int(checkpoint.get("iter", checkpoint.get("iteration", -1)))


def _resolve_checkpoints(cli):
    run_dir = Path(cli.run_dir) if cli.run_dir else Path(__file__).parents[3] / "logs/go2_moects" / cli.load_run
    paths = []
    for item in cli.checkpoints.split(","):
        item = item.strip()
        name = item if item.endswith(".pt") else (item if item.startswith("model_") else f"model_{item}") + ".pt"
        if item == "best_tracking":
            name = "best_tracking.pt"
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def _append(bank, **items):
    for key, value in items.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        bank.setdefault(key, []).append(np.asarray(value))


def _finish(bank):
    return {key: np.concatenate(values, axis=0) for key, values in bank.items()}


def _reset_and_warm(env, ac, command, warmup):
    env.reset()
    obs, priv, history, _ = env.get_observations()
    _set_command(env, command)
    with torch.no_grad():
        for _ in range(warmup):
            _set_command(env, command)
            action = ac.act_student(obs, history)
            obs, priv, history, _, _, _, _ = env.step(action)
    env.episode_length_buf.zero_()
    return obs, priv, history


def _common_cli(cli, num_envs):
    cli.num_envs = num_envs
    # The run stores the best 20500-equivalent checkpoint under this stable
    # alias; probing a nonexistent model_20500.pt would fail before scene build.
    cli.ckpt = "best_tracking"
    cli.nominal = False
    cli.cpu = False
    cli.prepare_axes = PROBE_AXES
    return cli


def run_grid(cli):
    axes = tuple(x.strip() for x in cli.axes.split(",") if x.strip())
    commands = COMMANDS[[int(x) for x in cli.command_ids.split(",")]]
    max_values = max(len(get_axis(name).default_grid) for name in axes)
    env, ac, ctx = _build_env_and_policy(_common_cli(cli, max_values * cli.per_point))
    _restore_curriculum_state(env, ctx["ckpt_path"], cli.seed)
    random_encoder = copy.deepcopy(ac.history_encoder)
    for module in random_encoder.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    random_encoder.eval()
    checkpoints = _resolve_checkpoints(cli)
    terrain_id, terrain_level = _terrain(env)
    all_data = []

    for checkpoint_path in checkpoints:
        ckpt_iter = _load_checkpoint(ac, checkpoint_path)
        tag = checkpoint_path.stem
        bank = {}
        episode_id = np.zeros(env.num_envs, dtype=np.int64)
        for axis_name in axes:
            axis = get_axis(axis_name)
            grid = np.asarray(axis.default_grid, dtype=np.float32)
            if cli.grid_limit:
                grid = grid[:cli.grid_limit]
            active = len(grid) * cli.per_point
            pin_others_to_nominal(env, axis_name)
            # pin_others_to_nominal only pins the oracle's privileged P vector
            # (friction/added_mass/com_bias). pd_gain_scale and control_delay have
            # no P slot, so they are left untouched by it -- explicitly pin them
            # here whenever they are not the swept axis, or a stale/randomized
            # value silently corrupts every other axis's physics_raw label.
            for other_name in AXIS_NAMES:
                if other_name != axis_name and other_name in PINNED_NON_P_AXES:
                    get_axis(other_name).pin_nominal(env)
            values = torch.full((env.num_envs,), float(axis.nominal), device=env.device)
            values[:active] = torch.as_tensor(np.repeat(grid, cli.per_point), device=env.device)
            axis.apply(env, values)
            axis.validate(env, values)
            raw = _physics_raw(env)[:active]
            norm = _physics_norm(raw)
            val_index = np.arange(active) // cli.per_point
            # OOD is defined by the actually-applied value falling outside the
            # axis's in-distribution training band, not by grid position: default
            # grids are intentionally wider than in_dist and not symmetric around
            # it, so a positional [0, len-1] check mislabels interior-but-OOD points.
            lo, hi = axis.in_dist
            swept_raw = raw[:, AXIS_NAMES.index(axis_name)]
            ood = (swept_raw < lo - 1e-6) | (swept_raw > hi + 1e-6)
            for command_id, command in zip([int(x) for x in cli.command_ids.split(",")], commands):
                obs, priv, history = _reset_and_warm(env, ac, command, cli.warmup)
                episode_id[:] = 0
                with torch.no_grad():
                    for step in range(cli.steps):
                        _set_command(env, command)
                        z_s, z_t, gate, e_norms = _latents(ac, history, priv)
                        z_random = random_encoder(history)
                        student_action = ac.actor(torch.cat([z_s, obs], dim=-1))
                        teacher_action = ac.actor(torch.cat([z_t, obs], dim=-1))
                        next_obs, next_priv, next_history, _, _, dones, _ = env.step(student_action)
                        if step % cli.stride == 0:
                            sl = slice(0, active)
                            contacts = _contact_pattern(env)[sl]
                            lin_err = torch.linalg.vector_norm(env.commands[sl, :2] - env.simulator.base_lin_vel[sl, :2], dim=1)
                            ang_err = torch.abs(env.commands[sl, 2] - env.simulator.base_ang_vel[sl, 2])
                            dof_acc = (env.simulator.dof_vel[sl] - env.simulator.last_dof_vel[sl]) / env.dt
                            gait_phase = (env.episode_length_buf[sl].float() * env.dt) % 1.0
                            _append(bank, obs=obs[sl], obs_history=history[sl], priv_obs=priv[sl],
                                    z_s=z_s[sl], z_t=z_t[sl], z_random=z_random[sl],
                                    g=gate[sl], E_norms=e_norms[sl],
                                    student_action=student_action[sl], teacher_action=teacher_action[sl],
                                    physics_raw=raw, physics_norm=norm,
                                    axis_code=np.full(active, AXIS_CODE[axis_name], np.int8),
                                    val_index=val_index.astype(np.int16),
                                    physics_combo_id=(AXIS_CODE[axis_name] * 100 + val_index).astype(np.int16),
                                    ood_flag=ood, env_id=np.arange(active, dtype=np.int32),
                                    episode_id=episode_id[:active].copy(), step=np.full(active, step, np.int32),
                                    done=dones[sl].bool(), fall=(dones[sl].bool() & ~env.time_out_buf[sl].bool()),
                                    time_out=env.time_out_buf[sl].bool(), terrain_id=terrain_id[sl],
                                    terrain_level=terrain_level[sl], command_id=np.full(active, command_id, np.int8),
                                    command=env.commands[sl, :3], base_lin_vel=env.simulator.base_lin_vel[sl],
                                    base_ang_vel=env.simulator.base_ang_vel[sl], contact_pattern=contacts,
                                    tracking_lin_err=lin_err, tracking_ang_err=ang_err,
                                    gait_phase=gait_phase, torque_norm=torch.linalg.vector_norm(env.simulator.torques[sl], dim=1),
                                    dof_acc_norm=torch.linalg.vector_norm(dof_acc, dim=1),
                                    command_change_time=np.full(active, step * env.dt, np.float32),
                                    ckpt_tag=np.full(active, tag), ckpt_iter=np.full(active, ckpt_iter, np.int32))
                        episode_id += dones.detach().cpu().numpy().astype(np.int64)
                        obs, priv, history = next_obs, next_priv, next_history
        data = _finish(bank)
        out = Path(cli.out_dir) / tag
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "samples.npz", **data)
        with open(out / "meta.json", "w") as handle:
            json.dump({"mode": "grid", "checkpoint": str(checkpoint_path), "n_samples": len(data["obs"]),
                       "axes": axes, "commands": commands.tolist(),
                       "excluded_axes": ["motor_strength", "zero_offset"]}, handle, indent=2)
        all_data.append(str(out / "samples.npz"))
        print(f"[moe-latent] {tag}: {len(data['obs'])} rows -> {out / 'samples.npz'}")
    return all_data


def run_paper_pca(cli):
    """Collect a clean bank for the paper-style terrain/command PCA plots."""
    env, ac, ctx = _build_env_and_policy(_common_cli(cli, cli.num_envs))
    _restore_curriculum_state(env, ctx["ckpt_path"], cli.seed)
    checkpoints = _resolve_checkpoints(cli)
    terrain_id, terrain_level = _terrain(env)
    outputs = []
    for checkpoint_path in checkpoints:
        ckpt_iter = _load_checkpoint(ac, checkpoint_path)
        tag = checkpoint_path.stem
        bank = {}
        for command_id, command in enumerate(PAPER_PCA_COMMANDS):
            obs, priv, history = _reset_and_warm(env, ac, command, cli.warmup)
            with torch.no_grad():
                for step in range(cli.steps):
                    _set_command(env, command)
                    z_s, z_t, gate, e_norms = _latents(ac, history, priv)
                    action = ac.actor(torch.cat([z_s, obs], dim=-1))
                    next_obs, next_priv, next_history, _, _, dones, _ = env.step(action)
                    if step % cli.stride == 0:
                        _append(
                            bank, z_s=z_s, z_t=z_t, g=gate, E_norms=e_norms,
                            terrain_id=terrain_id, terrain_level=terrain_level,
                            command_id=np.full(env.num_envs, command_id, np.int8),
                            command=env.commands[:, :3],
                            env_id=np.arange(env.num_envs, dtype=np.int32),
                            step=np.full(env.num_envs, step, np.int32), done=dones.bool(),
                            ckpt_tag=np.full(env.num_envs, tag),
                            ckpt_iter=np.full(env.num_envs, ckpt_iter, np.int32),
                        )
                    obs, priv, history = next_obs, next_priv, next_history
        data = _finish(bank)
        out = Path(cli.out_dir) / tag / "paper_pca"
        out.mkdir(parents=True, exist_ok=True)
        sample_path = out / "samples.npz"
        np.savez_compressed(sample_path, **data)
        with open(out / "meta.json", "w") as handle:
            json.dump({
                "mode": "paper_pca", "checkpoint": str(checkpoint_path),
                "n_samples": len(data["z_s"]),
                "commands": PAPER_PCA_COMMANDS.tolist(),
                "command_labels": ["Forward", "Backward", "Strafe Left",
                                   "Strafe Right", "Turn Left", "Turn Right"],
            }, handle, indent=2)
        outputs.append(str(sample_path))
        print(f"[moe-latent] {tag} paper PCA: {len(data['z_s'])} rows -> {sample_path}")
    return outputs


def _pool_key(command, terrain, contact, cell):
    return int(command), int(terrain), int(contact), int(cell)


def _latent_pools(samples):
    pools = {}
    for i in range(len(samples["z_s"])):
        key = _pool_key(samples["command_id"][i], samples["terrain_id"][i],
                        samples["contact_pattern"][i], samples["physics_combo_id"][i])
        pools.setdefault(key, []).append(samples["z_s"][i])
    return {key: np.asarray(value) for key, value in pools.items()}


def _sample_wrong_latent(pools, command, terrain, contact, cell, rng, mode):
    """Draw an on-manifold latent for one intervention mode.

    The two swap modes must differ in WHAT they hold fixed, otherwise their
    results are not comparable:

    ``shuffled_matched`` -- same (command, terrain, contact) as the live state,
    any other physics cell.  Tests "does the actor need THIS episode's latent,
    or does any latent from a matching state do?".

    ``wrong_regime`` -- same (command, terrain, contact) but a cell on the SAME
    swept axis, picked as far from the live cell as the bank allows (e.g. run
    at friction 1.25 while fed a friction 0.50 latent).  Without the same-axis
    constraint (``key[3] // 100 == cell // 100``, the axis code encoded in
    physics_combo_id) this mode silently degenerates into shuffled_matched by
    drawing from an unrelated axis.

    Returns ``(latent, tier)``.  ``tier`` records how far the fallback ladder
    had to walk -- 0 is the exact match.  A run whose wrong_regime samples are
    mostly tier>=2 has not tested what its name claims, so the tier is written
    into the bank instead of being silently absorbed.
    """
    same_state = [k for k in pools if k[:3] == (command, terrain, contact) and k[3] != cell]
    if mode == "wrong_regime":
        same_axis = [k for k in same_state if k[3] // 100 == cell // 100]
        if same_axis:
            # farthest value index on this axis == strongest regime mismatch
            far = max(abs(k[3] % 100 - cell % 100) for k in same_axis)
            ladder = [[k for k in same_axis if abs(k[3] % 100 - cell % 100) == far],
                      same_axis,
                      [k for k in pools if k[0] == command and k[3] // 100 == cell // 100
                       and k[3] != cell]]
        else:
            ladder = [[], [],
                      [k for k in pools if k[0] == command and k[3] // 100 == cell // 100
                       and k[3] != cell]]
        ladder.append([k for k in pools if k[3] // 100 == cell // 100 and k[3] != cell])
    else:
        ladder = [same_state,
                  [k for k in pools if k[:2] == (command, terrain) and k[3] != cell],
                  [k for k in pools if k[0] == command and k[3] != cell],
                  [k for k in pools if k[3] != cell]]
    for tier, candidates in enumerate(ladder):
        if candidates:
            key = candidates[int(rng.integers(len(candidates)))]
            values = pools[key]
            return values[int(rng.integers(len(values)))], tier
    raise RuntimeError(
        f"no {mode} latent available for command={command} terrain={terrain} "
        f"contact={contact} cell={cell}; the grid bank is too small to intervene")


def run_intervene(cli):
    samples = dict(np.load(cli.samples, allow_pickle=True))
    pools = _latent_pools(samples)
    env, ac, ctx = _build_env_and_policy(_common_cli(cli, cli.num_envs))
    checkpoint = _resolve_checkpoints(cli)[0]
    _load_checkpoint(ac, checkpoint)
    _restore_curriculum_state(env, str(checkpoint), cli.seed)
    axis = get_axis(cli.intervene_axis)
    grid = np.asarray(axis.default_grid, dtype=np.float32)
    value_index = cli.intervene_val_index if cli.intervene_val_index >= 0 else len(grid) - 1
    if value_index >= len(grid):
        raise ValueError(f"intervene_val_index={value_index} outside {cli.intervene_axis} grid")
    pin_others_to_nominal(env, cli.intervene_axis)
    values = torch.full((env.num_envs,), float(grid[value_index]), device=env.device)
    axis.apply(env, values)
    axis.validate(env, values)
    active_cell = AXIS_CODE[cli.intervene_axis] * 100 + value_index
    rng = np.random.default_rng(cli.seed)
    terrain_id, _ = _terrain(env)
    modes = ("student", "teacher_true", "shuffled_matched", "wrong_regime")
    bank = {}
    for mode in modes:
        obs, priv, history = _reset_and_warm(env, ac, COMMANDS[cli.intervene_command], cli.warmup)
        episode_id = np.zeros(env.num_envs, np.int64)
        with torch.no_grad():
            for step in range(cli.steps):
                _set_command(env, COMMANDS[cli.intervene_command])
                z_s, z_t, _, _ = _latents(ac, history, priv)
                reference_action = ac.actor(torch.cat([z_s, obs], dim=-1))
                if mode == "student":
                    latent = z_s
                    tiers = np.full(env.num_envs, -1, dtype=np.int64)
                elif mode == "teacher_true":
                    latent = z_t
                    tiers = np.full(env.num_envs, -1, dtype=np.int64)
                else:
                    contact = _contact_pattern(env).cpu().numpy()
                    drawn = [_sample_wrong_latent(pools, cli.intervene_command, int(terrain_id[i]),
                             int(contact[i]), active_cell, rng, mode)
                             for i in range(env.num_envs)]
                    picked = [d[0] for d in drawn]
                    tiers = np.asarray([d[1] for d in drawn], dtype=np.int64)
                    latent = torch.as_tensor(np.asarray(picked), device=env.device, dtype=obs.dtype)
                action = ac.actor(torch.cat([latent, obs], dim=-1))
                next_obs, next_priv, next_history, _, _, dones, _ = env.step(action)
                lin = torch.linalg.vector_norm(env.commands[:, :2] - env.simulator.base_lin_vel[:, :2], dim=1)
                ang = torch.abs(env.commands[:, 2] - env.simulator.base_ang_vel[:, 2])
                _append(bank, mode=np.full(env.num_envs, mode), step=np.full(env.num_envs, step),
                        pool_tier=tiers,
                        env_id=np.arange(env.num_envs), episode_id=episode_id.copy(), done=dones.bool(),
                        fall=(dones.bool() & ~env.time_out_buf.bool()), tracking_lin_err=lin,
                        tracking_ang_err=ang, achieved_speed=env.simulator.base_lin_vel[:, :2].norm(dim=1),
                        action_mae=torch.mean(torch.abs(action - reference_action), dim=1))
                episode_id += dones.cpu().numpy().astype(np.int64)
                obs, priv, history = next_obs, next_priv, next_history
    data = _finish(bank)
    Path(cli.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cli.out_dir) / "intervene.npz"
    np.savez_compressed(path, **data)
    print(f"[moe-latent] intervention -> {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("grid", "intervene", "paper_pca"), required=True)
    parser.add_argument("--task", default="go2_moects")
    parser.add_argument("--load_run", default="Aug03_12-01-45_moe_cts_genesis")
    parser.add_argument("--run_dir")
    parser.add_argument("--checkpoints", default="best_tracking,23500")
    parser.add_argument("--out_dir", default="logs/eval/moe_latent_probe")
    parser.add_argument("--samples")
    parser.add_argument("--axes", default=",".join(AXIS_NAMES))
    parser.add_argument("--command_ids", default="0,1,2,3,4,5")
    parser.add_argument("--per_point", type=int, default=8)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--grid_limit", type=int, default=0)
    parser.add_argument("--intervene_command", type=int, default=1)
    parser.add_argument("--intervene_axis", choices=AXIS_NAMES, default="friction")
    parser.add_argument("--intervene_val_index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.mode == "grid":
        run_grid(cli)
    elif cli.mode == "paper_pca":
        run_paper_pca(cli)
    else:
        if not cli.samples:
            raise ValueError("--samples is required for mode=intervene")
        run_intervene(cli)


if __name__ == "__main__":
    main()
