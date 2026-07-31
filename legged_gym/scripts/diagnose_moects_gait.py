"""Measure what a go2_moects policy actually does on FLAT ground, headlessly.

Watching Viser tells you a gait looks wrong; it does not tell you which of the
many wiring/physics knobs is wrong. This runs the same inference path play.py
uses (``get_inference_policy`` -> ``act_student``) on plane terrain with domain
randomization and observation noise off, and reports numbers that separate the
plausible causes:

  stand phase (zero command)
    base height, its drift, and per-joint deviation from the default pose. A
    correctly wired policy holds ~0.30-0.34 m with a small, steady deviation.
    Collapse or a large deviation means the actions themselves are wrong.

  walk phase (vx command)
    tracking error, plus the foot contact statistics that distinguish gaits:
      duty       -- fraction of steps each foot is loaded (trot ~0.5)
      flight     -- fraction of steps with ALL feet off the ground; a trot at
                    these speeds is near 0, so a high value IS the "hopping"
                    the viewer shows, quantified
      contacts   -- mean number of loaded feet (trot ~2)
      action_rate-- mean |a_t - a_{t-1}|; the "stiff/juddering" signature

``--compare-unpermuted`` additionally runs the same policy with the dof
permutation undone, so the joint-order fix is checked against behaviour and not
only against the algebra in import_go2_rl_gym_policy.py.

Usage:
    SIMULATOR=genesis python legged_gym/scripts/diagnose_moects_gait.py \
        --task=go2_moects --load_run wty_go2_moe_cts_137k --ckpt 0
"""

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.envs import *  # noqa: F401,F403  (populates task_registry)
from legged_gym.utils import get_args, task_registry

if SIMULATOR == "genesis":
    import genesis as gs

FOOT_CONTACT_N = 1.0  # [N] a foot carrying less than this is treated as swinging


def _flat_env_overrides(env_cfg, num_envs):
    """Plane terrain, nominal physics, clean observations, no auto-reset."""
    env_cfg.env.num_envs = num_envs
    env_cfg.env.debug = False
    env_cfg.env.auto_reset = False
    if getattr(env_cfg.env, "ued_enabled", False):
        env_cfg.env.ued_enabled = False

    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.terrain_kwargs = None
    env_cfg.terrain.measure_heights = False
    for flag in ("moe_grid", "moe_showcase", "taxonomy_showcase",
                 "v6_frontier_showcase", "ued_training_grid"):
        if hasattr(env_cfg.terrain, flag):
            setattr(env_cfg.terrain, flag, False)

    # Same nominal-physics contract play.py now uses by default.
    from legged_gym.scripts.play import _disable_play_domain_rand
    _disable_play_domain_rand(env_cfg)

    env_cfg.commands.resampling_time = 10 ** 6  # we set commands by hand
    if hasattr(env_cfg.commands, "heading_command"):
        env_cfg.commands.heading_command = False
    return env_cfg


def _foot_contacts(env):
    """(num_envs, 4) bool: which feet carry load this step."""
    forces = env.simulator.link_contact_forces[:, env.simulator.feet_indices, :]
    return forces.norm(dim=-1) > FOOT_CONTACT_N


def run_phase(env, policy, obs, history, steps, command, label, trace=0):
    """Drive `command` for `steps` policy steps and summarise the motion."""
    env.commands[:, :3] = torch.tensor(command, device=env.device, dtype=torch.float)
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0

    heights, vx, contacts, flight, duty = [], [], [], [], []
    dof_dev, action_rate = [], []
    act_absmax, act_absmean = [], []
    obs_blocks = []
    last_actions = None
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for step in range(steps):
        with torch.no_grad():
            actions = policy(obs, history)
        if trace and step < trace:
            # env 0 only, before stepping: the input that produced `actions`.
            j = actions[0].abs().argmax().item()
            print(f"    t={step:3d} h={env.simulator.base_pos[0,2]:.3f} "
                  f"vx={env.simulator.base_lin_vel[0,0]:+.2f} "
                  f"|a|max={actions[0].abs().max():8.2f}@dof{j:2d} "
                  f"|a|mean={actions[0].abs().mean():7.2f} "
                  f"dofdev={(env.simulator.dof_pos[0]-env.simulator.default_dof_pos[0]).abs().max():.2f} "
                  f"dofvel={env.simulator.dof_vel[0].abs().max():6.2f} "
                  f"prevact={obs[0,33:45].abs().max():7.2f} "
                  f"contacts={_foot_contacts(env)[0].sum().item()}")
        obs, _, history, _, _, dones, _ = env.step(actions)
        env.commands[:, :3] = torch.tensor(command, device=env.device, dtype=torch.float)

        terminated |= dones.bool()
        heights.append(env.simulator.base_pos[:, 2].cpu().numpy())
        vx.append(env.simulator.base_lin_vel[:, 0].cpu().numpy())
        c = _foot_contacts(env)
        contacts.append(c.float().sum(dim=1).cpu().numpy())
        flight.append((~c.any(dim=1)).float().cpu().numpy())
        duty.append(c.float().cpu().numpy())
        dof_dev.append((env.simulator.dof_pos - env.simulator.default_dof_pos)
                       .abs().mean(dim=1).cpu().numpy())
        act_absmax.append(actions.abs().max().item())
        act_absmean.append(actions.abs().mean().item())
        # Per-block observation ranges: a saturated or mis-scaled block is the
        # difference between "policy is bad here" and "policy is being fed junk".
        obs_blocks.append([obs[:, s:e].abs().max().item() for s, e in
                           ((0, 3), (3, 6), (6, 9), (9, 21), (21, 33), (33, 45))])
        if last_actions is not None:
            action_rate.append((actions - last_actions).abs().mean(dim=1).cpu().numpy())
        last_actions = actions.clone()

    h = np.asarray(heights)
    stats = {
        "base_height_mean": float(h.mean()),
        "base_height_std": float(h.std()),
        "base_height_final": float(h[-1].mean()),
        "vx_mean": float(np.asarray(vx).mean()),
        "vx_cmd": float(command[0]),
        "contacts_mean": float(np.asarray(contacts).mean()),
        "flight_fraction": float(np.asarray(flight).mean()),
        "duty_per_foot": np.asarray(duty).mean(axis=(0, 1)).tolist(),
        "dof_dev_mean": float(np.asarray(dof_dev).mean()),
        "action_rate_mean": float(np.asarray(action_rate).mean()) if action_rate else float("nan"),
        "terminated": int(terminated.sum().item()),
        "num_envs": int(env.num_envs),
    }

    print(f"\n--- {label} (cmd vx={command[0]:.2f} vy={command[1]:.2f} "
          f"wz={command[2]:.2f}, {steps} steps) ---")
    print(f"  base height   : {stats['base_height_mean']:.3f} m "
          f"(std {stats['base_height_std']:.3f}, final {stats['base_height_final']:.3f})")
    print(f"  vx achieved   : {stats['vx_mean']:+.3f} m/s  (commanded {command[0]:+.2f})")
    print(f"  feet loaded   : {stats['contacts_mean']:.2f} / 4")
    print(f"  flight frac   : {stats['flight_fraction']:.3f}   <- all four feet airborne")
    print("  duty per foot : " + ", ".join(f"{d:.2f}" for d in stats["duty_per_foot"]))
    print(f"  |dof - default|: {stats['dof_dev_mean']:.3f} rad")
    print(f"  action rate   : {stats['action_rate_mean']:.4f}")
    print(f"  |action|      : mean {np.mean(act_absmean):.3f}, max {np.max(act_absmax):.3f} "
          f"(clip at {env.cfg.normalization.clip_actions})")
    ob = np.asarray(obs_blocks).max(axis=0)
    print("  obs |max| per block: ang_vel {:.2f}  gravity {:.2f}  cmd {:.2f}  "
          "dof_pos {:.2f}  dof_vel {:.2f}  prev_act {:.2f}".format(*ob))
    print(f"  terminated    : {stats['terminated']} / {stats['num_envs']} envs")
    return obs, history, stats


def unpermute_policy(actor_critic, perm):
    """Undo the import-time dof permutation, in place, for an A/B run."""
    from legged_gym.scripts.import_go2_rl_gym_policy import (
        apply_dof_permutation, inverse_permutation)
    sd = actor_critic.state_dict()
    apply_dof_permutation(sd, inverse_permutation(perm))
    actor_critic.load_state_dict(sd, strict=True)


def _get_args():
    """``get_args()`` plus this script's own flags.

    The shared parser uses strict ``parse_args``, so an unknown flag would abort
    it; strip ours from argv first rather than adding script-specific noise to
    every entry point's ``--help``.
    """
    import sys
    flag = "--compare-unpermuted"
    compare = flag in sys.argv
    if compare:
        sys.argv.remove(flag)
    # Our go2 URDF is not the one go2_rl_gym trained against: 42 links vs 29,
    # 16.09 kg vs 15.02 kg, and collision primitives up to 2x larger. This swaps
    # the asset so the difference can be measured instead of argued about.
    trace = 0
    for i, a in enumerate(list(sys.argv)):
        if a == "--trace":
            trace = int(sys.argv[i + 1])
            del sys.argv[i:i + 2]
            break
    asset_file = None
    for i, a in enumerate(list(sys.argv)):
        if a == "--asset-file":
            asset_file = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            break
    args = get_args()
    args.compare_unpermuted = compare
    args.asset_file = asset_file
    args.trace = trace
    # This script only ever prints numbers, and it is meant to run on headless
    # boxes; opening a viewer would just fail there.
    args.headless = True
    return args


def main():
    args = _get_args()
    if SIMULATOR == "genesis":
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _flat_env_overrides(env_cfg, num_envs=args.num_envs or 8)
    if args.asset_file:
        env_cfg.asset.file = args.asset_file
        print(f"[asset] overriding URDF -> {args.asset_file}")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # make_alg_runner only reads the checkpoint when runner.resume is set; play.py
    # forces it too. Without this the runner hands back a freshly initialized
    # network and every number below describes random weights, not the policy.
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    # Fail loudly rather than silently profiling noise: a loaded go2_rl_gym MoE
    # policy answers a plausible standing observation with |action| ~ 0.6, a
    # fresh one with ~0.03.
    probe_obs = torch.zeros(1, env_cfg.env.num_observations, device=env.device)
    probe_obs[0, 5] = -1.0  # projected gravity
    with torch.no_grad():
        probe = runner.alg.actor_critic.act_student(
            probe_obs, probe_obs.repeat(1, env_cfg.env.frame_stack)).abs().mean().item()
    print(f"loaded-policy check: |action| on a nominal standing obs = {probe:.3f}")
    if probe < 0.1:
        raise SystemExit(
            f"policy looks untrained (|action|={probe:.3f}); the checkpoint did "
            "not load -- check --load_run/--ckpt")

    obs, _, history, _ = env.get_observations()

    print(f"\n{'=' * 68}\nflat-ground diagnosis: {args.load_run} @ ckpt {args.ckpt}"
          f"\n  dt={env.dt:.4f}s  action_scale={env_cfg.control.action_scale}"
          f"  kp={env_cfg.control.stiffness}  kd={env_cfg.control.damping}"
          f"\n  domain_rand OFF, obs noise={env_cfg.noise.add_noise}\n{'=' * 68}")

    obs, history, _ = run_phase(env, policy, obs, history, 200, (0.0, 0.0, 0.0),
                                "STAND")
    obs, history, _ = run_phase(env, policy, obs, history, 400, (0.5, 0.0, 0.0),
                                "WALK 0.5 m/s", trace=args.trace)
    run_phase(env, policy, obs, history, 400, (1.0, 0.0, 0.0), "WALK 1.0 m/s")

    if args.compare_unpermuted:
        import os
        from legged_gym.utils.helpers import get_load_path
        path = get_load_path(os.path.join(LEGGED_GYM_ROOT_DIR, "logs",
                                          train_cfg.runner.experiment_name),
                             load_run=train_cfg.runner.load_run,
                             checkpoint=train_cfg.runner.checkpoint)
        perm = torch.load(path, map_location="cpu",
                          weights_only=False)["infos"].get("dof_permutation")
        if perm is None:
            print("\n[compare] checkpoint carries no dof_permutation; skipping")
            return
        print(f"\n{'=' * 68}\nA/B: same policy with the dof permutation UNDONE"
              f"\n(this is the wrong wiring on purpose -- it should look worse)"
              f"\n{'=' * 68}")
        unpermute_policy(runner.alg.actor_critic, perm)
        obs, _, history, _ = env.reset()
        obs, history, _ = run_phase(env, policy, obs, history, 200,
                                    (0.0, 0.0, 0.0), "STAND (unpermuted)")
        run_phase(env, policy, obs, history, 400, (0.5, 0.0, 0.0),
                  "WALK 0.5 m/s (unpermuted)")


if __name__ == "__main__":
    main()
