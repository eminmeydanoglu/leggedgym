"""MoE-CTS gating diagnostic probe (go2_moects).

Diagnoses why the StudentMoEEncoder softmax gate ``g`` looks near-uniform.
The encoder applies L2Norm AFTER the gate-weighted expert mix
(``rsl_rl/modules/moe_utils.py``), so expert i's true mixture contribution is
``g_i * ||e_i||``, not ``g_i``: routing can hide in the expert-output norm
axis even when ``g`` is uniform. This probe measures that and five related
questions (M1..M6, see ``run_analyze``).

Modes:
  collect  -- GPU/Genesis: roll out the STUDENT policy path on the training
              terrain/DR distribution and bank (obs, history, privileged_obs,
              commands, velocities, terrain id/level) samples.
  analyze  -- CPU-only: load the bank + checkpoint and compute M1..M6,
              writing results.json / REPORT.md next to samples.pt.

Import contract: module top level imports ONLY stdlib + numpy + torch so the
pure-math API below is loadable CPU-only without the legged_gym/genesis
package chain (unit tests import this file directly via importlib). Every
legged_gym / rsl_rl / genesis import is lazy, inside functions.

Example (GPU box):
    PYTHONPATH=. SIMULATOR=genesis .venv/bin/python \
        legged_gym/scripts/eval/probe_moe_gate.py --mode collect \
        --num_envs 384 --target_samples 35000

Example (CPU analysis):
    PYTHONPATH=. SIMULATOR=genesis .venv/bin/python \
        legged_gym/scripts/eval/probe_moe_gate.py --mode analyze
"""

import argparse
import copy
import json
import math
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Pinned observation/model contract (go2_moects, see run_manifest.json and
# tests/test_moects_telemetry.py::_make_ac)
# ---------------------------------------------------------------------------
NUM_OBS = 45
NUM_PRIVILEGED = 263
NUM_HISTORY = 225          # frame_stack 5 x 45
NUM_LATENT = 32
NUM_CRITIC = 263
NUM_ACTIONS = 12
EXPERT_NUM = 8

TASK_DEFAULT = "go2_moects"
LOAD_RUN_DEFAULT = "Aug03_12-01-45_moe_cts_genesis"
EXPERIMENT_DEFAULT = "go2_moects"
OUT_ROOT_DEFAULT = "logs/eval/moe_gate_probe"
TREND_CKPTS_DEFAULT = "500,2500,5000,7500"

# Fallback only; analyze prefers rsl_rl.utils.moe_terrain_gate.TERRAIN_NAMES.
TERRAIN_NAMES_FALLBACK = (
    "wave", "slope", "rough_slope", "stairs_up", "stairs_down",
    "obstacles", "stepping_stones", "gap", "flat",
)

ANALYZE_BATCH = 8192


# ---------------------------------------------------------------------------
# SHARED PURE-MATH API (unit-tested contract -- signatures are frozen)
# ---------------------------------------------------------------------------

def effective_weights(g, E):
    """(N,K),(N,K,D) -> (N,K): w = g*||E||_dim2, row-normalized.

    The denominator is clamped away from zero so a degenerate all-zero expert
    row yields a zero row instead of NaNs (rows are otherwise stochastic).
    """
    w = g * torch.linalg.vector_norm(E, dim=2)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return w / denom


def row_entropy(p, eps=1e-12):
    """(N,K) probs -> (N,) natural-log entropy."""
    return -(p * torch.log(p + eps)).sum(dim=-1)


def effective_experts(p):
    """(N,K) probs -> (N,) perplexity of the distribution: exp(H(p))."""
    return torch.exp(row_entropy(p))


def pairwise_cosine_mean(F):
    """(N,K,D) -> (K,K): mean over samples of cos(F[:,i,:], F[:,j,:])."""
    Fn = torch.nn.functional.normalize(F, dim=-1)
    return (Fn @ Fn.transpose(1, 2)).mean(dim=0)


def linear_cka(X, Y):
    """(N,D),(N,D) -> float: centered linear CKA.

    ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F), denominator guarded.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    num = (Yc.t() @ Xc).square().sum()
    den = (Xc.t() @ Xc).square().sum().sqrt() * (Yc.t() @ Yc).square().sum().sqrt()
    return float(num / den.clamp_min(1e-12))


def responsibilities(d, tau):
    """(N,K), float -> softmax(-d/tau, dim=1): distance-based expert posteriors."""
    return torch.softmax(-d / tau, dim=1)


def _labels_to_zero_based(x):
    x = x.reshape(-1).long()
    return x - x.min() if x.numel() else x


def _entropy_from_counts(counts):
    p = counts.float()
    p = p[p > 0] / counts.sum().clamp_min(1)
    return float(-(p * torch.log(p)).sum())


def normalized_mi(a, b):
    """1-D int tensors -> MI(a;b)/min(H(a),H(b)); 0.0 if either entropy is 0."""
    a, b = _labels_to_zero_based(a), _labels_to_zero_based(b)
    if a.numel() == 0 or a.numel() != b.numel():
        return 0.0
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    ca = torch.bincount(a, minlength=na)
    cb = torch.bincount(b, minlength=nb)
    joint = torch.bincount(a * nb + b, minlength=na * nb)
    ha, hb = _entropy_from_counts(ca), _entropy_from_counts(cb)
    if ha <= 0.0 or hb <= 0.0:
        return 0.0
    mi = ha + hb - _entropy_from_counts(joint)
    return float(mi / min(ha, hb))


def chi2_stat(a, b):
    """1-D int tensors -> (chi2: float, dof: int); pure torch, no scipy."""
    a, b = _labels_to_zero_based(a), _labels_to_zero_based(b)
    if a.numel() == 0 or a.numel() != b.numel():
        return 0.0, 0
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    obs = torch.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).float()
    n = obs.sum()
    rows, cols = obs.sum(dim=1, keepdim=True), obs.sum(dim=0, keepdim=True)
    exp = (rows @ cols) / n.clamp_min(1.0)
    mask = exp > 0
    chi2 = (((obs - exp).square() / exp.clamp_min(1e-12)) * mask).sum()
    dof = (na - 1) * (nb - 1)
    return float(chi2), int(dof)


def ablation_weights(g, d, variant, generator=None):
    """(N,K) gate weights / (N,K) distances -> (N,K) row-stochastic ablated weights.

    variant in {'learned','uniform','shuffled','top1','oracle'}:
      learned  -- g unchanged
      uniform  -- 1/K everywhere
      shuffled -- g with rows permuted across samples (breaks sample alignment;
                  deterministic when `generator` is a seeded torch.Generator)
      top1     -- one-hot at argmax g
      oracle   -- one-hot at argmin d (best-matching expert per sample)
    """
    n, k = g.shape
    if variant == "learned":
        return g
    if variant == "uniform":
        return torch.full_like(g, 1.0 / k)
    if variant == "shuffled":
        perm = torch.randperm(n, generator=generator).to(g.device)
        return g[perm]
    if variant == "top1":
        w = torch.zeros_like(g)
        w.scatter_(1, g.argmax(dim=1, keepdim=True), 1.0)
        return w
    if variant == "oracle":
        w = torch.zeros_like(d)
        w.scatter_(1, d.argmin(dim=1, keepdim=True), 1.0)
        return w
    raise ValueError(f"unknown ablation variant: {variant!r}")


def mix_latent(E, w):
    """(N,K,D),(N,K) -> L2-normalized gate-weighted mix (the encoder's latent)."""
    return torch.nn.functional.normalize((w.unsqueeze(-1) * E).sum(dim=1), dim=-1)


# ---------------------------------------------------------------------------
# small shared helpers (pure)
# ---------------------------------------------------------------------------

def _offdiag_mean_abs(M):
    """Mean |off-diagonal| of a square matrix tensor."""
    k = M.shape[0]
    mask = ~torch.eye(k, dtype=torch.bool, device=M.device)
    return float(M[mask].abs().mean())


def _jsonable(obj):
    if torch.is_tensor(obj):
        return _jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _batched(x, batch=ANALYZE_BATCH):
    for i in range(0, x.shape[0], batch):
        yield x[i:i + batch]


# ---------------------------------------------------------------------------
# MODE collect (GPU/Genesis; heavy imports are lazy)
# ---------------------------------------------------------------------------

def _override_cfg_for_probe(env_cfg, cli):
    """Training-fidelity override: ONLY env count/seed/auto_reset/debug.

    Unlike sweep.override_cfg_for_eval this keeps the wty moe_grid terrain and
    its curriculum config, the vendored command ranges/curricula, and the full
    domain randomization exactly as in training. --nominal opts out of DR.
    """
    env_cfg.env.num_envs = cli.num_envs
    env_cfg.env.auto_reset = True
    env_cfg.env.debug = False
    env_cfg.seed = cli.seed
    for axis_name in getattr(cli, "prepare_axes", ()):
        from legged_gym.scripts.eval.dr_axes import get_axis
        get_axis(axis_name).prepare_cfg(env_cfg)
    if cli.nominal:
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_com_displacement = False
        env_cfg.domain_rand.push_robots = False


def _build_env_and_policy(cli):
    """Env + runner build, mirroring probe_rma_latent.build_env_and_policy.

    Deviations (both forced by the constraints of this probe):
      * log_root=None + train_cfg.runner.resume=False, then a manual
        runner.load(ckpt, load_optimizer=False, load_env_curriculum=False).
        With the default log_root, runner.load() would carry best_tracking.pt
        forward into a NEW directory under logs/go2_moects/
        (on_policy_runner.py:874-882) -- writes into the live training folder
        are forbidden. log_dir=None skips that branch.
      * load_env_curriculum=False: the checkpoint's env_curriculum_state has
        training geometry (num_envs=8192) and the env-side loader fails closed
        on mismatch (wty_curriculum_mixin.py:332-345). The probe restores the
        level distribution itself in _restore_curriculum_state.
    """
    from legged_gym import SIMULATOR, LEGGED_GYM_ROOT_DIR  # noqa: N813
    if SIMULATOR == "genesis":
        import genesis as gs
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level="warning")
        # gs.init installs a global torch default device of cuda; factory calls
        # without an explicit device (and bare CPU Generators) then misfire.
        # Same trap is documented in play.py:_drop_torch_default_device_mode.
        torch.set_default_device(None)
    import legged_gym.envs  # noqa: F401  (importing the package registers all tasks)
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import get_load_path
    from legged_gym.scripts.eval.sweep import resolve_load_run, make_registry_args

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)
    _override_cfg_for_probe(env_cfg, cli)

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    chosen_run = resolve_load_run(log_root, train_cfg.runner.run_name, cli.load_run)
    ckpt_path = get_load_path(log_root, load_run=chosen_run, checkpoint=cli.ckpt)
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    print(f"[probe-collect] checkpoint: {ckpt_path}")

    cli.load_run = chosen_run
    reg_args = make_registry_args(cli)
    # make_registry_args hardcodes resume=True (sweep convention); the probe
    # loads the checkpoint manually below, and update_cfg_from_args inside
    # make_alg_runner would flip train_cfg.runner.resume back to True and make
    # the registry attempt its own get_load_path(log_root=None) load.
    reg_args.resume = False
    env, _ = task_registry.make_env(name=cli.task, args=reg_args, env_cfg=env_cfg)

    train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=cli.task, args=reg_args, train_cfg=train_cfg, log_root=None)
    infos = ppo_runner.load(ckpt_path, load_optimizer=False, load_env_curriculum=False)
    ac = ppo_runner.alg.actor_critic
    ac.eval()

    # Mirror the learn()-time resume protocol: with common_step_counter just
    # restored by runner.load(), land every ratio-based curriculum (command
    # ranges, zero-command probability, reward ramps) on the iter-7500 stage.
    if hasattr(env, "set_wty_total_iterations"):
        env.set_wty_total_iterations(int(train_cfg.runner.max_iterations))

    ctx = dict(env_cfg=env_cfg, train_cfg=train_cfg, chosen_run=chosen_run,
               ckpt_path=ckpt_path, ckpt_tag=ckpt_tag,
               ckpt_iter=int(ppo_runner.current_learning_iteration),
               log_root=log_root, infos=infos)
    return env, ac, ctx


def _restore_curriculum_state(env, ckpt_path, seed):
    """Restore iter-7500 per-env terrain levels from env_curriculum_state.

    Exact path (probe env geometry == training geometry): delegate to the
    env-side loader (wty_curriculum_mixin.load_curriculum_state_dict).
    Mismatched geometry (e.g. 384 probe envs vs 8192 training envs): resample
    each probe env's level from the checkpoint's empirical level distribution
    OF ITS OWN TERRAIN COLUMN (seeded), then refresh env origins exactly like
    the mixin's loader does. Returns a meta dict describing what happened.
    """
    info = {"method": None, "reason": None, "ckpt_geometry": None,
            "level_mean": None, "level_histogram": None,
            "common_step_counter": int(getattr(env, "common_step_counter", -1))}
    load_fn = getattr(env, "load_curriculum_state_dict", None)
    if not callable(load_fn) or getattr(env, "wty_terrain_ids", None) is None:
        info["method"] = "unavailable"
        info["reason"] = ("env has no wty curriculum state (non-moe_grid terrain "
                          "or curriculum inactive); levels stay at fresh round-robin")
        return info

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("env_curriculum_state")
    if state is None:
        info["method"] = "unavailable"
        info["reason"] = "checkpoint carries no env_curriculum_state key"
        return info
    info["ckpt_geometry"] = {"num_envs": int(state["num_envs"]),
                             "num_cols": int(state["num_cols"]),
                             "version": int(state.get("version", -1))}

    same_geometry = (
        int(state["num_envs"]) == int(env.num_envs)
        and int(state["num_cols"]) == int(env.cfg.terrain.num_cols)
        and torch.equal(state["terrain_types"].long().to(env.device),
                        env.simulator.terrain_types.long())
    )
    if same_geometry:
        load_fn(state)  # the env-side loader; refreshes env origins itself
        info["method"] = "exact"
        info["reason"] = "probe env geometry matches the checkpoint"
    else:
        # Per-column resampling: column c's probe envs draw levels (with
        # replacement) from the checkpoint levels of training envs in column
        # c. Reproduces the marginal per-terrain level distribution at iter
        # 7500; per-env identity is impossible across different num_envs.
        gen = torch.Generator().manual_seed(seed)
        levels_ck = state["terrain_levels"].long().cpu()
        types_ck = state["terrain_types"].long().cpu()
        types_env = env.simulator.terrain_types.long().cpu()
        new_levels = torch.zeros(env.num_envs, dtype=torch.long)
        global_pool = levels_ck
        for c in range(int(env.cfg.terrain.num_cols)):
            envs_c = (types_env == c).nonzero(as_tuple=False).flatten()
            if envs_c.numel() == 0:
                continue
            pool = levels_ck[types_ck == c]
            if pool.numel() == 0:
                pool = global_pool  # column absent in ckpt: global fallback
            draw = torch.randint(pool.numel(), (envs_c.numel(),), generator=gen)
            new_levels[envs_c] = pool[draw]
        sim = env.simulator
        sim._terrain_levels[:] = new_levels.to(
            device=sim._terrain_levels.device, dtype=sim._terrain_levels.dtype)
        sim._env_origins[:] = sim._terrain_origins[
            sim.terrain_levels, sim.terrain_types]
        info["method"] = "resampled_per_column"
        info["reason"] = (
            f"geometry mismatch (checkpoint num_envs={int(state['num_envs'])} vs probe "
            f"num_envs={env.num_envs}); per-env levels cannot be mapped, so each probe "
            "env's level was drawn from the checkpoint's empirical level distribution "
            "of its own terrain column (seeded); env origins refreshed like "
            "wty_curriculum_mixin.load_curriculum_state_dict does")
    levels = env.simulator.terrain_levels.detach().cpu().long()
    info["level_mean"] = float(levels.float().mean())
    info["level_histogram"] = torch.bincount(
        levels, minlength=int(env.cfg.terrain.num_rows)).tolist()
    print(f"[probe-collect] curriculum restore: {info['method']} "
          f"(level mean {info['level_mean']:.2f})")
    return info


def run_collect(cli):
    t0 = time.time()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: N813
    from legged_gym.scripts.eval.sweep import git_commit
    try:
        from rsl_rl.utils.moe_terrain_gate import TERRAIN_NAMES
    except Exception:
        TERRAIN_NAMES = TERRAIN_NAMES_FALLBACK

    env, ac, ctx = _build_env_and_policy(cli)
    restore_info = _restore_curriculum_state(env, ctx["ckpt_path"], cli.seed)
    device = env.device
    num_envs = env.num_envs

    # per-env semantic terrain ids are fixed (terrain columns never change)
    tids = getattr(env, "wty_terrain_ids", None)
    if tids is None:
        tids = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    else:
        tids = tids.detach().long()
    active_types = sorted(t for t in tids.cpu().unique().tolist() if 0 <= t < len(TERRAIN_NAMES))
    never_seen = [TERRAIN_NAMES[t] for t in range(len(TERRAIN_NAMES)) if t not in active_types]
    if never_seen:
        print(f"[probe-collect] terrain types with ZERO envs (curriculum layout): {never_seen}")

    keys = ["obs", "obs_history", "privileged_obs", "commands",
            "base_lin_vel", "base_ang_vel", "terrain_id", "terrain_level"]
    bank = {k: [] for k in keys}
    counts = torch.zeros(len(TERRAIN_NAMES), dtype=torch.long)
    total = 0
    recorded = 0
    max_record_steps = max(1, math.ceil(cli.max_samples / num_envs))

    env.reset()
    obs, priv_obs, history, _critic = env.get_observations()
    with torch.no_grad():
        for _ in range(cli.warmup):
            actions = ac.act_student(obs.detach(), history.detach())
            obs, priv_obs, history, _c, _r, _d, _i = env.step(actions.detach())

    def _stop_due():
        if total < cli.target_samples:
            return False
        return all(int(counts[t]) >= cli.per_terrain_min for t in active_types)

    step = 0
    stop_reason = f"max_samples hard cap ({cli.max_samples})"
    with torch.no_grad():
        while recorded < max_record_steps:
            actions = ac.act_student(obs.detach(), history.detach())
            if step % cli.sample_every == 0:
                bank["obs"].append(obs.detach().float().cpu())
                bank["obs_history"].append(history.detach().float().cpu())
                bank["privileged_obs"].append(priv_obs.detach().float().cpu())
                bank["commands"].append(env.commands[:, :3].detach().float().cpu())
                bank["base_lin_vel"].append(env.simulator.base_lin_vel.detach().float().cpu())
                bank["base_ang_vel"].append(env.simulator.base_ang_vel.detach().float().cpu())
                bank["terrain_id"].append(tids.to(torch.int8).cpu())
                levels = getattr(env.simulator, "terrain_levels", None)
                if levels is None:
                    levels = torch.full((num_envs,), -1, dtype=torch.long, device=device)
                bank["terrain_level"].append(levels.detach().to(torch.int8).cpu())
                valid = tids.cpu()[(tids.cpu() >= 0) & (tids.cpu() < len(TERRAIN_NAMES))]
                counts += torch.bincount(valid, minlength=len(TERRAIN_NAMES))
                total += num_envs
                recorded += 1
                if recorded % 10 == 0 or _stop_due():
                    print(f"[probe-collect] step={step} total={total} "
                          f"counts={ {TERRAIN_NAMES[t]: int(counts[t]) for t in active_types} }")
                if _stop_due():
                    stop_reason = (f"target_samples ({cli.target_samples}) reached AND every "
                                   f"active terrain type >= per_terrain_min ({cli.per_terrain_min})")
                    break
            obs, priv_obs, history, _c, _r, _d, _i = env.step(actions.detach())
            step += 1

    samples = {k: torch.cat(v, dim=0) for k, v in bank.items()}
    assert samples["obs"].shape[0] == total

    out_root = cli.out_root if os.path.isabs(cli.out_root) else os.path.join(
        LEGGED_GYM_ROOT_DIR, cli.out_root)
    out_dir = os.path.join(out_root, ctx["chosen_run"], ctx["ckpt_tag"])
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, "samples.pt")
    torch.save(samples, samples_path)

    env_cfg = ctx["env_cfg"]
    dr = env_cfg.domain_rand
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=LEGGED_GYM_ROOT_DIR,
            stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        dirty = None
    meta = {
        "probe": "moe_gate", "mode": "collect",
        "created": datetime.now().isoformat(timespec="seconds"),
        "task": cli.task, "load_run": ctx["chosen_run"],
        "ckpt_path": ctx["ckpt_path"], "ckpt_tag": ctx["ckpt_tag"],
        "ckpt_iter": ctx["ckpt_iter"], "ckpt_spec": cli.ckpt,
        "seed": cli.seed, "git_commit": git_commit(), "git_dirty": dirty,
        "env": {
            "num_envs": num_envs, "auto_reset": True,
            "nominal": bool(cli.nominal),
            "domain_rand": {
                "randomize_friction": bool(dr.randomize_friction),
                "friction_range": list(dr.friction_range),
                "randomize_base_mass": bool(dr.randomize_base_mass),
                "added_mass_range": list(dr.added_mass_range),
                "randomize_com_displacement": bool(dr.randomize_com_displacement),
                "push_robots": bool(dr.push_robots),
            },
            "terrain": {
                "mesh_type": env_cfg.terrain.mesh_type,
                "moe_grid": bool(getattr(env_cfg.terrain, "moe_grid", False)),
                "num_rows": int(env_cfg.terrain.num_rows),
                "num_cols": int(env_cfg.terrain.num_cols),
                "max_init_terrain_level": int(getattr(env_cfg.terrain, "max_init_terrain_level", -1)),
                "curriculum_flag": bool(env_cfg.terrain.curriculum),
            },
            "commands_curriculum": bool(env_cfg.commands.curriculum),
        },
        "curriculum_restore": restore_info,
        "collection": {
            "warmup": cli.warmup, "sample_every": cli.sample_every,
            "target_samples": cli.target_samples,
            "per_terrain_min": cli.per_terrain_min,
            "max_samples": cli.max_samples,
            "recorded_steps": recorded, "total_samples": total,
            "stop_reason": stop_reason,
            "terrain_names": list(TERRAIN_NAMES),
            "active_terrain_types": [TERRAIN_NAMES[t] for t in active_types],
            "never_appearing_terrain_types": never_seen,
            "per_terrain_counts": {TERRAIN_NAMES[t]: int(counts[t])
                                   for t in range(len(TERRAIN_NAMES))},
        },
        "wall_time_s": round(time.time() - t0, 2),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(_jsonable(meta), f, indent=2)
    print(f"[probe-collect] stop: {stop_reason}")
    print(f"[probe-collect] saved {total} samples -> {samples_path}")
    print(f"[probe-collect] meta -> {meta_path}")


# ---------------------------------------------------------------------------
# MODE analyze (CPU-only: stdlib/numpy/torch + rsl_rl.modules +
# rsl_rl.utils.moe_terrain_gate; NEVER legged_gym.envs / genesis)
# ---------------------------------------------------------------------------

def _resolve_analyze_paths(cli):
    """stdlib-only path resolution (no legged_gym import in analyze)."""
    repo = _repo_root()
    run_dir = cli.run_dir or os.path.join(
        repo, "logs", cli.experiment, cli.load_run)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    if int(cli.ckpt) == -1:
        cands = []
        for f in os.listdir(run_dir):
            if f.startswith("model_") and f.endswith(".pt"):
                try:
                    cands.append((int(f[len("model_"):-len(".pt")]), f))
                except ValueError:
                    pass
        if not cands:
            raise FileNotFoundError(f"no model_*.pt in {run_dir}")
        ckpt_name = sorted(cands)[-1][1]
    else:
        ckpt_name = f"model_{int(cli.ckpt)}.pt"
    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt_tag = os.path.splitext(ckpt_name)[0]
    out_root = cli.out_root if os.path.isabs(cli.out_root) else os.path.join(
        repo, cli.out_root)
    samples_path = cli.samples or os.path.join(
        out_root, cli.load_run, ckpt_tag, "samples.pt")
    if not os.path.isfile(samples_path):
        raise FileNotFoundError(
            f"samples not found: {samples_path} (pass --samples explicitly)")
    return samples_path, ckpt_path, os.path.dirname(samples_path), ckpt_tag


def _load_ac(ckpt_path):
    """Build the pinned-dims ActorCriticMoECTS on CPU and load the weights."""
    from rsl_rl.modules import ActorCriticMoECTS
    ac = ActorCriticMoECTS(
        NUM_OBS, NUM_ACTIONS, NUM_PRIVILEGED, NUM_HISTORY, NUM_LATENT, NUM_CRITIC,
        expert_num=EXPERT_NUM, student_encoder_hidden_dims=[512, 256, 256],
        norm_type="l2norm", init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        privilege_encoder_hidden_dims=[512, 256])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ac.load_state_dict(ckpt["model_state_dict"], strict=True)
    ac.eval()
    return ac, ckpt


def _forward_bank(ac, hist, priv):
    """Batched no-grad forward: E (N,K,D), g (N,K), t (N,D)."""
    Es, gs_, ts = [], [], []
    with torch.no_grad():
        for hb, pb in zip(_batched(hist), _batched(priv)):
            Es.append(ac.history_encoder.moe.experts(hb))
            gs_.append(ac.history_encoder.moe.gating_network(hb))
            ts.append(ac.privilege_encoder(pb))
    return torch.cat(Es), torch.cat(gs_), torch.cat(ts)


def _expert_head_param_cos(sd):
    """(K,K) pairwise cosine of flattened Conv1d expert head matrices.

    Expert i's head = weight[i*32:(i+1)*32, :, 0] (32x256 slice of the
    grouped Conv1d); the backbone is shared, so only heads differ.
    """
    W = sd["history_encoder.moe.experts.experts.weight"]
    heads = torch.stack([
        W[i * NUM_LATENT:(i + 1) * NUM_LATENT, :, 0].reshape(-1)
        for i in range(EXPERT_NUM)])
    Hn = torch.nn.functional.normalize(heads, dim=-1)
    return Hn @ Hn.t()


def _m1(g, E):
    n_i = torch.linalg.vector_norm(E, dim=2)             # (N,K)
    w_eff = effective_weights(g, E)
    out = {"per_expert_norm_mean": n_i.mean(dim=0),
           "per_expert_norm_std": n_i.std(dim=0)}
    for name, p in (("g", g), ("w_eff", w_eff)):
        ent = row_entropy(p)
        out[name] = {
            "mean_entropy": float(ent.mean()),
            "mean_effective_experts": float(effective_experts(p).mean()),
            "mean_max_weight": float(p.max(dim=1).values.mean()),
        }
    eff_g, eff_w = out["g"]["mean_effective_experts"], out["w_eff"]["mean_effective_experts"]
    # "Clearly lower": >25% drop in effective expert count after norm-reweighting
    # (the motivating example: 7.4 -> <5 is a ~33% drop).
    hidden = eff_w < 0.75 * eff_g
    out["gate"] = {
        "rule": "hidden norm-axis routing EXISTS iff mean_eff_experts(w_eff) < 0.75 * mean_eff_experts(g)",
        "drop": eff_g - eff_w, "hidden_norm_axis_routing": bool(hidden),
        "verdict": (
            f"effective_experts drops {eff_g:.2f} (raw g) -> {eff_w:.2f} (norm-weighted): "
            + ("HIDDEN NORM-AXIS ROUTING EXISTS -- the 'gate is uniform / no "
               "specialization' diagnosis is OVERTURNED" if hidden else
               "no meaningful drop: routing really is near-uniform, the "
               "'no specialization' hypothesis stands"))
    }
    return out


def _m2(ac, E, hist, ckpt, trend_paths):
    En = torch.nn.functional.normalize(E, dim=-1)
    func_cos = pairwise_cosine_mean(En)
    cka = torch.tensor([[linear_cka(E[:, i, :], E[:, j, :])
                         for j in range(EXPERT_NUM)] for i in range(EXPERT_NUM)])
    param_cos = _expert_head_param_cos(ckpt["model_state_dict"])
    func_off, param_off = _offdiag_mean_abs(func_cos), _offdiag_mean_abs(param_cos)
    out = {
        "functional_cosine": func_cos, "linear_cka": cka,
        "param_head_cosine": param_cos,
        "mean_abs_offdiag_functional": func_off,
        "mean_abs_offdiag_param": param_off,
        "gate": {
            "rule": "functional ~1 -> experts are effectively copies; low param cosine "
                    "with high functional cosine corrects the parameter-space "
                    "'not copies' conclusion",
            "experts_effectively_copies": bool(func_off > 0.95),
            "param_functional_contradiction": bool(param_off < 0.3 and func_off > 0.9),
            "verdict": (
                f"mean|offdiag| functional cos={func_off:.3f}, param cos={param_off:.3f}: "
                + ("experts are effectively COPIES functionally" if func_off > 0.95
                   else "experts are functionally DISTINCT")
                + ("; param-space said 'not copies' but function-space says copies -- "
                   "the parameter-space conclusion is explicitly CORRECTED"
                   if (param_off < 0.3 and func_off > 0.9) else ""))
        },
        "trend": [],
    }
    for ck, path in trend_paths:
        entry = {"ckpt": ck, "path": path}
        if path is None:
            entry["skipped"] = "checkpoint file missing"
        else:
            try:
                ckpt_k = torch.load(path, map_location="cpu", weights_only=False)
                pc = _expert_head_param_cos(ckpt_k["model_state_dict"])
                ac_k, _ = _load_ac(path)
                with torch.no_grad():
                    Ek = torch.cat([ac_k.history_encoder.moe.experts(hb)
                                    for hb in _batched(hist)])
                fc = pairwise_cosine_mean(torch.nn.functional.normalize(Ek, dim=-1))
                entry.update(mean_abs_offdiag_param=_offdiag_mean_abs(pc),
                             mean_abs_offdiag_functional=_offdiag_mean_abs(fc))
                del ac_k, Ek
            except Exception as exc:  # trend is secondary; never kill analyze
                entry["skipped"] = f"error: {exc!r}"
        out["trend"].append(entry)
        print(f"[probe-analyze] trend ckpt={ck}: {entry.get('mean_abs_offdiag_functional', entry.get('skipped'))}")
    return out


def _m3(E, t):
    En = torch.nn.functional.normalize(E, dim=-1)
    d = ((En - t.unsqueeze(1)) ** 2).sum(dim=-1)          # (N,K) in [0,4]
    std = d.std(dim=1)                                    # per-sample spread over experts
    median_std = float(std.median())
    rng = (d.max(dim=1).values - d.min(dim=1).values) / d.mean(dim=1).clamp_min(1e-12)
    taus = [0.2, 0.5, median_std, 0.5 * median_std, 0.1 * median_std]
    labels = ["0.2", "0.5", "1.0xmedian_std", "0.5xmedian_std", "0.1xmedian_std"]
    table = []
    for lab, tau in zip(labels, taus):
        r = responsibilities(d, tau)
        table.append({"tau_label": lab, "tau": float(tau),
                      "mean_max_r": float(r.max(dim=1).values.mean()),
                      "mean_entropy_r": float(row_entropy(r).mean())})
    uniform_like = 0.125 <= table[0]["mean_max_r"] <= 0.15
    # concrete c: among c in {1.0, 0.5, 0.1} (x median_std), pick the one whose
    # mean max responsibility is closest to 0.5 (decisive but not one-hot).
    cands = table[2:]
    best = min(cands, key=lambda row: abs(row["mean_max_r"] - 0.5))
    c_val = float(best["tau_label"].split("x")[0])
    out = {
        "per_expert_mean_d": d.mean(dim=0),
        "per_sample_std": {"median": median_std,
                           "p10": float(torch.quantile(std, 0.1)),
                           "p90": float(torch.quantile(std, 0.9))},
        "range_over_mean": {"median": float(rng.median()), "mean": float(rng.mean())},
        "tau_table": table,
        "gate": {
            "rule": "tau=0.2 yields mean max r in [0.125,0.15] (~uniform over 8) -> tau too large",
            "tau_0.2_mean_max_r": table[0]["mean_max_r"],
            "tau_0.2_uniform_like": bool(uniform_like),
            "recommended_c": c_val,
            "recommended_tau": float(best["tau"]),
            "verdict": (
                f"tau=0.2 -> mean max r = {table[0]['mean_max_r']:.3f} "
                + ("(~uniform 1/8=0.125): tau=0.2 is TOO LARGE, the gate sees an "
                   "almost flat responsibility landscape; " if uniform_like else
                   "(decisive, not flat); ")
                + f"recommend tau* = {c_val} x median_std = {best['tau']:.4f} "
                  f"(mean max r {best['mean_max_r']:.3f})")
        },
    }
    return out


def _m4(ac, E, g, d, t, obs, seed):
    gen = torch.Generator().manual_seed(seed)
    variants = ["learned", "uniform", "shuffled", "top1", "oracle"]
    per_variant = {}
    with torch.no_grad():
        teacher_action = torch.cat([ac.actor(torch.cat([tb, ob], dim=-1))
                                    for tb, ob in zip(_batched(t), _batched(obs))])
        for v in variants:
            w = ablation_weights(g, d, v, generator=gen)
            z = mix_latent(E, w)
            mse = float(((z - t) ** 2).sum(dim=-1).mean())
            cos = float(torch.nn.functional.cosine_similarity(z, t, dim=-1).mean())
            acts = torch.cat([ac.actor(torch.cat([zb, ob], dim=-1))
                              for zb, ob in zip(_batched(z), _batched(obs))])
            gap = float(((acts - teacher_action) ** 2).sum(dim=-1).mean())
            per_variant[v] = {"latent_mse": mse, "latent_cos": cos, "action_gap": gap}
            print(f"[probe-analyze] M4 {v:8s}: mse={mse:.4f} cos={cos:.4f} action_gap={gap:.5f}")
    def _close(a, b, rtol=0.05):
        return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-12)
    L, U, S, O = (per_variant[k]["action_gap"] for k in ("learned", "uniform", "shuffled", "oracle"))
    gate_useless = _close(L, U) and _close(L, S)
    diversity_problem = _close(O, L) and not gate_useless
    gate_helps = L < U and not _close(L, U)
    verdict = (
        f"action_gap: learned={L:.5f} uniform={U:.5f} shuffled={S:.5f} top1={per_variant['top1']['action_gap']:.5f} "
        f"oracle={O:.5f}: "
        + ("learned ~= uniform ~= shuffled -> the gate contributes NOTHING"
           if gate_useless else
           "oracle ~= learned -> even perfect routing gains nothing; the problem is "
           "EXPERT DIVERSITY, not the gate" if diversity_problem else
           "learned < uniform -> the gate GENUINELY HELPS" if gate_helps else
           "mixed picture (see numbers)"))
    return {"per_variant": per_variant,
            "gate": {"rule": "compare action_gap across variants (~ = within 5%)",
                     "gate_contributes_nothing": bool(gate_useless),
                     "diversity_problem_not_gate": bool(diversity_problem),
                     "gate_genuinely_helps": bool(gate_helps),
                     "verdict": verdict}}


def _bin_vx(vx):
    b = torch.zeros(vx.shape, dtype=torch.long)
    b[vx < -0.1] = 0
    b[(vx >= -0.1) & (vx <= 0.1)] = 1
    b[(vx > 0.1) & (vx <= 0.6)] = 2
    b[vx > 0.6] = 3
    return b


def _bin_yaw(yaw):
    a = yaw.abs()
    b = torch.zeros(yaw.shape, dtype=torch.long)
    b[a <= 0.1] = 0
    b[(a > 0.1) & (a <= 0.5)] = 1
    b[a > 0.5] = 2
    return b


def _contingency(a, b, n_rows=None, n_cols=None):
    """(N,),(N,) int labels -> (n_rows, n_cols) count table.

    Labels are used as-is when non-negative (never remapped, so class indices
    stay aligned with a fixed class space); a negative minimum (e.g. -1 for
    "unknown") is shifted to 0. Out-of-range labels are dropped defensively.
    """
    a = a.reshape(-1).long()
    b = b.reshape(-1).long()
    if a.numel() == 0:
        return torch.zeros(n_rows or 1, n_cols or 1, dtype=torch.long)
    a = a - min(0, int(a.min()))
    b = b - min(0, int(b.min()))
    na = n_rows or (int(a.max()) + 1)
    nb = n_cols or (int(b.max()) + 1)
    valid = (a < na) & (b < nb)
    idx = a[valid] * nb + b[valid]
    return torch.bincount(idx, minlength=na * nb).reshape(na, nb)


def _m5(g, d, terrain_id, terrain_level, commands, per_terrain_min, terrain_names):
    best_i = d.argmin(dim=1)
    gate_i = g.argmax(dim=1)
    vx_bin = _bin_vx(commands[:, 0])
    yaw_bin = _bin_yaw(commands[:, 2])
    factors = {"terrain_id": terrain_id.long(), "terrain_level": terrain_level.long(),
               "vx_bin": vx_bin, "yaw_bin": yaw_bin}
    pairings = {}
    for src_name, src in (("best_expert", best_i), ("gate_argmax", gate_i)):
        for fac_name, fac in factors.items():
            nmi = normalized_mi(src, fac)
            chi2, dof = chi2_stat(src, fac)
            pairings[f"{src_name}_vs_{fac_name}"] = {
                "normalized_mi": nmi, "chi2": chi2, "dof": dof,
                "contingency": _contingency(src, fac)}
    counts = {terrain_names[t]: int((terrain_id == t).sum())
              for t in range(len(terrain_names))}
    unreliable = [name for name, c in counts.items() if c < per_terrain_min]
    max_nmi = max(v["normalized_mi"] for v in pairings.values())
    out = {
        "pairings": pairings,
        "per_terrain_counts": counts,
        "unreliable_terrains": unreliable,
        "unreliable_rule": f"terrain types with < {per_terrain_min} samples",
        "gate": {
            "rule": "max normalized MI ~ 0 (< 0.05) -> no exploitable regime structure",
            "max_normalized_mi": max_nmi,
            "no_regime_structure": bool(max_nmi < 0.05),
            "verdict": (f"max normalized MI over all pairings = {max_nmi:.4f}: "
                        + ("NO exploitable regime structure (gate argmax and best-expert "
                           "are both ~independent of terrain/level/command)"
                           if max_nmi < 0.05 else
                           "some regime structure exists -- see pairings")),
        },
    }
    return out


def _make_probe_mlp(in_dim, out_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, out_dim))


def _train_probe(X, y, seed, task, num_classes=None, max_epochs=200, patience=20,
                 batch=1024, lr=1e-3):
    """Small CPU MLP probe: 80/20 split, Adam, early stopping on val loss."""
    gen = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=gen)
    n_tr = int(0.8 * n)
    tr, va = perm[:n_tr], perm[n_tr:]
    mu, sd = X[tr].mean(dim=0, keepdim=True), X[tr].std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sd
    out_dim = num_classes if task == "cls" else 1
    model = _make_probe_mlp(X.shape[1], out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss() if task == "cls" else torch.nn.MSELoss()
    y_tr, y_va = (y[tr], y[va]) if task == "cls" else (y[tr].float(), y[va].float())

    def _eval_loss():
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, va.numel(), batch):
                xb, yb = Xn[va[i:i + batch]], y_va[i:i + batch]
                out = model(xb)
                l = loss_fn(out, yb if task == "cls" else yb.unsqueeze(-1))
                tot += float(l) * xb.shape[0]
                cnt += xb.shape[0]
        return tot / max(cnt, 1)

    best_loss, best_state, bad = float("inf"), None, 0
    epochs_run = 0
    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        shuf = tr[torch.randperm(tr.numel(), generator=gen)]
        for i in range(0, shuf.numel(), batch):
            xb, yb = Xn[shuf[i:i + batch]], y_tr[i:i + batch]
            loss = loss_fn(model(xb), yb if task == "cls" else yb.unsqueeze(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = _eval_loss()
        if vl < best_loss - 1e-5:
            best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(Xn[va[i:i + batch]]) for i in range(0, va.numel(), batch)])
    return model, preds, va, best_loss, epochs_run


def _m6(hist, terrain_id, terrain_level, seed, terrain_names):
    out = {"features": "obs_history (225), standardized with train-split mean/std",
           "classifier": "MLP 225->256->128->9 (ELU), Adam 1e-3, 80/20 split, "
                         "early stopping on val loss (patience 20, max 200 epochs)",
           "regressor": "same trunk, 1 output, MSE"}
    valid_cls = (terrain_id >= 0) & (terrain_id < len(terrain_names))
    Xc, yc = hist[valid_cls], terrain_id[valid_cls].long()
    if Xc.shape[0] < 50 or yc.unique().numel() < 2:
        out["classification"] = {"skipped": "not enough samples/classes"}
    else:
        _m, preds, va, best_loss, epochs = _train_probe(
            Xc, yc, seed, task="cls", num_classes=len(terrain_names))
        y_va = yc[va]
        pred_cls = preds.argmax(dim=1)
        acc = float((pred_cls == y_va).float().mean())
        counts_va = torch.bincount(y_va, minlength=len(terrain_names))
        majority = float(counts_va.max() / counts_va.sum().clamp_min(1))
        cm = _contingency(y_va, pred_cls,
                          n_rows=len(terrain_names), n_cols=len(terrain_names))
        recall = {}
        for t in range(len(terrain_names)):
            tot = int(counts_va[t])
            recall[terrain_names[t]] = (float(cm[t, t]) / tot) if tot > 0 else None
        out["classification"] = {
            "val_accuracy": acc, "majority_baseline": majority,
            "per_class_recall": recall, "confusion_matrix": cm,
            "best_val_loss": best_loss, "epochs_run": epochs,
            "gate": {
                "rule": "accuracy near majority baseline (+0.05) -> terrain not decodable "
                        "from obs_history; per-terrain-expert goal is physically "
                        "impossible in this obs space",
                "unobservable": bool(acc <= majority + 0.05),
                "verdict": (f"val acc {acc:.3f} vs majority {majority:.3f}: "
                            + ("terrain is NOT observable from obs_history -- the "
                               "per-terrain-expert goal is physically impossible in this "
                               "obs space; shift the specialization target to "
                               "command/phase/contact axes"
                               if acc <= majority + 0.05 else
                               "terrain IS decodable from obs_history -- the "
                               "'terrain unobservable' hypothesis is CONTRADICTED")),
            },
        }
    valid_reg = terrain_level >= 0
    Xr, yr = hist[valid_reg], terrain_level[valid_reg].float()
    if Xr.shape[0] < 50 or yr.std() <= 0:
        out["regression"] = {"skipped": "not enough samples or constant target"}
    else:
        _m, preds, va, best_loss, epochs = _train_probe(Xr, yr, seed, task="reg")
        y_va = yr[va]
        ss_res = float(((preds.squeeze(-1) - y_va) ** 2).sum())
        ss_tot = float(((y_va - y_va.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        out["regression"] = {"val_r2": r2, "best_val_mse": best_loss,
                             "epochs_run": epochs,
                             "target": "terrain_level (0..num_rows-1)"}
    return out


def _plot_pngs(out_dir, m1, m4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped (matplotlib unavailable)"
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    labels = ["g (raw)", "w_eff (norm-weighted)"]
    ent = [m1["g"]["mean_entropy"], m1["w_eff"]["mean_entropy"]]
    eff = [m1["g"]["mean_effective_experts"], m1["w_eff"]["mean_effective_experts"]]
    axes[0].bar(labels, ent); axes[0].set_title("M1 mean gate entropy (nats)")
    axes[1].bar(labels, eff); axes[1].set_title("M1 mean effective experts")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    p1 = os.path.join(out_dir, "m1_effective_routing.png")
    fig.savefig(p1, dpi=130); plt.close(fig); paths.append(p1)

    variants = ["learned", "uniform", "shuffled", "top1", "oracle"]
    metrics = ["latent_mse", "latent_cos", "action_gap"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, met in zip(axes, metrics):
        ax.bar(variants, [m4["per_variant"][v][met] for v in variants])
        ax.set_title(f"M4 {met}")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p2 = os.path.join(out_dir, "m4_gate_ablation.png")
    fig.savefig(p2, dpi=130); plt.close(fig); paths.append(p2)
    return paths


def _write_report(path, results, gates):
    L = []
    L.append("# MoE-CTS Gating Probe -- REPORT")
    L.append("")
    L.append(f"- checkpoint: `{results['paths']['ckpt']}` (iter {results['ckpt_iter']})")
    L.append(f"- samples: `{results['paths']['samples']}` "
             f"(N={results['data']['num_samples']}, DR "
             f"{'OFF (nominal)' if results['data'].get('nominal') else 'ON (training-fidelity)'})")
    L.append(f"- created: {results['created']}")
    L.append("")
    L.append("## Executive summary")
    L.append("")
    for g in gates:
        L.append(f"- **{g['id']}** {g['verdict']}")
    L.append("")
    L.append("Hypotheses under test (no specialization / gate useless / tau=0.2 too "
             "large / terrain unobservable) are marked OVERTURNED where the data "
             "contradicts them; contradictions are the headline, not the exception.")
    L.append("")
    m1 = results["M1"]
    L.append("## M1 -- Effective routing (gate weights vs norm-weighted)")
    L.append("")
    L.append(f"- per-expert output-norm mean: {[round(x, 4) for x in m1['per_expert_norm_mean']]}")
    L.append(f"- per-expert output-norm std:  {[round(x, 4) for x in m1['per_expert_norm_std']]}")
    for name in ("g", "w_eff"):
        r = m1[name]
        L.append(f"- `{name}`: entropy {r['mean_entropy']:.4f} nats, "
                 f"effective experts {r['mean_effective_experts']:.3f}, "
                 f"mean max weight {r['mean_max_weight']:.4f}")
    L.append(f"- decision gate: {m1['gate']['rule']}")
    L.append(f"- **verdict**: {m1['gate']['verdict']}")
    L.append("")
    m2 = results["M2"]
    L.append("## M2 -- Functional copying (are the 8 experts distinct?)")
    L.append("")
    L.append(f"- mean |off-diag| functional cosine (encoder outputs): "
             f"{m2['mean_abs_offdiag_functional']:.4f}")
    L.append(f"- mean |off-diag| parameter head cosine (Conv1d slices): "
             f"{m2['mean_abs_offdiag_param']:.4f}")
    L.append(f"- decision gate: {m2['gate']['rule']}")
    L.append(f"- **verdict**: {m2['gate']['verdict']}")
    L.append("- functional cosine (8x8):")
    for row in m2["functional_cosine"]:
        L.append("  - " + " ".join(f"{x:6.3f}" for x in row))
    L.append("- linear CKA (8x8):")
    for row in m2["linear_cka"]:
        L.append("  - " + " ".join(f"{x:6.3f}" for x in row))
    L.append("- trend over checkpoints (mean |off-diag| cosine):")
    L.append("  | ckpt | param | functional |")
    L.append("  |---|---|---|")
    for e in m2["trend"]:
        if "skipped" in e:
            L.append(f"  | {e['ckpt']} | -- | {e['skipped']} |")
        else:
            L.append(f"  | {e['ckpt']} | {e['mean_abs_offdiag_param']:.4f} | "
                     f"{e['mean_abs_offdiag_functional']:.4f} |")
    L.append("")
    m3 = results["M3"]
    L.append("## M3 -- Tau calibration (responsibility sharpness)")
    L.append("")
    L.append(f"- per-expert mean d_i (||normalize(E_i) - t||^2): "
             f"{[round(x, 4) for x in m3['per_expert_mean_d']]}")
    s = m3["per_sample_std"]
    L.append(f"- per-sample std_i(d_i): median {s['median']:.4f} "
             f"(p10 {s['p10']:.4f}, p90 {s['p90']:.4f}); "
             f"(max-min)/mean median {m3['range_over_mean']['median']:.3f}")
    L.append("- tau table (r = softmax(-d/tau)):")
    L.append("  | tau | mean max r | mean entropy(r) |")
    L.append("  |---|---|---|")
    for row in m3["tau_table"]:
        L.append(f"  | {row['tau_label']} ({row['tau']:.4f}) | {row['mean_max_r']:.4f} | "
                 f"{row['mean_entropy_r']:.4f} |")
    L.append(f"- decision gate: {m3['gate']['rule']}")
    L.append(f"- **verdict**: {m3['gate']['verdict']}")
    L.append("")
    m4 = results["M4"]
    L.append("## M4 -- Gate ablation (does the gate matter?)")
    L.append("")
    L.append("  | variant | mean ||z-t||^2 | mean cos(z,t) | action gap |")
    L.append("  |---|---|---|---|")
    for v, r in m4["per_variant"].items():
        L.append(f"  | {v} | {r['latent_mse']:.4f} | {r['latent_cos']:.4f} | "
                 f"{r['action_gap']:.5f} |")
    L.append(f"- decision gate: {m4['gate']['rule']}")
    L.append(f"- **verdict**: {m4['gate']['verdict']}")
    L.append("")
    m5 = results["M5"]
    L.append("## M5 -- Regime structure (routing vs terrain/level/command)")
    L.append("")
    L.append("  | pairing | norm. MI | chi2 (dof) |")
    L.append("  |---|---|---|")
    for name, r in m5["pairings"].items():
        L.append(f"  | {name} | {r['normalized_mi']:.4f} | {r['chi2']:.1f} ({r['dof']}) |")
    L.append(f"- per-terrain counts: {m5['per_terrain_counts']}")
    L.append(f"- unreliable (< min samples): {m5['unreliable_terrains']}")
    L.append(f"- decision gate: {m5['gate']['rule']}")
    L.append(f"- **verdict**: {m5['gate']['verdict']}")
    L.append("")
    m6 = results["M6"]
    L.append("## M6 -- Observability probe (can obs_history even see terrain?)")
    L.append("")
    cls = m6.get("classification", {})
    if "skipped" in cls:
        L.append(f"- classification skipped: {cls['skipped']}")
    else:
        L.append(f"- classifier val accuracy {cls['val_accuracy']:.4f} vs majority "
                 f"baseline {cls['majority_baseline']:.4f} "
                 f"(epochs {cls['epochs_run']})")
        L.append(f"- per-class recall: "
                 f"{ {k: (None if v is None else round(v, 3)) for k, v in cls['per_class_recall'].items()} }")
        L.append(f"- **verdict**: {cls['gate']['verdict']}")
    reg = m6.get("regression", {})
    if "skipped" in reg:
        L.append(f"- level regression skipped: {reg['skipped']}")
    else:
        L.append(f"- terrain_level regression val R^2 = {reg['val_r2']:.4f} "
                 f"(epochs {reg['epochs_run']})")
    L.append("")
    if results.get("plots"):
        L.append(f"## Plots\n")
        for p in (results["plots"] if isinstance(results["plots"], list) else []):
            L.append(f"- `{p}`")
        if isinstance(results["plots"], str):
            L.append(f"- {results['plots']}")
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def run_analyze(cli):
    t0 = time.time()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    try:
        from rsl_rl.utils.moe_terrain_gate import TERRAIN_NAMES
    except Exception:
        TERRAIN_NAMES = TERRAIN_NAMES_FALLBACK

    samples_path, ckpt_path, out_dir, ckpt_tag = _resolve_analyze_paths(cli)
    print(f"[probe-analyze] samples: {samples_path}")
    print(f"[probe-analyze] checkpoint: {ckpt_path}")
    bank = torch.load(samples_path, map_location="cpu", weights_only=False)
    meta = None
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    obs = bank["obs"].float()
    hist = bank["obs_history"].float()
    priv = bank["privileged_obs"].float()
    commands = bank["commands"].float()
    terrain_id = bank["terrain_id"].long()
    terrain_level = bank["terrain_level"].long()
    n = obs.shape[0]
    print(f"[probe-analyze] N={n} samples; forwarding encoders...")

    ac, ckpt = _load_ac(ckpt_path)
    E, g, t = _forward_bank(ac, hist, priv)

    # sanity: sum(g*E) then L2-norm must equal forward_with_weights latent
    with torch.no_grad():
        hb, pb = hist[:256], priv[:256]
        Eb = ac.history_encoder.moe.experts(hb)
        gb = ac.history_encoder.moe.gating_network(hb)
        ref, _w = ac.history_encoder.forward_with_weights(hb)
        chk = torch.nn.functional.normalize((gb.unsqueeze(-1) * Eb).sum(dim=1), dim=-1)
        sanity_diff = float((ref - chk).abs().max())
    sanity = {"max_abs_diff": sanity_diff, "passed": bool(sanity_diff < 1e-4),
              "note": "L2norm(sum(g*E)) vs forward_with_weights latent, batch 256"}
    print(f"[probe-analyze] sanity: max|diff|={sanity_diff:.2e} "
          f"({'ok' if sanity['passed'] else 'MISMATCH -- check obs contract!'})")

    # trend checkpoint paths (missing files tolerated)
    trend_list = [int(x) for x in str(cli.trend_ckpts).split(",") if x.strip()]
    trend_paths = []
    for ck in trend_list:
        p = os.path.join(os.path.dirname(ckpt_path), f"model_{ck}.pt")
        trend_paths.append((ck, p if os.path.isfile(p) else None))

    print("[probe-analyze] M1 effective routing...")
    m1 = _m1(g, E)
    print("[probe-analyze] M2 functional copying (+trend)...")
    m2 = _m2(ac, E, hist, ckpt, trend_paths)
    print("[probe-analyze] M3 tau calibration...")
    m3 = _m3(E, t)
    d = ((torch.nn.functional.normalize(E, dim=-1) - t.unsqueeze(1)) ** 2).sum(dim=-1)
    print("[probe-analyze] M4 gate ablation...")
    m4 = _m4(ac, E, g, d, t, obs, cli.seed)
    print("[probe-analyze] M5 regime structure...")
    m5 = _m5(g, d, terrain_id, terrain_level, commands, cli.per_terrain_min, TERRAIN_NAMES)
    print("[probe-analyze] M6 observability probes (CPU MLP training)...")
    m6 = _m6(hist, terrain_id, terrain_level, cli.seed, TERRAIN_NAMES)

    counts = {TERRAIN_NAMES[i]: int((terrain_id == i).sum()) for i in range(len(TERRAIN_NAMES))}
    results = {
        "probe": "moe_gate", "mode": "analyze",
        "created": datetime.now().isoformat(timespec="seconds"),
        "paths": {"samples": samples_path, "ckpt": ckpt_path, "out_dir": out_dir},
        "ckpt_tag": ckpt_tag, "ckpt_iter": int(ckpt.get("iter", -1)),
        "seed": cli.seed,
        "data": {"num_samples": int(n), "per_terrain_counts": counts,
                 "nominal": bool(meta.get("env", {}).get("nominal")) if meta else None},
        "sanity": sanity,
        "M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "M6": m6,
        "meta_from_collect": meta,
        "wall_time_s": None,
    }
    gates = [
        {"id": "M1 effective routing", "verdict": m1["gate"]["verdict"]},
        {"id": "M2 functional copying", "verdict": m2["gate"]["verdict"]},
        {"id": "M3 tau calibration", "verdict": m3["gate"]["verdict"]},
        {"id": "M4 gate ablation", "verdict": m4["gate"]["verdict"]},
        {"id": "M5 regime structure", "verdict": m5["gate"]["verdict"]},
        {"id": "M6 observability",
         "verdict": m6.get("classification", {}).get("gate", {}).get(
             "verdict", "classification skipped")},
    ]
    results["gates"] = gates
    results["plots"] = _plot_pngs(out_dir, m1, m4)
    results["wall_time_s"] = round(time.time() - t0, 2)

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    report_path = os.path.join(out_dir, "REPORT.md")
    _write_report(report_path, _jsonable(results), gates)

    print("\n================ GATE SUMMARY ================")
    for g in gates:
        print(f"[{g['id']}] {g['verdict']}")
    if not sanity["passed"]:
        print("WARNING: latent sanity check FAILED -- treat all numbers as suspect")
    print(f"\nresults -> {results_path}")
    print(f"report  -> {report_path}")
    if isinstance(results["plots"], list):
        for p in results["plots"]:
            print(f"plot    -> {p}")
    else:
        print(f"plots   : {results['plots']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MoE-CTS gating diagnostic probe (collect on GPU, analyze on CPU)")
    p.add_argument("--mode", choices=["collect", "analyze"], required=True)
    p.add_argument("--task", type=str, default=TASK_DEFAULT)
    p.add_argument("--load_run", type=str, default=LOAD_RUN_DEFAULT)
    p.add_argument("--ckpt", type=int, default=7500,
                   help="checkpoint iteration int, or -1 for latest (default 7500)")
    p.add_argument("--num_envs", type=int, default=384,
                   help="parallel envs (384 keeps the 3:1 MoE role split exact)")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=5,
                   help="record every Nth control step after warmup")
    p.add_argument("--target_samples", type=int, default=35000)
    p.add_argument("--per_terrain_min", type=int, default=1000,
                   help="stop only when every ACTIVE terrain type has this many samples")
    p.add_argument("--max_samples", type=int, default=80000, help="hard cap")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--nominal", action="store_true", default=False,
                   help="disable domain randomization (DEFAULT keeps training DR)")
    p.add_argument("--out_root", type=str, default=OUT_ROOT_DEFAULT)
    p.add_argument("--trend_ckpts", type=str, default=TREND_CKPTS_DEFAULT,
                   help="comma list of iterations for the M2 trend (analyze)")
    p.add_argument("--cpu", action="store_true", default=False,
                   help="Genesis CPU backend for collect (very slow; testing only)")
    # analyze-only conveniences
    p.add_argument("--samples", type=str, default=None,
                   help="explicit samples.pt path (analyze; default derives from "
                        "out_root/load_run/ckpt)")
    p.add_argument("--run_dir", type=str, default=None,
                   help="explicit checkpoint run dir (analyze; default "
                        "logs/<experiment>/<load_run>)")
    p.add_argument("--experiment", type=str, default=EXPERIMENT_DEFAULT,
                   help="experiment log folder name (analyze path derivation)")
    return p.parse_args(argv)


def main():
    cli = parse_args()
    if cli.sample_every < 1:
        raise ValueError("--sample_every must be >= 1")
    if cli.mode == "collect":
        run_collect(cli)
    elif cli.mode == "analyze":
        run_analyze(cli)
    else:
        raise ValueError(f"unknown mode: {cli.mode}")


if __name__ == "__main__":
    main()
