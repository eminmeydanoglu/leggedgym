import os
import json
import subprocess


from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import shutil


def _git_commit():
    """Short git commit of the repo, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=LEGGED_GYM_ROOT_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _git_state():
    """Return the clean/dirty provenance required by future Eval V2 runs."""
    try:
        raw = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=LEGGED_GYM_ROOT_DIR,
            stderr=subprocess.DEVNULL,
        )
        import hashlib
        return bool(raw.strip()), hashlib.sha256(raw).hexdigest()
    except Exception:
        return None, None


def _simulator_versions():
    """SIMULATOR name + relevant package versions (best-effort)."""
    import torch
    versions = {"simulator": SIMULATOR, "torch": torch.__version__}
    if SIMULATOR == "genesis":
        try:
            import genesis as gs
            versions["genesis"] = getattr(gs, "__version__", "unknown")
        except Exception:
            versions["genesis"] = "unknown"
    return versions


def _dashboard_requested(args) -> bool:
    """Keep Curriculum Atlas strictly opt-in (CLI or explicit environment)."""
    env_value = os.getenv("LPACRL_DASHBOARD", "").strip().lower()
    return bool(getattr(args, "dashboard", False)) or env_value in {"1", "true", "yes", "on"}


def write_run_manifest(log_dir, args, env_cfg, train_cfg):
    """Write run_manifest.json into the run folder: full provenance for the run
    so a checkpoint can be traced back to its exact protocol (see codex_plan.md
    sec. 2). Best-effort: never let manifest writing break training."""
    try:
        dr = env_cfg.domain_rand
        runner = train_cfg.runner
        git_dirty, git_diff_sha256 = _git_state()
        manifest = {
            "task": args.task,
            "training_seed": train_cfg.seed,
            "git_commit": _git_commit(),
            "git_dirty": git_dirty,
            "git_diff_sha256": git_diff_sha256,
            **_simulator_versions(),
            # P5 = [friction, added_base_mass, com_x, com_y, com_z]
            "p5_distribution": {
                "friction": list(getattr(dr, "friction_range", [])),
                "added_base_mass": list(getattr(dr, "added_mass_range", [])),
                "com_x": list(getattr(dr, "com_pos_x_range", [])),
                "com_y": list(getattr(dr, "com_pos_y_range", [])),
                "com_z": list(getattr(dr, "com_pos_z_range", [])),
            },
            "dynamic_physics": {
                "enabled": bool(getattr(dr, "resample_physics_within_episode", False)),
                "switches_per_episode": 1,
                "switch_window_frac": list(getattr(dr, "physics_resample_switch_window", [])),
                "mass": bool(getattr(dr, "physics_resample_mass", False)),
                "com": bool(getattr(dr, "physics_resample_com", False)),
            },
            "observation_contract": {
                "actor_observations": env_cfg.env.num_observations,
                "critic_observations": getattr(env_cfg.env, "num_privileged_obs", None),
                "history_frames": getattr(env_cfg.env, "frame_stack", None),
                "estimator_labels": getattr(env_cfg.env, "num_estimator_labels", None),
            },
            "command_schedule": getattr(runner, "command_schedule", None),
            # V5 UED provenance: which episode-task sampler (if any) owns the
            # command/terrain distribution for this run (see go2_v5_config.py).
            "ued_enabled": bool(getattr(env_cfg.env, "ued_enabled", False)),
            "ued_curriculum_algorithm": getattr(
                getattr(env_cfg, "curriculum", None), "algorithm", None
            ),
            "max_iterations": runner.max_iterations,
            "num_envs": env_cfg.env.num_envs,
            "num_observations": env_cfg.env.num_observations,
            # V2 checkpoint-selection (best_tracking.pt) protocol
            "checkpoint_selection": {
                "protocol": "v2 tracking lexicographic: safe fall-rate then tracking score",
                "validation_lin_vel_x": list(env_cfg.commands.ranges.lin_vel_x),
                "eval_seed": getattr(runner, "eval_seed", None),
                "eval_steps": getattr(runner, "eval_steps", None),
                "eval_warmup": getattr(runner, "eval_warmup", None),
                "eval_fall_guard": getattr(runner, "eval_fall_guard", None),
            },
        }
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "run_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"[manifest] WARNING: failed to write run_manifest.json: {e}")


def train(args):
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning')
    # Make environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    # V5 UED arms (uniform/lp_acrl/alp) opt into an episode-task sampler via
    # cfg.env.ued_enabled; every existing v3/v4/bench task leaves this False
    # and this block is a no-op, so their training stays byte-for-byte
    # unaffected. Must run before make_alg_runner(): a --resume load happens
    # inside that call and needs env.episode_curriculum already installed to
    # restore the teacher state into (see rsl_rl OnPolicyRunner.load).
    if getattr(env_cfg.env, "ued_enabled", False):
        from legged_gym.envs.go2.go2_v5_config import build_ued_teacher

        episode_curriculum, task_space = build_ued_teacher(env_cfg)
        env.enable_ued(episode_curriculum, task_space)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # Copy env.py and env_config.py to log_dir for backup
    log_dir = ppo_runner.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if env_cfg.asset.name == args.task:
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+".py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+"_config.py")
    else:
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+".py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+"_config.py")
    # A registered task need not map one-to-one to a source file.  In
    # particular, V4 reuses the V3 environment implementations and supplies
    # its contract through go2_v4_config.py.  Source snapshots are useful
    # provenance, but they must never prevent an otherwise valid training run.
    for source_path in (robot_file_path, robot_config_path):
        if os.path.isfile(source_path):
            shutil.copy(source_path, log_dir)
        else:
            print(f"[manifest] source snapshot unavailable; skipping: {source_path}")

    if args.task.startswith("go2_v4_"):
        v4_config_path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", "go2", "go2_v4_config.py"
        )
        if os.path.isfile(v4_config_path):
            shutil.copy(v4_config_path, log_dir)

    if args.task.startswith("go2_v5_"):
        v5_config_path = os.path.join(
            LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", "go2", "go2_v5_config.py"
        )
        if os.path.isfile(v5_config_path):
            shutil.copy(v5_config_path, log_dir)

    # Provenance manifest for the run (task/seed/git/simulator/P5/schedule/...)
    write_run_manifest(log_dir, args, env_cfg, train_cfg)

    # Curriculum Atlas is deliberately attached only after the UED teacher and
    # runner exist.  Disabled runs import no dashboard module, create no worker
    # thread, and do not touch their reset/training semantics.
    dashboard_bridge = None
    if _dashboard_requested(args):
        from lpacr.dashboard.v5_integration import create_v5_dashboard_bridge

        dashboard_url = (
            getattr(args, "dashboard_url", None)
            or os.getenv("LPACRL_DASHBOARD_URL")
            or "http://127.0.0.1:8765"
        )
        default_run_id = f"{args.task}-{os.path.basename(os.path.normpath(log_dir))}"
        dashboard_bridge = create_v5_dashboard_bridge(
            env,
            task=args.task,
            training_seed=train_cfg.seed,
            server_url=dashboard_url,
            run_id=getattr(args, "dashboard_run_id", None) or default_run_id,
        )
        if dashboard_bridge is None:
            print("[ued-dashboard] requested, but this non-UED arm has no stage frames to publish")
        else:
            env.set_ued_snapshot_listener(dashboard_bridge.publish)
            print(
                f"[ued-dashboard] enabled: run={dashboard_bridge.run_id} "
                f"server={dashboard_url}"
            )

    # Start training session.  ``close`` flushes the final stage frame without
    # changing error propagation from PPO training.
    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    finally:
        if dashboard_bridge is not None:
            dashboard_bridge.close()

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.num_envs = 1
    train(args)
