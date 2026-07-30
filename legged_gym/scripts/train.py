import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path


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
    """Curriculum Atlas publishing is on by default for UED arms (non-UED
    tasks already no-op in create_v5_dashboard_bridge, and publishing to an
    absent server just retries quietly in a daemon thread -- see
    lpacr/dashboard/plugger.py), so this only has to honour explicit opt-outs.
    """
    if getattr(args, "no_dashboard", False):
        return False
    env_value = os.getenv("LPACRL_DASHBOARD", "").strip().lower()
    if env_value in {"0", "false", "no", "off"}:
        return False
    return bool(getattr(args, "dashboard", True)) or env_value in {"1", "true", "yes", "on"}


_V7_INIT_TARGETS = {
    "go2_v6_frontier_him",
    "go2_v7_lpacrl_him",
    "go2_v7_uniform_him",
}


def load_v7_flat_initialization(runner, task: str, checkpoint: str) -> dict[str, object]:
    """Strictly initialize only deploy-relevant HIM policy state.

    This is intentionally separate from ``runner.load``: a source prior is
    not a resume.  Optimizer, critic, UED, RNG, selection and iteration state
    therefore remain fresh in the downstream run.
    """
    if task not in _V7_INIT_TARGETS:
        raise ValueError(
            "--init_checkpoint is supported only by go2_v6_frontier_him, "
            "go2_v7_lpacrl_him, and go2_v7_uniform_him"
        )
    source = Path(checkpoint).expanduser().resolve()
    if source.name != "model_1000.pt":
        raise ValueError("--init_checkpoint must name the flat source artifact model_1000.pt")
    if not source.is_file():
        raise FileNotFoundError(f"init checkpoint does not exist: {source}")
    manifest_path = source.parent / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"init checkpoint lacks sibling provenance manifest: {manifest_path}")
    try:
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read init checkpoint manifest: {manifest_path}") from exc
    if source_manifest.get("task") != "go2_v7_flat_lpacrl_him":
        raise ValueError(
            "init checkpoint must originate from go2_v7_flat_lpacrl_him; "
            f"got {source_manifest.get('task')!r}"
        )
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    source_iteration = payload.get("iter")
    if source_iteration != 1000:
        raise ValueError(
            "init checkpoint must contain the completed flat iteration 1000; "
            f"got {source_iteration!r}"
        )
    prefixes = tuple(runner.deploy_state_prefixes())
    if prefixes != ("actor.", "estimator."):
        raise ValueError("--init_checkpoint requires a HIM runner with actor and estimator deploy state")
    # This strict component loader rejects a missing or structurally incompatible
    # actor/estimator. It never sees critic / optimizers / curriculum state.
    runner.load_deploy_state(str(source), map_location=runner.device)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "mode": "deploy_him_actor_estimator_only_v1",
        "checkpoint_path": str(source),
        "sha256": digest,
        "source_task": source_manifest["task"],
        "source_seed": source_manifest.get("training_seed", payload.get("training_seed")),
        "source_iteration": source_iteration,
    }


def _v7_provenance(env):
    """V7 task-space identity for the run manifest.

    The semantic arms persist a family x |vx| x level LP identity and sample a
    physical V6 column only at reset; the flat prior owns four |vx| cells and no
    terrain.  Recording both fingerprints lets a paired run prove that V6-HIM,
    V7-LP and V7-uniform share the exact 10x10 physical terrain bank while the
    flat source deliberately does not teleport at all.  Returns ``None`` for
    non-V7 (or non-UED) runs so the manifest stays unchanged for them.
    """
    adapter = getattr(env, "ued_adapter", None)
    task_space = getattr(adapter, "task_space", None)
    if task_space is None:
        return None
    # V5 must be able to write its provenance manifest from an intentionally
    # minimal isolated source tree.  V7 is an optional sibling feature there,
    # not a dependency of a non-V7 run.
    try:
        from legged_gym.utils.v7.task_space import V7SemanticTaskSpace, V7VelocityTaskSpace
    except ModuleNotFoundError:
        return None

    if isinstance(task_space, V7SemanticTaskSpace):
        return {
            "task_identity": "v7_semantic_family_abs_vx_level",
            "task_space_fingerprint": task_space.fingerprint(),
            "terrain_bank_fingerprint": task_space.terrain_bank_fingerprint(),
            "num_cells": int(task_space.size),
            "num_families": int(task_space.num_families),
            "num_speed_bins": int(task_space.num_speed_bins),
            "num_levels": int(task_space.NUM_LEVELS),
            "family_columns": {
                int(f): [int(c) for c in task_space.columns_for_family(f)]
                for f in range(task_space.num_families)
            },
            "velocity_bin_edges": [float(x) for x in task_space.velocity_bin_edges],
            "vx_sign": "persistent_per_episode_rademacher_0.5",
            "physical_column_sampling": "uniform_within_family_at_reset",
            "standstill_attribution": "reserved_bucket_never_learning_progress",
            "requires_terrain_origins": bool(
                getattr(adapter, "requires_terrain_origins", True)
            ),
        }
    if isinstance(task_space, V7VelocityTaskSpace):
        return {
            "task_identity": "v7_flat_abs_vx",
            "task_space_fingerprint": task_space.fingerprint(),
            "terrain_bank_fingerprint": None,
            "num_cells": int(task_space.size),
            "velocity_bin_edges": [float(x) for x in task_space.velocity_bin_edges],
            "vx_sign": "persistent_per_episode_rademacher_0.5",
            "physical_column_sampling": None,
            "standstill_attribution": "reserved_bucket_never_learning_progress",
            "requires_terrain_origins": bool(
                getattr(adapter, "requires_terrain_origins", True)
            ),
        }
    return None


def write_run_manifest(log_dir, args, env_cfg, train_cfg, env=None):
    """Write run_manifest.json into the run folder: full provenance for the run
    so a checkpoint can be traced back to its exact protocol (see codex_plan.md
    sec. 2). Best-effort: never let manifest writing break training."""
    try:
        dr = env_cfg.domain_rand
        runner = train_cfg.runner
        git_dirty, git_diff_sha256 = _git_state()
        curriculum = getattr(env, "episode_curriculum", None)
        task_space = getattr(curriculum, "task_space", None)
        curriculum_schema = None
        if curriculum is not None:
            # Reading this small state dictionary is provenance-only and occurs
            # before learning; it does not sample either PPO or sampler RNG.
            curriculum_schema = curriculum.state_dict().get("schema_version")
        shadow_telemetry = None
        if curriculum is not None and args.task == "go2_v5_uniform":
            shadow_telemetry = {
                "telemetry_schema_version": 6,
                "checkpoint_schema_version": curriculum_schema,
                "curriculum_config_fingerprint": getattr(curriculum, "config_fingerprint", None),
                "task_space_fingerprint": task_space.fingerprint() if task_space is not None else None,
                "stage_admission_semantics": "completion_stage_all_moving_v1",
                "stage_checkpoint_contract": "even_closed_stages_after_ppo_update_v1",
                "shadow_metrics": ["pvl_raw_gae", "absolute_raw_gae", "success_rate", "frontier"],
            }
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
            "v5_shadow_telemetry": shadow_telemetry,
            "initialization": getattr(args, "_init_provenance", None),
            # V7 semantic/flat task-space identity and shared physical-bank proof.
            "v7_task_space": _v7_provenance(env),
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
        algorithm = getattr(getattr(env_cfg, "curriculum", None), "algorithm", None)
        if algorithm == "frontier":
            from legged_gym.envs.go2.go2_v6_frontier_config import build_frontier_teacher

            episode_curriculum, task_space = build_frontier_teacher(env_cfg)
        elif args.task.startswith("go2_v7_"):
            from legged_gym.envs.go2.go2_v7_lpacrl_him_config import build_v7_teacher

            episode_curriculum, task_space = build_v7_teacher(env_cfg)
        else:
            from legged_gym.envs.go2.go2_v5_config import build_ued_teacher

            episode_curriculum, task_space = build_ued_teacher(env_cfg)
        env.enable_ued(episode_curriculum, task_space)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    if getattr(args, "init_checkpoint", None):
        args._init_provenance = load_v7_flat_initialization(
            ppo_runner, args.task, args.init_checkpoint
        )
        print(
            "[init] loaded flat HIM actor/estimator only "
            f"from {args._init_provenance['checkpoint_path']}"
        )

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

    if args.task.startswith("go2_v6_"):
        v6_config_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "legged_gym",
            "envs",
            "go2",
            "go2_v6_frontier_config.py",
        )
        if os.path.isfile(v6_config_path):
            shutil.copy(v6_config_path, log_dir)

    if args.task.startswith("go2_v7_"):
        v7_config_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "legged_gym",
            "envs",
            "go2",
            "go2_v7_lpacrl_him_config.py",
        )
        if os.path.isfile(v7_config_path):
            shutil.copy(v7_config_path, log_dir)

    # Provenance manifest for the run (task/seed/git/simulator/P5/schedule/...)
    write_run_manifest(log_dir, args, env_cfg, train_cfg, env=env)

    # UED stage frames (sampling_probability, LP, episode counts, ...) are
    # always written under <log_dir>/curriculum_atlas/frames.ndjson for UED
    # arms -- independent of the live Curriculum Atlas server.  When the
    # dashboard is requested we also POST to it; a missing server is still
    # harmless because the local write is the durable analysis trail.
    # Bridge construction is best-effort: a failure must never cost GPU hours.
    dashboard_bridge = None
    if getattr(env_cfg.env, "ued_enabled", False):
        try:
            if LEGGED_GYM_ROOT_DIR not in sys.path:
                sys.path.insert(0, LEGGED_GYM_ROOT_DIR)
            from lpacr.dashboard.v5_integration import create_v5_dashboard_bridge

            want_live = _dashboard_requested(args)
            dashboard_url = None
            if want_live:
                dashboard_url = (
                    getattr(args, "dashboard_url", None)
                    or os.getenv("LPACRL_DASHBOARD_URL")
                    or "http://127.0.0.1:8765"
                )
            default_run_id = f"{args.task}-{os.path.basename(os.path.normpath(log_dir))}"
            local_frames_dir = os.path.join(log_dir, "curriculum_atlas")
            dashboard_bridge = create_v5_dashboard_bridge(
                env,
                task=args.task,
                training_seed=train_cfg.seed,
                server_url=dashboard_url,
                run_id=getattr(args, "dashboard_run_id", None) or default_run_id,
                local_dir=local_frames_dir,
            )
            if dashboard_bridge is None:
                print("[ued-frames] non-UED arm has no stage frames to record")
        except Exception as error:
            print(f"[ued-frames] disabled: could not attach ({error!r})")
            dashboard_bridge = None
        if dashboard_bridge is not None:
            env.set_ued_snapshot_listener(dashboard_bridge.publish)
            where = f"local={dashboard_bridge.local_dir}"
            if dashboard_url:
                where += f" server={dashboard_url}"
            else:
                where += " server=off"
            print(f"[ued-frames] enabled: run={dashboard_bridge.run_id} {where}")

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
