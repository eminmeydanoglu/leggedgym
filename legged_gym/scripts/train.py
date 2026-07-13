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


def write_run_manifest(log_dir, args, env_cfg, train_cfg):
    """Write run_manifest.json into the run folder: full provenance for the run
    so a checkpoint can be traced back to its exact protocol (see codex_plan.md
    sec. 2). Best-effort: never let manifest writing break training."""
    try:
        dr = env_cfg.domain_rand
        runner = train_cfg.runner
        manifest = {
            "task": args.task,
            "training_seed": train_cfg.seed,
            "git_commit": _git_commit(),
            **_simulator_versions(),
            # P5 = [friction, added_base_mass, com_x, com_y, com_z]
            "p5_distribution": {
                "friction": list(getattr(dr, "friction_range", [])),
                "added_base_mass": list(getattr(dr, "added_mass_range", [])),
                "com_x": list(getattr(dr, "com_pos_x_range", [])),
                "com_y": list(getattr(dr, "com_pos_y_range", [])),
                "com_z": list(getattr(dr, "com_pos_z_range", [])),
            },
            "command_schedule": getattr(runner, "command_schedule", None),
            "max_iterations": runner.max_iterations,
            "num_envs": env_cfg.env.num_envs,
            "num_observations": env_cfg.env.num_observations,
            # checkpoint-selection (best.pt) protocol
            "checkpoint_selection": {
                "protocol": "in-dist eval, deterministic policy, mean_return + fall guard",
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
    shutil.copy(robot_file_path, log_dir)
    shutil.copy(robot_config_path, log_dir)

    # Provenance manifest for the run (task/seed/git/simulator/P5/schedule/...)
    write_run_manifest(log_dir, args, env_cfg, train_cfg)

    # Start training session
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.num_envs = 1
    train(args)
