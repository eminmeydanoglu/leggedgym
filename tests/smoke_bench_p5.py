"""Runtime smoke test for the P5 plan (codex_plan.md sec. 5, runtime smoke).

GPU/genesis required. For each of go2_bench_mlp / go2_bench_oracle_id it:
  * builds the env with a small num_envs and a SHRUNK command schedule
    (boundary at iter 4 instead of 500) + small eval/save intervals,
  * runs a handful of PPO iterations,
  * asserts: seed plumbed through, schedule transition fired at the boundary,
    best.pt written, checkpoint carries seed + schedule-stage metadata,
    run_manifest.json written with the expected fields.

Run on a GPU node:
  SIMULATOR=genesis WANDB_MODE=disabled .venv/bin/python -m tests.smoke_bench_p5
"""

import os
import json
import types

os.environ.setdefault("SIMULATOR", "genesis")

import torch  # noqa: E402
import genesis as gs  # noqa: E402

from legged_gym import SIMULATOR  # noqa: E402
import legged_gym.envs  # noqa: E402  (registers tasks)
from legged_gym.utils import task_registry  # noqa: E402
from legged_gym.scripts.train import write_run_manifest  # noqa: E402

SEED = 7
NUM_ENVS = 32
MAX_ITERS = 12
SHRUNK_SCHEDULE = [
    {"start_iteration": 0, "lin_vel_x": [-0.5, 0.5]},
    {"start_iteration": 4, "lin_vel_x": [-1.0, 1.0]},
]


def _args():
    return types.SimpleNamespace(
        task=None, seed=SEED, headless=True, cpu=False, num_envs=NUM_ENVS,
        max_iterations=MAX_ITERS, resume=False, sync_wandb=False, export_onnx=False,
        debug=False, load_run=None, ckpt=-1, use_joystick=False, joystick_type="xbox",
        follow_robot=False, viewer="native", viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None,
    )


def smoke_one(task, expect_obs):
    print(f"\n{'='*70}\n[SMOKE] task={task} expect_obs={expect_obs}\n{'='*70}")
    args = _args(); args.task = task

    env_cfg, train_cfg = task_registry.get_cfgs(task)
    # small, fast overrides
    train_cfg.runner.max_iterations = MAX_ITERS
    train_cfg.runner.save_interval = 6
    train_cfg.runner.eval_interval = 6
    train_cfg.runner.eval_steps = 200
    train_cfg.runner.eval_warmup = 10
    train_cfg.runner.command_schedule = SHRUNK_SCHEDULE

    env, env_cfg = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
    log_dir = runner.log_dir
    print(f"[SMOKE] log_dir={log_dir}")

    checks = {}
    checks["run_folder_has_seed"] = f"_seed{SEED}" in os.path.basename(log_dir)
    checks["obs_dim"] = (env.num_obs == expect_obs)
    checks["training_seed_set"] = (runner.training_seed == SEED)

    write_run_manifest(log_dir, args, env_cfg, train_cfg)
    runner.learn(num_learning_iterations=MAX_ITERS, init_at_random_ep_len=True)

    # schedule transition should have advanced to the wide stage (boundary=4)
    checks["schedule_advanced"] = (runner._active_schedule_start == 4
                                   and runner._active_schedule_range == [-1.0, 1.0])
    checks["curriculum_off"] = (env.cfg.commands.curriculum is False)

    # valid (stage_start, lin_vel_x) pairs from the shrunk schedule
    valid_stages = {s["start_iteration"]: s["lin_vel_x"] for s in SHRUNK_SCHEDULE}

    # best.pt + metadata. best.pt is saved whenever an eval improves the score, so
    # its schedule stage reflects the stage ACTIVE AT THAT eval iteration (not the
    # final stage) -- assert the fields are present and internally consistent.
    best_path = os.path.join(log_dir, "best.pt")
    checks["best_pt_exists"] = os.path.isfile(best_path)
    if checks["best_pt_exists"]:
        ck = torch.load(best_path, weights_only=False, map_location="cpu")
        checks["ckpt_seed"] = (ck.get("training_seed") == SEED)
        st = ck.get("schedule_stage_start")
        checks["ckpt_schedule_consistent"] = (st in valid_stages
                                              and ck.get("schedule_lin_vel_x") == valid_stages[st])
        infos = ck.get("infos") or {}
        checks["ckpt_eval_it"] = ("eval_it" in infos and "eval_score" in infos)

    # final model checkpoint must carry the ADVANCED (post-boundary) stage, proving
    # the schedule stage is persisted as it changes over training.
    final_path = os.path.join(log_dir, f"model_{MAX_ITERS}.pt")
    checks["final_ckpt_exists"] = os.path.isfile(final_path)
    if checks["final_ckpt_exists"]:
        fk = torch.load(final_path, weights_only=False, map_location="cpu")
        checks["final_schedule_stage"] = (fk.get("schedule_stage_start") == 4
                                          and fk.get("schedule_lin_vel_x") == [-1.0, 1.0])
        checks["final_seed"] = (fk.get("training_seed") == SEED)

    # run_manifest.json
    man_path = os.path.join(log_dir, "run_manifest.json")
    checks["manifest_exists"] = os.path.isfile(man_path)
    if checks["manifest_exists"]:
        man = json.load(open(man_path))
        checks["manifest_seed"] = (man.get("training_seed") == SEED)
        checks["manifest_schedule"] = (man.get("command_schedule") == SHRUNK_SCHEDULE)
        checks["manifest_p5"] = ("p5_distribution" in man
                                 and set(man["p5_distribution"]) ==
                                 {"friction", "added_base_mass", "com_x", "com_y", "com_z"})
        checks["manifest_valfield"] = (man["checkpoint_selection"]["validation_lin_vel_x"] == [-1.0, 1.0])
        checks["manifest_gitcommit"] = bool(man.get("git_commit"))

    ok = all(checks.values())
    print(f"\n[SMOKE RESULT] task={task}")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k} = {v}")
    print(f"[SMOKE] {task}: {'ALL PASS' if ok else 'HAS FAILURES'}")
    return ok, log_dir


def main():
    gs.init(backend=gs.gpu, logging_level="warning")
    results = {}
    results["go2_bench_mlp"], mlp_dir = smoke_one("go2_bench_mlp", 45)
    results["go2_bench_oracle_id"], _ = smoke_one("go2_bench_oracle_id", 50)

    print(f"\n{'#'*70}")
    print("[SMOKE SUMMARY]")
    for t, ok in results.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {t}")
    # emit the mlp run dir so the loader smoke (indist/sweep/transient) can target it
    print(f"[SMOKE] MLP_RUN_DIR={os.path.basename(mlp_dir)}")
    print(f"{'#'*70}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
