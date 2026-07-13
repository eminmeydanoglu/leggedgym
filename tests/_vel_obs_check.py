"""Runtime actor/critic observation-layout check for the benchmark baselines.

Builds one task, steps a few times, and verifies that actor proprio remains noisy
while the asymmetric critic's proprio block is clean.  It also checks the P5 and
velocity blocks against the simulator quantities.

Run one of:
  TASK=go2_bench_mlp SIMULATOR=genesis .venv/bin/python tests/_vel_obs_check.py
  TASK=go2_bench_oracle_id SIMULATOR=genesis .venv/bin/python tests/_vel_obs_check.py
  TASK=go2_bench_oracle_id_vel SIMULATOR=genesis .venv/bin/python tests/_vel_obs_check.py
"""
import os, types
os.environ.setdefault("SIMULATOR", "genesis")

import torch
import genesis as gs
import legged_gym.envs  # noqa: F401
from legged_gym.utils import task_registry

TASK = os.environ.get("TASK", "go2_bench_oracle_id_vel")

EXPECTED_DIMS = {
    "go2_bench_mlp": (45, 48),
    "go2_bench_oracle_id": (50, 53),
    "go2_bench_oracle_id_vel": (53, 53),
}


def main():
    if TASK not in EXPECTED_DIMS:
        raise ValueError(f"unsupported task: {TASK}")
    gs.init(backend=gs.gpu, logging_level="warning")
    args = types.SimpleNamespace(
        task=TASK, seed=1, headless=True, cpu=False, num_envs=16, max_iterations=None,
        resume=False, sync_wandb=False, export_onnx=False, debug=False, load_run=None,
        ckpt=-1, use_joystick=False, joystick_type="xbox", follow_robot=False,
        viewer="native", viser_port=8080, motion_file=None, motion_out_dir=None, num_student=None,
    )
    env, env_cfg = task_registry.make_env(name=TASK, args=args)
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=TASK, args=args)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations().to(env.device)
    for _ in range(20):  # settle
        obs, _, _, _, _ = env.step(policy(obs.detach()))
        obs = obs.to(env.device)

    actor_dim, critic_dim = EXPECTED_DIMS[TASK]
    critic = env.get_privileged_observations()
    checks = {
        "actor_obs_dim": obs.shape[1] == actor_dim,
        "critic_obs_dim": critic.shape[1] == critic_dim,
    }

    clean_proprio = torch.cat((
        env.commands[:, :3] * env.commands_scale,
        env.simulator.projected_gravity,
        env.simulator.base_ang_vel * env.obs_scales.ang_vel,
        (env.simulator.dof_pos - env.simulator.default_dof_pos) * env.obs_scales.dof_pos,
        env.simulator.dof_vel * env.obs_scales.dof_vel,
        env.actions,
    ), dim=-1)
    checks["critic_proprio_clean"] = torch.allclose(
        critic[:, :45], clean_proprio.to(critic.dtype), atol=1e-4)
    checks["actor_proprio_noisy"] = not torch.allclose(
        obs[:, :45], clean_proprio.to(obs.dtype), atol=1e-4)

    # recompute the appended blocks straight from the simulator
    P = torch.cat((
        env.simulator.dr_friction_values.view(env.num_envs, -1),
        env.simulator.dr_added_base_mass.view(env.num_envs, -1),
        env.simulator.dr_base_com_bias.view(env.num_envs, -1),
    ), dim=-1)
    vel = env.simulator.base_lin_vel * env.obs_scales.lin_vel

    if TASK != "go2_bench_mlp":
        checks["actor_P_block_matches"] = torch.allclose(
            obs[:, 45:50], P.to(obs.dtype), atol=1e-4)
        checks["critic_P_block_matches"] = torch.allclose(
            critic[:, 45:50], P.to(critic.dtype), atol=1e-4)

    vel_start = 45 if TASK == "go2_bench_mlp" else 50
    checks["critic_vel_block_matches"] = torch.allclose(
        critic[:, vel_start:vel_start + 3], vel.to(critic.dtype), atol=1e-4)
    if TASK == "go2_bench_oracle_id_vel":
        checks["actor_vel_block_matches"] = torch.allclose(
            obs[:, 50:53], vel.to(obs.dtype), atol=1e-4)
    # velocity should actually be nonzero after settling (sanity: it is a real state)
    checks["vel_nonzero"] = bool((vel.abs().sum() > 0).item())

    print("[VEL OBS CHECK]")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k} = {v}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
