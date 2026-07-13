"""Runtime obs-layout check for go2_bench_oracle_id_vel (GPU/genesis).

Builds the env, steps a few times, and asserts the observation is exactly
[ proprio(45), P(5), base_lin_vel(3) ] = 53 with the appended blocks matching
the raw simulator quantities. Cheap (few steps) -- run before the real training.
"""
import os, types
os.environ.setdefault("SIMULATOR", "genesis")

import torch
import genesis as gs
import legged_gym.envs  # noqa: F401
from legged_gym.utils import task_registry

TASK = "go2_bench_oracle_id_vel"


def main():
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

    checks = {}
    checks["obs_dim_53"] = (obs.shape[1] == 53)

    # recompute the appended blocks straight from the simulator
    P = torch.cat((
        env.simulator.dr_friction_values.view(env.num_envs, -1),
        env.simulator.dr_added_base_mass.view(env.num_envs, -1),
        env.simulator.dr_base_com_bias.view(env.num_envs, -1),
    ), dim=-1)
    vel = env.simulator.base_lin_vel * env.obs_scales.lin_vel

    checks["P_block_matches"] = torch.allclose(obs[:, 45:50], P.to(obs.dtype), atol=1e-4)
    checks["vel_block_matches"] = torch.allclose(obs[:, 50:53], vel.to(obs.dtype), atol=1e-4)
    # velocity should actually be nonzero after settling (sanity: it is a real state)
    checks["vel_nonzero"] = bool((obs[:, 50:53].abs().sum() > 0).item())

    print("[VEL OBS CHECK]")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k} = {v}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
