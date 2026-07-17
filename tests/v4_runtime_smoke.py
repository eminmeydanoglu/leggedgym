#!/usr/bin/env python3
"""Genesis-only V4 runtime smoke; no PPO runner or training is created."""

import argparse
import os
import types

os.environ.setdefault("SIMULATOR", "genesis")

import genesis as gs
import torch

import legged_gym.envs  # register V4 tasks
from legged_gym.utils import task_registry


TASKS = {
    "go2_v4_mlp": (45, 48),
    "go2_v4_sysid": (900, 265),
    "go2_v4_rma": (45, 265),
    "go2_v4_dreamwaq": (45, 265),
    "go2_v4_him_fixed": (270, 265),
    "go2_v4_superset_oracle": (1095, 1095),
}


def _args(task):
    return types.SimpleNamespace(
        task=task, seed=17, debug=False, headless=True, cpu=False, num_envs=20,
        max_iterations=None, resume=False, sync_wandb=False, ckpt=None,
        load_run=None, export_onnx=False, motion_file=None, num_student=None,
    )


def _actor_and_critic(env, task):
    obs = env.get_observations()
    if isinstance(obs, tuple):
        critic = obs[-1] if task in {"go2_v4_sysid", "go2_v4_rma"} else obs[1]
        return obs[0], critic
    return obs, env.get_privileged_observations()


def _smoke(task, expected_actor, expected_critic):
    env_cfg, _ = task_registry.get_cfgs(task)
    env, _ = task_registry.make_env(task, args=_args(task), env_cfg=env_cfg)
    try:
        assert torch.equal(
            env.simulator.terrain_levels,
            torch.full_like(env.simulator.terrain_levels, 5),
        ), "V4 terrain level drifted from the fixed medium row"
        env._v3_switch_step[:] = 3
        env._v3_switch_done[:] = False
        for _ in range(12):
            env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        actor, critic = _actor_and_critic(env, task)
        assert actor.shape == (env.num_envs, expected_actor), actor.shape
        assert critic.shape == (env.num_envs, expected_critic), critic.shape
        assert torch.isfinite(actor).all()
        assert torch.isfinite(critic).all()
        assert env._v3_switch_done.all(), "V4 live physics switch did not fire"
        if task == "go2_v4_rma":
            _, teacher_input, _, _ = env.get_observations()
            assert teacher_input.shape == (env.num_envs, 5 + 3 + 17 * 11), teacher_input.shape
            expected_map = torch.clip(
                env.simulator.base_pos[:, 2].unsqueeze(1) - 0.5
                - env.simulator.measured_heights,
                -1.0,
                1.0,
            ) * env.obs_scales.height_measurements
            assert torch.allclose(teacher_input[:, -187:], expected_map), (
                "RMA teacher input does not contain the current scaled height map"
            )
        if task == "go2_v4_superset_oracle":
            expected_map = torch.clip(
                env.simulator.base_pos[:, 2].unsqueeze(1) - 0.5
                - env.simulator.measured_heights,
                -1.0,
                1.0,
            ) * env.obs_scales.height_measurements
            assert torch.allclose(actor[:, -187:], expected_map), (
                "oracle observation does not contain the current scaled height map"
            )
        # V4 uses ``curriculum=True`` only to generate the terrain grid. Verify
        # that a real reset cannot move a policy to a performance-dependent row.
        reset_ids = torch.arange(min(4, env.num_envs), device=env.device)
        env.reset_idx(reset_ids)
        assert torch.equal(
            env.simulator.terrain_levels,
            torch.full_like(env.simulator.terrain_levels, 5),
        ), "V4 terrain level drifted after reset"
        print(f"PASS {task}: actor={tuple(actor.shape)}, critic={tuple(critic.shape)}")
    finally:
        if hasattr(env, "destroy"):
            env.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=tuple(TASKS))
    args = parser.parse_args()
    gs.init(backend=gs.gpu, logging_level="warning")
    for task in args.tasks:
        _smoke(task, *TASKS[task])


if __name__ == "__main__":
    main()
