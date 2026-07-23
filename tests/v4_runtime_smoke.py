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
        # Standard ETH curriculum: envs start on the easiest rows
        # (0..max_init_terrain_level), not pinned to a hard fixed row.
        max_init = env_cfg.terrain.max_init_terrain_level
        assert (env.simulator.terrain_levels >= 0).all() and (
            env.simulator.terrain_levels <= max_init
        ).all(), "V4 initial terrain levels outside [0, max_init_terrain_level]"
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
        # V4 runs the game-inspired curriculum (not pinned to a fixed row).
        # Prove the progression is actually wired end-to-end: force a level below
        # the max, place the base a full terrain length from its origin so
        # distance > env_length/2 (the move_up condition), then call
        # _update_terrain_curriculum directly.  Because V4 sets no
        # fixed_terrain_level, the V3 physics mixin's guard must fall through to
        # the base game curriculum and promote the env exactly one row, relocating
        # its origin to the new terrain grid cell.  A pinned/no-op implementation
        # would leave the level unchanged and fail here.
        probe = torch.arange(min(4, env.num_envs), device=env.device)
        env.init_done = True
        env.simulator.terrain_levels[probe] = 0
        env_length = env.simulator._terrain.env_length
        env.simulator.base_pos[probe, :2] = (
            env.simulator.env_origins[probe, :2] + env_length
        )
        env._update_terrain_curriculum(probe)
        assert (env.simulator.terrain_levels[probe] == 1).all(), (
            "curriculum did not promote a far-walking env exactly one row: "
            f"{env.simulator.terrain_levels[probe].tolist()}"
        )
        expected_origin = env.simulator._terrain_origins[
            env.simulator.terrain_levels[probe], env.simulator._terrain_types[probe]
        ]
        assert torch.allclose(env.simulator.env_origins[probe], expected_origin), (
            "curriculum promoted the level but did not relocate the env origin"
        )
        # A full reset keeps levels within the valid grid.
        env.reset_idx(probe)
        assert (env.simulator.terrain_levels >= 0).all() and (
            env.simulator.terrain_levels < env_cfg.terrain.num_rows
        ).all(), "V4 terrain level outside [0, num_rows) after reset"
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
