#!/usr/bin/env python3
"""Genesis-only V3 runtime smoke.  It never creates a PPO runner or trains."""

import argparse
import os
import types

os.environ.setdefault("SIMULATOR", "genesis")

import torch
import genesis as gs

import legged_gym.envs  # registers V3 tasks
from legged_gym.utils import task_registry


TASKS = {
    "go2_v3_mlp": (45, 48),
    "go2_v3_sysid": (900, 265),
    "go2_v3_rma": (45, 265),
    "go2_v3_dreamwaq": (45, 265),
    "go2_v3_him_fixed": (270, 265),
    "go2_v3_superset_oracle": (908, 908),
}


def args_for(task):
    return types.SimpleNamespace(
        task=task,
        seed=17,
        debug=False,
        headless=True,
        cpu=False,
        num_envs=16,
        max_iterations=None,
        resume=False,
        sync_wandb=False,
        ckpt=None,
        load_run=None,
        export_onnx=False,
        motion_file=None,
        num_student=None,
    )


def first_and_critic(env, task):
    obs = env.get_observations()
    if isinstance(obs, tuple):
        # SysID: (history, labels, critic); RMA: (obs, teacher, history,
        # critic); DreamWaQ: (obs, critic, history, labels, decoder_target).
        # All three deploy from their first tensor, but their critic location
        # differs by method contract.
        critic = obs[-1] if task in {"go2_v3_sysid", "go2_v3_rma"} else obs[1]
        return obs[0], critic
    return obs, env.get_privileged_observations()


def smoke_one(task, expected_actor, expected_critic):
    args = args_for(task)
    env_cfg, _ = task_registry.get_cfgs(task)
    env, _ = task_registry.make_env(task, args=args, env_cfg=env_cfg)
    # One switch per episode.  Force the pending switch to a few steps in so the
    # live path fires inside this short loop instead of mid-episode; the real
    # schedule is exercised by the contract test.
    env._v3_switch_step[:] = 3
    env._v3_switch_done[:] = False
    calls = []
    original = env.simulator.resample_v3_physics

    def counted(env_ids, *, mass, com):
        # Genesis stores runtime payload/COM deltas in the solver's *state*
        # (``links_state.mass_shift`` / ``i_pos_shift``), not in static
        # ``links_info.inertial_mass``.  Snapshot those actual live solver
        # states around the setter call; label buffers alone would not suffice.
        robot = env.simulator._robot
        base_link_idx = torch.tensor(
            [robot._link_start + env.simulator._base_link_index],
            device=env.device,
            dtype=torch.int,
        )
        before_effective_mass = robot._solver.get_links_mass_shift(
            base_link_idx, env_ids
        ).clone()
        before_effective_com = robot._solver.get_links_COM_shift(
            base_link_idx, env_ids
        ).clone()
        before_mass = env.simulator._added_base_mass[env_ids].clone()
        before_com = env.simulator._base_com_bias[env_ids].clone()
        original(env_ids, mass=mass, com=com)
        after_effective_mass = robot._solver.get_links_mass_shift(base_link_idx, env_ids)
        after_effective_com = robot._solver.get_links_COM_shift(base_link_idx, env_ids)
        calls.append((
            len(env_ids),
            mass,
            com,
            not torch.equal(before_effective_mass, after_effective_mass),
            not torch.equal(before_effective_com, after_effective_com),
            not torch.equal(before_mass, env.simulator._added_base_mass[env_ids]),
            not torch.equal(before_com, env.simulator._base_com_bias[env_ids]),
        ))

    env.simulator.resample_v3_physics = counted
    try:
        for _ in range(12):
            env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
        actor, critic = first_and_critic(env, task)
        assert actor.shape == (env.num_envs, expected_actor), actor.shape
        assert critic.shape == (env.num_envs, expected_critic), critic.shape
        assert calls, "no V3 live physics switch occurred"
        assert all(n > 0 and mass and com for n, mass, com, *rest in calls)
        assert any(mass_changed for _, _, _, mass_changed, _, _, _ in calls), (
            "set_mass_shift did not change live Genesis solver mass_shift"
        )
        assert any(com_changed for _, _, _, _, com_changed, _, _ in calls), (
            "set_COM_shift did not change live Genesis solver i_pos_shift"
        )
        assert any(label_changed for _, _, _, _, _, label_changed, _ in calls), (
            "mass label buffer never changed"
        )
        assert any(label_changed for _, _, _, _, _, _, label_changed in calls), (
            "CoM label buffer never changed"
        )
        assert torch.isfinite(actor).all()
        assert torch.isfinite(critic).all()
        if task == "go2_v3_mlp":
            mass_api = [name for name in dir(env.simulator._robot) if "mass" in name.lower()]
            print(f"INFO {task}: robot_mass_api={mass_api}")
        print(f"PASS {task}: switches={len(calls)}, actor={tuple(actor.shape)}, critic={tuple(critic.shape)}")
    finally:
        if hasattr(env, "destroy"):
            env.destroy()


def main():
    parser = argparse.ArgumentParser(description="Genesis V3 runtime smoke")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=tuple(TASKS),
                        help="optional subset; enables one independent smoke per GPU")
    args = parser.parse_args()
    gs.init(backend=gs.gpu, logging_level="warning")
    for task in args.tasks:
        shapes = TASKS[task]
        smoke_one(task, *shapes)


if __name__ == "__main__":
    main()
