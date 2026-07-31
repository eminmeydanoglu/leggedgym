# Time env.step for the shrunk moe showcase (7 cols x 3 levels = 21 envs),
# with and without the measured-heights raycast, on the local GPU.
import os
os.environ.setdefault("SIMULATOR", "genesis")
import time

import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import task_registry
from legged_gym.utils.terrain import MOE_SHOWCASE_LEVELS, moe_showcase_num_columns
from legged_gym.envs.go2.go2_moects.go2_moects_config import Go2MoECTSCfg


def build(measure_heights: bool, force_envs=None, flat=False, small=False):
    env_cfg = Go2MoECTSCfg()
    env_cfg.seed = 1
    t = env_cfg.terrain
    t.mesh_type = "plane" if flat else "heightfield"
    t.curriculum = False
    t.selected = False
    t.terrain_kwargs = None
    t.ued_training_grid = False
    t.moe_grid = not flat
    t.moe_showcase = not flat
    t.moe_showcase_levels = MOE_SHOWCASE_LEVELS
    t.num_rows = len(MOE_SHOWCASE_LEVELS)
    t.num_cols = moe_showcase_num_columns(list(t.terrain_proportions))
    t.border_size = 2.0
    t.max_init_terrain_level = t.num_rows - 1
    t.measure_heights = measure_heights
    if small:
        t.terrain_length = 6.0
        t.terrain_width = 6.0
        t.border_size = 1.0
    env_cfg.env.num_envs = force_envs or (t.num_rows * t.num_cols)
    env_cfg.env.auto_reset = False
    env_cfg.noise.add_noise = False
    dr = env_cfg.domain_rand
    for name in ("randomize_friction", "randomize_base_mass",
                 "randomize_com_displacement", "randomize_pd_gain",
                 "push_robots", "randomize_ctrl_delay"):
        if hasattr(dr, name):
            setattr(dr, name, False)

    import sys
    from legged_gym.utils.helpers import get_args
    argv = sys.argv
    sys.argv = [argv[0], "--task", "go2_moects", "--headless"]
    args = get_args()
    sys.argv = argv
    env, _ = task_registry.make_env(name="go2_moects", args=args, env_cfg=env_cfg)
    return env


def bench(env, steps=300):
    env.reset()
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(20):  # warmup
        env.step(actions)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps
    print(f"  measure_heights={env.cfg.terrain.measure_heights} "
          f"envs={env.num_envs}: {dt*1000:.2f} ms/control-step "
          f"(~{1/dt:.0f} control steps/s)")
    # breakdown: physics (4 substeps + torque computes) vs env-side tail
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.simulator.step(actions)
    torch.cuda.synchronize()
    dt_sim = (time.perf_counter() - t0) / steps
    print(f"    physics only: {dt_sim*1000:.2f} ms  |  env tail (obs/rewards/etc): "
          f"{(dt-dt_sim)*1000:.2f} ms")


def main():
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level="warning")
    # NOTE: Genesis allows only one Scene per process, so run the two variants
    # in sequence via subprocess instead. This file benches ONE variant chosen
    # by argv[1].
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "off"
    if mode == "old":
        # The pre-shrink exhibit: 8 columns (slope_down included) x 5 levels
        # = 40 envs, raycast on. Monkeypatched back for comparison only.
        import legged_gym.utils.terrain as terrain_mod
        old_cols = terrain_mod.moe_showcase_columns
        def with_slope_down(proportions):
            cols = old_cols(proportions)
            p = [float(terrain_mod.np.sum(proportions[:i + 1]))
                 for i in range(len(proportions))]
            cols.insert(1, ("slope_down", 0.5 * (p[0] + (p[0] + p[1]) / 2)))
            return cols
        terrain_mod.moe_showcase_columns = with_slope_down
        terrain_mod.MOE_SHOWCASE_LEVELS = (0, 2, 4, 6, 9)
        global MOE_SHOWCASE_LEVELS
        MOE_SHOWCASE_LEVELS = (0, 2, 4, 6, 9)
        env = build(True)
    else:
        if mode == "flat":
            env = build(False, force_envs=21, flat=True)
        elif mode == "n3":
            env = build(False, force_envs=3)
        elif mode == "small":
            env = build(False, small=True)
        else:
            env = build(mode == "on")
    bench(env)


if __name__ == "__main__":
    main()
