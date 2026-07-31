# Headless repro of the play.py respawn path for go2_moects on flat terrain.
# Mirrors play.py overrides (_disable_play_domain_rand, auto_reset=False,
# resampling off) and the exact reset call (env.reset_idx([0])).
import os
os.environ.setdefault("SIMULATOR", "genesis")

import torch

from legged_gym.envs import *  # noqa: F401,F403  (registers tasks)
from legged_gym.utils import task_registry
from legged_gym.envs.go2.go2_moects.go2_moects_config import Go2MoECTSCfg
from legged_gym.scripts.import_go2_rl_gym_policy import build_actor_critic

CKPT = "logs/go2_moects/wty_go2_moe_cts_137k/model_0.pt"


def make_flat_env():
    env_cfg = Go2MoECTSCfg()
    env_cfg.env.num_envs = 2
    env_cfg.seed = 1
    # --terrain flat (configure_play_terrain)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.terrain_kwargs = None
    env_cfg.terrain.ued_training_grid = False
    env_cfg.terrain.moe_grid = False
    # play.py overrides
    env_cfg.env.auto_reset = False
    env_cfg.commands.resampling_time = 1.0e9
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_cmd_prob = 0.0
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


def load_policy(device):
    ac = build_actor_critic("cpu")
    ckpt = torch.load(CKPT, map_location="cpu")
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.to(device).eval()

    def policy(obs, hist):
        with torch.no_grad():
            return ac.act_student(obs, hist)
    return policy


def report(env, tag):
    z = env.simulator.base_pos[:, 2].detach().cpu()
    quat = env.simulator.base_quat[0].detach().cpu()
    print(f"[{tag}] base_z={z.numpy().round(3)} quat0={quat.numpy().round(3)}")


def run(env, policy, n, cmd=None, tag="run"):
    obs, priv, hist, critic = env.get_observations()
    for i in range(n):
        if cmd is not None:
            env.commands[:, 0] = cmd[0]
            env.commands[:, 1] = cmd[1]
            env.commands[:, 2] = cmd[2]
        actions = policy(obs, hist)
        obs, priv, hist, critic, rew, done, info = env.step(actions.detach())
        if i % 50 == 49:
            report(env, f"{tag} step {i+1}")
    return obs, hist


def main():
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level="warning")
    env = make_flat_env()
    policy = load_policy(env.device)
    env.reset()
    report(env, "after env.reset()")

    # 1) stand with zero command
    obs, hist = run(env, policy, 200, cmd=(0.0, 0.0, 0.0), tag="stand")

    # 2) drive harder: 1.0 m/s + yaw, like keyboard driving
    obs, hist = run(env, policy, 400, cmd=(1.0, 0.0, 0.5), tag="drive hard")

    # 3) EXACT play.py respawn flow: reset, but the FIRST policy call still
    # uses the stale (falling) obs/history, like the play loop does.
    env.reset_idx(torch.tensor([0], device=env.device))
    report(env, "right after reset_idx")
    env.commands[:, 0] = 0.0
    env.commands[:, 1] = 0.0
    env.commands[:, 2] = 0.0
    stale_actions = policy(obs, hist)  # stale falling obs -> first action
    print(f"stale-obs first action abs mean: {stale_actions.abs().mean().item():.3f}")
    obs, priv, hist, critic, rew, done, info = env.step(stale_actions.detach())
    report(env, "1 step after stale action")
    for i in range(300):
        env.commands[:, 0] = 0.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0
        actions = policy(obs, hist)
        obs, priv, hist, critic, rew, done, info = env.step(actions.detach())
        if i % 50 == 49:
            report(env, f"post-respawn(zero cmd) step {i+1}")

    # 4) "user still holding the drive key": command ramps back right away
    env.reset_idx(torch.tensor([0], device=env.device))
    obs, hist = env.get_observations()[0], env.get_observations()[2]
    for i in range(400):
        env.commands[:, 0] = 1.0
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.5
        actions = policy(obs, hist)
        obs, priv, hist, critic, rew, done, info = env.step(actions.detach())
        if i % 50 == 49:
            report(env, f"post-respawn(drive again) step {i+1}")


if __name__ == "__main__":
    main()
