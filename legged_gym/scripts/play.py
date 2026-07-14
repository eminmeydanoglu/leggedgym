import time

from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.utils.viser_viewer import create_viser_viewer

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick


def configure_play_terrain(env_cfg, terrain_mode):
    """Apply a visual-playback terrain override without changing task training cfgs."""
    if terrain_mode == "flat":
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.terrain_kwargs = None
        return

    if terrain_mode not in {"bumpy", "course"}:
        raise ValueError(f"Unsupported play terrain mode: {terrain_mode}")

    # Genesis currently validates heightfields, while the other backends use the
    # same selected terrain through their trimesh path.
    env_cfg.terrain.mesh_type = "heightfield" if SIMULATOR == "genesis" else "trimesh"
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_length = 8.0
    env_cfg.terrain.terrain_width = 8.0
    env_cfg.terrain.border_size = 2.0
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    if terrain_mode == "bumpy":
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.random_uniform_terrain",
            "min_height": -0.025,
            "max_height": 0.025,
            "step": 0.005,
            "downsampled_scale": 0.25,
        }
    else:
        # Quantize the collider dimensions exactly as rough_stairs_course does
        # when it turns metres into heightfield samples.
        stair_width = 0.45
        horizontal_scale = env_cfg.terrain.horizontal_scale
        env_cfg.terrain.play_stair_collision_spec = {
            "x_start": int(0.58 * (env_cfg.terrain.terrain_length / horizontal_scale)) * horizontal_scale,
            "step_width": max(1, int(stair_width / horizontal_scale)) * horizontal_scale,
            "y_start": int(0.15 * (env_cfg.terrain.terrain_width / horizontal_scale)) * horizontal_scale,
            "y_stop": (int(env_cfg.terrain.terrain_width / horizontal_scale)
                       - int(0.15 * (env_cfg.terrain.terrain_width / horizontal_scale))) * horizontal_scale,
            "step_height": 0.08,
            "num_steps": 4,
        }
        env_cfg.terrain.terrain_kwargs = {
            "type": "terrain_utils.rough_stairs_course",
            "min_height": -0.025,
            "max_height": 0.025,
            "roughness_step": 0.005,
            "stair_height": 0.08,
            "stair_width": 0.45,
        }


def override_configs(env_cfg, args, task_type):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
        task_type: type of the task
    """
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = 2
    if task_type == "cts" or task_type == "cts_amp": # concurrent teacher-student specific
        env_cfg.env.num_teacher = 1
    elif "depth" in task_type:  # depth specific
        env_cfg.env.num_envs = 1 # for depth observation, only support num_envs=1 for now
        env_cfg.env.num_camera_envs = 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))

    configure_play_terrain(env_cfg, args.terrain)

    # Keep selected non-flat terrain compact and deterministic for interactive play.
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.env.debug_draw_terrain_height_points = False
        env_cfg.domain_rand.push_robots = False

    env_cfg.env.debug = True
    # disable automatic reset on failure/timeout for interactive viewing - respawn is
    # triggered manually instead (e.g. via the viser "Respawn" button)
    env_cfg.env.auto_reset = False
    env_cfg.commands.zero_cmd_prob = 0.0 # for testing, use non-zero commands all the time
    env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    # Interactive controls write a yaw-rate directly to commands[:, 2].  Heading
    # mode would overwrite that value every physics step from commands[:, 3], so
    # disable it for both joystick and Viser control.
    env_cfg.commands.heading_command = False
    
    if args.viewer == "viser":
        args.headless = True

def print_debug_info(env, robot_index):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
    """
    # print debug info
    # print("base lin vel: ", env.simulator.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.simulator.base_euler[robot_index, 2].item())
    # print("base height: ", env.simulator.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.simulator.feet_pos[robot_index, :, 2].cpu().numpy())
    # print(f"knee pitch: {env.simulator.dof_pos[robot_index, [13,19]].cpu().numpy()}")
    # print(f"feet distance: {torch.norm(env.simulator.feet_pos[robot_index, 0, [0, 1]] - env.simulator.feet_pos[robot_index, 1, [0, 1]]).item()}")
    # print(f"actions: {env.simulator.dof_pos[robot_index].cpu().numpy()}")
    # print(f"command: {env.commands[robot_index].cpu().numpy()}")
    # print(f"dr_ctrl_delay: {env.simulator.dr_ctrl_delay[robot_index].item()}")
    pass

def interaction_loop(env, policy, args, task_type, viser_viewer=None):
    """Run interaction loop between environment and policy

    Args:
        env: environment object
        policy : a policy that takes observations and outputs actions
        args: command line arguments
        viser_viewer: optional ViserViewer for web-based visualization
    """
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
        
    # Get initial observations according to task type
    if task_type == "ts_depth":
        obs_buf, privileged_obs_buf, depth_image, critic_obs = env.get_observations()
    elif task_type == "ts" or task_type == "cat" or task_type == "cts" or task_type == "cts_amp": # teacher-student specific (including AMP)
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif task_type == "ee":
        estimator_features, _, _ = env.get_observations()
    elif task_type == "dreamwaq":  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    else: # vanilla
        obs_buf = env.get_observations()
    
    # Setup joystick if needed
    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)
    
    frame_dt = 1 / 60.0 # 30Hz
    # interaction loop - runs indefinitely so the viewer stays up continuously
    # instead of the whole simulator being torn down and rebuilt every ~10 episodes
    i = 0
    while True:

        t_start = time.perf_counter()
        # update commands from joystick
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx
        # update commands from viser GUI sliders
        elif viser_viewer is not None:
            cmd = viser_viewer.get_command()
            env.commands[:, 0] = cmd[0]
            env.commands[:, 1] = cmd[1]
            env.commands[:, 2] = cmd[2]
        
        # set the viewer camera to follow the first environment by default
        if args.follow_robot and viser_viewer is None:
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.pos, dtype=np.float32)
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.lookat, dtype=np.float32)
            env.set_viewer_camera(pos, lookat)
            
        # Step the environment according to task type
        if task_type == "ts_depth":
            actions = policy(obs_buf, depth_image)
            obs_buf, privileged_obs_buf, depth_image, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif task_type == "ts" or task_type == "cat" or task_type == "cts":
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif task_type == "ee":
            actions = policy(estimator_features.detach())
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        elif task_type == "dreamwaq":
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        elif task_type == "amp":
            actions = policy(obs_buf.detach())
            obs_buf, _, rews, dones, infos, _, _ = env.step(actions.detach())
        elif task_type == "cts_amp":
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos, _, _ = env.step(actions.detach())
        else:
            actions = policy(obs_buf.detach())
            obs_buf, _, rews, dones, infos = env.step(actions.detach())
        
        if viser_viewer is not None:
            viser_viewer.update_from_simulator(env, robot_index)

        print_debug_info(env, robot_index)
        
        # Update logger info
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.simulator.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.simulator.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.simulator.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.simulator.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
                    # 'contact_forces_z': env.feet_max_force_z[robot_index, 
                    #                                             env.simulator.feet_contact_indices].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        # sleep for the remainder of the frame budget to match real-time playback
        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        i += 1

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg, task_type):
    """export the policy as jit script according to different task types

    Args:
        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    if task_type == "ts_depth":
        exporter = PolicyExporterDepth(alg_runner.alg.actor_critic, train_cfg)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif task_type == "ts" or task_type == "cat" or task_type == "cts" or task_type == "cts_amp":
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif task_type == "ee":
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif task_type == "dreamwaq":
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)


def load_oracle_id_actor_for_playback(alg_runner, train_cfg):
    """Load the deployable P5 actor without requiring an old critic to match.

    The July P5 checkpoint predates the current velocity-augmented critic.  The
    critic is never called by play.py; retaining its shape mismatch must not
    block an otherwise byte-compatible 50-D actor from visual playback.
    """
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    checkpoint_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    checkpoint = torch.load(checkpoint_path, map_location=alg_runner.device, weights_only=False)
    state = checkpoint['model_state_dict']
    actor_state = {
        name.removeprefix('actor.'): value
        for name, value in state.items()
        if name.startswith('actor.')
    }
    alg_runner.alg.actor_critic.actor.load_state_dict(actor_state, strict=True)
    if 'std' in state:
        alg_runner.alg.actor_critic.std.data.copy_(state['std'])
    print(f"Loaded playback actor from: {checkpoint_path} (legacy critic skipped)")
    

def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning',
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    splitted = args.task.split("_")
    # by default, the first part of the task name is robot name, and the second part is task type, e.g. go2_ts, go2_cat, go2_ee, go2_cts, go2_dreamwaq, go2_ts_depth
    # concatenate the parts after the first part to get the task type, e.g. ts, cat, ee, cts, dreamwaq, ts_depth
    task_type = "_".join(splitted[1:])
    print("Task type: ", task_type)
    override_configs(env_cfg, args, task_type)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    # P5's selected July checkpoint has a 50-D critic whereas current code
    # builds a 53-D critic.  The 50-D actor is unchanged and is all play needs.
    actor_only_playback = args.task == 'go2_bench_oracle_id'
    if actor_only_playback:
        train_cfg.runner.resume = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    if actor_only_playback:
        load_oracle_id_actor_for_playback(ppo_runner, train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++ or python)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 
                            train_cfg.runner.load_run, 'exported')
    export_policy(ppo_runner, path, args, env_cfg, train_cfg, task_type)
    
    viser_viewer = None
    if args.viewer == 'viser':
        viser_viewer = create_viser_viewer(env, port=args.viser_port)
        robot_index = 0  # which robot the viewer tracks / respawns, matches interaction_loop
        viser_viewer.set_respawn_callback(
            lambda: env.reset_idx(torch.tensor([robot_index], device=env.device))
        )
        print(f"Viser web viewer started at http://localhost:{args.viser_port}")
    
    interaction_loop(env, policy, args, task_type, viser_viewer=viser_viewer)
    
    if viser_viewer is not None:
        viser_viewer.stop()
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
