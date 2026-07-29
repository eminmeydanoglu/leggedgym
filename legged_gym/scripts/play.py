import time

from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.utils.viser_viewer import create_viser_viewer
from legged_gym.utils.terrain import (
    build_taxonomy_label_map,
    build_v6_showcase_label_map,
    format_taxonomy_console_map,
    format_v6_showcase_console_map,
    is_taxonomy_terrain_cfg,
    is_v6_showcase_terrain_cfg,
    teleport_env_to_taxonomy_tile,
    TAXONOMY_LABEL_Z_OFFSET,
    TAXONOMY_NUM_LEVELS,
    TAXONOMY_NUM_TYPES,
    V6_LABEL_Z_OFFSET,
    V6_PLAY_PLATFORM_SIZE,
    V6_PLAY_TILE_LENGTH,
    V6_PLAY_TILE_WIDTH,
    V6_SHOWCASE_COLUMNS,
    V6_SHOWCASE_LEVELS,
    V6_TRAINING_NUM_COLS,
    V6_TRAINING_NUM_ROWS,
)

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick


# ``task_type`` is everything after the robot prefix:
# ``"_".join(args.task.split("_")[1:])``.  DreamWaQ tasks return a 5-tuple from
# ``get_observations()`` / ``step()`` instead of a bare obs tensor, so every
# DreamWaQ-family task name has to be listed here or play unpacks the wrong
# arity.  Keep this as the single source of truth -- it is consumed by
# ``interaction_loop``, the step loop and ``export_policy``.
# NOTE: ``v3_dreamwaq`` and ``bench_dreamwaq`` are deliberately NOT added here;
# they were already absent before v6 and adding them would silently change how
# those existing tasks replay.  If they turn out to be broken too, that is a
# separate fix with its own verification.
DREAMWAQ_TASK_TYPES = frozenset({
    "dreamwaq",
    "v4_dreamwaq",
    "v6_frontier_dreamwaq",
})


def _pin_terrain_difficulty(env_cfg, terrain_level, default_level, max_level):
    """Pin the curriculum difficulty to a user-chosen level, fully decoupled from
    the checkpoint.  ``curriculum()`` picks difficulty from the terrain row, and
    the number of rows caps how hard the hardest row is; we size the grid so the
    top (hardest) row *is* the requested level and start every env there.  This
    replaces the old behaviour of reading the difficulty off the model's own cfg
    (``fixed_terrain_level``), which made the same command mean different things
    for different checkpoints."""
    level = default_level if terrain_level is None else terrain_level
    level = int(max(0, min(level, max_level)))
    # rows are difficulty levels 0..num_rows-1; make the requested level the top
    # row and spawn there so the shown difficulty is exactly what was asked for.
    env_cfg.terrain.num_rows = level + 1
    env_cfg.terrain.max_init_terrain_level = level
    # honoured by cfgs whose curriculum supports a hard-pinned level; a harmless
    # no-op attribute otherwise (during play the curriculum never advances).
    env_cfg.terrain.fixed_terrain_level = level
    return level


def configure_play_terrain(
    env_cfg,
    terrain_mode,
    terrain_level=None,
    v6_tile_size="play",
):
    """Apply a visual-playback terrain override.

    Terrain is chosen by the *user* (``--terrain`` / ``--terrain_level``), never
    derived from the loaded checkpoint - the only exception is the explicit
    ``train`` mode, which deliberately reproduces the checkpoint's own training
    terrain (e.g. to mirror eval conditions).  All modes shrink the grid so the
    world builds fast and the Viser heightfield mesh actually renders in the
    browser (the full 10x10 / border=20 training grid is a 1200x1200 sample
    heightfield ~120 m x 120 m; even downsampled that mesh is too heavy and gets
    silently dropped, so the ground vanishes and the robot looks buried).

    For V6/frontier showcases, ``v6_tile_size`` selects tile envelope only:
    - ``play`` (default): legacy 8×8 m tiles so multi-tile grids stay light
    - ``train``: live ``Go2V6FrontierCfg`` length/width/platform (e.g. 80 m)

    Severity formulas always come from the training cfg either way.
    """
    if terrain_mode == "flat":
        env_cfg.terrain.mesh_type = "plane"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.terrain_kwargs = None
        # V5 UED tasks bake ued_training_grid=True; clear it for play overrides
        # so Terrain.__init__ does not keep building the 84-cell training map.
        env_cfg.terrain.ued_training_grid = False
        return

    if terrain_mode == "rough":
        # Model-agnostic ETH game-inspired curriculum terrain: force it onto ANY
        # policy regardless of what it trained on.  num_cols=5 gives one tile of
        # every terrain type (slope | random-uniform | stairs-up | stairs-down |
        # discrete); difficulty (num_rows) is the user's choice via --terrain_level.
        env_cfg.terrain.mesh_type = "heightfield" if SIMULATOR == "genesis" else "trimesh"
        env_cfg.terrain.selected = False
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.border_size = 2.0
        env_cfg.terrain.ued_training_grid = False
        _pin_terrain_difficulty(env_cfg, terrain_level, default_level=3, max_level=9)
        return

    if terrain_mode == "train":
        # Deliberately model-coupled: keep the task's own training terrain type
        # (whatever the checkpoint used - flat for V3, rough curriculum for V4),
        # just shrink the grid to something renderable.  Difficulty still honours
        # --terrain_level when the underlying terrain is a curriculum.
        # Leave ued_training_grid alone so V5 train-mode still rebuilds the
        # checkpoint's 84-cell UED map when that flag is set on the task cfg.
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.border_size = 2.0
        if getattr(env_cfg.terrain, "curriculum", False):
            _pin_terrain_difficulty(env_cfg, terrain_level, default_level=2, max_level=9)
        return

    if terrain_mode in {"taxonomy", "showcase"}:
        # LP-ACRL paper taxonomy exhibit: 6 semantic types × 4 difficulty levels.
        # Static showcase — no curriculum promote/demote.
        env_cfg.terrain.mesh_type = "heightfield" if SIMULATOR == "genesis" else "trimesh"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.terrain_kwargs = None
        env_cfg.terrain.mode = "taxonomy"
        env_cfg.terrain.taxonomy_showcase = True
        env_cfg.terrain.v6_frontier_showcase = False
        # UED training grid has higher priority in Terrain.__init__; must clear
        # it or V5 play still builds the 21-config / 84-task training map.
        env_cfg.terrain.ued_training_grid = False
        env_cfg.terrain.num_rows = TAXONOMY_NUM_LEVELS
        env_cfg.terrain.num_cols = TAXONOMY_NUM_TYPES
        env_cfg.terrain.border_size = 2.0
        # Spawn on the easiest row so the robot does not bury itself in L3 stairs.
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.fixed_terrain_level = 0
        return

    if terrain_mode in {"v6", "v6_frontier", "frontier", "v6_full"}:
        # Real go2_v6_frontier task space (V4 ETH curriculum generators), not
        # the V5 taxonomy tables.  Subsampled by default so play tiles fit in
        # Viser; v6_full keeps both replicas of every family.
        from legged_gym.envs.go2.go2_v6_frontier_config import Go2V6FrontierCfg

        src = Go2V6FrontierCfg.terrain
        tile_mode = str(v6_tile_size or "play").lower()
        if tile_mode not in {"play", "train"}:
            raise ValueError(
                f"Unsupported v6_tile_size {v6_tile_size!r}; expected 'play' or 'train'"
            )
        env_cfg.terrain.mesh_type = "heightfield" if SIMULATOR == "genesis" else "trimesh"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.terrain_kwargs = None
        env_cfg.terrain.mode = "v6_full" if terrain_mode == "v6_full" else "v6"
        env_cfg.terrain.v6_frontier_showcase = True
        env_cfg.terrain.taxonomy_showcase = False
        # UED grid would otherwise win if left True on a V5/V6 task cfg.
        env_cfg.terrain.ued_training_grid = False
        # Proportions + difficulty formulas always mirror live training so
        # severity stays truthful.  Tile envelope is selectable: play default
        # keeps the legacy 8 m multi-tile showcase light; train copies the
        # current training length/width/platform (heavy for multi-tile grids).
        env_cfg.terrain.terrain_proportions = list(src.terrain_proportions)
        env_cfg.terrain.terrain_curriculum_difficulty = dict(
            src.terrain_curriculum_difficulty
        )
        env_cfg.terrain.terrain_replica_variation = float(
            src.terrain_replica_variation
        )
        env_cfg.terrain.v6_tile_size = tile_mode
        if tile_mode == "train":
            env_cfg.terrain.terrain_length = float(src.terrain_length)
            env_cfg.terrain.terrain_width = float(src.terrain_width)
            env_cfg.terrain.platform_size = float(src.platform_size)
        else:
            env_cfg.terrain.terrain_length = float(V6_PLAY_TILE_LENGTH)
            env_cfg.terrain.terrain_width = float(V6_PLAY_TILE_WIDTH)
            env_cfg.terrain.platform_size = float(V6_PLAY_PLATFORM_SIZE)
        env_cfg.terrain.v6_showcase_seed = int(
            getattr(env_cfg.terrain, "v6_showcase_seed", 0)
        )
        # Sharp stair risers for trimesh backends; heightfield path uses Viser
        # edge-preserving meshing separately.
        env_cfg.terrain.simplify_mesh = False
        if terrain_mode == "v6_full":
            env_cfg.terrain.v6_showcase_levels = tuple(range(V6_TRAINING_NUM_ROWS))
            env_cfg.terrain.v6_showcase_columns = tuple(range(V6_TRAINING_NUM_COLS))
            env_cfg.terrain.num_rows = V6_TRAINING_NUM_ROWS
            env_cfg.terrain.num_cols = V6_TRAINING_NUM_COLS
        else:
            env_cfg.terrain.v6_showcase_levels = tuple(V6_SHOWCASE_LEVELS)
            env_cfg.terrain.v6_showcase_columns = tuple(V6_SHOWCASE_COLUMNS)
            env_cfg.terrain.num_rows = len(V6_SHOWCASE_LEVELS)
            env_cfg.terrain.num_cols = len(V6_SHOWCASE_COLUMNS)
        env_cfg.terrain.border_size = 2.0
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.fixed_terrain_level = 0
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
    # number of environments (respect --num_envs when given, e.g. to spread
    # robots across every terrain type on the training grid)
    showcase_terrains = {
        "taxonomy", "showcase", "v6", "v6_frontier", "frontier", "v6_full",
    }
    default_num_envs = 1 if getattr(args, "terrain", None) in showcase_terrains else 2
    env_cfg.env.num_envs = args.num_envs if getattr(args, "num_envs", None) else default_num_envs
    if task_type == "cts" or task_type == "cts_amp": # concurrent teacher-student specific
        env_cfg.env.num_teacher = 1
    elif "depth" in task_type:  # depth specific
        env_cfg.env.num_envs = 1 # for depth observation, only support num_envs=1 for now
        env_cfg.env.num_camera_envs = 1
    # Showcase exhibits: keep robots few so they do not occlude the grid.
    if getattr(args, "terrain", None) in showcase_terrains:
        env_cfg.env.num_envs = min(int(env_cfg.env.num_envs), 4)
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))

    configure_play_terrain(
        env_cfg,
        args.terrain,
        getattr(args, "terrain_level", None),
        v6_tile_size=getattr(args, "v6_tile_size", "play"),
    )

    # Keep selected non-flat terrain compact and deterministic for interactive play.
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.env.debug_draw_terrain_height_points = False
        env_cfg.domain_rand.push_robots = False

    env_cfg.env.debug = True
    # Play drives commands/terrain from the viewer (or fixed ranges), not from the
    # training UED teacher.  Same contract as ued_rollout: leave ued_enabled False
    # so reset_idx does not require an installed episode curriculum.
    if getattr(env_cfg.env, "ued_enabled", False):
        env_cfg.env.ued_enabled = False
    # disable automatic reset on failure/timeout for interactive viewing - respawn is
    # triggered manually instead (e.g. via the viser "Respawn" button)
    env_cfg.env.auto_reset = False
    env_cfg.commands.zero_cmd_prob = 0.0 # for testing, use non-zero commands all the time
    # Interactive control writes env.commands every frame; stop the env from
    # periodically resampling a random command (which would otherwise leak into
    # the policy observation for one step every resampling_time seconds).
    env_cfg.commands.resampling_time = 1.0e9
    env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    # Interactive controls write a yaw-rate directly to commands[:, 2].  Heading
    # mode would overwrite that value every physics step from commands[:, 3], so
    # disable it for both joystick and Viser control.
    env_cfg.commands.heading_command = False

    # Elevated / oblique camera defaults for showcase exhibits so the full grid
    # is visible without --follow_robot.  Offsets scale with the larger grid
    # extent (taxonomy 4x6×6 m vs v6 5x6×8 m need different distances).
    if getattr(args, "terrain", None) in showcase_terrains:
        length = float(env_cfg.terrain.terrain_length)
        width = float(env_cfg.terrain.terrain_width)
        cx = 0.5 * env_cfg.terrain.num_rows * length
        cy = 0.5 * env_cfg.terrain.num_cols * width
        extent = max(
            float(env_cfg.terrain.num_rows) * length,
            float(env_cfg.terrain.num_cols) * width,
        )
        # Proportional oblique: ~0.45× extent XY pull-back, ~0.7× extent height.
        cam_xy = 0.45 * extent
        cam_z = 0.70 * extent
        env_cfg.viewer.pos = [cx - cam_xy, cy - cam_xy * (22.0 / 18.0), cam_z]
        env_cfg.viewer.lookat = [cx, cy, 0.0]
        # Relative-to-robot offsets unused for showcase fixed camera, but keep
        # sensible values if the user later enables --follow_robot.
    
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
    elif task_type in {"ts", "cat", "cts", "cts_amp", "rma", "bench_rma", "v3_rma", "v4_rma"}: # teacher-student specific (including RMA)
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif task_type == "ee":
        estimator_features, _, _ = env.get_observations()
    elif task_type in DREAMWAQ_TASK_TYPES:  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    else: # vanilla
        obs_buf = env.get_observations()
    
    # Setup joystick if needed
    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)

    showcase_mode = (
        is_taxonomy_terrain_cfg(env.cfg.terrain)
        or is_v6_showcase_terrain_cfg(env.cfg.terrain)
        or getattr(args, "terrain", None) in {
            "taxonomy", "showcase", "v6", "v6_frontier", "frontier", "v6_full",
        }
    )
    v6_mode = is_v6_showcase_terrain_cfg(env.cfg.terrain) or getattr(args, "terrain", None) in {
        "v6", "v6_frontier", "frontier", "v6_full",
    }
    taxonomy_camera_set = False
    if showcase_mode:
        # Console fallback for labels (Genesis has no draw_debug_text).
        terrain = getattr(env.simulator, "_terrain", None)
        label_map = None
        if terrain is not None and getattr(terrain, "taxonomy_labels", None):
            label_map = terrain.taxonomy_labels
        elif terrain is not None and hasattr(terrain, "env_origins"):
            if v6_mode:
                label_map = build_v6_showcase_label_map(
                    terrain.env_origins,
                    levels=getattr(env.cfg.terrain, "v6_showcase_levels", V6_SHOWCASE_LEVELS),
                    columns=getattr(env.cfg.terrain, "v6_showcase_columns", V6_SHOWCASE_COLUMNS),
                    difficulty_cfg=getattr(
                        env.cfg.terrain, "terrain_curriculum_difficulty", None
                    ),
                    z_offset=V6_LABEL_Z_OFFSET,
                )
            else:
                label_map = build_taxonomy_label_map(
                    terrain.env_origins, z_offset=TAXONOMY_LABEL_Z_OFFSET
                )
        if label_map:
            if v6_mode:
                print(format_v6_showcase_console_map(label_map))
            else:
                print(format_taxonomy_console_map(label_map))
            if viser_viewer is not None and hasattr(viser_viewer, "set_taxonomy_labels"):
                viser_viewer.set_taxonomy_labels(label_map)
    
    frame_dt = 1 / 60.0 # 30Hz
    # interaction loop - runs indefinitely so the viewer stays up continuously
    # instead of the whole simulator being torn down and rebuilt every ~10 episodes
    i = 0
    while True:

        t_start = time.perf_counter()
        # Drain Viser GUI actions on the sim thread.  Teleport/Respawn buttons
        # only queue work; executing env.reset_idx on the GUI worker races
        # oracle history deques (RuntimeError: deque mutated during iteration).
        if viser_viewer is not None:
            if hasattr(viser_viewer, "take_pending_respawn") and viser_viewer.take_pending_respawn():
                env.reset_idx(torch.tensor([robot_index], device=env.device))
                if hasattr(viser_viewer, "clear_drive_command"):
                    viser_viewer.clear_drive_command()
            if hasattr(viser_viewer, "take_pending_taxonomy_spawn"):
                spawn = viser_viewer.take_pending_taxonomy_spawn()
                if spawn is not None:
                    level, type_idx = spawn
                    ok = teleport_env_to_taxonomy_tile(env, robot_index, level, type_idx)
                    if ok and hasattr(viser_viewer, "clear_drive_command"):
                        viser_viewer.clear_drive_command()
                    if hasattr(viser_viewer, "report_taxonomy_spawn_result"):
                        viser_viewer.report_taxonomy_spawn_result(level, type_idx, ok)
                    print(f"[play] taxonomy spawn → type={type_idx} L{level} (ok={ok})")

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
        elif showcase_mode and not taxonomy_camera_set and viser_viewer is None:
            # One-shot elevated oblique view of the full showcase grid.
            # Headless / no native viewer: skip (scene.viewer is None).
            try:
                env.set_viewer_camera(
                    np.array(env.cfg.viewer.pos, dtype=np.float32),
                    np.array(env.cfg.viewer.lookat, dtype=np.float32),
                )
            except Exception:
                pass
            taxonomy_camera_set = True
            
        # Step the environment according to task type
        if task_type == "ts_depth":
            actions = policy(obs_buf, depth_image)
            obs_buf, privileged_obs_buf, depth_image, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif task_type in {"ts", "cat", "cts", "rma", "bench_rma", "v3_rma", "v4_rma"}:
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif task_type == "ee":
            actions = policy(estimator_features.detach())
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        elif task_type in DREAMWAQ_TASK_TYPES:
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
            viser_viewer.update_from_simulator(env, [robot_index])
            viser_viewer.update_live_telemetry(
                env.simulator.base_lin_vel[robot_index].detach().cpu().numpy(),
                env.simulator.base_ang_vel[robot_index, 2].item(),
                env.commands[robot_index, :3].detach().cpu().numpy(),
            )

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
    elif task_type in DREAMWAQ_TASK_TYPES:
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
        # Respawn/Teleport are drained on the main thread in interaction_loop
        # (see take_pending_respawn / take_pending_taxonomy_spawn).  Do not
        # call env.reset_idx from Viser GUI worker callbacks.
        if (
            is_taxonomy_terrain_cfg(env_cfg.terrain)
            or is_v6_showcase_terrain_cfg(env_cfg.terrain)
            or getattr(args, "terrain", None) in {
                "taxonomy", "showcase", "v6", "v6_frontier", "frontier", "v6_full",
            }
        ):
            # Panel may already exist from create_viser_viewer; ensure it exists.
            if not hasattr(viser_viewer, "_taxonomy_spawn_gui"):
                if is_v6_showcase_terrain_cfg(env_cfg.terrain) or getattr(
                    args, "terrain", None
                ) in {"v6", "v6_frontier", "frontier", "v6_full"}:
                    from legged_gym.utils.terrain import (
                        V6_FAMILY_NAMES,
                        v6_family_index_for_column,
                    )
                    cols = getattr(
                        env_cfg.terrain,
                        "v6_showcase_columns",
                        tuple(range(int(env_cfg.terrain.num_cols))),
                    )
                    type_names = tuple(
                        f"{V6_FAMILY_NAMES[v6_family_index_for_column(c)]} c{c}"
                        for c in cols
                    )
                else:
                    from legged_gym.utils.terrain import TAXONOMY_TYPE_NAMES
                    type_names = TAXONOMY_TYPE_NAMES
                viser_viewer.setup_taxonomy_spawn_panel(
                    type_names=type_names,
                    num_levels=int(env_cfg.terrain.num_rows),
                )
        print(f"Viser web viewer started at http://localhost:{args.viser_port}")
    
    interaction_loop(env, policy, args, task_type, viser_viewer=viser_viewer)
    
    if viser_viewer is not None:
        viser_viewer.stop()
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
