import time

from contextlib import contextmanager

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
    MOE_SHOWCASE_LEVELS,
    MOE_TRAINING_NUM_ROWS,
    moe_showcase_columns,
    moe_showcase_num_columns,
    format_moe_showcase_console_map,
)
from legged_gym.envs.go2.go2_moects.go2_moects_config import Go2MoECTSCommonCfg

from legged_gym.utils.model_swap import CheckpointSwapper
from legged_gym.utils.native_ui import NativeArenaUI
from legged_gym.scripts.input_source import build_input_source

import numpy as np
import torch


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


def _disable_play_domain_rand(env_cfg):
    """Put the env on nominal physics and clean observations.

    Clears every ``domain_rand.randomize_*`` switch plus the pushers, and turns
    off observation noise.  Nominal values are whatever the cfg/asset already
    carries (friction ratio stays at the simulator default of 1.0), so nothing
    here invents numbers -- it only stops the per-env sampling.
    """
    dr = env_cfg.domain_rand
    for name in dir(dr):
        if name.startswith("randomize_") and isinstance(getattr(dr, name), bool):
            setattr(dr, name, False)
    for name in ("push_robots", "push_links"):
        if hasattr(dr, name):
            setattr(dr, name, False)
    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False


def _drop_torch_default_device_mode():
    """Undo the global ``torch.set_default_device`` that ``gs.init`` installs.

    ``genesis/__init__.py`` calls ``torch.set_default_device(device)`` "just in
    case" when the backend comes up.  That does not merely record a preference:
    it pushes a global ``TorchFunctionMode``, so from then on EVERY torch call
    in the process detours into Python (``torch/utils/_device.py``
    ``__torch_function__``), tests whether it is a tensor constructor, and
    re-dispatches.  Measured on this box that is ~1.9 us per op -- invisible in
    training, where one op covers 4096 envs, but decisive in interactive play,
    where a single robot runs hundreds of tiny ops per control step at 50 Hz.
    py-spy attributes 58%% of play's main-thread samples to that one frame, and
    dropping the mode alone moves a go2_moects play session from 0.65x to 1.0x
    real time on an RTX 3050 laptop.

    Safe here because nothing in this stack relies on the implicit default:
    legged_gym and Genesis both pass ``device=`` explicitly at every tensor
    construction (``gs.device`` / ``self.device``).  The default *dtype* that
    ``gs.init`` also sets is deliberately left alone -- it costs nothing and
    Genesis's float width really is a global contract.
    """
    torch.set_default_device(None)


def _disable_play_rewards(env):
    """Stop computing rewards during play; keep the command-resample tail.

    Play never reads ``rew_buf``: ``auto_reset`` is off (see override_configs),
    so ``reset_idx`` -- the only place that fills ``infos["episode"]`` from
    ``episode_sums`` -- does not run on its own, and the Logger below therefore
    has nothing to print.  Meanwhile the reward loop evaluates every term and
    accumulates ``episode_sums`` on every one of the 50 control steps a second,
    which profiles at ~18%% of the frame.

    The one side effect that must survive is the tail of
    ``WtyCurriculumMixin.compute_reward``: it calls
    ``_resample_commands_if_due()`` after scoring, deliberately, so a resample
    cannot change the command a reward was scored against.  Play pins
    ``resampling_time`` to 1e9 so it never fires, but the call is kept anyway
    rather than silently dropping a contract that lives in another module.

    Instance attribute, not a class patch: it shadows the bound method for this
    env only and cannot leak into a training process that imports this module.
    """
    resample = getattr(env, "_resample_commands_if_due", None)

    def compute_reward():
        if resample is not None:
            resample()

    env.compute_reward = compute_reward


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
        env_cfg.terrain.moe_grid = False
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
        env_cfg.terrain.moe_grid = False
        _pin_terrain_difficulty(env_cfg, terrain_level, default_level=3, max_level=9)
        return

    if terrain_mode == "train":
        # Deliberately model-coupled: keep the task's own training terrain type
        # (whatever the checkpoint used - flat for V3, rough curriculum for V4),
        # just shrink the grid to something renderable.  Difficulty still honours
        # --terrain_level when the underlying terrain is a curriculum.
        # Leave ued_training_grid / moe_grid alone so V5 and go2_moects train-mode
        # still rebuild the checkpoint's own map when those flags are set.
        if not getattr(env_cfg.terrain, "moe_grid", False):
            # The moe grid's columns ARE the terrain-type taxonomy (cols2id /
            # name2cols drive per-type command limits and tracking sigma);
            # shrinking to 5 would drop 15 of the 20 types.
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
        # Same for the go2_moects grid: its dispatch branch also outranks
        # curriculum/selected in Terrain.__init__.
        env_cfg.terrain.moe_grid = False
        env_cfg.terrain.num_rows = TAXONOMY_NUM_LEVELS
        env_cfg.terrain.num_cols = TAXONOMY_NUM_TYPES
        env_cfg.terrain.border_size = 2.0
        # Spawn on the easiest row so the robot does not bury itself in L3 stairs.
        env_cfg.terrain.max_init_terrain_level = 0
        env_cfg.terrain.fixed_terrain_level = 0
        return

    if terrain_mode == "moe":
        # go2_moects terrain bank as a static exhibit: every DISTINCT terrain
        # type once (the training grid repeats them proportionally) x a few
        # spaced difficulty levels, generated by the same tile code with the
        # same `level / 10` difficulty the training grid would use.
        if SIMULATOR != "genesis":
            raise ValueError("--terrain moe is heightfield-only (Genesis); "
                             f"SIMULATOR={SIMULATOR}")
        env_cfg.terrain.mesh_type = "heightfield"
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = False
        env_cfg.terrain.terrain_kwargs = None
        env_cfg.terrain.ued_training_grid = False
        env_cfg.terrain.taxonomy_showcase = False
        env_cfg.terrain.v6_frontier_showcase = False
        env_cfg.terrain.moe_grid = True
        env_cfg.terrain.moe_showcase = True
        # The student inference path never reads heights (they only feed the
        # privileged/critic obs, which play leaves unused).  The Genesis
        # heightfield raycast is cheap on modern GPUs (<2% of step time on an
        # RTX 3050 Laptop), but with the flag off the simulator hands out its
        # zero buffer and skips the cast entirely -- free on weak setups.
        env_cfg.terrain.measure_heights = False
        # Tasks other than go2_moects have no moe proportions of their own; use
        # the go2_moects bank so --terrain moe means the same thing everywhere.
        # Keep the checkpoint's actual training bank.  Its zero-weight
        # stepping_stones and gap bands are intentionally absent, leaving a
        # smaller seven-column showcase made only of trained terrain types.
        env_cfg.terrain.terrain_proportions = list(
            Go2MoECTSCommonCfg.terrain.terrain_proportions)
        env_cfg.terrain.terrain_length = float(Go2MoECTSCommonCfg.terrain.terrain_length)
        env_cfg.terrain.terrain_width = float(Go2MoECTSCommonCfg.terrain.terrain_width)
        # Play-only raster reduction: training remains at the original 0.1 m
        # sample spacing, while the interactive showcase uses 0.15 m.  This
        # cuts each 8 m tile from 80x80 to about 54x54 samples without changing
        # tile dimensions, terrain amplitudes, or policy/control timing.
        env_cfg.terrain.horizontal_scale = max(
            float(getattr(env_cfg.terrain, "horizontal_scale", 0.1)), 0.15
        )
        env_cfg.terrain.terrain_spacing = float(Go2MoECTSCommonCfg.terrain.terrain_spacing)
        levels = tuple(getattr(env_cfg.terrain, "moe_showcase_levels", MOE_SHOWCASE_LEVELS))
        if terrain_level is not None:
            levels = (int(max(0, min(terrain_level, MOE_TRAINING_NUM_ROWS - 1))),)
        env_cfg.terrain.moe_showcase_levels = levels
        env_cfg.terrain.num_rows = len(levels)
        env_cfg.terrain.num_cols = moe_showcase_num_columns(
            env_cfg.terrain.terrain_proportions)
        env_cfg.terrain.border_size = 2.0
        # Interactive play drives ONE robot (see override_configs); this line is
        # what lets the optional full-grid exhibit still work.  The wty mixin
        # assigns terrain_types = arange // (num_envs / num_cols) and
        # terrain_levels = arange % (max_init_terrain_level + 1), so with
        # num_envs = num_rows * num_cols each column block walks every row, and
        # with num_envs = 1 the single robot lands on (row 0, col 0), from which
        # Tab / [ ] reach every other cell by teleport.
        env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
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
        # UED grid would otherwise win if left True on a V5/V6 task cfg; the
        # go2_moects grid outranks both, so clear it too.
        env_cfg.terrain.ued_training_grid = False
        env_cfg.terrain.moe_grid = False
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
    # Both flags outrank `selected` in the Terrain.__init__ dispatch.
    env_cfg.terrain.ued_training_grid = False
    env_cfg.terrain.moe_grid = False
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
    # The moe exhibit gets the camera framing below but not the "few robots"
    # clamp the others use.
    moe_showcase = getattr(args, "terrain", None) == "moe"
    # Every terrain -- exhibits and plain alike -- defaults to ONE robot.  The
    # moe grid used to default to one robot per (level, type) cell (3 rows x 7
    # columns = 21).  Each of those streams ~13 body meshes per frame, and that,
    # not the terrain size, is what made play feel slow: the robot is a single
    # batched Genesis entity and heightfield collision scales with contacts,
    # not area.  Plain terrains kept a legacy default of 2, which buys nothing
    # on screen but doubles the per-frame sim + viewer-stream cost on a laptop
    # GPU.  The old full-grid exhibit is still one flag away: ``--num_envs 21``.
    default_num_envs = 1
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

    if moe_showcase and env_cfg.env.num_envs == 1:
        # With curriculum off and no fixed level, _get_env_origins draws the
        # terrain row at random -- fine when 21 robots cover the grid, but with
        # the single arena robot it means "which difficulty am I on?" changes
        # every launch.  Pin row 0 (the easiest showcase level); Tab / [ ] move
        # from there.
        env_cfg.terrain.fixed_terrain_level = 0

    # Keep selected non-flat terrain compact and deterministic for interactive play.
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.env.debug_draw_terrain_height_points = False

    if not getattr(args, "domain_rand", False):
        # Play answers "how does this policy move?", so it must not also be
        # sampling adversarial physics.  The go2 training ranges draw ground
        # friction from [0.0, 2.0]: an env that lands near 0 skates and hops on
        # flat ground no matter how good the checkpoint is, and with one robot
        # per showcase tile that reads as a broken policy.  --domain_rand keeps
        # the training ranges.
        _disable_play_domain_rand(env_cfg)

    # Debug geometry is extremely expensive in interactive Genesis playback:
    # post_physics_step() clears and rebuilds debug objects every control step.
    # It is not needed for driving or HUD input, so keep it opt-in for play.
    env_cfg.env.debug = bool(getattr(args, "debug_vis", False))
    # ``LeggedRobot._pre_sim_step`` historically owns a training-viewer follow
    # camera whenever debug is off.  Play has its own explicit ``--follow_robot``
    # path (and the native arena's T toggle).  Letting both paths write the
    # camera in one control frame causes visible jumps, particularly when a
    # showcase config has an elevated overview offset.  This flag changes only
    # camera ownership; it leaves debug-draw, FPS and terrain configuration as
    # configured above.
    env_cfg.env.manual_viewer_camera = True
    # Native play camera only: a wider view helps both robot-relative modes,
    # and a world-up lock prevents a prior mouse-orbit roll from tilting the
    # horizon the next time play writes a camera pose.
    env_cfg.viewer.camera_fov = 55.0
    env_cfg.viewer.lock_camera_world_up = True
    # Visual smoothness, in two halves (see GenesisSimulator for the full rate
    # table).  The window is redrawn at ``render_refresh_rate`` Hz by the
    # viewer thread, while the POSE it draws only changes when the simulator
    # publishes one.  Publishing once per control step (the old setting here)
    # meant 50 new poses a second under a 60 Hz window: every sixth frame was a
    # duplicate, which looks like judder however fast the window redraws.
    # Publishing every 2nd substep gives 100 Hz of fresh poses -- above the
    # window rate, so every drawn frame is new -- for one extra graphics
    # submission per control step.  This costs sim-thread time and is only
    # affordable because play no longer burns ~30%% of its frame on the torch
    # default-device mode and the unused reward loop.
    env_cfg.viewer.render_refresh_rate = 60
    env_cfg.viewer.visualizer_update_stride = 2
    # Camera smoothing runs every control frame.  Sampling it less frequently
    # made the view visibly step at 25 Hz without moving the measured RTF, so
    # do not trade visual quality for a non-existent throughput gain.
    env_cfg.viewer.follow_camera_stride = 1
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

    # Preserve the task's normal relative follow view before showcase mode
    # replaces ``viewer.pos`` with a large, fixed overview of the whole grid.
    # The two camera modes have different coordinate contracts: overview is a
    # world-space pose, while follow is an offset from the robot.
    env_cfg.viewer.follow_robot_pos = list(env_cfg.viewer.pos)
    env_cfg.viewer.follow_robot_lookat = list(env_cfg.viewer.lookat)

    # Elevated / oblique camera defaults for showcase exhibits so the full grid
    # is visible without --follow_robot.  Offsets scale with the larger grid
    # extent (taxonomy 4x6×6 m vs v6 5x6×8 m need different distances).
    if getattr(args, "terrain", None) in showcase_terrains or moe_showcase:
        length = float(env_cfg.terrain.terrain_length)
        width = float(env_cfg.terrain.terrain_width)
        # the moe layout leaves terrain_spacing metre gaps between tiles
        gap = float(getattr(env_cfg.terrain, "terrain_spacing", 0.0)) if moe_showcase else 0.0
        span_x = env_cfg.terrain.num_rows * length + (env_cfg.terrain.num_rows - 1) * gap
        span_y = env_cfg.terrain.num_cols * width + (env_cfg.terrain.num_cols - 1) * gap
        cx, cy = 0.5 * span_x, 0.5 * span_y
        extent = max(span_x, span_y)
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

def interaction_loop(env, policy, args, task_type, viser_viewer=None,
                     input_source=None, arena_ui=None):
    """Run interaction loop between environment and policy

    Args:
        env: environment object
        policy : a policy that takes observations and outputs actions
        args: command line arguments
        viser_viewer: optional ViserViewer for web-based visualization
        input_source: optional MergedSource (native viewer: keyboard + gamepad)
        arena_ui: optional NativeArenaUI whose events are drained here, on the
            sim thread -- Genesis keybind callbacks run on the viewer thread and
            must not call env.reset_idx themselves.
    """

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
        
    # Get initial observations according to task type
    if task_type == "ts_depth":
        obs_buf, privileged_obs_buf, depth_image, critic_obs = env.get_observations()
    elif task_type in {"ts", "cat", "cts", "cts_amp", "moects", "rma", "bench_rma", "v3_rma", "v4_rma"}: # teacher-student specific (including RMA and MoE-CTS)
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif task_type == "ee":
        estimator_features, _, _ = env.get_observations()
    elif task_type in DREAMWAQ_TASK_TYPES:  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    else: # vanilla
        obs_buf = env.get_observations()
    
    # go2_moects terrain bank exhibit: print the column legend so the grid is
    # readable at a glance (which column is which terrain type, which row is
    # which difficulty level).
    if getattr(env.cfg.terrain, "moe_showcase", False):
        print(format_moe_showcase_console_map(
            moe_showcase_columns(env.cfg.terrain.terrain_proportions),
            tuple(getattr(env.cfg.terrain, "moe_showcase_levels", MOE_SHOWCASE_LEVELS)),
        ))

    showcase_mode = (
        is_taxonomy_terrain_cfg(env.cfg.terrain)
        or is_v6_showcase_terrain_cfg(env.cfg.terrain)
        or getattr(env.cfg.terrain, "moe_showcase", False)
        or getattr(args, "terrain", None) in {
            "taxonomy", "showcase", "v6", "v6_frontier", "frontier", "v6_full", "moe",
        }
    )
    v6_mode = is_v6_showcase_terrain_cfg(env.cfg.terrain) or getattr(args, "terrain", None) in {
        "v6", "v6_frontier", "frontier", "v6_full",
    }
    taxonomy_camera_set = False
    if showcase_mode and not getattr(env.cfg.terrain, "moe_showcase", False):
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
    
    # One env.step() advances a FULL control step -- sim dt x control.decimation
    # -- so the real-time budget for a frame is exactly ``env.dt``.  This used
    # to be a hard-coded 1/60 s, which asked the loop to produce 20 ms of
    # simulated time every 16.7 ms: a 1.2x-real-time target that could never be
    # met, so the sleep below never fired and playback always read as "behind".
    # Deriving it from the env keeps it correct for any task/decimation.
    frame_dt = float(getattr(env, "dt", 0.0)) or (1.0 / 50.0)
    print(f"[play] real-time frame budget: {frame_dt * 1000:.1f} ms "
          f"({1.0 / frame_dt:.1f} Hz control rate)")
    # interaction loop - runs indefinitely so the viewer stays up continuously
    # instead of the whole simulator being torn down and rebuilt every ~10 episodes
    i = 0
    t_prev = time.perf_counter()
    # Smoothed frame rate for the on-screen HUD.  EMA rather than an instant
    # value so the number is readable instead of flickering.
    fps_ema = None
    realtime_ema = None
    # Throttled console mirror of the HUD telemetry: the HUD does not exist in
    # headless runs and is easy to miss on screen, while this line is also what
    # makes scripted RTF measurement possible.  The guard is one
    # time.monotonic() read per frame -- negligible -- and the timestamp is
    # pushed forward at print time, so it can never double-print an iteration.
    rtf_print_next = 0.0
    while True:

        t_start = time.perf_counter()
        # Real elapsed time, so the keyboard ramp does not depend on how many
        # frames the sim managed to run.  Capped so a hitch cannot teleport the
        # command straight to its target.
        input_dt = min(max(t_start - t_prev, 0.0), 0.1)
        # Wall-clock rate of the previous frame, including the real-time sleep,
        # so "1.00x real-time" on the HUD means exactly that.
        if input_dt > 1e-6:
            inst_fps = 1.0 / input_dt
            inst_rt = frame_dt / input_dt
            fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps
            realtime_ema = inst_rt if realtime_ema is None else 0.9 * realtime_ema + 0.1 * inst_rt
        t_prev = t_start
        if fps_ema is not None and time.monotonic() >= rtf_print_next:
            print(f"[play] {fps_ema:5.1f} FPS   {realtime_ema:.2f}x real-time")
            rtf_print_next = time.monotonic() + 5.0

        # Arena actions (tab / level / respawn / model swap) are queued by
        # Genesis keybind callbacks on the viewer thread; execute them here.
        if arena_ui is not None:
            arena_ui.drain_and_apply()

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

        # Native viewer: keyboard and gamepad coexist inside MergedSource (the
        # pad wins only while a stick is off-centre), so there is no longer an
        # exclusive if/elif between input backends.
        if input_source is not None:
            cmd = input_source.poll(input_dt)
            env.commands[:, 0] = cmd[0]
            env.commands[:, 1] = cmd[1]
            env.commands[:, 2] = cmd[2]
        # Viser stays the remote / headless path (Lightning.ai, UHeM) and keeps
        # its own key handling unchanged.
        elif viser_viewer is not None:
            cmd = viser_viewer.get_command()
            env.commands[:, 0] = cmd[0]
            env.commands[:, 1] = cmd[1]
            env.commands[:, 2] = cmd[2]


        # Camera.  After a native-arena T press, the arena is the only camera
        # owner; its rear/front modes update after the full control step and
        # its free mode leaves the Genesis trackball untouched.
        arena_camera_override = (
            arena_ui is not None and arena_ui.camera_override_active
        )
        if args.follow_robot and viser_viewer is None and not arena_camera_override:
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(
                getattr(env.cfg.viewer, "follow_robot_pos", env.cfg.viewer.pos),
                dtype=np.float32,
            )
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(
                getattr(env.cfg.viewer, "follow_robot_lookat", env.cfg.viewer.lookat),
                dtype=np.float32,
            )
            env.set_viewer_camera(pos, lookat)
        elif showcase_mode and not taxonomy_camera_set and viser_viewer is None:
            # One-shot camera placement.  With the arena the interesting frame is
            # the tile the robot actually stands on, not the whole grid.
            if arena_ui is not None and arena_ui.frame_current_tile():
                pass
            else:
                # Elevated oblique view of the full showcase grid.
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
        elif task_type in {"ts", "cat", "cts", "moects", "rma", "bench_rma", "v3_rma", "v4_rma"}:
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

        # Native follow is updated once per full control frame.  Genesis's
        # built-in entity follow runs on every physics substep and visibly
        # jitters on a decimated legged-robot simulation.
        if arena_ui is not None and arena_ui.following:
            arena_ui.update_follow_camera()
        
        if viser_viewer is not None:
            viser_viewer.update_from_simulator(env, [robot_index])
            viser_viewer.update_live_telemetry(
                env.simulator.base_lin_vel[robot_index].detach().cpu().numpy(),
                env.simulator.base_ang_vel[robot_index, 2].item(),
                env.commands[robot_index, :3].detach().cpu().numpy(),
            )

        # Native on-screen HUD.  Cheap to call every frame -- NativeArenaUI
        # throttles the string rebuild itself -- and a no-op when the viewer is
        # headless or the caption layer is unavailable.  The command shown is
        # the one actually applied to the env, so the envelope ramp is visible.
        if arena_ui is not None and arena_ui.hud_due():
            arena_ui.update_hud(
                command=env.commands[robot_index, :3].detach().cpu().tolist(),
                achieved=(env.simulator.base_lin_vel[robot_index, 0].item(),
                          env.simulator.base_lin_vel[robot_index, 1].item(),
                          env.simulator.base_ang_vel[robot_index, 2].item()),
                fps=fps_ema,
                realtime=realtime_ema,
            )

        print_debug_info(env, robot_index)

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
    elif task_type == "moects":
        # latent-first actor concat -> its own exporter (see PolicyExporterMoECTS)
        exporter = PolicyExporterMoECTS(alg_runner.alg.actor_critic)
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


@contextmanager
def skip_rollout_storage(train_cfg):
    """Build the runner WITHOUT allocating PPO rollout storage.

    Playback needs exactly two things from the runner: the loaded weights and
    ``get_inference_policy()`` (which returns a bound method of
    ``alg.actor_critic``).  The rollout storage is a pure *training* artifact --
    ``num_steps_per_env x num_envs`` transition buffers that play never writes
    and never reads -- so allocating it is wasted memory at best.

    At worst it is fatal.  MoE-CTS splits the env batch into interleaved
    teacher/student roles at ``init_storage`` time
    (``rsl_rl/algorithms/ppo_moe_cts.py:compute_role_env_idxs``): every 4th env
    is a student, the rest teachers.  That split has no solution for a SINGLE
    env -- index 0 is always a student, so there are 0 teachers, while the
    guard wants ``max(int(1 * 0.75), 1) = 1`` -- and ``WtyCurriculumMixin``
    independently rescales ``num_teacher`` to ``int(1 * 0.75) = 0``.  num_envs=1
    is in fact the *only* count that fails; 2 and up all resolve.  Since the
    arena deliberately drives one robot, play would otherwise crash before the
    viewer ever opened.  Rather than inflate the exhibit to two robots or
    weaken a training-side contract, play simply does not build the training
    buffer it has no use for.  Nothing about the policy, the observations or
    the physics changes.

    Scoped to the runner construction call and restored afterwards, so nothing
    a training entry point does is affected.
    """
    from rsl_rl.utils.runner_registry import runner_registry

    try:
        runner_class = runner_registry.get_runner_class(train_cfg.runner_class_name)
    except Exception as exc:
        print(f"[play] rollout-storage skip unavailable ({exc}); allocating it")
        yield
        return
    had_own = "_init_storage" in vars(runner_class)
    original = vars(runner_class).get("_init_storage")
    runner_class._init_storage = lambda self: None
    try:
        yield
    finally:
        if had_own:
            runner_class._init_storage = original
        else:
            # It was inherited; drop the override so the MRO takes over again.
            delattr(runner_class, "_init_storage")


def _play_inference_policy(policy):
    """Run deployment policy without constructing autograd graphs."""
    def infer(*inputs, **kwargs):
        with torch.inference_mode():
            return policy(*inputs, **kwargs)
    return infer


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
        # Must come after gs.init: that is what installs the mode (see the
        # helper's docstring -- this is the single biggest play-time win).
        _drop_torch_default_device_mode()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    splitted = args.task.split("_")
    # by default, the first part of the task name is robot name, and the second part is task type, e.g. go2_ts, go2_cat, go2_ee, go2_cts, go2_dreamwaq, go2_ts_depth
    # concatenate the parts after the first part to get the task type, e.g. ts, cat, ee, cts, dreamwaq, ts_depth
    task_type = "_".join(splitted[1:])
    print("Task type: ", task_type)
    override_configs(env_cfg, args, task_type)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not getattr(args, "play_rewards", False):
        _disable_play_rewards(env)
    # load policy
    train_cfg.runner.resume = True
    # P5's selected July checkpoint has a 50-D critic whereas current code
    # builds a 53-D critic.  The 50-D actor is unchanged and is all play needs.
    actor_only_playback = args.task == 'go2_bench_oracle_id'
    if actor_only_playback:
        train_cfg.runner.resume = False
    # Play builds its own env (few envs, a user-chosen --terrain), so the
    # checkpoint's per-env terrain levels describe a geometry that does not
    # exist here; restoring them is impossible and the strict resume protocol
    # would hard-fail. Playback uses the play terrain's own levels.
    # Play never rolls out into PPO storage; skipping it also decouples the
    # arena's one-robot default from the training-only env-count contracts
    # (see skip_rollout_storage).
    with skip_rollout_storage(train_cfg):
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg,
                                                              load_env_curriculum=False)
    if actor_only_playback:
        load_oracle_id_actor_for_playback(ppo_runner, train_cfg)
    policy = _play_inference_policy(ppo_runner.get_inference_policy(device=env.device))
    
    # export policy as a jit module (used to run it from C++ or python)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 
                            train_cfg.runner.load_run, 'exported')
    export_policy(ppo_runner, path, args, env_cfg, train_cfg, task_type)
    
    viser_viewer = None
    input_source = None
    arena_ui = None
    if args.viewer == 'native':
        # Native path: real key-down/key-up from Genesis + optional gamepad.
        input_source = build_input_source()
        # Hot-swap candidates are the sibling checkpoints of the one that was
        # loaded -- same task, so a state-dict reload is enough (see model_swap).
        swapper = None
        try:
            loaded_ckpt = get_load_path(
                os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                             train_cfg.runner.experiment_name),
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            swapper = CheckpointSwapper(
                ppo_runner, os.path.dirname(loaded_ckpt),
                current=os.path.basename(loaded_ckpt),
            )
        except Exception as exc:
            print(f"[play] checkpoint hot-swap disabled: {exc}")
        arena_ui = NativeArenaUI(
            env,
            input_source,
            robot_index=0,
            model_swap_cb=(
                swapper.step if (swapper is not None and swapper.available) else None
            ),
        )
        arena_ui.print_legend()

    if args.viewer == 'viser':
        viser_viewer = create_viser_viewer(env, port=args.viser_port)
        robot_index = 0  # which robot the viewer tracks / respawns, matches interaction_loop
        # Respawn/Teleport are drained on the main thread in interaction_loop
        # (see take_pending_respawn / take_pending_taxonomy_spawn).  Do not
        # call env.reset_idx from Viser GUI worker callbacks.
        if (
            is_taxonomy_terrain_cfg(env_cfg.terrain)
            or is_v6_showcase_terrain_cfg(env_cfg.terrain)
            or getattr(env_cfg.terrain, "moe_showcase", False)
            or getattr(args, "terrain", None) in {
                "taxonomy", "showcase", "v6", "v6_frontier", "frontier", "v6_full", "moe",
            }
        ):
            # Panel may already exist from create_viser_viewer; ensure it exists.
            if not hasattr(viser_viewer, "_taxonomy_spawn_gui"):
                level_names = None
                panel_title = "Spawn (taxonomy)"
                if getattr(env_cfg.terrain, "moe_showcase", False):
                    # The moe grid's rows are a sparse pick of training levels,
                    # so label the dropdown with the difficulty each row really
                    # carries (L0/L2/...) while still queueing the row index.
                    levels = tuple(getattr(env_cfg.terrain, "moe_showcase_levels",
                                           MOE_SHOWCASE_LEVELS))
                    type_names = tuple(
                        name for name, _ in moe_showcase_columns(
                            env_cfg.terrain.terrain_proportions))
                    level_names = tuple(f"L{lvl}" for lvl in levels)
                    panel_title = "Spawn (moe terrain)"
                elif is_v6_showcase_terrain_cfg(env_cfg.terrain) or getattr(
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
                    level_names=level_names,
                    title=panel_title,
                )
        print(f"Viser web viewer started at http://localhost:{args.viser_port}")
    
    try:
        interaction_loop(
            env, policy, args, task_type,
            viser_viewer=viser_viewer,
            input_source=input_source,
            arena_ui=arena_ui,
        )
    finally:
        if arena_ui is not None:
            arena_ui.unregister()
        if input_source is not None and input_source.gamepad is not None:
            input_source.gamepad.close()

    if viser_viewer is not None:
        viser_viewer.stop()
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
