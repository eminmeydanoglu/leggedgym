"""V5: UED episode-task sampler arms on the frozen V4 locomotion substrate.

See ``lpacr/solun_plani.md`` (esp. §2, §3, §4, §14) and
``lpacr/subplans/05_arms_config_and_fazB.md`` for the source-of-truth design.
``lpacr/faz0_freeze.md`` collects every frozen number defined here in one place.

One task family, one flag.  Four arms (``curriculum.algorithm``) share an
IDENTICAL V4 physics/reward/DR/actor-critic/PPO/budget contract; only the
episode-task distribution mechanism differs:

  * ``handcrafted_v4``: unmodified V4 terrain promotion + command schedule
    (the real-world baseline).  No UED teacher is installed.
  * ``uniform`` / ``lp_acrl`` / ``alp``: an ``EpisodeCurriculum`` teacher owns
    the command/terrain distribution instead; the legacy terrain-promotion and
    command-schedule/curriculum mechanisms are switched off (§14.5).

``build_ued_teacher(env_cfg)`` is the single factory used by
``legged_gym/scripts/train.py`` to construct the (curriculum, task_space) pair
implied by a cfg's ``curriculum.algorithm`` (``None, None`` for
``handcrafted_v4``, which needs no teacher).
"""
from __future__ import annotations

from legged_gym.envs.base.common_cfgs import Go2BenchmarkV4TerrainCfg, get_simulator_suffix
from legged_gym.envs.go2.go2_v3_mlp.go2_v3_mlp_config import Go2V3CfgPPO
from legged_gym.utils.terrain import TAXONOMY_NUM_LEVELS, TAXONOMY_NUM_TYPES


# ---------------------------------------------------------------------------
# Faz 0 freeze (lpacr/solun_plani.md §11 Faz 0 / §14) -- see lpacr/faz0_freeze.md
# for the full rationale behind every number below.
# ---------------------------------------------------------------------------

V5_ALGORITHMS = ("handcrafted_v4", "uniform", "lp_acrl", "alp")

# Standing-exposure mixture rho (solun_plani.md §4 "Standstill kontratı"):
# the explicit budget for the reserved standstill bucket.  Under UED each env
# is drawn standstill with probability rho and an LP moving task otherwise, so
# rho is a plain budget line, not a hidden curriculum-contamination rate.
# Constant across every arm so standing exposure is not an experiment variable;
# lower than the legacy base default (0.4) so LP cells stay denser.
V5_STANDSTILL_RHO = 0.12

# Fixed global_control_steps stage length for the episode-task sampler (§5,
# §11 Faz 0).  common_step_counter advances once per simulated control tick
# (shared by all 4096 envs), so at num_steps_per_env=24 this is ~83 PPO
# iterations/stage (~36 stages across the 3000-iteration budget).  Each stage
# yields on the order of (2000 steps * 4096 envs / ~1000-step episode) ~= 8192
# completed episodes spread over the 84 (+1 standstill) cells before the LP
# update fires -- enough for a per-cell mean-return estimate.
V5_STAGE_LENGTH_CONTROL_STEPS = 2000

# Fixed LP temperature.  Frozen after the LP-ACRL curriculum flattened to
# ~uniform sampling under adaptive_ess: offline replay on the real logged LP
# data measured a steady-state median |LP| ~= 0.5, so beta = 1.0 makes
# LP/beta ~= 1 and the Eq. 7 softmax actually bites (beta = 5 gave LP/beta ~=
# 0.1 => effectively uniform; the adaptive_ess confidence gate flattened
# everything further).  Chosen so LP/beta ~= 1 at the measured steady-state
# |LP|.  Adaptive mode treats this as its starting value, but is no longer the
# selected mode.
V5_BETA = 1.0

# Small uniform exploration floor: a plain observability guarantee that every
# cell keeps a little sampling mass, not an adaptive coverage controller (that
# belongs to the ESS mode, which is no longer selected).
V5_EPSILON = 0.02
V5_TEMPERATURE_MODE = "fixed"
V5_BETA_MIN = 0.75
V5_BETA_MAX = 8.0
V5_BETA_EMA = 0.8
V5_TARGET_ESS_RATIO_MIN = 0.5
# Per-cell ceiling, enforced in BOTH temperature modes.  It binds: the first
# beta = 1.0 run collapsed to ESS ~1.3 with a single cell at p = 0.90 (see the
# episode gate in _FiniteEpisodeCurriculum.advance), so 0.25 is what keeps at
# least four cells sharing the top of the distribution.
V5_MAX_CELL_PROBABILITY = 0.25
# Minimum episodes in BOTH consecutive stages before a cell's LP is believed.
# Below this the LP is a difference of one- or two-episode means: |LP| ~ 6-9
# against a steady-state signal of ~0.5.  Ungated, e^{noise} at beta = 1 hands
# the distribution to whichever cell is currently mismeasured worst.
V5_MIN_STAGE_EPISODES_FOR_LP = 16
V5_CONFIDENCE_SCALE = 1.0

# Deterministic UED training-grid geometry seed (terrain.ued_training_seed).
# Shared by all three UED-teacher arms so their static 4x6 taxonomy grid is
# byte-identical (solun_plani.md §14.6, "Geometry hash ... bütün kollarda
# aynı" -- for the UED-teacher arms; handcrafted_v4 intentionally keeps the
# legacy V4 terrain builder instead, see the allowlist in
# tests/test_v5_training_contract.py).
V5_TERRAIN_SEED = 0

# Forward (vx) command-support ceiling shared by all four arms: [0, 2] m/s
# (SPNTE lin v_scale=2.0, solun_plani.md §2/§10/§14.1).  This schedule only
# ramps the vx support; vy / omega_z are full-support 3-axis nuisance from the
# start (see Go2V5CommonCfg.commands.ranges).  handcrafted_v4 reaches the vx
# ceiling via this iteration-based runner schedule; the other arms reach it
# directly through the episode-task sampler's vx bins.
V5_COMMAND_SCHEDULE = [
    {"start_iteration": 0, "lin_vel_x": [0.0, 1.0]},
    {"start_iteration": 500, "lin_vel_x": [0.0, 2.0]},
]


# ---------------------------------------------------------------------------
# Env cfgs
# ---------------------------------------------------------------------------

class Go2V5CommonCfg(Go2BenchmarkV4TerrainCfg):
    """Frozen V5 substrate (§2), shared by all four arms.

    Physics (V3 contract), reward (V4 rough-terrain set), domain
    randomization, heightfield terrain generation, actor/critic contract and
    budget are inherited unchanged from ``Go2BenchmarkV4TerrainCfg``.  Only
    the command support ceiling and the (per-arm) UED sampler section are
    added here; everything else an arm overrides is documented on that arm.
    """

    class env(Go2BenchmarkV4TerrainCfg.env):
        num_observations = 45
        num_privileged_obs = 48  # clean proprio + true base velocity (no height map)
        num_actions = 12
        ued_enabled = False  # only the three UED-teacher arms flip this on

    class commands(Go2BenchmarkV4TerrainCfg.commands):
        # Shared standstill mixture for all four arms (handcrafted + UED).
        zero_cmd_prob = V5_STANDSTILL_RHO

        # 3-axis velocity command (vx, vy, ang_vel_yaw).  We command angular
        # VELOCITY omega_z directly (heading_command=False) rather than a
        # world-frame heading pose: omega_z is what the gyro measures directly,
        # so it survives sim-to-real without an absolute (drift-prone) yaw
        # estimate, and it matches the velocity-command convention of the
        # LP-ACRL reference (see docstring, solun_plani.md §2).  The vx bin is
        # the UED task axis; vy and omega_z ride along as within-cell nuisance,
        # sampled uniformly in training and frozen per-replica in the offline
        # validation bank so training and eval share one command distribution.
        heading_command = False

        class ranges(Go2BenchmarkV4TerrainCfg.commands.ranges):
            # Shared [0, 2] m/s forward support for every arm (§2, §14.1):
            # SPNTE's lin v_scale is derived from this field, so it must be
            # identical everywhere for the arms to be comparable.
            lin_vel_x = [0.0, 2.0]
            # Shared lateral / yaw-rate nuisance support (identical across arms
            # and mirrored by the eval bank).  Symmetric so both turn/strafe
            # directions are covered; |bound| sets each axis' SPNTE scale.
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]

    class curriculum:
        """UED episode-task sampler knobs (§4/§5, Faz 0 freeze).

        Meaningless (but harmless) for ``handcrafted_v4``, which never builds
        a teacher.  ``seed=None`` means "derive from the resolved training
        seed" (see ``build_ued_teacher``), so a paired ``--seed`` CLI run
        gets a reproducible-but-distinct teacher RNG without a second flag.
        """
        algorithm: str = "handcrafted_v4"
        stage_length_control_steps: int = V5_STAGE_LENGTH_CONTROL_STEPS
        beta: float = V5_BETA
        epsilon: float = V5_EPSILON
        temperature_mode: str = V5_TEMPERATURE_MODE
        beta_min: float = V5_BETA_MIN
        beta_max: float = V5_BETA_MAX
        beta_ema: float = V5_BETA_EMA
        target_ess_ratio_min: float = V5_TARGET_ESS_RATIO_MIN
        max_cell_probability: float = V5_MAX_CELL_PROBABILITY
        min_stage_episodes_for_lp: int = V5_MIN_STAGE_EPISODES_FOR_LP
        confidence_scale: float = V5_CONFIDENCE_SCALE
        seed: int | None = None


class Go2V5HandcraftedCfg(Go2V5CommonCfg):
    """``handcrafted_v4``: the literal, unmodified V4 baseline (§3).

    Terrain promotion (``terrain.curriculum=True`` on the standard 10x10
    grid) and the legacy batch-wide standstill draw are inherited unchanged
    from ``Go2BenchmarkV4TerrainCfg`` / ``Go2V5CommonCfg``; the runner-level
    command schedule below is this arm's only active distribution mechanism.
    """

    class curriculum(Go2V5CommonCfg.curriculum):
        algorithm = "handcrafted_v4"


class _Go2V5UedArmCfg(Go2V5CommonCfg):
    """Shared substrate for the three UED-teacher arms (uniform/lp_acrl/alp).

    The episode-task sampler becomes the single owner of the command/terrain
    distribution (§14.5): legacy terrain promotion and command curriculum are
    switched off, and standstill becomes a reserved mixture bucket drawn per-env
    beside the sampler's task space (§4 "Standstill kontratı", §14.4).
    """

    class env(Go2V5CommonCfg.env):
        ued_enabled = True

    class terrain(Go2V5CommonCfg.terrain):
        # Deterministic 4-level x 6-type teleport grid (§14.6): the sampler
        # teleports envs onto it every reset: no distance-based promotion.
        curriculum = False
        num_rows = TAXONOMY_NUM_LEVELS
        num_cols = TAXONOMY_NUM_TYPES
        ued_training_grid = True
        ued_training_seed = V5_TERRAIN_SEED

    class commands(Go2V5CommonCfg.commands):
        # The sampler is the single owner of the command/terrain distribution
        # for these arms; the legacy performance-based widening never fires.
        # zero_cmd_prob inherits V5_STANDSTILL_RHO from Go2V5CommonCfg and is the
        # reserved standstill budget.  per_env_standstill is kept True for
        # continuity but the UED reserve draw is always per-env regardless.
        legacy_performance_command_curriculum_enabled = False
        per_env_standstill = True


class Go2V5UniformCfg(_Go2V5UedArmCfg):
    """``uniform``: equal probability across moving tasks (curriculum-off control)."""

    class curriculum(_Go2V5UedArmCfg.curriculum):
        algorithm = "uniform"


class Go2V5LPACRLCfg(_Go2V5UedArmCfg):
    """``lp_acrl``: signed learning-progress softmax (the main method)."""

    class curriculum(_Go2V5UedArmCfg.curriculum):
        algorithm = "lp_acrl"


class Go2V5ALPCfg(_Go2V5UedArmCfg):
    """``alp``: absolute learning-progress softmax (the signed-LP ablation)."""

    class curriculum(_Go2V5UedArmCfg.curriculum):
        algorithm = "alp"


# ---------------------------------------------------------------------------
# PPO / runner cfgs -- identical PPO hyperparameters and budget across arms;
# only experiment naming and the (allowlisted) command_schedule differ.
# ---------------------------------------------------------------------------

class Go2V5CfgPPO(Go2V3CfgPPO):
    """Shared V5 PPO/runner protocol (identical HP/budget across all arms)."""

    class runner(Go2V3CfgPPO.runner):
        experiment_name = "go2_v5_ued"
        # V3's three-stage schedule (incl. its 1500-iter re-widen to [-1, 2])
        # does not apply to V5: only handcrafted_v4 overrides this, with the
        # two-stage vx-support schedule above.
        command_schedule = None
        # V3 inherits online in-distribution eval (eval_interval=200) that runs
        # on the *same live training env*. Under UED that path observes partial
        # episodes into the teacher, advances common_step_counter, and mutates
        # sampler RNG / assignments -- so it trains the curriculum, not just
        # measures it. 0 disables OnPolicyRunner._run_eval entirely. Checkpoint
        # selection for V5 is offline (SPNTE / UED validation), not best_tracking.
        eval_interval = 0


class Go2V5HandcraftedCfgPPO(Go2V5CfgPPO):
    class runner(Go2V5CfgPPO.runner):
        run_name = "v5_handcrafted_v4" + get_simulator_suffix()
        command_schedule = V5_COMMAND_SCHEDULE


class Go2V5UniformCfgPPO(Go2V5CfgPPO):
    class runner(Go2V5CfgPPO.runner):
        run_name = "v5_uniform" + get_simulator_suffix()


class Go2V5LPACRLCfgPPO(Go2V5CfgPPO):
    class runner(Go2V5CfgPPO.runner):
        run_name = "v5_lp_acrl" + get_simulator_suffix()


class Go2V5ALPCfgPPO(Go2V5CfgPPO):
    class runner(Go2V5CfgPPO.runner):
        run_name = "v5_alp" + get_simulator_suffix()


# ---------------------------------------------------------------------------
# Teacher factory (Topic 05 deliverable): the only place that turns a cfg's
# ``curriculum.algorithm`` into a (curriculum, task_space) pair.
# ``legged_gym/scripts/train.py`` calls this and then ``env.enable_ued(...)``.
# ---------------------------------------------------------------------------

def build_ued_teacher(env_cfg):
    """Build the (episode_curriculum, task_space) pair implied by ``env_cfg``.

    Returns ``(None, None)`` for ``handcrafted_v4``, which installs no UED
    teacher at all (``env_cfg.env.ued_enabled`` stays ``False`` for that arm,
    so ``env.enable_ued`` must never be called for it either).
    """
    algorithm = getattr(env_cfg.curriculum, "algorithm", "handcrafted_v4")
    if algorithm not in V5_ALGORITHMS:
        raise ValueError(
            f"unknown V5 curriculum.algorithm={algorithm!r}; expected one of {V5_ALGORITHMS}"
        )
    if algorithm == "handcrafted_v4":
        return None, None

    # Local imports: keep this factory (and therefore train.py, which only
    # imports it when ued_enabled) from pulling numpy/UED-teacher modules into
    # every non-UED task's import graph.
    from legged_gym.utils.terrain import ued_training_builder_parameters
    from legged_gym.utils.ued.task_space import TaskSpace
    from legged_gym.utils.ued.episode_curriculum import (
        ALPEpisodeCurriculum,
        LPACRLEpisodeCurriculum,
        UniformEpisodeCurriculum,
    )

    curriculum_classes = {
        "uniform": UniformEpisodeCurriculum,
        "lp_acrl": LPACRLEpisodeCurriculum,
        "alp": ALPEpisodeCurriculum,
    }
    task_space = TaskSpace(
        builder_parameters=ued_training_builder_parameters(env_cfg.terrain)
    )
    seed = env_cfg.curriculum.seed
    if seed is None:
        # Paired-seed default: derive the teacher RNG from the resolved
        # training seed (env_cfg.seed, set by --seed / task_registry) so a
        # `--seed N` run is reproducible without a second CLI flag.
        seed = int(getattr(env_cfg, "seed", 0) or 0)
    curriculum = curriculum_classes[algorithm](
        task_space,
        stage_length_control_steps=int(env_cfg.curriculum.stage_length_control_steps),
        beta=float(env_cfg.curriculum.beta),
        epsilon=float(getattr(env_cfg.curriculum, "epsilon", 0.0)),
        temperature_mode=str(getattr(env_cfg.curriculum, "temperature_mode", "fixed")),
        beta_min=float(getattr(env_cfg.curriculum, "beta_min", 0.75)),
        beta_max=float(getattr(env_cfg.curriculum, "beta_max", 8.0)),
        beta_ema=float(getattr(env_cfg.curriculum, "beta_ema", 0.8)),
        target_ess_ratio_min=float(getattr(env_cfg.curriculum, "target_ess_ratio_min", 0.5)),
        max_cell_probability=float(getattr(env_cfg.curriculum, "max_cell_probability", 0.08)),
        min_stage_episodes_for_lp=getattr(env_cfg.curriculum, "min_stage_episodes_for_lp", 16),
        confidence_scale=float(getattr(env_cfg.curriculum, "confidence_scale", 1.0)),
        seed=seed,
    )
    return curriculum, task_space
