"""Brake probe: high-speed hard-stop, recorded identically on Genesis and MuJoCo.

WHY THIS EXISTS
---------------
Sim-to-sim play showed the rear legs lifting noticeably after a fast run is
stopped in MuJoCo, but not (to the eye) in Genesis.  "To the eye" is not a
measurement, and the obvious metric -- peak foot height -- is actively
misleading here: a single nominal smoke had a HIGHER peak foot height in
Genesis while the event we actually care about (both rear feet leaving the
ground TOGETHER) was twice as frequent in MuJoCo.  So this script does not try
to score anything.  It is a RECORDER: it drives one deterministic braking
manoeuvre over a (speed x stop-ramp x gait-phase) grid on whichever backend is
selected and dumps the raw per-step time series.  Every metric is derived
offline by ``brake_analyze.py``, so a metric can be redefined without paying
for the rollouts again.

THE MANOEUVRE
-------------
Per condition: settle at a constant forward command ``vx`` (unrecorded warmup),
record ``pre_steps`` of steady cruise, then at a per-env trigger step ramp the
forward command to zero over ``ramp`` seconds (``ramp = 0`` is an instantaneous
hard stop) and keep recording for ``post_steps``.  Lateral and yaw commands are
held at zero the whole time: this reproduces the interactive "run fast, release
the stick" case, which -- as the training config shows -- is a *sparse corner*
of the training distribution (zero-command draws zero x/y but usually leave yaw
nonzero).

GAIT PHASE IS THE CONFOUNDER, SO IT IS SWEPT
--------------------------------------------
With domain randomization off and observation noise off, the pre-trigger
trajectory is deterministic and IDENTICAL for every env in a batch.  The only
thing that differs is WHEN the stop fires, so a trigger offset of k control
steps is exactly a gait-phase offset of k steps.  That is the whole point of
``--phase_span``: braking mid-flight and braking at touchdown are different
mechanical problems, and averaging them together is how you miss the effect.

Genesis and MuJoCo cannot be phase-matched against each other (they diverge
dynamically from the first substep -- different contact solvers), so the two
backends are compared as DISTRIBUTIONS over the phase sweep, never trajectory
against trajectory.

BACKEND BATCHING
----------------
MuJoCo has one ``MjModel``/``MjData`` pair, i.e. ``num_envs == 1`` hard
(``mujoco_simulator.py``), so its conditions run serially in one process with a
reset between them; determinism makes the repeated pre-trigger stretch
identical across those rollouts.  Genesis runs the whole grid as one batched
rollout, one env per condition.  Both paths execute the same
``run_window`` code.

MODES
-----
``--arm student|teacher``  Which head drives.  ``act_student`` is the deployed
    policy (obs + history); ``act_teacher`` reads the privileged vector.  If
    the teacher brakes cleanly in MuJoCo and the student does not, the problem
    is the history/distillation contract, not the actor or the physics.

``--replay <npz>``  Open-loop: ignore the policy and feed the action sequence
    recorded on the *other* backend.  If the rear-leg lift survives with the
    control loop cut, the difference is plant/contact, not the policy having
    memorized one simulator.

Example (grid is 3 speeds x 3 ramps x 16 phases = 144 conditions)::

    python legged_gym/scripts/eval/brake_probe.py --sim genesis \
        --load_run Aug03_12-01-45_moe_cts_genesis --ckpt 20500 \
        --out logs/eval/brake/genesis_student.npz

    python legged_gym/scripts/eval/brake_probe.py --sim mujoco \
        --load_run Aug03_12-01-45_moe_cts_genesis --ckpt 20500 \
        --out logs/eval/brake/mujoco_student.npz
"""

import os
import sys


def _bootstrap_simulator_from_argv() -> None:
    """Translate ``--sim <backend>`` into ``os.environ["SIMULATOR"]``.

    Verbatim from ``legged_gym/scripts/play.py``: ``legged_gym`` freezes
    SIMULATOR at import time, so this has to run before the first
    ``from legged_gym import ...`` below.
    """
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--sim="):
            os.environ["SIMULATOR"] = arg.split("=", 1)[1]
            return
        if arg == "--sim" and i + 1 < len(sys.argv):
            os.environ["SIMULATOR"] = sys.argv[i + 1]
            return


_bootstrap_simulator_from_argv()

import argparse
import itertools
import json
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from legged_gym import *          # noqa: F401,F403  (gs, SIMULATOR, LEGGED_GYM_ROOT_DIR)
from legged_gym.envs import *     # noqa: F401,F403  (registers tasks)
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_load_path
from legged_gym.scripts.eval.ckpt_utils import parse_ckpt_cli


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Sim-to-sim hard-stop (brake) recorder")
    p.add_argument("--sim", type=str, default=None,
                   help="backend: genesis | mujoco (consumed before imports)")
    p.add_argument("--task", type=str, default="go2_moects")
    p.add_argument("--load_run", type=str, required=True,
                   help="run dir under logs/<experiment_name>, e.g. Aug03_12-01-45_moe_cts_genesis")
    p.add_argument("--ckpt", type=parse_ckpt_cli, default=-1,
                   help="checkpoint: 'best', 'latest'/-1, or iteration int (e.g. 20500)")
    p.add_argument("--arm", type=str, default="student", choices=["student", "teacher"],
                   help="which head drives: deployed student (obs+history) or privileged teacher")

    # --- the grid ---
    p.add_argument("--vx", type=float, nargs="+", default=[1.0, 1.5, 2.0],
                   help="cruise forward commands [m/s] (training range on flat is [-2, 2])")
    p.add_argument("--ramp", type=float, nargs="+", default=[0.0, 0.3, 0.6],
                   help="stop ramp durations [s]; 0 = instantaneous hard stop")
    p.add_argument("--phases", type=int, default=16,
                   help="gait phases sampled per (vx, ramp) cell")
    p.add_argument("--phase_span", type=int, default=20,
                   help="control steps the phase sweep spans; ~one gait period "
                        "(trot at ~2.5 Hz over dt=0.02 s is 20 steps)")

    # --- the window ---
    p.add_argument("--warmup", type=int, default=150,
                   help="unrecorded settling steps at the cruise command")
    p.add_argument("--pre_steps", type=int, default=30,
                   help="recorded steady-cruise steps before the earliest trigger (baseline)")
    p.add_argument("--post_steps", type=int, default=100,
                   help="recorded steps after each env's own trigger")

    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--randomize_start", action="store_true", default=False,
                   help="keep legged_robot's randomized spawn (joints +-0.2 rad, base "
                        "vel +-0.5); off by default so a trigger offset means a gait "
                        "phase and not a different rollout -- see make_reset_deterministic")
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--out", type=str, required=True, help="output .npz path")
    p.add_argument("--replay", type=str, default=None,
                   help="open-loop: npz from another run whose recorded actions are "
                        "replayed instead of querying the policy (grid must match)")
    p.add_argument("--replay_from", type=str, default="trigger",
                   choices=["trigger", "window"],
                   help="when the control loop is cut. 'trigger' (default) keeps the "
                        "policy closed-loop through the cruise and goes open-loop only "
                        "from each env's own stop trigger, so the open-loop horizon is "
                        "just the brake response; 'window' replays the whole recorded "
                        "window, which on a legged robot diverges long before the stop "
                        "and answers nothing")
    p.add_argument("--replay_sync_state", action="store_true", default=False,
                   help="at each env's trigger, overwrite this sim's state with the "
                        "replay source's recorded state, so the open-loop comparison is "
                        "same-state + same-torques + different-plant (see inject_state). "
                        "Requires --replay_from trigger")
    p.add_argument("--max_conditions", type=int, default=None,
                   help="debug: truncate the grid to the first N conditions")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Runner construction
# --------------------------------------------------------------------------- #
@contextmanager
def skip_rollout_storage(train_cfg):
    """Build the runner WITHOUT allocating PPO rollout storage.

    Copied from ``play.py`` (same reasoning, and deliberately not imported --
    importing play.py drags in the viewer stack and re-runs its argv bootstrap).
    Beyond the wasted memory, MoE-CTS splits the env batch into interleaved
    teacher/student roles at ``init_storage`` time, and ``num_envs == 1`` (the
    MuJoCo case) is the one count that split has no solution for.  This probe
    never writes rollout storage, so it simply does not build it.
    """
    from rsl_rl.utils.runner_registry import runner_registry

    try:
        runner_class = runner_registry.get_runner_class(train_cfg.runner_class_name)
    except Exception as exc:
        print(f"[brake] rollout-storage skip unavailable ({exc}); allocating it")
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
            delattr(runner_class, "_init_storage")


def freeze_cfg_for_brake(env_cfg, cli, num_envs):
    """Nominal physics, flat ground, no noise, externally driven commands.

    Three separate freezes, each for its own reason:

    * **Domain randomization off.**  Sim-to-sim asks "do the two plants disagree
      at the nominal point"; leaving DR on would answer a different question
      (and desynchronise the two backends' RNG streams on top of it).  Cleared
      the same way ``play.py::_disable_play_domain_rand`` does -- every
      ``randomize_*`` bool plus the pushers -- so nothing here invents physics
      values, it only stops the per-env sampling.
    * **Observation noise off.**  Determinism is what makes a trigger offset
      mean "gait phase" instead of "different rollout".
    * **Flat plane.**  The braking question is about contact impulse and pitch
      inertia, not terrain.  Recipe matches ``play.py``'s ``terrain_mode ==
      "flat"`` branch exactly, including clearing ``moe_grid`` /
      ``ued_training_grid`` so ``Terrain.__init__`` does not keep building the
      training map.
    """
    env_cfg.env.num_envs = num_envs
    env_cfg.env.auto_reset = True     # a fall inside the window must be observable
    env_cfg.env.debug = False
    env_cfg.seed = cli.seed

    dr = env_cfg.domain_rand
    for name in dir(dr):
        if name.startswith("randomize_") and isinstance(getattr(dr, name), bool):
            setattr(dr, name, False)
    for name in ("push_robots", "push_links"):
        if hasattr(dr, name):
            setattr(dr, name, False)

    env_cfg.noise.add_noise = False

    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.terrain_kwargs = None
    env_cfg.terrain.ued_training_grid = False
    env_cfg.terrain.moe_grid = False
    if hasattr(env_cfg.terrain, "moe_showcase"):
        env_cfg.terrain.moe_showcase = False

    # Commands are written by hand every step.  Killing curriculum/heading and
    # pushing the resample clock past the window means ``_resample_commands``
    # never gets a nonempty env set and so can never overwrite the schedule.
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.resampling_time = 1e6
    if hasattr(env_cfg.commands, "dynamic_resample_commands"):
        env_cfg.commands.dynamic_resample_commands = False
    # Collapse the ranges onto the point we actually hold, so that even if some
    # code path did resample, it would resample the same thing.
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    if hasattr(env_cfg.commands.ranges, "heading"):
        env_cfg.commands.ranges.heading = [0.0, 0.0]


def make_reset_deterministic(env):
    """Replace the randomized reset draws with the nominal spawn state.

    ``legged_robot._reset_dofs`` / ``_reset_root_states`` are not config-gated:
    every reset jitters the joints by +-0.2 rad and the base linear/angular
    velocity by +-0.5 m/s and rad/s.  Measured, that jitter dominates: two envs
    in the same batch differed by up to 0.5 m/s well before either trigger.
    That would silently turn "trigger offset k" from *a controlled gait phase*
    into *a different random rollout*, which is precisely the confound this
    probe is built to remove -- and it would also give the two backends
    different starting states, so any measured gap would be partly initial
    conditions rather than plant.

    The overrides are bound on the INSTANCE, so ``reset_idx``'s
    ``self._reset_dofs(...)`` picks them up while the class stays untouched for
    every other caller.  Nothing here invents physics: the values written are
    the asset's own defaults (``default_dof_pos``, ``base_init_pos/quat``, zero
    velocity).
    """
    sim = env.simulator
    device = env.device
    default_dof = sim.default_dof_pos.reshape(1, -1).to(device)

    def _reset_dofs(env_ids):
        n = len(env_ids)
        dof_pos = default_dof.repeat(n, 1).clone()
        sim.reset_dofs(env_ids, dof_pos, torch.zeros_like(dof_pos))

    def _reset_root_states(env_ids):
        n = len(env_ids)
        base_pos = sim.base_init_pos.reshape(1, -1).repeat(n, 1).clone()
        base_pos += sim.env_origins[env_ids]
        base_quat = sim.base_init_quat.reshape(1, -1).repeat(n, 1).clone()
        zeros = torch.zeros((n, 3), dtype=torch.float, device=device)
        sim.reset_root_states(env_ids, base_pos, base_quat, zeros, zeros.clone())

    env._reset_dofs = _reset_dofs
    env._reset_root_states = _reset_root_states

    # The other reset-time draw: ``reset_idx`` resamples a RANDOM command, and
    # although this probe overwrites ``env.commands`` before every step, the
    # observation returned by ``reset()`` is computed BEFORE the first
    # overwrite -- so the very first action of each rollout would be taken
    # against a random command in [-2, 2].  In a gait that is a large kick, and
    # it is the one that survived the deterministic spawn fix above.  Pinning
    # the resampler to the cruise command (``env._brake_cruise``, written by
    # ``run_window`` before ``reset()``) removes the draw at its source rather
    # than patching the observation after the fact.
    def _resample_commands(env_ids):
        env.commands[env_ids, 0] = env._brake_cruise[env_ids]
        env.commands[env_ids, 1:] = 0.0

    env._resample_commands = _resample_commands


def make_registry_args(cli):
    """The arg namespace ``task_registry`` expects (mirrors ``helpers.get_args``)."""
    return SimpleNamespace(
        task=cli.task, headless=True, cpu=cli.cpu, num_envs=None, max_iterations=None,
        # resume=False so ``update_cfg_from_args`` leaves ``train_cfg.runner.resume``
        # alone (it only ever forces it True); the checkpoint is loaded by hand in
        # ``build_env_and_arms`` -- see the comment there.
        resume=False, sync_wandb=False, export_onnx=False, debug=False,
        load_run=cli.load_run, ckpt=cli.ckpt, use_joystick=False, joystick_type="xbox",
        follow_robot=False, viewer="native", viser_port=8080, motion_file=None,
        motion_out_dir=None, num_student=None, seed=None,
    )


def build_env_and_arms(cli, num_envs):
    """Return ``(env, arms, ckpt_path)`` where ``arms`` maps name -> callable.

    ``load_run`` is passed through explicitly rather than resolved by
    ``sweep.resolve_load_run``: that helper gates on ``train_cfg.runner.run_name``,
    which for this task is ``'moe_cts' + get_simulator_suffix()`` -- so under the
    MuJoCo backend it reads ``moe_cts_mujoco`` and would refuse the very
    ``..._moe_cts_genesis`` run this probe exists to cross-check.  The identity of
    the loaded weights is instead recorded in the output npz.
    """
    if SIMULATOR == "genesis":  # noqa: F405
        gs.init(backend=gs.cpu if cli.cpu else gs.gpu, logging_level="warning")  # noqa: F405
    elif SIMULATOR == "mujoco":  # noqa: F405
        # MuJoCo steps on CPU; keeping env tensors on the GPU would only buy a
        # per-step round trip (play.py does the same).
        cli.cpu = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=cli.task)
    freeze_cfg_for_brake(env_cfg, cli, num_envs)

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)  # noqa: F405
    ckpt_path = get_load_path(log_root, load_run=cli.load_run, checkpoint=cli.ckpt)
    print(f"[brake] backend={SIMULATOR} checkpoint={ckpt_path}")  # noqa: F405

    reg_args = make_registry_args(cli)
    env, _ = task_registry.make_env(name=cli.task, args=reg_args, env_cfg=env_cfg)

    # Build with ``resume=False`` + ``log_root=None`` and load the checkpoint by
    # hand.  ``make_alg_runner`` derives its resume path from the same ``log_root``
    # it uses to CREATE a timestamped run folder, so letting it resume would have
    # this read-only probe litter logs/<experiment>/ with a fresh dir (and a copy
    # of best_*.pt) on every launch.  ``load_env_curriculum=False``: this env is a
    # flat plane, so the checkpoint's per-env terrain levels describe geometry
    # that does not exist here.
    train_cfg.runner.resume = False
    with skip_rollout_storage(train_cfg):
        runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=cli.task, args=reg_args, train_cfg=train_cfg,
            log_root=None, load_env_curriculum=False)
    runner.load(ckpt_path, load_env_curriculum=False)

    if not cli.randomize_start:
        make_reset_deterministic(env)

    actor_critic = runner.alg.actor_critic
    actor_critic.eval()

    def student(obs, priv, hist):
        return actor_critic.act_student(obs, hist)

    def teacher(obs, priv, hist):
        return actor_critic.act_teacher(obs, priv)

    return env, {"student": student, "teacher": teacher}, ckpt_path


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
def build_conditions(cli):
    """(vx, ramp, phase_offset) for every cell, plus the offsets used.

    Offsets are spread as evenly as ``phase_span`` allows; with the default
    16 phases over 20 steps they are the integers 0,1,2,3,5,6,...,18 -- i.e.
    16 distinct points inside one nominal gait period.
    """
    offsets = np.unique(np.round(
        np.linspace(0, cli.phase_span - 1, cli.phases)).astype(int))
    conds = [
        (float(vx), float(ramp), int(off))
        for vx, ramp, off in itertools.product(cli.vx, cli.ramp, offsets)
    ]
    if cli.max_conditions is not None:
        conds = conds[:cli.max_conditions]
    return conds, offsets


def command_schedule(conds, T, pre_steps, dt):
    """(T, N) forward command and (N,) trigger step index.

    Before its trigger an env holds ``vx``; after it the command falls linearly
    to zero over ``ramp`` seconds (``ramp <= 0`` drops to zero in one step).
    """
    n = len(conds)
    cmd = np.zeros((T, n), dtype=np.float32)
    trig = np.zeros(n, dtype=np.int64)
    for i, (vx, ramp, off) in enumerate(conds):
        g = pre_steps + off
        trig[i] = g
        for t in range(T):
            if t < g:
                cmd[t, i] = vx
            elif ramp <= 0.0:
                cmd[t, i] = 0.0
            else:
                frac = 1.0 - ((t - g) * dt) / ramp
                cmd[t, i] = vx * max(0.0, frac)
    return cmd, trig


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
# What each recorded channel is for, so the analyzer never has to guess:
#   base_pos/base_quat        pitch, base-height overshoot, stopping distance
#   base_lin_vel/base_ang_vel body-frame velocity -> tracking error, pitch rate
#   projected_gravity         tilt without unwrapping Euler angles
#   dof_pos/dof_vel           rear hip/thigh/calf extrema
#   torques                   saturation during the brake
#   actions                   the open-loop replay source (--replay)
#   feet_pos                  foot height; calibrated per-backend by the analyzer
#                             because the foot frame origin sits at a different
#                             height above ground in the URDF and the MJCF
#   feet_contact_forces       the bilateral rear contact-loss event + impulses
RECORD_SPECS = (
    ("base_pos", lambda e: e.simulator.base_pos),
    ("base_quat", lambda e: e.simulator.base_quat),
    ("base_lin_vel", lambda e: e.simulator.base_lin_vel),
    ("base_ang_vel", lambda e: e.simulator.base_ang_vel),
    ("projected_gravity", lambda e: e.simulator.projected_gravity),
    ("dof_pos", lambda e: e.simulator.dof_pos),
    ("dof_vel", lambda e: e.simulator.dof_vel),
    ("torques", lambda e: e.simulator.torques),
    ("feet_pos", lambda e: e.simulator.feet_pos),
    ("feet_contact_forces",
     lambda e: e.simulator.link_contact_forces[:, e.simulator.feet_contact_indices, :]),
)


class WindowRecorder:
    """Per-step CPU-side buffers for one chunk of conditions."""

    def __init__(self, T, n):
        self.T = T
        self.n = n
        self.data = {}
        self.t = 0

    def _slot(self, key, sample):
        if key not in self.data:
            self.data[key] = np.zeros((self.T, self.n) + tuple(sample.shape[1:]),
                                      dtype=np.float32)
        return self.data[key]

    @torch.no_grad()
    def record(self, env, actions):
        for key, getter in RECORD_SPECS:
            val = getter(env)
            arr = self._slot(key, val)
            arr[self.t] = val.detach().float().cpu().numpy()
        act = self._slot("actions", actions)
        act[self.t] = actions.detach().float().cpu().numpy()
        cmd = self._slot("commands", env.commands[:, :3])
        cmd[self.t] = env.commands[:, :3].detach().float().cpu().numpy()

        # Falls, latched: an env that terminates non-timeout inside the window
        # stays flagged, so a late fall after an auto_reset respawn is not lost
        # and a single fall is not double-counted.
        fell_now = (env.reset_buf.bool() & ~env.time_out_buf.bool()).cpu().numpy()
        flags = self.data.setdefault("fell", np.zeros((self.T, self.n), dtype=np.float32))
        prev = flags[self.t - 1] if self.t > 0 else np.zeros(self.n, dtype=np.float32)
        flags[self.t] = np.maximum(prev, fell_now.astype(np.float32))
        self.t += 1


@torch.no_grad()
def run_window(env, act_fn, cmd_chunk, cli, replay_actions=None, trigger=None,
               sync_src=None):
    """Warm up at the cruise command, then record the braking window.

    ``cmd_chunk`` is (T, n_chunk) forward commands.  The warmup command is row 0
    of the schedule (still full cruise for every env by construction) and the
    episode clock is zeroed after it, so the settled physics state is kept while
    the timeout is restarted -- otherwise ``max_episode_length`` would fire
    ``warmup`` steps early and truncate the recovery window.
    """
    device = env.device
    T, n = cmd_chunk.shape

    cruise = torch.as_tensor(cmd_chunk[0], dtype=torch.float, device=device)
    # Read by the pinned resampler installed in make_reset_deterministic, so the
    # command is already the cruise value in the observation reset() returns.
    env._brake_cruise = cruise

    env.reset()
    out = env.get_observations()
    obs, priv, hist = out[0], out[1], out[2]
    for _ in range(cli.warmup):
        env.commands[:, 0] = cruise
        env.commands[:, 1:3] = 0.0
        actions = act_fn(obs.detach(), priv.detach(), hist.detach())
        out = env.step(actions.detach())
        obs, priv, hist = out[0], out[1], out[2]
    env.episode_length_buf.zero_()

    rec = WindowRecorder(T, n)
    sched = torch.as_tensor(cmd_chunk, dtype=torch.float, device=device)
    for t in range(T):
        # Write the command for THIS step so the obs produced by this step --
        # which the next action consumes -- already reflects it.
        env.commands[:, 0] = sched[t]
        env.commands[:, 1:3] = 0.0
        if sync_src is not None:
            fire = torch.nonzero(
                torch.as_tensor(trigger, device=device) == t, as_tuple=False).flatten()
            if fire.numel():
                inject_state(env, fire, sync_src, t)
        if replay_actions is None:
            actions = act_fn(obs.detach(), priv.detach(), hist.detach())
        elif cli.replay_from == "window":
            actions = torch.as_tensor(replay_actions[t], dtype=torch.float, device=device)
        else:
            # Cut the loop at each env's OWN trigger.  Before it the policy still
            # drives, so the cruise gait is this backend's own and the state at
            # the trigger is the state the closed-loop run actually reached; only
            # the brake response is open-loop.  Replaying the pre-trigger stretch
            # too (``--replay_from window``) diverges within the cruise -- measured:
            # both rear feet already airborne in 100% of conditions BEFORE the
            # stop -- which answers nothing about braking.
            policy_actions = act_fn(obs.detach(), priv.detach(), hist.detach())
            replay_t = torch.as_tensor(replay_actions[t], dtype=torch.float, device=device)
            open_loop = (torch.as_tensor(trigger, device=device) <= t).unsqueeze(1)
            actions = torch.where(open_loop, replay_t, policy_actions)
        out = env.step(actions.detach())
        obs, priv, hist = out[0], out[1], out[2]
        rec.record(env, actions)
    return rec.data


@torch.no_grad()
def inject_state(env, env_ids, src, t):
    """Overwrite the sim state of ``env_ids`` with the source run's state at step ``t``.

    Turns the cross-backend replay into a controlled experiment: without this,
    the two backends arrive at the trigger in DIFFERENT states (measured: 1.37 vs
    1.55 m/s cruise for the same command), so any post-trigger divergence mixes
    "different initial condition" with "different plant".  With it, the replay
    starts from a byte-identical state and is driven by an identical action
    sequence -- the only remaining difference is the physics.

    Velocities are recorded in the BASE frame (that is what ``simulator.base_*_vel``
    holds) while ``reset_root_states`` takes WORLD frame, hence ``quat_apply``.
    ``reset_root_states`` also zeroes joint velocity, so ``reset_dofs`` must come
    after it, not before.
    """
    from legged_gym.utils.math_utils import quat_apply

    device = env.device
    idx = env_ids.detach().cpu().numpy()
    tt = lambda key: torch.as_tensor(src[key][t][idx], dtype=torch.float, device=device)

    quat = tt("base_quat")
    lin_w = quat_apply(quat, tt("base_lin_vel"))
    ang_w = quat_apply(quat, tt("base_ang_vel"))
    env.simulator.reset_root_states(env_ids, tt("base_pos"), quat, lin_w, ang_w)
    env.simulator.reset_dofs(env_ids, tt("dof_pos"), tt("dof_vel"))


def chunk_ranges(total, size):
    for lo in range(0, total, size):
        yield lo, min(lo + size, total)


# --------------------------------------------------------------------------- #
def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    conds, offsets = build_conditions(cli)
    n_cond = len(conds)
    max_off = int(max(off for _, _, off in conds))
    T = cli.pre_steps + max_off + cli.post_steps

    # MuJoCo is a single-world backend (num_envs == 1 is enforced in
    # mujoco_simulator._parse_cfg), so its conditions run serially.
    chunk = 1 if SIMULATOR == "mujoco" else n_cond  # noqa: F405
    print(f"[brake] {n_cond} conditions, T={T} steps, chunk={chunk}, "
          f"phase offsets={list(offsets)}")

    env, arms, ckpt_path = build_env_and_arms(cli, chunk)
    act_fn = arms[cli.arm]
    dt = float(env.dt)
    cmd_all, trig_all = command_schedule(conds, T, cli.pre_steps, dt)

    replay_src = None
    sync_src = None
    if cli.replay is not None:
        src = np.load(cli.replay, allow_pickle=True)
        replay_src = src["actions"]                      # (T, n_cond, 12)
        if replay_src.shape[:2] != (T, n_cond):
            raise SystemExit(
                f"--replay grid mismatch: file has {replay_src.shape[:2]}, "
                f"this run needs {(T, n_cond)}. Re-run the source with the same grid flags.")
        if not np.array_equal(src["trigger_step"], trig_all):
            raise SystemExit(
                "--replay trigger schedule differs from this run's; the open-loop cut "
                "would happen at a different point than the recorded one.")
        if cli.replay_sync_state:
            if cli.replay_from != "trigger":
                raise SystemExit("--replay_sync_state requires --replay_from trigger")
            sync_src = {k: src[k] for k in
                        ("base_pos", "base_quat", "base_lin_vel", "base_ang_vel",
                         "dof_pos", "dof_vel")}
        print(f"[brake] OPEN-LOOP replay of {cli.replay} "
              f"(source backend={src['backend'] if 'backend' in src else '?'}, "
              f"cut at {cli.replay_from}, state sync={bool(sync_src)})")

    feet_names = list(getattr(env.simulator, "_feet_names", []))
    print(f"[brake] feet order: {feet_names}")

    collected = {}
    t0 = time.time()
    for ci, (lo, hi) in enumerate(chunk_ranges(n_cond, chunk)):
        part = run_window(
            env, act_fn, cmd_all[:, lo:hi], cli,
            replay_actions=None if replay_src is None else replay_src[:, lo:hi],
            trigger=trig_all[lo:hi],
            sync_src=None if sync_src is None
                     else {k: v[:, lo:hi] for k, v in sync_src.items()})
        for key, arr in part.items():
            collected.setdefault(key, []).append(arr)
        done = hi
        rate = (time.time() - t0) / done
        print(f"[brake] {done}/{n_cond} conditions "
              f"({rate:.1f} s/cond, eta {rate * (n_cond - done) / 60:.1f} min)", flush=True)

    merged = {k: np.concatenate(v, axis=1) for k, v in collected.items()}

    meta = dict(
        backend=SIMULATOR,                                   # noqa: F405
        task=cli.task, load_run=cli.load_run, ckpt_path=ckpt_path,
        arm=cli.arm if replay_src is None
            else (f"replay[{cli.replay_from}"
                  f"{'+sync' if sync_src is not None else ''}]:"
                  f"{os.path.basename(cli.replay).replace('.npz', '')}"),
        dt=dt, T=T, pre_steps=cli.pre_steps, post_steps=cli.post_steps,
        warmup=cli.warmup, seed=cli.seed,
        # Names, not indices: the two backends build their link/DOF tables from
        # different assets (URDF vs MJCF), so the analyzer must resolve "rear
        # feet" and "RR_calf" by name or it can silently compare a hip against a
        # calf across backends.
        feet_names=np.asarray(feet_names),
        dof_names=np.asarray(list(getattr(env.simulator, "_dof_names", []))),
        torque_limits=np.asarray(env.simulator.torque_limits.detach().cpu().numpy()),
        cond_vx=np.asarray([c[0] for c in conds], dtype=np.float32),
        cond_ramp=np.asarray([c[1] for c in conds], dtype=np.float32),
        cond_phase=np.asarray([c[2] for c in conds], dtype=np.int64),
        trigger_step=trig_all,
        command_vx=cmd_all,
        cli=json.dumps(vars(cli)),
    )

    out = os.path.abspath(cli.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **merged, **meta)
    os.replace(tmp, out)
    print(f"[brake] wrote {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
