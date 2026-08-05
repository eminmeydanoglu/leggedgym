"""MuJoCo backend for the `Simulator` abstraction (CONSTRUCTION HALF).

Why this file exists
--------------------
The env/observation/reward code in `legged_gym/envs` never talks to a physics
engine directly; it talks to the `Simulator` ABC.  Genesis, IsaacGym and
IsaacLab already implement that surface, so adding MuJoCo lets the SAME policy,
the SAME observation assembly and the SAME reward terms be replayed on a third
solver -- i.e. a real sim-to-sim check rather than a re-implementation that
happens to look similar.

That only works if this backend is *semantically identical* to the reference
(`genesis_simulator.py`) everywhere the policy can observe a difference.  Any
divergence in DOF ordering, quaternion convention, buffer staleness or terrain
registration does not raise -- it silently feeds the policy a different vector
than it was trained on and shows up as "MuJoCo just walks worse".  Hence the
dense comments below: they exist to pin down the places where MuJoCo's data
layout differs from Genesis' and to say what was done about it.

Split of work
-------------
This file is the CONSTRUCTION half: config parsing, scene/model construction,
index resolution, buffer allocation, reset paths, terrain curriculum.
Everything that advances or perturbs the dynamics -- `step`, torque
computation, `post_physics_step` state read-back, pushes, debug drawing, the
domain-randomization draw methods -- is left as an explicitly marked
`NotImplementedError` stub for a second agent.  The stubs are still *called*
from the correct places in the control flow (see `reset_idx` / `_create_envs`)
so that the ordering contract is already correct when they land.

Genuine physics differences from Genesis (not papered over)
-----------------------------------------------------------
* Contact model.  Genesis uses a Newton constraint solver over collision pairs
  with per-entity friction; MuJoCo uses a soft-constraint (softness/impratio)
  formulation and `go2.xml` sets `cone="elliptic" impratio="100"`.  Normal
  forces, and therefore `link_contact_forces`, will not match Genesis
  step-for-step even from an identical state.  This is expected and is exactly
  what a sim-to-sim transfer test is meant to expose.
* Friction combination.  MuJoCo picks the geom pair's friction by `solmix` /
  priority rather than a per-entity scalar, so `_randomize_friction` cannot be
  a single `set_friction` call the way it is in Genesis (agent 2's problem, but
  noted here because the DR semantics differ, not just the API).
* One world only.  MuJoCo has a single `MjModel`/`MjData` pair, so this backend
  is hard-limited to `num_envs == 1` (see `_parse_cfg`).  All buffers keep their
  leading `num_envs` dimension anyway so the consuming env code is unchanged.
"""

import os

import numpy as np
import torch

import mujoco

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.simulator.simulator import Simulator
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import (
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    quat_rotate_inverse,
    quat_apply_yaw,
    get_euler_xyz,
    torch_rand_float,
)
# The world/scene (compiled go2.xml + terrain heightfield or plane, timestep,
# gravity, cleared actuator ctrl limits) is built by a sibling module.  Import
# it by contract -- do NOT stub it locally, or a broken scene builder would be
# masked by a silently-working fallback.
from legged_gym.simulator.mujoco_scene import build_scene, hfield_world_origin


""" ********** MuJoCo Simulator ********** """
class MujocoSimulator(Simulator):
    def __init__(self, cfg, sim_params: dict, device, headless):
        # Stored before super().__init__ because the ABC constructor immediately
        # runs _parse_cfg/_create_sim/_create_envs/_init_buffers, all of which
        # may read the sim params.
        self._sim_params = sim_params
        # The MuJoCo window (legged_gym/utils/mujoco_viewer.py :: MujocoViewer)
        # is created by agent 2 in _create_sim when not headless.  Keeping the
        # attribute defined unconditionally means every `if self._viewer is not
        # None` guard in the rest of the class is valid from construction time.
        self._viewer = None
        # Populated in _create_sim; declared here so `mesh_type == 'plane'`
        # builds still have the attribute (several consumers do `getattr`-free
        # access on it).
        self._terrain = None
        self._height_samples = None
        self._edge_mask = None
        super().__init__(cfg, sim_params, device, headless)

    # ----- Public methods -----#
    def step(self, actions):
        """One CONTROL step = `cfg.control.decimation` MuJoCo substeps.

        Statement-for-statement port of genesis_simulator.py:48-115.  The only
        backend-specific parts are (a) the torque handoff -- Genesis takes a
        policy-ordered tensor via `control_dofs_force`, MuJoCo takes a plain
        float array in ACTUATOR order via `data.ctrl` -- and (b) the DOF
        read-back, which is a direct qpos/qvel gather instead of an API call.
        Everything else (the last_* snapshots, the substep delay blend, the
        motor-strength scaling AFTER the torque clip) is the reference's.
        """
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._last_base_ang_vel[:] = self._base_ang_vel[:]
        self._last_feet_vel[:] = self._feet_vel[:]
        actions = self._pre_simulator_step(actions)
        if self._ctrl_delay_substep_mode:
            # Inlined genesis `_draw_substep_delay`: one per-env draw per control
            # step, in SIM SUBSTEP units, exposed on `_action_delay` because that
            # buffer is the privileged `dr_ctrl_delay` label the oracle reads.
            lo, hi = self._ctrl_delay_substep_range
            substep_delay = torch.randint(
                lo, hi + 1, (self._num_envs, 1), device=self._device)
            self._action_delay[:] = substep_delay.squeeze(1)

        ctrl = self._data.ctrl
        qpos = self._data.qpos
        qvel = self._data.qvel
        for i in range(self._cfg.control.decimation):
            self._last_dof_vel[:] = self._dof_vel[:]
            if self._ctrl_delay_substep_mode:
                # go2_rl_gym blend: substeps before the drawn delay keep the
                # previous control step's target, the rest use the current one
                use_current = (i >= substep_delay).float()
                step_actions = (1 - use_current) * self._prev_action_targets \
                    + use_current * actions
            else:
                step_actions = actions
            self._torques = self._compute_torques(step_actions)
            if self._randomize_motor_strength:
                # go2_rl_gym legged_robot.py:79-81 scales the torque AFTER
                # _compute_torques' clip, so the multiplier also scales the
                # effective torque limit.  Genesis then re-clamps to the model's
                # dof force_range; MuJoCo does NOT re-clamp, because
                # mujoco_scene._apply_sim_options cleared `actuator_ctrllimited`
                # precisely so this file owns the one and only limiter.  With
                # motor_strength_range <= 1 (the established config) the scaled
                # torque is strictly inside the limit either way, so the two
                # backends agree; a range above 1.0 would let MuJoCo exceed the
                # asset effort limit where Genesis clamps.  Called out rather
                # than "fixed" by re-clipping, since re-clipping would instead
                # break the reference's semantics for ranges below 1.
                self._torques = self._torques * self._motor_strengths
            # POLICY order -> ACTUATOR order.  `_dof_act_adr[i]` is the ctrl slot
            # driving policy DOF i; for go2 the actuator block is FR/FL/RR/RL
            # while the policy order is FL/FR/RL/RR, so a contiguous `ctrl[:] =`
            # would swap the left and right legs and still run.
            ctrl[self._dof_act_adr] = self._torques[0].detach().cpu().numpy()
            mujoco.mj_step(self._model, self._data)
            # `push_links` writes data.xfrc_applied, which -- unlike Genesis'
            # external force -- PERSISTS until overwritten.  Genesis clears it
            # once per scene.step() (engine/simulator.py:293), i.e. the push
            # lasts exactly one physics substep, so clearing here reproduces the
            # reference's impulse.  Leaving it set would turn a one-substep kick
            # into a constant body force for the rest of the episode.
            self._data.xfrc_applied[:] = 0.0
            # dof_pos and dof_vel use joint sequence of policy.  qpos/qvel are
            # the integrated state and need no mj_forward -- unlike every
            # DERIVED field, which post_physics_step refreshes (see there).
            self._dof_pos[0] = torch.as_tensor(
                qpos[self._dof_qpos_adr], dtype=torch.float, device=self._device)
            self._dof_vel[0] = torch.as_tensor(
                qvel[self._dof_qvel_adr], dtype=torch.float, device=self._device)
        if self._ctrl_delay_substep_mode:
            # the current target becomes the previous one for the next control
            # step (reference updates last_actions in post_physics_step)
            self._prev_action_targets[:] = actions[:]

        # One frame per CONTROL step.  MujocoViewer renders synchronously from
        # sync() on this thread (it has no draw loop of its own), so a missing
        # call here is not judder -- it is a frozen window.  Genesis' per-substep
        # `update_visualizer` stride has no analogue: there is no separate viewer
        # thread to keep fed, and rendering mid-decimation would only cost frames
        # showing states the policy never acted on.
        if self._viewer is not None:
            self._viewer.sync()

    def post_physics_step(self):
        """Read the solver state back into the buffers, mirroring genesis:117-142.

        The first line is the one that is easy to get wrong.  `mj_step` performs
        forward dynamics at the CURRENT state and then integrates, so on return
        `qpos`/`qvel` describe t+dt while every DERIVED field -- `xpos`, `xquat`,
        `cvel`, the contact list and `efc_force` -- still describes t.  Genesis'
        getters all report the post-step state, so reading MuJoCo's derived
        fields directly would hand the policy a base position from qpos at t+dt
        alongside foot positions at t: internally inconsistent by one substep,
        and completely silent.  One `mj_forward` re-derives everything from the
        integrated state, at the cost of roughly one extra substep per control
        step (25% at decimation=4, on a single-env backend that is not the
        bottleneck).
        """
        mujoco.mj_forward(self._model, self._data)
        qpos = self._data.qpos
        qvel = self._data.qvel

        # prepare quantities
        self._base_pos[0] = torch.as_tensor(
            qpos[0:3], dtype=torch.float, device=self._device)
        self._check_base_pos_out_of_bound()       # check if the pos of the robot is out of terrain bounds
        # re-read: the bounds check may have teleported the base back inside
        self._base_pos[0] = torch.as_tensor(
            qpos[0:3], dtype=torch.float, device=self._device)
        self._base_quat_wxyz[0] = torch.as_tensor(
            qpos[3:7], dtype=torch.float, device=self._device)
        self._base_quat[:] = quat_wxyz_to_xyzw(self._base_quat_wxyz)
        self._base_euler[:] = get_euler_xyz(self._base_quat)

        # --- base velocity: the highest-risk frame conversion in this file ---
        # Genesis returns both root velocities in the WORLD frame and rotates
        # both into the base frame here.  MuJoCo's free joint is asymmetric
        # (verified empirically, not from the docs): qvel[0:3] is the linear
        # velocity of the base FRAME ORIGIN in WORLD coordinates, while
        # qvel[3:6] is the angular velocity already expressed in the BODY frame.
        # So the linear half needs the same quat_rotate_inverse Genesis applies
        # and the angular half needs NO rotation at all.  Rotating both (the
        # obvious "port it verbatim" mistake) yields a base_ang_vel that is
        # wrong by one extra body->world rotation -- identical under an identity
        # base quaternion, so it survives every flat-ground smoke test and only
        # shows up as bad yaw tracking once the robot turns.
        self._base_lin_vel[:] = quat_rotate_inverse(
            self._base_quat,
            torch.as_tensor(qvel[0:3], dtype=torch.float,
                            device=self._device).unsqueeze(0))
        self._base_ang_vel[0] = torch.as_tensor(
            qvel[3:6], dtype=torch.float, device=self._device)

        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)
        self._dof_pos[0] = torch.as_tensor(
            qpos[self._dof_qpos_adr], dtype=torch.float, device=self._device)
        self._dof_vel[0] = torch.as_tensor(
            qvel[self._dof_qvel_adr], dtype=torch.float, device=self._device)

        self._link_contact_forces[0] = torch.as_tensor(
            self._net_contact_forces_world(), dtype=torch.float, device=self._device)

        # Link poses.  `data.xpos`/`data.xquat` are the body FRAME ORIGIN, which
        # is what Genesis' get_links_pos returns (its default ref is
        # "link_origin", not the CoM) -- MuJoCo's CoM equivalents are
        # xipos/ximat, and using those would shift every foot position by the
        # foot's inertial offset.
        body_xpos = self._data.xpos[self._body_ids]
        self._feet_pos[0] = torch.as_tensor(
            body_xpos[self._feet_indices], dtype=torch.float, device=self._device)
        self._feet_vel[0] = torch.as_tensor(
            self._body_linvel_world(self._feet_indices),
            dtype=torch.float, device=self._device)
        self._key_body_pos[0] = torch.as_tensor(
            body_xpos[self._key_body_indices], dtype=torch.float, device=self._device)
        # Genesis leaves feet_quat at zero (it never fills the buffer); the Isaac
        # backends fill it, and here it is one gather away, so it is filled --
        # no go2 path reads it, so this cannot make the two backends disagree on
        # anything a policy observes, and the humanoid tasks that DO read it get
        # correct data instead of zeros.
        self._feet_quat[0] = quat_wxyz_to_xyzw(torch.as_tensor(
            self._data.xquat[self._body_ids][self._feet_indices],
            dtype=torch.float, device=self._device))

        # Link contact state
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = 1. * (torch.norm(
                self._link_contact_forces[:, self._contact_state_link_indices, :], dim=-1) > 1.)
        # update terrain heights info
        if self._cfg.terrain.measure_heights:
            self._update_surrounding_heights()
            if self._cfg.terrain.obtain_terrain_info_around_feet:
                self._calc_terrain_info_around_feet()

    def _net_contact_forces_world(self):
        """Per-link net CONTACT force, world frame, shape (num_links, 3).

        Accumulated from `data.contact` + `mj_contactForce` rather than read out
        of `data.cfrc_ext`, even though cfrc_ext[:, 3:6] is the same world-frame
        force and needs no Python loop.  Two reasons, both about matching what
        Genesis' `get_links_net_contact_force` means:

        * cfrc_ext is the net EXTERNAL force, which also carries `xfrc_applied`
          and equality-constraint forces.  `push_links` writes xfrc_applied, so
          using cfrc_ext would fold the domain-randomization push into the very
          buffer the termination check and the foot-contact rewards read -- a
          push would register as a contact.
        * cfrc_ext additionally requires an `mj_rnePostConstraint` pass, so the
          "cheap" option is not actually free.

        Sign convention (verified against a box resting on a plane, not from
        memory): `mj_contactForce` returns the force in the CONTACT frame acting
        on geom2's body, with `contact.frame` holding the frame axes as ROWS --
        hence `frame.T @ f` to get world coordinates, added to geom2's body and
        subtracted from geom1's.
        """
        forces = np.zeros((self._num_links, 3), dtype=np.float64)
        ncon = self._data.ncon
        if ncon == 0:
            return forces
        wrench = np.zeros(6, dtype=np.float64)
        geom_bodyid = self._model.geom_bodyid
        for i in range(ncon):
            con = self._data.contact[i]
            mujoco.mj_contactForce(self._model, self._data, i, wrench)
            f_world = con.frame.reshape(3, 3).T @ wrench[0:3]
            # body 0 is `world`; link index = body id - 1 (see _create_envs).
            b1 = int(geom_bodyid[con.geom1]) - 1
            b2 = int(geom_bodyid[con.geom2]) - 1
            if b1 >= 0:
                forces[b1] -= f_world
            if b2 >= 0:
                forces[b2] += f_world
        return forces

    def _body_linvel_world(self, link_indices):
        """World-frame linear velocity of the given links' FRAME ORIGINS.

        `mj_objectVelocity` with mjOBJ_XBODY (not mjOBJ_BODY -- that variant
        reports the velocity of the body's INERTIAL frame) and flg_local=0 gives
        [angular(3), linear(3)] about the body frame origin in world
        orientation, which is exactly Genesis' get_links_vel(ref="link_origin").
        `data.cvel` is not usable directly: it is expressed about the kinematic
        tree's subtree CoM, so for any link off that point it differs by
        omega x r.
        """
        out = np.zeros((len(link_indices), 3), dtype=np.float64)
        res = np.zeros(6, dtype=np.float64)
        for row, li in enumerate(link_indices):
            mujoco.mj_objectVelocity(
                self._model, self._data, mujoco.mjtObj.mjOBJ_XBODY,
                int(self._body_ids[li]), res, 0)
            out[row] = res[3:6]
        return out

    def reset_idx(self, env_ids):
        """Per-reset domain-randomization re-draws and last-* buffer clearing.

        Mirrors genesis_simulator.py:144-189 statement for statement.  The draw
        methods themselves are agent 2's stubs, but they are called from here
        already so the control flow (which DR is per-reset vs. per-build, and in
        what order) is fixed now and does not have to be re-derived later.
        """
        # domain randomization
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(env_ids)
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(env_ids)
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(env_ids)
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(env_ids)
        # motor DR trio (go2_rl_gym legged_robot.py:198-206): both are per-RESET,
        # per-env, per-DOF draws, exactly like the reference's reset_idx block.
        if self._randomize_motor_strength:
            self._draw_motor_strengths(env_ids)
        if self._randomize_motor_zero_offset:
            self._draw_motor_zero_offsets(env_ids)

        self._last_dof_vel[env_ids] = 0.
        self._last_feet_vel[env_ids] = 0.
        self._last_base_lin_vel[env_ids] = 0.
        self._last_base_ang_vel[env_ids] = 0.

        # reset action queue and delay
        if self._cfg.domain_rand.randomize_ctrl_delay:
            if self._ctrl_delay_substep_mode:
                # reference zeroes last_actions at episode start, so the first
                # substeps of a new episode fall back to a zero action target
                self._prev_action_targets[env_ids] = 0.
            else:
                self._action_queue[env_ids] *= 0.
                self._action_queue[env_ids] = 0.
                # Eval V2 needs a fixed delay over resets.  Keep the established
                # randomization behaviour for every normal training configuration.
                if not getattr(self._cfg.domain_rand, "eval_fixed_ctrl_delay", False):
                    self._action_delay[env_ids] = torch.randint(
                        self._cfg.domain_rand.ctrl_delay_step_range[0],
                        self._cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                        (len(env_ids),), device=self._device, requires_grad=False)

        # No depth-image reset branch here: _parse_cfg rejects sensor.add_depth
        # outright for this backend, so self._depth_images is always None.

    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        """Reset DOF position/velocity of the selected envs.

        `dof_pos` / `dof_vel` arrive in POLICY order, so they are scattered into
        qpos/qvel through the policy-order address arrays rather than written as
        a contiguous slice.  Writing them contiguously would look right, run
        fine, and permute the legs -- the single most damaging bug available in
        this file.
        """
        self._dof_pos[env_ids] = dof_pos[:]
        self._dof_vel[env_ids] = dof_vel[:]

        # Genesis passes zero_velocity=True here and then zeroes ALL dofs, so the
        # reference semantics are "position from the caller, velocity zero" --
        # the caller's dof_vel only reaches the mirror buffer, not the solver.
        # Reproduce that literally rather than honouring dof_vel in the solver.
        qpos = self._data.qpos
        qvel = self._data.qvel
        pos_np = self._dof_pos[env_ids].reshape(-1).detach().cpu().numpy()
        qpos[self._dof_qpos_adr] = pos_np
        qvel[self._dof_qvel_adr] = 0.0
        # Zero the free joint's 6 DOF too: Genesis' zero_all_dofs_velocity()
        # includes the floating base.
        qvel[0:6] = 0.0

        # MuJoCo equivalent of the IsaacGym "call forward before reading state
        # after a reset" rule: mj_forward recomputes xpos/xquat/cvel/sensordata
        # from the freshly written qpos/qvel.  Without it, anything read between
        # this call and the next mj_step (contact forces, body poses, the
        # viewer) still reflects the PRE-reset state.
        mujoco.mj_forward(self._model, self._data)

    def reset_root_states(self,
                          env_ids,
                          base_pos,
                          base_quat,
                          base_lin_vel_w,
                          base_ang_vel_w):
        """Reset the floating base. `base_quat` is XYZW (codebase convention)."""
        # base pos -> qpos[0:3]
        self._base_pos[env_ids, :] = base_pos[:]

        # base quat: the codebase carries XYZW everywhere; MuJoCo's freejoint
        # stores WXYZ in qpos[3:7].  Convert through math_utils, never by hand --
        # a hand-rolled swizzle that is wrong by a component transposition
        # produces a plausible-looking but rotated projected_gravity.
        self._base_quat[env_ids, :] = base_quat[:]
        self._base_quat_wxyz[env_ids] = quat_xyzw_to_wxyz(self._base_quat[env_ids])

        qpos = self._data.qpos
        qvel = self._data.qvel
        qpos[0:3] = self._base_pos[env_ids].reshape(-1).detach().cpu().numpy()
        qpos[3:7] = self._base_quat_wxyz[env_ids].reshape(-1).detach().cpu().numpy()
        # Genesis calls zero_all_dofs_velocity() between the pose writes and the
        # velocity write, so joint velocities are cleared on a root reset too.
        qvel[6:] = 0.0

        # update projected gravity (Genesis recomputes it for ALL envs here, not
        # just env_ids -- kept identical; with num_envs == 1 it is the same set)
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)

        # reset root states - velocity.  MuJoCo's freejoint qvel is
        # [world linear (3), world angular (3)], which is exactly the
        # (lin_vel_w, ang_vel_w) pair the caller supplies -- no frame change.
        qvel[0:3] = base_lin_vel_w.reshape(-1).detach().cpu().numpy()
        qvel[3:6] = base_ang_vel_w.reshape(-1).detach().cpu().numpy()

        # See reset_dofs: forward before anything reads state back.
        mujoco.mj_forward(self._model, self._data)

        # Mirror buffers in the BASE frame, matching Genesis' post-reset refresh.
        self._base_lin_vel[env_ids] = quat_rotate_inverse(
            self._base_quat[env_ids], base_lin_vel_w)
        self._base_ang_vel[env_ids] = quat_rotate_inverse(
            self._base_quat[env_ids], base_ang_vel_w)

    def update_terrain_curriculum(self, env_ids, move_up, move_down):
        """Port of genesis_simulator.py:247-255 -- pure index bookkeeping.

        Terrain levels/types are host-side tensors, so this is backend-agnostic;
        the only thing the physics engine ever sees is `_env_origins`, which the
        reset path then writes into qpos.
        """
        self._terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self._terrain_levels[env_ids] = torch.where(
            self._terrain_levels[env_ids] >= self._max_terrain_level,
            torch.randint_like(self._terrain_levels[env_ids], self._max_terrain_level),
            torch.clip(self._terrain_levels[env_ids], 0))  # (the minimum level is zero)
        self._env_origins[env_ids] = self._terrain_origins[self._terrain_levels[env_ids],
                                                           self._terrain_types[env_ids]]

    def update_sensors(self):
        """No-op for this backend.

        The only sensor the abstraction defines is the depth camera, and
        `_parse_cfg` rejects `sensor.add_depth` for MuJoCo (there is no Warp
        camera wired to an MjModel here).  Kept as an explicit, documented no-op
        rather than a stub, because the env calls it every control step and it is
        genuinely nothing to do -- not missing work.
        """
        return

    def push_robots(self):
        max_push_vel_xy = self._cfg.domain_rand.max_push_vel_xy
        push_vel = torch_rand_float(
            -max_push_vel_xy, max_push_vel_xy,
            (self._num_envs, 2), self._device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        self._data.qvel[0:2] += push_vel[0].detach().cpu().numpy()
        mujoco.mj_forward(self._model, self._data)
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._base_lin_vel[:] = quat_rotate_inverse(
            self._base_quat,
            torch.as_tensor(self._data.qvel[0:3], dtype=torch.float,
                            device=self._device).unsqueeze(0))

    def push_robots_overwrite(self, env_ids):
        # TODO(agent2): go2_rl_gym-style push that OVERWRITES the base velocity
        # of `env_ids` instead of adding (genesis_simulator.py:270-298):
        # qvel[0:2] <- U(+-max_push_vel_xy), qvel[3:6] <- U(+-max_push_ang_vel),
        # qvel[2] and all joint DOF untouched.
        if len(env_ids) == 0:
            return
        max_push_vel_xy = self._cfg.domain_rand.max_push_vel_xy
        max_push_ang_vel = self._cfg.domain_rand.max_push_ang_vel
        lin = torch_rand_float(
            -max_push_vel_xy, max_push_vel_xy, (len(env_ids), 2), self._device)
        ang = torch_rand_float(
            -max_push_ang_vel, max_push_ang_vel, (len(env_ids), 3), self._device)
        # This backend is deliberately single-env.  Keep env_ids validation
        # explicit so a malformed caller cannot silently perturb env 0.
        if 0 not in torch.as_tensor(env_ids).flatten().tolist():
            return
        self._data.qvel[0:2] = lin[0].detach().cpu().numpy()
        self._data.qvel[3:6] = ang[0].detach().cpu().numpy()
        self._rand_push_vels[0, :2] = lin[0]
        mujoco.mj_forward(self._model, self._data)
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._last_base_ang_vel[:] = self._base_ang_vel[:]
        self._base_lin_vel[:] = quat_rotate_inverse(
            self._base_quat,
            torch.as_tensor(self._data.qvel[0:3], dtype=torch.float,
                            device=self._device).unsqueeze(0))
        # Free-joint angular qvel is already in the body frame.
        self._base_ang_vel[0] = torch.as_tensor(
            self._data.qvel[3:6], dtype=torch.float, device=self._device)

    def push_links(self):
        # TODO(agent2): random per-link external force.  MuJoCo's equivalent of
        # Genesis' apply_links_external_force is writing data.xfrc_applied[body_id,
        # 0:3] for every body in self._body_ids (note: xfrc_applied PERSISTS across
        # mj_step, unlike Genesis' per-step force, so it must be zeroed again).
        max_force = self._cfg.domain_rand.max_push_force
        forces = torch.rand(
            (self._num_links, 3), device=self._device) * 2 * max_force - max_force
        self._data.xfrc_applied[self._body_ids, 0:3] = \
            forces.detach().cpu().numpy()

    def draw_debug_vis(self, ref_key_body_pos=None):
        # TODO(agent2): debug geometry (height sample points, key-body markers)
        # pushed into the MujocoViewer's user scene via mjv_initGeom.
        # Height/key-body debug primitives are visual diagnostics only; no
        # policy observation or simulator state depends on them.  Until the
        # viewer exposes a persistent user-scene API, an explicit no-op is safer
        # than drawing stale or incorrectly registered geometry.
        return

    def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray):
        # TODO(agent2): set self._viewer's mjvCamera -- MuJoCo cameras are
        # (lookat, distance, azimuth, elevation), so `eye` has to be converted
        # into that spherical form rather than assigned directly.
        if self._viewer is not None:
            self._viewer.set_camera(eye, target)

    def calc_feet_near_edge(self):
        # TODO(agent2): port genesis_simulator.py:336-368 verbatim -- it is pure
        # tensor logic over self._edge_mask (allocated here in _create_sim) and
        # self._feet_pos, so no MuJoCo API is involved.
        if self._cfg.terrain.mesh_type == 'plane':
            return torch.zeros(
                (self._num_envs, len(self._feet_indices)), device=self._device,
                dtype=torch.bool)
        if self._cfg.terrain.mesh_type == 'none':
            raise NameError("Can't calculate feet near edge with terrain mesh type 'none'")

        feet_pos_xy = self._feet_pos[:, :, :2]
        points = ((feet_pos_xy + self._cfg.terrain.border_size) /
                  self._cfg.terrain.horizontal_scale).long()
        px = torch.clip(points[:, :, 0], 0, self._edge_mask.shape[0] - 1)
        py = torch.clip(points[:, :, 1], 0, self._edge_mask.shape[1] - 1)
        threshold = self._cfg.rewards.feet_edge_threshold
        near = torch.zeros(
            (self._num_envs, len(self._feet_indices)), device=self._device,
            dtype=torch.bool)
        for i in range(-1, 2):
            for j in range(-1, 2):
                ix = torch.clip(px + i, 0, self._edge_mask.shape[0] - 1)
                iy = torch.clip(py + j, 0, self._edge_mask.shape[1] - 1)
                edge = self._edge_mask[ix, iy] == 1
                if edge.any():
                    pos = torch.zeros_like(feet_pos_xy)
                    pos[:, :, 0] = ix.float() * self._cfg.terrain.horizontal_scale \
                        - self._cfg.terrain.border_size
                    pos[:, :, 1] = iy.float() * self._cfg.terrain.horizontal_scale \
                        - self._cfg.terrain.border_size
                    near |= edge & (torch.norm(feet_pos_xy - pos, dim=-1) < threshold)
        return near

    # ----- Protected methods -----#
    def _pre_simulator_step(self, actions):
        # TODO(agent2): action-delay handling before the decimation loop --
        # queue mode (shift self._action_queue, gather per-env delay) and
        # substep mode (self._prev_action_targets bookkeeping), matching
        # genesis_simulator.py:371-115.
        if self._cfg.domain_rand.randomize_ctrl_delay \
                and not self._ctrl_delay_substep_mode:
            self._action_queue[:, 1:] = self._action_queue[:, :-1].clone()
            self._action_queue[:, 0] = actions.clone()
            actions = self._action_queue[
                torch.arange(self._num_envs, device=self._device),
                self._action_delay].clone()
        return actions

    def _parse_cfg(self):
        """Port of genesis_simulator.py:394-416, plus the MuJoCo-only guards."""
        # --- MuJoCo-only structural limits, checked first so the error the user
        # --- sees names the flag they must change, not a downstream shape error.
        if self._cfg.env.num_envs != 1:
            raise ValueError(
                f"MujocoSimulator is single-environment (MuJoCo has one world), but "
                f"cfg.env.num_envs = {self._cfg.env.num_envs}. Re-run with --num_envs 1."
            )
        if getattr(self._cfg.sensor, "add_depth", False):
            raise ValueError(
                "MujocoSimulator has no depth-camera support (sensor.add_depth=True is "
                "unsupported); the Warp camera path is wired to Genesis meshes only. "
                "Disable sensor.add_depth to run on MuJoCo."
            )

        self._debug = self._cfg.env.debug
        self._control_dt = self._cfg.sim.dt * self._cfg.control.decimation
        # Genesis needs this flag to decide whether to batch per-env dof/link
        # info at scene-build time.  MuJoCo has exactly one world, so per-env
        # batching is meaningless -- the flag is still computed because the DR
        # code paths (agent 2) branch on it for buffer shapes.
        self._batch_dofs_links_info = (
            self._cfg.domain_rand.randomize_joint_armature or
            self._cfg.domain_rand.randomize_joint_friction or
            self._cfg.domain_rand.randomize_joint_damping)
        # reference go2_rl_gym action delay: per-env SIM-SUBSTEP delay re-drawn
        # every control step, blended inside the decimation loop (see step()).
        # Off by default ("queue" mode keeps the legacy per-episode delay).
        self._ctrl_delay_substep_mode = (
            self._cfg.domain_rand.randomize_ctrl_delay
            and getattr(self._cfg.domain_rand, "ctrl_delay_mode", "queue") == "substep")
        if self._ctrl_delay_substep_mode:
            self._ctrl_delay_substep_range = getattr(
                self._cfg.domain_rand, "ctrl_delay_substep_range",
                [0, self._cfg.control.decimation])
            lo, hi = self._ctrl_delay_substep_range
            if not (0 <= lo <= hi <= self._cfg.control.decimation):
                raise ValueError(
                    f"domain_rand.ctrl_delay_substep_range {list(self._ctrl_delay_substep_range)} "
                    f"must satisfy 0 <= min <= max <= control.decimation "
                    f"({self._cfg.control.decimation}); units are sim substeps, not control steps")

    def _create_sim(self):
        """Build the terrain + the MjModel/MjData pair.

        Division of labour with `mujoco_scene.py`: that module owns everything
        that ends up inside the compiled model (go2.xml + hfield/plane geoms,
        opt.timestep, gravity, actuator ctrl-limit clearing).  This method owns
        the *host-side* terrain artefacts the rest of the class and the reward
        terms read directly -- `self._terrain`, `self._height_samples`,
        `self._edge_mask`, and the world x/y bounds.
        """
        mesh_type = self._cfg.terrain.mesh_type
        if mesh_type == 'plane':
            # No Terrain object at all: the scene module emits a ground plane and
            # height queries short-circuit to zero (see _update_surrounding_heights).
            self._terrain = None
        elif mesh_type == 'heightfield':
            self._terrain = Terrain(self._cfg.terrain)
            # Exactly genesis_simulator.py:1325-1327.  These two grids are the
            # *authoritative* terrain for every height/edge query in the reward
            # code -- deliberately NOT re-derived from the MuJoCo hfield, so that
            # Genesis and MuJoCo runs read bit-identical height samples and any
            # observed difference is attributable to the solver, not the map.
            self._height_samples = torch.tensor(self._terrain.heightsamples).view(
                self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
            self._edge_mask = torch.tensor(self._terrain.edge_mask).view(
                self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
        elif mesh_type == 'trimesh':
            raise NotImplementedError(
                "Trimesh terrain is not supported by the MuJoCo backend; use 'heightfield'.")
        else:
            raise ValueError(f"Unsupported terrain mesh type: {mesh_type}")

        # --- compile the world (sibling module owns the MJCF assembly) ---
        self._model = build_scene(self._cfg, terrain=self._terrain)
        self._data = mujoco.MjData(self._model)

        if mesh_type == 'heightfield':
            # Registration check.  `_update_surrounding_heights` /
            # `calc_feet_near_edge` index `_height_samples` as
            # (world_xy + border_size) / horizontal_scale, i.e. they assume the
            # raw heightfield's [0, 0] cell sits at world (-border_size,
            # -border_size) -- which is where Genesis places its terrain entity.
            # If the scene module registers the hfield anywhere else, every
            # height observation is offset by a constant and the policy walks
            # into geometry it cannot see.  Warn loudly rather than raise, since
            # the scene module owns the placement and may legitimately evolve.
            self._hfield_world_origin = np.asarray(hfield_world_origin(self._cfg), dtype=np.float64)
            expected = np.array([-self._cfg.terrain.border_size,
                                 -self._cfg.terrain.border_size], dtype=np.float64)
            if not np.allclose(self._hfield_world_origin, expected, atol=1e-6):
                print(f"[MujocoSimulator] WARNING: heightfield world origin "
                      f"{self._hfield_world_origin.tolist()} != expected {expected.tolist()}; "
                      f"height-sample lookups assume the Genesis convention "
                      f"(raw cell [0,0] at (-border_size, -border_size)) and will be "
                      f"mis-registered by the difference.")
        else:
            self._hfield_world_origin = np.zeros(2, dtype=np.float64)

        # specify the boundary of the heightfield (genesis_simulator.py:494-511)
        self._terrain_x_range = torch.zeros(2, device=self._device)
        self._terrain_y_range = torch.zeros(2, device=self._device)
        if self._cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            # give a small margin(1.0m)
            self._terrain_x_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_x_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_rows * self._cfg.terrain.terrain_length - 1.0
            self._terrain_y_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_y_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_cols * self._cfg.terrain.terrain_width - 1.0
        elif self._cfg.terrain.mesh_type == 'plane':  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self._terrain_x_range[0] = -self._cfg.terrain.plane_length / 2 + 1
            self._terrain_x_range[1] = self._cfg.terrain.plane_length / 2 - 1
            # the plane is a square
            self._terrain_y_range[0] = -self._cfg.terrain.plane_length / 2 + 1
            self._terrain_y_range[1] = self._cfg.terrain.plane_length / 2 - 1

        if not self._headless:
            from legged_gym.utils.mujoco_viewer import MujocoViewer
            self._viewer = MujocoViewer(self._model, self._data, self._cfg.viewer)

    def _create_envs(self):
        """Resolve every index this class hands out, and read static limits.

        There is no "env creation" in the Genesis/Isaac sense here -- `_create_sim`
        already compiled the one and only world.  What this method really does is
        establish the two index spaces the rest of the codebase consumes:

          1. POLICY DOF ORDER (`cfg.asset.dof_names`).  Every DOF-shaped tensor
             this class exposes is in this order.  MuJoCo's qpos/qvel are in MJCF
             *joint* order (FL/FR/RL/RR for go2) and `data.ctrl` is in MJCF
             *actuator* order (FR/FL/RR/RL) -- three different orderings that
             happen to have the same length, so a mix-up cannot be caught by any
             shape check.  Resolved once, here, into address arrays.
          2. LINK INDEX SPACE: the robot's bodies excluding `world`, in MuJoCo
             body order.  This deliberately does NOT match Genesis' link order.
             It does not need to: consumers only ever use the indices this class
             hands them (`feet_indices`, `penalized_contact_indices`, ...), and
             those are resolved by the same substring-matching rule Genesis uses,
             so they select the same physical set of bodies.
        """
        if self._cfg.asset.xml_file == "":
            raise NotImplementedError("Please specify asset.xml_file for the MuJoCo simulator!")
        # Resolved for the log line only -- the model itself was compiled by
        # mujoco_scene.build_scene, which owns the path formatting.
        asset_path = self._cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        print(f"[MujocoSimulator] robot asset: {os.path.basename(asset_path)}")

        self._get_env_origins()

        self._dof_names = self._cfg.asset.dof_names
        self._num_dof = len(self._cfg.asset.dof_names)

        # ---------------- policy-order DOF address resolution ----------------
        # actuator -> joint transmission map, so a policy joint name can be
        # traced to the ctrl slot that drives it.  Never assume actuator i drives
        # joint i: for go2 the actuator block is FR/FL/RR/RL while the body/joint
        # tree is FL/FR/RL/RR.
        act_of_joint = {}
        for a in range(self._model.nu):
            if self._model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                act_of_joint[int(self._model.actuator_trnid[a, 0])] = a

        dof_joint_ids, dof_qpos_adr, dof_qvel_adr, dof_act_adr = [], [], [], []
        for name in self._cfg.asset.dof_names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(
                    f"asset.dof_names entry '{name}' is not a joint in the compiled MuJoCo "
                    f"model. Available joints: "
                    f"{[mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self._model.njnt)]}")
            jtype = self._model.jnt_type[jid]
            if jtype not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(
                    f"asset.dof_names entry '{name}' is not a 1-DOF joint (jnt_type={jtype}); "
                    f"the address arrays assume one qpos/qvel slot per policy DOF.")
            if jid not in act_of_joint:
                raise ValueError(
                    f"joint '{name}' has no MuJoCo actuator driving it; _compute_torques "
                    f"would have nowhere to write its torque.")
            dof_joint_ids.append(int(jid))
            dof_qpos_adr.append(int(self._model.jnt_qposadr[jid]))
            dof_qvel_adr.append(int(self._model.jnt_dofadr[jid]))
            dof_act_adr.append(int(act_of_joint[jid]))

        self._dof_joint_ids = np.array(dof_joint_ids, dtype=np.int32)
        self._dof_qpos_adr = np.array(dof_qpos_adr, dtype=np.int32)
        self._dof_qvel_adr = np.array(dof_qvel_adr, dtype=np.int32)
        self._dof_act_adr = np.array(dof_act_adr, dtype=np.int32)
        # `_dof_indices` is the ABC-level name for "the DOF addresses in policy
        # order".  For MuJoCo the natural analogue is the qvel/dof address array
        # (that is what indexes velocities, torques and the dof-space model
        # fields such as dof_damping / dof_armature).
        self._dof_indices = self._dof_qvel_adr

        # Print the resolved mapping at build time, exactly like the Genesis
        # backend does.  This line is the only cheap way to catch a permuted leg
        # before it costs a training run.
        print("[MujocoSimulator] policy DOF order -> mujoco addresses:")
        for i, name in enumerate(self._cfg.asset.dof_names):
            act_name = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(self._dof_act_adr[i]))
            print(f"    [{i:2d}] {name:<24s} jnt={self._dof_joint_ids[i]:2d} "
                  f"qpos={self._dof_qpos_adr[i]:2d} qvel={self._dof_qvel_adr[i]:2d} "
                  f"act={self._dof_act_adr[i]:2d} ({act_name})")

        # ---------------- link (body) index space ----------------
        # body 0 is always MuJoCo's `world`; excluding it makes link index 0 the
        # robot's root body, matching what every consumer expects.
        self._body_ids = np.arange(1, self._model.nbody, dtype=np.int32)
        self._link_names = [
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, int(b))
            for b in self._body_ids]
        self._num_links = len(self._link_names)

        def find_link_indices(names):
            """Substring match over link names -- the same rule as the Genesis
            `find_link_indices` helper, so both backends select the same set of
            physical bodies even though the resulting integers differ."""
            link_indices = list()
            for idx, link_name in enumerate(self._link_names):
                flag = False
                for name in names:
                    if name in link_name:
                        flag = True
                if flag:
                    link_indices.append(idx)
            return link_indices

        print(f"all link names: {self._link_names}")
        self._termination_contact_indices = find_link_indices(
            self._cfg.asset.terminate_after_contacts_on)
        print("termination link indices:", self._termination_contact_indices)
        self._penalized_contact_indices = find_link_indices(
            self._cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self._penalized_contact_indices}")
        self._feet_names = [
            link_name for link_name in self._link_names
            if self._cfg.asset.foot_name in link_name]
        self._feet_indices = find_link_indices(self._feet_names)
        print(f"feet names: {self._feet_names}, feet link indices: {self._feet_indices}")
        assert len(self._feet_indices) > 0
        # MuJoCo has no separate contact-sensor body ordering (that split exists
        # only to work around IsaacLab), so the contact-space feet indices are
        # the articulation-space ones.
        self._feet_contact_indices = self._feet_indices
        self._key_body_indices = find_link_indices(self._cfg.asset.key_bodies)
        print(f"key body link indices: {self._key_body_indices}")
        base_link_name = getattr(self._cfg.asset, "base_link_name", "base")
        base_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, base_link_name)
        if base_body_id < 0:
            raise ValueError(f"asset.base_link_name '{base_link_name}' is not a body in the model.")
        self._base_body_id = int(base_body_id)
        self._base_link_index = self._base_body_id - 1  # world excluded
        print(f"base link index: {self._base_link_index} (mujoco body id {self._base_body_id})")

        # DR writes model arrays in-place.  Cache nominal values once so every
        # re-draw is absolute rather than accumulating mass/COM/damping shifts.
        self._robot_geom_ids = np.flatnonzero(self._model.geom_bodyid != 0)
        self._nominal_geom_friction = self._model.geom_friction.copy()
        self._nominal_geom_solref = self._model.geom_solref.copy()
        self._nominal_base_mass = float(self._model.body_mass[self._base_body_id])
        self._nominal_base_ipos = self._model.body_ipos[self._base_body_id].copy()
        self._nominal_dof_armature = self._model.dof_armature.copy()
        self._nominal_dof_frictionloss = self._model.dof_frictionloss.copy()
        self._nominal_dof_damping = self._model.dof_damping.copy()

        if self._cfg.asset.obtain_link_contact_states:
            self._contact_state_link_indices = find_link_indices(
                self._cfg.asset.contact_state_link_names)
        else:
            self._contact_state_link_indices = []

        # ---------------- static joint / actuator limits ----------------
        # dof position limits, policy order, straight from the model.
        self._dof_pos_limits = torch.tensor(
            np.asarray(self._model.jnt_range[self._dof_joint_ids], dtype=np.float32),
            device=self._device, dtype=torch.float)
        # MuJoCo exposes no per-joint velocity limit (velocity limits are an
        # actuator/solver concept there), so -- as in Genesis -- take them from
        # the config when the asset declares them.
        self._dof_vel_limits = None
        if hasattr(self._cfg.asset, "dof_vel_limits"):
            self._dof_vel_limits = torch.tensor(
                self._cfg.asset.dof_vel_limits, device=self._device).unsqueeze(0)
        # Torque limits come from the actuators' ctrlrange upper bound.  NOTE the
        # scene module CLEARS ctrllimited on the model (so mj_step does not clip
        # our torques a second time with different semantics); the limits are read
        # here from `ctrlrange`, which the scene module leaves intact, and the
        # clipping is done explicitly in _compute_torques instead.
        self._torque_limits = torch.tensor(
            np.asarray(self._model.actuator_ctrlrange[self._dof_act_adr, 1], dtype=np.float32),
            device=self._device, dtype=torch.float)
        print(f"[MujocoSimulator] torque limits (policy order): "
              f"{self._torque_limits.detach().cpu().numpy().tolist()}")

        # soft limits, genesis_simulator.py:611-620 (in place, same arithmetic)
        for i in range(self._dof_pos_limits.shape[0]):
            m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
            r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
            self._dof_pos_limits[i, 0] = (
                m - 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
            )
            self._dof_pos_limits[i, 1] = (
                m + 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
            )

        self._init_domain_params()
        # Startup DR draws, in the same order and under the same conditions as
        # genesis_simulator.py:623-653.  All of these land on agent 2's stubs;
        # they are wired up now so the *when* is already settled.
        # randomize friction (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self._num_envs))
        # randomize restitution (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_restitution:
            self._randomize_restitution(torch.arange(self._num_envs))
        # randomize base mass (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self._num_envs))
        # randomize COM displacement (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self._num_envs))
        # randomize joint armature
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(np.arange(self._num_envs))
        # randomize joint friction
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(np.arange(self._num_envs))
        # randomize joint damping
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(np.arange(self._num_envs))
        # randomize pd gain
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(np.arange(self._num_envs))
        # motor strength / zero offset: drawn here too so the very first episode
        # (before any reset_idx) already sees a randomized motor, matching the
        # reference where reset_idx runs for all envs at startup.
        if self._randomize_motor_strength:
            self._draw_motor_strengths(np.arange(self._num_envs))
        if self._randomize_motor_zero_offset:
            self._draw_motor_zero_offsets(np.arange(self._num_envs))

    def _init_buffers(self):
        """Allocate every `_`-prefixed tensor behind the ABC's properties.

        Shapes and dtypes are copied from genesis_simulator.py:656-771 so that
        consumers cannot tell the backends apart.  Everything keeps the leading
        `num_envs` dimension even though `num_envs == 1` here.
        """
        self._base_init_pos = torch.tensor(
            self._cfg.init_state.pos, device=self._device
        )
        self._base_init_quat = torch.tensor(
            self._cfg.init_state.rot, device=self._device
        )
        self._base_lin_vel = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_ang_vel = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._last_base_lin_vel = torch.zeros_like(self._base_lin_vel)
        self._last_base_ang_vel = torch.zeros_like(self._base_ang_vel)
        self._projected_gravity = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        # Unit "down" vector in the WORLD frame; rotated into the base frame each
        # step.  Note this is a normalized direction, not g*[0,0,-1] -- the
        # observation scaling downstream assumes unit length.
        self._global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=self._device, dtype=torch.float).repeat(self._num_envs, 1)
        self._dof_pos = torch.zeros(
            self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._dof_vel = torch.zeros(
            self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._last_dof_vel = torch.zeros_like(self._dof_vel)
        self._torques = torch.zeros(
            self._num_envs, self._num_actions, device=self._device, dtype=torch.float)
        self._base_pos = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        self._base_quat = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float)  # xyzw (codebase convention)
        # Scratch buffer holding the SAME orientation in MuJoCo's wxyz layout, so
        # the qpos[3:7] write is a straight copy and the conversion happens in
        # exactly one place (mirrors Genesis' _base_quat_gs).
        self._base_quat_wxyz = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float)
        self._base_euler = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float)
        # One row per link in THIS backend's link index space (bodies minus
        # world).  agent 2 fills it: MuJoCo's data.cfrc_ext is a 6-vector
        # [torque, force] in the body frame at the body's CoM, so it needs the
        # force half extracted (and, for a like-for-like comparison with Genesis'
        # net contact force, ideally accumulated from data.contact instead --
        # cfrc_ext also carries applied external forces such as xfrc_applied).
        self._link_contact_forces = torch.zeros(
            (self._num_envs, self._num_links, 3), device=self._device, dtype=torch.float)
        self._feet_pos = torch.zeros(
            (self._num_envs, len(self._feet_indices), 3), device=self._device, dtype=torch.float)
        self._feet_vel = torch.zeros(
            (self._num_envs, len(self._feet_indices), 3), device=self._device, dtype=torch.float)
        # Not populated by the Genesis backend either, but the ABC exposes a
        # `feet_quat` property, so allocate it (xyzw) rather than let attribute
        # access explode.
        self._feet_quat = torch.zeros(
            (self._num_envs, len(self._feet_indices), 4), device=self._device, dtype=torch.float)
        self._key_body_pos = torch.zeros(
            (self._num_envs, len(self._key_body_indices), 3), device=self._device, dtype=torch.float)
        self._last_feet_vel = torch.zeros_like(self._feet_vel)

        # Depth camera: rejected in _parse_cfg, but the property must not raise
        # on attribute access, so keep an explicit None.
        self._depth_images = None

        # Terrain information around feet.  Allocated unconditionally (zero-width
        # when the feature is off) so the properties are always safe to read.
        if self._cfg.terrain.obtain_terrain_info_around_feet:
            self._normal_vector_around_feet = torch.zeros(
                self._num_envs, len(self._feet_indices) * 3,
                dtype=torch.float, device=self._device, requires_grad=False)
            self._height_around_feet = torch.zeros(
                self._num_envs, len(self._feet_indices), 9,
                dtype=torch.float, device=self._device, requires_grad=False)
        else:
            self._normal_vector_around_feet = torch.zeros(
                self._num_envs, 0, dtype=torch.float, device=self._device, requires_grad=False)
            self._height_around_feet = torch.zeros(
                self._num_envs, 0, 9, dtype=torch.float, device=self._device, requires_grad=False)

        self._link_contact_states = torch.zeros(
            self._num_envs, len(self._contact_state_link_indices),
            dtype=torch.float, device=self._device, requires_grad=False)

        # default_dof_pos uses the POLICY joint sequence (it is added to the
        # policy's action to form the PD target, so it must not be in MJCF order)
        self._default_dof_pos = torch.tensor(
            [self._cfg.init_state.default_joint_angles[name]
                for name in self._cfg.asset.dof_names],
            device=self._device,
            dtype=torch.float,
        )
        self._default_dof_pos = self._default_dof_pos.unsqueeze(0)

        # PD control gains, policy order.
        stiffness = self._cfg.control.stiffness
        damping = self._cfg.control.damping
        self._p_gains, self._d_gains = [], []
        for dof_name in self._cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self._p_gains.append(stiffness[key])
                    self._d_gains.append(damping[key])
        self._p_gains = torch.tensor(self._p_gains, device=self._device)
        self._d_gains = torch.tensor(self._d_gains, device=self._device)
        if self._batch_dofs_links_info:
            self._p_gains = self._p_gains[None, :].repeat(self._num_envs, 1)
            self._d_gains = self._d_gains[None, :].repeat(self._num_envs, 1)
        # Deliberately NOT pushed into the model, unlike Genesis' set_dofs_kp/kv.
        # go2.xml's actuators are plain <motor> (direct torque), and the reward
        # terms need the applied torque as a tensor anyway, so the PD law is
        # evaluated in _compute_torques on the host and written to data.ctrl.
        # That also keeps the torque path bit-comparable with Genesis'.

        # control delay
        if self._cfg.domain_rand.randomize_ctrl_delay and not self._ctrl_delay_substep_mode:
            self._action_queue = torch.zeros(
                self._num_envs, self._cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
            self._action_delay = torch.randint(
                self._cfg.domain_rand.ctrl_delay_step_range[0],
                self._cfg.domain_rand.ctrl_delay_step_range[1] + 1,
                (self._num_envs,), device=self._device, requires_grad=False)
        if self._ctrl_delay_substep_mode:
            # previous control step's (clipped) action target, blended with the
            # current target inside step()'s decimation loop; zeroed at resets
            self._prev_action_targets = torch.zeros(
                self._num_envs, self._num_actions,
                dtype=torch.float, device=self._device, requires_grad=False)

        self._init_height_points()
        self._measured_heights = torch.zeros(
            self._num_envs, self._num_height_points, device=self._device, requires_grad=False)

    def _init_height_points(self):
        """Height sampling grid in the BASE frame (genesis_simulator.py:773-784).

        Pure tensor logic, ported unchanged -- the grid definition is part of the
        observation contract, so it must not drift between backends.
        """
        y = torch.tensor(self._cfg.terrain.measured_points_y,
                         device=self._device, requires_grad=False)
        x = torch.tensor(self._cfg.terrain.measured_points_x,
                         device=self._device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self._num_height_points = grid_x.numel()
        self._height_points = torch.zeros(self._num_envs, self._num_height_points,
                                          3, device=self._device, requires_grad=False)
        self._height_points[:, :, 0] = grid_x.flatten()
        self._height_points[:, :, 1] = grid_y.flatten()

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.

        Ported unchanged from genesis_simulator.py:786-844: it is pure tensor
        logic over the shared `Terrain` object, and the curriculum bookkeeping it
        sets up (`_terrain_levels`, `_terrain_types`, `_terrain_origins`,
        `_max_terrain_level`) is what `update_terrain_curriculum` mutates.
        """
        if self._cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self._custom_origins = True
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self._cfg.terrain.max_init_terrain_level
            if not self._cfg.terrain.curriculum:
                max_init_level = self._cfg.terrain.num_rows - 1
            fixed_level = getattr(self._cfg.terrain, "fixed_terrain_level", None)
            if fixed_level is None:
                self._terrain_levels = torch.randint(
                    0, max_init_level + 1, (self._num_envs,), device=self._device)
            else:
                fixed_level = int(fixed_level)
                if not 0 <= fixed_level < self._cfg.terrain.num_rows:
                    raise ValueError(
                        "terrain.fixed_terrain_level must be in "
                        f"[0, {self._cfg.terrain.num_rows - 1}], got {fixed_level}"
                    )
                self._terrain_levels = torch.full(
                    (self._num_envs,), fixed_level, dtype=torch.long, device=self._device)
            self._terrain_types = torch.zeros(
                self._num_envs, dtype=torch.long, device=self._device)
            if hasattr(self._cfg.env, "num_camera_envs"):
                # split envs with cameras across all terrain types, ensuring the
                # diversity of data collected by cameras
                self._terrain_types[:self._cfg.env.num_camera_envs] = torch.div(
                    torch.arange(self._cfg.env.num_camera_envs, device=self._device),
                    (self._cfg.env.num_camera_envs / self._cfg.terrain.num_cols),
                    rounding_mode='floor').to(torch.long)
                self._terrain_types[self._cfg.env.num_camera_envs:] = torch.div(
                    torch.arange(self._num_envs - self._cfg.env.num_camera_envs, device=self._device),
                    ((self._num_envs - self._cfg.env.num_camera_envs) / self._cfg.terrain.num_cols),
                    rounding_mode='floor').to(torch.long)
            else:
                self._terrain_types = torch.div(
                    torch.arange(self._num_envs, device=self._device),
                    (self._num_envs / self._cfg.terrain.num_cols),
                    rounding_mode='floor').to(torch.long)
            self._max_terrain_level = self._cfg.terrain.num_rows
            self._terrain_origins = torch.from_numpy(
                self._terrain.env_origins).to(self._device).to(torch.float)
            self._env_origins[:] = self._terrain_origins[self._terrain_levels,
                                                         self._terrain_types]
        else:
            self._custom_origins = False
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self._num_envs))
            num_rows = np.ceil(self._num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
            # plane has limited size, we need to specify spacing based on num_envs,
            # to make sure all robots are within the plane; restrict envs to a
            # square of [plane_length/2, plane_length/2]
            spacing = self._cfg.env.env_spacing
            if num_rows * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2 or \
                    num_cols * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2:
                spacing = min((self._cfg.terrain.plane_length / 2) / (num_rows - 1),
                              (self._cfg.terrain.plane_length / 2) / (num_cols - 1))
            self._env_origins[:, 0] = spacing * xx.flatten()[:self._num_envs]
            self._env_origins[:, 1] = spacing * yy.flatten()[:self._num_envs]
            self._env_origins[:, 2] = 0.
            self._env_origins[:, 0] -= self._cfg.terrain.plane_length / 4
            self._env_origins[:, 1] -= self._cfg.terrain.plane_length / 4

    def _update_surrounding_heights(self):
        # TODO(agent2): port genesis_simulator.py:846-905 verbatim.  It is pure
        # tensor logic over self._height_samples (allocated in _create_sim) plus
        # self._base_quat / self._base_pos, so nothing MuJoCo-specific is needed
        # -- and porting it verbatim is what keeps the height observation
        # identical between the two backends.
        if self._cfg.terrain.mesh_type == 'plane':
            self._measured_heights.zero_()
            return self._measured_heights
        if self._cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        points = quat_apply_yaw(
            self._base_quat.repeat(1, self._num_height_points),
            self._height_points) + self._base_pos[:, :3].unsqueeze(1)
        points += self._cfg.terrain.border_size
        points = (points / self._cfg.terrain.horizontal_scale).long()
        px = torch.clip(points[:, :, 0].reshape(-1), 0,
                        self._height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1].reshape(-1), 0,
                        self._height_samples.shape[1] - 2)
        heights = torch.minimum(
            torch.minimum(self._height_samples[px, py],
                          self._height_samples[px + 1, py]),
            self._height_samples[px, py + 1])
        self._measured_heights = heights.view(self._num_envs, -1) \
            * self._cfg.terrain.vertical_scale
        return self._measured_heights

    def _check_base_pos_out_of_bound(self):
        # TODO(agent2): clamp/flag bases that leave self._terrain_x_range /
        # self._terrain_y_range (computed in _create_sim).  In Genesis this
        # teleports the robot back inside; the MuJoCo version must write qpos and
        # call mujoco.mj_forward afterwards.
        out = ((self._base_pos[:, 0] >= self._terrain_x_range[1]) |
               (self._base_pos[:, 0] <= self._terrain_x_range[0]) |
               (self._base_pos[:, 1] >= self._terrain_y_range[1]) |
               (self._base_pos[:, 1] <= self._terrain_y_range[0]))
        if not bool(out[0]):
            return
        self._base_pos[0] = self.base_init_pos + self._env_origins[0]
        self._data.qpos[0:3] = self._base_pos[0].detach().cpu().numpy()
        mujoco.mj_forward(self._model, self._data)

    def _compute_torques(self, actions):
        # TODO(agent2): the PD law, genesis_simulator.py:924-945:
        #   target = actions * cfg.control.action_scale + self._default_dof_pos
        #   tau = kp*kp_scale*(target + motor_zero_offset - dof_pos) - kd*kd_scale*dof_vel
        #   return torch.clip(tau, -self._torque_limits, self._torque_limits)
        # self._p_gains / _d_gains / _kp_scale / _kd_scale / _motor_zero_offsets /
        # _torque_limits are all allocated here, all in POLICY order.  Write the
        # result into data.ctrl[self._dof_act_adr] (actuator order) -- do not
        # assume ctrl is already in policy order for other robots.
        actions_scaled = actions * self._cfg.control.action_scale
        if self._p_gains.ndim == 1:
            self._p_gains = self._p_gains.unsqueeze(0).repeat(self._num_envs, 1)
            self._d_gains = self._d_gains.unsqueeze(0).repeat(self._num_envs, 1)
        torques = (
            self._kp_scale * self._p_gains *
            (actions_scaled + self._default_dof_pos - self._dof_pos
             + self._motor_zero_offsets)
            - self._kd_scale * self._d_gains * self._dof_vel
        )
        return torch.clip(torques, -self._torque_limits, self._torque_limits)

    def _init_domain_params(self):
        """ Initializes domain randomization parameters, which are used to randomize the environment.

        Port of genesis_simulator.py:947-990.  These buffers are the *privileged
        observation* the teacher/oracle reads, so their shapes and nominal values
        are part of the observation contract, not an implementation detail: a
        backend that allocated them differently would silently change the
        privileged vector even before any randomization is applied.
        """
        self._friction_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._restitution_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._added_base_mass = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._rand_push_vels = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_com_bias = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_armature = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_friction = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_damping = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._kp_scale = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        self._kd_scale = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        # per-env SCALAR pd-gain draw (broadcast to all DOF in _randomize_pd_gain).
        # Kept as a clean 1-dim privileged label for the oracle's P vector; inits
        # to nominal 1.0 so dr_kp_scale is safe even when randomize_pd_gain is off.
        self._kp_scale_scalar = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        # motor DR trio (go2_rl_gym legged_robot.py:198-206). Both buffers are
        # always allocated at their nominal value (1.0 / 0.0), so a task that
        # leaves the flags off keeps a bit-identical torque path. The flags are
        # cached because they are read inside the per-substep loop; getattr keeps
        # domain_rand blocks that do not inherit LeggedRobotCfg.domain_rand safe.
        self._randomize_motor_strength = getattr(
            self._cfg.domain_rand, "randomize_motor_strength", False)
        self._randomize_motor_zero_offset = getattr(
            self._cfg.domain_rand, "randomize_motor_zero_offset", False)
        self._motor_strengths = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        self._motor_zero_offsets = torch.zeros(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        # always-present control-delay buffer (int steps). The build path re-inits
        # this when randomize_ctrl_delay is on; default 0 keeps dr_ctrl_delay safe
        # for tasks that never enable delay randomization.
        if not hasattr(self, "_action_delay"):
            self._action_delay = torch.zeros(
                self._num_envs, dtype=torch.long, device=self._device, requires_grad=False)

    # --- domain-randomization draws (dynamics half) ---------------------------
    # Every one of these is called from _create_envs and/or reset_idx above, so
    # the *when* is already fixed; only the MuJoCo-side application is missing.

    def _randomize_friction(self, env_ids=None):
        # TODO(agent2): draw U(friction_range) into self._friction_values and
        # apply it to model.geom_friction[:, 0] for the robot's geoms.  NOTE the
        # genuine difference from Genesis: MuJoCo combines the two geoms'
        # friction (max, or the higher-priority geom's value when geom_priority
        # differs), so the terrain/plane geom's friction has to be handled too
        # or the effective coefficient will not equal the drawn one.
        if env_ids is None:
            env_ids = np.arange(self._num_envs)
        lo, hi = self._cfg.domain_rand.friction_range
        value = torch_rand_float(lo, hi, (len(env_ids), 1), self._device)
        self._friction_values[env_ids] = value
        # go2.xml gives foot geoms priority=1, so their friction wins over the
        # terrain geom instead of Genesis' max-combine rule.  Writing every
        # robot geom (including those feet) makes the sampled coefficient the
        # effective foot-ground value; non-foot contacts can still combine
        # differently, which is a genuine simulator distinction.
        self._model.geom_friction[self._robot_geom_ids, 0] = float(value[0, 0])

    def _randomize_restitution(self, env_ids):
        # TODO(agent2): draw into self._restitution_values and map onto
        # model.geom_solref / solimp.  MuJoCo has no direct restitution
        # parameter -- bounciness comes out of the constraint solref, so this is
        # an approximation, not a one-to-one port.
        lo, hi = self._cfg.domain_rand.restitution_range
        value = torch_rand_float(lo, hi, (len(env_ids), 1), self._device)
        self._restitution_values[env_ids] = value
        # MuJoCo has no restitution scalar.  In positive solref format the
        # second component is damping ratio; lowering it increases rebound.
        # Preserve each geom's time constant and map r in [0,1] to a positive
        # damping ratio.  This is necessarily an approximation, documented
        # rather than presented as Genesis-identical physics.
        damping_ratio = max(0.05, 1.0 - float(value[0, 0]))
        self._model.geom_solref[self._robot_geom_ids, 1] = damping_ratio

    def _randomize_base_mass(self, env_ids=None):
        # TODO(agent2): self._added_base_mass draw ->
        # model.body_mass[self._base_body_id] = nominal + added.  Cache the
        # nominal mass at build time if you need to re-draw idempotently.
        if env_ids is None:
            env_ids = np.arange(self._num_envs)
        lo, hi = self._cfg.domain_rand.added_mass_range
        value = torch_rand_float(lo, hi, (len(env_ids), 1), self._device)
        self._added_base_mass[env_ids] = value
        self._model.body_mass[self._base_body_id] = \
            self._nominal_base_mass + float(value[0, 0])

    def _randomize_com_displacement(self, env_ids):
        # TODO(agent2): self._base_com_bias draw ->
        # model.body_ipos[self._base_body_id] += bias (cache the nominal ipos).
        n = len(env_ids)
        value = torch.zeros((n, 3), device=self._device)
        for axis, name in enumerate(('x', 'y', 'z')):
            lo, hi = getattr(self._cfg.domain_rand, f'com_pos_{name}_range')
            value[:, axis] = torch_rand_float(lo, hi, (n, 1), self._device).squeeze(1)
        self._base_com_bias[env_ids] = value
        self._model.body_ipos[self._base_body_id] = \
            self._nominal_base_ipos + value[0].detach().cpu().numpy()

    def _randomize_joint_armature(self, env_ids):
        # TODO(agent2): self._joint_armature draw ->
        # model.dof_armature[self._dof_qvel_adr] (policy order in, dof order out).
        lo, hi = self._cfg.domain_rand.joint_armature_range
        value = torch_rand_float(lo, hi, (len(env_ids), 1), self._device)
        self._joint_armature[env_ids] = value
        self._model.dof_armature[self._dof_qvel_adr] = float(value[0, 0])

    def _randomize_joint_friction(self, env_ids):
        # TODO(agent2): self._joint_friction draw ->
        # model.dof_frictionloss[self._dof_qvel_adr].
        lo, hi = self._cfg.domain_rand.joint_friction_range
        value = torch_rand_float(lo, hi, (len(env_ids), 1), self._device)
        self._joint_friction[env_ids] = value
        self._model.dof_frictionloss[self._dof_qvel_adr] = float(value[0, 0])

    def _randomize_joint_damping(self, env_ids):
        # TODO(agent2): self._joint_damping draw ->
        # model.dof_damping[self._dof_qvel_adr].
        lo, hi = self._cfg.domain_rand.joint_damping_range
        value = torch_rand_float(lo, hi, (len(env_ids), 1), self._device)
        self._joint_damping[env_ids] = value
        self._model.dof_damping[self._dof_qvel_adr] = float(value[0, 0])

    def _randomize_pd_gain(self, env_ids):
        # TODO(agent2): draw kp/kd scales into self._kp_scale / self._kd_scale
        # (and self._kp_scale_scalar when cfg.domain_rand.pd_gain_scalar).  No
        # model write is needed -- the PD law lives in _compute_torques.
        n = len(env_ids)
        kp_lo, kp_hi = self._cfg.domain_rand.kp_range
        kd_lo, kd_hi = self._cfg.domain_rand.kd_range
        if getattr(self._cfg.domain_rand, "pd_gain_scalar", False):
            value = torch_rand_float(kp_lo, kp_hi, (n, 1), self._device)
            self._kp_scale[env_ids] = value.expand(-1, self._num_dof)
            self._kd_scale[env_ids] = value.expand(-1, self._num_dof)
            self._kp_scale_scalar[env_ids] = value
        else:
            self._kp_scale[env_ids] = torch_rand_float(
                kp_lo, kp_hi, (n, self._num_dof), self._device)
            self._kd_scale[env_ids] = torch_rand_float(
                kd_lo, kd_hi, (n, self._num_dof), self._device)
            self._kp_scale_scalar[env_ids] = \
                self._kp_scale[env_ids].mean(dim=1, keepdim=True)

    def _draw_motor_strengths(self, env_ids):
        # TODO(agent2): per-env, per-DOF U(motor_strength_range) into
        # self._motor_strengths (applied to the CLIPPED torque in step()).
        lo, hi = self._cfg.domain_rand.motor_strength_range
        self._motor_strengths[env_ids] = torch_rand_float(
            lo, hi, (len(env_ids), self._num_dof), self._device)

    def _draw_motor_zero_offsets(self, env_ids):
        # TODO(agent2): per-env, per-DOF U(motor_zero_offset_range) into
        # self._motor_zero_offsets (added to the PD position target).
        lo, hi = self._cfg.domain_rand.motor_zero_offset_range
        self._motor_zero_offsets[env_ids] = torch_rand_float(
            lo, hi, (len(env_ids), self._num_dof), self._device)

    def _update_depth_camera(self):
        # TODO(agent2): unreachable by construction -- _parse_cfg rejects
        # sensor.add_depth for this backend.  Left as a stub so the ABC surface
        # is complete; if depth is ever wanted here it needs an MjvCamera +
        # offscreen mjr_render depth read, not the Warp path.
        # _parse_cfg rejects depth cameras before model construction.  The env
        # can still call this method through a generic debug path, where a no-op
        # is correct because no depth observation buffer exists on this backend.
        return
