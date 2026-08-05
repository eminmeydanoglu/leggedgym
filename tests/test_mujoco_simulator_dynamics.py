from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

# Import through the established package path; importing the backend module
# first hits the repository's pre-existing simulator/env circular import.
import legged_gym.envs  # noqa: F401
from legged_gym.simulator.mujoco_simulator import MujocoSimulator
from legged_gym.utils.math_utils import quat_wxyz_to_xyzw, quat_xyzw_to_wxyz


def _tiny_model():
    xml = """
    <mujoco>
      <worldbody>
        <geom name="ground" type="plane" size="2 2 .1"/>
        <body name="base" pos="0 0 1">
          <freejoint/>
          <geom name="base_geom" type="box" size=".1 .1 .1" mass="2"/>
          <body name="foot" pos=".2 0 0">
            <joint name="joint" axis="0 1 0" damping=".2" armature=".01" frictionloss=".03"/>
            <geom name="foot_geom" type="sphere" size=".05" mass=".1" priority="1"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor name="motor" joint="joint" ctrlrange="-20 20"/></actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def _bare_sim():
    sim = MujocoSimulator.__new__(MujocoSimulator)
    # Simulator constructors carry the device as a string; torch_rand_float's
    # scripted signature intentionally requires that exact contract.
    sim._device = "cpu"
    sim._num_envs = 1
    sim._num_dof = sim._num_actions = 1
    return sim


def test_compute_torques_matches_pd_law_with_bias_scales_and_clip():
    sim = _bare_sim()
    sim._cfg = SimpleNamespace(control=SimpleNamespace(action_scale=2.0))
    sim._p_gains = torch.tensor([10.0])
    sim._d_gains = torch.tensor([3.0])
    sim._kp_scale = torch.tensor([[1.5]])
    sim._kd_scale = torch.tensor([[0.5]])
    sim._default_dof_pos = torch.tensor([[0.2]])
    sim._dof_pos = torch.tensor([[-0.1]])
    sim._dof_vel = torch.tensor([[2.0]])
    sim._motor_zero_offsets = torch.tensor([[0.4]])
    sim._torque_limits = torch.tensor([5.0])

    result = sim._compute_torques(torch.tensor([[0.25]]))

    # Unclipped value is 15*1.2 - 1.5*2 = 15, so the actuator limit must win.
    torch.testing.assert_close(result, torch.tensor([[5.0]]))


def test_quaternion_layout_and_post_step_velocity_frames_are_distinguishable():
    sim = _bare_sim()
    model, data = _tiny_model()
    sim._model, sim._data = model, data
    sim._num_links = 2
    sim._body_ids = np.array([1, 2], dtype=np.int32)
    sim._feet_indices = []
    sim._key_body_indices = []
    sim._contact_state_link_indices = []
    sim._dof_qpos_adr = np.array([7], dtype=np.int32)
    sim._dof_qvel_adr = np.array([6], dtype=np.int32)
    sim._terrain_x_range = torch.tensor([-10.0, 10.0])
    sim._terrain_y_range = torch.tensor([-10.0, 10.0])
    sim._env_origins = torch.zeros((1, 3))
    sim._base_init_pos = torch.zeros(3)
    sim._cfg = SimpleNamespace(
        asset=SimpleNamespace(obtain_link_contact_states=False),
        terrain=SimpleNamespace(measure_heights=False),
    )
    sim._base_pos = torch.zeros((1, 3))
    sim._base_quat = torch.zeros((1, 4))
    sim._base_quat_wxyz = torch.zeros((1, 4))
    sim._base_euler = torch.zeros((1, 3))
    sim._base_lin_vel = torch.zeros((1, 3))
    sim._base_ang_vel = torch.zeros((1, 3))
    sim._projected_gravity = torch.zeros((1, 3))
    sim._global_gravity = torch.tensor([[0.0, 0.0, -1.0]])
    sim._dof_pos = torch.zeros((1, 1))
    sim._dof_vel = torch.zeros((1, 1))
    sim._link_contact_forces = torch.zeros((1, 2, 3))
    sim._feet_pos = torch.zeros((1, 0, 3))
    sim._feet_vel = torch.zeros((1, 0, 3))
    sim._feet_quat = torch.zeros((1, 0, 4))
    sim._key_body_pos = torch.zeros((1, 0, 3))

    # 90 deg yaw: world +X linear velocity becomes body -Y, while MuJoCo's
    # angular qvel is already body-frame and must remain component-identical.
    xyzw = torch.tensor([[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]], dtype=torch.float)
    wxyz = quat_xyzw_to_wxyz(xyzw)
    torch.testing.assert_close(quat_wxyz_to_xyzw(wxyz), xyzw)
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = wxyz[0].numpy()
    data.qvel[0:3] = [1.0, 0.0, 0.0]
    data.qvel[3:6] = [0.3, -0.4, 0.7]

    sim.post_physics_step()

    torch.testing.assert_close(sim._base_quat, xyzw, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        sim._base_lin_vel, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        sim._base_ang_vel, torch.tensor([[0.3, -0.4, 0.7]]), atol=1e-6, rtol=0)


def test_push_overwrite_replaces_velocity_and_viewer_noops_are_safe(monkeypatch):
    sim = _bare_sim()
    sim._model, sim._data = _tiny_model()
    sim._cfg = SimpleNamespace(domain_rand=SimpleNamespace(
        max_push_vel_xy=0.8, max_push_ang_vel=1.2))
    sim._base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    sim._base_lin_vel = torch.zeros((1, 3))
    sim._base_ang_vel = torch.zeros((1, 3))
    sim._last_base_lin_vel = torch.zeros((1, 3))
    sim._last_base_ang_vel = torch.zeros((1, 3))
    sim._rand_push_vels = torch.zeros((1, 3))
    sim._viewer = None
    sim._data.qvel[:6] = [9, 9, 4, 9, 9, 9]
    monkeypatch.setattr(
        "legged_gym.simulator.mujoco_simulator.torch_rand_float",
        lambda lo, hi, shape, device: torch.full(shape, (lo + hi) / 4, device=device))

    sim.push_robots_overwrite(torch.tensor([0]))

    assert sim._data.qvel[2] == 4  # untouched linear z
    assert np.all(sim._data.qvel[:2] != 9)
    assert np.all(sim._data.qvel[3:6] != 9)
    sim.set_viewer_camera(np.zeros(3), np.ones(3))
    sim.draw_debug_vis()


def test_domain_randomization_updates_mujoco_model_fields():
    sim = _bare_sim()
    sim._model, sim._data = _tiny_model()
    sim._base_body_id = 1
    sim._dof_qvel_adr = np.array([6], dtype=np.int32)
    sim._robot_geom_ids = np.flatnonzero(sim._model.geom_bodyid != 0)
    sim._nominal_geom_friction = sim._model.geom_friction.copy()
    sim._nominal_geom_solref = sim._model.geom_solref.copy()
    sim._nominal_base_mass = float(sim._model.body_mass[1])
    sim._nominal_base_ipos = sim._model.body_ipos[1].copy()
    sim._nominal_dof_armature = sim._model.dof_armature.copy()
    sim._nominal_dof_frictionloss = sim._model.dof_frictionloss.copy()
    sim._nominal_dof_damping = sim._model.dof_damping.copy()
    sim._cfg = SimpleNamespace(domain_rand=SimpleNamespace(
        friction_range=[0.7, 0.7], restitution_range=[0.4, 0.4],
        added_mass_range=[0.5, 0.5],
        com_pos_x_range=[0.01, 0.01], com_pos_y_range=[-0.02, -0.02],
        com_pos_z_range=[0.03, 0.03], joint_armature_range=[0.2, 0.2],
        joint_friction_range=[0.3, 0.3], joint_damping_range=[0.4, 0.4],
        kp_range=[0.8, 0.8], kd_range=[1.2, 1.2], pd_gain_scalar=False,
        motor_strength_range=[0.9, 0.9], motor_zero_offset_range=[0.05, 0.05],
        randomize_motor_strength=True, randomize_motor_zero_offset=True,
    ))
    sim._init_domain_params()
    ids = np.array([0])

    sim._randomize_friction(ids)
    sim._randomize_restitution(ids)
    sim._randomize_base_mass(ids)
    sim._randomize_com_displacement(ids)
    sim._randomize_joint_armature(ids)
    sim._randomize_joint_friction(ids)
    sim._randomize_joint_damping(ids)
    sim._randomize_pd_gain(ids)
    sim._draw_motor_strengths(ids)
    sim._draw_motor_zero_offsets(ids)

    assert np.allclose(sim._model.geom_friction[sim._robot_geom_ids, 0], 0.7)
    assert np.allclose(sim._model.geom_solref[sim._robot_geom_ids, 1], 0.6)
    assert sim._model.body_mass[1] == pytest.approx(sim._nominal_base_mass + 0.5)
    assert np.allclose(sim._model.body_ipos[1], sim._nominal_base_ipos + [0.01, -0.02, 0.03])
    assert sim._model.dof_armature[6] == pytest.approx(0.2)
    assert sim._model.dof_frictionloss[6] == pytest.approx(0.3)
    assert sim._model.dof_damping[6] == pytest.approx(0.4)
    torch.testing.assert_close(sim._kp_scale, torch.tensor([[0.8]]))
    torch.testing.assert_close(sim._kd_scale, torch.tensor([[1.2]]))
    torch.testing.assert_close(sim._motor_strengths, torch.tensor([[0.9]]))
    torch.testing.assert_close(sim._motor_zero_offsets, torch.tensor([[0.05]]))
